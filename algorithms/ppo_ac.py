import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from tests.torch_np import np

from models.action_conditioned import ActionConditionedActor, CentralValueNet

device = torch.device("cuda")
autocast_device = 'cuda'
autocast_dtype = torch.float16


class ActionConditionedPPO:
    """
    Skeleton PPO per policy action-conditioned con mask implicito
    (passando solo le azioni legali). Non è CTDE completo, ma prepara la struttura.
    """
    def __init__(self,
                 obs_dim: int = 10823,
                 action_dim: int = 80,
                 lr: float = 3e-4,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 value_clip: float = 0.2,
                 target_kl: float = 0.02):
        self.actor = ActionConditionedActor(obs_dim, action_dim)
        self.critic = CentralValueNet(obs_dim)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip

        # Optional schedulers (can be set from trainer)
        self._lr_schedulers = []
        self._entropy_schedule = None
        self.update_steps = 0
        self.target_kl = target_kl
        self._high_kl_count = 0
        self._high_kl_patience = 5
        self._lr_decay_factor = 0.5
        # save config for reproducibility
        self.run_config = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'lr': lr,
            'clip_ratio': clip_ratio,
            'value_coef': value_coef,
            'entropy_coef': entropy_coef,
            'value_clip': value_clip,
            'target_kl': target_kl,
        }

        # Perf flags
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # AMP GradScaler su CUDA
        try:
            self.scaler = torch.amp.GradScaler('cuda')
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler()

    @torch.inference_mode()
    def select_action(self, obs, legal_actions: List, seat_team_vec = None, belief_summary = None) -> Tuple[torch.Tensor, float, int]:
        if len(legal_actions) == 0:
            raise ValueError("No legal actions")
        # Avoid warning: if obs already tensor, use clone().detach().to(device)
        if torch.is_tensor(obs):
            obs_t = obs.clone().detach().to(device=device, dtype=torch.float32).unsqueeze(0)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if len(legal_actions) > 0 and torch.is_tensor(legal_actions[0]):
            actions_t = torch.stack(legal_actions).to(device=device, dtype=torch.float32)
        else:
            actions_t = torch.stack([
                x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32, device=device)
            for x in legal_actions], dim=0)
        st = None
        if seat_team_vec is not None:
            if torch.is_tensor(seat_team_vec):
                st = seat_team_vec.clone().detach().to(device=device, dtype=torch.float32).unsqueeze(0)
            else:
                st = torch.tensor(seat_team_vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = self.actor(obs_t, actions_t, st)
        logp = torch.log_softmax(logits, dim=0)
        idx = int(torch.multinomial(torch.exp(logp), num_samples=1).item())
        return legal_actions[idx], logp[idx], idx

    def compute_loss(self, batch):
        """
        batch:
          - obs: (B, obs_dim)
          - act: (B, 80)
          - old_logp: (B)
          - ret: (B)
          - adv: (B)
          - legals: (M, 80) stack di tutte le azioni legali in ordine
          - legals_offset: (B) offset in legals per ciascun sample
          - legals_count: (B) numero di azioni legali per sample
          - chosen_index: (B) indice della scelta nel proprio sottoinsieme legale
        """
        to_f32 = lambda x: x.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device=device)
        to_long = lambda x: x.clone().detach().to(device=device, dtype=torch.long) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.long, device=device)

        obs = to_f32(batch['obs'])
        seat = to_f32(batch.get('seat_team', torch.zeros((obs.size(0), 6), device=device, dtype=torch.float32)))
        belief = to_f32(batch.get('belief_summary', torch.zeros((obs.size(0), 120), device=device, dtype=torch.float32)))
        act = to_f32(batch['act'])
        old_logp = to_f32(batch['old_logp'])
        ret = to_f32(batch['ret'])
        adv = to_f32(batch['adv'])
        legals = to_f32(batch['legals'])
        offs = to_long(batch['legals_offset'])
        cnts = to_long(batch['legals_count'])
        chosen_idx = to_long(batch['chosen_index'])

        # Ricalcola logp sulle sole azioni legali per ciascun sample (ragged via loop)
        new_logp_list = []
        entropy_list = []
        for i in range(obs.size(0)):
            start = offs[i].item()
            end = start + cnts[i].item()
            legal_i = legals[start:end]
            logits_i = self.actor(obs[i], legal_i, seat[i])
            logp_i = torch.log_softmax(logits_i, dim=0)
            new_logp_list.append(logp_i[chosen_idx[i]])
            # entropia media per diagnosi
            probs_i = torch.softmax(logits_i, dim=0)
            entropy_list.append(-(probs_i * logp_i).sum())

        logp_new = torch.stack(new_logp_list)
        entropy = torch.stack(entropy_list).mean()

        ratio = torch.exp(logp_new - old_logp)
        clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clipped)).mean()

        v = self.critic(obs, seat, belief)
        if self.value_clip is not None and self.value_clip > 0:
            v_clipped = torch.clamp(v, ret - self.value_clip, ret + self.value_clip)
            loss_v = torch.max((v - ret) ** 2, (v_clipped - ret) ** 2).mean()
        else:
            loss_v = nn.MSELoss()(v, ret)

        loss = loss_pi + self.value_coef * loss_v - self.entropy_coef * entropy
        approx_kl = (old_logp - logp_new).mean().abs()
        # clip frac non calcolata su tutte le azioni: metti 0 per logging (puoi rimpiazzare con stima migliore)
        clip_frac = torch.tensor(0.0, device=device)
        return loss, {
            'loss_pi': loss_pi.item(),
            'loss_v': loss_v.item(),
            'entropy': entropy.item(),
            'approx_kl': float(approx_kl.item()),
            'clip_frac': float(clip_frac.item())
        }

    def update(self, batch, epochs: int = 4, minibatch_size: int = 256):
        """
        Esegue update PPO con più epoche e minibatch sul batch corrente.
        Le azioni legali restano passate come array globale (ragged via offset/len per sample).
        """
        num_samples = len(batch['obs'])
        last_info = {}
        avg_kl_acc, avg_clip_acc, count_mb = 0.0, 0.0, 0
        early_stop = False

        def _grad_norm(params):
            total_sq = torch.zeros((), device=device)
            for p in params:
                if p.grad is not None:
                    total_sq = total_sq + p.grad.data.norm(2).pow(2)
            return float(total_sq.sqrt().item())

        for _ in range(epochs):
            perm = torch.randperm(num_samples).tolist()
            for start in range(0, num_samples, minibatch_size):
                idx = perm[start:start+minibatch_size]
                mini = {
                    'obs': batch['obs'][idx],
                    'act': batch['act'][idx],
                    'old_logp': batch['old_logp'][idx],
                    'ret': batch['ret'][idx],
                    'adv': batch['adv'][idx],
                    'legals': batch['legals'],  # globale
                    'legals_offset': batch['legals_offset'][idx],
                    'legals_count': batch['legals_count'][idx],
                    'chosen_index': batch['chosen_index'][idx],
                    'seat_team': batch.get('seat_team', None)[idx] if batch.get('seat_team', None) is not None else None,
                    'belief_summary': batch.get('belief_summary', None)[idx] if batch.get('belief_summary', None) is not None else None,
                }
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
                        loss, info = self.compute_loss(mini)
                    self.scaler.scale(loss).backward()
                    # Unscale prima del grad clip
                    self.scaler.unscale_(self.opt_actor)
                    self.scaler.unscale_(self.opt_critic)
                    # grad norm (prima del clip) per logging
                    gn_actor = _grad_norm(self.actor.parameters())
                    gn_critic = _grad_norm(self.critic.parameters())
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.scaler.step(self.opt_actor)
                    self.scaler.step(self.opt_critic)
                    self.scaler.update()
                else:
                    loss, info = self.compute_loss(mini)
                    loss.backward()
                    # grad norm (prima del clip) per logging
                    gn_actor = _grad_norm(self.actor.parameters())
                    gn_critic = _grad_norm(self.critic.parameters())
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_actor.step()
                    self.opt_critic.step()
                last_info = info
                self.update_steps += 1
                avg_kl_acc += info.get('approx_kl', 0.0)
                avg_clip_acc += info.get('clip_frac', 0.0)
                count_mb += 1
                last_info['grad_norm_actor'] = gn_actor
                last_info['grad_norm_critic'] = gn_critic
                last_info['lr_actor'] = self.opt_actor.param_groups[0]['lr']
                last_info['lr_critic'] = self.opt_critic.param_groups[0]['lr']
                # early stop per target KL
                if info.get('approx_kl', 0.0) > self.target_kl:
                    self._high_kl_count += 1
                    early_stop = True
                    break
                else:
                    self._high_kl_count = max(0, self._high_kl_count - 1)
            # Step any schedulers
            for sch in self._lr_schedulers:
                sch.step()
            # entropy schedule opzionale
            if self._entropy_schedule is not None:
                try:
                    self.entropy_coef = float(self._entropy_schedule(self.update_steps))
                except Exception:
                    pass
            if early_stop:
                break
        # riduzione LR automatica quando KL alto ripetuto
        if self._high_kl_count >= self._high_kl_patience:
            for opt in (self.opt_actor, self.opt_critic):
                for g in opt.param_groups:
                    g['lr'] = g['lr'] * self._lr_decay_factor
            self._high_kl_count = 0
            last_info['lr_reduced'] = True
        # medie su minibatch
        if count_mb > 0:
            last_info['avg_kl'] = avg_kl_acc / count_mb
            last_info['avg_clip_frac'] = avg_clip_acc / count_mb
            last_info['early_stop'] = early_stop
        return last_info

    def add_lr_schedulers(self, actor_scheduler, critic_scheduler):
        self._lr_schedulers = [sch for sch in [actor_scheduler, critic_scheduler] if sch is not None]

    def set_entropy_schedule(self, schedule_fn):
        """Imposta una funzione schedule_fn(step)->entropy_coef."""
        self._entropy_schedule = schedule_fn

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor': self.opt_actor.state_dict(),
            'opt_critic': self.opt_critic.state_dict(),
            'run_config': self.run_config,
            'update_steps': self.update_steps,
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        try:
            self.opt_actor.load_state_dict(ckpt['opt_actor'])
            self.opt_critic.load_state_dict(ckpt['opt_critic'])
        except Exception:
            pass
        self.run_config = ckpt.get('run_config', self.run_config)
        self.update_steps = ckpt.get('update_steps', 0)


