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

        # Optimizers with fused/foreach when available to reduce kernel launches
        try:
            self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr, fused=True)
        except TypeError:
            self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr, foreach=True)
        try:
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr, fused=True)
        except TypeError:
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr, foreach=True)
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
    def select_action(self, obs, legal_actions: List, seat_team_vec = None, belief_summary = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(legal_actions) == 0:
            raise ValueError("No legal actions")
        # Prepare CPU-pinned → CUDA non_blocking transfers for per-step inference
        if torch.is_tensor(obs):
            obs_cpu = obs.detach().to(device='cpu', dtype=torch.float32)
        else:
            obs_cpu = torch.as_tensor(obs, dtype=torch.float32, device='cpu')
        obs_t = obs_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)

        if len(legal_actions) > 0 and torch.is_tensor(legal_actions[0]):
            actions_cpu = torch.stack(legal_actions).detach().to(device='cpu', dtype=torch.float32)
        else:
            actions_cpu = torch.stack([
                (x.detach().to(device='cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device='cpu'))
            for x in legal_actions], dim=0)
        actions_t = actions_cpu.pin_memory().to(device=device, non_blocking=True)

        st = None
        if seat_team_vec is not None:
            if torch.is_tensor(seat_team_vec):
                st_cpu = seat_team_vec.detach().to('cpu', dtype=torch.float32)
            else:
                st_cpu = torch.as_tensor(seat_team_vec, dtype=torch.float32, device='cpu')
            st = st_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
        bsum_t = None
        if belief_summary is not None:
            if torch.is_tensor(belief_summary):
                b_cpu = belief_summary.detach().to('cpu', dtype=torch.float32)
            else:
                b_cpu = torch.as_tensor(belief_summary, dtype=torch.float32, device='cpu')
            bsum_t = b_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = self.actor(obs_t, actions_t, st, bsum_t)
        logp = torch.log_softmax(logits, dim=0)
        idx_t = torch.multinomial(torch.exp(logp), num_samples=1).squeeze(0)
        # Move chosen action and metadata back to CPU for env.step
        chosen_act = actions_t[idx_t].detach().to('cpu', non_blocking=False)
        return chosen_act, logp[idx_t].detach().to('cpu'), idx_t.detach().to('cpu')

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

        # Accept CPU inputs; move to CUDA once in a pinned, non_blocking way
        def to_cuda_nb(x, dtype):
            if torch.is_tensor(x):
                x_cpu = x.detach().to('cpu', dtype=dtype)
            else:
                x_cpu = torch.as_tensor(x, dtype=dtype, device='cpu')
            return x_cpu.pin_memory().to(device=device, non_blocking=True)

        obs = to_cuda_nb(batch['obs'], torch.float32)
        seat = to_cuda_nb(batch['seat_team'], torch.float32)
        belief = to_cuda_nb(batch['belief_summary'], torch.float32)
        act = to_cuda_nb(batch['act'], torch.float32)
        old_logp = to_f32(batch['old_logp'])
        ret = to_f32(batch['ret'])
        adv = to_f32(batch['adv'])
        legals = to_cuda_nb(batch['legals'], torch.float32)
        offs = to_cuda_nb(batch['legals_offset'], torch.long)
        cnts = to_cuda_nb(batch['legals_count'], torch.long)
        chosen_idx = to_cuda_nb(batch['chosen_index'], torch.long)

        # Filtra eventuali sample senza azioni legali per evitare NaN
        valid_mask = cnts > 0
        if not bool(valid_mask.all()):
            if not bool(valid_mask.any()):
                zero = torch.tensor(0.0, device=device)
                return zero, {'loss_pi': 0.0, 'loss_v': 0.0, 'entropy': 0.0, 'approx_kl': 0.0, 'clip_frac': 0.0}
            obs = obs[valid_mask]
            seat = seat[valid_mask]
            belief = belief[valid_mask]
            act = act[valid_mask]
            old_logp = old_logp[valid_mask]
            ret = ret[valid_mask]
            adv = adv[valid_mask]
            offs = offs[valid_mask]
            cnts = cnts[valid_mask]
            chosen_idx = chosen_idx[valid_mask]
        B = obs.size(0)
        # Forward actor una sola volta sul batch intero (logits pieni)
        # torch.compile could reduce overhead if available
        raw_logits = self.actor(obs, None, seat, belief)  # (B, action_dim) or (action_dim,)
        if raw_logits.dim() == 1:
            raw_logits = raw_logits.unsqueeze(0)
        # Prepara indici legal per minibatch usando offs/cnts contro legals globali
        max_cnt = int(cnts.max().item()) if B > 0 else 0
        if max_cnt > 0:
            pos = torch.arange(max_cnt, device=device, dtype=torch.long)
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)                    # (B, max_cnt)
            mask = rel_pos_2d < cnts.unsqueeze(1)                               # (B, max_cnt)
            rel_pos = rel_pos_2d[mask]                                          # (M_mb)
            sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            abs_idx_2d = offs.unsqueeze(1) + rel_pos_2d                         # (B, max_cnt)
            abs_idx = abs_idx_2d[mask]                                          # (M_mb)
            legals_mb = legals[abs_idx].contiguous()                            # (M_mb, 80)
            # Score legali per le rispettive osservazioni: dot product per riga
            legal_scores = (legals_mb * raw_logits[sample_idx_per_legal]).sum(dim=1)  # (M_mb)
            # Costruisci tensore padded (B, max_cnt) per softmax per-gruppo senza loop Python
            padded = torch.full((B, max_cnt), fill_value=-float('inf'), device=device, dtype=legal_scores.dtype)
            padded[mask] = legal_scores
        else:
            padded = torch.full((B, 0), fill_value=-float('inf'), device=device, dtype=raw_logits.dtype)
        # log-softmax per gruppo
        logp_group = torch.log_softmax(padded, dim=1)
        # estrai logp scelti
        row_idx = torch.arange(B, device=device, dtype=torch.long)
        # clamp chosen_idx to valid range per-row
        if max_cnt > 0:
            chosen_clamped = torch.minimum(chosen_idx, (cnts - 1).clamp_min(0))
            logp_new = logp_group[row_idx, chosen_clamped]
        else:
            logp_new = torch.zeros((B,), device=device, dtype=padded.dtype)
        # entropia media per gruppo: evita 0 * -inf applicando mask
        probs_group = torch.softmax(padded, dim=1)
        if max_cnt > 0:
            logp_group_masked = logp_group.masked_fill(~mask, 0)
            entropy = (-(probs_group * logp_group_masked).sum(dim=1)).mean()
        else:
            entropy = torch.tensor(0.0, device=device)

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
        # Restituisci TENSORS su device; conversione a float avverrà nel trainer in un'unica sync
        return loss, {
            'loss_pi': loss_pi.detach(),
            'loss_v': loss_v.detach(),
            'entropy': entropy.detach(),
            'approx_kl': approx_kl.detach(),
            'clip_frac': clip_frac.detach()
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
            return total_sq.sqrt()

        check_every = 8  # reduce CPU syncs for early-stop
        for ep in range(epochs):
            # Keep indices on CPU to index CPU-stored batch efficiently; tensors will be moved in compute_loss
            perm = torch.randperm(num_samples, device='cpu')
            for start in range(0, num_samples, minibatch_size):
                idx_t = perm[start:start+minibatch_size]
                # Use index_select to slice tensors by CUDA indices
                def sel(x):
                    # Ensure indexing on CPU tensors
                    if torch.is_tensor(x) and x.device.type != 'cpu':
                        x = x.detach().to('cpu')
                    return torch.index_select(x, 0, idx_t)
                mini = {
                    'obs': sel(batch['obs']),
                    'act': sel(batch['act']),
                    'old_logp': sel(batch['old_logp']),
                    'ret': sel(batch['ret']),
                    'adv': sel(batch['adv']),
                    'legals': batch['legals'],  # globale
                    'legals_offset': sel(batch['legals_offset']),
                    'legals_count': sel(batch['legals_count']),
                    'chosen_index': sel(batch['chosen_index']),
                    'seat_team': sel(batch['seat_team']) if batch.get('seat_team', None) is not None else None,
                    'belief_summary': sel(batch['belief_summary']) if batch.get('belief_summary', None) is not None else None,
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
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.scaler.step(self.opt_actor)
                    self.scaler.step(self.opt_critic)
                    self.scaler.update()
                else:
                    loss, info = self.compute_loss(mini)
                    loss.backward()
                    # grad norm (prima del clip) per logging
                    gn_actor = _grad_norm(self.actor.parameters())
                    gn_critic = _grad_norm(self.critic.parameters())
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.opt_actor.step()
                    self.opt_critic.step()
                last_info = info
                self.update_steps += 1
                avg_kl_acc += info.get('approx_kl', torch.tensor(0.0, device=device))
                avg_clip_acc += info.get('clip_frac', torch.tensor(0.0, device=device))
                count_mb += 1
                last_info['grad_norm_actor'] = gn_actor.detach()
                last_info['grad_norm_critic'] = gn_critic.detach()
                last_info['lr_actor'] = self.opt_actor.param_groups[0]['lr']
                last_info['lr_critic'] = self.opt_critic.param_groups[0]['lr']
                # early stop per target KL (controlla meno spesso per ridurre sync CPU)
                if (count_mb % check_every) == 0:
                    _kl = info.get('approx_kl', torch.tensor(0.0, device=device)).detach()
                    if bool((_kl > self.target_kl).item()):
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
            last_info['avg_kl'] = (avg_kl_acc / count_mb).detach()
            last_info['avg_clip_frac'] = (avg_clip_acc / count_mb).detach()
            last_info['early_stop'] = torch.tensor(1.0 if early_stop else 0.0, device=device)
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


