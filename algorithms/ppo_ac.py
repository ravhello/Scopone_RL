import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from contextlib import nullcontext

from models.action_conditioned import ActionConditionedActor, CentralValueNet, StateEncoderCompact

import os as _os
_DEVICE_STR = _os.environ.get(
    'SCOPONE_DEVICE',
    ('cuda' if torch.cuda.is_available() and _os.environ.get('TESTS_FORCE_CPU') != '1' else 'cpu')
)
device = torch.device(_DEVICE_STR)
autocast_device = device.type
_AMP_DTYPE = _os.environ.get('AMP_DTYPE', 'fp16')
autocast_dtype = torch.bfloat16 if _AMP_DTYPE == 'bf16' else torch.float16


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
        shared_enc = StateEncoderCompact()
        self.actor = ActionConditionedActor(obs_dim, action_dim, state_encoder=shared_enc)
        self.critic = CentralValueNet(obs_dim, state_encoder=shared_enc)
        # esecuzione diretta (rimosso torch.compile)

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

        # AMP GradScaler su CUDA; disabilitato su CPU
        self.scaler = None
        if device.type == 'cuda':
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except Exception:
                try:
                    self.scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    self.scaler = None

    @torch.inference_mode()
    def select_action(self, obs, legal_actions: List, seat_team_vec = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # belief handled internally by the actor

        cm = torch.autocast(device_type=autocast_device, dtype=autocast_dtype) if device.type == 'cuda' else nullcontext()
        with cm:
            # Scoring simultaneo di tutte le azioni legali via fattorizzazione
            state_proj = self.actor.compute_state_proj(obs_t, st)  # (1,64)
            card_emb = self.actor.card_emb_play.to(device)
            card_logits_all = torch.matmul(state_proj, card_emb.t()).squeeze(0)  # (40)
            played_ids_all = torch.argmax(actions_t[:, :40], dim=1)  # (A)
            # logp carta con mask sulle sole carte presenti nei legali
            allowed_mask = torch.zeros(40, dtype=torch.bool, device=device)
            allowed_mask[played_ids_all] = True
            masked_card_logits = card_logits_all.masked_fill(~allowed_mask, float('-inf'))
            logp_cards = torch.log_softmax(masked_card_logits, dim=0)  # (40)
            # capture logits per-legal
            a_emb = self.actor.action_enc(actions_t)  # (A,64)
            cap_logits = torch.matmul(a_emb, state_proj.squeeze(0))  # (A)
            # log-softmax within group (card)
            group_ids = played_ids_all  # (A)
            num_groups = 40
            group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
            try:
                group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
            except Exception:
                tmp = torch.zeros_like(group_max)
                tmp.index_copy_(0, group_ids, cap_logits)
                group_max = torch.maximum(group_max, tmp)
            gmax_per_legal = group_max[group_ids]
            exp_shifted = torch.exp(cap_logits - gmax_per_legal)
            group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
            group_sum.index_add_(0, group_ids, exp_shifted)
            lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
            logp_cap_per_legal = cap_logits - lse_per_legal  # (A)
            # logp totale per legal = logp(card played) + logp(capture|card)
            logp_totals = logp_cards[played_ids_all] + logp_cap_per_legal  # (A)
            probs = torch.softmax(logp_totals, dim=0)
            # Sanitize before sampling: clamp negatives/NaN, renormalize; fallback to uniform
            probs = probs.nan_to_num(0.0)
            probs = torch.clamp(probs, min=0.0)
            s = probs.sum()
            if not torch.isfinite(s) or s <= 0:
                A = probs.numel()
                probs = torch.full_like(probs, 1.0 / max(1, A))
            else:
                probs = probs / s
            try:
                idx_t = torch.multinomial(probs, num_samples=1).squeeze(0)
            except Exception:
                idx_t = torch.argmax(probs)
            logp_total = logp_totals[idx_t].detach()
        # Move chosen action and metadata back to CPU for env.step
        chosen_act = actions_t[idx_t].detach().to('cpu', non_blocking=False)
        return chosen_act, logp_total.to('cpu'), idx_t.detach().to('cpu')

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
        act = to_cuda_nb(batch['act'], torch.float32)
        old_logp = to_f32(batch['old_logp'])
        ret = to_f32(batch['ret'])
        adv = to_f32(batch['adv'])
        legals = to_cuda_nb(batch['legals'], torch.float32)
        offs = to_cuda_nb(batch['legals_offset'], torch.long)
        cnts = to_cuda_nb(batch['legals_count'], torch.long)
        chosen_idx = to_cuda_nb(batch['chosen_index'], torch.long)
        # distillazione MCTS (targets raggruppati per sample): policy piatta e peso per-sample
        mcts_policy_flat = to_cuda_nb(batch.get('mcts_policy', torch.zeros((0,), dtype=torch.float32)), torch.float32)
        mcts_weight = to_cuda_nb(batch.get('mcts_weight', torch.zeros((0,), dtype=torch.float32)), torch.float32)

        # Filtra eventuali sample senza azioni legali per evitare NaN
        valid_mask = cnts > 0
        if not bool(valid_mask.all()):
            if not bool(valid_mask.any()):
                zero = torch.tensor(0.0, device=device)
                return zero, {'loss_pi': 0.0, 'loss_v': 0.0, 'entropy': 0.0, 'approx_kl': 0.0, 'clip_frac': 0.0}
            obs = obs[valid_mask]
            seat = seat[valid_mask]
            act = act[valid_mask]
            old_logp = old_logp[valid_mask]
            ret = ret[valid_mask]
            adv = adv[valid_mask]
            offs = offs[valid_mask]
            cnts = cnts[valid_mask]
            chosen_idx = chosen_idx[valid_mask]
        B = obs.size(0)
        # Prepara indici legal per minibatch usando offs/cnts contro legals globali
        max_cnt = int(cnts.max().item()) if B > 0 else 0
        row_idx = torch.arange(B, device=device, dtype=torch.long)
        # State projection e logits per carta
        state_proj = self.actor.compute_state_proj(obs, seat)  # (B,64)
        card_emb = self.actor.card_emb_play.to(device)
        card_logits_all = torch.matmul(state_proj, card_emb.t())       # (B,40)
        if max_cnt > 0:
            pos = torch.arange(max_cnt, device=device, dtype=torch.long)
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts.unsqueeze(1)
            abs_idx_2d = offs.unsqueeze(1) + rel_pos_2d
            abs_idx = abs_idx_2d[mask]
            sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            legals_mb = legals[abs_idx].contiguous()                   # (M_mb,80)
            played_ids_mb = torch.argmax(legals_mb[:, :40], dim=1)     # (M_mb)
            # mask per carte consentite per sample
            card_mask = torch.zeros((B, 40), dtype=torch.bool, device=device)
            card_mask[sample_idx_per_legal, played_ids_mb] = True
            masked_card_logits = card_logits_all.masked_fill(~card_mask, float('-inf'))
            logp_cards = torch.log_softmax(masked_card_logits, dim=1)   # (B,40)
            # chosen abs indices e played ids
            chosen_clamped = torch.minimum(chosen_idx, (cnts - 1).clamp_min(0))
            chosen_abs_idx = (offs + chosen_clamped)
            total_legals = legals.size(0)
            pos_map = torch.full((total_legals,), -1, dtype=torch.long, device=device)
            pos_map[abs_idx] = torch.arange(abs_idx.numel(), device=device, dtype=torch.long)
            chosen_pos = pos_map[chosen_abs_idx]
            played_ids_all = torch.argmax(legals[:, :40], dim=1)
            chosen_card_ids = played_ids_all[chosen_abs_idx]
            logp_card = logp_cards[row_idx, chosen_card_ids]
            # capture logits per-legal via action embedding
            a_emb_mb = self.actor.action_enc(legals_mb)                # (M_mb,64)
            cap_logits = (a_emb_mb * state_proj[sample_idx_per_legal]).sum(dim=1)
            # segment logsumexp per gruppo (sample, card)
            group_ids = sample_idx_per_legal * 40 + played_ids_mb
            num_groups = B * 40
            group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
            try:
                group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
            except Exception:
                # fallback semplice
                tmp = torch.zeros_like(group_max)
                tmp.index_copy_(0, group_ids, cap_logits)
                group_max = torch.maximum(group_max, tmp)
            gmax_per_legal = group_max[group_ids]
            exp_shifted = torch.exp(cap_logits - gmax_per_legal)
            group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
            group_sum.index_add_(0, group_ids, exp_shifted)
            lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
            logp_cap_per_legal = cap_logits - lse_per_legal
            logp_cap = logp_cap_per_legal[chosen_pos]
            logp_new = logp_card + logp_cap
            # Distribuzione completa sui legali per entropia/KL
            logp_total_per_legal = logp_cards[sample_idx_per_legal, played_ids_mb] + logp_cap_per_legal
            logp_total_padded = torch.full((B, max_cnt), float('-inf'), dtype=cap_logits.dtype, device=device)
            logp_total_padded[mask] = logp_total_per_legal
            probs_group = torch.exp(logp_total_padded)
            entropy = (-(probs_group * logp_total_padded).nan_to_num(0.0).sum(dim=1)).mean()
        else:
            logp_new = torch.zeros((B,), device=device, dtype=state_proj.dtype)
            entropy = torch.tensor(0.0, device=device)

        ratio = torch.exp(logp_new - old_logp)
        clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clipped)).mean()

        v = self.critic(obs, seat)
        if self.value_clip is not None and self.value_clip > 0:
            v_clipped = torch.clamp(v, ret - self.value_clip, ret + self.value_clip)
            loss_v = torch.max((v - ret) ** 2, (v_clipped - ret) ** 2).mean()
        else:
            loss_v = nn.MSELoss()(v, ret)

        # Distillazione MCTS: costruisci target per-gruppo da mcts_policy_flat (raggruppato con offs/cnts)
        distill_loss = torch.tensor(0.0, device=device)
        # Loss ausiliaria per BeliefNet: cross-entropy 3x40 verso mani reali altrui (mascherando carte visibili)
        belief_aux = torch.tensor(0.0, device=device)
        if max_cnt > 0 and mcts_policy_flat.numel() >= int(cnts.sum().item()) and mcts_weight.numel() == B:
            # Ricostruisci (B, max_cnt) target evitando contributi sui padding (restano a 0)
            target = torch.full((B, max_cnt), 0.0, device=device, dtype=torch.float32)
            start = 0
            for i in range(B):
                ni = int(cnts[i].item())
                if ni > 0:
                    target[i, :ni] = mcts_policy_flat[start:start+ni]
                    start += ni
            # KL(pi_target || pi_actor) solo per posizioni con target>0 → niente 0 * (-inf)
            eps = 1e-8
            mask_pos = (target > 0)
            safe_log_t = torch.zeros_like(target)
            safe_log_t[mask_pos] = torch.log(torch.clamp(target[mask_pos], min=eps))
            diff = safe_log_t - logp_total_padded
            kl_per_row = (target * diff).sum(dim=1)
            # Peso per incertezza calcolato sulle posizioni valide
            ent_row = torch.zeros_like(kl_per_row)
            if int(mask_pos.sum().item()) > 0:
                log_t_pos = torch.zeros_like(target)
                log_t_pos[mask_pos] = torch.log(torch.clamp(target[mask_pos], min=eps))
                ent_row = (-(target * log_t_pos).sum(dim=1))
            denom_h = torch.log(torch.clamp_min(cnts.to(torch.float32), 1.0))
            ent_norm = torch.where(denom_h > 0, ent_row / denom_h, torch.zeros_like(ent_row))
            w_unc = torch.clamp(ent_norm, 0.0, 1.0)
            # Pesa solo i sample con MCTS attivo
            w = (mcts_weight.clamp(0.0, 1.0) * w_unc)
            if w.sum() > 0:
                distill_loss = (kl_per_row * w).sum() / torch.clamp_min(w.sum(), 1.0)
        # Prepara target belief supervision (se batch fornisce mani reali degli altri)
        real_hands = batch.get('others_hands', None)  # shape (B,3,40) one-hot o multi-hot per altri giocatori
        if real_hands is not None:
            rh = to_cuda_nb(real_hands, torch.float32)
            # calcola logits/probs dal BeliefNet dell'actor per il batch
            with torch.no_grad():
                state_feat_all = self.actor.state_enc(obs, seat)
            logits_b = self.actor.belief_net(state_feat_all)  # (B,120)
            # visible mask da obs per masking nella CE
            hand_table = obs[:, :83]
            hand_mask = hand_table[:, :40] > 0.5
            table_mask = hand_table[:, 43:83] > 0.5
            captured = obs[:, 83:165]
            cap0_mask = captured[:, :40] > 0.5
            cap1_mask = captured[:, 40:80] > 0.5
            visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)  # (B,40)
            Bsz = logits_b.size(0)
            logits_3x40 = logits_b.view(Bsz, 3, 40)
            # softmax over players dim
            log_probs = torch.log_softmax(logits_3x40, dim=1)
            # mask visible cards: zero their contribution
            m = (~visible_mask).to(log_probs.dtype).unsqueeze(1)  # True on unknown
            ce_per_card = -(rh * log_probs).sum(dim=1)  # (B,40)
            ce_per_card = ce_per_card * m.squeeze(1)
            denom = m.sum(dim=(1,2)).clamp_min(1.0) if m.dim()==3 else m.sum(dim=2).clamp_min(1.0)
            belief_aux = (ce_per_card.sum(dim=1) / torch.clamp_min((~visible_mask).sum(dim=1).to(torch.float32), 1.0)).mean()

        # Schedule coefficiente distillazione
        distill_coef_base = float(_os.environ.get('DISTILL_COEF', '0.1'))
        warm = int(_os.environ.get('DISTILL_WARMUP', '100'))
        coef = 0.0 if self.update_steps < warm else distill_coef_base
        # coefficiente belief aux
        belief_coef = float(_os.environ.get('BELIEF_AUX_COEF', '0.1'))
        loss = loss_pi + self.value_coef * loss_v - self.entropy_coef * entropy + coef * distill_loss + belief_coef * belief_aux
        approx_kl = (old_logp - logp_new).mean().abs()
        # clip fraction reale: frazione di sample con |ratio-1| > clip_ratio
        if B > 0:
            clip_frac = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()
        else:
            clip_frac = torch.tensor(0.0, device=device)
        # Restituisci TENSORS su device; conversione a float avverrà nel trainer in un'unica sync
        return loss, {
            'loss_pi': loss_pi.detach(),
            'loss_v': loss_v.detach(),
            'entropy': entropy.detach(),
            'approx_kl': approx_kl.detach(),
            'clip_frac': clip_frac.detach(),
            'distill_kl': distill_loss.detach(),
            'belief_aux': belief_aux.detach()
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
                    'others_hands': sel(batch['others_hands']) if batch.get('others_hands', None) is not None else None,
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


