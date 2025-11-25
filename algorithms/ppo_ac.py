import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext

from models.action_conditioned import ActionConditionedActor, CentralValueNet, StateEncoderCompact
from utils.compile import maybe_compile_module, maybe_compile_function
from utils.torch_load import safe_torch_load
import os as _os
device = torch.device('cpu')
autocast_device = device.type
autocast_dtype = torch.float16
import os as _os
STRICT_CHECKS = (_os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1')


class ActionConditionedPPO:
    """
    Skeleton PPO per policy action-conditioned con mask implicito
    (passando solo le azioni legali). Non è CTDE completo, ma prepara la struttura.
    """
    def __init__(self,
                 obs_dim: int = 10823,
                 action_dim: int = 80,
                 lr: float = 0.0001989908416605655,
                 clip_ratio: float = 0.15,
                 value_coef: float = 1.0,
                 entropy_coef: float = 0.0069646788699474806,
                 value_clip: float = 0.1,
                 target_kl: float = 0.015,
                 k_history: int = None):
        shared_enc = StateEncoderCompact(k_history=k_history)
        self.actor = ActionConditionedActor(obs_dim, action_dim, state_encoder=shared_enc)
        self.critic = CentralValueNet(obs_dim, state_encoder=shared_enc)

        # Training device forced to CPU; disable CUDA/autocast branches
        self.train_device = torch.device('cpu')
        self.scaler = None

        # Unified compile: allow CPU and CUDA. Avoid compiling backward-heavy compute_loss on CPU.
        self.actor = maybe_compile_module(self.actor, name='ActionConditionedActor')
        self.critic = maybe_compile_module(self.critic, name='CentralValueNet')
        # Forward hotspots (safe on CPU/CUDA)
        try:
            self.actor.compute_state_proj = maybe_compile_function(self.actor.compute_state_proj, name='ActionConditionedActor.compute_state_proj')
        except Exception as e:
            raise RuntimeError('ppo.init.compile_compute_state_proj_failed') from e
        try:
            self._select_action_core = maybe_compile_function(self._select_action_core, name='ActionConditionedPPO._select_action_core')
        except Exception as e:
            raise RuntimeError('ppo.init.compile_select_action_core_failed') from e
        # Additional forward hotspots
        try:
            self.actor.compute_state_features = maybe_compile_function(self.actor.compute_state_features, name='ActionConditionedActor.compute_state_features')
        except Exception as e:
            raise RuntimeError('ppo.init.compile_compute_state_features_failed') from e
        try:
            self.actor.compute_state_proj_from_state = maybe_compile_function(self.actor.compute_state_proj_from_state, name='ActionConditionedActor.compute_state_proj_from_state')
        except Exception as e:
            raise RuntimeError('ppo.init.compile_compute_state_proj_from_state_failed') from e
        try:
            self.critic.forward_from_state = maybe_compile_function(self.critic.forward_from_state, name='CentralValueNet.forward_from_state')
        except Exception as e:
            raise RuntimeError('ppo.init.compile_critic_forward_from_state_failed') from e
        # Compile compute_loss as well (CPU/CUDA). We ensure stable strides/shapes in update() to support CPU.
        try:
            self.compute_loss = maybe_compile_function(self.compute_loss, name='ActionConditionedPPO.compute_loss')
        except Exception as e:
            raise RuntimeError('ppo.init.compile_compute_loss_failed') from e

        # Optimizers with deduplicated shared encoder params to avoid double updates
        shared_ids = {id(p) for p in self.actor.state_enc.parameters()}
        actor_params = list(self.actor.parameters())
        critic_params = [p for p in self.critic.parameters() if id(p) not in shared_ids]
        self.opt_actor = optim.Adam(actor_params, lr=lr, fused=False)
        self.opt_critic = optim.Adam(critic_params, lr=lr, fused=False)
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

        # Perf flags (CPU-only)
        torch.set_float32_matmul_precision('high')

        # CPU buffers only
        self._obs_cpu_pinned = None
        self._seat_cpu_pinned = None
        self._actions_cpu_pinned = None
        self._actions_cpu_capacity = 0

    def select_action(self, obs, legal_actions: List, seat_team_vec = None, profiling_bins: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(legal_actions) == 0:
            raise ValueError("No legal actions")
        # Validate legal actions dimensionality (accept list of (80,) or tensor (A,80))
        if torch.is_tensor(legal_actions):
            if legal_actions.dim() != 2 or legal_actions.size(1) != self.run_config['action_dim']:
                raise ValueError(f"legal actions tensor must be shape (A, 80), got {tuple(legal_actions.shape)}")
            if STRICT_CHECKS:
                # Validate binary structure of action encoding (hot-path disabled unless STRICT)
                la = legal_actions
                if (la.size(0) > 0) and (not torch.allclose(la[:, :40].sum(dim=1), torch.ones((la.size(0),), dtype=la.dtype, device=la.device))):
                    raise RuntimeError("select_action: each legal action must have exactly one played bit in [:40]")
                cap = la[:, 40:]
                if bool((((cap > 0.0 + 1e-6) & (cap < 1.0 - 1e-6)) | (cap < -1e-6) | (cap > 1.0 + 1e-6)).any().item() if torch.is_tensor(cap) else False):
                    raise RuntimeError("select_action: captured section must be binary (0/1)")
        elif isinstance(legal_actions, list) and len(legal_actions) > 0:
            la0 = legal_actions[0]
            if STRICT_CHECKS:
                if torch.is_tensor(la0):
                    if la0.dim() != 1 or la0.numel() != self.run_config['action_dim']:
                        raise ValueError(f"each legal action vector must be shape (80,), got {tuple(la0.shape)}")
                    if not torch.allclose(la0[:40].sum(), torch.tensor(1.0, dtype=la0.dtype, device=la0.device)):
                        raise RuntimeError("select_action: first legal action must have exactly one played bit in [:40]")
                else:
                    import numpy as _np
                    a0 = _np.asarray(la0)
                    if a0.ndim != 1 or a0.shape[0] != self.run_config['action_dim']:
                        raise ValueError(f"each legal action vector must be shape (80,), got {a0.shape}")
                    s = float(a0[:40].sum())
                    if abs(s - 1.0) > 1e-6:
                        raise RuntimeError("select_action: first legal action must have exactly one played bit in [:40]")
        # Accept obs on CPU (CPU-only path)
        if torch.is_tensor(obs):
            obs_t = obs.detach().to(dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
        else:
            o_cpu = torch.as_tensor(obs, dtype=torch.float32, device='cpu')
            if o_cpu.dim() == 1:
                o_cpu = o_cpu.unsqueeze(0)
            obs_t = o_cpu

        # Legal actions: always stage to CPU tensor
        if torch.is_tensor(legal_actions):
            if legal_actions.dim() != 2 or legal_actions.size(1) != self.run_config['action_dim']:
                raise ValueError(f"legal actions tensor must be shape (A, 80), got {tuple(legal_actions.shape)}")
            actions_t = legal_actions.detach().to(dtype=torch.float32)
        elif isinstance(legal_actions, list) and len(legal_actions) > 0:
            if torch.is_tensor(legal_actions[0]):
                stacked = torch.stack([a.detach().to(dtype=torch.float32) for a in legal_actions], dim=0)
            else:
                stacked = torch.as_tensor(legal_actions, dtype=torch.float32, device='cpu')
            if stacked.dim() != 2 or stacked.size(1) != self.run_config['action_dim']:
                raise ValueError(f"each legal action vector must be shape (80,), got {tuple(stacked.shape)}")
            actions_t = stacked
        else:
            raise RuntimeError('select_action expects non-empty legal_actions')
        # Validate each legal row has exactly one played bit
        ones_per_row = actions_t[:, :40].sum(dim=1)
        if STRICT_CHECKS:
            if not torch.allclose(ones_per_row, torch.ones_like(ones_per_row)):
                raise RuntimeError("select_action: each legal action must have exactly one played bit in [:40]")
        # Validate captured is binary only in STRICT mode
        if STRICT_CHECKS:
            cap = actions_t[:, 40:]
            cap_bad = ((cap > 0.0 + 1e-6) & (cap < 1.0 - 1e-6)) | (cap < -1e-6) | (cap > 1.0 + 1e-6)
            if bool(cap_bad.any().item() if torch.is_tensor(cap_bad) else cap_bad.any()):
                raise RuntimeError("select_action: captured section must be binary (0/1)")

        st = None
        if seat_team_vec is not None:
            if torch.is_tensor(seat_team_vec):
                st = seat_team_vec.detach().to(dtype=torch.float32)
            else:
                st = torch.as_tensor(seat_team_vec, dtype=torch.float32, device='cpu')
            if st.dim() == 1:
                st = st.unsqueeze(0)
            # Validate seat one-hot + team flags (after staging)
            if st is not None:
                if st.size(1) != 6:
                    raise ValueError("select_action: seat_team_vec must have shape (B,6)")
                if not (st[:, :4].sum(dim=1) == 1).all():
                    raise RuntimeError("select_action: seat one-hot invalid (sum != 1)")
                if ((st[:, 4:6] < 0) | (st[:, 4:6] > 1)).any():
                    raise RuntimeError("select_action: team flags out of [0,1]")
        # belief handled internally by the actor

        # inference_mode disables autograd and some dispatcher overhead vs no_grad
        with torch.inference_mode():
            if profiling_bins is not None:
                chosen_act_d, logp_total_d, idx_t_d = self._select_action_core_profiled(obs_t, actions_t, st, profiling_bins=profiling_bins)
            else:
                chosen_act_d, logp_total_d, idx_t_d = self._select_action_core(obs_t, actions_t, st)
        # Move chosen action and metadata back to CPU for env.step
        chosen_act = chosen_act_d.detach().detach()
        return chosen_act, logp_total_d.detach().detach(), idx_t_d.detach().detach()

    def _select_action_core(self, obs_t: torch.Tensor, actions_t: torch.Tensor, seat_team_t: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure compute core for select_action. Expects inputs already on device.
        Returns (chosen_action_tensor_on_device, logp_total_on_device, idx_on_device).
        """
        # Strict shape/device checks
        if obs_t.dim() != 2 or obs_t.size(0) != 1:
            raise ValueError(f"_select_action_core expects obs_t with shape (1, D), got {tuple(obs_t.shape)}")
        if actions_t.dim() != 2 or actions_t.size(1) != self.run_config['action_dim']:
            raise ValueError(f"_select_action_core expects actions_t with shape (A, {self.run_config['action_dim']}), got {tuple(actions_t.shape)}")
        if seat_team_t is not None and (seat_team_t.dim() != 2 or seat_team_t.size(1) != 6):
            raise ValueError(f"_select_action_core expects seat_team_t with shape (1, 6), got {None if seat_team_t is None else tuple(seat_team_t.shape)}")
        with nullcontext():
            # Scoring simultaneo via fattorizzazione: proiezione di stato e logits carta
            state_proj = self.actor.compute_state_proj(obs_t, seat_team_t)  # (1,64)
            # Logits carta su tutte le 40 carte (mantieni dentro autocast per allineamento dtype)
            card_logits_all = torch.matmul(state_proj, self.actor.card_emb_play.t()).squeeze(0)  # (40)
        # Early diagnostics on non-finite values
        if STRICT_CHECKS and (not torch.isfinite(state_proj).all()):
            bad = state_proj[~torch.isfinite(state_proj)]
            raise RuntimeError(f"Actor state_proj contains non-finite values (count={int(bad.numel())})")
        if STRICT_CHECKS and (not torch.isfinite(card_logits_all).all()):
            bad = card_logits_all[~torch.isfinite(card_logits_all)]
            raise RuntimeError(f"Actor card_logits_all contains non-finite values (count={int(bad.numel())})")
        # Estrai card id per ciascuna azione legale
        played_ids_all = torch.argmax(actions_t[:, :40], dim=1)  # (A)
        # Validate legal actions have exactly one played bit per row
        ones_per_row = actions_t[:, :40].sum(dim=1)
        if STRICT_CHECKS:
            if not torch.allclose(ones_per_row, torch.ones_like(ones_per_row)):
                raise RuntimeError(f"Invalid legal actions: expected exactly one played card per row, got sums={ones_per_row.tolist()}")

        # Two-stage sampling (equivalente alla softmax su logp_totals):
        # 1) campiona la carta tra quelle presenti nelle legali usando p_card ristretto alle carte ammissibili
        logp_cards_all = torch.log_softmax(card_logits_all, dim=0)  # (40)
        unique_cards, inv_idx = torch.unique(played_ids_all, sorted=False, return_inverse=True)
        logp_cards_allowed = logp_cards_all[unique_cards]  # (G)
        probs_card_allowed = torch.softmax(logp_cards_allowed, dim=0)  # rinormalizza su carte consentite
        # sample card
        cdf_c = torch.cumsum(probs_card_allowed, dim=0)
        # Avoid scalarization: clamp last element to valid positive value
        last_c = torch.clamp(cdf_c[-1], min=torch.finfo(cdf_c.dtype).eps)
        if STRICT_CHECKS:
            torch._assert(torch.isfinite(last_c), "Invalid card CDF: non-finite last element")
        u_c = torch.rand((), device=cdf_c.device, dtype=cdf_c.dtype) * last_c
        sel_card_pos = torch.searchsorted(cdf_c, u_c, right=True)
        sel_card_pos = torch.clamp(sel_card_pos, max=cdf_c.numel() - 1)
        sel_card_id = unique_cards[sel_card_pos]

        # 2) campiona la presa condizionata alla carta scelta, usando solo le azioni del gruppo
        group_mask = (played_ids_all == sel_card_id)
        group_idx = torch.nonzero(group_mask, as_tuple=False).flatten()
        if group_idx.numel() <= 0:
            # Non dovrebbe accadere: carta scelta deve comparire tra le legali
            raise RuntimeError("Two-stage sampling: empty group for selected card")
        actions_grp = actions_t[group_idx]
        a_tbl = self.actor.get_action_emb_table_cached(device=actions_t.device, dtype=state_proj.dtype)  # (80,64)
        a_emb_grp = actions_grp.to(dtype=a_tbl.dtype) @ a_tbl  # (Gk,64)
        if STRICT_CHECKS and (not torch.isfinite(a_emb_grp).all()):
            bad = a_emb_grp[~torch.isfinite(a_emb_grp)]
            raise RuntimeError(f"Action embeddings (group) contain non-finite values (count={int(bad.numel())})")
        cap_logits_grp = torch.matmul(a_emb_grp, state_proj.squeeze(0).to(dtype=a_emb_grp.dtype))  # (Gk)
        if STRICT_CHECKS and (not torch.isfinite(cap_logits_grp).all()):
            bad = cap_logits_grp[~torch.isfinite(cap_logits_grp)]
            raise RuntimeError(f"Capture logits (group) contain non-finite values (count={int(bad.numel())})")
        # softmax nel gruppo
        probs_cap_grp = torch.softmax(cap_logits_grp, dim=0)
        if STRICT_CHECKS and (not torch.isfinite(probs_cap_grp).all()):
            raise RuntimeError("select_action: non-finite capture group probabilities before sanitize")
        probs_cap_grp = probs_cap_grp.nan_to_num(0.0)
        s_g = probs_cap_grp.sum()
        if STRICT_CHECKS and ((not torch.isfinite(s_g)) or (s_g <= 0)):
            raise RuntimeError("Invalid group probabilities for capture selection")
        cdf_g = torch.cumsum(probs_cap_grp, dim=0)
        u_g = torch.rand((), device=cdf_g.device, dtype=cdf_g.dtype) * cdf_g[-1]
        idx_in_group = torch.searchsorted(cdf_g, u_g, right=True)
        idx_in_group = torch.clamp(idx_in_group, max=cdf_g.numel() - 1)

        # Indice assoluto e log-prob totale coerente con training (log-softmax su 40 carte + log-softmax nel gruppo)
        idx_t = group_idx[idx_in_group]
        logp_card_sel = logp_cards_all[sel_card_id]
        logp_cap_sel = torch.log_softmax(cap_logits_grp, dim=0)[idx_in_group]
        logp_total = (logp_card_sel + logp_cap_sel).detach()
        chosen_act = actions_t[idx_t].detach()
        return chosen_act, logp_total, idx_t

    def _select_action_core_profiled(self, obs_t: torch.Tensor, actions_t: torch.Tensor, seat_team_t: torch.Tensor = None,
                                     profiling_bins: Dict[str, float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if profiling_bins is None:
            return self._select_action_core(obs_t, actions_t, seat_team_t)
        profiling_bins.setdefault('state_proj', 0.0)
        profiling_bins.setdefault('action_enc', 0.0)
        profiling_bins.setdefault('sampling', 0.0)

        if obs_t.dim() != 2 or obs_t.size(0) != 1:
            raise ValueError(f"_select_action_core expects obs_t with shape (1, D), got {tuple(obs_t.shape)}")
        if actions_t.dim() != 2 or actions_t.size(1) != self.run_config['action_dim']:
            raise ValueError(f"_select_action_core expects actions_t with shape (A, {self.run_config['action_dim']}), got {tuple(actions_t.shape)}")
        if seat_team_t is not None and (seat_team_t.dim() != 2 or seat_team_t.size(1) != 6):
            raise ValueError(f"_select_action_core expects seat_team_t with shape (1, 6), got {None if seat_team_t is None else tuple(seat_team_t.shape)}")

        t_state = time.perf_counter()
        with nullcontext():
            state_proj = self.actor.compute_state_proj(obs_t, seat_team_t)  # (1,64)
            card_logits_all = torch.matmul(state_proj, self.actor.card_emb_play.t()).squeeze(0)  # (40)
        profiling_bins['state_proj'] += (time.perf_counter() - t_state)

        if STRICT_CHECKS and (not torch.isfinite(state_proj).all()):
            bad = state_proj[~torch.isfinite(state_proj)]
            raise RuntimeError(f"Actor state_proj contains non-finite values (count={int(bad.numel())})")
        if STRICT_CHECKS and (not torch.isfinite(card_logits_all).all()):
            bad = card_logits_all[~torch.isfinite(card_logits_all)]
            raise RuntimeError(f"Actor card_logits_all contains non-finite values (count={int(bad.numel())})")

        played_ids_all = torch.argmax(actions_t[:, :40], dim=1)
        ones_per_row = actions_t[:, :40].sum(dim=1)
        if STRICT_CHECKS:
            if not torch.allclose(ones_per_row, torch.ones_like(ones_per_row)):
                raise RuntimeError(f"Invalid legal actions: expected exactly one played card per row, got sums={ones_per_row.tolist()}")

        logp_cards_all = torch.log_softmax(card_logits_all, dim=0)  # (40)
        unique_cards, inv_idx = torch.unique(played_ids_all, sorted=False, return_inverse=True)
        logp_cards_allowed = logp_cards_all[unique_cards]
        probs_card_allowed = torch.softmax(logp_cards_allowed, dim=0)
        cdf_c = torch.cumsum(probs_card_allowed, dim=0)
        last_c = torch.clamp(cdf_c[-1], min=torch.finfo(cdf_c.dtype).eps)
        if STRICT_CHECKS:
            torch._assert(torch.isfinite(last_c), "Invalid card CDF: non-finite last element")
        u_c = torch.rand((), device=cdf_c.device, dtype=cdf_c.dtype) * last_c
        sel_card_pos = torch.searchsorted(cdf_c, u_c, right=True)
        sel_card_pos = torch.clamp(sel_card_pos, max=cdf_c.numel() - 1)
        sel_card_id = unique_cards[sel_card_pos]

        group_mask = (played_ids_all == sel_card_id)
        group_idx = torch.nonzero(group_mask, as_tuple=False).flatten()
        if group_idx.numel() <= 0:
            raise RuntimeError("Two-stage sampling: empty group for selected card")
        actions_grp = actions_t[group_idx]

        t_action = time.perf_counter()
        a_tbl = self.actor.get_action_emb_table_cached(device=actions_t.device, dtype=state_proj.dtype)
        a_emb_grp = actions_grp.to(dtype=a_tbl.dtype) @ a_tbl  # (Gk,64)
        profiling_bins['action_enc'] += (time.perf_counter() - t_action)

        if STRICT_CHECKS and (not torch.isfinite(a_emb_grp).all()):
            bad = a_emb_grp[~torch.isfinite(a_emb_grp)]
            raise RuntimeError(f"Action embeddings (group) contain non-finite values (count={int(bad.numel())})")

        t_sampling = time.perf_counter()
        cap_logits_grp = torch.matmul(a_emb_grp, state_proj.squeeze(0).to(dtype=a_emb_grp.dtype))  # (Gk)
        if STRICT_CHECKS and (not torch.isfinite(cap_logits_grp).all()):
            bad = cap_logits_grp[~torch.isfinite(cap_logits_grp)]
            raise RuntimeError(f"Capture logits (group) contain non-finite values (count={int(bad.numel())})")
        probs_cap_grp = torch.softmax(cap_logits_grp, dim=0)
        if STRICT_CHECKS and (not torch.isfinite(probs_cap_grp).all()):
            raise RuntimeError("select_action: non-finite capture group probabilities before sanitize")
        probs_cap_grp = probs_cap_grp.nan_to_num(0.0)
        s_g = probs_cap_grp.sum()
        if STRICT_CHECKS and ((not torch.isfinite(s_g)) or (s_g <= 0)):
            raise RuntimeError("Invalid group probabilities for capture selection")
        cdf_g = torch.cumsum(probs_cap_grp, dim=0)
        u_g = torch.rand((), device=cdf_g.device, dtype=cdf_g.dtype) * cdf_g[-1]
        idx_in_group = torch.searchsorted(cdf_g, u_g, right=True)
        idx_in_group = torch.clamp(idx_in_group, max=cdf_g.numel() - 1)

        idx_t = group_idx[idx_in_group]
        logp_card_sel = logp_cards_all[sel_card_id]
        logp_cap_sel = torch.log_softmax(cap_logits_grp, dim=0)[idx_in_group]
        logp_total = (logp_card_sel + logp_cap_sel).detach()
        chosen_act = actions_t[idx_t].detach()
        profiling_bins['sampling'] += (time.perf_counter() - t_sampling)
        return chosen_act, logp_total, idx_t

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
        def to_f32(x):
            if torch.is_tensor(x):
                if (x.device.type == 'cpu') and (x.dtype == torch.float32):
                    return x
                return x.detach().to(dtype=torch.float32)
            return torch.as_tensor(x, dtype=torch.float32, device='cpu')
        def to_long(x):
            if torch.is_tensor(x):
                if (x.device.type == 'cpu') and (x.dtype == torch.long):
                    return x
                return x.detach().to(dtype=torch.long)
            return torch.as_tensor(x, dtype=torch.long, device='cpu')

        # CPU-only compute path
        device = torch.device('cpu')

        def to_cpu(x, dtype):
            if torch.is_tensor(x):
                if (x.device.type == 'cpu') and (x.dtype == dtype):
                    return x
                return x.detach().to(dtype=dtype)
            return torch.as_tensor(x, dtype=dtype, device='cpu')

        # Required keys validation
        for key in ('obs','act','old_logp','ret','adv','legals','legals_offset','legals_count','chosen_index','seat_team'):
            if key not in batch:
                raise KeyError(f"compute_loss: missing batch key '{key}'")
        obs = to_cpu(batch['obs'], torch.float32)
        seat = to_cpu(batch['seat_team'], torch.float32)
        act = to_cpu(batch['act'], torch.float32)
        old_logp = to_cpu(batch['old_logp'], torch.float32)
        ret = to_cpu(batch['ret'], torch.float32)
        adv = to_cpu(batch['adv'], torch.float32)
        legals = to_cpu(batch['legals'], torch.float32)
        offs = to_cpu(batch['legals_offset'], torch.long)
        cnts = to_cpu(batch['legals_count'], torch.long)
        chosen_idx = to_cpu(batch['chosen_index'], torch.long)
        # Plausibility checks on old_logp early to catch bad batches
        if STRICT_CHECKS and (not torch.isfinite(old_logp).all()):
            raise RuntimeError("compute_loss: old_logp contains non-finite values")
        if STRICT_CHECKS and old_logp.numel() > 0:
            pos_any = (old_logp > 1.0e-6).any()
            small_any = (old_logp < -120.0).any()
            if bool(pos_any.detach().cpu().item()):
                mx = float(old_logp.max().detach().cpu().item())
                raise RuntimeError(f"compute_loss: old_logp contains positive values (max={mx})")
            if bool(small_any.detach().cpu().item()):
                mn = float(old_logp.min().detach().cpu().item())
                raise RuntimeError(f"compute_loss: old_logp too small (min={mn})")
        # Sanity on ragged indices
        if obs.size(0) != seat.size(0) or obs.size(0) != act.size(0):
            raise RuntimeError("compute_loss: batch size mismatch among obs/seat/act")
        if obs.size(0) != offs.size(0) or obs.size(0) != cnts.size(0) or obs.size(0) != chosen_idx.size(0):
            raise RuntimeError("compute_loss: ragged indices sizes mismatch with batch size")
        # Optional precomputed global action embeddings to avoid recomputation per minibatch
        a_emb_global = batch.get('a_emb_global', None)
        # distillazione MCTS (targets raggruppati per sample): policy piatta e peso per-sample
        mcts_policy_flat = to_cpu(batch.get('mcts_policy', torch.zeros((0,), dtype=torch.float32, device=device)), torch.float32)
        mcts_weight = to_cpu(batch.get('mcts_weight', torch.zeros((0,), dtype=torch.float32, device=device)), torch.float32)

        # Filtra eventuali sample senza azioni legali per evitare NaN
        valid_mask = cnts > 0
        if not bool(valid_mask.all()):
            if not bool(valid_mask.any()):
                if STRICT_CHECKS:
                    raise RuntimeError("compute_loss: batch has no samples with legal actions (cnts>0)")
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
        # Validate ragged structure before use
        if STRICT_CHECKS:
            neg_offs = (offs < 0).any()
            neg_cnts = (cnts < 0).any()
            if bool((neg_offs | neg_cnts).detach().cpu().item()):
                raise RuntimeError("compute_loss: negative offsets or counts in legals structure")
        if STRICT_CHECKS:
            if (B > 0) and (legals.size(0) > 0):
                last_sum = int((offs[-1] + cnts[-1]).detach().cpu().item())
                if last_sum > int(legals.size(0)):
                    raise RuntimeError("compute_loss: last window exceeds legals length")
        if STRICT_CHECKS:
            bad_low = (chosen_idx < 0).any()
            bad_high = ((chosen_idx >= cnts) & (cnts > 0)).any()
            if bool((bad_low | bad_high).detach().cpu().item()):
                raise RuntimeError("compute_loss: chosen_index out of range for some rows")
        row_idx = torch.arange(B, device=device, dtype=torch.long)
        # Compute state features once; reuse for actor and critic
        state_feat = self.actor.compute_state_features(obs, seat)  # (B,256)
        if STRICT_CHECKS and (not torch.isfinite(state_feat).all()):
            bad = state_feat[~torch.isfinite(state_feat)]
            raise RuntimeError(f"compute_loss: state_feat non-finite (count={int(bad.numel())})")
        # Precompute visible mask once and reuse
        hand_table = obs[:, :83]
        hand_mask = hand_table[:, :40] > 0.5
        table_mask = hand_table[:, 43:83] > 0.5
        captured = obs[:, 83:165]
        cap0_mask = captured[:, :40] > 0.5
        cap1_mask = captured[:, 40:80] > 0.5
        visible_mask_40 = (hand_mask | table_mask | cap0_mask | cap1_mask)
        # State projection e logits per carta
        state_proj = self.actor.compute_state_proj_from_state(state_feat, obs, visible_mask_40=visible_mask_40)  # (B,64)
        if STRICT_CHECKS and (not torch.isfinite(state_proj).all()):
            bad = state_proj[~torch.isfinite(state_proj)]
            raise RuntimeError(f"compute_loss: state_proj non-finite (count={int(bad.numel())})")
        card_logits_all = torch.matmul(state_proj, self.actor.card_emb_play.t())       # (B,40)
        # Guard extreme magnitudes before exp/softmax usage
        if STRICT_CHECKS and (not torch.isfinite(card_logits_all).all()):
            bad = card_logits_all[~torch.isfinite(card_logits_all)]
            raise RuntimeError(f"compute_loss: card_logits_all non-finite (count={int(bad.numel())})")
        max_abs_cl = float(card_logits_all.abs().max().item()) if card_logits_all.numel() > 0 else 0.0
        if max_abs_cl > 1e3:
            raise RuntimeError(f"compute_loss: card_logits_all magnitude too large (max_abs={max_abs_cl})")
        if max_cnt > 0:
            pos = torch.arange(max_cnt, device=device, dtype=torch.long)
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts.unsqueeze(1)
            abs_idx_2d = offs.unsqueeze(1) + rel_pos_2d
            abs_idx = abs_idx_2d[mask]
            sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            legals_mb = legals[abs_idx].contiguous()                   # (M_mb,80)
            ones_per_row = legals_mb[:, :40].sum(dim=1)
            if STRICT_CHECKS:
                if not torch.allclose(ones_per_row, torch.ones_like(ones_per_row)):
                    raise RuntimeError("compute_loss: legals_mb must have exactly one played bit per row in [:40]")
            played_ids_mb = torch.argmax(legals_mb[:, :40], dim=1)     # (M_mb)
            # Card log-prob restricted to allowed set per sample (two-stage policy)
            # Build allowed mask per sample over the 40 cards
            allowed_mask = torch.zeros((B, 40), dtype=torch.bool, device=device)
            allowed_mask[sample_idx_per_legal, played_ids_mb] = True
            # LSE over allowed set per sample (out-of-place; avoid in-place masking + exp)
            neg_inf = torch.full_like(card_logits_all, float('-inf'))
            masked_logits = torch.where(allowed_mask, card_logits_all, neg_inf)
            lse_allowed = torch.logsumexp(masked_logits, dim=1)
            # chosen abs indices e played ids (evita pos_map grande)
            chosen_clamped = torch.minimum(chosen_idx, (cnts - 1).clamp_min(0))
            chosen_abs_idx = (offs + chosen_clamped)                                 # (B)
            # posizioni relative nella maschera (per-legal all'interno del proprio sample)
            pos_in_sample = rel_pos_2d[mask]                                         # (M_mb)
            chosen_abs_idx_per_legal = chosen_abs_idx[sample_idx_per_legal]          # (M_mb)
            match = (abs_idx == chosen_abs_idx_per_legal)                            # (M_mb)
            # Raccogli posizioni scelte per ciascun sample tramite index_copy
            chosen_pos = torch.full((B,), -1, dtype=torch.long, device=device)
            if bool(match.any()):
                chosen_pos_vals = pos_in_sample[match]
                chosen_pos_idx = sample_idx_per_legal[match]
                chosen_pos.index_copy_(0, chosen_pos_idx, chosen_pos_vals)
            if STRICT_CHECKS:
                has_neg = (chosen_pos < 0).any() if chosen_pos.numel() > 0 else torch.tensor(False, device=device)
                if bool(has_neg.detach().cpu().item()):
                    bad_rows = torch.nonzero(chosen_pos < 0, as_tuple=False).flatten().tolist()
                    raise RuntimeError(f"compute_loss: chosen_pos mapping failed for rows {bad_rows}")
            played_ids_all = torch.argmax(legals[:, :40], dim=1)
            chosen_card_ids = played_ids_all[chosen_abs_idx]
            logp_card = card_logits_all[row_idx, chosen_card_ids] - lse_allowed[row_idx]
            # capture logits per-legal via action embedding
            # Prefer precomputed global embeddings to avoid recomputation per minibatch
            if a_emb_global is not None:
                a_emb_mb = a_emb_global[abs_idx]
            else:
                # In training, avoid cached table to keep gradients flowing
                if self.actor.training:
                    a_emb_mb = self.actor.action_enc(legals_mb)
                else:
                    a_tbl = self.actor.get_action_emb_table_cached(device=legals_mb.device, dtype=state_proj.dtype)
                    a_emb_mb = torch.matmul(legals_mb, a_tbl)             # (M_mb,64)
            if STRICT_CHECKS and (not torch.isfinite(a_emb_mb).all()):
                bad = a_emb_mb[~torch.isfinite(a_emb_mb)]
                raise RuntimeError(f"compute_loss: a_emb_mb non-finite (count={int(bad.numel())})")
            cap_logits = (a_emb_mb * state_proj[sample_idx_per_legal]).sum(dim=1)
            if STRICT_CHECKS and (not torch.isfinite(cap_logits).all()):
                bad = cap_logits[~torch.isfinite(cap_logits)]
                raise RuntimeError(f"compute_loss: cap_logits non-finite (count={int(bad.numel())})")
            # segment logsumexp per gruppo (sample, card)
            group_ids = sample_idx_per_legal * 40 + played_ids_mb
            num_groups = B * 40
            group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
            group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
            gmax_per_legal = group_max[group_ids]
            # Ensure dtype consistency under autocast (exp may return float32)
            exp_shifted = torch.exp(cap_logits - gmax_per_legal).to(cap_logits.dtype)
            group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
            group_sum.index_add_(0, group_ids, exp_shifted)
            lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
            logp_cap_per_legal = cap_logits - lse_per_legal
            logp_cap = logp_cap_per_legal[chosen_pos]
            if STRICT_CHECKS and (not torch.isfinite(logp_card).all()):
                raise RuntimeError("compute_loss: logp_card non-finite")
            if STRICT_CHECKS and (not torch.isfinite(logp_cap).all()):
                raise RuntimeError("compute_loss: logp_cap non-finite")
            logp_new = logp_card + logp_cap
            if not torch.isfinite(logp_new).all():
                raise RuntimeError("compute_loss: logp_new non-finite")
            # Distribuzione completa sui legali per entropia/KL (computata solo se serve)
            need_entropy = (float(self.entropy_coef) > 0.0)
            if need_entropy:
                # Entropia per-sample sui soli sample con legali: H = -Σ p * log p
                logp_cards_allowed_per_legal = (card_logits_all[sample_idx_per_legal, played_ids_mb] - lse_allowed[sample_idx_per_legal])
                logp_total_per_legal = logp_cards_allowed_per_legal + logp_cap_per_legal
                probs = torch.exp(logp_total_per_legal)
                neg_p_logp = -(probs * logp_total_per_legal)
                ent_per_row = torch.zeros((B,), dtype=neg_p_logp.dtype, device=device)
                ent_per_row.index_add_(0, sample_idx_per_legal, neg_p_logp)
                valid_rows = torch.zeros((B,), dtype=torch.bool, device=device)
                valid_rows.index_fill_(0, sample_idx_per_legal.unique(), True)
                denom = valid_rows.to(ent_per_row.dtype).sum().clamp_min(1.0)
                entropy = (ent_per_row[valid_rows].sum() / denom)
            else:
                entropy = torch.tensor(0.0, device=device)
            if not torch.isfinite(entropy).all():
                raise RuntimeError("compute_loss: entropy non-finite")
        else:
            logp_new = torch.zeros((B,), device=device, dtype=state_proj.dtype)
            entropy = torch.tensor(0.0, device=device)

        # Ratio: assert inputs finite and gaps not extreme; raise early with diagnostics
        if STRICT_CHECKS and (not torch.isfinite(logp_new).all()):
            raise RuntimeError("compute_loss: logp_new non-finite before ratio")
        if STRICT_CHECKS and (not torch.isfinite(old_logp).all()):
            raise RuntimeError("compute_loss: old_logp non-finite before ratio")
        diff = (logp_new - old_logp)
        if STRICT_CHECKS and (not torch.isfinite(diff).all()):
            raise RuntimeError("compute_loss: non-finite (logp_new - old_logp)")
        if diff.numel() > 0:
            max_gap = diff.abs().amax()
            torch._assert((max_gap <= 80.0), "compute_loss: log-prob gap too large; likely to overflow exp")
        ratio = torch.exp(diff)
        if STRICT_CHECKS and (not torch.isfinite(ratio).all()):
            raise RuntimeError("compute_loss: ratio non-finite (exp overflow)")
        clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        # Maschera off-policy per step MCTS: quando presente mcts_weight>0, escludi la loss di policy
        mcts_w = batch.get('mcts_weight', None)
        if mcts_w is not None:
            if not torch.is_tensor(mcts_w):
                mcts_w = torch.as_tensor(mcts_w, dtype=torch.float32, device=device)
            else:
                mcts_w = mcts_w.to(device=device, dtype=torch.float32)
            # costruiamo un mask 0/1 allineato al minibatch corrente (B,)
            if mcts_w.dim() > 1:
                mcts_w = mcts_w.view(-1)
            # clamp and invert: 1.0 per step senza MCTS, 0.0 per step con MCTS
            mask_no_mcts = (mcts_w <= 0.0).to(torch.float32)
            # applica la maschera alla parte di policy; somma eps per evitare divisione 0
            ppo_term = torch.min(ratio * adv, clipped) * mask_no_mcts
            denom = mask_no_mcts.sum().clamp_min(1.0)
            loss_pi = -(ppo_term.sum() / denom)
        else:
            loss_pi = -(torch.min(ratio * adv, clipped)).mean()
        # Guard against non-finite intermediates spilling into the loss
        for name, tensor in (("ratio", ratio), ("adv", adv), ("ret", ret)):
            if STRICT_CHECKS and (not torch.isfinite(tensor).all()):
                raise RuntimeError(f"compute_loss: non-finite {name}")

        # Passa others_hands (se disponibile) al critico per percorso CTDE opzionale
        v = self.critic.forward_from_state(state_feat, obs, batch.get('others_hands', None), visible_mask_40=visible_mask_40)
        if STRICT_CHECKS and (not torch.isfinite(v).all()):
            bad = v[~torch.isfinite(v)]
            raise RuntimeError(f"compute_loss: critic value non-finite (count={int(bad.numel())})")
        if self.value_clip is not None and self.value_clip > 0:
            v_clipped = torch.clamp(v, ret - self.value_clip, ret + self.value_clip)
            loss_v = torch.max((v - ret) ** 2, (v_clipped - ret) ** 2).mean()
        else:
            loss_v = nn.MSELoss()(v, ret)

        # Schedula coefficienti (prima per evitare calcoli inutili)
        distill_coef_base = float(_os.environ.get('DISTILL_COEF', '0.1'))
        warm = int(_os.environ.get('DISTILL_WARMUP', '100'))
        coef = 0.0 if self.update_steps < warm else distill_coef_base
        belief_coef = float(_os.environ.get('BELIEF_AUX_COEF', '0.1'))

        # Distillazione MCTS: costruisci target per-gruppo solo se necessario
        distill_loss = torch.tensor(0.0, device=device)
        # Loss ausiliaria per BeliefNet solo se necessario
        belief_aux = torch.tensor(0.0, device=device)
        if (coef > 0.0) and max_cnt > 0 and mcts_weight.numel() == B and (mcts_weight.sum() > 0) and mcts_policy_flat.numel() >= int(cnts.sum().item()):
            if STRICT_CHECKS and (not torch.isfinite(mcts_policy_flat).all()):
                raise RuntimeError("compute_loss: mcts_policy_flat non-finite")
            # Ricostruisci (B, max_cnt) target evitando loop Python: usa masked_scatter
            target = torch.zeros((B, max_cnt), device=device, dtype=torch.float32)
            valid_mask = torch.arange(max_cnt, device=device).unsqueeze(0).expand(B, max_cnt) < cnts.unsqueeze(1)
            flat_len = int(cnts.sum().item())
            target.masked_scatter_(valid_mask, mcts_policy_flat[:flat_len])
            # KL(pi_target || pi_actor) solo per posizioni con target>0 → niente 0 * (-inf)
            eps = 1e-8
            mask_pos = (target > 0)
            safe_log_t = torch.zeros_like(target)
            safe_log_t[mask_pos] = torch.log(torch.clamp(target[mask_pos], min=eps))
            # Costruisci sempre i log-prob totali per-legal paddati (usati dalla KL MCTS)
            logp_cards_allowed_per_legal = (card_logits_all[sample_idx_per_legal, played_ids_mb] - lse_allowed[sample_idx_per_legal])
            logp_total_per_legal = logp_cards_allowed_per_legal + logp_cap_per_legal
            logp_total_padded = torch.full((B, max_cnt), float('-inf'), dtype=cap_logits.dtype, device=device)
            logp_total_padded[mask] = logp_total_per_legal
            diff = safe_log_t - logp_total_padded
            # Avoid 0 * inf -> NaN: only accumulate where target > 0
            diff = torch.where(mask_pos, diff, torch.zeros_like(diff))
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
            if STRICT_CHECKS and (not torch.isfinite(distill_loss).all()):
                raise RuntimeError("compute_loss: distill_loss non-finite")
        # Prepara target belief supervision (se batch fornisce mani reali degli altri)
        real_hands = batch.get('others_hands', None)  # shape (B,3,40) one-hot o multi-hot per altri giocatori
        if (belief_coef > 0.0) and (real_hands is not None):
            rh = to_cpu(real_hands, torch.float32)
            # riusa state_feat già calcolato e la visible_mask_40 già costruita
            logits_b = self.actor.belief_net(state_feat)  # (B,120)
            if STRICT_CHECKS and (not torch.isfinite(logits_b).all()):
                raise RuntimeError("compute_loss: belief logits non-finite")
            visible_mask = visible_mask_40  # (B,40)
            Bsz = logits_b.size(0)
            logits_3x40 = logits_b.view(Bsz, 3, 40)
            # softmax over players dim
            log_probs = torch.log_softmax(logits_3x40, dim=1)
            # mask visible cards: zero their contribution, then average over unknown only
            m = (~visible_mask).to(log_probs.dtype).unsqueeze(1)  # True on unknown
            ce_per_card = -(rh * log_probs).sum(dim=1)  # (B,40)
            ce_per_card = ce_per_card * m.squeeze(1)
            denom = m.sum(dim=(1,2)).clamp_min(1.0)
            belief_aux = (ce_per_card.sum(dim=1) / denom).mean()

        loss = loss_pi + self.value_coef * loss_v - self.entropy_coef * entropy + coef * distill_loss + belief_coef * belief_aux
        approx_kl = (old_logp - logp_new).mean()
        # Finite checks on core scalars
        for name, tensor in (
            ('loss', loss),
            ('loss_pi', loss_pi),
            ('loss_v', loss_v),
            ('entropy', entropy),
            ('approx_kl', approx_kl),
        ):
            if STRICT_CHECKS and (not torch.isfinite(tensor).all()):
                raise RuntimeError(f"compute_loss: non-finite {name}")
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
        # Modelli già su CPU: nessun move necessario
        models_were_cpu = False
        num_samples = len(batch['obs'])
        last_info = {}
        avg_kl_acc, avg_clip_acc, count_mb = 0.0, 0.0, 0
        early_stop = False

        def _grad_norm(params):
            # Deprecated: we now use the value returned by clip_grad_norm_ to cut kernel launches
            total_sq = torch.zeros((), device=device)
            for p in params:
                if p.grad is not None:
                    total_sq = total_sq + p.grad.data.norm(2).pow(2)
            return total_sq.sqrt()

        check_every = 1  # sample ogni minibatch: più controlli -> KL aderente al target
        # Stage intero batch su CPU (unico device supportato)
        batch_cpu = {}
        for k, v in batch.items():
            if k in ('routing_log',):
                continue
            if torch.is_tensor(v):
                dtype = torch.float32 if k in ('obs', 'act', 'ret', 'adv') else v.dtype
                batch_cpu[k] = v.detach().to(device='cpu', dtype=dtype)
            else:
                batch_cpu[k] = v
        if 'legals' in batch_cpu:
            lv = batch_cpu['legals']
            if not (torch.is_tensor(lv) and lv.device.type == device.type and lv.dtype == torch.float32):
                batch_cpu['legals'] = lv.to(device=device, dtype=torch.float32)
            if not self.actor.training:
                try:
                    batch_cpu['a_emb_global'] = self.actor.action_enc(batch_cpu['legals']).detach()
                except Exception as e:
                    raise RuntimeError("Failed to precompute global action embeddings (a_emb_global)") from e
            else:
                batch_cpu['a_emb_global'] = None

        global_max_cnt = int(batch_cpu['legals_count'].max().item()) if num_samples > 0 else 0
        pos_global = (torch.arange(global_max_cnt, device=device, dtype=torch.long) if global_max_cnt > 0 else torch.zeros((0,), device=device, dtype=torch.long))

        for ep in range(epochs):
            # Indici su device; scegli un minibatch_size effettivo che divida num_samples
            perm = torch.randperm(num_samples, device=device)
            mb_eff = int(minibatch_size)
            if num_samples > 0 and mb_eff > 0 and (num_samples % mb_eff) != 0:
                d = min(mb_eff, num_samples)
                found = False
                for k in range(d, 0, -1):
                    if (num_samples % k) == 0:
                        mb_eff = k
                        found = True
                        break
                if not found:
                    mb_eff = max(1, num_samples)
            for start in range(0, num_samples, mb_eff):
                idx_t = perm[start:start+mb_eff]
                # Slice helper su CPU
                def sel_cpu(x):
                    return torch.index_select(x, 0, idx_t)
                # Seleziona sottovettori offs/cnts del minibatch (no fallback)
                offs_mb = sel_cpu(batch_cpu['legals_offset'])
                cnts_mb = sel_cpu(batch_cpu['legals_count'])
                B_mb = int(cnts_mb.size(0))
                # Fissa la dimensione al massimo globale per stabilità delle forme compilate
                max_cnt_mb = int(global_max_cnt)
                # Verifica presenza mcts_weight
                if batch_cpu.get('mcts_weight', None) is None:
                    raise RuntimeError('update: missing mcts_weight in batch')
                mcts_weight_mb = sel_cpu(batch_cpu['mcts_weight'])
                # Costruisci indice assoluto sui legali globali per estrarre la porzione di mcts_policy, solo se necessario
                if max_cnt_mb > 0 and bool((mcts_weight_mb > 0).any().item()):
                    if batch_cpu.get('mcts_policy', None) is None:
                        raise RuntimeError("update: 'mcts_policy' is required when mcts_weight>0 in minibatch")
                    if batch_cpu['mcts_policy'].dim() != 1:
                        raise RuntimeError('update: mcts_policy must be 1D flat vector')
                    if int(batch_cpu['mcts_policy'].numel()) < int(batch_cpu['legals'].size(0)):
                        raise RuntimeError('update: mcts_policy length smaller than total legals')
                    pos = pos_global
                    rel_pos_2d = (pos.unsqueeze(0).expand(B_mb, max_cnt_mb) if max_cnt_mb > 0 else torch.zeros((B_mb, 0), device=device, dtype=torch.long))
                    mask = rel_pos_2d < cnts_mb.unsqueeze(1)
                    abs_idx = (offs_mb.unsqueeze(1) + rel_pos_2d)[mask]
                    mcts_policy_mb = batch_cpu['mcts_policy'][abs_idx].contiguous()
                else:
                    mcts_policy_mb = torch.zeros((0,), dtype=torch.float32, device=device)

                mini = {
                    'obs': sel_cpu(batch_cpu['obs']),
                    'act': sel_cpu(batch_cpu['act']),
                    'old_logp': sel_cpu(batch_cpu['old_logp']),
                    'ret': sel_cpu(batch_cpu['ret']),
                    'adv': sel_cpu(batch_cpu['adv']),
                    'legals': batch_cpu['legals'],  # globale su device
                    'legals_offset': offs_mb,
                    'legals_count': cnts_mb,
                    'chosen_index': sel_cpu(batch_cpu['chosen_index']),
                    'seat_team': sel_cpu(batch_cpu['seat_team']) if batch_cpu.get('seat_team', None) is not None else None,
                    'others_hands': sel_cpu(batch_cpu['others_hands']) if batch_cpu.get('others_hands', None) is not None else None,
                    'a_emb_global': batch_cpu.get('a_emb_global', None),
                    'mcts_policy': mcts_policy_mb,
                    'mcts_weight': mcts_weight_mb,
                }
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                loss, info = self.compute_loss(mini)
                loss.backward()
                # grad norm (usa il valore restituito da clip_grad_norm_ per evitare un secondo pass)
                gn_actor = nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                gn_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                if STRICT_CHECKS and ((not torch.isfinite(gn_actor)) or (not torch.isfinite(gn_critic))):
                    raise RuntimeError("update: non-finite gradients detected (norm)")
                gn_actor = torch.as_tensor(gn_actor, device=device, dtype=torch.float32)
                gn_critic = torch.as_tensor(gn_critic, device=device, dtype=torch.float32)
                self.opt_actor.step()
                self.opt_critic.step()
                # Post-step: parametri finiti (cattura divergenze immediate)
                if hasattr(self.actor, 'state_enc') and hasattr(self.actor.state_enc, 'card_emb'):
                    if STRICT_CHECKS and (not torch.isfinite(self.actor.state_enc.card_emb).all()):
                        raise RuntimeError("update: state_enc.card_emb became non-finite after step")
                # invalidate any inference caches after params changed
                try:
                    self.actor.invalidate_action_cache()
                except Exception as e:
                    raise RuntimeError('ppo.update.invalidate_action_cache_failed') from e
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
                    # Use averaged KL so far to decide early stop (smoother)
                    _kl = (avg_kl_acc / max(1, count_mb)).detach()
                    if bool((_kl > self.target_kl).item()):
                        self._high_kl_count += 1
                        early_stop = True
                        break
                    else:
                        self._high_kl_count = max(0, self._high_kl_count - 1)
            # Step any schedulers
            for sch in self._lr_schedulers:
                sch.step()
            # entropy schedule opzionale (clamp in range sicuro)
            if self._entropy_schedule is not None:
                try:
                    new_coef = float(self._entropy_schedule(self.update_steps))
                    self.entropy_coef = float(max(0.0, min(0.1, new_coef)))
                except Exception as e:
                    raise RuntimeError('ppo.update.entropy_schedule_failed') from e
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
        # Modelli restano su CPU
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
        ckpt = safe_torch_load(path, map_location=map_location or device)
        # Support multiple checkpoint formats:
        # 1) Full PPO checkpoint: {'actor','critic','opt_actor','opt_critic',...}
        # 2) Actor-only checkpoint: {'actor', ...}
        # 3) Raw actor state_dict (plain mapping)
        if isinstance(ckpt, dict) and ('actor' in ckpt or 'critic' in ckpt or 'opt_actor' in ckpt or 'run_config' in ckpt):
            # Load actor (required in this branch)
            if 'actor' in ckpt:
                self.actor.load_state_dict(ckpt['actor'])
            else:
                # Some older actor-only checkpoints may have saved the actor dict at top-level (handled below)
                pass
            # Load critic if present; otherwise keep randomly initialized critic
            if 'critic' in ckpt:
                self.critic.load_state_dict(ckpt['critic'])
            # Load optimizers only if both are present (skip safely otherwise)
            if ('opt_actor' in ckpt) and ('opt_critic' in ckpt):
                try:
                    self.opt_actor.load_state_dict(ckpt['opt_actor'])
                    self.opt_critic.load_state_dict(ckpt['opt_critic'])
                except Exception as e:
                    raise RuntimeError('ppo.load.optim_state_load_failed') from e
            # Restore metadata if present
            self.run_config = ckpt.get('run_config', self.run_config)
            self.update_steps = ckpt.get('update_steps', 0)
        else:
            # Fallback: attempt to treat checkpoint as a raw actor state_dict
            # This covers files like bootstrap_random.pth with only actor weights.
            try:
                self.actor.load_state_dict(ckpt)
            except Exception as e:
                raise RuntimeError('ppo.load.unsupported_checkpoint_format') from e
            # Keep critic/optimizers as freshly initialized
            # Metadata not available in this format
