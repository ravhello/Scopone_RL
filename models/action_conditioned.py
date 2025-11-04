import torch
import torch.nn as nn
import math
import time
import os
from contextlib import nullcontext
from typing import Dict, Tuple, Optional
from utils.fallback import notify_fallback
try:
    # Prefer new SDPA backend selector if available (PyTorch >= 2.3)
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_ctx, SDPBackend as _SDPBackend  # type: ignore
except Exception:
    _sdpa_kernel_ctx = None  # type: ignore
    _SDPBackend = None  # type: ignore

def _device_from_env(env_key: str, default: str = 'cpu') -> torch.device:
    val = os.environ.get(env_key, default)
    try:
        if val.startswith('cuda') and not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(val)
    except Exception:
        return torch.device('cpu')


def _get_amp_dtype() -> torch.dtype:
    amp = os.environ.get('AMP_DTYPE', 'fp16').lower()
    return torch.bfloat16 if amp == 'bf16' else torch.float16


device = _device_from_env('SCOPONE_DEVICE')
autocast_device = device.type
autocast_dtype = _get_amp_dtype()
STRICT = (os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1')

# Observation flags are now imported from observation.py to ensure consistency.
from observation import (
    OBS_INCLUDE_INFERRED as _OBS_INCLUDE_INFERRED,
    OBS_INCLUDE_RANK_PROBS as _OBS_INCLUDE_RANK_PROBS,
    OBS_INCLUDE_SCOPA_PROBS as _OBS_INCLUDE_SCOPA_PROBS,
    OBS_INCLUDE_DEALER as _OBS_INCLUDE_DEALER,
)

import torch._dynamo as _dynamo  # type: ignore
_dynamo_disable = _dynamo.disable  # type: ignore[attr-defined]



class StateEncoderCompact(nn.Module):
    """
    Encoder per osservazione compatta con storia-k (blocchi da 61) e stats a
    dimensione variabile (in base ai flag di osservazione). Le sezioni sono:
      - hand_table: 83
      - captured: 82
      - history: 61*k (k variabile)
      - stats: variabile (resto delle feature)
      - seat/team: 6 (passato separatamente)
    """
    def __init__(self, k_history: Optional[int] = None):
        super().__init__()
        self.k_history_hint: Optional[int] = k_history
        # Card embedding for permutation-invariant set encoding (40 card IDs)
        # Initialize deterministically and ensure finite values
        torch.manual_seed(0)
        ce = torch.randn(40, 32, device=device, dtype=torch.float32) * 0.02
        ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
        self.card_emb = nn.Parameter(ce)
        # Small processors for counts and concatenations
        self.counts_head_hand = nn.Sequential(nn.Linear(3, 16), nn.ReLU())    # other hands sizes
        self.counts_head_cap = nn.Sequential(nn.Linear(2, 16), nn.ReLU())     # captured counts
        # Cross-attention mano↔tavolo per relazioni pari-rank e subset-sum
        self.cross_attn_h2t = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.cross_attn_t2h = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        # set_merge: [hand(32) | table(32) | cap0(32) | cap1(32) | hand_attn(32) | table_attn(32) | counts(16+16)]
        self.set_merge_head = nn.Sequential(nn.Linear(32 * 6 + 16 + 16, 64), nn.ReLU())

        # History Transformer (sequence encoder)
        self.hist_proj = nn.Linear(61, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.hist_pos_emb = nn.Embedding(40, 64)  # up to 40 recent moves

        # Stats and seat/team: fix input dim deterministically from flags to avoid LazyLinear pitfalls
        self.stats_in_dim = 99 \
            + (120 if _OBS_INCLUDE_INFERRED else 0) \
            + (10 if _OBS_INCLUDE_SCOPA_PROBS else 0) \
            + (150 if _OBS_INCLUDE_RANK_PROBS else 0) \
            + (4 if _OBS_INCLUDE_DEALER else 0)
        self.stats_processor = nn.Sequential(nn.Linear(self.stats_in_dim, 64), nn.ReLU())
        self.seat_head = nn.Sequential(nn.Linear(6, 32), nn.ReLU())

        # Combiner to 256-d state context
        # Inputs: set_merge(64) + hist(64) + stats(64) + seat(32) = 224 → 256
        # Se OBS_INCLUDE_DEALER=1 aggiungiamo +4 nelle stats. Usiamo Linear(224,256) e
        # deleghiamo a self.stats_processor (LazyLinear) l'adattamento alla nuova dimensione.
        self.combiner = nn.Sequential(nn.Linear(224, 256), nn.ReLU())
        self.to(device)
        # Final guard after move to device
        with torch.no_grad():
            if (os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1') and (not torch.isfinite(self.card_emb).all()):
                raise RuntimeError("ActionConditionedActor: non-finite card_emb on initialization")
        # Pre-create positional ids for up to 40 steps to avoid per-forward arange
        self.register_buffer('_hist_pos_ids', torch.arange(40, dtype=torch.long, device=device))

    def _attn_ctx(self):
        if device.type == 'cuda':
            if _sdpa_kernel_ctx is not None and _SDPBackend is not None:
                return _sdpa_kernel_ctx([_SDPBackend.FLASH_ATTENTION, _SDPBackend.EFFICIENT_ATTENTION])
            return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        return nullcontext()

    def _safe_mha(self, mha: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kpm: torch.Tensor) -> torch.Tensor:
        """Run MHA with key padding mask safely; returns zeros where all tokens are masked.
        Expects batch-first (B, T, E)."""
        out = torch.zeros_like(q)
        if kpm.dim() == 2:
            with self._attn_ctx():
                o, _ = mha(query=q, key=k, value=v, key_padding_mask=kpm, need_weights=False)
            if o.dtype != out.dtype:
                o = o.to(dtype=out.dtype)
            all_masked = kpm.all(dim=1)
            if all_masked.dtype != torch.bool:
                all_masked = all_masked.to(torch.bool)
            # Zero-out rows where all keys are masked (out-of-place to avoid autograd versioning issues)
            o = torch.where(all_masked.view(-1, 1, 1), torch.zeros_like(o), o)
            out = o
        return out

    def _mha_masked_mean(self, mha: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          q_present: torch.Tensor, k_present: torch.Tensor) -> torch.Tensor:
        """Compute MHA and return masked mean over query tokens using a single key_padding_mask path.
        Returns shape (B, E)."""
        B, _, E = q.shape
        kpm = (~k_present)
        # Compute per-row validity and, if possible, avoid MHA on fully-masked rows
        q_any = q_present.any(dim=1)
        k_any = k_present.any(dim=1)
        valid_rows = (q_any & k_any)
        out = torch.zeros((B, q.size(1), E), dtype=q.dtype, device=q.device)
        if bool(valid_rows.any()):
            idx = valid_rows.nonzero(as_tuple=True)[0]
            q_sel = q.index_select(0, idx)
            k_sel = k.index_select(0, idx)
            v_sel = v.index_select(0, idx)
            kpm_sel = kpm.index_select(0, idx)
            with self._attn_ctx():
                o_sel, _ = mha(query=q_sel, key=k_sel, value=v_sel, key_padding_mask=kpm_sel, need_weights=False)
            if o_sel.dtype != out.dtype:
                o_sel = o_sel.to(dtype=out.dtype)
            out.index_copy_(0, idx, o_sel)
        # Mask invalid rows to zeros (already zero) and compute masked mean across query tokens
        m = q_present.unsqueeze(-1).to(out.dtype)
        summed = (out * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(self, obs: torch.Tensor, seat_team_vec: torch.Tensor = None) -> torch.Tensor:
        import torch.nn.functional as F
        # Use the module's parameter device as the target to avoid CPU/GPU mismatches after .to(...)
        target_device = self.card_emb.device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if (obs.device != target_device) or (obs.dtype != torch.float32):
            obs = obs.to(device=target_device, dtype=torch.float32)

        B = obs.size(0)
        D = obs.size(1)
        # Deterministic k inference from flags and total observation size.
        # D = 165 + 61*k + stats_len, stats_len = 99 + [inferred(120)] + [scopa(10)] + [rank(150)] + [dealer(4)]
        base_prefix = 165
        base_stats = 99
        stats_len = base_stats \
            + (120 if _OBS_INCLUDE_INFERRED else 0) \
            + (10 if _OBS_INCLUDE_SCOPA_PROBS else 0) \
            + (150 if _OBS_INCLUDE_RANK_PROBS else 0) \
            + (4 if _OBS_INCLUDE_DEALER else 0)
        k = 0
        if self.k_history_hint is not None:
            k = int(self.k_history_hint)
        else:
            rem = int(D) - base_prefix - stats_len
            if rem >= 0 and (rem % 61) == 0:
                k = rem // 61
            else:
                # Fallback to heuristic search (legacy), but signal mismatch
                notify_fallback('models.state_encoder_compact.heuristic_k')
                found = False
                for kk in range(40, -1, -1):
                    rem2 = D - base_prefix - 61 * kk
                    if rem2 < base_stats:
                        continue
                    delta = rem2 - base_stats
                    # Option dims: inferred(120), scopa(10), rank(150), dealer(4)
                    option_dims = [120, 10, 150, 4]
                    ok = False
                    for a in (0, 1):
                        for b in (0, 1):
                            for c in (0, 1):
                                for d in (0, 1):
                                    if (a * option_dims[0] + b * option_dims[1] + c * option_dims[2] + d * option_dims[3]) == delta:
                                        ok = True
                                        break
                                if ok:
                                    break
                            if ok:
                                break
                        if ok:
                            break
                    if ok:
                        k = kk
                        found = True
                        break

        # Autocast per tutto il compute del forward compatto
        cm = torch.autocast(device_type=target_device.type, dtype=autocast_dtype) if target_device.type == 'cuda' else nullcontext()
        with cm:
            # Sezioni
            hand_table = obs[:, :83]
            captured = obs[:, 83:165]
            hist_start = 165
            hist_end = 165 + 61 * k
            history = obs[:, hist_start:hist_end]
            stats = obs[:, hist_end:]
            # Sanitize stats to avoid dynamic asserts/guards under torch.compile
            stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e6, 1e6)
            # Early structure guard: stats segment must be non-empty
            if STRICT:
                if stats.size(1) <= 0:
                    raise RuntimeError(
                        f"StateEncoderCompact: empty stats segment (D={int(D)}, hist_end={int(hist_end)}, k={int(k)}, expected_stats_len≈{int(stats_len)})"
                    )
            # Strict shape check for stats segment to match configured flags
            if STRICT:
                if int(stats.size(1)) != int(stats_len):
                    raise RuntimeError(
                        f"StateEncoderCompact: stats segment dimension mismatch: got {int(stats.size(1))}, expected {int(stats_len)}; "
                        f"D={int(D)} hist_end={int(hist_end)} k={int(k)} flags(inferred={int(_OBS_INCLUDE_INFERRED)}, scopa={int(_OBS_INCLUDE_SCOPA_PROBS)}, rank={int(_OBS_INCLUDE_RANK_PROBS)}, dealer={int(_OBS_INCLUDE_DEALER)})"
                    )
            # Validate stats input before feeding to LazyLinear
            if STRICT:
                torch._assert(torch.isfinite(stats).all(), "StateEncoderCompact: stats input contains non-finite values")
                if stats.numel() > 0:
                    torch._assert((stats.abs().amax() <= 1e6), "StateEncoderCompact: stats input magnitude too large (>1e6)")
            # If LazyLinear already materialized, ensure weights are finite
            lin: nn.Linear = self.stats_processor[0]  # type: ignore
            w = getattr(lin, 'weight', None)
            b = getattr(lin, 'bias', None)
            if STRICT:
                if (w is not None) and (not isinstance(w, torch.nn.parameter.UninitializedParameter)) and w.numel() > 0:
                    torch._assert(torch.isfinite(w).all(), "StateEncoderCompact: stats_processor.weight non-finite")
                    if w.numel() > 0:
                        torch._assert((w.abs().amax() <= 1e6), "StateEncoderCompact: stats_processor.weight magnitude too large (>1e6)")
                if (b is not None) and (not isinstance(b, torch.nn.parameter.UninitializedParameter)) and b.numel() > 0:
                    torch._assert(torch.isfinite(b).all(), "StateEncoderCompact: stats_processor.bias non-finite")
                    if b.numel() > 0:
                        torch._assert((b.abs().amax() <= 1e6), "StateEncoderCompact: stats_processor.bias magnitude too large (>1e6)")

            # ----- Set encoders -----
            hand_mask = hand_table[:, :40]
            other_counts = hand_table[:, 40:43]
            table_mask = hand_table[:, 43:83]
            # Validate inputs are finite and in [0,1]
            if STRICT:
                for name, t in (('hand_mask', hand_mask), ('other_counts', other_counts), ('table_mask', table_mask)):
                    torch._assert(torch.isfinite(t).all(), f"StateEncoderCompact: {name} contains non-finite values")
                    torch._assert((t.min().ge(0) & t.max().le(1)), f"StateEncoderCompact: {name} out of [0,1] range")
            card_emb = self.card_emb
            # Parameter sanity before use
            if STRICT:
                torch._assert(torch.isfinite(card_emb).all(), "StateEncoderCompact: card_emb contains non-finite values")
            w_cnt: nn.Linear = self.counts_head_hand[0]  # type: ignore
            if STRICT:
                if (not torch.isfinite(w_cnt.weight).all()) or (not torch.isfinite(w_cnt.bias).all()):
                    raise RuntimeError("StateEncoderCompact: counts_head_hand weights contain non-finite values")
            hand_feat = torch.matmul(hand_mask, card_emb)           # (B,32)
            table_feat = torch.matmul(table_mask, card_emb)         # (B,32)
            other_cnt_feat = self.counts_head_hand(other_counts)    # (B,16)
            if STRICT:
                torch._assert((torch.isfinite(hand_feat).all() & torch.isfinite(table_feat).all() & torch.isfinite(other_cnt_feat).all()), "StateEncoderCompact: non-finite set base features (hand/table/counts)")
            # Cross-attention mano↔tavolo
            hand_present = (hand_mask > 0.5)
            table_present = (table_mask > 0.5)
            hand_seq = hand_mask.unsqueeze(-1) * card_emb           # (B,40,32)
            table_seq = table_mask.unsqueeze(-1) * card_emb         # (B,40,32)
            hand_kpm = (~hand_present)
            table_kpm = (~table_present)
            # Safe MHA wrapper batched + SDPA
            hand_attn_feat = self._mha_masked_mean(self.cross_attn_h2t, hand_seq, table_seq, table_seq,
                                                   hand_present, table_present)  # (B,32)
            table_attn_feat = self._mha_masked_mean(self.cross_attn_t2h, table_seq, hand_seq, hand_seq,
                                                    table_present, hand_present)  # (B,32)
            if STRICT:
                torch._assert((torch.isfinite(hand_attn_feat).all() & torch.isfinite(table_attn_feat).all()), "StateEncoderCompact: non-finite attention features")

            # captured
            cap0_mask = captured[:, :40]
            cap1_mask = captured[:, 40:80]
            cap_counts = captured[:, 80:82]
            cap0_feat = torch.matmul(cap0_mask, card_emb)           # (B,32)
            cap1_feat = torch.matmul(cap1_mask, card_emb)           # (B,32)
            cap_cnt_feat = self.counts_head_cap(cap_counts)         # (B,16)

            # Reduce cat overhead by preallocating and slicing
            set_merged = torch.empty((B, 32*6 + 16 + 16), dtype=hand_feat.dtype, device=hand_feat.device)
            pos = 0
            set_merged[:, pos:pos+32] = hand_feat; pos += 32
            set_merged[:, pos:pos+32] = table_feat; pos += 32
            set_merged[:, pos:pos+32] = cap0_feat; pos += 32
            set_merged[:, pos:pos+32] = cap1_feat; pos += 32
            set_merged[:, pos:pos+32] = hand_attn_feat; pos += 32
            set_merged[:, pos:pos+32] = table_attn_feat; pos += 32
            set_merged[:, pos:pos+16] = other_cnt_feat; pos += 16
            set_merged[:, pos:pos+16] = cap_cnt_feat; pos += 16
            set_feat = self.set_merge_head(set_merged)               # (B,64)
            if STRICT:
                torch._assert(torch.isfinite(set_feat).all(), "StateEncoderCompact: non-finite set_feat")
                if set_feat.numel() > 0:
                    torch._assert((set_feat.abs().amax() <= 1e6), "StateEncoderCompact: set_feat magnitude too large (>1e6)")

            # ----- History Transformer -----
            if k > 0:
                hist_reshaped = history.view(B, k, 61)               # (B,k,61)
                hproj = self.hist_proj(hist_reshaped)                # (B,k,64)
                # Use precomputed position ids buffer (max 40)
                pos_idx = self._hist_pos_ids[:k].unsqueeze(0).expand(B, k)
                hpos = self.hist_pos_emb(pos_idx)
                hseq = hproj + hpos
                # Enable efficient SDPA kernels during history attention when available
                with self._attn_ctx():
                    henc = self.hist_encoder(hseq)                   # (B,k,64)
                hist_feat = henc.mean(dim=1)                         # (B,64)
            else:
                hist_feat = torch.zeros((B, 64), dtype=obs.dtype, device=obs.device)
            if STRICT:
                torch._assert(torch.isfinite(hist_feat).all(), "StateEncoderCompact: non-finite hist_feat")
                if hist_feat.numel() > 0:
                    torch._assert((hist_feat.abs().amax() <= 1e6), "StateEncoderCompact: hist_feat magnitude too large (>1e6)")

            # Stats e seat/team
            stats_feat = self.stats_processor(stats)
            if STRICT:
                if not torch.isfinite(stats_feat).all():
                    # Detailed diagnostics on failure
                    lin: nn.Linear = self.stats_processor[0]  # type: ignore
                    w = getattr(lin, 'weight', None)
                    b = getattr(lin, 'bias', None)
                    def _st(t: torch.Tensor):
                        return {
                            'min': float(t.min().item()) if t.numel() > 0 else None,
                            'max': float(t.max().item()) if t.numel() > 0 else None,
                            'mean': float(t.mean().item()) if t.numel() > 0 else None,
                            'numel': int(t.numel())
                        }
                    w_stats = _st(w)
                    b_stats = _st(b)
                    s_stats = _st(stats)
                    raise RuntimeError(f"StateEncoderCompact: non-finite stats_feat; weight_stats={w_stats}, bias_stats={b_stats}, stats_input_stats={s_stats}")
                if stats_feat.numel() > 0:
                    torch._assert((stats_feat.abs().amax() <= 1e6), "StateEncoderCompact: stats_feat magnitude too large (>1e6)")
            if seat_team_vec is None:
                seat_team_vec = torch.zeros((B, 6), dtype=torch.float32, device=obs.device)
            elif seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            # Validate seat/team vector: one-hot seat and flags in [0,1]
            if STRICT:
                if seat_team_vec.size(-1) != 6:
                    raise ValueError("StateEncoderCompact: seat_team_vec must have shape (B,6)")
                if not (seat_team_vec[:, :4].sum(dim=1) == 1).all():
                    raise RuntimeError("StateEncoderCompact: seat one-hot invalid (sum != 1)")
                if ((seat_team_vec[:, 4:6] < 0) | (seat_team_vec[:, 4:6] > 1)).any():
                    raise RuntimeError("StateEncoderCompact: team flags out of [0,1]")
            seat_feat = F.relu(self.seat_head[0](seat_team_vec), inplace=True)
            if STRICT:
                torch._assert(torch.isfinite(seat_feat).all(), "StateEncoderCompact: non-finite seat_feat")
                if seat_feat.numel() > 0:
                    torch._assert((seat_feat.abs().amax() <= 1e6), "StateEncoderCompact: seat_feat magnitude too large (>1e6)")

            combined = torch.empty((B, 64+64+64+32), dtype=set_feat.dtype, device=set_feat.device)
            p2 = 0
            combined[:, p2:p2+64] = set_feat; p2 += 64
            combined[:, p2:p2+64] = hist_feat; p2 += 64
            combined[:, p2:p2+64] = stats_feat; p2 += 64
            combined[:, p2:p2+32] = seat_feat; p2 += 32
            context = F.relu(self.combiner[0](combined), inplace=True)
            if STRICT:
                torch._assert(torch.isfinite(context).all(), "StateEncoderCompact: non-finite context output")
                if context.numel() > 0:
                    torch._assert((context.abs().amax() <= 1e6), "StateEncoderCompact: context magnitude too large (>1e6)")
        return context


class ActionEncoder80(nn.Module):
    """Encoda l'azione binaria 80-dim in un embedding 64-dim."""
    def __init__(self, action_dim: int = 80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.to(device)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        target_device = next(self.parameters()).device
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=target_device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if (actions.device != target_device) or (actions.dtype != torch.float32):
            actions = actions.to(device=target_device, dtype=torch.float32)
        # Validate last-dimension matches expected input features
        try:
            in_dim = int(self.net[0].in_features)  # type: ignore[attr-defined]
        except Exception:
            in_dim = 80
        if actions.size(-1) != in_dim:
            raise ValueError(f"ActionEncoder80: expected last-dim {in_dim}, got {int(actions.size(-1))}")
        out = self.net(actions)
        # Keep checks only in STRICT mode
        if STRICT:
            torch._assert(torch.isfinite(out).all(), "ActionEncoder80.forward: non-finite output embedding")
            if out.numel() > 0:
                max_abs = out.abs().amax()
                torch._assert((max_abs <= 1e6), "ActionEncoder80.forward: output magnitude too large (>1e6)")
        return out


class BeliefNet(nn.Module):
    """
    Belief network che predice 3x40 logits (altri giocatori) a partire da state features (256).
    - Architettura profonda con residual, LayerNorm e GELU per capacità e stabilità.
    - Parametro di temperatura per calibrazione delle probabilità.
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc_mid1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_mid2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 120)
        # temperatura appresa (clampata in (0.25, 4.0))
        self._log_temp = nn.Parameter(torch.log(torch.tensor(1.0)))
        # CPU-friendly: allow approximate GELU when enabled via env (preserves quality in practice)
        approx_gelu = (os.environ.get('SCOPONE_APPROX_GELU', '0') == '1')
        self.act = nn.GELU(approximate='tanh' if approx_gelu else 'none')
        self.dropout = nn.Dropout(p=0.1)
        self.to(device)

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        x = state_feat
        h = self.act(self.ln1(self.fc_in(x)))
        h = self.dropout(h)
        # residuo 1
        r = self.act(self.ln2(self.fc_mid1(h)))
        h = self.dropout(h + r)
        # residuo 2
        r2 = self.act(self.ln3(self.fc_mid2(h)))
        h = self.dropout(h + r2)
        logits = self.fc_out(h)
        return logits  # (B,120) logits per 3x40

    def temperature(self) -> torch.Tensor:
        log_temp = self._log_temp
        if not torch.isfinite(log_temp).all():
            raise RuntimeError("BeliefNet.temperature: _log_temp contains non-finite values")
        temp = torch.exp(log_temp)
        if STRICT:
            torch._assert(torch.isfinite(temp).all(), "BeliefNet.temperature: exp(log_temp) produced non-finite values")
        return torch.clamp(temp, 0.25, 4.0)

    def probs(self, logits: torch.Tensor, visible_mask_40: torch.Tensor = None) -> torch.Tensor:
        """
        Converte logits 3x40 in probabilità normalizzate per-carta tra i 3 giocatori.
        - logits: (B,120)
        - visible_mask_40: (B,40) boolean, True = carta visibile → probabilità a 0
        Ritorna: (B,120) flatten di (B,3,40) probabilità
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        B = logits.size(0)
        t = self.temperature().to(logits.device, dtype=logits.dtype)
        x = logits.view(B, 3, 40)
        x = x / t
        # softmax per-carta (dim=1 sui 3 giocatori)
        probs = torch.softmax(x, dim=1)
        if STRICT:
            torch._assert(torch.isfinite(probs).all(), "BeliefNet.probs: softmax produced non-finite probabilities")
        if visible_mask_40 is not None:
            if visible_mask_40.dim() == 1:
                visible_mask_40 = visible_mask_40.unsqueeze(0)
            # Azzera probabilità per carte visibili e rinormalizza solo dove la somma per carta > 0
            m = visible_mask_40.to(probs.dtype).unsqueeze(1)  # (B,1,40)
            probs = probs * (1.0 - m)
            sums = probs.sum(dim=1, keepdim=True)  # (B,1,40)
            nz = (sums > 0)
            # Broadcast-safe: divide solo sulle posizioni nz, altrimenti lascia 0
            probs = torch.where(nz.expand_as(probs), probs / torch.clamp_min(sums, 1e-12), probs)
        return probs.view(B, 120)


class ActionConditionedActor(torch.nn.Module):
    """
    Actor realmente action-conditioned:
      - State encoder compatto (usa storia-k con pooling) → 256-d
      - Belief head (120 → 64)
      - Proiezione stato → 64 e scoring via prodotto scalare con embedding azione (80 → 64)
    """
    def __init__(self, obs_dim=10823, action_dim=80, state_encoder: StateEncoderCompact = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Encoders
        self.state_enc = state_encoder if state_encoder is not None else StateEncoderCompact()
        self.belief_head = nn.Sequential(nn.Linear(120, 64), nn.ReLU())
        # BeliefNet neurale migliorata (state_feat 256 -> logits 120)
        self.belief_net = BeliefNet(in_dim=256, hidden_dim=512)
        # Partner-aware: embed per-carta (40→32) e gating separato partner/opps
        self.belief_card_emb = nn.Parameter(torch.randn(40, 32, device=device, dtype=torch.float32) * 0.02)
        self.partner_gate = nn.Sequential(nn.Linear(256, 32), nn.Sigmoid())
        self.opp_gate = nn.Sequential(nn.Linear(256, 32), nn.Sigmoid())
        # Merge: stato 256 + belief_head 64 + partner 32 + opp 32 = 384
        self.merge = nn.Sequential(nn.Linear(384, 256), nn.ReLU())
        self.state_to_action = nn.Linear(256, 64)
        # Initialize small and stable
        nn.init.kaiming_uniform_(self.state_to_action.weight, a=math.sqrt(5))
        if self.state_to_action.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.state_to_action.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.state_to_action.bias, -bound, bound)
        self.action_enc = ActionEncoder80(action_dim)
        # Embedding per la selezione carta (40 carte)
        self.card_emb_play = nn.Parameter(torch.randn(40, 64, device=device, dtype=torch.float32) * 0.02)
        # Cache di tutte le azioni one-hot (80 x 80) per calcolare logits pieni
        self.register_buffer('all_actions_eye', torch.eye(action_dim, dtype=torch.float32))
        # Cache embedding azioni per device/dtype (solo per inference)
        self._cached_action_emb = None  # legacy single-cache for backward compat
        self._cached_action_emb_variants: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}
        self.to(device)

    @staticmethod
    def _visible_mask_from_obs(x_obs: torch.Tensor) -> torch.Tensor:
        hand_table = x_obs[:, :83]
        # Single threshold mask and slicing to avoid repeated > 0.5 ops
        mask_all = hand_table > 0.5
        hand_mask = mask_all[:, :40]
        table_mask = mask_all[:, 43:83]
        captured = x_obs[:, 83:165] > 0.5
        cap0_mask = captured[:, :40]
        cap1_mask = captured[:, 40:80]
        return hand_mask | table_mask | cap0_mask | cap1_mask

    def compute_state_proj(self, obs: torch.Tensor, seat_team_vec: torch.Tensor) -> torch.Tensor:
        target_device = next(self.parameters()).device
        _par = (os.environ.get('SCOPONE_PROFILE', '0') != '0')
        t_state_enc = 0.0; t_belief_logits = 0.0; t_belief_probs = 0.0
        t_partner = 0.0; t_opp = 0.0; t_merge = 0.0; t_proj = 0.0
        _t0 = time.time() if _par else 0.0
        if torch.is_tensor(obs):
            if (obs.device == target_device) and (obs.dtype == torch.float32):
                x_obs = obs
            elif obs.device == target_device:
                x_obs = obs.to(dtype=torch.float32)
            else:
                x_obs = obs.to(device=target_device, dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if seat_team_vec is None:
            raise ValueError("seat_team_vec is required (B,6)")
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            if (seat_team_vec.device != x_obs.device) or (seat_team_vec.dtype != torch.float32):
                seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        t1 = time.time() if _par else 0.0
        state_feat = self.state_enc(x_obs, seat_team_vec)  # (B,256)
        if _par: t_state_enc += (time.time() - t1)
        if STRICT:
            torch._assert(torch.isfinite(state_feat).all(), "state_enc produced non-finite features")
            if state_feat.numel() > 0:
                torch._assert((state_feat.abs().amax() <= 1e6), "compute_state_proj: state_feat magnitude too large (>1e6)")
        # Ensure BeliefNet receives its parameter dtype (avoids Half/Float mismatch outside autocast)
        bn_dtype = self.belief_net.fc_in.weight.dtype
        if state_feat.dtype != bn_dtype:
            state_feat = state_feat.to(dtype=bn_dtype)
        # belief neurale interno con maschera carte visibili
        visible_mask = self._visible_mask_from_obs(x_obs)
        t1 = time.time() if _par else 0.0
        belief_logits = self.belief_net(state_feat)        # (B,120)
        if _par: t_belief_logits += (time.time() - t1)
        # Safety clamp before softmax to avoid NaNs from extreme values
        belief_logits = torch.nan_to_num(belief_logits, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-30.0, 30.0)
        if STRICT:
            torch._assert(torch.isfinite(belief_logits).all(), "BeliefNet produced non-finite logits")
        t1 = time.time() if _par else 0.0
        belief_probs_flat = self.belief_net.probs(belief_logits, visible_mask)
        if _par: t_belief_probs += (time.time() - t1)
        belief_probs_flat = torch.nan_to_num(belief_probs_flat, nan=0.0, posinf=0.0, neginf=0.0)
        if STRICT:
            torch._assert(torch.isfinite(belief_probs_flat).all(), "BeliefNet.probs produced non-finite probabilities")
        # Probability range guard
        if STRICT:
            torch._assert(((belief_probs_flat >= 0).all() & (belief_probs_flat <= 1).all()), "compute_state_proj: belief_probs out of [0,1]")
        belief_feat = self.belief_head(belief_probs_flat)  # (B,64)
        if STRICT:
            torch._assert(torch.isfinite(belief_feat).all(), "compute_state_proj: belief_feat non-finite")
            if belief_feat.numel() > 0:
                torch._assert((belief_feat.abs().amax() <= 1e6), "compute_state_proj: belief_feat magnitude too large (>1e6)")
        partner_slice = belief_probs_flat[:, 40:80]
        opps_slice = belief_probs_flat[:, 0:40] + belief_probs_flat[:, 80:120]
        emb = self.belief_card_emb
        t1 = time.time() if _par else 0.0
        partner_feat = torch.matmul(partner_slice, emb)     # (B,32)
        opp_feat = torch.matmul(opps_slice, emb)            # (B,32)
        if _par: t_partner += (time.time() - t1)
        t1 = time.time() if _par else 0.0
        pg = self.partner_gate(state_feat)
        og = self.opp_gate(state_feat)
        if _par: t_opp += (time.time() - t1)
        # Gating range guards
        if STRICT:
            torch._assert(((pg >= 0).all() & (pg <= 1).all()), "compute_state_proj: partner_gate out of [0,1]")
            torch._assert(((og >= 0).all() & (og <= 1).all()), "compute_state_proj: opp_gate out of [0,1]")
        partner_feat = partner_feat * pg
        opp_feat = opp_feat * og
        if STRICT:
            torch._assert(torch.isfinite(partner_feat).all(), "compute_state_proj: partner_feat non-finite before merge")
            torch._assert(torch.isfinite(opp_feat).all(), "compute_state_proj: opp_feat non-finite before merge")
            if partner_feat.numel() > 0:
                torch._assert((partner_feat.abs().amax() <= 1e6), "compute_state_proj: partner_feat magnitude too large (>1e6)")
            if opp_feat.numel() > 0:
                torch._assert((opp_feat.abs().amax() <= 1e6), "compute_state_proj: opp_feat magnitude too large (>1e6)")
        B_ctx = state_feat.size(0)
        t1 = time.time() if _par else 0.0
        ctx_in = torch.empty((B_ctx, 256 + 64 + 32 + 32), dtype=state_feat.dtype, device=state_feat.device)
        p = 0
        ctx_in[:, p:p+256] = state_feat; p += 256
        ctx_in[:, p:p+64] = belief_feat; p += 64
        ctx_in[:, p:p+32] = partner_feat; p += 32
        ctx_in[:, p:p+32] = opp_feat; p += 32
        # Lightweight compile-friendly guard (tensor assert)
        if STRICT:
            if ctx_in.numel() > 0:
                torch._assert((ctx_in.abs().amax() <= 1e6), "merge input (ctx_in) magnitude too large (>1e6)")
        state_ctx = self.merge(ctx_in)  # (B,256)
        if _par: t_merge += (time.time() - t1)
        if STRICT:
            torch._assert(torch.isfinite(state_ctx).all(), "merge produced non-finite state_ctx")
            # Guard extremely large activations (tensor assert)
            if state_ctx.numel() > 0:
                torch._assert((state_ctx.abs().amax() <= 1e6), "merge produced extremely large state_ctx (>1e6)")
        t1 = time.time() if _par else 0.0
        state_proj = self.state_to_action(state_ctx)  # (B,64)
        if _par: t_proj += (time.time() - t1)
        if STRICT:
            torch._assert(torch.isfinite(state_proj).all(), "state_to_action produced non-finite state_proj")
            if state_proj.numel() > 0:
                torch._assert((state_proj.abs().amax() <= 1e6), "state_to_action produced extremely large state_proj (>1e6)")
        # Parameter guard for card_emb_play
        # Evita sync CPU: controlla in modo leggero
        if STRICT:
            torch._assert((self.card_emb_play.abs().amax() <= 1e3), "card_emb_play parameter magnitude exploded (>1e3)")
        # Export sub-timers via a lightweight global (avoid imports to trainer)
        if _par:
            from utils.prof import accum_actor_stateproj
            accum_actor_stateproj(t_state_enc, t_belief_logits, t_belief_probs, t_partner + t_opp, t_merge, t_proj)
        return state_proj

    def compute_state_features(self, obs: torch.Tensor, seat_team_vec: torch.Tensor) -> torch.Tensor:
        """Calcola solo le feature di stato (256) dal pair (obs, seat)."""
        target_device = next(self.parameters()).device
        if torch.is_tensor(obs):
            if (obs.device == target_device) and (obs.dtype == torch.float32):
                x_obs = obs
            elif obs.device == target_device:
                x_obs = obs.to(dtype=torch.float32)
            else:
                x_obs = obs.to(device=target_device, dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if seat_team_vec is None:
            raise ValueError("seat_team_vec is required (B,6)")
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            if (seat_team_vec.device != x_obs.device) or (seat_team_vec.dtype != torch.float32):
                seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        sf = self.state_enc(x_obs, seat_team_vec)  # (B,256)
        if STRICT:
            if not torch.isfinite(sf).all():
                bad = sf[~torch.isfinite(sf)]
                raise RuntimeError(f"state_enc produced non-finite features (count={int(bad.numel())})")
        return sf

    def compute_state_proj_from_state(self, state_feat: torch.Tensor, x_obs: torch.Tensor, visible_mask_40: torch.Tensor = None) -> torch.Tensor:
        """Proietta feature di stato (256) in spazio azione (64) usando belief/gating dell'actor.
        Richiede l'osservazione per calcolare la maschera carte visibili.
        """
        target_device = next(self.parameters()).device
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if x_obs.device != target_device:
            x_obs = x_obs.to(device=target_device)
        if state_feat.device != target_device:
            state_feat = state_feat.to(device=target_device)
        visible_mask = (visible_mask_40 if visible_mask_40 is not None else self._visible_mask_from_obs(x_obs))
        belief_logits = self.belief_net(state_feat)        # (B,120)
        belief_logits = torch.nan_to_num(belief_logits, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-30.0, 30.0)
        if STRICT:
            if not torch.isfinite(belief_logits).all():
                bad = belief_logits[~torch.isfinite(belief_logits)]
                raise RuntimeError(f"BeliefNet produced non-finite logits (count={int(bad.numel())})")
        belief_probs_flat = self.belief_net.probs(belief_logits, visible_mask)
        belief_probs_flat = torch.nan_to_num(belief_probs_flat, nan=0.0, posinf=0.0, neginf=0.0)
        if STRICT:
            if not torch.isfinite(belief_probs_flat).all():
                bad = belief_probs_flat[~torch.isfinite(belief_probs_flat)]
                raise RuntimeError(f"BeliefNet.probs produced non-finite probabilities (count={int(bad.numel())})")
        belief_feat = self.belief_head(belief_probs_flat)  # (B,64)
        partner_slice = belief_probs_flat[:, 40:80]
        opps_slice = belief_probs_flat[:, 0:40] + belief_probs_flat[:, 80:120]
        emb = self.belief_card_emb
        partner_feat = torch.matmul(partner_slice, emb)     # (B,32)
        opp_feat = torch.matmul(opps_slice, emb)            # (B,32)
        pg = self.partner_gate(state_feat)
        og = self.opp_gate(state_feat)
        partner_feat = partner_feat * pg
        opp_feat = opp_feat * og
        B_ctx = state_feat.size(0)
        ctx_in = torch.empty((B_ctx, 256 + 64 + 32 + 32), dtype=state_feat.dtype, device=state_feat.device)
        p = 0
        ctx_in[:, p:p+256] = state_feat; p += 256
        ctx_in[:, p:p+64] = belief_feat; p += 64
        ctx_in[:, p:p+32] = partner_feat; p += 32
        ctx_in[:, p:p+32] = opp_feat; p += 32
        state_ctx = self.merge(ctx_in)  # (B,256)
        if STRICT:
            if not torch.isfinite(state_ctx).all():
                bad = state_ctx[~torch.isfinite(state_ctx)]
                raise RuntimeError(f"merge produced non-finite state_ctx (count={int(bad.numel())})")
        if STRICT:
            if state_ctx.numel() > 0:
                max_abs_ctx = state_ctx.abs().amax()
                torch._assert((max_abs_ctx <= 1e6), "merge produced extremely large state_ctx (>1e6)")
        sp = self.state_to_action(state_ctx)
        if STRICT:
            if not torch.isfinite(sp).all():
                bad = sp[~torch.isfinite(sp)]
                raise RuntimeError(f"state_to_action produced non-finite state_proj (count={int(bad.numel())})")
        if STRICT:
            if sp.numel() > 0:
                max_abs_proj = sp.abs().amax()
                torch._assert((max_abs_proj <= 1e6), "state_to_action produced extremely large state_proj (>1e6)")
            torch._assert((self.card_emb_play.abs().amax() <= 1e3), "card_emb_play parameter magnitude exploded")
        return sp

    def invalidate_action_cache(self) -> None:
        """Invalida la cache degli embedding delle azioni (usata in inference)."""
        self._cached_action_emb = None
        self._cached_action_emb_variants.clear()

    def get_action_emb_table_cached(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Ritorna la tabella (80,64) degli embedding azione.
        Mantiene una cache per (device,dtype) per evitare copie .to ripetute.
        Se device/dtype non sono specificati, usa il device del modulo e float32.
        """
        target_device = device or next(self.action_enc.parameters()).device
        target_dtype = dtype or next(self.action_enc.parameters()).dtype
        key = (str(target_device), target_dtype)
        cached = self._cached_action_emb_variants.get(key, None)
        if cached is not None:
            return cached

        eye = self.all_actions_eye.to(device=target_device, dtype=torch.float32)
        tbl = self.action_enc(eye).to(device=target_device, dtype=target_dtype).contiguous()
        self._cached_action_emb_variants[key] = tbl
        return tbl

    def forward(self, obs: torch.Tensor, legals: torch.Tensor = None,
                seat_team_vec: torch.Tensor = None) -> torch.Tensor:
        # Stato: (B, D)
        target_device = next(self.parameters()).device
        if torch.is_tensor(obs):
            if (obs.device == target_device) and (obs.dtype == torch.float32):
                x_obs = obs
            elif obs.device == target_device:
                x_obs = obs.to(dtype=torch.float32)
            else:
                x_obs = obs.to(device=target_device, dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if seat_team_vec is None:
            raise ValueError("seat_team_vec is required (B,6)")
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            if (seat_team_vec.device != x_obs.device) or (seat_team_vec.dtype != torch.float32):
                seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        # belief neurale interno con maschera carte visibili
        state_feat = self.state_enc(x_obs, seat_team_vec)  # (B,256)
        bn_dtype = self.belief_net.fc_in.weight.dtype
        if state_feat.dtype != bn_dtype:
            state_feat = state_feat.to(dtype=bn_dtype)
        visible_mask = self._visible_mask_from_obs(x_obs)
        belief_logits = self.belief_net(state_feat)
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1' and (not torch.isfinite(belief_logits).all()):
            bad = belief_logits[~torch.isfinite(belief_logits)]
            raise RuntimeError(f"BeliefNet produced non-finite logits (count={int(bad.numel())})")
        belief_logits = torch.nan_to_num(belief_logits, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-30.0, 30.0)
        if not torch.isfinite(belief_logits).all():
            bad = belief_logits[~torch.isfinite(belief_logits)]
            raise RuntimeError(f"BeliefNet produced non-finite logits (count={int(bad.numel())})")
        belief_probs_flat = self.belief_net.probs(belief_logits, visible_mask)
        if not torch.isfinite(belief_probs_flat).all():
            bad = belief_probs_flat[~torch.isfinite(belief_probs_flat)]
            raise RuntimeError(f"BeliefNet.probs produced non-finite probabilities (count={int(bad.numel())})")
        belief_feat = self.belief_head(belief_probs_flat)  # (B,64)
        # Partner index fisso nel nostro belief: slice centrale [40:80]
        partner_slice = belief_probs_flat[:, 40:80]
        opps_slice = belief_probs_flat[:, 0:40] + belief_probs_flat[:, 80:120]
        emb = self.belief_card_emb
        partner_feat = torch.matmul(partner_slice, emb)     # (B,32)
        opp_feat = torch.matmul(opps_slice, emb)            # (B,32)
        # Gating dipendente dallo stato
        pg = self.partner_gate(state_feat)
        og = self.opp_gate(state_feat)
        partner_feat = partner_feat * pg
        opp_feat = opp_feat * og
        B_ctx2 = state_feat.size(0)
        ctx2_in = torch.empty((B_ctx2, 256 + 64 + 32 + 32), dtype=state_feat.dtype, device=state_feat.device)
        p2 = 0
        ctx2_in[:, p2:p2+256] = state_feat; p2 += 256
        ctx2_in[:, p2:p2+64] = belief_feat; p2 += 64
        ctx2_in[:, p2:p2+32] = partner_feat; p2 += 32
        ctx2_in[:, p2:p2+32] = opp_feat; p2 += 32
        state_ctx = self.merge(ctx2_in)  # (B,256)
        if not torch.isfinite(state_ctx).all():
            bad = state_ctx[~torch.isfinite(state_ctx)]
            raise RuntimeError(f"merge produced non-finite state_ctx (count={int(bad.numel())})")
        state_proj = self.state_to_action(state_ctx)  # (B,64)
        if not torch.isfinite(state_proj).all():
            bad = state_proj[~torch.isfinite(state_proj)]
            raise RuntimeError(f"state_to_action produced non-finite state_proj (count={int(bad.numel())})")

        if legals is None:
            # Calcola logits per tutte le 80 azioni: (B,64) @ (64,80) -> (B,80)
            all_actions = self.all_actions_eye
            # usa cache embedding azioni se disponibile
            action_emb = self.get_action_emb_table_cached(device=state_proj.device, dtype=state_proj.dtype)
            logits = torch.matmul(state_proj, action_emb.t())  # (B,80)
            return logits if logits.size(0) > 1 else logits.squeeze(0)
        # legals: (A,80). Calcola score per azioni legali via prodotto scalare
        if not torch.is_tensor(legals):
            legals_t = torch.as_tensor(legals, dtype=torch.float32, device=state_proj.device)
        else:
            legals_t = legals.to(state_proj.device, dtype=torch.float32)
        # Validate legals shape and structure
        if legals_t.dim() != 2 or legals_t.size(1) != 80:
            raise ValueError(f"Actor.forward: legals must be (A,80), got {tuple(legals_t.shape)}")
        ones = legals_t[:, :40].sum(dim=1)
        if STRICT:
            torch._assert(torch.allclose(ones, torch.ones_like(ones)), "Actor.forward: each legal must have exactly one played bit in [:40]")
        cap = legals_t[:, 40:]
        cap_bad = ((cap > 0.0 + 1e-6) & (cap < 1.0 - 1e-6)) | (cap < -1e-6) | (cap > 1.0 + 1.0e-6)
        torch._assert((~cap_bad).all(), "Actor.forward: captured section must be binary (0/1)")
        # B atteso = 1 in path di selezione
        # In training evita la tabella cache per mantenere gradiente
        if self.training:
            a_emb = self.action_enc(legals_t)
        else:
            a_tbl = self.get_action_emb_table_cached(device=legals_t.device, dtype=state_proj.dtype)
            a_emb = torch.matmul(legals_t, a_tbl)
        if state_proj.size(0) == 1:
            scores = torch.matmul(a_emb, state_proj.squeeze(0))  # (A)
        else:
            # Row-wise dot product: (A,64) ⊙ (A,64) → (A)
            scores = (a_emb * state_proj).sum(dim=1)
        return scores

    def compute_two_stage_logp(self,
                                card_logits_all: torch.Tensor,
                                legals_mb: torch.Tensor,
                                sample_idx_per_legal: torch.Tensor,
                                state_proj: torch.Tensor,
                                a_emb_mb: torch.Tensor) -> torch.Tensor:
        """Compute per-legal total log-prob under the two-stage policy.
        Inputs:
          - card_logits_all: (B,40)
          - legals_mb: (M,80)
          - sample_idx_per_legal: (M,) mapping each legal row to its sample index [0..B)
          - state_proj: (B,64)
          - a_emb_mb: (M,64) action embeddings for each legal row (use action_enc during training)
        Returns:
          - logp_total_per_legal: (M,)
        """
        device_local = card_logits_all.device
        dtype_local = card_logits_all.dtype
        if legals_mb.numel() == 0:
            return torch.zeros((0,), dtype=dtype_local, device=device_local)
        # Validate shapes
        if card_logits_all.dim() != 2 or card_logits_all.size(1) != 40:
            raise ValueError(f"compute_two_stage_logp: card_logits_all must be (B,40), got {tuple(card_logits_all.shape)}")
        if state_proj.dim() != 2 or state_proj.size(1) != 64:
            raise ValueError(f"compute_two_stage_logp: state_proj must be (B,64), got {tuple(state_proj.shape)}")
        if legals_mb.dim() != 2 or legals_mb.size(1) != 80:
            raise ValueError(f"compute_two_stage_logp: legals_mb must be (M,80), got {tuple(legals_mb.shape)}")
        if a_emb_mb.dim() != 2 or a_emb_mb.size(1) != 64:
            raise ValueError(f"compute_two_stage_logp: a_emb_mb must be (M,64), got {tuple(a_emb_mb.shape)}")

        # Played card ids per legal
        ones_per_row = legals_mb[:, :40].sum(dim=1)
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
            if not torch.allclose(ones_per_row, torch.ones_like(ones_per_row)):
                raise RuntimeError("compute_two_stage_logp: legals must have exactly one played bit in [:40]")
        played_ids_mb = torch.argmax(legals_mb[:, :40], dim=1)  # (M)

        # Card log-prob restricted to allowed set per sample
        B = int(card_logits_all.size(0))
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
            if (sample_idx_per_legal.min() < 0) or (sample_idx_per_legal.max() >= B):
                raise RuntimeError(f"compute_two_stage_logp: sample_idx_per_legal out of range (min={int(sample_idx_per_legal.min().item())}, max={int(sample_idx_per_legal.max().item())}, B={B})")
        allowed_mask = torch.zeros((B, 40), dtype=torch.bool, device=device_local)
        allowed_mask[sample_idx_per_legal, played_ids_mb] = True
        # STRICT: rows that appear must have at least one allowed card
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
            rows_present = torch.zeros((B,), dtype=torch.bool, device=device_local)
            rows_present[sample_idx_per_legal] = True
            allow_any = allowed_mask.any(dim=1)
            if bool(((rows_present & (~allow_any)).any()).item()):
                bad_rows = torch.nonzero(rows_present & (~allow_any), as_tuple=False).flatten().tolist()
                raise RuntimeError(f"compute_two_stage_logp: allowed_mask empty for rows {bad_rows}")
            # Also ensure logits are finite at allowed positions
            bad_allowed = (~torch.isfinite(card_logits_all)) & allowed_mask
            rows_bad = bad_allowed.any(dim=1) & rows_present
            if bool(rows_bad.any().item() if rows_bad.numel() > 0 else False):
                bad_rows = torch.nonzero(rows_bad, as_tuple=False).flatten().tolist()
                # collect indices of bad allowed positions for the first bad row
                r = bad_rows[0]
                ids = torch.nonzero(bad_allowed[r], as_tuple=False).flatten()
                raise RuntimeError(f"compute_two_stage_logp: non-finite card_logits at allowed positions for rows {bad_rows}; first_row_bad_ids={[int(i.item()) for i in ids]} ")
        masked_logits = torch.where(allowed_mask, card_logits_all, torch.full_like(card_logits_all, float('-inf')))
        max_allowed = torch.amax(masked_logits, dim=1)
        # Numerically stable exp: operate on masked logits so disallowed are -inf → exp=0
        # Avoid in-place ops on tensors needed for gradient; keep computation out-of-place
        exp_shift_allowed = torch.exp(masked_logits - max_allowed.unsqueeze(1))
        sum_allowed = (exp_shift_allowed * allowed_mask.to(card_logits_all.dtype)).sum(dim=1)
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
            rows_present = torch.zeros((B,), dtype=torch.bool, device=device_local)
            rows_present[sample_idx_per_legal] = True
            bad_sum = (~torch.isfinite(sum_allowed)) | (sum_allowed <= 0)
            bad_rows = (bad_sum & rows_present)
            if bool(bad_rows.any().item() if bad_rows.numel() > 0 else False):
                r = int(torch.nonzero(bad_rows, as_tuple=False).flatten()[0].item())
                allow_ids = torch.nonzero(allowed_mask[r], as_tuple=False).flatten()
                v = card_logits_all[r, allow_ids] if allow_ids.numel() > 0 else torch.empty(0, device=device_local)
                ma = float(max_allowed[r].item()) if torch.isfinite(max_allowed[r]) else None
                ssum = float(sum_allowed[r].item()) if torch.isfinite(sum_allowed[r]) else None
                # Detailed per-row numerics
                diff = (card_logits_all[r] - max_allowed[r])
                diff_allowed = diff[allow_ids] if allow_ids.numel() > 0 else torch.empty(0, device=device_local)
                exp_allowed = torch.exp(diff_allowed) if diff_allowed.numel() > 0 else torch.empty(0, device=device_local)
                def _to_list(t):
                    return [float(x.item()) for x in t] if t.numel() > 0 else []
                raise RuntimeError(
                    f"compute_two_stage_logp: sum_allowed invalid at row={r}; allow_ids={[int(i.item()) for i in allow_ids]}, "
                    f"logits_allowed={_to_list(v)}, max_allowed={ma}, sum_allowed={ssum}, "
                    f"diff_allowed={_to_list(diff_allowed)}, exp_diff_allowed={_to_list(exp_allowed)}"
                )
        lse_allowed = max_allowed + torch.log(torch.clamp_min(sum_allowed, 1e-12))  # (B)
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
            # For rows that actually appear in sample_idx_per_legal, we must have at least one allowed card
            rows_present = torch.zeros((B,), dtype=torch.bool, device=device_local)
            rows_present[sample_idx_per_legal] = True
            allow_any = allowed_mask.any(dim=1)
            if bool(((rows_present & (~allow_any)).any()).item()):
                bad_rows = torch.nonzero(rows_present & (~allow_any), as_tuple=False).flatten().tolist()
                raise RuntimeError(f"compute_two_stage_logp: allowed_mask empty for rows {bad_rows}")
            if not torch.isfinite(card_logits_all[rows_present]).all():
                raise RuntimeError("compute_two_stage_logp: card_logits_all non-finite on present rows")
            if not torch.isfinite(lse_allowed[rows_present]).all():
                bad_rows = torch.nonzero(~torch.isfinite(lse_allowed) & rows_present, as_tuple=False).flatten().tolist()
                raise RuntimeError(f"compute_two_stage_logp: lse_allowed non-finite for rows {bad_rows}")
        logp_cards_allowed_per_legal = (card_logits_all[sample_idx_per_legal, played_ids_mb] - lse_allowed[sample_idx_per_legal])  # (M)

        # Capture logits per legal using action embeddings
        cap_logits = (a_emb_mb * state_proj[sample_idx_per_legal]).sum(dim=1)
        if not torch.isfinite(cap_logits).all():
            raise RuntimeError("compute_two_stage_logp: cap_logits non-finite")
        # Group-wise logsumexp over (sample, card)
        group_ids = sample_idx_per_legal * 40 + played_ids_mb
        num_groups = B * 40
        group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device_local)
        group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
        gmax_per_legal = group_max[group_ids]
        exp_shifted = torch.exp(cap_logits - gmax_per_legal).to(cap_logits.dtype)
        group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device_local)
        group_sum.index_add_(0, group_ids, exp_shifted)
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
            denom = group_sum[group_ids]
            bad = (~torch.isfinite(denom)) | (denom <= 0)
            if bool(bad.any().item() if denom.numel() > 0 else False):
                r = int(torch.nonzero(bad, as_tuple=False).flatten()[0].item()) if denom.numel() > 0 else -1
                s = int(sample_idx_per_legal[r].item()) if r >= 0 else -1
                chosen = int(played_ids_mb[r].item()) if r >= 0 else -1
                # capture stats for that sample/card
                cap_grp = cap_logits[(sample_idx_per_legal == s) & (played_ids_mb == chosen)]
                def _st(t):
                    return {'min': float(t.min().item()) if t.numel() > 0 else None,
                            'max': float(t.max().item()) if t.numel() > 0 else None,
                            'mean': float(t.mean().item()) if t.numel() > 0 else None,
                            'numel': int(t.numel())}
                raise RuntimeError(f"compute_two_stage_logp: capture group denominator invalid at row={r}, sample={s}, card={chosen}, cap_logits_stats={_st(cap_grp)}")
        lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
        logp_cap_per_legal = cap_logits - lse_per_legal

        logp_total_per_legal = (logp_cards_allowed_per_legal + logp_cap_per_legal)
        if not torch.isfinite(logp_total_per_legal).all():
            # Provide diagnostics for the first bad row
            idx_bad = torch.nonzero(~torch.isfinite(logp_total_per_legal), as_tuple=False).flatten()
            r = int(idx_bad[0].item()) if idx_bad.numel() > 0 else -1
            s = int(sample_idx_per_legal[r].item()) if r >= 0 else -1
            chosen = int(played_ids_mb[r].item()) if r >= 0 else -1
            # Per-sample debug
            allow_count = int(allowed_mask[s].sum().item()) if s >= 0 else -1
            max_allowed_s = float(max_allowed[s].item()) if s >= 0 else None
            masked_row = masked_logits[s] if s >= 0 else None
            def _stats_row(t):
                return {'min': float(t.min().item()) if t is not None and t.numel() > 0 else None,
                        'max': float(t.max().item()) if t is not None and t.numel() > 0 else None,
                        'mean': float(t.mean().item()) if t is not None and t.numel() > 0 else None,
                        'numel': int(t.numel()) if t is not None else 0}
            raise RuntimeError(
                f"compute_two_stage_logp: logp_total_per_legal non-finite at row={r}, sample={s}, card={chosen}; "
                f"allow_count={allow_count}, max_allowed={max_allowed_s}, masked_logits_stats={_stats_row(masked_row)}, "
                f"card_logits_stats={_stats_row(card_logits_all[s]) if s>=0 else None}, cap_logits_stats={_stats_row(cap_logits[(sample_idx_per_legal==s)]) if s>=0 else None}"
            )
        return logp_total_per_legal


class CentralValueNet(torch.nn.Module):
    """
    Critico condizionato: usa StateEncoderCompact (256) + belief (120→64).
    """
    def __init__(self, obs_dim=10823, state_encoder: StateEncoderCompact = None):
        super().__init__()
        self.state_enc = state_encoder if state_encoder is not None else StateEncoderCompact()
        self.belief_head = nn.Sequential(nn.Linear(120, 64), nn.ReLU())
        self.belief_net = BeliefNet(in_dim=256, hidden_dim=512)
        # Partner-aware belief features (come nell'actor)
        self.belief_card_emb = nn.Parameter(torch.randn(40, 32) * 0.02)
        self.partner_gate = nn.Sequential(nn.Linear(256, 32), nn.Sigmoid())
        self.opp_gate = nn.Sequential(nn.Linear(256, 32), nn.Sigmoid())
        # CTDE opzionale: usa others_hands (3x40) per modulare state_feat via FiLM
        self.ctde_cond = nn.Sequential(
            nn.Linear(120, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.ctde_scale = nn.Linear(128, 256)
        self.ctde_shift = nn.Linear(128, 256)
        # Stato 256 + belief 64 + partner 32 + opp 32 = 384
        self.head = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.to(device)

    def forward(self, obs: torch.Tensor, seat_team_vec: torch.Tensor = None, others_hands: torch.Tensor = None) -> torch.Tensor:
        target_device = next(self.parameters()).device
        if torch.is_tensor(obs):
            if (obs.device == target_device) and (obs.dtype == torch.float32):
                x_obs = obs
            elif obs.device == target_device:
                x_obs = obs.to(dtype=torch.float32)
            else:
                x_obs = obs.to(device=target_device, dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if seat_team_vec is None:
            seat_team_vec = torch.zeros((x_obs.size(0), 6), dtype=torch.float32, device=x_obs.device)
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            if (seat_team_vec.device != x_obs.device) or (seat_team_vec.dtype != torch.float32):
                seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        # Belief neurale interno (ignora belief_summary esterno)
        state_feat = self.state_enc(x_obs, seat_team_vec)
        bn_dtype = self.belief_net.fc_in.weight.dtype
        if state_feat.dtype != bn_dtype:
            state_feat = state_feat.to(dtype=bn_dtype)
        # CTDE FiLM gating se others_hands è disponibile (training centralizzato)
        if others_hands is not None:
            oh = others_hands
            if not torch.is_tensor(oh):
                oh = torch.as_tensor(oh, dtype=torch.float32, device=x_obs.device)
            else:
                oh = oh.to(x_obs.device, dtype=torch.float32)
            if oh.dim() == 2 and oh.size(1) == 120:
                oh_flat = oh
            elif oh.dim() == 3 and oh.size(1) == 3 and oh.size(2) == 40:
                oh_flat = oh.view(oh.size(0), -1)
            else:
                from utils.fallback import notify_fallback
                notify_fallback('models.critic.forward.others_hands_shape')
            cond = self.ctde_cond(oh_flat)
            scale = torch.sigmoid(self.ctde_scale(cond))  # (B,256) in (0,1)
            shift = torch.tanh(self.ctde_shift(cond)) * 0.1  # small bias
            state_feat = state_feat * (0.5 + scale) + shift
        hand_table = x_obs[:, :83]
        hand_mask = hand_table[:, :40] > 0.5
        table_mask = hand_table[:, 43:83] > 0.5
        captured = x_obs[:, 83:165]
        cap0_mask = captured[:, :40] > 0.5
        cap1_mask = captured[:, 40:80] > 0.5
        visible_mask = hand_mask | table_mask | cap0_mask | cap1_mask
        b_logits = self.belief_net(state_feat)
        b_logits = torch.nan_to_num(b_logits, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-30.0, 30.0)
        b_probs_flat = self.belief_net.probs(b_logits, visible_mask)
        belief_feat = self.belief_head(b_probs_flat)
        # Partner/opponent channel split con gating
        partner_slice = b_probs_flat[:, 40:80]
        opps_slice = b_probs_flat[:, 0:40] + b_probs_flat[:, 80:120]
        emb = self.belief_card_emb
        partner_feat = torch.matmul(partner_slice, emb)
        opp_feat = torch.matmul(opps_slice, emb)
        pg = self.partner_gate(state_feat)
        og = self.opp_gate(state_feat)
        partner_feat = partner_feat * pg
        opp_feat = opp_feat * og
        B_head = state_feat.size(0)
        head_in = torch.empty((B_head, 256 + 64 + 32 + 32), dtype=state_feat.dtype, device=state_feat.device)
        hp = 0
        head_in[:, hp:hp+256] = state_feat; hp += 256
        head_in[:, hp:hp+64] = belief_feat; hp += 64
        head_in[:, hp:hp+32] = partner_feat; hp += 32
        head_in[:, hp:hp+32] = opp_feat; hp += 32
        out = self.head(head_in)
        return out.squeeze(-1)

    def forward_from_state(self, state_feat: torch.Tensor, x_obs: torch.Tensor,
                            others_hands: torch.Tensor = None, visible_mask_40: torch.Tensor = None) -> torch.Tensor:
        """Valuta il valore partendo da feature di stato (256) già calcolate.
        Usa la stessa testa belief/gating del critico.
        """
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        sf = state_feat
        # CTDE gating opzionale
        if others_hands is not None:
            oh = others_hands
            if not torch.is_tensor(oh):
                oh = torch.as_tensor(oh, dtype=torch.float32, device=x_obs.device)
            else:
                oh = oh.to(x_obs.device, dtype=torch.float32)
            if oh.dim() == 2 and oh.size(1) == 120:
                oh_flat = oh
            elif oh.dim() == 3 and oh.size(1) == 3 and oh.size(2) == 40:
                oh_flat = oh.view(oh.size(0), -1)
            else:
                notify_fallback('models.critic.forward_from_state.others_hands_shape')
            cond = self.ctde_cond(oh_flat)
            scale = torch.sigmoid(self.ctde_scale(cond))  # (B,256) in (0,1)
            shift = torch.tanh(self.ctde_shift(cond)) * 0.1
            sf = sf * (0.5 + scale) + shift
        # visible mask
        hand_table = x_obs[:, :83]
        hand_mask = hand_table[:, :40] > 0.5
        table_mask = hand_table[:, 43:83] > 0.5
        captured = x_obs[:, 83:165]
        cap0_mask = captured[:, :40] > 0.5
        cap1_mask = captured[:, 40:80] > 0.5
        visible_mask_local = hand_mask | table_mask | cap0_mask | cap1_mask
        visible_mask = (visible_mask_40 if visible_mask_40 is not None else visible_mask_local)
        b_logits = self.belief_net(sf)
        b_logits = torch.nan_to_num(b_logits, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-30.0, 30.0)
        b_probs_flat = self.belief_net.probs(b_logits, visible_mask)
        b_probs_flat = torch.nan_to_num(b_probs_flat, nan=0.0, posinf=0.0, neginf=0.0)
        belief_feat = self.belief_head(b_probs_flat)
        partner_slice = b_probs_flat[:, 40:80]
        opps_slice = b_probs_flat[:, 0:40] + b_probs_flat[:, 80:120]
        emb = self.belief_card_emb
        partner_feat = torch.matmul(partner_slice, emb)
        opp_feat = torch.matmul(opps_slice, emb)
        pg = self.partner_gate(sf)
        og = self.opp_gate(sf)
        partner_feat = partner_feat * pg
        opp_feat = opp_feat * og
        head_in = torch.empty((sf.size(0), 256 + 64 + 32 + 32), dtype=sf.dtype, device=sf.device)
        hp = 0
        head_in[:, hp:hp+256] = sf; hp += 256
        head_in[:, hp:hp+64] = belief_feat; hp += 64
        head_in[:, hp:hp+32] = partner_feat; hp += 32
        head_in[:, hp:hp+32] = opp_feat; hp += 32
        out = self.head(head_in)
        return out.squeeze(-1)

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        _ = strict  # keep arg for external callers; force non-strict loading
        return super().load_state_dict(state_dict, strict=False)

