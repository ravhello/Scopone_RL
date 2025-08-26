import torch
import torch.nn as nn
import os
from utils.device import get_compute_device
from utils.fallback import notify_fallback

device = get_compute_device()

# Optional: alias to torch._dynamo.disable for eager-only helpers
try:
    import torch._dynamo as _dynamo  # type: ignore
    _dynamo_disable = _dynamo.disable  # type: ignore[attr-defined]
except Exception:
    def _dynamo_disable(fn):  # type: ignore
        return fn


class StateEncoder10823(nn.Module):
    """
    Encoda l'osservazione 10823-dim in un vettore di contesto (256-dim),
    riutilizzando la stessa scomposizione semantica già usata nelle reti correnti.
    """
    def __init__(self, obs_dim: int = 10823):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.hand_table_processor = nn.Sequential(nn.Linear(83, 64), nn.ReLU())
        self.captured_processor = nn.Sequential(nn.Linear(82, 64), nn.ReLU())
        self.history_processor = nn.Sequential(nn.Linear(10320, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.stats_processor = nn.Sequential(nn.Linear(334, 64), nn.ReLU())
        # Seat/team embedding: 6-dim (4 seat one-hot + 1 ally flag + 1 opponent flag)
        self.seat_head = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        self.combiner = nn.Sequential(nn.Linear(128 + 64 * 4 + 32, 256), nn.ReLU())
        self.to(device)

    def forward(self, obs: torch.Tensor, seat_team_vec: torch.Tensor = None) -> torch.Tensor:
        import torch.nn.functional as F
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.device != device:
            obs = obs.to(device)

        x1 = F.relu(self.backbone[0](obs), inplace=True)
        x2 = F.relu(self.backbone[2](x1), inplace=True)
        x3 = F.relu(self.backbone[4](x2), inplace=True)
        backbone_features = F.relu(self.backbone[6](x3), inplace=True)

        hand_table = obs[:, :83]
        captured = obs[:, 83:165]
        history = obs[:, 169:10489]
        stats = obs[:, 10489:]

        hand_table_features = F.relu(self.hand_table_processor[0](hand_table), inplace=True)
        captured_features = F.relu(self.captured_processor[0](captured), inplace=True)
        history_features = F.relu(self.history_processor[0](history), inplace=True)
        history_features = F.relu(self.history_processor[2](history_features), inplace=True)
        stats_features = F.relu(self.stats_processor[0](stats), inplace=True)

        if seat_team_vec is None:
            # fallback: vettore zero
            notify_fallback('models.state_encoder10823.seat_team_missing')
            seat_team_vec = torch.zeros((obs.size(0), 6), dtype=torch.float32, device=obs.device)
        elif seat_team_vec.dim() == 1:
            seat_team_vec = seat_team_vec.unsqueeze(0)

        seat_feat = F.relu(self.seat_head[0](seat_team_vec), inplace=True)

        combined = torch.cat([
            backbone_features,
            hand_table_features,
            captured_features,
            history_features,
            stats_features,
            seat_feat
        ], dim=1)

        context = F.relu(self.combiner[0](combined), inplace=True)
        return context  # (B,256)


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
    def __init__(self):
        super().__init__()
        # Card embedding for permutation-invariant set encoding (40 card IDs)
        self.card_emb = nn.Parameter(torch.randn(40, 32) * 0.02)
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

        # Stats and seat/team (dimensione variabile → LazyLinear)
        self.stats_processor = nn.Sequential(nn.LazyLinear(64), nn.ReLU())
        self.seat_head = nn.Sequential(nn.Linear(6, 32), nn.ReLU())

        # Combiner to 256-d state context
        # Inputs: set_merge(64) + hist(64) + stats(64) + seat(32) = 224 → 256
        # Se OBS_INCLUDE_DEALER=1 aggiungiamo +4 nelle stats. Usiamo Linear(224,256) e
        # deleghiamo a self.stats_processor (LazyLinear) l'adattamento alla nuova dimensione.
        self.combiner = nn.Sequential(nn.Linear(224, 256), nn.ReLU())
        self.to(device)

    def forward(self, obs: torch.Tensor, seat_team_vec: torch.Tensor = None) -> torch.Tensor:
        import torch.nn.functional as F
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.device != device:
            obs = obs.to(device)

        B = obs.size(0)
        D = obs.size(1)
        # Calcola k in modo robusto dal totale D sapendo le combinazioni possibili delle stats.
        # D = 165 + 61*k + stats_len, con stats_len = 99 + [0/120] + [0/10] + [0/150]
        base_prefix = 165
        base_stats = 99
        # Opzioni possibili nelle stats: inferred(120), scopa_probs(10), rank_probs(150), dealer(4)
        option_dims = [120, 10, 150, 4]
        # Trova k (<=40) che rende remainder compatibile con una somma base_stats + subset(option_dims)
        k = 0
        found = False
        for kk in range(40, -1, -1):
            rem = D - base_prefix - 61 * kk
            if rem < base_stats:
                continue
            delta = rem - base_stats
            # Verifica se delta è somma di un sottoinsieme di option_dims
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
        if not found:
            # Fallback conservativo
            notify_fallback('models.state_encoder_compact.heuristic_k')
            k = max(0, min(40, int((D - base_prefix - base_stats) // 61)))
        # Sezioni
        hand_table = obs[:, :83]
        captured = obs[:, 83:165]
        hist_start = 165
        hist_end = 165 + 61 * k
        history = obs[:, hist_start:hist_end]
        stats = obs[:, hist_end:]

        # ----- Set encoders -----
        # hand_table: [hand(40) | other_counts(3) | table(40)]
        hand_mask = hand_table[:, :40]
        other_counts = hand_table[:, 40:43]
        table_mask = hand_table[:, 43:83]
        # Sum-pool embeddings
        card_emb = self.card_emb
        hand_feat = torch.matmul(hand_mask, card_emb)           # (B,32)
        table_feat = torch.matmul(table_mask, card_emb)         # (B,32)
        other_cnt_feat = self.counts_head_hand(other_counts)    # (B,16)
        # Cross-attention mano↔tavolo (usa sequenze lunghezza 40 con padding via mask)
        hand_present = (hand_mask > 0.5)
        table_present = (table_mask > 0.5)
        hand_seq = hand_mask.unsqueeze(-1) * card_emb           # (B,40,32)
        table_seq = table_mask.unsqueeze(-1) * card_emb         # (B,40,32)
        # key_padding_mask: True = ignora (assenza)
        hand_kpm = (~hand_present)
        table_kpm = (~table_present)
        # Safe MHA wrapper per evitare NaN quando tutte le key sono mascherate
        def _safe_mha(mha: nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kpm: torch.Tensor) -> torch.Tensor:
            # kpm shape (B,S) boolean; True = masked
            B = q.size(0)
            out = torch.zeros_like(q)
            if kpm.dim() == 2 and bool((~kpm.all(dim=1)).any()):
                valid = (~kpm.all(dim=1)).nonzero(as_tuple=False).flatten()
                o, _ = mha(query=q.index_select(0, valid),
                           key=k.index_select(0, valid),
                           value=v.index_select(0, valid),
                           key_padding_mask=kpm.index_select(0, valid))
                # Ensure dtype matches destination to avoid Float/Half mismatch under autocast
                if o.dtype != out.dtype:
                    o = o.to(dtype=out.dtype)
                out.index_copy_(0, valid, o)
            return out
        # Q=mano, K,V=tavolo → informazioni su prese possibili dal tavolo
        h2t = _safe_mha(self.cross_attn_h2t, hand_seq, table_seq, table_seq, table_kpm)
        # Q=tavolo, K,V=mano → come il tavolo è influenzato dalla mano
        t2h = _safe_mha(self.cross_attn_t2h, table_seq, hand_seq, hand_seq, hand_kpm)
        # masked mean pooling su dimensione sequenza
        def masked_mean(x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
            m = present.unsqueeze(-1).to(x.dtype)
            summed = (x * m).sum(dim=1)
            denom = m.sum(dim=1).clamp_min(1.0)
            return summed / denom
        hand_attn_feat = masked_mean(h2t, hand_present)         # (B,32)
        table_attn_feat = masked_mean(t2h, table_present)       # (B,32)

        # captured: [team0(40) | team1(40) | counts(2)]
        cap0_mask = captured[:, :40]
        cap1_mask = captured[:, 40:80]
        cap_counts = captured[:, 80:82]
        cap0_feat = torch.matmul(cap0_mask, card_emb)           # (B,32)
        cap1_feat = torch.matmul(cap1_mask, card_emb)           # (B,32)
        cap_cnt_feat = self.counts_head_cap(cap_counts)         # (B,16)

        set_merged = torch.cat([
            hand_feat, table_feat, cap0_feat, cap1_feat,
            hand_attn_feat, table_attn_feat,
            other_cnt_feat, cap_cnt_feat
        ], dim=1)
        set_feat = self.set_merge_head(set_merged)               # (B,64)

        # ----- History Transformer -----
        if k > 0:
            hist_reshaped = history.view(B, k, 61)               # (B,k,61)
            hproj = self.hist_proj(hist_reshaped)                # (B,k,64)
            pos_idx = torch.arange(k, device=obs.device, dtype=torch.long).unsqueeze(0).expand(B, k)
            hpos = self.hist_pos_emb(pos_idx)
            hseq = hproj + hpos
            henc = self.hist_encoder(hseq)                       # (B,k,64)
            hist_feat = henc.mean(dim=1)                         # (B,64)
        else:
            hist_feat = torch.zeros((B, 64), dtype=torch.float32, device=obs.device)

        # Stats and seat/team (stats_processor è LazyLinear e si adatta alla prima forward)
        stats_feat = self.stats_processor(stats)
        if seat_team_vec is None:
            seat_team_vec = torch.zeros((B, 6), dtype=torch.float32, device=obs.device)
        elif seat_team_vec.dim() == 1:
            seat_team_vec = seat_team_vec.unsqueeze(0)
        seat_feat = F.relu(self.seat_head[0](seat_team_vec), inplace=True)

        # Combine
        combined = torch.cat([set_feat, hist_feat, stats_feat, seat_feat], dim=1)  # (B,224)
        context = F.relu(self.combiner[0](combined), inplace=True)
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
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.device != device:
            actions = actions.to(device)
        return self.net(actions)


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
        self.act = nn.GELU()
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
        return torch.clamp(torch.exp(self._log_temp), 0.25, 4.0)

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
        if visible_mask_40 is not None:
            if visible_mask_40.dim() == 1:
                visible_mask_40 = visible_mask_40.unsqueeze(0)
            # azzera probabilità per carte visibili
            m = visible_mask_40.to(probs.dtype).unsqueeze(1)  # (B,1,40)
            probs = probs * (1.0 - m)
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
        self.belief_card_emb = nn.Parameter(torch.randn(40, 32) * 0.02)
        self.partner_gate = nn.Sequential(nn.Linear(256, 32), nn.Sigmoid())
        self.opp_gate = nn.Sequential(nn.Linear(256, 32), nn.Sigmoid())
        # Merge: stato 256 + belief_head 64 + partner 32 + opp 32 = 384
        self.merge = nn.Sequential(nn.Linear(384, 256), nn.ReLU())
        self.state_to_action = nn.Linear(256, 64)
        self.action_enc = ActionEncoder80(action_dim)
        # Embedding per la selezione carta (40 carte)
        self.card_emb_play = nn.Parameter(torch.randn(40, 64) * 0.02)
        # Cache di tutte le azioni one-hot (80 x 80) per calcolare logits pieni
        self.register_buffer('all_actions_eye', torch.eye(action_dim, dtype=torch.float32))
        # Cache embedding azioni (solo per inference)
        self._cached_action_emb = None
        self.to(device)

    @staticmethod
    def _visible_mask_from_obs(x_obs: torch.Tensor) -> torch.Tensor:
        hand_table = x_obs[:, :83]
        hand_mask = hand_table[:, :40] > 0.5
        table_mask = hand_table[:, 43:83] > 0.5
        captured = x_obs[:, 83:165]
        cap0_mask = captured[:, :40] > 0.5
        cap1_mask = captured[:, 40:80] > 0.5
        visible = hand_mask | table_mask | cap0_mask | cap1_mask
        return visible

    def compute_state_proj(self, obs: torch.Tensor, seat_team_vec: torch.Tensor = None) -> torch.Tensor:
        if torch.is_tensor(obs):
            x_obs = obs.to(device, dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if seat_team_vec is None:
            notify_fallback('models.actor.compute_state_proj.seat_team_missing')
            seat_team_vec = torch.zeros((x_obs.size(0), 6), dtype=torch.float32, device=x_obs.device)
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        state_feat = self.state_enc(x_obs, seat_team_vec)  # (B,256)
        # belief neurale interno con maschera carte visibili
        visible_mask = self._visible_mask_from_obs(x_obs)
        belief_logits = self.belief_net(state_feat)        # (B,120)
        belief_probs_flat = self.belief_net.probs(belief_logits, visible_mask)
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
        state_ctx = self.merge(torch.cat([state_feat, belief_feat, partner_feat, opp_feat], dim=1))  # (B,256)
        state_proj = self.state_to_action(state_ctx)  # (B,64)
        return state_proj

    def invalidate_action_cache(self) -> None:
        """Invalida la cache degli embedding delle azioni (usata in inference)."""
        self._cached_action_emb = None

    def get_action_emb_table_cached(self) -> torch.Tensor:
        """Ritorna la tabella (80,64) degli embedding azione usando la cache in inference.
        Viene ricalcolata on-demand quando assente o invalidata. Non traccia gradiente.
        """
        if self._cached_action_emb is not None:
            return self._cached_action_emb
        # Evita cattura in cudagraph/compile: calcola in eager e clona storage
        return self._compute_action_emb_table_eager()

    @_dynamo_disable
    def _compute_action_emb_table_eager(self) -> torch.Tensor:
        with torch.no_grad():
            # Clona l'input per assicurare storage indipendente
            eye = self.all_actions_eye.to(device=device, dtype=torch.float32).clone()
            tbl = self.action_enc(eye).detach().clone()
            # Materializza su device predefinito e contiguo
            tbl = tbl.to(device=device, dtype=torch.float32).contiguous()
            self._cached_action_emb = tbl
            return self._cached_action_emb

    def forward(self, obs: torch.Tensor, legals: torch.Tensor = None,
                seat_team_vec: torch.Tensor = None) -> torch.Tensor:
        # Stato: (B, D)
        if torch.is_tensor(obs):
            x_obs = obs.to(device, dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        # Seat/team: opzionale
        if seat_team_vec is None:
            from utils.fallback import notify_fallback
            notify_fallback('models.critic.forward.seat_team_missing')
            seat_team_vec = torch.zeros((x_obs.size(0), 6), dtype=torch.float32, device=x_obs.device)
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        # belief neurale interno con maschera carte visibili
        state_feat = self.state_enc(x_obs, seat_team_vec)  # (B,256)
        visible_mask = self._visible_mask_from_obs(x_obs)
        belief_logits = self.belief_net(state_feat)
        belief_probs_flat = self.belief_net.probs(belief_logits, visible_mask)
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
        state_ctx = self.merge(torch.cat([state_feat, belief_feat, partner_feat, opp_feat], dim=1))  # (B,256)
        state_proj = self.state_to_action(state_ctx)  # (B,64)

        if legals is None:
            # Calcola logits per tutte le 80 azioni: (B,64) @ (64,80) -> (B,80)
            all_actions = self.all_actions_eye
            # usa cache embedding azioni se disponibile
            try:
                action_emb = self.get_action_emb_table_cached()
            except Exception:
                from utils.fallback import notify_fallback
                notify_fallback('models.actor.action_emb_table_failed')
                action_emb = self.action_enc(all_actions)
            logits = torch.matmul(state_proj, action_emb.t())  # (B,80)
            return logits if logits.size(0) > 1 else logits.squeeze(0)
        # legals: (A,80). Calcola score per azioni legali via prodotto scalare
        if not torch.is_tensor(legals):
            legals_t = torch.as_tensor(legals, dtype=torch.float32, device=state_proj.device)
        else:
            legals_t = legals.to(state_proj.device, dtype=torch.float32)
        # B atteso = 1 in path di selezione
        # In training evita la tabella cache per mantenere gradiente
        if self.training:
            a_emb = self.action_enc(legals_t)
        else:
            try:
                a_tbl = self.get_action_emb_table_cached().to(dtype=state_proj.dtype, device=legals_t.device)
                a_emb = torch.matmul(legals_t, a_tbl)
            except Exception:
                from utils.fallback import notify_fallback
                notify_fallback('models.actor.action_emb_table_failed.legals')
                a_emb = self.action_enc(legals_t)  # fallback
        if state_proj.size(0) == 1:
            scores = torch.matmul(a_emb, state_proj.squeeze(0))  # (A)
        else:
            # Row-wise dot product: (A,64) ⊙ (A,64) → (A)
            scores = (a_emb * state_proj).sum(dim=1)
        return scores


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
        if torch.is_tensor(obs):
            x_obs = obs.to(device, dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if seat_team_vec is None:
            seat_team_vec = torch.zeros((x_obs.size(0), 6), dtype=torch.float32, device=x_obs.device)
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        # Belief neurale interno (ignora belief_summary esterno)
        state_feat = self.state_enc(x_obs, seat_team_vec)
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
                # fallback: tenta reshape
                from utils.fallback import notify_fallback
                notify_fallback('models.critic.forward.others_hands_shape')
                oh_flat = oh.view(oh.size(0), -1)
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
        out = self.head(torch.cat([state_feat, belief_feat, partner_feat, opp_feat], dim=1))
        return out.squeeze(-1)

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        _ = strict  # keep arg for external callers; force non-strict loading
        return super().load_state_dict(state_dict, strict=False)


