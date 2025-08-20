import torch
import torch.nn as nn
from tests.torch_np import np

device = torch.device("cuda")


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
    Encoder per osservazione compatta (dim = 499 + 61*k), con storia-k trattata come
    k blocchi da 61 e pooling, senza slicing fisso dipendente da k.
    Sezioni:
      - hand_table: 83
      - captured: 82
      - history: 61*k  (k variabile)
      - stats: 334 (resto delle feature)
      - seat/team: 6 (passato separatamente)
    """
    def __init__(self):
        super().__init__()
        self.hand_table_processor = nn.Sequential(nn.Linear(83, 64), nn.ReLU())
        self.captured_processor = nn.Sequential(nn.Linear(82, 64), nn.ReLU())
        # history per-move head (61 -> 64), pooling su k mosse, poi un ulteriore layer 64->64
        self.history_move_head = nn.Sequential(nn.Linear(61, 64), nn.ReLU())
        self.history_pool_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.stats_processor = nn.Sequential(nn.Linear(334, 64), nn.ReLU())
        # Seat/team embedding: 6 -> 32
        self.seat_head = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        # Combiner: 64*4 + 32 -> 256
        self.combiner = nn.Sequential(nn.Linear(64 * 4 + 32, 256), nn.ReLU())
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
        # Calcola k dalla dimensione complessiva: D = 499 + 61*k
        k = max(0, int((D - 499) // 61))
        # Sezioni
        hand_table = obs[:, :83]
        captured = obs[:, 83:165]
        hist_start = 165
        hist_end = 165 + 61 * k
        history = obs[:, hist_start:hist_end]
        stats = obs[:, hist_end:hist_end + 334]

        # processa sezioni
        ht_feat = F.relu(self.hand_table_processor[0](hand_table), inplace=True)
        cap_feat = F.relu(self.captured_processor[0](captured), inplace=True)
        if k > 0:
            # reshaping in (B, k, 61)
            hist_reshaped = history.view(B, k, 61)
            # applica head per-move e poi pooling medio
            move_feats = self.history_move_head[0](hist_reshaped)  # (B, k, 64)
            move_feats = F.relu(move_feats, inplace=True)
            pooled = move_feats.mean(dim=1)  # (B, 64)
            hist_feat = F.relu(self.history_pool_head[0](pooled), inplace=True)
        else:
            hist_feat = torch.zeros((B, 64), dtype=torch.float32, device=obs.device)
        stats_feat = F.relu(self.stats_processor[0](stats), inplace=True)

        if seat_team_vec is None:
            seat_team_vec = torch.zeros((B, 6), dtype=torch.float32, device=obs.device)
        elif seat_team_vec.dim() == 1:
            seat_team_vec = seat_team_vec.unsqueeze(0)
        seat_feat = F.relu(self.seat_head[0](seat_team_vec), inplace=True)

        combined = torch.cat([ht_feat, cap_feat, hist_feat, stats_feat, seat_feat], dim=1)
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


class ActionConditionedActor(torch.nn.Module):
    """
    Actor realmente action-conditioned:
      - State encoder compatto (usa storia-k con pooling) → 256-d
      - Belief head (120 → 64)
      - Proiezione stato → 64 e scoring via prodotto scalare con embedding azione (80 → 64)
    """
    def __init__(self, obs_dim=10823, action_dim=80):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # Encoders
        self.state_enc = StateEncoderCompact()
        self.belief_head = nn.Sequential(nn.Linear(120, 64), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(256 + 64, 256), nn.ReLU())
        self.state_to_action = nn.Linear(256, 64)
        self.action_enc = ActionEncoder80(action_dim)
        # Cache di tutte le azioni one-hot (80 x 80) per calcolare logits pieni
        self.register_buffer('all_actions_eye', torch.eye(action_dim, dtype=torch.float32))
        self.to(torch.device('cuda'))

    def forward(self, obs: torch.Tensor, legals: torch.Tensor = None,
                seat_team_vec: torch.Tensor = None, belief_summary: torch.Tensor = None) -> torch.Tensor:
        # Stato: (B, D)
        if torch.is_tensor(obs):
            x_obs = obs.to(torch.device('cuda'), dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda'))
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        # Seat/team: opzionale
        if seat_team_vec is None:
            seat_team_vec = torch.zeros((x_obs.size(0), 6), dtype=torch.float32, device=x_obs.device)
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        # Belief: opzionale (120)
        if belief_summary is None:
            belief_summary = torch.zeros((x_obs.size(0), 120), dtype=torch.float32, device=x_obs.device)
        else:
            belief_summary = (belief_summary if torch.is_tensor(belief_summary) else torch.as_tensor(belief_summary, dtype=torch.float32))
            if belief_summary.dim() == 1:
                belief_summary = belief_summary.unsqueeze(0)
            belief_summary = belief_summary.to(x_obs.device, dtype=torch.float32)

        # Encode stato (già incorpora seat_team_vec internamente)
        state_feat = self.state_enc(x_obs, seat_team_vec)  # (B,256)
        belief_feat = self.belief_head(belief_summary)     # (B,64)
        state_ctx = self.merge(torch.cat([state_feat, belief_feat], dim=1))  # (B,256)
        state_proj = self.state_to_action(state_ctx)  # (B,64)

        if legals is None:
            # Calcola logits per tutte le 80 azioni: (B,64) @ (64,80) -> (B,80)
            all_actions = self.all_actions_eye.to(state_proj.device)
            action_emb = self.action_enc(all_actions)  # (80,64)
            logits = torch.matmul(state_proj, action_emb.t())  # (B,80)
            return logits if logits.size(0) > 1 else logits.squeeze(0)
        # legals: (A,80). Calcola score per azioni legali via prodotto scalare
        if not torch.is_tensor(legals):
            legals_t = torch.as_tensor(legals, dtype=torch.float32, device=state_proj.device)
        else:
            legals_t = legals.to(state_proj.device, dtype=torch.float32)
        # B atteso = 1 in path di selezione
        a_emb = self.action_enc(legals_t)  # (A,64)
        if state_proj.size(0) == 1:
            scores = torch.matmul(a_emb, state_proj.squeeze(0))  # (A)
        else:
            # Broadcast: per semplicità calcola logits pieni e poi indicizza (fallback raro)
            all_actions = self.all_actions_eye.to(state_proj.device)
            action_emb = self.action_enc(all_actions)  # (80,64)
            scores_full = torch.matmul(state_proj, action_emb.t())  # (B,80)
            # Mappa legals a indici (assume one-hot) per B=1 usa prima riga
            idx = torch.argmax(legals_t, dim=1)
            scores = scores_full[torch.arange(state_proj.size(0))[0], idx]
        return scores


class CentralValueNet(torch.nn.Module):
    """
    Critico condizionato: usa StateEncoderCompact (256) + belief (120→64).
    """
    def __init__(self, obs_dim=10823):
        super().__init__()
        self.state_enc = StateEncoderCompact()
        self.belief_head = nn.Sequential(nn.Linear(120, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.to(torch.device('cuda'))

    def forward(self, obs: torch.Tensor, seat_team_vec: torch.Tensor = None, belief_summary: torch.Tensor = None) -> torch.Tensor:
        if torch.is_tensor(obs):
            x_obs = obs.to(torch.device('cuda'), dtype=torch.float32)
        else:
            x_obs = torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda'))
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        if seat_team_vec is None:
            seat_team_vec = torch.zeros((x_obs.size(0), 6), dtype=torch.float32, device=x_obs.device)
        else:
            seat_team_vec = (seat_team_vec if torch.is_tensor(seat_team_vec) else torch.as_tensor(seat_team_vec, dtype=torch.float32))
            if seat_team_vec.dim() == 1:
                seat_team_vec = seat_team_vec.unsqueeze(0)
            seat_team_vec = seat_team_vec.to(x_obs.device, dtype=torch.float32)
        if belief_summary is None:
            belief_summary = torch.zeros((x_obs.size(0), 120), dtype=torch.float32, device=x_obs.device)
        else:
            belief_summary = (belief_summary if torch.is_tensor(belief_summary) else torch.as_tensor(belief_summary, dtype=torch.float32))
            if belief_summary.dim() == 1:
                belief_summary = belief_summary.unsqueeze(0)
            belief_summary = belief_summary.to(x_obs.device, dtype=torch.float32)
        state_feat = self.state_enc(x_obs, seat_team_vec)
        belief_feat = self.belief_head(belief_summary)
        out = self.head(torch.cat([state_feat, belief_feat], dim=1))
        return out.squeeze(-1)

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        return super().load_state_dict(state_dict, strict=False)


