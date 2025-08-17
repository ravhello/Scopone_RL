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
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
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
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
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
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32, device=device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.device != device:
            actions = actions.to(device)
        return self.net(actions)


class ActionConditionedActor(torch.nn.Module):
    def __init__(self, obs_dim=10823, action_dim=80):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim)
        )
        self.to(torch.device('cuda'))

    def forward(self, obs: torch.Tensor, legals: torch.Tensor = None, seat_team_vec: torch.Tensor = None) -> torch.Tensor:
        # obs: (B, obs_dim) or (obs_dim,)
        if torch.is_tensor(obs):
            x = obs.to(torch.device('cuda'), dtype=torch.float32)
        else:
            x = torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda'))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Raw action logits for full action space
        raw_logits = self.net(x)  # (B, action_dim)
        if legals is None:
            # Keep batch if B>1, otherwise return vector
            return raw_logits if raw_logits.size(0) > 1 else raw_logits.squeeze(0)
        # legals: (A, 80) one-hot-like masks per legal action. Score each by dot product
        if not torch.is_tensor(legals):
            legals_t = torch.as_tensor(legals, dtype=torch.float32, device=torch.device('cuda'))
        else:
            legals_t = legals.to(torch.device('cuda'), dtype=torch.float32)
        # B is expected to be 1 in this path
        logits_vec = raw_logits.squeeze(0)
        legal_scores = torch.matmul(legals_t, logits_vec.reshape(-1, 1)).squeeze(-1)  # (A)
        return legal_scores

    # Accept legacy checkpoints by loading non-strictly
    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        return super().load_state_dict(state_dict, strict=False)


class CentralValueNet(torch.nn.Module):
    def __init__(self, obs_dim=10823):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.to(torch.device('cuda'))

    def forward(self, obs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if torch.is_tensor(obs):
            x = obs.to(torch.device('cuda'), dtype=torch.float32)
        else:
            x = torch.as_tensor(obs, dtype=torch.float32, device=torch.device('cuda'))
        return self.net(x).squeeze(-1)

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        return super().load_state_dict(state_dict, strict=False)


