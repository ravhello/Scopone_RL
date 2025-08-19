import torch
from observation import RANK_OF_ID, SUITCOL_OF_ID, PRIMIERA_VAL_T

def compute_final_score_breakdown_torch(game_state, rules=None):
    """
    Breakdown finale calcolato su CUDA; i valori restituiti sono tensori CUDA 0-D.
    """
    device = torch.device('cuda')
    squads = game_state["captured_squads"]

    rules = rules or {}
    re_bello_enabled = bool(rules.get("re_bello", False))
    napola_enabled = bool(rules.get("napola", False))
    napola_scoring = rules.get("napola_scoring", "fixed3")

    # Scope per team (conteggio semplice)
    scope0 = sum(1 for m in game_state["history"] if m.get("capture_type") == "scopa" and (m.get("player") in [0, 2]))
    scope1 = sum(1 for m in game_state["history"] if m.get("capture_type") == "scopa" and (m.get("player") in [1, 3]))

    # Carte totali (confronto su GPU) e ID catturati: preferisci mirror bitset se disponibili
    from observation import IDS_CUDA
    captured_bits_t = game_state.get('_captured_bits_t', None)
    if captured_bits_t is not None:
        mask0 = (((captured_bits_t[0] >> IDS_CUDA) & 1).to(torch.bool))
        mask1 = (((captured_bits_t[1] >> IDS_CUDA) & 1).to(torch.bool))
        ids0 = IDS_CUDA[mask0]
        ids1 = IDS_CUDA[mask1]
        c0_t = ids0.numel().to(torch.long)
        c1_t = ids1.numel().to(torch.long)
    else:
        ids0 = torch.as_tensor(squads[0] or [], dtype=torch.long, device=device)
        ids1 = torch.as_tensor(squads[1] or [], dtype=torch.long, device=device)
        c0_t = torch.tensor(len(squads[0]), dtype=torch.long, device=device)
        c1_t = torch.tensor(len(squads[1]), dtype=torch.long, device=device)
    pt_c0_t = (c0_t > c1_t).to(dtype=torch.long)
    pt_c1_t = (c1_t > c0_t).to(dtype=torch.long)

    # Denari (suit==0)
    den0_t = (SUITCOL_OF_ID[ids0] == 0).sum() if ids0.numel() > 0 else torch.zeros((), dtype=torch.long, device=device)
    den1_t = (SUITCOL_OF_ID[ids1] == 0).sum() if ids1.numel() > 0 else torch.zeros((), dtype=torch.long, device=device)
    pt_d0_t = (den0_t > den1_t).to(dtype=torch.long)
    pt_d1_t = (den1_t > den0_t).to(dtype=torch.long)

    # Settebello (ID=24)
    sb0_t = (ids0 == 24).any().to(dtype=torch.long)
    sb1_t = (ids1 == 24).any().to(dtype=torch.long)

    # Primiera: max per seme
    def _primiera_points(ids_t: torch.Tensor) -> torch.Tensor:
        if ids_t.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=device)
        ranks = RANK_OF_ID[ids_t].to(torch.long)
        suits = SUITCOL_OF_ID[ids_t].to(torch.long)
        vals = PRIMIERA_VAL_T[ranks]
        out = torch.zeros(4, dtype=torch.float32, device=device)
        try:
            out.scatter_reduce_(0, suits, vals, reduce='amax', include_self=True)
        except Exception:
            for s in range(4):
                mask = (suits == s)
                if mask.any():
                    out[s] = torch.max(vals[mask])
        return out.sum()
    prim0_t = _primiera_points(ids0)
    prim1_t = _primiera_points(ids1)
    pt_p0_t = (prim0_t > prim1_t).to(dtype=torch.long)
    pt_p1_t = (prim1_t > prim0_t).to(dtype=torch.long)

    # re_bello
    def _has_card_id_tensor(ids_t: torch.Tensor, rank: int, suit_col: int) -> int:
        target = (rank - 1) * 4 + suit_col
        return (ids_t == target).any().to(dtype=torch.long)
    rb0_t = _has_card_id_tensor(ids0, 10, 0) if re_bello_enabled else torch.zeros((), dtype=torch.long, device=device)
    rb1_t = _has_card_id_tensor(ids1, 10, 0) if re_bello_enabled else torch.zeros((), dtype=torch.long, device=device)

    # Napola
    def _napola_points(ids_t: torch.Tensor) -> torch.Tensor:
        if not napola_enabled or ids_t.numel() == 0:
            return torch.zeros((), dtype=torch.long, device=device)
        ranks = RANK_OF_ID[ids_t].to(torch.long)
        suits = SUITCOL_OF_ID[ids_t].to(torch.long)
        denari_ranks = ranks[suits == 0]
        if denari_ranks.numel() == 0:
            return torch.zeros((), dtype=torch.long, device=device)
        present = torch.zeros(11, dtype=torch.float32, device=device)
        present[denari_ranks.clamp(1, 10)] = 1.0
        # Richiede 1,2,3
        has123 = (present[1:4].sum() >= 3.0)
        if napola_scoring == "length":
            cprod = present[1:].cumprod(dim=0)
            length_t = cprod.sum().to(dtype=torch.long)
            return torch.where(has123, length_t, torch.zeros((), dtype=torch.long, device=device))
        fixed3_t = torch.tensor(3, dtype=torch.long, device=device)
        return torch.where(has123, fixed3_t, torch.zeros((), dtype=torch.long, device=device))
    np0_t = _napola_points(ids0)
    np1_t = _napola_points(ids1)

    scope0_t = torch.tensor(scope0, dtype=torch.long, device=device)
    scope1_t = torch.tensor(scope1, dtype=torch.long, device=device)
    total0_t = (pt_c0_t + pt_d0_t + sb0_t + pt_p0_t + scope0_t + (rb0_t if re_bello_enabled else torch.zeros((), dtype=torch.long, device=device)) + (np0_t if napola_enabled else torch.zeros((), dtype=torch.long, device=device))).to(torch.long)
    total1_t = (pt_c1_t + pt_d1_t + sb1_t + pt_p1_t + scope1_t + (rb1_t if re_bello_enabled else torch.zeros((), dtype=torch.long, device=device)) + (np1_t if napola_enabled else torch.zeros((), dtype=torch.long, device=device))).to(torch.long)

    breakdown = {
      0: {
         "carte": pt_c0_t,
         "denari": pt_d0_t,
         "settebello": sb0_t,
         "primiera": pt_p0_t,
         "scope": scope0_t,
         "re_bello": (rb0_t if re_bello_enabled else torch.zeros((), dtype=torch.long, device=device)),
         "napola": (np0_t if napola_enabled else torch.zeros((), dtype=torch.long, device=device)),
         "total": total0_t
      },
      1: {
         "carte": pt_c1_t,
         "denari": pt_d1_t,
         "settebello": sb1_t,
         "primiera": pt_p1_t,
         "scope": scope1_t,
         "re_bello": (rb1_t if re_bello_enabled else torch.zeros((), dtype=torch.long, device=device)),
         "napola": (np1_t if napola_enabled else torch.zeros((), dtype=torch.long, device=device)),
         "total": total1_t
      }
    }
    return breakdown

def compute_final_reward_from_breakdown_torch(breakdown):
    """
    Calcolo su CUDA della ricompensa finale; ritorna un tensore CUDA (2,).
    """
    device = torch.device('cuda')
    # I totali sono tensori 0-D long
    diff = (breakdown[0]["total"].to(torch.float32) - breakdown[1]["total"].to(torch.float32))
    pos = torch.round(diff * 10.0).to(dtype=torch.float32)
    return torch.stack([pos, -pos])  # shape (2,) CUDA

def compute_final_team_rewards_torch(game_state, rules=None):
    """Ritorna tensor CUDA shape (2,) con i punteggi finali team0, team1."""
    breakdown = compute_final_score_breakdown_torch(game_state, rules)
    return compute_final_reward_from_breakdown_torch(breakdown)
