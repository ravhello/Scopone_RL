import os
import torch
from observation import RANK_OF_ID, SUITCOL_OF_ID, PRIMIERA_VAL_T

def compute_final_score_breakdown(game_state, rules=None):
    """
    Breakdown finale calcolato su CUDA; i valori restituiti sono scalari Python.
    """
    # Esegui su CPU per evitare dipendenze forzate da GPU (env/logica punteggio lato CPU)
    device = torch.device(os.environ.get('REW_DEVICE', os.environ.get('SCOPONE_DEVICE', 'cpu')))
    # Allinea i tensori costanti di indicizzazione al device di lavoro per evitare mismatch
    RID = RANK_OF_ID.to(device=device)
    SUID = SUITCOL_OF_ID.to(device=device)
    PV = PRIMIERA_VAL_T.to(device=device)
    squads = game_state["captured_squads"]

    rules = rules or {}
    re_bello_enabled = bool(rules.get("re_bello", False))
    napola_enabled = bool(rules.get("napola", False))
    napola_scoring = rules.get("napola_scoring", "fixed3")

    # Scope per team (conteggio su CPU per estrazione, ma computo è banale)
    scope0 = 0
    scope1 = 0
    for move in game_state["history"]:
        if move.get("capture_type") == "scopa":
            if move.get("player") in [0, 2]:
                scope0 += 1
            else:
                scope1 += 1

    # Carte totali
    c0 = len(squads[0])
    c1 = len(squads[1])
    pt_c0, pt_c1 = ((1, 0) if c0 > c1 else (0, 1) if c1 > c0 else (0, 0))

    # Denari (suit==0)
    ids0 = torch.as_tensor(squads[0] or [], dtype=torch.long, device=device)
    ids1 = torch.as_tensor(squads[1] or [], dtype=torch.long, device=device)
    den0 = int(((ids0.numel() > 0) and (SUID[ids0] == 0).sum().item()) or 0)
    den1 = int(((ids1.numel() > 0) and (SUID[ids1] == 0).sum().item()) or 0)
    pt_d0, pt_d1 = ((1, 0) if den0 > den1 else (0, 1) if den1 > den0 else (0, 0))

    # Settebello (ID=24)
    sb0 = int((ids0 == 24).any().item())
    sb1 = int((ids1 == 24).any().item())

    # Primiera: max per seme
    def _primiera_points(ids_t: torch.Tensor) -> float:
        if ids_t.numel() == 0:
            return 0.0
        ranks = RID[ids_t].to(torch.long)
        suits = SUID[ids_t].to(torch.long)
        vals = PV[ranks]
        out = torch.zeros(4, dtype=torch.float32, device=device)
        try:
            out.scatter_reduce_(0, suits, vals, reduce='amax', include_self=True)
        except Exception:
            for s in range(4):
                mask = (suits == s)
                if mask.any():
                    out[s] = torch.max(vals[mask])
        return float(out.sum().item())
    prim0 = _primiera_points(ids0)
    prim1 = _primiera_points(ids1)
    pt_p0, pt_p1 = ((1, 0) if prim0 > prim1 else (0, 1) if prim1 > prim0 else (0, 0))

    # re_bello
    def _has_card_id_tensor(ids_t: torch.Tensor, rank: int, suit_col: int) -> int:
        target = (rank - 1) * 4 + suit_col
        return int((ids_t == target).any().item())
    rb0 = _has_card_id_tensor(ids0, 10, 0) if re_bello_enabled else 0
    rb1 = _has_card_id_tensor(ids1, 10, 0) if re_bello_enabled else 0

    # Napola
    def _napola_points(ids_t: torch.Tensor) -> int:
        if not napola_enabled:
            return 0
        if ids_t.numel() == 0:
            return 0
        ranks = RANK_OF_ID[ids_t].to(torch.long)
        suits = SUITCOL_OF_ID[ids_t].to(torch.long)
        denari_ranks = ranks[suits == 0]
        if denari_ranks.numel() == 0:
            return 0
        present = torch.zeros(11, dtype=torch.float32, device=device)
        present[denari_ranks.clamp(1, 10)] = 1.0
        # Richiede 1,2,3
        if float(present[1:4].sum().item()) < 3.0:
            return 0
        if napola_scoring == "length":
            cprod = present[1:].cumprod(dim=0)
            return int(cprod.sum().item())
        return 3
    np0 = _napola_points(ids0)
    np1 = _napola_points(ids1)

    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scope0 + rb0 + np0
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scope1 + rb1 + np1

    breakdown = {
      0: {
         "carte": pt_c0,
         "denari": pt_d0,
         "settebello": sb0,
         "primiera": pt_p0,
         "scope": scope0,
         "re_bello": rb0 if re_bello_enabled else 0,
         "napola": np0 if napola_enabled else 0,
         "total": total0
      },
      1: {
         "carte": pt_c1,
         "denari": pt_d1,
         "settebello": sb1,
         "primiera": pt_p1,
         "scope": scope1,
         "re_bello": rb1 if re_bello_enabled else 0,
         "napola": np1 if napola_enabled else 0,
         "total": total1
      }
    }
    return breakdown

def compute_final_reward_from_breakdown(breakdown):
    """
    Calcola la ricompensa finale come differenza di punteggio tra i team.
    Ritorna un dizionario {0: reward_team0, 1: reward_team1}.
    """
    device = torch.device(os.environ.get('REW_DEVICE', os.environ.get('SCOPONE_DEVICE', 'cpu')))
    diff = torch.tensor(breakdown[0]["total"] - breakdown[1]["total"], dtype=torch.float32, device=device)
    # Mantieni reward final-only senza moltiplicatore artificiale
    pos = int(diff.item())
    neg = -pos
    return {0: pos, 1: neg}
