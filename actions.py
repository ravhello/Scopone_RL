# actions.py
from collections import defaultdict
import os
import torch

# Rimosso supporto numba e path legacy: usiamo solo implementazioni torch/python
# Device for action encoding/decoding. Default to CPU for env-side logic to avoid GPU micro-kernels
_ACTIONS_DEVICE_STR = os.environ.get("ACTIONS_DEVICE", os.environ.get('SCOPONE_DEVICE', 'cpu'))
device = torch.device(_ACTIONS_DEVICE_STR)

def encode_action(card, cards_to_capture):
    """
    Codifica un'azione utilizzando la rappresentazione a matrice:
    - Carta giocata: matrice 10x4 con 1 solo bit attivo (40 dim)
    - Carte catturate: matrice 10x4 con i bit attivi corrispondenti (40 dim)
    
    Totale: 80 dimensioni
    """
    # Inizializza le matrici sul device selezionato (di default CPU)
    played_card_matrix = torch.zeros((10, 4), dtype=torch.float32, device=device)
    captured_cards_matrix = torch.zeros((10, 4), dtype=torch.float32, device=device)
    
    # Codifica la carta giocata (forza ID)
    if isinstance(card, int):
        cid = int(card)
    else:
        rank, suit = card
        suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
        cid = (rank - 1) * 4 + suit_to_col[suit]
    row = (cid // 4)
    col = (cid % 4)
    played_card_matrix[row, col] = 1.0
    
    # Codifica le carte catturate
    for capt_card in cards_to_capture:
        if isinstance(capt_card, int):
            cid = int(capt_card)
        else:
            capt_rank, capt_suit = capt_card
            suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
            cid = (capt_rank - 1) * 4 + suit_to_col[capt_suit]
        capt_row = (cid // 4)
        capt_col = (cid % 4)
        captured_cards_matrix[capt_row, capt_col] = 1.0
    
    # Appiattisci le matrici e concatenale
    return torch.cat([played_card_matrix.reshape(-1), captured_cards_matrix.reshape(-1)], dim=0)

def encode_action_from_ids_gpu(played_id_t: torch.Tensor, captured_ids_t: torch.Tensor) -> torch.Tensor:
    """
    GPU-native variant: build the 80-d action vector directly from CUDA int tensors
    without converting to Python ints. Inputs can be scalar int64 tensor for played_id
    and 1-D int64 tensor for captured_ids (possibly empty).
    """
    # Ensure tensors are on target device (CPU by default) and correct dtypes/shapes
    pid = played_id_t.to(device=device, dtype=torch.long).reshape(())
    cap = captured_ids_t.to(device=device, dtype=torch.long).reshape(-1)

    played_card_matrix = torch.zeros((10, 4), dtype=torch.float32, device=device)
    captured_cards_matrix = torch.zeros((10, 4), dtype=torch.float32, device=device)

    # Set played bit
    prow = pid // 4
    pcol = pid % 4
    played_card_matrix[prow, pcol] = 1.0

    # Set captured bits if any
    if cap.numel() > 0:
        rows = (cap // 4).clamp_(0, 9)
        cols = (cap % 4).clamp_(0, 3)
        captured_cards_matrix[rows, cols] = 1.0

    return torch.cat([played_card_matrix.reshape(-1), captured_cards_matrix.reshape(-1)], dim=0)

def decode_action(action_vec):
    """
    Decodifica un vettore azione 80-dim in (played_id, [captured_ids]).
    """
    return decode_action_ids(action_vec)

# ----- FAST PATH: decode to card IDs (0..39) -----

# Helpers legacy rimossi: non necessari nelle nuove API ID-only

def _decode_ids(vec_t: torch.Tensor):
    if vec_t.dim() != 1 or vec_t.numel() != 80:
        raise ValueError("decode_action_ids richiede un vettore 80-dim")
    # Avoid per-element .item(); move once to CPU for indexing scalars
    played_idx = torch.argmax(vec_t[:40])
    captured_ids = torch.nonzero(vec_t[40:] > 0, as_tuple=False).flatten()
    played_idx_cpu = int(played_idx.item())
    captured_cpu = captured_ids.tolist()
    return played_idx_cpu, captured_cpu

def decode_action_ids(action_vec):
    """
    Decodifica un vettore azione 80-dim in (played_id, [captured_ids]) con card IDs 0..39.
    """
    if torch.is_tensor(action_vec):
        vec_t = action_vec.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        # supporta list o numpy array
        try:
            vec_t = torch.as_tensor(action_vec, dtype=torch.float32, device=device).reshape(-1)
        except Exception:
            vec_t = torch.tensor(list(action_vec), dtype=torch.float32, device=device).reshape(-1)
    return _decode_ids(vec_t)

# ----- FAST PATH: subset-sum su ID con numba -----
def find_sum_subsets_ids(table_ids, target_rank: int):
    """
    Restituisce liste di ID dal tavolo la cui somma dei rank è target_rank.
    Usa numba se disponibile per enumerare i sottoinsiemi via bitmask sulle rank.
    """
    if not table_ids:
        return []
    ids = torch.as_tensor([int(x) for x in table_ids], dtype=torch.long, device=device)
    ranks = (ids // 4) + 1
    n = int(ids.numel())
    if n <= 0:
        return []
    pos = torch.arange(n, device=device, dtype=torch.long)
    masks = torch.arange(1, 1 << n, device=device, dtype=torch.long)
    sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.long)
    sums = (sel * ranks.unsqueeze(0)).sum(dim=1)
    good = (sums == int(target_rank)).nonzero(as_tuple=False).flatten()
    results = []
    for gi in good:
        subset = ids[((masks[gi].unsqueeze(0) >> pos) & 1).bool()].tolist()
        results.append([int(x) for x in subset])
    return results

def get_valid_actions(game_state, current_player):
    """Azioni valide compute interamente su CUDA. Richiede stato in ID (0..39)."""
    cp = current_player
    hand = game_state["hands"][cp]
    table = game_state["table"]
    valid_actions = []
    if not hand:
        return valid_actions
    if not isinstance(hand[0], int) or (len(table) > 0 and not isinstance(table[0], int)):
        raise TypeError("get_valid_actions richiede game_state in ID (int)")
    from observation import RANK_OF_ID as _RANK_OF_ID
    hand_ids_t = torch.as_tensor(hand, dtype=torch.long, device=device)
    table_ids_t = torch.as_tensor(table or [], dtype=torch.long, device=device)
    for pid_t in hand_ids_t:
        pid = int(pid_t.item())
        prank = int(_RANK_OF_ID[pid].item())
        if table_ids_t.numel() > 0:
            table_ranks = _RANK_OF_ID[table_ids_t].to(torch.long)
            direct_ids_t = table_ids_t[table_ranks == prank]
        else:
            direct_ids_t = torch.empty(0, dtype=torch.long, device=device)
        if direct_ids_t.numel() > 0:
            for did_t in direct_ids_t:
                valid_actions.append(encode_action(pid, [int(did_t.item())]))
        else:
            n = int(table_ids_t.numel())
            if n > 0:
                pos = torch.arange(n, device=device, dtype=torch.long)
                masks = torch.arange(1, 1 << n, device=device, dtype=torch.long)
                sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.long)
                sums = (sel * (table_ranks.unsqueeze(0))).sum(dim=1)
                good = (sums == prank).nonzero(as_tuple=False).flatten()
                if good.numel() > 0:
                    for gi in good:
                        subset_ids = table_ids_t[((masks[gi].unsqueeze(0) >> pos) & 1).bool()].tolist()
                        valid_actions.append(encode_action(pid, subset_ids))
                else:
                    valid_actions.append(encode_action(pid, []))
            else:
                valid_actions.append(encode_action(pid, []))
    return valid_actions