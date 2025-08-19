# actions.py
from collections import defaultdict
import torch

# Accel opzionale con numba per enumerazione subset-sum (DISABLED to keep torch-only path)
NUMBA_AVAILABLE = False
def njit(*args, **kwargs):
    def _decorator(fn):
        return fn
    return _decorator

def _find_sum_subsets_fast_gpu(table_cards, target_rank):
    """GPU-accelerated version that finds all subsets summing to target_rank"""
    n = len(table_cards)
    if n == 0:
        return []
    if n > 10:  # Limit per evitare esplosione di memoria (2^10 = 1024 masks)
        # Fallback to CPU for very large tables
        return _find_sum_subsets_fast_cpu(table_cards, target_rank)
    
    # Crea tensori su GPU
    ranks = torch.tensor([c[0] if isinstance(c, tuple) else (c // 4 + 1) for c in table_cards], 
                         dtype=torch.long, device=device)
    
    # Genera tutte le maschere possibili (2^n - 1, escludendo il set vuoto)
    num_masks = (1 << n) - 1
    masks = torch.arange(1, num_masks + 1, dtype=torch.long, device=device)
    
    # Crea matrice binaria per tutte le maschere (num_masks x n)
    positions = torch.arange(n, device=device).unsqueeze(0)
    mask_bits = ((masks.unsqueeze(1) >> positions) & 1).to(torch.bool)
    
    # Calcola somme per ogni maschera in parallelo
    sums = (mask_bits * ranks.unsqueeze(0)).sum(dim=1)
    
    # Trova maschere valide
    valid_masks = masks[sums == target_rank]
    
    # Converti maschere valide in subset (solo qui torniamo a Python)
    subsets = []
    for mask in valid_masks.cpu().tolist():
        subset = [table_cards[i] for i in range(n) if (mask >> i) & 1]
        subsets.append(subset)
    
    return subsets

def _find_sum_subsets_fast_cpu(table_cards, target_rank):
    """CPU fallback for compatibility"""
    n = len(table_cards)
    subsets = []
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if (mask >> i) & 1:
                s += table_cards[i][0] if isinstance(table_cards[i], tuple) else (table_cards[i] // 4 + 1)
        if s == target_rank:
            subset = [table_cards[i] for i in range(n) if (mask >> i) & 1]
            subsets.append(subset)
    return subsets

# Usa versione GPU di default
_find_sum_subsets_fast = _find_sum_subsets_fast_gpu
device = torch.device("cuda")

def encode_action(card, cards_to_capture):
    """
    Codifica un'azione utilizzando la rappresentazione a matrice:
    - Carta giocata: matrice 10x4 con 1 solo bit attivo (40 dim)
    - Carte catturate: matrice 10x4 con i bit attivi corrispondenti (40 dim)
    
    Totale: 80 dimensioni
    """
    # Inizializza le matrici su GPU
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
    # Ensure tensors are on CUDA and correct dtypes/shapes
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

# (Removed numba path; tensor-only paths used elsewhere)

_COL_TO_SUIT = {0: 'denari', 1: 'coppe', 2: 'spade', 3: 'bastoni'}

def _idx_to_id(row: int, col: int) -> int:
    return row * 4 + col

def _decode_ids(vec_t: torch.Tensor):
    if vec_t.dim() != 1 or vec_t.numel() != 80:
        raise ValueError("decode_action_ids richiede un vettore 80-dim")
    # GPU-only: ritorna tensori CUDA (long)
    played_idx = torch.argmax(vec_t[:40]).to(dtype=torch.long)
    captured_mask = (vec_t[40:] > 0)
    captured_ids = torch.nonzero(captured_mask, as_tuple=False).flatten().to(dtype=torch.long)
    return played_idx, captured_ids

def decode_action_ids(action_vec):
    """
    Decodifica un vettore azione 80-dim e ritorna (played_idx_t, captured_ids_t) su CUDA (long).
    """
    if torch.is_tensor(action_vec):
        vec_t = action_vec.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        vec_t = torch.as_tensor(action_vec, dtype=torch.float32, device=device).reshape(-1)
    return _decode_ids(vec_t)

def _decode_ids_torch(vec_t: torch.Tensor):
    if vec_t.dim() != 1 or vec_t.numel() != 80:
        raise ValueError("decode_action_ids_torch richiede un vettore 80-dim")
    played_idx = torch.argmax(vec_t[:40]).to(dtype=torch.long)
    captured_ids = torch.nonzero(vec_t[40:] > 0, as_tuple=False).flatten().to(dtype=torch.long)
    # map 0..39 for captured since indices 40..79 map directly 0..39
    return played_idx, captured_ids

def decode_action_ids_torch(action_vec: torch.Tensor):
    """
    GPU-native: ritorna (played_id_t, captured_ids_t) come tensori CUDA (long).
    Non effettua conversioni a Python e non sincronizza la GPU.
    """
    if torch.is_tensor(action_vec):
        vec_t = action_vec.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        vec_t = torch.as_tensor(action_vec, dtype=torch.float32, device=device).reshape(-1)
    return _decode_ids_torch(vec_t)

def get_valid_actions(game_state, current_player):
    """Azioni valide compute interamente su CUDA. Restituisce un tensore (K,80) su CUDA."""
    cp = current_player
    hand = game_state["hands"][cp]
    table = game_state["table"]
    actions = []
    if not hand:
        return torch.zeros((0, 80), dtype=torch.float32, device=device)
    if not isinstance(hand[0], int) or (len(table) > 0 and not isinstance(table[0], int)):
        raise TypeError("get_valid_actions richiede game_state in ID (int)")
    from observation import RANK_OF_ID as _RANK_OF_ID
    hand_ids_t = torch.as_tensor(hand, dtype=torch.long, device=device)
    table_ids_t = torch.as_tensor(table or [], dtype=torch.long, device=device)
    for pid_t in hand_ids_t:
        prank_t = _RANK_OF_ID[pid_t].to(torch.long)
        if table_ids_t.numel() > 0:
            table_ranks = _RANK_OF_ID[table_ids_t].to(torch.long)
            direct_ids_t = table_ids_t[table_ranks == prank_t]
        else:
            direct_ids_t = torch.empty(0, dtype=torch.long, device=device)
        if direct_ids_t.numel() > 0:
            for did_t in direct_ids_t:
                actions.append(encode_action_from_ids_gpu(pid_t, did_t.view(1)))
        else:
            n = int(table_ids_t.numel())
            if n > 0:
                pos = torch.arange(n, device=device, dtype=torch.long)
                masks = torch.arange(1, 1 << n, device=device, dtype=torch.long)
                sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.long)
                sums = (sel * (table_ranks.unsqueeze(0))).sum(dim=1)
                good = (sums == prank_t).nonzero(as_tuple=False).flatten()
                if good.numel() > 0:
                    for gi in good:
                        subset_ids = table_ids_t[((masks[gi].unsqueeze(0) >> pos) & 1).bool()]
                        actions.append(encode_action_from_ids_gpu(pid_t, subset_ids))
                else:
                    actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))
            else:
                actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))
    if actions:
        return torch.stack(actions, dim=0)
    return torch.zeros((0, 80), dtype=torch.float32, device=device)