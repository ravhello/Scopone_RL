# actions.py
from collections import defaultdict
import torch

# Accel opzionale con numba per enumerazione subset-sum (DISABLED to keep torch-only path)
NUMBA_AVAILABLE = False
def njit(*args, **kwargs):
    def _decorator(fn):
        return fn
    return _decorator

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _subset_masks_with_sum(ranks, target):
        n = ranks.shape[0]
        results = []
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if (mask >> i) & 1:
                    s += ranks[i]
            if s == target:
                results.append(mask)
        return results

    def _find_sum_subsets_fast(table_cards, target_rank):
        # Pure python fallback always
        n = len(table_cards)
        masks = []
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if (mask >> i) & 1:
                    s += table_cards[i][0]
            if s == target_rank:
                masks.append(mask)
        subsets = []
        for mask in masks:
            subset = []
            for i in range(n):
                if (mask >> i) & 1:
                    subset.append(table_cards[i])
            subsets.append(subset)
        return subsets
else:
    def _find_sum_subsets_fast(table_cards, target_rank):
        # Fallback: brute via bitmask in puro Python (n di solito piccolo)
        n = len(table_cards)
        subsets = []
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if (mask >> i) & 1:
                    s += table_cards[i][0]
            if s == target_rank:
                subset = [table_cards[i] for i in range(n) if (mask >> i) & 1]
                subsets.append(subset)
        return subsets
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
    
    # Mappatura dei semi agli indici di colonna
    suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    
    # Codifica la carta giocata (accetta ID o tuple)
    if isinstance(card, int):
        cid = int(card)
        row = (cid // 4)
        col = (cid % 4)
    else:
        rank, suit = card
        row = rank - 1
        col = suit_to_col[suit]
    played_card_matrix[row, col] = 1.0
    
    # Codifica le carte catturate
    for capt_card in cards_to_capture:
        if isinstance(capt_card, int):
            cid = int(capt_card)
            capt_row = (cid // 4)
            capt_col = (cid % 4)
        else:
            capt_rank, capt_suit = capt_card
            capt_row = capt_rank - 1
            capt_col = suit_to_col[capt_suit]
        captured_cards_matrix[capt_row, capt_col] = 1.0
    
    # Appiattisci le matrici e concatenale
    return torch.cat([played_card_matrix.reshape(-1), captured_cards_matrix.reshape(-1)], dim=0)

def decode_action(action_vec):
    """
    Decodifica un vettore azione 80-dim in (played_id, [captured_ids]).
    """
    return decode_action_ids(action_vec)

# ----- FAST PATH: decode to card IDs (0..39) -----
try:
    from numba import njit
    NUMBA_JIT = True
except Exception:
    NUMBA_JIT = False

_COL_TO_SUIT = {0: 'denari', 1: 'coppe', 2: 'spade', 3: 'bastoni'}

def _idx_to_id(row: int, col: int) -> int:
    return row * 4 + col

def _decode_ids(vec_t: torch.Tensor):
    if vec_t.dim() != 1 or vec_t.numel() != 80:
        raise ValueError("decode_action_ids richiede un vettore 80-dim")
    played_idx = int(torch.argmax(vec_t[:40]).item())
    captured_ids = torch.nonzero(vec_t[40:] > 0, as_tuple=False).flatten()
    return played_idx, [int(i.item()) for i in captured_ids]

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
    # Usa puro Python per evitare CPU array ops; le liste restano leggere
    ids = [int(x) for x in table_ids]
    ranks = [(cid // 4) + 1 for cid in ids]
    if False:
        masks = _subset_masks_with_sum(torch.tensor(ranks, dtype=torch.int64), int(target_rank))
        n = len(ids)
        results = []
        for mask in masks:
            sub = []
            for i in range(n):
                if (mask >> i) & 1:
                    sub.append(int(ids[i]))
            results.append(sub)
        return results
    # Python path: bitmask
    n = len(ids)
    results = []
    for mask in range(1, 1 << n):
        s = 0
        sub = []
        for i in range(n):
            if (mask >> i) & 1:
                cid = ids[i]
                s += (cid // 4) + 1
                sub.append(cid)
        if s == target_rank:
            results.append(sub)
    return results

def get_valid_actions(game_state, current_player):
    """
    Restituisce una lista di azioni valide nel formato matrice per il giocatore corrente.
    
    Args:
        game_state: Stato del gioco
        current_player: ID del giocatore corrente
    
    Returns:
        Lista di azioni valide nel formato matrice (80 bit)
    """
    cp = current_player
    hand = game_state["hands"][cp]
    table = game_state["table"]

    valid_actions = []

    # Se lo stato è in ID (int), usa path ottimizzato con bitmask/numba
    if len(hand) > 0 and isinstance(hand[0], int):
        # Helpers: ID <-> tuple
        def _id_to_tuple(cid: int):
            rank = cid // 4 + 1
            suit = _COL_TO_SUIT[cid % 4]
            return (rank, suit)

        from actions import find_sum_subsets_ids

        table_ids = list(table)
        # Per ogni carta in mano
        for pid in hand:
            prank = pid // 4 + 1
            # pari-rank
            direct_ids = [tid for tid in table_ids if (tid // 4 + 1) == prank]
            if direct_ids:
                for did in direct_ids:
                    action_vec = encode_action(_id_to_tuple(pid), [_id_to_tuple(did)])
                    valid_actions.append(action_vec)
            else:
                subs = find_sum_subsets_ids(table_ids, prank)
                if subs:
                    for sub in subs:
                        action_vec = encode_action(_id_to_tuple(pid), [_id_to_tuple(x) for x in sub])
                        valid_actions.append(action_vec)
                else:
                    valid_actions.append(encode_action(_id_to_tuple(pid), []))
        return valid_actions

    # Altrimenti, percorso legacy su tuple con acceleratori locali
    def _find_sum_subsets(table_cards, target_rank):
        # Usa accelerazione numba/bitmask se possibile, altrimenti DP
        if len(table_cards) <= 15:
            return _find_sum_subsets_fast(table_cards, target_rank)
        dp = defaultdict(list)
        dp[0] = [tuple()]
        for card in table_cards:
            r, _ = card
            for s in sorted(list(dp.keys()), reverse=True):
                new_sum = s + r
                if new_sum > target_rank:
                    continue
                for subset in dp[s]:
                    dp[new_sum].append(subset + (card,))
        return [list(subset) for subset in dp.get(target_rank, [])]

    for card in hand:
        rank, suit = card
        same_rank_cards = [t_c for t_c in table if t_c[0] == rank]
        if same_rank_cards:
            for direct_card in same_rank_cards:
                valid_actions.append(encode_action(card, [direct_card]))
        else:
            sum_subsets = _find_sum_subsets(table, rank)
            if sum_subsets:
                for subset in sum_subsets:
                    valid_actions.append(encode_action(card, subset))
            else:
                valid_actions.append(encode_action(card, []))

    return valid_actions