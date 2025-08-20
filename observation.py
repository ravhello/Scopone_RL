# observation.py - Versione Torch CUDA (no NumPy)
import itertools
from functools import lru_cache
import os
import torch

SUITS = ['denari', 'coppe', 'spade', 'bastoni']
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Valori per la primiera
PRIMIERA_VAL = {1:16, 2:12, 3:13, 4:14, 5:15, 6:18, 7:21, 8:10, 9:10, 10:10}

# Mappa condivisa per conversione suit → index
suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}

# ===== Device control (CPU default) =====
OBS_DEVICE = torch.device(os.environ.get('OBS_DEVICE', 'cpu'))
OBS_INCLUDE_INFERRED = os.environ.get('OBS_INCLUDE_INFERRED', '1') == '1'
OBS_INCLUDE_RANK_PROBS = os.environ.get('OBS_INCLUDE_RANK_PROBS', '1') == '1'
OBS_INCLUDE_SCOPA_PROBS = os.environ.get('OBS_INCLUDE_SCOPA_PROBS', '1') == '1'

# ===== ID/Bitset helpers (device = OBS_DEVICE) =====
RANK_OF_ID = torch.tensor([i // 4 + 1 for i in range(40)], dtype=torch.int16, device=OBS_DEVICE)
SUITCOL_OF_ID = torch.tensor([i % 4 for i in range(40)], dtype=torch.int16, device=OBS_DEVICE)
MASK_RANK = [(sum(1 << j for j in range(40) if (j // 4 + 1) == r)) for r in range(1, 11)]
PRIMIERA_VAL_T = torch.tensor([0, 16, 12, 13, 14, 15, 18, 21, 10, 10, 10], dtype=torch.float32, device=OBS_DEVICE)
IDS_CUDA = torch.arange(40, device=OBS_DEVICE, dtype=torch.int64)
IS_DENARI_MASK_40 = (SUITCOL_OF_ID.to(torch.long) == 0)
PRIMIERA_PER_ID = PRIMIERA_VAL_T[RANK_OF_ID.to(torch.long)]  # (40,)

def bitset_popcount(x: int) -> int:
    return int(x.bit_count()) if hasattr(int, 'bit_count') else bin(x).count('1')

def bitset_rank_counts(bits: int) -> torch.Tensor:
    # Rank counts via bit manip on OBS_DEVICE
    ids = torch.arange(40, device=OBS_DEVICE, dtype=torch.int64)
    bits_t = torch.tensor(int(bits), dtype=torch.int64, device=OBS_DEVICE)
    active = ((bits_t >> ids) & 1).to(torch.float32)
    ranks = RANK_OF_ID.to(torch.int64) - 1
    counts = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
    counts.index_add_(0, ranks, active)
    return counts.to(torch.int32)

def bitset_table_sum(bits: int) -> int:
    # Sum of ranks via vectorization on OBS_DEVICE
    ids = torch.arange(40, device=OBS_DEVICE, dtype=torch.int64)
    bits_t = torch.tensor(int(bits), dtype=torch.int64, device=OBS_DEVICE)
    active = ((bits_t >> ids) & 1).to(torch.float32)
    ranks = RANK_OF_ID.to(torch.float32)
    return int(torch.sum(active * ranks).item())

# ----- OTTIMIZZAZIONE: CACHE PER FUNZIONI COSTOSE -----
# Cache per risultati costosi
cards_matrix_cache = {}

def encode_cards_as_matrix(cards):
    """Codifica un insieme di carte come vettore 40-d su CUDA usando scatter, ID-only."""
    if not cards:
        return torch.zeros(40, dtype=torch.float32, device=OBS_DEVICE)
    if not isinstance(cards[0], int):
        raise TypeError("encode_cards_as_matrix expects card IDs (int)")
    idx = torch.as_tensor(cards, dtype=torch.long, device=OBS_DEVICE).clamp_(0, 39)
    vec = torch.zeros(40, dtype=torch.float32, device=OBS_DEVICE)
    vec[idx] = 1.0
    return vec

def encode_hands(hands, player_id):
    """
    Codifica la mano del giocatore corrente come matrice.
    Versione ottimizzata con meno allocazioni ma risultati identici.
    """
    # Accetta sia liste di ID sia liste di tuple; usa bitset se ID
    ph = hands[player_id]
    if ph and not isinstance(ph[0], int):
        raise TypeError("encode_hands expects ID lists")
    player_hand = encode_cards_as_matrix(ph)
    
    other_counts = torch.zeros(3, dtype=torch.float32, device=OBS_DEVICE)
    count_idx = 0
    for p in range(4):
        if p != player_id:
            other_counts[count_idx] = float(len(hands[p]) / 10.0)
            count_idx += 1
    
    return torch.cat([player_hand.reshape(-1), other_counts.reshape(-1)], dim=0)  # 43 dim

def encode_table(table):
    """
    Codifica il tavolo come matrice rank x suit.
    Versione cachizzata con risultati identici.
    """
    if table and not isinstance(table[0], int):
        raise TypeError("encode_table expects ID list")
    return encode_cards_as_matrix(table)  # 40 dim

def encode_captured_squads(captured_squads):
    """
    Versione ottimizzata: vettori 40-d via bitset per ciascun team + conteggi.
    Output: 40 + 40 + 2 = 82 dim.
    """
    t0 = captured_squads[0]
    t1 = captured_squads[1]
    v0 = encode_cards_as_matrix(t0)
    v1 = encode_cards_as_matrix(t1)
    team0_count = torch.tensor([len(t0) / 40.0], dtype=torch.float32, device=OBS_DEVICE)
    team1_count = torch.tensor([len(t1) / 40.0], dtype=torch.float32, device=OBS_DEVICE)
    return torch.cat([v0.reshape(-1), v1.reshape(-1), team0_count, team1_count], dim=0)

# One-hot encoding per player (pre-calcolato per velocità)
ONE_HOT_PLAYERS = {
    0: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=OBS_DEVICE),
    1: torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=OBS_DEVICE),
    2: torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=OBS_DEVICE),
    3: torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=OBS_DEVICE)
}

def encode_current_player(cp):
    """
    Versione ottimizzata con risultati identici.
    Usa array pre-calcolati invece di crearli ogni volta.
    """
    return ONE_HOT_PLAYERS[cp].clone()  # Copia per evitare modifiche esterne

def encode_move(move):
    """
    Codifica una mossa. Versione ottimizzata con risultati identici.
    """
    player_vec = torch.zeros(4, dtype=torch.float32, device=OBS_DEVICE)
    player_vec[move["player"]] = 1.0
    
    # Carta giocata (14 dim)
    played = move["played_card"]
    # Supporta ID oppure tuple
    if isinstance(played, int):
        rank = int(RANK_OF_ID[played])
        suit_idx = int(SUITCOL_OF_ID[played])
    else:
        rank, suit = played
        suit_idx = suit_to_col[suit]
    
    rank_vec = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
    rank_vec[rank-1] = 1.0
    
    suit_vec = torch.zeros(4, dtype=torch.float32, device=OBS_DEVICE)
    suit_vec[suit_idx] = 1.0
    
    capture_vec = torch.zeros(3, dtype=torch.float32, device=OBS_DEVICE)
    capture_map = {"no_capture": 0, "capture": 1, "scopa": 2}
    ctype_idx = capture_map.get(move["capture_type"], 0)
    capture_vec[ctype_idx] = 1.0
    
    captured_cards = move["captured_cards"]
    captured_vec = encode_cards_as_matrix(captured_cards)
    
    return torch.cat([player_vec, rank_vec, suit_vec, capture_vec, captured_vec.reshape(-1)], dim=0)  # 61 dim

# [LEGACY REMOVED] La codifica della storia completa (40 turni, 10320 dim) è stata rimossa.

# Cache condivisa per matrice di carte mancanti
missing_cards_cache = {}

def compute_missing_cards_matrix(game_state, player_id):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Costruisci chiave cache efficiente
    # Supporta ID o tuple
    player_hand = tuple(sorted(game_state["hands"][player_id]))
    table = tuple(sorted(game_state["table"]))
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards)
    
    # Controlla cache
    if cache_key in missing_cards_cache:
        return missing_cards_cache[cache_key].clone()
    
    # GPU mask-based computation
    vis = torch.zeros(40, dtype=torch.bool, device=OBS_DEVICE)
    for lst in (game_state["hands"][player_id], game_state["table"], game_state["captured_squads"][0], game_state["captured_squads"][1]):
        if lst:
            idx = torch.as_tensor(lst, dtype=torch.long, device=OBS_DEVICE)
            vis[idx] = True
    missing = (~vis).to(torch.float32)
    result = missing
    
    # Salva in cache e ritorna tensore
    missing_cards_cache[cache_key] = result.clone()
    
    # Gestisci dimensione cache
    if len(missing_cards_cache) > 100:
        import random
        keys_to_remove = random.sample(list(missing_cards_cache.keys()), 50)
        for k in keys_to_remove:
            del missing_cards_cache[k]
    
    return result

# Cache per probabilità inferite
inferred_probs_cache = {}

def compute_inferred_probabilities(game_state, player_id):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Costruisci chiave cache
    player_hand = tuple(sorted(game_state["hands"][player_id]))
    table = tuple(sorted(game_state["table"]))
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    
    # Includi anche le dimensioni delle mani degli altri giocatori
    other_hands_sizes = tuple(len(game_state["hands"].get(p, [])) for p in range(4) if p != player_id)
    
    # Considera anche una rappresentazione condensata della storia
    history_key = tuple((m["player"], m["played_card"]) for m in game_state["history"])
    
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards, 
                other_hands_sizes, hash(history_key))
    
    # Controlla cache
    if cache_key in inferred_probs_cache:
        return inferred_probs_cache[cache_key].clone()
    
    # GPU implementation (use bitset mirrors if available)
    hands_bits_t = game_state.get('_hands_bits_t', None)
    table_bits_t = game_state.get('_table_bits_t', None)
    captured_bits_t = game_state.get('_captured_bits_t', None)
    if hands_bits_t is not None and table_bits_t is not None and captured_bits_t is not None:
        visible_bits = hands_bits_t[player_id] | table_bits_t | captured_bits_t[0] | captured_bits_t[1]
        vis = (((visible_bits >> IDS_CUDA) & 1).to(torch.bool))
    else:
        vis = torch.zeros(40, dtype=torch.bool, device=OBS_DEVICE)
        for lst in (game_state["hands"][player_id], game_state["table"], game_state["captured_squads"][0], game_state["captured_squads"][1]):
            if lst:
                vis[torch.as_tensor(lst, dtype=torch.long, device=OBS_DEVICE)] = True
    invisible = ~vis
    total_unknown = int(invisible.sum().item())
    probs = []
    other_players = [p for p in range(4) if p != player_id]
    for p in other_players:
        hand_size = len(game_state["hands"].get(p, []))
        pm = torch.zeros((10, 4), dtype=torch.float32, device=OBS_DEVICE)
        if hand_size == 0 or total_unknown == 0:
            probs.append(pm.reshape(-1))
            continue
        played_mask = torch.zeros(40, dtype=torch.bool, device=OBS_DEVICE)
        for mv in game_state["history"]:
            if mv["player"] == p:
                pc = mv["played_card"]
                cid = int(pc if isinstance(pc, int) else (pc[0] - 1) * 4 + suit_to_col[pc[1]])
                if 0 <= cid < 40:
                    played_mask[cid] = True
        possible_mask = invisible & (~played_mask)
        idx = torch.nonzero(possible_mask, as_tuple=False).flatten()
        if idx.numel() > 0:
            rows = (RANK_OF_ID[idx].to(torch.long) - 1).clamp_(0, 9)
            cols = SUITCOL_OF_ID[idx].to(torch.long).clamp_(0, 3)
            pm[rows, cols] = float(hand_size / total_unknown)
        probs.append(pm.reshape(-1))
    result = torch.cat(probs)
    
    # Salva in cache
    inferred_probs_cache[cache_key] = result.clone()
    
    # Gestisci dimensione cache
    if len(inferred_probs_cache) > 100:
        import random
        keys_to_remove = random.sample(list(inferred_probs_cache.keys()), 50)
        for k in keys_to_remove:
            del inferred_probs_cache[k]
    
    return result

# Cache per primiera
primiera_cache = {}

def compute_primiera_status(game_state):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Chiave cache semplice ma efficace
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    cache_key = (team0_cards, team1_cards)
    
    # Controlla cache
    if cache_key in primiera_cache:
        return primiera_cache[cache_key].clone()
    
    # Calcolo GPU vectorized
    def _max_per_suit(ids_list):
        if not ids_list:
            return torch.zeros(4, dtype=torch.float32, device=OBS_DEVICE)
        # Accept tuple cards; convert to IDs if needed
        if isinstance(ids_list[0], int):
            ids = torch.as_tensor(ids_list, dtype=torch.long, device=OBS_DEVICE)
        else:
            suit_to_idx = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
            ids_py = [int((r - 1) * 4 + suit_to_idx[s]) for (r, s) in ids_list]
            ids = torch.as_tensor(ids_py, dtype=torch.long, device=OBS_DEVICE)
        ranks = RANK_OF_ID[ids].to(torch.long)
        suits = SUITCOL_OF_ID[ids].to(torch.long)
        vals = PRIMIERA_VAL_T[ranks]
        out = torch.zeros(4, dtype=torch.float32, device=OBS_DEVICE)
        # scatter_reduce for max if available
        try:
            out.scatter_reduce_(0, suits, vals, reduce='amax', include_self=True)
        except Exception:
            for s in range(4):
                mask = (suits == s)
                if mask.any():
                    out[s] = torch.max(vals[mask])
        return out / 21.0
    team0_primiera = _max_per_suit(game_state["captured_squads"][0])
    team1_primiera = _max_per_suit(game_state["captured_squads"][1])
    result = torch.cat([team0_primiera, team1_primiera]).to(dtype=torch.float32)
    
    # Salva in cache
    primiera_cache[cache_key] = result.clone()
    
    # Gestisci dimensione cache
    if len(primiera_cache) > 100:
        import random
        keys_to_remove = random.sample(list(primiera_cache.keys()), 50)
        for k in keys_to_remove:
            del primiera_cache[k]
    
    return result

# Cache per denari count
denari_cache = {}

def compute_denari_count(game_state):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Chiave cache
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    cache_key = (team0_cards, team1_cards)
    
    # Controlla cache
    if cache_key in denari_cache:
        return denari_cache[cache_key].clone()
    
    cs0 = torch.as_tensor(game_state["captured_squads"][0] or [], dtype=torch.long, device=OBS_DEVICE)
    cs1 = torch.as_tensor(game_state["captured_squads"][1] or [], dtype=torch.long, device=OBS_DEVICE)
    den0 = (cs0.numel() > 0) and (SUITCOL_OF_ID[cs0] == 0).sum().item() or 0
    den1 = (cs1.numel() > 0) and (SUITCOL_OF_ID[cs1] == 0).sum().item() or 0
    result = torch.tensor([den0 / 10.0, den1 / 10.0], dtype=torch.float32, device=OBS_DEVICE)
    
    # Salva in cache
    denari_cache[cache_key] = result.clone()
    
    # Gestisci dimensione cache
    if len(denari_cache) > 100:
        import random
        keys_to_remove = random.sample(list(denari_cache.keys()), 50)
        for k in keys_to_remove:
            del denari_cache[k]
    
    return result

# Cache per settebello
settebello_cache = {}

def compute_settebello_status(game_state):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Chiave cache
    # Lavora su ID o tuple con fast-path ID
    SETTEBELLO_ID = 24  # (7-1)*4 + denari(0)
    def _has_settebello(seq):
        return SETTEBELLO_ID in seq
    team0_has_settebello = _has_settebello(game_state["captured_squads"][0])
    team1_has_settebello = _has_settebello(game_state["captured_squads"][1])
    table_has_settebello = _has_settebello(game_state["table"])
    cache_key = (team0_has_settebello, team1_has_settebello, table_has_settebello)
    
    # Controlla cache
    if cache_key in settebello_cache:
        return settebello_cache[cache_key].clone()
    
    # Calcolo originale
    if team0_has_settebello:
        status = 1
    elif team1_has_settebello:
        status = 2
    elif table_has_settebello:
        status = 3
    else:
        status = 0
    
    # Normalizza
    result = torch.tensor([status / 3.0], dtype=torch.float32, device=OBS_DEVICE)
    
    # Salva in cache
    settebello_cache[cache_key] = result.clone()
    
    return result

# Cache per score estimate
score_cache = {}

def compute_current_score_estimate(game_state):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Chiave cache
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    # Considera le scope
    scope_history = tuple((m["capture_type"], m["player"]) for m in game_state["history"] 
                         if m["capture_type"] == "scopa")
    
    cache_key = (team0_cards, team1_cards, scope_history)
    
    # Controlla cache
    if cache_key in score_cache:
        return score_cache[cache_key].clone()
    
    # Calcolo originale
    # Conteggio scope
    scope0 = 0
    scope1 = 0
    for mv in game_state["history"]:
        if mv.get("capture_type") == "scopa":
            if mv.get("player") in [0, 2]:
                scope0 += 1
            else:
                scope1 += 1
    
    # Carte totali
    c0 = len(game_state["captured_squads"][0])
    c1 = len(game_state["captured_squads"][1])
    pt_c0, pt_c1 = (1, 0) if c0 > c1 else (0, 1) if c1 > c0 else (0, 0)
    
    # Denari (ID-safe)
    den0 = sum(1 for c in game_state["captured_squads"][0] if (int(SUITCOL_OF_ID[c]) == 0 if isinstance(c, int) else c[1] == 'denari'))
    den1 = sum(1 for c in game_state["captured_squads"][1] if (int(SUITCOL_OF_ID[c]) == 0 if isinstance(c, int) else c[1] == 'denari'))
    pt_d0, pt_d1 = (1, 0) if den0 > den1 else (0, 1) if den1 > den0 else (0, 0)
    
    # Settebello (ID-safe)
    sb0 = 1 if (24 in game_state["captured_squads"][0]) else 0
    sb1 = 1 if (24 in game_state["captured_squads"][1]) else 0
    
    # Primiera (calcolo semplificato)
    primiera_status = compute_primiera_status(game_state)
    team0_prim_sum = float(torch.sum(primiera_status[:4]).item() * 21.0)  # Denormalizza
    team1_prim_sum = float(torch.sum(primiera_status[4:]).item() * 21.0)  # Denormalizza
    pt_p0, pt_p1 = (1, 0) if team0_prim_sum > team1_prim_sum else (0, 1) if team1_prim_sum > team0_prim_sum else (0, 0)
    
    # Punteggio totale
    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scope0
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scope1
    
    # Normalizza
    result = torch.tensor([total0 / 12.0, total1 / 12.0], dtype=torch.float32, device=OBS_DEVICE)
    
    # Salva in cache
    score_cache[cache_key] = result.clone()
    
    # Gestisci dimensione cache
    if len(score_cache) > 100:
        import random
        keys_to_remove = random.sample(list(score_cache.keys()), 50)
        for k in keys_to_remove:
            del score_cache[k]
    
    return result

# Cache per table sum
table_sum_cache = {}

def compute_table_sum(game_state):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Chiave cache che funziona sia per ID che per tuple
    tbl = game_state["table"]
    table_key = tuple(sorted(tbl))
    if table_key in table_sum_cache:
        return table_sum_cache[table_key].clone()
    if len(tbl) > 0:
        if isinstance(tbl[0], int):
            ids = torch.as_tensor(tbl, dtype=torch.long, device=OBS_DEVICE)
        else:
            suit_to_idx = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
            ids_py = [int((r - 1) * 4 + suit_to_idx[s]) for (r, s) in tbl]
            ids = torch.as_tensor(ids_py, dtype=torch.long, device=OBS_DEVICE)
        table_sum = int(RANK_OF_ID[ids].to(torch.int64).sum().item())
    else:
        table_sum = 0
    result = torch.tensor([table_sum / 30.0], dtype=torch.float32, device=OBS_DEVICE)
    table_sum_cache[table_key] = result.clone()
    
    # Gestisci dimensione cache
    if len(table_sum_cache) > 100:
        import random
        keys_to_remove = random.sample(list(table_sum_cache.keys()), 50)
        for k in keys_to_remove:
            del table_sum_cache[k]
    
    return result

# Cache per scopa probabilities
scopa_probs_cache = {}

def compute_next_player_scopa_probabilities(game_state, player_id, rank_probabilities=None):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Chiave cache che cattura lo stato rilevante
    player_hand = tuple(sorted(game_state["hands"][player_id]))
    table = tuple(sorted(game_state["table"]))
    next_player = (player_id + 1) % 4
    next_hand_size = len(game_state["hands"].get(next_player, []))
    
    cache_key = (player_id, next_player, player_hand, table, next_hand_size)
    
    # Controlla cache
    if cache_key in scopa_probs_cache:
        return scopa_probs_cache[cache_key].clone()
    
    # CPU-based computation
    next_player_idx = [i for i, p in enumerate([p for p in range(4) if p != player_id]) if p == next_player][0]
    scopa_probs = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
    if rank_probabilities is None:
        rank_probabilities = compute_rank_probabilities_by_player(game_state, player_id)
    hand_size = len(game_state["hands"].get(next_player, []))
    if hand_size == 0:
        return scopa_probs
    table = game_state["table"]
    if table and not isinstance(table[0], int):
        suit_to_idx = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
        table = [int((r - 1) * 4 + suit_to_idx[s]) for (r, s) in table]
    table_ids = torch.as_tensor(table or [], dtype=torch.long, device=OBS_DEVICE)
    table_ranks = RANK_OF_ID[table_ids].to(torch.long) if table_ids.numel() > 0 else torch.empty(0, dtype=torch.long, device=OBS_DEVICE)
    # For each current_rank 1..10
    for current_rank in range(1, 11):
        # Direct capture removes all of that rank
        if table_ranks.numel() > 0:
            if (table_ranks == current_rank).any():
                remaining = table_ranks[table_ranks != current_rank]
            else:
                n = int(table_ranks.numel())
                if n > 0:
                    pos = torch.arange(n, device=OBS_DEVICE, dtype=torch.long)
                    masks = torch.arange(1, 1 << n, device=OBS_DEVICE, dtype=torch.long)
                    sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.long)
                    sums = (sel * table_ranks.unsqueeze(0)).sum(dim=1)
                    good = (sums == current_rank)
                    if bool(good.any().item()):
                        # if a subset sums, simulate remove that subset
                        gi = int(torch.nonzero(good, as_tuple=False)[0].item())
                        remaining = table_ranks[((masks[gi].unsqueeze(0) >> pos) & 1) == 0]
                    else:
                        remaining = torch.cat([table_ranks, torch.tensor([current_rank], device=OBS_DEVICE)])
                else:
                    remaining = table_ranks
        else:
            remaining = table_ranks
        if remaining.numel() == 0:
            p_has = (1.0 - rank_probabilities[next_player_idx, 0, :].sum()).clamp_(0.0, 1.0)
            scopa_probs[current_rank-1] = p_has
            continue
        total_sum = int(remaining.sum().item()) if remaining.numel() > 0 else 0
        for next_rank in range(1, 11):
            can_capture_all = False
            if remaining.numel() > 0:
                if bool((remaining == next_rank).all().item()):
                    can_capture_all = True
                elif total_sum == next_rank:
                    can_capture_all = True
            if can_capture_all:
                p_zero = rank_probabilities[next_player_idx, 0, next_rank-1]
                scopa_probs[current_rank-1] += (1.0 - p_zero)
        scopa_probs[current_rank-1] = min(1.0, float(scopa_probs[current_rank-1].item()))
    
    # Salva in cache
    scopa_probs_cache[cache_key] = scopa_probs.clone()
    
    # Gestisci dimensione cache
    if len(scopa_probs_cache) > 100:
        import random
        keys_to_remove = random.sample(list(scopa_probs_cache.keys()), 50)
        for k in keys_to_remove:
            del scopa_probs_cache[k]
    
    return scopa_probs

# Cache per rank probabilities
rank_prob_cache = {}

def compute_rank_probabilities_by_player(game_state, player_id):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Costruisci chiave cache
    player_hand = tuple(sorted(game_state["hands"][player_id]))
    table = tuple(sorted(game_state["table"]))
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    
    # Includi dimensioni mani degli altri giocatori
    other_hands = tuple((p, len(game_state["hands"].get(p, []))) 
                       for p in range(4) if p != player_id)
    
    # Rappresentazione compatta della history (ID-safe)
    def _rank_of_played(pc):
        try:
            # se ID
            if isinstance(pc, int):
                return (pc // 4) + 1
            # altrimenti tuple
            return pc[0]
        except Exception:
            return -1
    history_summary = tuple((m["player"], _rank_of_played(m.get("played_card"))) for m in game_state.get("history", []))
    
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards, 
                other_hands, hash(history_summary))
    
    # Controlla cache
    if cache_key in rank_prob_cache:
        return rank_prob_cache[cache_key].clone()
    
    # GPU-based computation
    all_probs = torch.zeros((3, 5, 10), dtype=torch.float32, device=OBS_DEVICE)
    other_players = [p for p in range(4) if p != player_id]
    # Visible rank counts (use bitsets if available)
    hands_bits_t = game_state.get('_hands_bits_t', None)
    table_bits_t = game_state.get('_table_bits_t', None)
    captured_bits_t = game_state.get('_captured_bits_t', None)
    visible_rank_counts = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
    if hands_bits_t is not None and table_bits_t is not None and captured_bits_t is not None:
        visible_bits = hands_bits_t[player_id] | table_bits_t | captured_bits_t[0] | captured_bits_t[1]
        mask = (((visible_bits >> IDS_CUDA) & 1).to(torch.bool))
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() > 0:
            ranks = (RANK_OF_ID[idx].to(torch.long) - 1).clamp_(0, 9)
            add = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
            add.index_add_(0, ranks, torch.ones_like(ranks, dtype=torch.float32))
            visible_rank_counts += add
    else:
        def _acc_counts(lst):
            if lst:
                ids = torch.as_tensor(lst, dtype=torch.long, device=OBS_DEVICE)
                ranks = (RANK_OF_ID[ids].to(torch.long) - 1)
                add = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
                add.index_add_(0, ranks, torch.ones_like(ranks, dtype=torch.float32))
                return add
            return torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
        visible_rank_counts += _acc_counts(game_state["table"]) + _acc_counts(game_state["hands"][player_id])
        for team_cards in game_state["captured_squads"].values():
            visible_rank_counts += _acc_counts(team_cards)
    total_invisible = 40 - int(visible_rank_counts.sum().item())
    for i, p in enumerate(other_players):
        played_rank_counts = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
        for move in game_state.get("history", []):
            if move.get("player") == p:
                pc = move.get("played_card")
                rank = (pc // 4) + 1 if isinstance(pc, int) else pc[0]
                if 1 <= rank <= 10:
                    played_rank_counts[rank-1] += 1
        hand_size = len(game_state["hands"].get(p, []))
        if hand_size == 0:
            all_probs[i, 0, :] = 1.0
            continue
        total_rank = 4
        for rank in range(1, 11):
            rank_idx = rank - 1
            invisible_rank = total_rank - int(visible_rank_counts[rank_idx].item())
            played_rank = int(played_rank_counts[rank_idx].item())
            remaining_rank = total_rank - played_rank
            possible_rank = min(max(remaining_rank, 0), max(invisible_rank, 0))
            if possible_rank < 0:
                all_probs[i, 0, rank_idx] = 1.0
                continue
            # k from 0..min(4, hand_size)
            k_max = min(possible_rank, hand_size, 4)
            k = torch.arange(k_max + 1, device=OBS_DEVICE, dtype=torch.float32)
            N = torch.tensor(float(total_invisible), device=OBS_DEVICE)
            K = torch.tensor(float(invisible_rank), device=OBS_DEVICE)
            n = torch.tensor(float(hand_size), device=OBS_DEVICE)
            # log comb using lgamma
            def log_comb(a, b):
                return torch.lgamma(a + 1.0) - torch.lgamma(b + 1.0) - torch.lgamma(a - b + 1.0)
            log_num = log_comb(K, k) + log_comb(N - K, n - k)
            log_den = log_comb(N, n)
            probs_k = torch.exp(log_num - log_den)
            all_probs[i, :k.numel(), rank_idx] = probs_k
    
    # Salva in cache
    rank_prob_cache[cache_key] = all_probs.clone()
    
    # Gestisci dimensione cache
    if len(rank_prob_cache) > 50:  # Cache più piccola perché tensori più grandi
        import random
        keys_to_remove = random.sample(list(rank_prob_cache.keys()), 25)
        for k in keys_to_remove:
            del rank_prob_cache[k]
    
    return all_probs

# Cache per encode_state_for_player
state_cache = {}

def encode_state_for_player(game_state, player_id):
    """
    Versione ottimizzata con caching ma risultati identici.
    """
    # Chiave cache ottimizzata - include solo elementi che influenzano l'output
    player_hand = tuple(sorted(game_state["hands"][player_id]))
    table = tuple(sorted(game_state["table"]))
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    
    # Includiamo la lunghezza della history. Nota: per coerenza dell'osservazione
    # usiamo direttamente player_id come current player, dato che questa funzione
    # viene chiamata per lo specifico giocatore osservatore e lo stato non mantiene
    # una chiave affidabile "current_player".
    history_len = len(game_state["history"])
    cp = player_id
    
    # Carte in mano degli altri giocatori (solo lunghezze)
    other_hands = tuple((p, len(game_state["hands"].get(p, []))) 
                      for p in range(4) if p != player_id)
    
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards, 
                history_len, cp, other_hands)
    
    # Controlla cache
    if cache_key in state_cache:
        return state_cache[cache_key].clone()
    
    # Calcolo originale se non in cache
    # 1) Mani
    hands_enc = encode_hands(game_state["hands"], player_id)  # 43 dim
    
    # 2) Tavolo
    table_enc = encode_table(game_state["table"])  # 40 dim
    
    # 3) Catture squadre
    captured_enc = encode_captured_squads(game_state["captured_squads"])  # 82 dim
    
    # 4) current_player
    cp_enc = encode_current_player(cp)  # 4 dim
    
    # 5) history (compatta opzionale)
    # Mantieni la codifica completa per compatibilità; in futuro si può sostituire con storia corta
    # La storia completa legacy è rimossa: manteniamo un vettore nullo fisso (10320 dim) per compatibilità dimensionale
    hist_enc = torch.zeros(10320, dtype=torch.float32, device=OBS_DEVICE)
    
    # --- NUOVE INFORMAZIONI ---
    
    # 6) Matrice di carte mancanti
    missing_cards = compute_missing_cards_matrix(game_state, player_id)  # 40 dim
    
    # 7) Probabilità inferite per gli altri giocatori (opzionale)
    if OBS_INCLUDE_INFERRED:
        inferred_probs = compute_inferred_probabilities(game_state, player_id)  # 120 dim
    else:
        inferred_probs = torch.zeros(120, dtype=torch.float32, device=OBS_DEVICE)
    
    # 8) Stato della primiera
    primiera_status = compute_primiera_status(game_state)  # 8 dim
    
    # 9) Conteggio dei denari
    denari_count = compute_denari_count(game_state)  # 2 dim
    
    # 10) Stato del settebello
    settebello_status = compute_settebello_status(game_state)  # 1 dim
    
    # 11) Stima del punteggio attuale
    score_estimate = compute_current_score_estimate(game_state)  # 2 dim
    
    # 12) Somma totale sul tavolo
    table_sum = compute_table_sum(game_state)  # 1 dim
    
    # 13) Probabilità di scopa per ogni rank (opzionale)
    if OBS_INCLUDE_SCOPA_PROBS:
        scopa_probs = compute_next_player_scopa_probabilities(game_state, player_id)  # 10 dim
    else:
        scopa_probs = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
    
    # 14) Probabilità di rank per giocatore (opzionale)
    if OBS_INCLUDE_RANK_PROBS:
        rank_probs_by_player = compute_rank_probabilities_by_player(game_state, player_id).flatten()  # 150 dim
    else:
        rank_probs_by_player = torch.zeros(150, dtype=torch.float32, device=OBS_DEVICE)
    
    # Concatena tutte le features (tutte torch su CUDA -> converti a numpy solo in uscita per compat)
    parts = [
        hands_enc.reshape(-1),
        table_enc.reshape(-1),
        captured_enc.reshape(-1),
        cp_enc.reshape(-1),
        (hist_enc if torch.is_tensor(hist_enc) else torch.tensor(hist_enc, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (missing_cards if torch.is_tensor(missing_cards) else torch.tensor(missing_cards, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (inferred_probs if torch.is_tensor(inferred_probs) else torch.tensor(inferred_probs, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (primiera_status if torch.is_tensor(primiera_status) else torch.tensor(primiera_status, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (denari_count if torch.is_tensor(denari_count) else torch.tensor(denari_count, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (settebello_status if torch.is_tensor(settebello_status) else torch.tensor(settebello_status, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (score_estimate if torch.is_tensor(score_estimate) else torch.tensor(score_estimate, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (table_sum if torch.is_tensor(table_sum) else torch.tensor(table_sum, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (scopa_probs if torch.is_tensor(scopa_probs) else torch.tensor(scopa_probs, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
        (rank_probs_by_player if torch.is_tensor(rank_probs_by_player) else torch.tensor(rank_probs_by_player, dtype=torch.float32, device=OBS_DEVICE)).reshape(-1),
    ]
    result = torch.cat(parts)
    
    # Salva in cache, se non troppo grande
    if len(state_cache) < 30:  # Cache molto piccola per evitare problemi di memoria
        state_cache[cache_key] = result.clone()
    
    return result

# Funzione per pulizia cache - utile se si vuole liberare memoria
def clear_all_caches():
    """Pulisce tutte le cache usate per ottimizzare le funzioni"""
    cards_matrix_cache.clear()
    missing_cards_cache.clear()
    inferred_probs_cache.clear()
    primiera_cache.clear()
    denari_cache.clear()
    settebello_cache.clear()
    score_cache.clear()
    table_sum_cache.clear()
    scopa_probs_cache.clear()
    rank_prob_cache.clear()
    state_cache.clear()

###############################################
# Nuova osservazione compatta con storia corta
###############################################

# Cache per storia compatta
compact_history_cache = {}

def encode_recent_history_k(game_state, k=12):
    """
    Restituisce una codifica compatta delle ultime k mosse.
    Usa encode_move (61 dim per mossa) e padding a destra con zeri.
    Output: 61*k vettore float32.
    """
    # Chiave cache: lunghezza history e k
    hlen = len(game_state.get("history", []))
    cache_key = (hlen, k)
    if cache_key in compact_history_cache:
        return compact_history_cache[cache_key].clone()

    moves = game_state.get("history", [])[-k:]
    parts = []
    for mv in moves:
        try:
            enc = encode_move(mv)
            parts.append(enc)
        except Exception:
            parts.append(torch.zeros(61, dtype=torch.float32, device=OBS_DEVICE))
    while len(parts) < k:
        parts.insert(0, torch.zeros(61, dtype=torch.float32, device=OBS_DEVICE))
    result = torch.cat(parts) if parts else torch.zeros(61*k, dtype=torch.float32, device=OBS_DEVICE)
    compact_history_cache[cache_key] = result.clone()
    # Limita dimensione cache
    if len(compact_history_cache) > 100:
        import random
        for ck in random.sample(list(compact_history_cache.keys()), 50):
            del compact_history_cache[ck]
    return result

# Cache per stato compatto
state_compact_cache = {}

def encode_state_compact_for_player(game_state, player_id, k_history=12):
    """
    Osservazione compatta con storia corta:
      - 1) Mani: 43
      - 2) Tavolo: 40
      - 3) Catture squadre: 82
      - 4) History compatta: 61*k
      - 5) Missing cards: 40
      - 6) Inferred probs: 120
      - 7) Primiera status: 8
      - 8) Denari count: 2
      - 9) Settebello: 1
      - 10) Score estimate: 2
      - 11) Table sum: 1
      - 12) Scopa probs next: 10
      - 13) Rank probs by player: 150
    Tipico totale con k=12: 1231 dim.
    """
    player_hand = tuple(sorted(game_state["hands"][player_id]))
    table = tuple(sorted(game_state["table"]))
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    history_len = len(game_state["history"])
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards, history_len, k_history)
    if cache_key in state_compact_cache:
        _cached = state_compact_cache[cache_key]
        return _cached.clone() if hasattr(_cached, 'clone') else _cached.copy()

    device = torch.device('cuda')
    hands_enc = encode_hands(game_state["hands"], player_id)  # 43 (torch)
    table_enc = encode_table(game_state["table"])  # 40 (torch)
    captured_enc = encode_captured_squads(game_state["captured_squads"])  # 82 (torch)
    hist_k_np = encode_recent_history_k(game_state, k=k_history)  # np (61*k)
    missing_cards_np = compute_missing_cards_matrix(game_state, player_id)  # np (40)
    inferred_probs_np = compute_inferred_probabilities(game_state, player_id)  # np (120)
    primiera_status_np = compute_primiera_status(game_state)  # np (8)
    denari_count_np = compute_denari_count(game_state)  # np (2)
    settebello_status_np = compute_settebello_status(game_state)  # np (1)
    score_estimate_np = compute_current_score_estimate(game_state)  # np (2)
    table_sum_np = compute_table_sum(game_state)  # np (1)
    scopa_probs_np = compute_next_player_scopa_probabilities(game_state, player_id)  # np (10)
    rank_probs_by_player_np = compute_rank_probabilities_by_player(game_state, player_id).flatten()  # np (150)

    # Convert to torch CUDA tensors
    hist_k = (hist_k_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(hist_k_np) else torch.as_tensor(hist_k_np, dtype=torch.float32, device=device))
    missing_cards = (missing_cards_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(missing_cards_np) else torch.as_tensor(missing_cards_np, dtype=torch.float32, device=device))
    inferred_probs = (inferred_probs_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(inferred_probs_np) else torch.as_tensor(inferred_probs_np, dtype=torch.float32, device=device))
    primiera_status = (primiera_status_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(primiera_status_np) else torch.as_tensor(primiera_status_np, dtype=torch.float32, device=device))
    denari_count = (denari_count_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(denari_count_np) else torch.as_tensor(denari_count_np, dtype=torch.float32, device=device))
    settebello_status = (settebello_status_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(settebello_status_np) else torch.as_tensor(settebello_status_np, dtype=torch.float32, device=device))
    score_estimate = (score_estimate_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(score_estimate_np) else torch.as_tensor(score_estimate_np, dtype=torch.float32, device=device))
    table_sum = (table_sum_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(table_sum_np) else torch.as_tensor(table_sum_np, dtype=torch.float32, device=device))
    scopa_probs = (scopa_probs_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(scopa_probs_np) else torch.as_tensor(scopa_probs_np, dtype=torch.float32, device=device))
    rank_probs_by_player = (rank_probs_by_player_np.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(rank_probs_by_player_np) else torch.as_tensor(rank_probs_by_player_np, dtype=torch.float32, device=device))

    result = torch.cat([
        hands_enc,
        table_enc,
        captured_enc,
        hist_k,
        missing_cards,
        inferred_probs,
        primiera_status,
        denari_count,
        settebello_status,
        score_estimate,
        table_sum,
        scopa_probs,
        rank_probs_by_player
    ])

    # Cache limitata
    if len(state_compact_cache) < 50:
        state_compact_cache[cache_key] = result.clone()
    return result

def encode_state_compact_for_player_fast(game_state, player_id, k_history=12):
    """
    Variante interamente GPU-driven che usa i bitset mirror già residenti su CUDA
    (inseriti dall'ambiente come _hands_bits_t, _table_bits_t, _captured_bits_t)
    per minimizzare i trasferimenti H2D. Richiede che tali chiavi siano presenti.
    """
    # Use the same device of bitset mirrors (default CPU)
    hands_bits_t = game_state.get('_hands_bits_t', None)
    table_bits_t = game_state.get('_table_bits_t', None)
    captured_bits_t = game_state.get('_captured_bits_t', None)
    # Fallback handled below; pick device from any available tensor
    device = (hands_bits_t.device if torch.is_tensor(hands_bits_t) else
              table_bits_t.device if torch.is_tensor(table_bits_t) else
              captured_bits_t.device if torch.is_tensor(captured_bits_t) else
              OBS_DEVICE)
    if hands_bits_t is None or table_bits_t is None or captured_bits_t is None:
        # fallback
        return encode_state_compact_for_player(game_state, player_id, k_history=k_history)

    # 1) Mani (43): 40 one-hot + 3 conteggi altri giocatori
    hand_vec = (((hands_bits_t[player_id] >> IDS_CUDA) & 1).to(torch.float32))  # (40,)
    other_counts = []
    for p in range(4):
        if p == player_id:
            continue
        cnt = (((hands_bits_t[p] >> IDS_CUDA) & 1).to(torch.float32).sum() / 10.0)
        other_counts.append(cnt)
    other_counts_t = torch.stack(other_counts) if other_counts else torch.zeros(3, dtype=torch.float32, device=device)
    hands_enc = torch.cat([hand_vec.reshape(-1), other_counts_t.reshape(-1)], dim=0)

    # 2) Tavolo (40)
    table_enc = (((table_bits_t >> IDS_CUDA) & 1).to(torch.float32))

    # 3) Catture squadre (82): 40 + 40 + 2
    team0_vec = (((captured_bits_t[0] >> IDS_CUDA) & 1).to(torch.float32))
    team1_vec = (((captured_bits_t[1] >> IDS_CUDA) & 1).to(torch.float32))
    team0_count = torch.tensor([team0_vec.sum() / 40.0], dtype=torch.float32, device=device)
    team1_count = torch.tensor([team1_vec.sum() / 40.0], dtype=torch.float32, device=device)
    captured_enc = torch.cat([team0_vec, team1_vec, team0_count, team1_count], dim=0)

    # 4) History compatta (61*k)
    hist_k = encode_recent_history_k(game_state, k=k_history)

    # 5) Missing cards (40): inverti visibilità (mano osservatore + tavolo + captured)
    visible_bits = hands_bits_t[player_id] | table_bits_t | captured_bits_t[0] | captured_bits_t[1]
    missing_vec = (1 - (((visible_bits >> IDS_CUDA) & 1).to(torch.float32)))

    # 6) Inferred probs (120) - fallback GPU funzione esistente
    inferred_probs = compute_inferred_probabilities(game_state, player_id)

    # 7) Primiera status (8) via bitset
    def _primiera_from_bits(bits_t):
        present = (((bits_t >> IDS_CUDA) & 1).to(torch.bool))
        vals = PRIMIERA_PER_ID.to(torch.float32)
        prim = torch.zeros(4, dtype=torch.float32, device=device)
        for s in range(4):
            mask_s = (SUITCOL_OF_ID.to(torch.long) == s)
            v = vals.where(mask_s & present, torch.zeros((), device=device))
            prim[s] = v.max() if bool((mask_s & present).any().item()) else 0.0
        return prim / 21.0
    primiera_status = torch.cat([_primiera_from_bits(captured_bits_t[0]), _primiera_from_bits(captured_bits_t[1])])

    # 8) Denari count (2)
    den0 = (present := (((captured_bits_t[0] >> IDS_CUDA) & 1).to(torch.bool)))
    den1 = (((captured_bits_t[1] >> IDS_CUDA) & 1).to(torch.bool))
    denari_count = torch.stack([
        (den0 & IS_DENARI_MASK_40).to(torch.float32).sum() / 10.0,
        (den1 & IS_DENARI_MASK_40).to(torch.float32).sum() / 10.0,
    ])

    # 9) Settebello (1)
    settebello_id = 24
    settebello_status = torch.tensor([
        1.0 if bool(((captured_bits_t[0] >> settebello_id) & 1).item()) else
        (2.0 if bool(((captured_bits_t[1] >> settebello_id) & 1).item()) else
         (3.0 if bool(((table_bits_t >> settebello_id) & 1).item()) else 0.0))
    ], dtype=torch.float32, device=device) / 3.0

    # 10) Score estimate (2) - fallback
    score_estimate = compute_current_score_estimate(game_state)

    # 11) Table sum (1) via bitset
    table_sum = torch.tensor([
        (RANK_OF_ID.to(torch.int64)[IDS_CUDA].to(torch.float32) * (((table_bits_t >> IDS_CUDA) & 1).to(torch.float32))).sum() / 30.0
    ], dtype=torch.float32, device=device)

    # 12) Scopa probs next (10) - fallback
    scopa_probs = compute_next_player_scopa_probabilities(game_state, player_id)

    # 13) Rank probs by player (150) - fallback
    rank_probs_by_player = compute_rank_probabilities_by_player(game_state, player_id).flatten()

    return torch.cat([
        hands_enc,
        table_enc,
        captured_enc,
        hist_k,
        missing_vec,
        inferred_probs,
        primiera_status,
        denari_count,
        settebello_status,
        score_estimate,
        table_sum,
        scopa_probs,
        rank_probs_by_player
    ])