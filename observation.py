# observation.py - Versione Torch CUDA (no NumPy)
import os
import torch
import torch.nn.functional as F
import torch._dynamo as _dynamo  # type: ignore
_dynamo_disable = _dynamo.disable  # type: ignore[attr-defined]

SUITS = ['denari', 'coppe', 'spade', 'bastoni']
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Valori per la primiera
PRIMIERA_VAL = {1:16, 2:12, 3:13, 4:14, 5:15, 6:18, 7:21, 8:10, 9:10, 10:10}

# Mappa condivisa per conversione suit → index
suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}

# ===== Device control (CPU default) =====
import os as _os
OBS_DEVICE = torch.device(_os.environ.get('OBS_DEVICE', _os.environ.get('SCOPONE_DEVICE', 'cpu')))
# Disabilita di default le feature probabilistiche per lasciare che la rete impari il belief
OBS_INCLUDE_INFERRED = os.environ.get('OBS_INCLUDE_INFERRED', '0') == '1'
OBS_INCLUDE_RANK_PROBS = os.environ.get('OBS_INCLUDE_RANK_PROBS', '0') == '1'
OBS_INCLUDE_SCOPA_PROBS = os.environ.get('OBS_INCLUDE_SCOPA_PROBS', '0') == '1'
OBS_INCLUDE_DEALER = os.environ.get('OBS_INCLUDE_DEALER', '1') == '1'

# ===== ID/Bitset helpers (device = OBS_DEVICE) =====
RANK_OF_ID = torch.tensor([i // 4 + 1 for i in range(40)], dtype=torch.int16, device=OBS_DEVICE)
SUITCOL_OF_ID = torch.tensor([i % 4 for i in range(40)], dtype=torch.int16, device=OBS_DEVICE)
MASK_RANK = [(sum(1 << j for j in range(40) if (j // 4 + 1) == r)) for r in range(1, 11)]  # retained for potential bitset ops
PRIMIERA_VAL_T = torch.tensor([0, 16, 12, 13, 14, 15, 18, 21, 10, 10, 10], dtype=torch.float32, device=OBS_DEVICE)
IDS_CUDA = torch.arange(40, device=OBS_DEVICE, dtype=torch.int64)
IS_DENARI_MASK_40 = (SUITCOL_OF_ID.to(torch.long) == 0)
PRIMIERA_PER_ID = PRIMIERA_VAL_T[RANK_OF_ID.to(torch.long)]  # (40,)
# Precompute suits one-hot (4,40) for grouping by suit; rebuild on device change
SUITS_OH_4x40 = (torch.arange(4, device=OBS_DEVICE, dtype=torch.long).unsqueeze(1) == SUITCOL_OF_ID.to(torch.long).unsqueeze(0))

# ===== Runtime flags / small-const caches (CPU-friendly) =====
STRICT_CHECKS = (os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1')

_RANK_BINS_CACHE = {}
_IDX_1_10_CACHE = {}
_BIT_ONE_CACHE = {}
_MASK_BITS_CACHE = {}

def _get_rank_bins(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _RANK_BINS_CACHE.get(key)
    if t is None or t.device != device:
        t = torch.arange(1, 11, device=device, dtype=torch.int64).unsqueeze(1)  # (10,1)
        _RANK_BINS_CACHE[key] = t
    return t

def _get_idx_1_10(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _IDX_1_10_CACHE.get(key)
    if t is None or t.device != device:
        t = torch.arange(1, 11, device=device, dtype=torch.int64)
        _IDX_1_10_CACHE[key] = t
    return t

def _get_bit_one(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _BIT_ONE_CACHE.get(key)
    if t is None or t.device != device:
        t = torch.tensor(1, dtype=torch.int64, device=device)
        _BIT_ONE_CACHE[key] = t
    return t

def _get_mask_bits(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _MASK_BITS_CACHE.get(key)
    if t is None or t.device != device:
        t = torch.tensor((1 << 11) - 1, dtype=torch.int64, device=device)
        _MASK_BITS_CACHE[key] = t
    return t

def set_obs_device(device: torch.device) -> None:
    """Rebuild observation constant tensors on the requested device.
    Call this when the environment/device changes to avoid CPU/CUDA mismatches.
    """
    global OBS_DEVICE, RANK_OF_ID, SUITCOL_OF_ID, PRIMIERA_VAL_T, IDS_CUDA, IS_DENARI_MASK_40, PRIMIERA_PER_ID, ONE_HOT_PLAYERS, SUITS_OH_4x40
    if device == OBS_DEVICE:
        return
    OBS_DEVICE = torch.device(device)
    RANK_OF_ID = torch.tensor([i // 4 + 1 for i in range(40)], dtype=torch.int16, device=OBS_DEVICE)
    SUITCOL_OF_ID = torch.tensor([i % 4 for i in range(40)], dtype=torch.int16, device=OBS_DEVICE)
    PRIMIERA_VAL_T = torch.tensor([0, 16, 12, 13, 14, 15, 18, 21, 10, 10, 10], dtype=torch.float32, device=OBS_DEVICE)
    IDS_CUDA = torch.arange(40, device=OBS_DEVICE, dtype=torch.int64)
    IS_DENARI_MASK_40 = (SUITCOL_OF_ID.to(torch.long) == 0)
    PRIMIERA_PER_ID = PRIMIERA_VAL_T[RANK_OF_ID.to(torch.long)]
    SUITS_OH_4x40 = (torch.arange(4, device=OBS_DEVICE, dtype=torch.long).unsqueeze(1) == SUITCOL_OF_ID.to(torch.long).unsqueeze(0))
    ONE_HOT_PLAYERS = {
        0: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=OBS_DEVICE),
        1: torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=OBS_DEVICE),
        2: torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=OBS_DEVICE),
        3: torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=OBS_DEVICE),
    }

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
    return int((active * ranks).sum().detach().cpu().item())

# ----- OTTIMIZZAZIONE: CACHE PER FUNZIONI COSTOSE -----

def encode_cards_as_matrix(cards):
    """Codifica un insieme di carte come vettore 40-d su CUDA usando scatter, ID-only."""
    if not cards:
        return torch.zeros(40, dtype=torch.float32, device=OBS_DEVICE)
    if not isinstance(cards[0], int):
        raise TypeError("encode_cards_as_matrix expects card IDs (int)")
    idx = torch.as_tensor(cards, dtype=torch.long, device=OBS_DEVICE).clamp(0, 39)
    one_hot = F.one_hot(idx, num_classes=40).to(torch.float32)
    vec = one_hot.sum(dim=0)
    # Mantieni semantica originale: presenza (0/1), non conteggio
    vec = (vec > 0).to(torch.float32)
    return vec
 

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
    Codifica una mossa in un vettore 61-d totalmente tensoriale e compatibile con torch.compile.
    Layout: [player(4), rank(10), suit(4), capture_type(3), captured_cards(40)].
    """
    # Player one-hot (4)
    player_idx_py = int(move.get("player", 0))
    player_idx_py = 0 if player_idx_py < 0 else 3 if player_idx_py > 3 else player_idx_py
    # Use precomputed EYE_4 via ONE_HOT_PLAYERS to avoid one_hot kernel
    player_vec = ONE_HOT_PLAYERS[player_idx_py]

    # Carta giocata → rank (10) e suit (4)
    played = move.get("played_card", None)
    if isinstance(played, int):
        pid_t = torch.as_tensor(int(played), dtype=torch.long, device=OBS_DEVICE).clamp_(0, 39)
        rank_idx_t = (RANK_OF_ID[pid_t].to(torch.long) - 1).clamp_(0, 9)
        suit_idx_t = SUITCOL_OF_ID[pid_t].to(torch.long).clamp_(0, 3)
    else:
        if played is None:
            rank_idx_t = torch.zeros((), dtype=torch.long, device=OBS_DEVICE)
            suit_idx_t = torch.zeros((), dtype=torch.long, device=OBS_DEVICE)
        else:
            r_py, s_py = played
            r_py = int(r_py)
            rank_idx_t = torch.as_tensor(0 if r_py < 1 else 9 if r_py > 10 else (r_py - 1), dtype=torch.long, device=OBS_DEVICE)
            suit_idx_t = torch.as_tensor(int(suit_to_col.get(s_py, 0)), dtype=torch.long, device=OBS_DEVICE)
    # Replace one_hot with equality against small ranges to avoid launching kernels
    # Cached EYE tensors to avoid per-call arange
    global _EYE10, _EYE4, _EYE3
    if '_EYE10' not in globals() or _EYE10.device != OBS_DEVICE:
        _EYE10 = torch.eye(10, dtype=torch.float32, device=OBS_DEVICE)
        _EYE4 = torch.eye(4, dtype=torch.float32, device=OBS_DEVICE)
        _EYE3 = torch.eye(3, dtype=torch.float32, device=OBS_DEVICE)
    rank_vec = _EYE10[rank_idx_t.clamp_(0, 9)]
    suit_vec = _EYE4[suit_idx_t.clamp_(0, 3)]

    # Capture type (3)
    capture_map = {"no_capture": 0, "capture": 1, "scopa": 2}
    ctype_idx_py = int(capture_map.get(move.get("capture_type"), 0))
    ctype_idx_t = torch.as_tensor(ctype_idx_py, dtype=torch.long, device=OBS_DEVICE)
    capture_vec = _EYE3[ctype_idx_t.clamp_(0, 2)]

    # Carte catturate (40) – presenza 0/1
    captured_cards = move.get("captured_cards") or []
    if len(captured_cards) == 0:
        captured_vec = torch.zeros(40, dtype=torch.float32, device=OBS_DEVICE)
    else:
        if isinstance(captured_cards[0], int):
            idx = torch.as_tensor(captured_cards, dtype=torch.long, device=OBS_DEVICE)
        else:
            ids_py = [int((int(r) - 1) * 4 + suit_to_col[s]) for (r, s) in captured_cards]
            idx = torch.as_tensor(ids_py, dtype=torch.long, device=OBS_DEVICE)
        idx = idx.clamp(0, 39)
        one_hot = F.one_hot(idx, num_classes=40).to(torch.float32)
        captured_vec = (one_hot.sum(dim=0) > 0).to(torch.float32)

    return torch.cat([player_vec, rank_vec, suit_vec, capture_vec, captured_vec.reshape(-1)], dim=0)


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
    Versione mirror-only e compile-friendly: richiede i mirror tensoriali
    (_hands_bits_t, _table_bits_t, _captured_bits_t, _played_bits_by_player_t).
    Output: (3*40,) flatten per i tre avversari (10x4 per ciascuno).
    """
    hands_bits_t = game_state.get('_hands_bits_t', None)
    table_bits_t = game_state.get('_table_bits_t', None)
    captured_bits_t = game_state.get('_captured_bits_t', None)
    played_bits_by_player_t = game_state.get('_played_bits_by_player_t', None)
    torch._assert(torch.is_tensor(hands_bits_t), "_hands_bits_t required")
    torch._assert(torch.is_tensor(table_bits_t), "_table_bits_t required")
    torch._assert(torch.is_tensor(captured_bits_t), "_captured_bits_t required")
    torch._assert(torch.is_tensor(played_bits_by_player_t), "_played_bits_by_player_t required")

    visible_bits = hands_bits_t[player_id] | table_bits_t | captured_bits_t[0] | captured_bits_t[1]
    vis = (((visible_bits >> IDS_CUDA) & 1).to(torch.bool))
    invisible = ~vis
    total_unknown = invisible.to(torch.float32).sum().clamp(min=1.0)

    probs = []
    other_players = [p for p in range(4) if p != player_id]
    for p in other_players:
        hand_size = (((hands_bits_t[p] >> IDS_CUDA) & 1).to(torch.float32).sum())
        pm = torch.zeros((10, 4), dtype=torch.float32, device=OBS_DEVICE)
        played_mask = (((played_bits_by_player_t[p] >> IDS_CUDA) & 1).to(torch.bool))
        possible_mask = invisible & (~played_mask)
        idx = torch.nonzero(possible_mask, as_tuple=False).flatten()
        if idx.numel() > 0:
            rows = (RANK_OF_ID[idx].to(torch.long) - 1).clamp(0, 9)
            cols = SUITCOL_OF_ID[idx].to(torch.long).clamp(0, 3)
            pm[rows, cols] = (hand_size / total_unknown)
        probs.append(pm.reshape(-1))
    return torch.cat(probs)

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
        out.scatter_reduce_(0, suits, vals, reduce='amax', include_self=True)
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
    den0 = (cs0.numel() > 0) and int((SUITCOL_OF_ID[cs0] == 0).sum().detach().cpu().item()) or 0
    den1 = (cs1.numel() > 0) and int((SUITCOL_OF_ID[cs1] == 0).sum().detach().cpu().item()) or 0
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
    Versione interamente tensoriale (senza .item()/detach) per compatibilità con torch.compile.
    Restituisce un tensor [score_team0, score_team1] normalizzato in [0,1].
    """
    device = OBS_DEVICE

    # Sorgenti: richiede mirror bitset delle carte catturate
    cap_bits = game_state.get('_captured_bits_t', None)
    torch._assert(torch.is_tensor(cap_bits) and cap_bits.numel() >= 2, "compute_current_score_estimate requires _captured_bits_t")
    cap0_mask = (((cap_bits[0] >> IDS_CUDA) & 1).to(torch.bool))
    cap1_mask = (((cap_bits[1] >> IDS_CUDA) & 1).to(torch.bool))

    # 1) Carte totali (punto a chi ha più carte)
    c0 = cap0_mask.to(torch.float32).sum()
    c1 = cap1_mask.to(torch.float32).sum()
    pt_c0 = (c0 > c1).to(torch.float32)
    pt_c1 = (c1 > c0).to(torch.float32)

    # 2) Denari (punto a chi ne ha di più)
    den_mask = IS_DENARI_MASK_40.to(device=device)
    d0 = (cap0_mask & den_mask).to(torch.float32).sum()
    d1 = (cap1_mask & den_mask).to(torch.float32).sum()
    pt_d0 = (d0 > d1).to(torch.float32)
    pt_d1 = (d1 > d0).to(torch.float32)

    # 3) Settebello (ID 24)
    sette_id = 24
    sb0 = cap0_mask[sette_id].to(torch.float32)
    sb1 = cap1_mask[sette_id].to(torch.float32)

    # 4) Primiera (confronta la somma dei migliori per seme; lo scaling per 21 è costante e si annulla nel confronto)
    vals = PRIMIERA_PER_ID.to(device=device, dtype=torch.float32)  # (40)
    suits = SUITCOL_OF_ID.to(device=device, dtype=torch.long)      # (40)
    suits_oh = (suits.unsqueeze(0) == torch.arange(4, device=device, dtype=torch.long).unsqueeze(1))  # (4,40)
    def _primiera_sum(mask_bool):
        present = mask_bool.to(torch.bool)
        masked_vals = vals.unsqueeze(0) * (suits_oh & present.unsqueeze(0)).to(vals.dtype)
        best_per_suit = masked_vals.max(dim=1).values  # (4)
        return best_per_suit.sum()  # senza *21, confronto invariato
    p0 = _primiera_sum(cap0_mask)
    p1 = _primiera_sum(cap1_mask)
    pt_p0 = (p0 > p1).to(torch.float32)
    pt_p1 = (p1 > p0).to(torch.float32)

    # 5) Scopa counts: richiede mirror tensoriale
    scp = game_state.get('_scopa_counts_t', None)
    torch._assert(torch.is_tensor(scp) and scp.numel() >= 2, "compute_current_score_estimate requires _scopa_counts_t")
    scope0_t = scp[0]
    scope1_t = scp[1]

    # Totale e normalizzazione
    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scope0_t
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scope1_t
    result = torch.stack([total0, total1]).to(torch.float32) / 12.0
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
        table_sum = int(RANK_OF_ID[ids].to(torch.int64).sum().detach().cpu().item())
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

# Cache per somme possibili sul tavolo (subset-sum 1..10)
table_possible_sums_cache = {}

def compute_table_possible_sums(game_state):
    """
    Restituisce un vettore 10-d (rank 1..10) con 1.0 se esiste un sottoinsieme
    delle carte sul tavolo la cui somma dei rank è uguale a quel valore.
    Risultato normalizzato in [0,1] (booleana) e cache-ato per tavolo.
    """
    tbl = game_state["table"]
    table_key = tuple(sorted(tbl))
    cached = table_possible_sums_cache.get(table_key)
    if cached is not None:
        return cached.clone()
    if len(tbl) == 0:
        vec = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
        table_possible_sums_cache[table_key] = vec.clone()
        return vec
    # Converte in ID
    if isinstance(tbl[0], int):
        ids = torch.as_tensor(tbl, dtype=torch.long, device=OBS_DEVICE)
    else:
        suit_to_idx = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
        ids_py = [int((r - 1) * 4 + suit_to_idx[s]) for (r, s) in tbl]
        ids = torch.as_tensor(ids_py, dtype=torch.long, device=OBS_DEVICE)
    ranks = RANK_OF_ID[ids].to(torch.long)
    n = int(ranks.numel())
    possible = torch.zeros(10, dtype=torch.float32, device=OBS_DEVICE)
    if n == 0:
        table_possible_sums_cache[table_key] = possible.clone()
        return possible
    # Genera tutte le somme possibili via bitmask (n <= 10 tipicamente)
    pos = torch.arange(n, device=OBS_DEVICE, dtype=torch.long)
    masks = torch.arange(1, 1 << n, device=OBS_DEVICE, dtype=torch.long)
    sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.long)
    sums = (sel * ranks.unsqueeze(0)).sum(dim=1)
    # Booleana per tutti i rank 1..10 in un colpo solo
    ranks_10 = torch.arange(1, 11, device=OBS_DEVICE, dtype=sums.dtype)  # (10)
    # Confronto broadcasting: (M,1) vs (1,10) -> (M,10)
    any_per_rank = (sums.unsqueeze(1) == ranks_10.unsqueeze(0)).any(dim=0)
    possible = any_per_rank.to(torch.float32)
    table_possible_sums_cache[table_key] = possible.clone()
    # Limita cache
    if len(table_possible_sums_cache) > 100:
        import random
        for ck in random.sample(list(table_possible_sums_cache.keys()), 50):
            del table_possible_sums_cache[ck]
    return possible

def compute_scopa_counts(game_state):
    """
    Ritorna un vettore (2,) con il numero di scope per team [team0, team1],
    normalizzato dividendo per 10.0.
    """
    scp = game_state.get('_scopa_counts_t', None)
    torch._assert(torch.is_tensor(scp) and scp.numel() >= 2, "compute_scopa_counts requires _scopa_counts_t")
    return (scp / 10.0).to(torch.float32)

def compute_rank_presence_probs_from_inferred(game_state, player_id):
    """
    Condensa le inferred probs (3 x 40 carte per altri giocatori) in probabilità per-rank (1..10)
    per ciascuno dei 3 giocatori avversari, sommando le probabilità sulle 4 carte del rank.
    Output: (30,) flatten: [opp0(10), opp1(10), opp2(10)]. Se OBS_INCLUDE_INFERRED=0,
    restituisce zeri.
    """
    if not OBS_INCLUDE_INFERRED:
        return torch.zeros(30, dtype=torch.float32, device=OBS_DEVICE)
    probs_3x40 = compute_inferred_probabilities(game_state, player_id)
    if torch.is_tensor(probs_3x40):
        x = probs_3x40.view(3, 10, 4).sum(dim=2)  # (3,10)
        return x.reshape(-1)
    else:
        t = torch.as_tensor(probs_3x40, dtype=torch.float32, device=OBS_DEVICE).view(3, 10, 4)
        return t.sum(dim=2).reshape(-1)

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
                    if bool(good.any().detach().cpu().item()):
                        # if a subset sums, simulate remove that subset
                        gi = int(torch.nonzero(good, as_tuple=False)[0].detach().cpu().item())
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
        total_sum = int(remaining.sum().detach().cpu().item()) if remaining.numel() > 0 else 0
        for next_rank in range(1, 11):
            can_capture_all = False
            if remaining.numel() > 0:
                if bool((remaining == next_rank).all().detach().cpu().item()):
                    can_capture_all = True
                elif total_sum == next_rank:
                    can_capture_all = True
            if can_capture_all:
                p_zero = rank_probabilities[next_player_idx, 0, next_rank-1]
                scopa_probs[current_rank-1] += (1.0 - p_zero)
        scopa_probs[current_rank-1] = min(1.0, float(scopa_probs[current_rank-1].detach().cpu().item()))
    
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
        if isinstance(pc, int):
            return (pc // 4) + 1
        return pc[0]
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
    total_invisible = 40 - int(visible_rank_counts.sum().detach().cpu().item())
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
            invisible_rank = total_rank - int(visible_rank_counts[rank_idx].detach().cpu().item())
            played_rank = int(played_rank_counts[rank_idx].detach().cpu().item())
            remaining_rank = total_rank - played_rank
            possible_rank = min(max(remaining_rank, 0), max(invisible_rank, 0))
            if possible_rank < 0:
                all_probs[i, 0, rank_idx] = 1.0
                continue
            # k from 0..min(4, hand_size)
            k_max = min(possible_rank, hand_size, 4)
            k = torch.arange(k_max + 1, device=OBS_DEVICE, dtype=torch.float32)
            # Cast to float scalars on device
            N = torch.tensor(float(total_invisible), device=OBS_DEVICE)
            K = torch.tensor(float(invisible_rank), device=OBS_DEVICE)
            n = torch.tensor(float(hand_size), device=OBS_DEVICE)
            # Validity mask for hypergeometric terms: 0 <= k <= K and 0 <= n-k <= N-K and N,K,n >= 0
            valid = (k >= 0) & (k <= K) & ((n - k) >= 0) & ((n - k) <= (N - K)) & (N >= 0) & (K >= 0) & (n >= 0)
            probs_k = torch.zeros_like(k)
            if bool(valid.any().detach().cpu().item()):
                kv = k[valid]
                # log comb using lgamma only on valid entries
                def log_comb(a, b):
                    return torch.lgamma(a + 1.0) - torch.lgamma(b + 1.0) - torch.lgamma(a - b + 1.0)
                log_num = log_comb(K, kv) + log_comb(N - K, n - kv)
                log_den = log_comb(N, n)
                pv = torch.exp(log_num - log_den)
                probs_k[valid] = pv
            # Assign; remaining invalid stay 0
            all_probs[i, :k.numel(), rank_idx] = torch.nan_to_num(probs_k, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Salva in cache
    rank_prob_cache[cache_key] = all_probs.clone()
    
    # Gestisci dimensione cache
    if len(rank_prob_cache) > 50:  # Cache più piccola perché tensori più grandi
        import random
        keys_to_remove = random.sample(list(rank_prob_cache.keys()), 25)
        for k in keys_to_remove:
            del rank_prob_cache[k]
    
    return all_probs



# Funzione per pulizia cache - utile se si vuole liberare memoria
def clear_all_caches():
    """Pulisce tutte le cache usate per ottimizzare le funzioni"""
    missing_cards_cache.clear()
    inferred_probs_cache.clear()
    primiera_cache.clear()
    denari_cache.clear()
    settebello_cache.clear()
    score_cache.clear()
    table_sum_cache.clear()
    scopa_probs_cache.clear()
    rank_prob_cache.clear()
 

def encode_recent_history_k(game_state, k=12):
    """
    Restituisce una codifica compatta delle ultime k mosse (61*k) leggendo direttamente
    dal ring buffer senza copie intermedie e senza torch.roll, usando head/len.
    """
    hb = game_state.get('_hist_buf_t', None)
    hlen_t = game_state.get('_history_len_t', None)
    head_t = game_state.get('_hist_head_t', None)
    device = hb.device
    k_int = int(k)
    T = int(hb.size(0))
    n = int(torch.clamp(hlen_t.to(torch.long), min=0, max=T).item())
    if n <= 0:
        return torch.zeros(61 * k_int, dtype=hb.dtype, device=device)
    head = int(head_t.item()) if torch.is_tensor(head_t) else n % T
    take = min(k_int, n)
    # calcola start index circolare degli ultimi `take`
    start = (head - take) % T
    if start + take <= T:
        seg = hb[start:start+take]
    else:
        first = hb[start:]
        second = hb[:(start + take) % T]
        seg = torch.cat([first, second], dim=0)
    if take < k_int:
        pad = torch.zeros((k_int - take, 61), dtype=hb.dtype, device=device)
        seg = torch.cat([pad, seg], dim=0)
    return seg.reshape(-1)
 

def encode_state_compact_for_player_fast(game_state, player_id, k_history=12, out: torch.Tensor = None):
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
        raise ValueError('encode_state_compact_for_player_fast requires bitset mirrors (_hands_bits_t, _table_bits_t, _captured_bits_t)')

    # 1) Mani (43): 40 one-hot + 3 conteggi altri giocatori
    hand_vec = (((hands_bits_t[player_id] >> IDS_CUDA) & 1).to(torch.float32))  # (40,)
    if STRICT_CHECKS and (not torch.isfinite(hand_vec).all()):
        raise RuntimeError("Observation hand_vec contains non-finite values")
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
    if STRICT_CHECKS and (not torch.isfinite(table_enc).all()):
        raise RuntimeError("Observation table_enc contains non-finite values")

    # 3) Catture squadre (82): 40 + 40 + 2
    team0_vec = (((captured_bits_t[0] >> IDS_CUDA) & 1).to(torch.float32))
    team1_vec = (((captured_bits_t[1] >> IDS_CUDA) & 1).to(torch.float32))
    # Avoid tiny tensor constructors; compute then unsqueeze
    team0_count = (team0_vec.sum() / 40.0).unsqueeze(0)
    team1_count = (team1_vec.sum() / 40.0).unsqueeze(0)
    captured_enc = torch.cat([team0_vec, team1_vec, team0_count, team1_count], dim=0)
    if STRICT_CHECKS and (not torch.isfinite(captured_enc).all()):
        raise RuntimeError("Observation captured_enc contains non-finite values")

    # 4) History compatta (61*k)
    hist_k = encode_recent_history_k(game_state, k=k_history)
    if STRICT_CHECKS and (not torch.isfinite(hist_k).all()):
        raise RuntimeError("Observation history encoding contains non-finite values")

    # 5) Missing cards (40): inverti visibilità (mano osservatore + tavolo + captured)
    visible_bits = hands_bits_t[player_id] | table_bits_t | captured_bits_t[0] | captured_bits_t[1]
    missing_vec = (1 - (((visible_bits >> IDS_CUDA) & 1).to(torch.float32)))
    if STRICT_CHECKS and (not torch.isfinite(missing_vec).all()):
        raise RuntimeError("Observation missing_vec contains non-finite values")

    # 6) Inferred probs (120) - opzionale
    inferred_probs = (compute_inferred_probabilities(game_state, player_id)
                      if OBS_INCLUDE_INFERRED else None)

    # 7) Primiera status (8) via bitset (evita .item()/.any() sync)
    def _primiera_from_bits(bits_t):
        present = (((bits_t >> IDS_CUDA) & 1).to(torch.bool))           # (40)
        vals = PRIMIERA_PER_ID.to(device=device, dtype=torch.float32)   # (40)
        suits_oh = SUITS_OH_4x40.to(device=device)
        mask = present.unsqueeze(0) & suits_oh                          # (4,40)
        masked_vals = vals.unsqueeze(0) * mask.to(vals.dtype)
        prim = masked_vals.max(dim=1).values / 21.0
        return prim
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
    have0 = (((captured_bits_t[0] >> settebello_id) & 1).to(torch.float32))
    have1 = (((captured_bits_t[1] >> settebello_id) & 1).to(torch.float32))
    on_tbl = (((table_bits_t >> settebello_id) & 1).to(torch.float32))
    settebello_status = (have0 * 1.0 + have1 * 2.0 + on_tbl * 3.0).unsqueeze(0) / 3.0

    # 10) Score estimate (2) - tensor path using mirrors to stay compilable
    scopa_counts = game_state.get('_scopa_counts_t', None)
    if scopa_counts is None:
        scopa_counts = torch.zeros(2, dtype=torch.float32, device=device)
    # total cards winner
    c0 = (team0_vec > 0.5).to(torch.float32).sum()
    c1 = (team1_vec > 0.5).to(torch.float32).sum()
    pt_c0 = (c0 > c1).to(torch.float32)
    pt_c1 = (c1 > c0).to(torch.float32)
    # denari winner
    d0 = (den0 & IS_DENARI_MASK_40).to(torch.float32).sum()
    d1 = (den1 & IS_DENARI_MASK_40).to(torch.float32).sum()
    pt_d0 = (d0 > d1).to(torch.float32)
    pt_d1 = (d1 > d0).to(torch.float32)
    # settebello
    sb0 = have0
    sb1 = have1
    # primiera winner using per-suit maxima sums
    prim0 = primiera_status[:4].sum()
    prim1 = primiera_status[4:].sum()
    pt_p0 = (prim0 > prim1).to(torch.float32)
    pt_p1 = (prim1 > prim0).to(torch.float32)
    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scopa_counts[0]
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scopa_counts[1]
    score_estimate = torch.stack([total0, total1]).to(torch.float32) / 12.0

    # 11) Table sum (1) via bitset
    table_sum = (
        (RANK_OF_ID.to(torch.int64)[IDS_CUDA].to(torch.float32) * (((table_bits_t >> IDS_CUDA) & 1).to(torch.float32))).sum() / 30.0
    ).unsqueeze(0)

    # 12) Scopa probs next (10) - opzionale; prefer mirror to stay compilable
    if OBS_INCLUDE_SCOPA_PROBS:
        scopa_probs = game_state.get('_scopa_probs_t', torch.zeros(10, dtype=torch.float32, device=device))
    else:
        scopa_probs = torch.zeros(10, dtype=torch.float32, device=device)

    # 13) Rank probs by player (150) - opzionale; prefer mirror
    if OBS_INCLUDE_RANK_PROBS:
        rpb = game_state.get('_rank_probs_by_player_t', None)
        rank_probs_by_player = (rpb.flatten() if torch.is_tensor(rpb) else torch.zeros(150, dtype=torch.float32, device=device))
    else:
        rank_probs_by_player = torch.zeros(150, dtype=torch.float32, device=device)

    # 14) Scopa counts (2)
    scopa_counts = compute_scopa_counts(game_state)

    # 15) Table possible sums (10) — no Python branching on tensors
    present_tbl = (((table_bits_t >> IDS_CUDA) & 1).to(torch.bool))  # (40)
    ranks_all = RANK_OF_ID.to(torch.int64)[IDS_CUDA]  # (40)
    rank_bins = _get_rank_bins(device)  # (10,1)
    counts = ((ranks_all.unsqueeze(0) == rank_bins) & present_tbl.unsqueeze(0)).sum(dim=1).to(torch.int64)  # (10)
    bitset = _get_bit_one(device)  # bit 0 attivo (somma 0)
    mask_bits = _get_mask_bits(device)  # mantieni 11 bit (0..10)
    # Vectorize the four additions per rank using powers-of-two mask accumulation
    r_vals = torch.arange(1, 11, dtype=torch.int64, device=device)
    for ri in range(10):
        r = r_vals[ri]
        c = counts[ri]
        # Repeat-add r up to four times without per-iter where allocations
        # Each iteration: bitset = (bitset | (bitset << r)) & mask_bits if c > t
        if bool((c > 0).item()):
            bitset = torch.bitwise_and(bitset | torch.bitwise_left_shift(bitset, r), mask_bits)
        if bool((c > 1).item()):
            bitset = torch.bitwise_and(bitset | torch.bitwise_left_shift(bitset, r), mask_bits)
        if bool((c > 2).item()):
            bitset = torch.bitwise_and(bitset | torch.bitwise_left_shift(bitset, r), mask_bits)
        if bool((c > 3).item()):
            bitset = torch.bitwise_and(bitset | torch.bitwise_left_shift(bitset, r), mask_bits)
    # Estrai bit 1..10
    sums_bits = bitset
    idx = _get_idx_1_10(device)
    vals = (torch.bitwise_and(torch.bitwise_right_shift(sums_bits, idx), 1)).to(torch.float32)
    table_possible_sums = vals
    # 16) Progress (1) e 17) Last capturing team (2)
    p_m = game_state.get('_progress_t', None)
    lct_m = game_state.get('_last_capturing_team_t', None)
    progress = p_m.to(device=device, dtype=torch.float32)
    last_capturing_team = lct_m.to(device=device, dtype=torch.float32)

    # Prealloc result and write slices to reduce cat overhead
    include_scopa = OBS_INCLUDE_SCOPA_PROBS
    include_rank = OBS_INCLUDE_RANK_PROBS
    include_inferred = OBS_INCLUDE_INFERRED
    expected_dim = (43 + 40 + 82 + 61 * k_history + 40 + (120 if include_inferred else 0) + 8 + 2 + 1 + 2 + 1 + 10 + 2 + 30 + 3
                    + (10 if include_scopa else 0) + (150 if include_rank else 0) + (4 if OBS_INCLUDE_DEALER else 0))
    if (out is not None) and torch.is_tensor(out) and out.shape == (expected_dim,) and out.dtype == torch.float32 and out.device == device:
        result = out
    else:
        result = torch.empty((expected_dim,), dtype=torch.float32, device=device)
    pos = 0
    # Preallocate a small zero buffer for occasional zero fills to avoid creating many tiny tensors
    zero_buf = torch.zeros((1,), dtype=torch.float32, device=device)
    def _w(t):
        nonlocal pos
        # Assume all tensors are already float32 on correct device to avoid per-slice conversions
        n = int(t.numel())
        result[pos:pos+n] = t.reshape(-1)
        pos += n
    _w(hands_enc)
    _w(table_enc)
    _w(captured_enc)
    _w(hist_k)
    _w(missing_vec)
    if include_inferred and inferred_probs is not None:
        _w(inferred_probs if torch.is_tensor(inferred_probs) else torch.as_tensor(inferred_probs, dtype=torch.float32, device=device))
    _w(primiera_status)
    _w(denari_count)
    _w(settebello_status)
    _w(score_estimate)
    _w(table_sum)
    _w(table_possible_sums)
    _w(scopa_counts)
    if include_scopa:
        _w(scopa_probs)
    if include_rank:
        _w(rank_probs_by_player)
    _w(progress)
    _w(last_capturing_team)
    if OBS_INCLUDE_DEALER:
        # Compute dealer one-hot purely with tensor ops to stay compile-friendly
        curr = torch.as_tensor(int(game_state.get('current_player', -1)), dtype=torch.long, device=device)
        hlen_t = game_state.get('_history_len_t', None)
        hlen_mod4 = (hlen_t.to(torch.long) % 4) if torch.is_tensor(hlen_t) else torch.zeros((), dtype=torch.long, device=device)
        starting_seat = torch.remainder(curr - hlen_mod4, 4)
        dealer_idx = torch.remainder(starting_seat - 1, 4)
        dealer_vec = (torch.arange(4, device=device, dtype=torch.long) == dealer_idx).to(torch.float32)
        _w(dealer_vec)
    # rank_presence_from_inferred (30) always appended at end for fast-path coherence
    rpf = game_state.get('_rank_presence_from_inferred_t', None)
    _w(rpf)
    return result

