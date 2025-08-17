# observation.py - Versione Torch CUDA (no NumPy)
import itertools
from functools import lru_cache
import torch

SUITS = ['denari', 'coppe', 'spade', 'bastoni']
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Valori per la primiera
PRIMIERA_VAL = {1:16, 2:12, 3:13, 4:14, 5:15, 6:18, 7:21, 8:10, 9:10, 10:10}

# Mappa condivisa per conversione suit → index
suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}

# ===== ID/Bitset helpers =====
RANK_OF_ID = torch.tensor([i // 4 + 1 for i in range(40)], dtype=torch.int16, device=torch.device('cuda'))
SUITCOL_OF_ID = torch.tensor([i % 4 for i in range(40)], dtype=torch.int16, device=torch.device('cuda'))
MASK_RANK = [(sum(1 << j for j in range(40) if (j // 4 + 1) == r)) for r in range(1, 11)]

def bitset_popcount(x: int) -> int:
    return int(x.bit_count()) if hasattr(int, 'bit_count') else bin(x).count('1')

def bitset_rank_counts(bits: int) -> torch.Tensor:
    counts = torch.zeros(10, dtype=torch.int32, device=torch.device('cuda'))
    for r in range(10):
        counts[r] = bitset_popcount(bits & MASK_RANK[r])
    return counts

def bitset_table_sum(bits: int) -> int:
    # sum of ranks on table
    return int(sum((r+1) * bitset_popcount(bits & MASK_RANK[r]) for r in range(10)))

# ----- OTTIMIZZAZIONE: CACHE PER FUNZIONI COSTOSE -----
# Cache per risultati costosi
cards_matrix_cache = {}
# Cache alternativa basata su bitset per vettori 40-d (più stabile e senza sort)
cards_matrix_cache_bits = {}

def _vector_from_bitset(bits: int) -> torch.Tensor:
    """Crea un vettore 40-d (float32, CUDA) dai bit attivi in bits (ID 0..39). Usa cache."""
    cached = cards_matrix_cache_bits.get(bits)
    if cached is not None:
        return cached
    vec = torch.zeros(40, dtype=torch.float32, device=torch.device('cuda'))
    mb = bits
    while mb:
        lsb = mb & -mb
        idx = (lsb.bit_length() - 1)
        vec[idx] = 1.0
        mb ^= lsb
    cards_matrix_cache_bits[bits] = vec
    # mantieni cache entro ~2000 entry
    if len(cards_matrix_cache_bits) > 2000:
        import random
        for k in random.sample(list(cards_matrix_cache_bits.keys()), 400):
            del cards_matrix_cache_bits[k]
    return vec

def encode_cards_as_matrix(cards):
    """
    Codifica un insieme di carte come vettore 40-d. Richiede ID (0..39).
    """
    if not cards:
        return torch.zeros(40, dtype=torch.float32, device=torch.device('cuda'))
    if not isinstance(cards[0], int):
        # Enforce ID-only usage
        raise TypeError("encode_cards_as_matrix expects card IDs (int)")
    bits = 0
    for cid in cards:
        bits |= (1 << int(cid))
    vec = _vector_from_bitset(bits)
    return vec

def encode_hands(hands, player_id):
    """
    Codifica la mano del giocatore corrente come matrice.
    Versione ottimizzata con meno allocazioni ma risultati identici.
    """
    # Accetta sia liste di ID sia liste di tuple; usa bitset se ID
    ph = hands[player_id]
    if ph and isinstance(ph[0], int):
        bits = 0
        for cid in ph:
            bits |= (1 << int(cid))
        player_hand = _vector_from_bitset(bits)
    else:
        player_hand = encode_cards_as_matrix(ph)
    
    other_counts = torch.zeros(3, dtype=torch.float32, device=torch.device('cuda'))
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
    if table and isinstance(table[0], int):
        bits = 0
        for cid in table:
            bits |= (1 << int(cid))
        return _vector_from_bitset(bits)
    return encode_cards_as_matrix(table)  # 40 dim

def encode_captured_squads(captured_squads):
    """
    Versione ottimizzata: vettori 40-d via bitset per ciascun team + conteggi.
    Output: 40 + 40 + 2 = 82 dim.
    """
    t0 = captured_squads[0]
    t1 = captured_squads[1]
    if t0 and isinstance(t0[0], int):
        bits0 = 0
        for c in t0:
            bits0 |= (1 << int(c))
        v0 = _vector_from_bitset(bits0)
    else:
        v0 = encode_cards_as_matrix(t0)
    if t1 and isinstance(t1[0], int):
        bits1 = 0
        for c in t1:
            bits1 |= (1 << int(c))
        v1 = _vector_from_bitset(bits1)
    else:
        v1 = encode_cards_as_matrix(t1)
    team0_count = torch.tensor([len(t0) / 40.0], dtype=torch.float32, device=torch.device('cuda'))
    team1_count = torch.tensor([len(t1) / 40.0], dtype=torch.float32, device=torch.device('cuda'))
    return torch.cat([v0.reshape(-1), v1.reshape(-1), team0_count, team1_count], dim=0)

# One-hot encoding per player (pre-calcolato per velocità)
ONE_HOT_PLAYERS = {
    0: torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda')),
    1: torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda')),
    2: torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=torch.device('cuda')),
    3: torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=torch.device('cuda'))
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
    player_vec = torch.zeros(4, dtype=torch.float32, device=torch.device('cuda'))
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
    
    rank_vec = torch.zeros(10, dtype=torch.float32, device=torch.device('cuda'))
    rank_vec[rank-1] = 1.0
    
    suit_vec = torch.zeros(4, dtype=torch.float32, device=torch.device('cuda'))
    suit_vec[suit_idx] = 1.0
    
    capture_vec = torch.zeros(3, dtype=torch.float32, device=torch.device('cuda'))
    capture_map = {"no_capture": 0, "capture": 1, "scopa": 2}
    ctype_idx = capture_map.get(move["capture_type"], 0)
    capture_vec[ctype_idx] = 1.0
    
    captured_cards = move["captured_cards"]
    captured_vec = encode_cards_as_matrix(captured_cards)
    
    return torch.cat([player_vec, rank_vec, suit_vec, capture_vec, captured_vec.reshape(-1)], dim=0)  # 61 dim

# Cache per history (chiave: lunghezza history, player_id, hash di hands e table)
history_cache = {}

def encode_enhanced_history(game_state, player_id):
    """
    Versione ottimizzata con cache ma risultati identici.
    """
    # Costruisci una chiave di cache efficiente ma unica
    # Includiamo solo gli elementi che influenzano l'output
    history_len = len(game_state["history"])
    player_hand_tuple = tuple(sorted(game_state["hands"].get(player_id, [])))
    table_tuple = tuple(sorted(game_state["table"]))
    
    # Rappresentazione compatta delle carte catturate (solo lunghezze)
    captured0_len = len(game_state["captured_squads"][0])
    captured1_len = len(game_state["captured_squads"][1])
    
    # Chiave di cache
    cache_key = (history_len, player_id, player_hand_tuple, table_tuple, 
                captured0_len, captured1_len)
    
    # Controlla cache
    if cache_key in history_cache:
        return history_cache[cache_key].clone()
    
    # Se non in cache, calcola normalmente seguendo l'algoritmo originale
    max_turns = 40
    turn_dim = 4 + 40 + 3 + 40 + 40 + 40 + 40 + 40 + 1 + 10  # 258 dim per turno 
    hist_arr = torch.zeros(max_turns * turn_dim, dtype=torch.float32, device=torch.device('cuda'))
    
    # Ricostruisci la storia esattamente come nell'originale
    team0_captured = []
    team1_captured = []
    table_after_moves = []
    current_team0 = []
    current_team1 = []
    current_table = []
    table_sums = []
    scopa_probs_history = []
    
    # Ricostruisci la storia delle mani del giocatore osservatore
    observer_hands = []
    current_observer_hand = list(game_state["hands"].get(player_id, []))
    
    # Set per tenere traccia delle carte giocate dal player_id
    played_cards_by_observer = set()
    
    for move in game_state["history"]:
        player = move["player"]
        played_card = move["played_card"]
        capture_type = move["capture_type"]
        captured_cards = move["captured_cards"]
        
        # Se è il giocatore osservatore che sta giocando, aggiungi al set
        if player == player_id:
            played_cards_by_observer.add(played_card)
        
        team_id = 0 if player in [0, 2] else 1
        
        if capture_type in ["capture", "scopa"]:
            # Rimuovi le carte catturate dal tavolo
            current_table = [card for card in current_table if card not in captured_cards]
            
            # Aggiungi le carte catturate alla squadra
            if team_id == 0:
                current_team0.extend(captured_cards)
                current_team0.append(played_card)
            else:
                current_team1.extend(captured_cards)
                current_team1.append(played_card)
        else:
            # Aggiungi la carta al tavolo
            current_table.append(played_card)
        
        # [NUOVO] Calcola la somma totale sul tavolo
        current_table_sum = sum(card[0] for card in current_table) / 30.0  # Normalizzato come in compute_table_sum
        table_sums.append(current_table_sum)
        
        # [NUOVO] Calcola le probabilità di scopa per ogni rank
        # Versione semplificata - se il tavolo è vuoto, nessuna probabilità di scopa
        current_scopa_probs = torch.zeros(10, dtype=torch.float32, device=torch.device('cuda'))
        if not current_table:
            pass  # Lascia tutto a zero
        else:
            # Per ogni rank, verifica se può catturare tutte le carte del tavolo
            for rank in range(1, 11):
                # Cattura diretta - se tutti hanno lo stesso rank
                if all(card[0] == rank for card in current_table):
                    current_scopa_probs[rank-1] = 1.0
                # Cattura per somma
                elif sum(card[0] for card in current_table) == rank:
                    current_scopa_probs[rank-1] = 1.0
        
        scopa_probs_history.append(current_scopa_probs)
        
        # Ricrea la mano dell'osservatore per questo turno
        reconstructed_hand = []
        
        # Se abbiamo informazioni sulla mano attuale del giocatore
        if player_id in game_state["hands"]:
            for card in game_state["hands"][player_id]:
                # Le carte attualmente in mano erano sicuramente in mano anche nei turni precedenti
                reconstructed_hand.append(card)
            
            # Aggiungiamo le carte che l'osservatore ha giocato fino a questo punto
            for card in played_cards_by_observer:
                if card not in reconstructed_hand:  # evita duplicati
                    reconstructed_hand.append(card)
        
        # Salva la mano ricostruita
        observer_hands.append(list(reconstructed_hand))
        
        # Salva lo stato dopo questa mossa
        team0_captured.append(list(current_team0))
        team1_captured.append(list(current_team1))
        table_after_moves.append(list(current_table))
    
    # Prendi fino a max_turns mosse dalla storia
    hist_slice = game_state["history"][-max_turns:]
    team0_slice = team0_captured[-max_turns:]
    team1_slice = team1_captured[-max_turns:]
    table_slice = table_after_moves[-max_turns:]
    observer_hands_slice = observer_hands[-max_turns:]
    table_sums_slice = table_sums[-max_turns:]
    scopa_probs_slice = scopa_probs_history[-max_turns:]
    
    for turn_idx, (move, team0_cards, team1_cards, table_cards, observer_hand, table_sum, scopa_prob) in enumerate(
            zip(hist_slice, team0_slice, team1_slice, table_slice, observer_hands_slice, 
                table_sums_slice, scopa_probs_slice)):
        player = move["player"]
        played_card = move["played_card"]
        capture_type = move["capture_type"]
        captured_cards = move["captured_cards"]
        
        turn_offset = turn_idx * turn_dim
        
        # Giocatore di turno (4 dim)
        player_vec = torch.zeros(4, dtype=torch.float32, device=torch.device('cuda'))
        player_vec[player] = 1.0
        hist_arr[turn_offset:turn_offset+4] = player_vec
        
        # Carta giocata (40 dim)
        played_card_vec = encode_cards_as_matrix([played_card])
        hist_arr[turn_offset+4:turn_offset+44] = played_card_vec
        
        # Tipo di cattura (3 dim)
        capture_vec = torch.zeros(3, dtype=torch.float32, device=torch.device('cuda'))
        capture_map = {"no_capture": 0, "capture": 1, "scopa": 2}
        ctype_idx = capture_map.get(capture_type, 0)
        capture_vec[ctype_idx] = 1.0
        hist_arr[turn_offset+44:turn_offset+47] = capture_vec
        
        # Carte catturate in questa mossa (40 dim)
        captured_vec = encode_cards_as_matrix(captured_cards)
        hist_arr[turn_offset+47:turn_offset+87] = captured_vec
        
        # Carte catturate dalla squadra 0 fino a questo punto (40 dim)
        team0_vec = encode_cards_as_matrix(team0_cards)
        hist_arr[turn_offset+87:turn_offset+127] = team0_vec
        
        # Carte catturate dalla squadra 1 fino a questo punto (40 dim)
        team1_vec = encode_cards_as_matrix(team1_cards)
        hist_arr[turn_offset+127:turn_offset+167] = team1_vec
        
        # Carte sul tavolo dopo la mossa (40 dim)
        table_vec = encode_cards_as_matrix(table_cards)
        hist_arr[turn_offset+167:turn_offset+207] = table_vec
        
        # Carte in mano all'osservatore in quel turno (40 dim)
        observer_hand_vec = encode_cards_as_matrix(observer_hand)
        hist_arr[turn_offset+207:turn_offset+247] = observer_hand_vec
        
        # [NUOVO] Somma totale sul tavolo dopo la mossa (1 dim)
        hist_arr[turn_offset+247] = table_sum
        
        # [NUOVO] Probabilità di scopa per ogni rank (10 dim)
        hist_arr[turn_offset+248:turn_offset+258] = scopa_prob
    
    # Salva risultato in cache
    history_cache[cache_key] = hist_arr.clone()
    
    # Gestisci dimensione cache
    if len(history_cache) > 100:
        # Rimuovi elementi vecchi
        import random
        keys_to_remove = random.sample(list(history_cache.keys()), 50)
        for k in keys_to_remove:
            del history_cache[k]
    
    return hist_arr  # 40 * 258 = 10320 dim

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
    
    # Calcolo con bitset se possibile
    h = game_state["hands"][player_id]
    t = game_state["table"]
    c0 = game_state["captured_squads"][0]
    c1 = game_state["captured_squads"][1]
    if (h and isinstance(h[0], int)) or (t and isinstance(t[0], int)):
        visible_bits = 0
        for lst in (h, t, c0, c1):
            for cid in lst:
                if isinstance(cid, int):
                    visible_bits |= (1 << int(cid))
        all_bits = (1 << 40) - 1
        missing_bits = all_bits & (~visible_bits)
        vec = torch.zeros(40, dtype=torch.float32, device=torch.device('cuda'))
        mb = missing_bits
        while mb:
            lsb = mb & -mb
            idx = (lsb.bit_length() - 1)
            vec[idx] = 1.0
            mb ^= lsb
        result = vec
    else:
        all_cards = list(range(40))
        visible_cards = set()
        visible_cards.update(h)
        visible_cards.update(t)
        visible_cards.update(c0)
        visible_cards.update(c1)
        missing_cards = [card for card in all_cards if card not in set(visible_cards)]
        result = encode_cards_as_matrix(missing_cards)
    
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
    
    # Calcolo originale se non in cache
    # Tutte le carte nel mazzo
    # ID set
    all_ids = set(range(40))
    
    # Carte visibili (mano propria, tavolo, catturate)
    visible_cards = set()
    visible_cards.update(game_state["hands"][player_id])
    visible_cards.update(game_state["table"])
    visible_cards.update(game_state["captured_squads"][0])
    visible_cards.update(game_state["captured_squads"][1])
    
    # Carte invisibili
    invisible_cards = all_ids - visible_cards
    
    # Risultato finale
    probs = []
    other_players = [p for p in range(4) if p != player_id]
    
    for p in other_players:
        played_cards = set()
        for mv in game_state["history"]:
            if mv["player"] == p:
                pc = mv["played_card"]
                if isinstance(pc, int):
                    played_cards.add(pc)
                else:
                    # tuple → id
                    r, s = pc
                    pid = (r - 1) * 4 + suit_to_col[s]
                    played_cards.add(pid)
        
        # Dimensione mano attuale
        hand_size = len(game_state["hands"].get(p, []))
        
        # Matrice probabilità (10x4)
        prob_matrix = torch.zeros((10, 4), dtype=torch.float32, device=torch.device('cuda'))
        
        if hand_size == 0 or len(invisible_cards) == 0:
            probs.append(prob_matrix.reshape(-1))
            continue
        
        # Carte possibili per questo giocatore (invisibili e non già giocate)
        possible_cards = invisible_cards - played_cards
        
        # Totale carte rimaste nel gioco non visibili al giocatore corrente
        total_unknown_cards = len(invisible_cards)
        
        # Per ogni carta possibile
        for cid in possible_cards:
            row = int(RANK_OF_ID[cid].item()) - 1
            col = int(SUITCOL_OF_ID[cid].item())
            prob_matrix[row, col] = float(hand_size / total_unknown_cards)
        
        probs.append(prob_matrix.reshape(-1))
    result = torch.cat(probs)  # 120 dim
    
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
    
    # Calcolo originale se non in cache
    # Inizializza gli array per i valori primiera
    team0_primiera = torch.zeros(4, dtype=torch.float32, device=torch.device('cuda'))
    team1_primiera = torch.zeros(4, dtype=torch.float32, device=torch.device('cuda'))
    
    suit_to_idx = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    
    # Calcola i valori primiera per Team 0
    for c in game_state["captured_squads"][0]:
        if isinstance(c, int):
            rank = int(RANK_OF_ID[c])
            suit_idx = int(SUITCOL_OF_ID[c])
        else:
            rank, suit = c
            suit_idx = suit_to_idx[suit]
        primiera_val = PRIMIERA_VAL[rank]
        team0_primiera[suit_idx] = max(team0_primiera[suit_idx], primiera_val)
    
    # Calcola i valori primiera per Team 1
    for c in game_state["captured_squads"][1]:
        if isinstance(c, int):
            rank = int(RANK_OF_ID[c])
            suit_idx = int(SUITCOL_OF_ID[c])
        else:
            rank, suit = c
            suit_idx = suit_to_idx[suit]
        primiera_val = PRIMIERA_VAL[rank]
        team1_primiera[suit_idx] = max(team1_primiera[suit_idx], primiera_val)
    
    # Normalizza
    team0_primiera = team0_primiera / 21.0  # 21 è il valore massimo (7)
    team1_primiera = team1_primiera / 21.0
    
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
    
    # Calcolo ID-only
    cs0 = game_state["captured_squads"][0]
    cs1 = game_state["captured_squads"][1]
    team0_denari = sum(1 for c in cs0 if (c % 4) == 0)
    team1_denari = sum(1 for c in cs1 if (c % 4) == 0)
    
    # Normalizza
    result = torch.tensor([team0_denari / 10.0, team1_denari / 10.0], dtype=torch.float32, device=torch.device('cuda'))
    
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
    result = torch.tensor([status / 3.0], dtype=torch.float32, device=torch.device('cuda'))
    
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
    result = torch.tensor([total0 / 12.0, total1 / 12.0], dtype=torch.float32, device=torch.device('cuda'))
    
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
    if len(tbl) > 0 and isinstance(tbl[0], int):
        bits = 0
        for cid in tbl:
            bits |= (1 << int(cid))
        table_sum = bitset_table_sum(bits)
    else:
        table_sum = sum(card[0] for card in tbl)
    result = torch.tensor([table_sum / 30.0], dtype=torch.float32, device=torch.device('cuda'))
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
    
    # Calcolo originale se non in cache
    next_player_idx = [i for i, p in enumerate([p for p in range(4) if p != player_id]) if p == next_player][0]
    scopa_probs = torch.zeros(10, dtype=torch.float32, device=torch.device('cuda'))
    
    # Se non vengono fornite le probabilità di rank, le calcoliamo
    if rank_probabilities is None:
        rank_probabilities = compute_rank_probabilities_by_player(game_state, player_id)
    
    # Dimensione mano del prossimo giocatore
    hand_size = len(game_state["hands"].get(next_player, []))
    if hand_size == 0:
        return scopa_probs
    
    # Precompute mani ID bitset per rank presenti (se ID)
    hand_list = game_state["hands"][player_id]
    hand_bits = 0
    if hand_list and isinstance(hand_list[0], int):
        for cid in hand_list:
            hand_bits |= (1 << int(cid))
        hand_rank_counts = bitset_rank_counts(hand_bits)
    else:
        hand_rank_counts = None

    # Helper per ottenere rank da carta ID/tuple
    def _rank_of(x):
        return (x // 4) + 1 if isinstance(x, int) else x[0]

    # Per ogni rank che il giocatore corrente potrebbe giocare
    for current_rank in range(1, 11):
        # Verifica se il giocatore ha in mano una carta di questo rank
        if hand_rank_counts is not None:
            has_rank = hand_rank_counts[current_rank-1] > 0
        else:
            has_rank = any(card[0] == current_rank for card in hand_list)
        if not has_rank:
            continue
        
        # Simula cosa accadrebbe al tavolo dopo aver giocato questa carta (in spazio di rank)
        simulated_ranks = [_rank_of(c) for c in game_state["table"]]
        # Regola 1: Cattura diretta di tutte le carte con stesso rank
        if any(r == current_rank for r in simulated_ranks):
            simulated_ranks = [r for r in simulated_ranks if r != current_rank]
        else:
            # Regola 2: Cerca combinazioni che sommano al rank
            captured = None
            n = len(simulated_ranks)
            for subset_size in range(1, min(4, n) + 1):
                for idxs in itertools.combinations(range(n), subset_size):
                    if sum(simulated_ranks[i] for i in idxs) == current_rank:
                        captured = set(idxs)
                        break
                if captured is not None:
                    break
            if captured is not None:
                simulated_ranks = [r for i, r in enumerate(simulated_ranks) if i not in captured]
            else:
                # Nessuna cattura: la carta va sul tavolo (considera solo il rank)
                simulated_ranks.append(current_rank)
        
        # Caso 1: Se il tavolo è vuoto, qualsiasi carta porterebbe a scopa
        if not simulated_ranks:
            # Probabilità che il giocatore abbia almeno una carta = 1 - P(avere 0 carte)
            # Possiamo calcolare questa probabilità sommando tutte le probabilità che
            # il giocatore abbia almeno una carta di qualsiasi rank
            p_has_at_least_one_card = 0.0
            for rank_idx in range(10):
                # Probabilità di avere 0 carte di questo rank
                p_zero = rank_probabilities[next_player_idx, 0, rank_idx]
                # Probabilità di avere almeno una carta di questo rank
                p_at_least_one = 1.0 - p_zero
                p_has_at_least_one_card += p_at_least_one
            
            # Normalizza per evitare probabilità > 1.0
            p_has_at_least_one_card = min(1.0, p_has_at_least_one_card)
            scopa_probs[current_rank-1] = p_has_at_least_one_card
            continue
        
        # Caso 2: Il tavolo non è vuoto, serve una carta specifica
        for next_rank in range(1, 11):
            # Verifica se questo rank può catturare tutto il tavolo
            can_capture_all = False
            
            # Cattura diretta
            if simulated_ranks and all(r == next_rank for r in simulated_ranks):
                can_capture_all = True
            # Cattura per somma
            elif simulated_ranks and sum(simulated_ranks) == next_rank:
                can_capture_all = True
            
            if can_capture_all:
                # Utilizziamo direttamente le probabilità calcolate da compute_rank_probabilities_by_player
                # Probabilità di avere 0 carte di questo rank
                p_zero = rank_probabilities[next_player_idx, 0, next_rank-1]
                # Probabilità di avere almeno una carta di questo rank
                p_at_least_one = 1.0 - p_zero
                
                # Aggiungi alla probabilità totale per questo rank
                scopa_probs[current_rank-1] += p_at_least_one
        
        # Normalizza a 1.0 se necessario
        scopa_probs[current_rank-1] = min(1.0, scopa_probs[current_rank-1])
    
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
    
    # Calcolo originale se non in cache
    all_probs = torch.zeros((3, 5, 10), dtype=torch.float32, device=torch.device('cuda'))
    other_players = [p for p in range(4) if p != player_id]
    
    # Carte visibili per rank
    visible_rank_counts = [0] * 10
    
    # Conta carte visibili: usa bitset_rank_counts quando in ID
    def _accumulate_counts_from_list(lst):
        if lst and isinstance(lst[0], int):
            bits = 0
            for cid in lst:
                bits |= (1 << int(cid))
            cnt = bitset_rank_counts(bits)
            for i in range(10):
                visible_rank_counts[i] += int(cnt[i])
        else:
            for card in lst:
                r, _ = card
                visible_rank_counts[r-1] += 1
    _accumulate_counts_from_list(game_state["table"])
    _accumulate_counts_from_list(game_state["hands"][player_id])
    for team_cards in game_state["captured_squads"].values():
        _accumulate_counts_from_list(team_cards)
    
    # Totale carte visibili e invisibili
    total_invisible = 40 - sum(visible_rank_counts)
    
    for i, p in enumerate(other_players):
        # Carte già giocate da questo giocatore (per rank)
        played_rank_counts = [0] * 10
        for move in game_state.get("history", []):
            if move.get("player") == p:
                pc = move.get("played_card")
                rank = (pc // 4) + 1 if isinstance(pc, int) else pc[0]
                if 1 <= rank <= 10:
                    played_rank_counts[rank-1] += 1
        
        # Carte in mano attualmente
        hand_size = len(game_state["hands"].get(p, []))
        if hand_size == 0:
            # Tutti i rank hanno probabilità 1.0 di avere 0 carte
            for rank_idx in range(10):
                all_probs[i, 0, rank_idx] = 1.0
            continue
        
        for rank in range(1, 11):
            rank_idx = rank - 1
            
            # Carte totali di questo rank
            total_rank = 4
            
            # Carte invisibili di questo rank
            invisible_rank = total_rank - visible_rank_counts[rank_idx]
            
            # Carte già giocate di questo rank dal giocatore
            played_rank = played_rank_counts[rank_idx]
            
            # Carte potenzialmente rimaste di questo rank
            remaining_rank = total_rank - played_rank
            
            # Carte possibili per questo giocatore
            possible_rank = min(remaining_rank, invisible_rank)
            
            if possible_rank < 0:
                # Impossibile avere carte di questo rank
                all_probs[i, 0, rank_idx] = 1.0  # Probabilità 1.0 di avere 0 carte
                continue
            
            try:
                from scipy.special import comb
                
                # Per ogni possibile numero di carte (0-4)
                for k in range(min(possible_rank + 1, hand_size + 1, 5)):
                    # Probabilità ipergeometrica di avere esattamente k carte di questo rank
                    # P(X=k) = [C(K,k) × C(N-K,n-k)] / C(N,n)
                    
                    numerator = comb(invisible_rank, k, exact=True) * comb(total_invisible - invisible_rank, hand_size - k, exact=True)
                    denominator = comb(total_invisible, hand_size, exact=True)
                    
                    if denominator > 0:
                        prob = numerator / denominator
                        all_probs[i, k, rank_idx] = prob
            except:
                # Approssimazione se scipy non è disponibile
                for k in range(min(possible_rank + 1, hand_size + 1, 5)):
                    from math import comb as math_comb
                    try:
                        numerator = math_comb(invisible_rank, k) * math_comb(total_invisible - invisible_rank, hand_size - k)
                        denominator = math_comb(total_invisible, hand_size)
                        prob = numerator / denominator if denominator > 0 else 0
                    except:
                        # Fallback ancora più semplice
                        prob = 0
                        if k == 0:
                            prob = ((total_invisible - invisible_rank) / total_invisible) ** hand_size
                        else:
                            prob = (invisible_rank / total_invisible) ** k * ((total_invisible - invisible_rank) / total_invisible) ** (hand_size - k)
                    
                    all_probs[i, k, rank_idx] = prob
    
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
    # Usa storia completa solo se esplicitamente abilitata via ENV
    import os as _os
    if _os.environ.get('DEBUG_FULL_HISTORY', '0') == '1':
        hist_enc = encode_enhanced_history(game_state, player_id)
    else:
        hist_enc = torch.zeros(10320, dtype=torch.float32, device=torch.device('cuda'))
    
    # --- NUOVE INFORMAZIONI ---
    
    # 6) Matrice di carte mancanti
    missing_cards = compute_missing_cards_matrix(game_state, player_id)  # 40 dim
    
    # 7) Probabilità inferite per gli altri giocatori
    inferred_probs = compute_inferred_probabilities(game_state, player_id)  # 120 dim
    
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
    
    # 13) Probabilità di scopa per ogni rank
    scopa_probs = compute_next_player_scopa_probabilities(game_state, player_id)  # 10 dim
    
    # 14) Probabilità di rank per giocatore
    rank_probs_by_player = compute_rank_probabilities_by_player(game_state, player_id).flatten()  # 150 dim
    
    # Concatena tutte le features (tutte torch su CUDA -> converti a numpy solo in uscita per compat)
    parts = [
        hands_enc.reshape(-1),
        table_enc.reshape(-1),
        captured_enc.reshape(-1),
        cp_enc.reshape(-1),
        (hist_enc if torch.is_tensor(hist_enc) else torch.tensor(hist_enc, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (missing_cards if torch.is_tensor(missing_cards) else torch.tensor(missing_cards, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (inferred_probs if torch.is_tensor(inferred_probs) else torch.tensor(inferred_probs, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (primiera_status if torch.is_tensor(primiera_status) else torch.tensor(primiera_status, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (denari_count if torch.is_tensor(denari_count) else torch.tensor(denari_count, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (settebello_status if torch.is_tensor(settebello_status) else torch.tensor(settebello_status, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (score_estimate if torch.is_tensor(score_estimate) else torch.tensor(score_estimate, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (table_sum if torch.is_tensor(table_sum) else torch.tensor(table_sum, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (scopa_probs if torch.is_tensor(scopa_probs) else torch.tensor(scopa_probs, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
        (rank_probs_by_player if torch.is_tensor(rank_probs_by_player) else torch.tensor(rank_probs_by_player, dtype=torch.float32, device=torch.device('cuda'))).reshape(-1),
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
    history_cache.clear()
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
            parts.append(torch.zeros(61, dtype=torch.float32, device=torch.device('cuda')))
    while len(parts) < k:
        parts.insert(0, torch.zeros(61, dtype=torch.float32, device=torch.device('cuda')))
    result = torch.cat(parts) if parts else torch.zeros(61*k, dtype=torch.float32, device=torch.device('cuda'))
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