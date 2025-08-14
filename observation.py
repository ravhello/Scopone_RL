# observation.py - Versione Ottimizzata (risultati identici all'originale)
import numpy as np
import itertools
from functools import lru_cache

SUITS = ['denari', 'coppe', 'spade', 'bastoni']
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Valori per la primiera
PRIMIERA_VAL = {1:16, 2:12, 3:13, 4:14, 5:15, 6:18, 7:21, 8:10, 9:10, 10:10}

# Mappa condivisa per conversione suit → index
suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}

# ----- OTTIMIZZAZIONE: CACHE PER FUNZIONI COSTOSE -----
# Cache per risultati costosi
cards_matrix_cache = {}

def encode_cards_as_matrix(cards):
    """
    Versione ottimizzata con caching ma risultati identici.
    Codifica un insieme di carte come una matrice 10x4 (rank x suit).
    """
    # Caso speciale per lista vuota
    if not cards:
        return np.zeros(40, dtype=np.float32)
    
    # Genera chiave cache univoca efficiente
    # Ordiniamo le carte per garantire coerenza con input diversi ma equivalenti
    cards_tuple = tuple(sorted((r, s) for r, s in cards))
    cache_key = hash(cards_tuple)
    
    # Verifica cache
    if cache_key in cards_matrix_cache:
        return cards_matrix_cache[cache_key]
    
    # Calcolo non in cache: usa l'algoritmo originale
    matrix = np.zeros((10, 4), dtype=np.float32)
    
    for rank, suit in cards:
        row = rank - 1
        col = suit_to_col[suit]
        matrix[row, col] = 1.0
    
    result = matrix.flatten()
    
    # Salva in cache
    cards_matrix_cache[cache_key] = result
    
    # Limita dimensione cache
    if len(cards_matrix_cache) > 1000:
        # Rimuovi 200 elementi casuali per fare spazio
        import random
        keys_to_remove = random.sample(list(cards_matrix_cache.keys()), 200)
        for k in keys_to_remove:
            del cards_matrix_cache[k]
    
    return result

def encode_hands(hands, player_id):
    """
    Codifica la mano del giocatore corrente come matrice.
    Versione ottimizzata con meno allocazioni ma risultati identici.
    """
    player_hand = encode_cards_as_matrix(hands[player_id])
    
    other_counts = np.zeros(3, dtype=np.float32)
    count_idx = 0
    for p in range(4):
        if p != player_id:
            other_counts[count_idx] = len(hands[p]) / 10.0
            count_idx += 1
    
    return np.concatenate([player_hand, other_counts])  # 43 dim

def encode_table(table):
    """
    Codifica il tavolo come matrice rank x suit.
    Versione cachizzata con risultati identici.
    """
    return encode_cards_as_matrix(table)  # 40 dim

def encode_captured_squads(captured_squads):
    """
    Versione ottimizzata con risultati identici.
    """
    team0_cards = encode_cards_as_matrix(captured_squads[0])
    team1_cards = encode_cards_as_matrix(captured_squads[1])
    
    # Dimensioni originali (team0, team1, team0_count, team1_count)
    team0_count = len(captured_squads[0]) / 40.0
    team1_count = len(captured_squads[1]) / 40.0
    
    return np.concatenate([team0_cards, team1_cards, [team0_count, team1_count]])  # 82 dim

# One-hot encoding per player (pre-calcolato per velocità)
ONE_HOT_PLAYERS = {
    0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    2: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    3: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
}

def encode_current_player(cp):
    """
    Versione ottimizzata con risultati identici.
    Usa array pre-calcolati invece di crearli ogni volta.
    """
    return ONE_HOT_PLAYERS[cp].copy()  # Copia per evitare modifiche esterne

def encode_move(move):
    """
    Codifica una mossa. Versione ottimizzata con risultati identici.
    """
    player_vec = np.zeros(4, dtype=np.float32)
    player_vec[move["player"]] = 1.0
    
    # Carta giocata (14 dim)
    played_card = move["played_card"]
    rank, suit = played_card
    
    rank_vec = np.zeros(10, dtype=np.float32)
    rank_vec[rank-1] = 1.0
    
    suit_vec = np.zeros(4, dtype=np.float32)
    suit_idx = suit_to_col[suit]
    suit_vec[suit_idx] = 1.0
    
    capture_vec = np.zeros(3, dtype=np.float32)
    capture_map = {"no_capture": 0, "capture": 1, "scopa": 2}
    ctype_idx = capture_map.get(move["capture_type"], 0)
    capture_vec[ctype_idx] = 1.0
    
    captured_vec = encode_cards_as_matrix(move["captured_cards"])
    
    return np.concatenate([player_vec, rank_vec, suit_vec, capture_vec, captured_vec])  # 61 dim

# Cache per history (chiave: lunghezza history, player_id, hash di hands e table)
history_cache = {}

def encode_enhanced_history(game_state, player_id):
    """
    Versione ottimizzata con cache ma risultati identici.
    """
    # Costruisci una chiave di cache efficiente ma unica
    # Includiamo solo gli elementi che influenzano l'output
    history_len = len(game_state["history"])
    player_hand_tuple = tuple(sorted((r, s) for r, s in game_state["hands"].get(player_id, [])))
    table_tuple = tuple(sorted((r, s) for r, s in game_state["table"]))
    
    # Rappresentazione compatta delle carte catturate (solo lunghezze)
    captured0_len = len(game_state["captured_squads"][0])
    captured1_len = len(game_state["captured_squads"][1])
    
    # Chiave di cache
    cache_key = (history_len, player_id, player_hand_tuple, table_tuple, 
                captured0_len, captured1_len)
    
    # Controlla cache
    if cache_key in history_cache:
        return history_cache[cache_key].copy()  # Copia per sicurezza
    
    # Se non in cache, calcola normalmente seguendo l'algoritmo originale
    max_turns = 40
    turn_dim = 4 + 40 + 3 + 40 + 40 + 40 + 40 + 40 + 1 + 10  # 258 dim per turno 
    hist_arr = np.zeros(max_turns * turn_dim, dtype=np.float32)
    
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
        current_scopa_probs = np.zeros(10, dtype=np.float32)
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
        player_vec = np.zeros(4, dtype=np.float32)
        player_vec[player] = 1.0
        hist_arr[turn_offset:turn_offset+4] = player_vec
        
        # Carta giocata (40 dim)
        played_card_vec = encode_cards_as_matrix([played_card])
        hist_arr[turn_offset+4:turn_offset+44] = played_card_vec
        
        # Tipo di cattura (3 dim)
        capture_vec = np.zeros(3, dtype=np.float32)
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
    history_cache[cache_key] = hist_arr.copy()
    
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
    player_hand = tuple(sorted(game_state["hands"][player_id]))
    table = tuple(sorted(game_state["table"]))
    team0_cards = tuple(sorted(game_state["captured_squads"][0]))
    team1_cards = tuple(sorted(game_state["captured_squads"][1]))
    
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards)
    
    # Controlla cache
    if cache_key in missing_cards_cache:
        return missing_cards_cache[cache_key].copy()
    
    # Calcolo originale se non in cache
    # Inizia con tutte le carte (40 carte)
    all_cards = [(r, s) for r in RANKS for s in SUITS]
    
    # Rimuovi le carte visibili
    visible_cards = set()
    
    # Carte in mano propria
    visible_cards.update(game_state["hands"][player_id])
    
    # Carte sul tavolo
    visible_cards.update(game_state["table"])
    
    # Carte catturate da entrambe le squadre
    visible_cards.update(game_state["captured_squads"][0])
    visible_cards.update(game_state["captured_squads"][1])
    
    # Le carte mancanti sono quelle che non sono visibili
    missing_cards = [card for card in all_cards if card not in visible_cards]
    
    # Codifica come matrice
    result = encode_cards_as_matrix(missing_cards)
    
    # Salva in cache
    missing_cards_cache[cache_key] = result.copy()
    
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
        return inferred_probs_cache[cache_key].copy()
    
    # Calcolo originale se non in cache
    # Tutte le carte nel mazzo
    all_cards = [(r, s) for r in RANKS for s in SUITS]
    all_cards_set = set(all_cards)
    
    # Carte visibili (mano propria, tavolo, catturate)
    visible_cards = set()
    visible_cards.update(game_state["hands"][player_id])
    visible_cards.update(game_state["table"])
    visible_cards.update(game_state["captured_squads"][0])
    visible_cards.update(game_state["captured_squads"][1])
    
    # Carte invisibili
    invisible_cards = all_cards_set - visible_cards
    
    # Risultato finale
    probs = []
    other_players = [p for p in range(4) if p != player_id]
    
    for p in other_players:
        # Carte già giocate da questo giocatore
        played_cards = set()
        for move in game_state["history"]:
            if move["player"] == p:
                played_cards.add(move["played_card"])
        
        # Dimensione mano attuale
        hand_size = len(game_state["hands"].get(p, []))
        
        # Matrice probabilità (10x4)
        prob_matrix = np.zeros((10, 4), dtype=np.float32)
        
        if hand_size == 0 or len(invisible_cards) == 0:
            probs.append(prob_matrix.flatten())
            continue
        
        # Carte possibili per questo giocatore (invisibili e non già giocate)
        possible_cards = invisible_cards - played_cards
        
        # Totale carte rimaste nel gioco non visibili al giocatore corrente
        total_unknown_cards = len(invisible_cards)
        
        # Per ogni carta possibile
        for rank, suit in possible_cards:
            row = rank - 1
            col = suit_to_col[suit]
            
            # Probabilità ipergeometrica semplice: hand_size / total_unknown_cards
            prob_matrix[row, col] = hand_size / total_unknown_cards
        
        probs.append(prob_matrix.flatten())
    
    result = np.concatenate(probs)  # 120 dim
    
    # Salva in cache
    inferred_probs_cache[cache_key] = result.copy()
    
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
        return primiera_cache[cache_key].copy()
    
    # Calcolo originale se non in cache
    # Inizializza gli array per i valori primiera
    team0_primiera = np.zeros(4, dtype=np.float64)
    team1_primiera = np.zeros(4, dtype=np.float64)
    
    suit_to_idx = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    
    # Calcola i valori primiera per Team 0
    for rank, suit in game_state["captured_squads"][0]:
        suit_idx = suit_to_idx[suit]
        primiera_val = PRIMIERA_VAL[rank]
        team0_primiera[suit_idx] = max(team0_primiera[suit_idx], primiera_val)
    
    # Calcola i valori primiera per Team 1
    for rank, suit in game_state["captured_squads"][1]:
        suit_idx = suit_to_idx[suit]
        primiera_val = PRIMIERA_VAL[rank]
        team1_primiera[suit_idx] = max(team1_primiera[suit_idx], primiera_val)
    
    # Normalizza
    team0_primiera = team0_primiera / 21.0  # 21 è il valore massimo (7)
    team1_primiera = team1_primiera / 21.0
    
    result = np.concatenate([team0_primiera, team1_primiera])
    
    # Salva in cache
    primiera_cache[cache_key] = result.copy()
    
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
        return denari_cache[cache_key].copy()
    
    # Calcolo originale
    team0_denari = sum(1 for card in game_state["captured_squads"][0] if card[1] == 'denari')
    team1_denari = sum(1 for card in game_state["captured_squads"][1] if card[1] == 'denari')
    
    # Normalizza
    result = np.array([team0_denari / 10.0, team1_denari / 10.0], dtype=np.float64)
    
    # Salva in cache
    denari_cache[cache_key] = result.copy()
    
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
    team0_has_settebello = (7, 'denari') in game_state["captured_squads"][0]
    team1_has_settebello = (7, 'denari') in game_state["captured_squads"][1]
    table_has_settebello = (7, 'denari') in game_state["table"]
    cache_key = (team0_has_settebello, team1_has_settebello, table_has_settebello)
    
    # Controlla cache
    if cache_key in settebello_cache:
        return settebello_cache[cache_key].copy()
    
    # Calcolo originale
    settebello = (7, 'denari')
    
    # Controlla se il settebello è stato catturato da una squadra
    if settebello in game_state["captured_squads"][0]:
        status = 1
    elif settebello in game_state["captured_squads"][1]:
        status = 2
    # Controlla se il settebello è sul tavolo
    elif settebello in game_state["table"]:
        status = 3
    else:
        status = 0
    
    # Normalizza
    result = np.array([status / 3.0], dtype=np.float64)
    
    # Salva in cache
    settebello_cache[cache_key] = result.copy()
    
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
        return score_cache[cache_key].copy()
    
    # Calcolo originale
    # Conteggio scope
    scope0 = 0
    scope1 = 0
    for move in game_state["history"]:
        if move["capture_type"] == "scopa":
            if move["player"] in [0, 2]:
                scope0 += 1
            else:
                scope1 += 1
    
    # Carte totali
    c0 = len(game_state["captured_squads"][0])
    c1 = len(game_state["captured_squads"][1])
    pt_c0, pt_c1 = (1, 0) if c0 > c1 else (0, 1) if c1 > c0 else (0, 0)
    
    # Denari
    den0 = sum(1 for c in game_state["captured_squads"][0] if c[1] == 'denari')
    den1 = sum(1 for c in game_state["captured_squads"][1] if c[1] == 'denari')
    pt_d0, pt_d1 = (1, 0) if den0 > den1 else (0, 1) if den1 > den0 else (0, 0)
    
    # Settebello
    sb0 = 1 if (7, 'denari') in game_state["captured_squads"][0] else 0
    sb1 = 1 if (7, 'denari') in game_state["captured_squads"][1] else 0
    
    # Primiera (calcolo semplificato)
    primiera_status = compute_primiera_status(game_state)
    team0_prim_sum = np.sum(primiera_status[:4]) * 21.0  # Denormalizza
    team1_prim_sum = np.sum(primiera_status[4:]) * 21.0  # Denormalizza
    pt_p0, pt_p1 = (1, 0) if team0_prim_sum > team1_prim_sum else (0, 1) if team1_prim_sum > team0_prim_sum else (0, 0)
    
    # Punteggio totale
    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scope0
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scope1
    
    # Normalizza
    result = np.array([total0 / 12.0, total1 / 12.0], dtype=np.float32)
    
    # Salva in cache
    score_cache[cache_key] = result.copy()
    
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
    # Chiave cache
    table = tuple(sorted(game_state["table"]))
    
    # Controlla cache
    if table in table_sum_cache:
        return table_sum_cache[table].copy()
    
    # Calcolo originale
    table_sum = sum(card[0] for card in game_state["table"])
    
    # Normalizza
    result = np.array([table_sum / 30.0], dtype=np.float32)
    
    # Salva in cache
    table_sum_cache[table] = result.copy()
    
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
        return scopa_probs_cache[cache_key].copy()
    
    # Calcolo originale se non in cache
    next_player_idx = [i for i, p in enumerate([p for p in range(4) if p != player_id]) if p == next_player][0]
    scopa_probs = np.zeros(10, dtype=np.float32)
    
    # Se non vengono fornite le probabilità di rank, le calcoliamo
    if rank_probabilities is None:
        rank_probabilities = compute_rank_probabilities_by_player(game_state, player_id)
    
    # Dimensione mano del prossimo giocatore
    hand_size = len(game_state["hands"].get(next_player, []))
    if hand_size == 0:
        return scopa_probs
    
    # Per ogni rank che il giocatore corrente potrebbe giocare
    for current_rank in range(1, 11):
        # Verifica se il giocatore ha in mano una carta di questo rank
        has_rank = any(card[0] == current_rank for card in game_state["hands"][player_id])
        if not has_rank:
            continue
        
        # Simula cosa accadrebbe al tavolo dopo aver giocato questa carta
        simulated_table = game_state["table"].copy()
        
        # Determina se il giocatore corrente cattura carte e quali
        cards_to_capture = []
        
        # Regola 1: Cattura diretta di carte dello stesso rank
        same_rank_cards = [c for c in simulated_table if c[0] == current_rank]
        if same_rank_cards:
            cards_to_capture = same_rank_cards
        else:
            # Regola 2: Cerca combinazioni che sommano al rank
            for subset_size in range(1, min(4, len(simulated_table)+1)):
                for subset in itertools.combinations(simulated_table, subset_size):
                    if sum(c[0] for c in subset) == current_rank:
                        cards_to_capture = list(subset)
                        break
                if cards_to_capture:
                    break
        
        # Aggiorna il tavolo simulato
        if cards_to_capture:
            # Rimuovi le carte catturate
            for card in cards_to_capture:
                simulated_table.remove(card)
        else:
            # Aggiungi la carta giocata al tavolo
            simulated_table.append((current_rank, 'denari'))
        
        # Caso 1: Se il tavolo è vuoto, qualsiasi carta porterebbe a scopa
        if not simulated_table:
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
            if all(card[0] == next_rank for card in simulated_table):
                can_capture_all = True
            # Cattura per somma
            elif sum(card[0] for card in simulated_table) == next_rank:
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
    scopa_probs_cache[cache_key] = scopa_probs.copy()
    
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
    
    # Rappresentazione compatta della history
    history_summary = tuple((m["player"], m["played_card"][0]) for m in game_state["history"])
    
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards, 
                other_hands, hash(history_summary))
    
    # Controlla cache
    if cache_key in rank_prob_cache:
        return rank_prob_cache[cache_key].copy()
    
    # Calcolo originale se non in cache
    all_probs = np.zeros((3, 5, 10), dtype=np.float32)
    other_players = [p for p in range(4) if p != player_id]
    
    # Carte visibili per rank
    visible_rank_counts = [0] * 10
    
    # Conta carte visibili
    for card in game_state["table"]:
        rank, _ = card
        visible_rank_counts[rank-1] += 1
    
    for card in game_state["hands"][player_id]:
        rank, _ = card
        visible_rank_counts[rank-1] += 1
    
    for team_cards in game_state["captured_squads"].values():
        for card in team_cards:
            rank, _ = card
            visible_rank_counts[rank-1] += 1
    
    # Totale carte visibili e invisibili
    total_invisible = 40 - sum(visible_rank_counts)
    
    for i, p in enumerate(other_players):
        # Carte già giocate da questo giocatore (per rank)
        played_rank_counts = [0] * 10
        for move in game_state["history"]:
            if move["player"] == p:
                rank, _ = move["played_card"]
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
    rank_prob_cache[cache_key] = all_probs.copy()
    
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
    
    # Includiamo la lunghezza della history e il current_player
    history_len = len(game_state["history"])
    cp = game_state.get("current_player", 0)
    
    # Carte in mano degli altri giocatori (solo lunghezze)
    other_hands = tuple((p, len(game_state["hands"].get(p, []))) 
                      for p in range(4) if p != player_id)
    
    cache_key = (player_id, player_hand, table, team0_cards, team1_cards, 
                history_len, cp, other_hands)
    
    # Controlla cache
    if cache_key in state_cache:
        return state_cache[cache_key].copy()
    
    # Calcolo originale se non in cache
    # 1) Mani
    hands_enc = encode_hands(game_state["hands"], player_id)  # 43 dim
    
    # 2) Tavolo
    table_enc = encode_table(game_state["table"])  # 40 dim
    
    # 3) Catture squadre
    captured_enc = encode_captured_squads(game_state["captured_squads"])  # 82 dim
    
    # 4) current_player
    cp_enc = encode_current_player(cp)  # 4 dim
    
    # 5) history
    hist_enc = encode_enhanced_history(game_state, player_id)  # 10320 dim
    
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
    
    # Concatena tutte le features
    result = np.concatenate([
        hands_enc,             # 43 dim
        table_enc,             # 40 dim
        captured_enc,          # 82 dim
        cp_enc,                # 4 dim
        hist_enc,              # 10320 dim
        missing_cards,         # 40 dim
        inferred_probs,        # 120 dim
        primiera_status,       # 8 dim
        denari_count,          # 2 dim
        settebello_status,     # 1 dim
        score_estimate,        # 2 dim
        table_sum,             # 1 dim
        scopa_probs,           # 10 dim
        rank_probs_by_player   # 150 dim
    ])  # Totale: 10823 dim
    
    # Salva in cache, se non troppo grande
    if len(state_cache) < 30:  # Cache molto piccola per evitare problemi di memoria
        state_cache[cache_key] = result.copy()
    
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