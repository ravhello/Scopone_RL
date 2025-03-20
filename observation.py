# observation.py (Versione Avanzata)
import numpy as np
import itertools

SUITS = ['denari', 'coppe', 'spade', 'bastoni']
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Valori per la primiera
PRIMIERA_VAL = {1:16, 2:12, 3:13, 4:14, 5:15, 6:18, 7:21, 8:10, 9:10, 10:10}

def encode_cards_as_matrix(cards):
    """
    Codifica un insieme di carte come una matrice 10x4 (rank x suit).
    - Le righe rappresentano i rank (1-10)
    - Le colonne rappresentano i semi (denari, coppe, spade, bastoni)
    - Valori: 1 indica presenza, 0 assenza
    """
    matrix = np.zeros((10, 4), dtype=np.float32)
    suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    
    for rank, suit in cards:
        row = rank - 1
        col = suit_to_col[suit]
        matrix[row, col] = 1.0
    
    return matrix.flatten()

def encode_hands(hands, player_id):
    """
    Codifica la mano del giocatore corrente come matrice.
    Per gli altri giocatori, memorizza solo il numero di carte.
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
    """
    return encode_cards_as_matrix(table)  # 40 dim

def encode_captured_squads(captured_squads):
    """
    Codifica le carte catturate da ciascuna squadra come matrici.
    """
    team0_cards = encode_cards_as_matrix(captured_squads[0])
    team1_cards = encode_cards_as_matrix(captured_squads[1])
    
    team0_count = len(captured_squads[0]) / 40.0
    team1_count = len(captured_squads[1]) / 40.0
    
    return np.concatenate([team0_cards, team1_cards, [team0_count, team1_count]])  # 82 dim

def encode_current_player(cp):
    """
    4 dimensioni one-hot per il giocatore corrente
    """
    arr = np.zeros(4, dtype=np.float32)
    arr[cp] = 1.0
    return arr  # 4 dim

def encode_move(move):
    """
    Codifica una mossa
    """
    player_vec = np.zeros(4, dtype=np.float32)
    player_vec[move["player"]] = 1.0
    
    # Carta giocata (14 dim)
    played_card = move["played_card"]
    rank, suit = played_card
    
    rank_vec = np.zeros(10, dtype=np.float32)
    rank_vec[rank-1] = 1.0
    
    suit_vec = np.zeros(4, dtype=np.float32)
    suit_idx = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}[suit]
    suit_vec[suit_idx] = 1.0
    
    capture_vec = np.zeros(3, dtype=np.float32)
    capture_map = {"no_capture": 0, "capture": 1, "scopa": 2}
    ctype_idx = capture_map.get(move["capture_type"], 0)
    capture_vec[ctype_idx] = 1.0
    
    captured_vec = encode_cards_as_matrix(move["captured_cards"])
    
    return np.concatenate([player_vec, rank_vec, suit_vec, capture_vec, captured_vec])  # 61 dim

def encode_enhanced_history(game_state, player_id):
    """
    Codifica uno storico di gioco dettagliato per ogni turno della partita (fino a 40).
    Per ogni turno include:
    - Giocatore di turno (4 dim)
    - Carta giocata (40 dim)
    - Tipo di cattura (3 dim)
    - Carte catturate in questo turno (40 dim)
    - Carte catturate dalla squadra 0 fino a quel punto (40 dim)
    - Carte catturate dalla squadra 1 fino a quel punto (40 dim)
    - Carte sul tavolo dopo la mossa (40 dim)
    - Carte in mano all'osservatore in quel turno (40 dim)
    - Somma totale sul tavolo dopo la mossa (1 dim) [NUOVO]
    - Probabilità di scopa per ogni rank (10 dim) [NUOVO]
    """
    max_turns = 40
    turn_dim = 4 + 40 + 3 + 40 + 40 + 40 + 40 + 40 + 1 + 10  # 258 dim per turno 
    hist_arr = np.zeros(max_turns * turn_dim, dtype=np.float32)
    
    # Ricostruisci la storia delle carte catturate e del tavolo
    team0_captured = []
    team1_captured = []
    table_after_moves = []
    current_team0 = []
    current_team1 = []
    current_table = []
    table_sums = []  # [NUOVO] Somme del tavolo dopo ogni mossa
    scopa_probs_history = []  # [NUOVO] Probabilità di scopa dopo ogni mossa
    
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
        
        # Se è il giocatore osservatore che sta giocando, rimuovi la carta dalla sua mano ricostruita
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
    table_sums_slice = table_sums[-max_turns:]  # [NUOVO]
    scopa_probs_slice = scopa_probs_history[-max_turns:]  # [NUOVO]
    
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
    
    return hist_arr  # 40 * 258 = 10320 dim

# ----- NUOVE FUNZIONI PER LE INFORMAZIONI AVANZATE -----

def compute_missing_cards_matrix(game_state, player_id):
    """
    Crea una matrice 10x4 che indica le carte che non sono visibili al giocatore
    (non in mano propria, non sul tavolo, non catturate)
    """
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
    return encode_cards_as_matrix(missing_cards)  # 40 dim

def compute_inferred_probabilities(game_state, player_id):
    """
    Crea 3 matrici 10x4 di probabilità inferite per gli altri giocatori.
    """
    # Inizia con tutte le carte (40 carte)
    all_cards = [(r, s) for r in RANKS for s in SUITS]
    all_cards_set = set(all_cards)
    
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
    missing_cards = all_cards_set - visible_cards
    missing_cards_list = list(missing_cards)
    
    # Per ogni altro giocatore, calcola la probabilità di avere ciascuna carta
    probs = []
    other_players = [p for p in range(4) if p != player_id]
    
    for p in other_players:
        # Conteggio carte in mano
        cards_in_hand = len(game_state["hands"][p])
        
        # Matrice delle probabilità
        prob_matrix = np.zeros((10, 4), dtype=np.float32)
        suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
        
        # Se il giocatore ha 0 carte, tutte le probabilità sono 0
        if cards_in_hand == 0:
            probs.append(prob_matrix.flatten())
            continue
        
        # Probabilità di base per ogni carta mancante
        # (assumendo distribuzione uniforme tra i giocatori con carte)
        total_missing_cards = sum(len(game_state["hands"][j]) for j in range(4) if j != player_id)
        
        if total_missing_cards > 0:
            base_prob = cards_in_hand / total_missing_cards
            
            # Assegna la probabilità di base a tutte le carte mancanti
            for rank, suit in missing_cards_list:
                row = rank - 1
                col = suit_to_col[suit]
                prob_matrix[row, col] = base_prob
        
        probs.append(prob_matrix.flatten())
    
    return np.concatenate(probs)  # 120 dim (3 giocatori x 40 dim)

def compute_primiera_status(game_state):
    """
    Calcola lo stato attuale della primiera per entrambe le squadre.
    Per ogni seme, prende il valore più alto delle carte catturate.
    """
    # Inizializza gli array per i valori primiera
    team0_primiera = np.zeros(4, dtype=np.float32)
    team1_primiera = np.zeros(4, dtype=np.float32)
    
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
    
    return np.concatenate([team0_primiera, team1_primiera])  # 8 dim

def compute_denari_count(game_state):
    """
    Calcola il numero di denari catturati da ciascuna squadra.
    """
    team0_denari = sum(1 for card in game_state["captured_squads"][0] if card[1] == 'denari')
    team1_denari = sum(1 for card in game_state["captured_squads"][1] if card[1] == 'denari')
    
    # Normalizza
    return np.array([team0_denari / 10.0, team1_denari / 10.0], dtype=np.float32)  # 2 dim

def compute_settebello_status(game_state):
    """
    Indica lo stato del settebello:
    0 = non visto
    1 = catturato da squadra 0
    2 = catturato da squadra 1
    3 = sul tavolo
    """
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
    return np.array([status / 3.0], dtype=np.float32)  # 1 dim

def compute_current_score_estimate(game_state):
    """
    Stima il punteggio attuale basandosi sulle carte catturate finora.
    """
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
    
    # Normalizza (in genere il punteggio massimo è intorno a 11-12)
    return np.array([total0 / 12.0, total1 / 12.0], dtype=np.float32)  # 2 dim

def compute_table_sum(game_state):
    """
    Calcola la somma totale dei rank sul tavolo.
    """
    table_sum = sum(card[0] for card in game_state["table"])
    
    # Normalizza (la somma massima teorica è 78, ma in pratica raramente supera 30)
    return np.array([table_sum / 30.0], dtype=np.float32)  # 1 dim

def compute_next_player_scopa_probabilities(game_state, player_id):
    """
    Per ogni rank (1-10) che il giocatore corrente potrebbe giocare,
    calcola la probabilità che il giocatore successivo possa fare scopa.
    
    Args:
        game_state: Stato attuale del gioco
        player_id: ID del giocatore corrente
    
    Returns:
        Array di 10 dimensioni con le probabilità di scopa per il prossimo giocatore
        condizionate a ogni possibile rank giocato dal giocatore corrente.
    """
    # Identifica il prossimo giocatore
    next_player = (player_id + 1) % 4
    
    # Carte già giocate dal prossimo giocatore
    played_by_next = []
    for move in game_state["history"]:
        if move["player"] == next_player:
            played_by_next.append(move["played_card"])
    
    # Carte sul tavolo attuale
    current_table = game_state["table"].copy()
    
    # Carte visibili (per calcolare le invisibili)
    visible_cards = set()
    visible_cards.update(game_state["hands"][player_id])  # Mano propria
    visible_cards.update(current_table)  # Tavolo
    visible_cards.update(game_state["captured_squads"][0])  # Catturate team 0
    visible_cards.update(game_state["captured_squads"][1])  # Catturate team 1
    
    # Tutte le carte possibili
    all_cards = [(r, s) for r in RANKS for s in SUITS]
    
    # Carte invisibili (potenzialmente nelle mani degli altri)
    invisible_cards = [c for c in all_cards if c not in visible_cards]
    
    # Calcola quante carte ha in mano il prossimo giocatore
    next_player_hand_size = len(game_state["hands"].get(next_player, []))
    
    # Array di probabilità finale
    scopa_probs = np.zeros(10, dtype=np.float32)
    
    # Per ogni rank che il giocatore corrente potrebbe giocare
    for current_rank in range(1, 11):
        # Verifica se il giocatore ha in mano una carta di questo rank
        has_rank = any(card[0] == current_rank for card in game_state["hands"][player_id])
        if not has_rank:
            continue
        
        # Simula cosa accadrebbe al tavolo dopo aver giocato questa carta
        simulated_table = current_table.copy()
        
        # Determina se il giocatore corrente cattura carte e quali
        cards_to_capture = []
        
        # Regola 1: Cattura diretta di carte dello stesso rank
        same_rank_cards = [c for c in current_table if c[0] == current_rank]
        if same_rank_cards:
            cards_to_capture = same_rank_cards
        else:
            # Regola 2: Cerca combinazioni che sommano al rank
            # Per semplicità, implementiamo solo combinazioni fino a 3 carte
            for subset_size in range(1, min(4, len(current_table)+1)):
                for subset in itertools.combinations(current_table, subset_size):
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
            # Il seme non è importante per questa simulazione
            simulated_table.append((current_rank, 'denari'))
        
        # Ora calcola la probabilità che il prossimo giocatore possa fare scopa
        
        # Caso 1: Se il tavolo è vuoto, qualsiasi carta porterebbe a scopa
        if not simulated_table:
            if next_player_hand_size > 0:
                scopa_probs[current_rank-1] = 1.0
            continue
        
        # Caso 2: Il tavolo non è vuoto, calcola per ogni rank che il prossimo 
        # giocatore potrebbe avere
        for next_rank in range(1, 11):
            # Quante carte di questo rank il prossimo giocatore ha già giocato
            played_count = sum(1 for card in played_by_next if card[0] == next_rank)
            
            # Quante carte di questo rank potrebbero essere ancora in mano
            available_count = 4 - played_count
            
            # Quante carte di questo rank sono tra le invisibili
            invisible_count = sum(1 for card in invisible_cards if card[0] == next_rank)
            
            # Se nessuna carta disponibile, salta
            if available_count == 0 or invisible_count == 0:
                continue
            
            # Verifica se questo rank può catturare tutto il tavolo
            can_capture_all = False
            
            # Cattura diretta
            if all(card[0] == next_rank for card in simulated_table):
                can_capture_all = True
            else:
                # Cattura per somma
                if sum(card[0] for card in simulated_table) == next_rank:
                    can_capture_all = True
            
            if can_capture_all:
                # Calcola probabilità che il prossimo giocatore abbia questo rank
                # basata su: carte disponibili, invisibili e mano del giocatore
                
                # Probabilità base: carte di questo rank tra le invisibili / totale invisibili
                base_prob = invisible_count / max(1, len(invisible_cards))
                
                # Aggiustamento per numero di carte in mano
                hand_factor = next_player_hand_size / max(1, sum(len(game_state["hands"][p]) 
                                                           for p in range(4) if p != player_id))
                
                # Probabilità finale
                rank_prob = base_prob * hand_factor * available_count / 4.0
                
                # Aggiungi alla probabilità totale per questo rank corrente
                scopa_probs[current_rank-1] += min(1.0, rank_prob)
        
        # Normalizza a 1.0 se necessario
        scopa_probs[current_rank-1] = min(1.0, scopa_probs[current_rank-1])
    
    return scopa_probs  # 10 dim

def compute_rank_probabilities_by_player(game_state, player_id):
    """
    Calcola, per ogni giocatore esterno, la probabilità di avere 1-4 carte di ogni rank
    considerando le carte che hanno già giocato.
    
    Returns:
        Numpy array di dimensione (3, 4, 10) = 120 dimensioni totali
        - 3 giocatori esterni
        - 4 possibili quantità di carte (1, 2, 3, 4)
        - 10 rank possibili
    """
    # Inizializza l'array delle probabilità
    all_probs = np.zeros((3, 4, 10), dtype=np.float32)
    
    # Identifica i giocatori diversi dal giocatore corrente
    other_players = [p for p in range(4) if p != player_id]
    
    for i, p in enumerate(other_players):
        # Carte già giocate da questo giocatore specifico
        played_cards_by_p = []
        for move in game_state["history"]:
            if move["player"] == p:
                played_cards_by_p.append(move["played_card"])
        
        # Conteggio delle carte per rank giocate da questo giocatore
        played_rank_counts = [0] * 10
        for card in played_cards_by_p:
            rank, _ = card
            played_rank_counts[rank-1] += 1
        
        # Conteggio complessivo delle carte visibili per rank
        visible_rank_counts = [0] * 10
        
        # Carte sul tavolo
        for card in game_state["table"]:
            rank, _ = card
            visible_rank_counts[rank-1] += 1
        
        # Carte già giocate (nella storia)
        for move in game_state["history"]:
            rank, _ = move["played_card"]
            visible_rank_counts[rank-1] += 1
        
        # Carte nella mano del giocatore corrente
        for card in game_state["hands"][player_id]:
            rank, _ = card
            visible_rank_counts[rank-1] += 1
        
        # Carte catturate
        for team_cards in game_state["captured_squads"].values():
            for card in team_cards:
                rank, _ = card
                visible_rank_counts[rank-1] += 1
        
        # Calcolo delle probabilità
        cards_in_hand = len(game_state["hands"][p])
        
        for rank in range(1, 11):
            rank_idx = rank - 1
            
            # Carte già giocate da questo giocatore
            played = played_rank_counts[rank_idx]
            
            # Carte totali di questo rank
            total = 4
            
            # Potenziali carte rimanenti di questo rank
            potentially_remaining = total - played
            
            # Carte non visibili di questo rank
            invisible = max(0, total - visible_rank_counts[rank_idx])
            
            # Stima delle carte in mano
            estimated_in_hand = min(potentially_remaining, invisible)
            
            if cards_in_hand == 0:
                continue
                
            # Calcola probabilità per 1, 2, 3 o 4 carte
            for num_cards in range(1, 5):
                if num_cards <= estimated_in_hand and num_cards <= cards_in_hand:
                    # Fattori di probabilità
                    
                    # Base: quanto questo rank è rappresentato tra le carte invisibili
                    base_prob = estimated_in_hand / max(1, invisible)
                    
                    # Fattore mano: più carte ha, più probabile è che abbia questo rank
                    hand_factor = min(1.0, cards_in_hand / 10.0)
                    
                    # Fattore giocate: se ha già giocato carte di questo rank, è meno
                    # probabile che ne abbia altre
                    played_factor = max(0.1, 1.0 - (played / 4.0))
                    
                    # Probabilità finale
                    prob = base_prob * hand_factor * played_factor
                    
                    # Riduzione per più carte dello stesso rank
                    if num_cards > 1:
                        prob = prob / (2.0 ** (num_cards - 1))
                    
                    all_probs[i, num_cards-1, rank_idx] = min(1.0, prob)
    
    return all_probs  # Dimensione (3, 4, 10) = 120 dim

def encode_state_for_player(game_state, player_id):
    """
    Versione avanzata che include statistiche aggiuntive per migliorare il processo decisionale.
    Ora utilizza encode_enhanced_history per uno storico più dettagliato.
    """
    # 1) Mani
    hands_enc = encode_hands(game_state["hands"], player_id)  # 43 dim
    
    # 2) Tavolo
    table_enc = encode_table(game_state["table"])  # 40 dim
    
    # 3) Catture squadre
    captured_enc = encode_captured_squads(game_state["captured_squads"])  # 82 dim
    
    # 4) current_player
    cp = game_state.get("current_player", 0)
    cp_enc = encode_current_player(cp)  # 4 dim
    
    # 5) history - MODIFICATO: usiamo encode_enhanced_history invece di encode_history
    hist_enc = encode_enhanced_history(game_state, player_id)  # # 10320 dim
    
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
    rank_probs_by_player = compute_rank_probabilities_by_player(game_state, player_id).flatten()  # 120 dim
    
    # Concatena tutte le features
    return np.concatenate([
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
        scopa_probs,            # 10 dim
        rank_probs_by_player  # 120 dim
    ])  # Totale: 10793 dim