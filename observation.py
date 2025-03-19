# observation.py
import numpy as np

SUITS_ORDER = ['denari','coppe','spade','bastoni']
RANKS_ORDER = [1,2,3,4,5,6,7,8,9,10]

def encode_card_onehot(card):
    """
    Converte una carta (rank, suit) in un vettore one-hot di dimensione 14
    - 10 bit per rank (1-10)
    - 4 bit per suit (denari, coppe, spade, bastoni)
    """
    rank, suit = card
    vec = np.zeros(14, dtype=np.float32)
    
    # One-hot per il rank (1-10)
    vec[rank-1] = 1.0  # rank è 1-indexed, quindi sottraiamo 1
    
    # One-hot per il seme
    suit_map = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    vec[10 + suit_map[suit]] = 1.0
    
    return vec

def encode_hand(hand):
    """
    Converte una mano in un vettore.
    Ogni carta è codificata come 14 bit (10 per rank, 4 per suit).
    Poiché una mano può avere al massimo 10 carte, otteniamo un vettore di 10*14=140 dimensioni.
    """
    max_cards = 10
    card_size = 14
    vec = np.zeros(max_cards * card_size, dtype=np.float32)
    
    for i, card in enumerate(hand):
        if i >= max_cards:
            break
        vec[i*card_size:(i+1)*card_size] = encode_card_onehot(card)
    
    return vec

def encode_table(table):
    """
    Converte il tavolo in un vettore.
    Ogni carta è codificata come 14 bit (10 per rank, 4 per suit).
    Poiché sul tavolo ci possono essere al massimo 10 carte, otteniamo un vettore di 10*14=140 dimensioni.
    """
    max_cards = 10
    card_size = 14
    vec = np.zeros(max_cards * card_size, dtype=np.float32)
    
    for i, card in enumerate(table):
        if i >= max_cards:
            break
        vec[i*card_size:(i+1)*card_size] = encode_card_onehot(card)
    
    return vec

def encode_captured_squads(captured_squads):
    """
    Codifica le carte catturate da ciascuna squadra.
    Per ogni squadra, fino a 20 carte, ciascuna codificata come 14 bit.
    Totale: 2 squadre * 20 carte * 14 bit = 560 dimensioni.
    """
    max_cards_per_squad = 20
    card_size = 14
    vec_size = max_cards_per_squad * card_size
    
    out = []
    for sq in [0, 1]:
        arr = np.zeros(vec_size, dtype=np.float32)
        for i, card in enumerate(captured_squads[sq]):
            if i >= max_cards_per_squad:
                break
            arr[i*card_size:(i+1)*card_size] = encode_card_onehot(card)
        out.append(arr)
    
    return np.concatenate(out)  # 560 dimensioni

def encode_current_player(cp):
    """
    4 dimensioni one-hot
    """
    arr = np.zeros(4, dtype=np.float32)
    arr[cp] = 1.0
    return arr

def encode_move(move):
    """
    Ogni mossa è codificata come:
    - 4 dimensioni per il player (one-hot)
    - 14 dimensioni per la carta giocata (rank+suit one-hot)
    - 3 dimensioni per il tipo di cattura (no_capture=0, capture=1, scopa=2)
    - 10*14 = 140 dimensioni per le carte catturate (fino a 10 carte)
    
    Totale: 4 + 14 + 3 + 140 = 161 dimensioni
    """
    player_dim = 4
    card_dim = 14
    capture_type_dim = 3
    max_captured = 10
    
    out = np.zeros(player_dim + card_dim + capture_type_dim + max_captured * card_dim, dtype=np.float32)
    
    # Player
    p = move["player"]
    out[p] = 1.0
    
    # Carta giocata
    played_card = move["played_card"]
    out[player_dim:player_dim+card_dim] = encode_card_onehot(played_card)
    
    # Tipo di cattura
    capture_map = {"no_capture": 0, "capture": 1, "scopa": 2}
    ctype_idx = capture_map.get(move["capture_type"], 0)
    out[player_dim + card_dim + ctype_idx] = 1.0
    
    # Carte catturate
    base = player_dim + card_dim + capture_type_dim
    for i, card in enumerate(move["captured_cards"]):
        if i >= max_captured:
            break
        out[base + i*card_dim:base + (i+1)*card_dim] = encode_card_onehot(card)
    
    return out

def encode_history(history):
    """
    Massimo 20 mosse, ciascuna 161 dimensioni => 3220
    """
    max_moves = 20
    move_dim = 161
    hist_arr = np.zeros(max_moves * move_dim, dtype=np.float32)
    
    for i, m in enumerate(history):
        if i >= max_moves:
            break
        enc = encode_move(m)
        hist_arr[i*move_dim:(i+1)*move_dim] = enc
    
    return hist_arr

def encode_state_for_player(game_state, player_id):
    """
    Crea un vettore di dimensione fissa (4484) ma OSCURANDO le mani degli altri 3 giocatori.
      - 4 * 140 = 560 per le mani (solo player_id vede la propria, gli altri 0)
      - 140 per tavolo
      - 560 per catture di squadra
      - 4 per current_player
      - 3220 per history (max 20 mosse)

    TOT = 4484
    """
    # 1) Mani
    hands_vec = []
    for p in range(4):
        if p == player_id:
            arr = encode_hand(game_state["hands"][p])
        else:
            # Azzeriamo le mani di tutti tranne p
            arr = np.zeros(140, dtype=np.float32)
        hands_vec.append(arr)
    hands_enc = np.concatenate(hands_vec)  # 560
    
    # 2) Tavolo
    table_enc = encode_table(game_state["table"])  # 140
    
    # 3) Catture squadre
    captured_enc = encode_captured_squads(game_state["captured_squads"])  # 560
    
    # 4) current_player
    cp = game_state.get("current_player", 0)
    cp_enc = encode_current_player(cp)  # 4
    
    # 5) history
    hist_enc = encode_history(game_state["history"])  # 3220
    
    return np.concatenate([hands_enc, table_enc, captured_enc, cp_enc, hist_enc])