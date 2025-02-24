# observation.py
import numpy as np

SUITS_ORDER = ['denari','coppe','spade','bastoni']
RANKS_ORDER = [1,2,3,4,5,6,7,8,9,10]

card_to_index = {}
index_to_card = {}

idx=0
for s in SUITS_ORDER:
    for r in RANKS_ORDER:
        card_to_index[(r,s)] = idx
        index_to_card[idx] = (r,s)
        idx += 1

def encode_hand(hand):
    """
    Converte una mano in un vettore binario di dimensione 40.
    """
    vec = np.zeros(40, dtype=np.float32)
    for c in hand:
        i = card_to_index[c]
        vec[i] = 1.0
    return vec

def encode_table(table):
    """
    Converte il tavolo in 40 dimensioni binarie.
    """
    vec = np.zeros(40, dtype=np.float32)
    for c in table:
        i = card_to_index[c]
        vec[i] = 1.0
    return vec

def encode_captured_squads(captured_squads):
    """
    Concatena 2 vettori binari (ciascuno di dimensione 40) per le 2 squadre.
    """
    out = []
    for sq in [0,1]:
        arr = np.zeros(40, dtype=np.float32)
        for c in captured_squads[sq]:
            i = card_to_index[c]
            arr[i] = 1.0
        out.append(arr)
    return np.concatenate(out)  # 80 dimensioni

def encode_current_player(cp):
    """
    4 dimensioni one-hot
    """
    arr = np.zeros(4, dtype=np.float32)
    arr[cp] = 1.0
    return arr

def encode_move(move):
    """
    Ogni mossa in 87 dimensioni:
      - 0..3 => player
      - 4..43 => played_card
      - 44..46 => capture_type (no_capture=0, capture=1, scopa=2)
      - 47..86 => captured_cards
    """
    out = np.zeros(87, dtype=np.float32)
    p = move["player"]
    out[p] = 1.0

    played_idx = card_to_index[move["played_card"]]
    out[4 + played_idx] = 1.0

    capture_map = {"no_capture":0, "capture":1, "scopa":2}
    ctype_idx = capture_map.get(move["capture_type"], 0)
    out[4+40+ctype_idx] = 1.0

    base = 4+40+3  # 47
    for c in move["captured_cards"]:
        out[base + card_to_index[c]] = 1.0

    return out

def encode_history(history):
    """
    Massimo 40 mosse, ciascuna 87 dimensioni => 3480
    """
    max_moves = 40
    msize = 87
    hist_arr = np.zeros(max_moves * msize, dtype=np.float32)
    for i,m in enumerate(history):
        if i>=max_moves:
            break
        enc = encode_move(m)
        hist_arr[i*msize : (i+1)*msize] = enc
    return hist_arr

def encode_state_for_player(game_state, player_id):
    """
    Crea un vettore di dimensione fissa (3764) ma OSCURANDO le mani degli altri 3 giocatori.
      - 4 x 40 = 160 per le mani (solo player_id vede la propria, gli altri 0).
      - 40 per tavolo
      - 80 per catture di squadra
      - 4 per current_player
      - 3480 per history

    TOT = 3764
    """
    # 1) Mani
    hands_vec=[]
    for p in range(4):
        if p == player_id:
            arr = encode_hand(game_state["hands"][p])
        else:
            # Azzeriamo le mani di tutti tranne p
            arr = np.zeros(40, dtype=np.float32)
        hands_vec.append(arr)
    hands_enc = np.concatenate(hands_vec)  # 160

    # 2) Tavolo
    table_enc = encode_table(game_state["table"])  # 40

    # 3) Catture squadre
    captured_enc = encode_captured_squads(game_state["captured_squads"])  # 80

    # 4) current_player
    # (se p=player_id è uguale a current_player => indica tocca a me)
    cp = game_state.get("current_player", 0)
    cp_enc = encode_current_player(cp)  # 4

    # 5) history (40 x 87 = 3480)
    hist_enc = encode_history(game_state["history"])

    return np.concatenate([hands_enc, table_enc, captured_enc, cp_enc, hist_enc])
