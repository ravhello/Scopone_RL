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

def encode_current_player(cp):
    """
    4 dimensioni one-hot
    """
    arr = np.zeros(4, dtype=np.float32)
    arr[cp] = 1.0
    return arr

def encode_history_stub(history):
    """
    Non salviamo più nulla di player/catture. Se vuoi,
    puoi codificare semplicemente le 'played_card' e 'choice_index'.
    
    Esempio: 2 campi per mossa => (card_one_hot 40) + (choice_index) -> TOT 41 per mossa.
    Se vogliamo max 40 mosse => 1640 dimensioni.
    """
    max_moves = 40
    single_move_size = 41  # 40 per la carta + 1 per choice_index
    hist_arr = np.zeros(max_moves * single_move_size, dtype=np.float32)

    for i, move in enumerate(history):
        if i >= max_moves:
            break
        played_card = move["played_card"]
        choice_idx = move["choice_index"]

        # 1) Encode la carta
        card_enc = np.zeros(40, dtype=np.float32)
        cidx = card_to_index[played_card]
        card_enc[cidx] = 1.0

        # 2) Insert nel vettore
        base = i*single_move_size
        hist_arr[base : base+40] = card_enc
        hist_arr[base+40] = float(choice_idx)

    return hist_arr

def encode_state_for_player(game_state, player_id):
    """
    Esempio di encoding:
      - 4 x 40 = 160 per le mani (solo la propria mano, altrimenti 0)
      - 40 per tavolo
      - 4 per current_player
      - 1640 per la history
    TOT = 160 + 40 + 4 + 1640 = 1844
    (Se vuoi arrivare a 3764, puoi aggiungere zone "vuote" finché non raggiungi la dimensione desiderata.)
    """
    # 1) Mani
    hands_vec=[]
    for p in range(4):
        if p == player_id:
            arr = encode_hand(game_state["hands"][p])
        else:
            arr = np.zeros(40, dtype=np.float32)
        hands_vec.append(arr)
    hands_enc = np.concatenate(hands_vec)  # 160

    # 2) Tavolo
    table_enc = encode_table(game_state["table"])  # 40

    # 3) Current player
    cp_enc = encode_current_player(player_id)  # 4

    # 4) History ridotta
    hist_enc = encode_history_stub(game_state["history"])  # 1640

    # Metti tutto insieme
    full_enc = np.concatenate([hands_enc, table_enc, cp_enc, hist_enc])

    # Se vuoi raggiungere esattamente 3764, aggiungi uno zero-pad
    final_size = 3764
    if len(full_enc) < final_size:
        padded = np.zeros(final_size, dtype=np.float32)
        padded[:len(full_enc)] = full_enc
        return padded
    else:
        return full_enc
