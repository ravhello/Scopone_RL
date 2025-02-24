# actions.py
import itertools

MAX_ACTIONS = 512

def get_valid_actions(game_state, current_player):
    cp = current_player
    hand = game_state["hands"][cp]
    table = game_state["table"]
    
    valid_actions = []
    for h_i, card in enumerate(hand):
        rank = card[0]

        # 1) Verifica se sul tavolo c'è almeno una carta di rank uguale
        same_rank_indices = [i for i, t_c in enumerate(table) if t_c[0] == rank]
        if same_rank_indices:
            # REGOLA: se esiste cattura diretta, è obbligata.
            # => Aggiungiamo SOLO l'azione di cattura diretta.
            #    Niente somme, niente "gioca senza catturare".
            action_id = encode_action(h_i, same_rank_indices)
            if action_id < MAX_ACTIONS:
                valid_actions.append(action_id)
        
        else:
            # 2) NON c'è cattura diretta => cerchiamo possibili somme
            sum_options = []
            idx_range = range(len(table))
            for size_ in range(1, len(table)+1):
                for subset in itertools.combinations(idx_range, size_):
                    chosen_cards = [table[x] for x in subset]
                    if sum(c[0] for c in chosen_cards) == rank:
                        sum_options.append(subset)
            
            if sum_options:
                # Se ci sono combinazioni di somma, dobbiamo catturare con una di esse
                for subset in sum_options:
                    action_id = encode_action(h_i, subset)
                    if action_id < MAX_ACTIONS:
                        valid_actions.append(action_id)
            else:
                # 3) Se non c'è cattura diretta e non c'è cattura per somma,
                #    allora è possibile buttare la carta senza catturare.
                action_id = encode_action(h_i, ())
                if action_id < MAX_ACTIONS:
                    valid_actions.append(action_id)

    return valid_actions


def encode_action(hand_index, subset_indices):
    """
    Rimane invariato: mappa (hand_index, bitmask subset tavolo) in un intero < MAX_ACTIONS.
    """
    action_id = (hand_index & 0xF)
    bitmask = 0
    for s in subset_indices:
        bitmask |= (1<<s)
    action_id |= (bitmask << 4)
    return action_id

def decode_action(action_id):
    """
    Decodifica l'intero in (hand_index, subset_indices).
    """
    hand_index = action_id & 0xF
    bitmask = action_id>>4
    subset=[]
    i=0
    while bitmask>0:
        if (bitmask &1)==1:
            subset.append(i)
        i+=1
        bitmask >>=1
    return hand_index, tuple(subset)
