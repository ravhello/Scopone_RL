# actions.py
import torch
from functools import lru_cache

MAX_ACTIONS = 2048

@lru_cache(maxsize=None)
def precompute_all_subsets(table_tuple):
    """
    Precompute per *tutti* i rank (1..10) le combinazioni di indici che sommano quel rank.
    table_tuple: una tupla dei rank sul tavolo (e.g. (2,5,7,1,...))
                 L'indice di ogni elemento in table_tuple corrisponde all'indice di carta sul tavolo.
    
    Restituisce: 
      un dizionario { rank : [lista di tuple con gli indici del tavolo], ... }
      rank è un intero 1..10
    """
    results = {r: [] for r in range(1, 11)}
    table = list(table_tuple)

    def backtrack(start, current_indices, current_sum):
        if 1 <= current_sum <= 10:
            results[current_sum].append(tuple(current_indices))
        if current_sum >= 10:
            return

        for i in range(start, len(table)):
            new_sum = current_sum + table[i]
            if new_sum > 10:
                continue
            backtrack(i + 1, current_indices + [i], new_sum)

    backtrack(0, [], 0)
    return results


def get_valid_actions(game_state, current_player, device=None):
    """
    Restituisce un tensore con tutte le azioni valide per il current_player,
    usando una precomputazione globale per trovare i sottoinsiemi di indici che sommano il rank.
    
    Se device è specificato (es: torch.device('cuda')),
    allora il tensore finale verrà creato su quella device;
    altrimenti, rimane su CPU.
    """
    if device is None:
        device = torch.device("cpu")

    hand = game_state["hands"][current_player]
    table = game_state["table"]

    table_tuple = tuple(card[0] for card in table)
    all_subsets = precompute_all_subsets(table_tuple)

    valid_actions_list = []
    for h_i, card in enumerate(hand):
        rank = card[0]
        # 1) Cattura diretta (stesso rank)
        same_rank_indices = [idx for idx, t_c in enumerate(table) if t_c[0] == rank]
        if same_rank_indices:
            action_id = encode_action(h_i, same_rank_indices)
            if action_id < MAX_ACTIONS:
                valid_actions_list.append(action_id)
        else:
            # 2) Sottoinsiemi con somma == rank
            possible_subsets = all_subsets.get(rank, [])
            if possible_subsets:
                for subset_indices in possible_subsets:
                    action_id = encode_action(h_i, subset_indices)
                    if action_id < MAX_ACTIONS:
                        valid_actions_list.append(action_id)
            else:
                # 3) Gioca carta senza cattura
                action_id = encode_action(h_i, ())
                if action_id < MAX_ACTIONS:
                    valid_actions_list.append(action_id)

    # Creiamo il tensore su 'device' specificato
    return torch.tensor(valid_actions_list, dtype=torch.long, device=device)


def encode_action(hand_index, subset_indices):
    """
    Mappa (hand_index, subset_indices) -> intero < MAX_ACTIONS.
    subset_indices è una lista/tupla di indici (nel tavolo) che intendiamo catturare.
    """
    action_id = (hand_index & 0xF)
    bitmask = 0
    for s in subset_indices:
        bitmask |= (1 << s)
    action_id |= (bitmask << 4)
    return action_id


def decode_action(action_id):
    """
    Decodifica l'intero in (hand_index, subset_indices).
    """
    hand_index = action_id & 0xF
    bitmask = action_id >> 4
    subset = []
    i = 0
    while bitmask > 0:
        if (bitmask & 1) == 1:
            subset.append(i)
        i += 1
        bitmask >>= 1
    return hand_index, tuple(subset)
