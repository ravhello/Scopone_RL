# actions.py
import itertools
import numpy as np
from observation import encode_card_onehot

def encode_action(card, cards_to_capture):
    """
    Codifica un'azione come vettore:
    - 14 bit per la carta giocata (10 per rank, 4 per suit)
    - 10 * 14 = 140 bit per le carte da catturare (fino a 10 carte)
    
    Totale: 14 + 140 = 154 dimensioni
    """
    max_cards_to_capture = 10
    card_dim = 14  # 10 per rank + 4 per suit
    
    action_vec = np.zeros(card_dim + max_cards_to_capture * card_dim, dtype=np.float32)
    
    # Encoding della carta giocata (one-hot)
    action_vec[:card_dim] = encode_card_onehot(card)
    
    # Encoding delle carte da catturare (one-hot per ciascuna)
    for i, capture_card in enumerate(cards_to_capture):
        if i >= max_cards_to_capture:
            break
        base = card_dim + i * card_dim
        action_vec[base:base+card_dim] = encode_card_onehot(capture_card)
    
    return action_vec

def decode_action(action_vec):
    """
    Decodifica un vettore di azione in carta giocata e carte catturate.
    
    Args:
        action_vec: Numpy array
    
    Returns:
        Tupla (card, cards_to_capture) dove:
            - card: Tupla (rank, suit) della carta giocata
            - cards_to_capture: Lista di tuple (rank, suit) delle carte catturate
    """
    # Decodifica la carta giocata
    card_dim = 14
    
    # Rank (primi 10 bit)
    rank_hot = action_vec[0:10]
    if np.max(rank_hot) > 0:  # Verifica che ci sia almeno un bit attivo
        rank = np.argmax(rank_hot) + 1  # +1 perché rank è 1-indexed
    else:
        rank = 0  # Valore di default se nessun bit è attivo
    
    # Suit (successivi 4 bit)
    suit_hot = action_vec[10:14]
    if np.max(suit_hot) > 0:
        suit_idx = np.argmax(suit_hot)
        suits = ['denari', 'coppe', 'spade', 'bastoni']
        suit = suits[suit_idx]
    else:
        suit = 'denari'  # Valore di default
    
    played_card = (rank, suit)
    
    # Decodifica le carte da catturare
    cards_to_capture = []
    max_cards = 10
    
    for i in range(max_cards):
        base = card_dim + i * card_dim
        
        # Verifica se questa posizione ha una carta (almeno un bit attivo)
        if np.max(action_vec[base:base+card_dim]) > 0:
            # Decodifica rank
            c_rank_hot = action_vec[base:base+10]
            c_rank = np.argmax(c_rank_hot) + 1
            
            # Decodifica suit
            c_suit_hot = action_vec[base+10:base+card_dim]
            c_suit_idx = np.argmax(c_suit_hot)
            c_suit = suits[c_suit_idx]
            
            cards_to_capture.append((c_rank, c_suit))
    
    return played_card, cards_to_capture

def get_valid_actions(game_state, current_player):
    """
    Restituisce una lista di azioni valide nel formato one-hot per il giocatore corrente.
    
    Args:
        game_state: Stato del gioco
        current_player: ID del giocatore corrente
    
    Returns:
        Lista di azioni valide nel formato one-hot (154 bit)
    """
    cp = current_player
    hand = game_state["hands"][cp]
    table = game_state["table"]
    
    valid_actions = []
    
    # Per ogni carta nella mano del giocatore
    for card in hand:
        rank, suit = card
        
        # 1) Verifica se sul tavolo c'è almeno una carta di rank uguale (cattura diretta)
        same_rank_cards = [t_c for t_c in table if t_c[0] == rank]
        if same_rank_cards:
            # Cattura diretta obbligatoria
            action_vec = encode_action(card, same_rank_cards)
            valid_actions.append(action_vec)
        else:
            # 2) Cerca possibili somme
            captures_found = False
            for subset_size in range(1, len(table) + 1):
                for subset in itertools.combinations(table, subset_size):
                    if sum(c[0] for c in subset) == rank:
                        # Cattura per somma
                        action_vec = encode_action(card, list(subset))
                        valid_actions.append(action_vec)
                        captures_found = True
            
            # 3) Se non ci sono catture possibili, la carta viene buttata
            if not captures_found:
                action_vec = encode_action(card, [])
                valid_actions.append(action_vec)
    
    return valid_actions