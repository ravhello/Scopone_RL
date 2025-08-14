# actions.py
import itertools
import numpy as np

def encode_action(card, cards_to_capture):
    """
    Codifica un'azione utilizzando la rappresentazione a matrice:
    - Carta giocata: matrice 10x4 con 1 solo bit attivo (40 dim)
    - Carte catturate: matrice 10x4 con i bit attivi corrispondenti (40 dim)
    
    Totale: 80 dimensioni
    """
    # Inizializza le matrici
    played_card_matrix = np.zeros((10, 4), dtype=np.float32)
    captured_cards_matrix = np.zeros((10, 4), dtype=np.float32)
    
    # Mappatura dei semi agli indici di colonna
    suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    
    # Codifica la carta giocata
    rank, suit = card
    row = rank - 1
    col = suit_to_col[suit]
    played_card_matrix[row, col] = 1.0
    
    # Codifica le carte catturate
    for capt_card in cards_to_capture:
        capt_rank, capt_suit = capt_card
        capt_row = capt_rank - 1
        capt_col = suit_to_col[capt_suit]
        captured_cards_matrix[capt_row, capt_col] = 1.0
    
    # Appiattisci le matrici e concatenale
    return np.concatenate([played_card_matrix.flatten(), captured_cards_matrix.flatten()])

def decode_action(action_vec):
    """
    Decodifica un vettore di azione nella rappresentazione a matrice.
    
    Args:
        action_vec: Numpy array di dimensione 80
    
    Returns:
        Tupla (card, cards_to_capture)

    Raises:
        ValueError: se nessun bit è attivo per la carta giocata
    """
    # Separa la carta giocata e le carte catturate
    played_card_flat = action_vec[:40]
    captured_cards_flat = action_vec[40:]
    
    # Reshapa in matrici 10x4
    played_card_matrix = played_card_flat.reshape(10, 4)
    captured_cards_matrix = captured_cards_flat.reshape(10, 4)
    
    # Mappatura degli indici di colonna ai semi
    col_to_suit = {0: 'denari', 1: 'coppe', 2: 'spade', 3: 'bastoni'}
    
    # Trova la carta giocata (l'unico bit attivo nella matrice)
    if np.max(played_card_matrix) > 0:
        row, col = np.unravel_index(np.argmax(played_card_matrix), played_card_matrix.shape)
        rank = row + 1  # +1 perché rank è 1-indexed
        suit = col_to_suit[col]
        played_card = (rank, suit)
    else:
        # In precedenza veniva restituita una carta di default.
        # Ora solleviamo un'eccezione per segnalare un'azione non valida.
        raise ValueError("Nessuna carta giocata codificata nell'azione")
    
    # Trova le carte catturate (tutti i bit attivi nella matrice)
    cards_to_capture = []
    for row in range(10):
        for col in range(4):
            if captured_cards_matrix[row, col] > 0:
                rank = row + 1
                suit = col_to_suit[col]
                cards_to_capture.append((rank, suit))
    
    return played_card, cards_to_capture

def get_valid_actions(game_state, current_player):
    """
    Restituisce una lista di azioni valide nel formato matrice per il giocatore corrente.
    
    Args:
        game_state: Stato del gioco
        current_player: ID del giocatore corrente
    
    Returns:
        Lista di azioni valide nel formato matrice (80 bit)
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