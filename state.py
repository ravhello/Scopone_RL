# state.py
import random

SUITS = ['denari', 'coppe', 'spade', 'bastoni']
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def create_deck():
    """
    Crea e restituisce un mazzo di 40 carte (rank, suit).
    """
    deck = [(r, s) for s in SUITS for r in RANKS]
    return deck

def initialize_game():
    """
    Inizializza lo stato 'completo' del gioco:
      - 4 giocatori, 10 carte ciascuno
      - Tavolo vuoto
      - captured_squads -> {0:[], 1:[]}
      - history -> lista di mosse
    """
    deck = create_deck()
    random.shuffle(deck)

    hands = {}
    for i in range(4):
        hands[i] = deck[i*10 : (i+1)*10]

    state = {
        "hands": hands,               # dict: {0: [...], 1: [...], 2: [...], 3: [...]}
        "table": [],                  # carte sul tavolo
        "captured_squads": {0:[], 1:[]}, 
        "history": []
    }
    return state
