# state.py
import random

SUITS = ['denari', 'coppe', 'spade', 'bastoni']
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def card_to_id(card):
    rank, suit = card
    suit_to_int = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    return (rank - 1) * 4 + suit_to_int[suit]

def create_deck():
    """
    Crea e restituisce un mazzo di 40 carte come ID 0..39.
    """
    deck = list(range(40))
    return deck

def initialize_game(rules=None):
    """
    Inizializza lo stato 'completo' del gioco:
      - Default: 4 giocatori, 10 carte ciascuno, tavolo vuoto (Scopone scientifico)
      - Variante opzionale (scopone non scientifico): 4 carte scoperte sul tavolo e 9 carte a testa
      - captured_squads -> {0:[], 1:[]}
      - history -> lista di mosse
    """
    rules = rules or {}
    # Abilita la variante se esplicitata via flag o alias di variant
    start_with_4_on_table = bool(rules.get("start_with_4_on_table", False))
    variant = rules.get("variant")
    if isinstance(variant, str) and variant.lower() in ("scopone_non_scientifico", "non_scientifico", "scopone-non-scientifico"):
        start_with_4_on_table = True

    deck = create_deck()
    random.shuffle(deck)

    if start_with_4_on_table:
        # 4 carte scoperte sul tavolo, 9 carte ciascuno (36 carte in mano totali)
        # Regola "a monte": se all'apertura ci sono 3 o 4 Re (rank=10) sul tavolo, si ridistribuisce
        attempts = 0
        while True:
            random.shuffle(deck)
            table_cards = deck[:4]
            kings_on_table = sum(1 for cid in table_cards if (cid // 4 + 1) == 10)
            if kings_on_table < 3:
                break
            attempts += 1
            if attempts > 2000:
                # fallback di sicurezza: accetta la distribuzione corrente per evitare loop infiniti
                break

        start_idx = 4
        hands = {}
        for i in range(4):
            hands[i] = deck[start_idx + i*9 : start_idx + (i+1)*9]
        table = list(table_cards)
    else:
        # Comportamento standard: 10 carte a testa, tavolo vuoto
        hands = {}
        for i in range(4):
            hands[i] = deck[i*10 : (i+1)*10]
        table = []

    state = {
        "hands": hands,               # dict: {0: [ids], 1: [ids], 2: [ids], 3: [ids]}
        "table": table,               # ids sul tavolo
        "captured_squads": {0:[], 1:[]}, 
        "history": []
    }
    return state
