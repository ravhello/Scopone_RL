"""Bot euristico semplice per Scopone.
Non è ottimale ma fornisce un prior ragionevole per distillazione/valutazione.
"""
from tests.torch_np import np
from typing import List, Tuple


def score_heuristic_action(hand_card: Tuple[int, str], captured: List[Tuple[int, str]], table_after: List[Tuple[int, str]]) -> float:
    score = 0.0
    # preferisci prese (più carte meglio)
    score += 0.2 * len(captured)
    # scopa (tavolo vuoto)
    if len(table_after) == 0 and len(captured) > 0:
        score += 1.0
    # cattura sette/denari
    for r, s in captured:
        if r == 7:
            score += 0.5
        if s == 'denari':
            score += 0.2
        if (r, s) == (7, 'denari'):
            score += 0.5
    # evita regali evidenti: tavolo grande con somma facile
    tsum = sum(c[0] for c in table_after)
    if tsum in [7, 10]:
        score -= 0.2
    return score


def pick_action_heuristic(valid_actions: List):
    """Seleziona l'azione migliore secondo euristica grezza.
    valid_actions: lista di vettori 80-dim (played 40 + captured 40).
    """
    best_score = -1e9
    best = valid_actions[0]
    for a in valid_actions:
        played = a[:40].reshape(10, 4)
        captured = a[40:].reshape(10, 4)
        # decodifica minima
        pr, ps = np.unravel_index(np.argmax(played), (10, 4))
        played_card = (pr + 1, ['denari', 'coppe', 'spade', 'bastoni'][ps])
        captured_cards = []
        for r in range(10):
            for s in range(4):
                if captured[r, s] > 0:
                    captured_cards.append((r + 1, ['denari', 'coppe', 'spade', 'bastoni'][s]))
        # tavolo dopo la mossa sconosciuto qui; penalità grossolana
        score = score_heuristic_action(played_card, captured_cards, table_after=captured_cards and [] or [played_card])
        if score > best_score:
            best_score = score
            best = a
    return best



