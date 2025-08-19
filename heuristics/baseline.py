"""Bot euristico semplice per Scopone.
Non è ottimale ma fornisce un prior ragionevole per distillazione/valutazione.
"""
from tests.torch_np import np
from typing import List, Tuple
import torch


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


def pick_action_heuristic(valid_actions):
    """Seleziona l'azione migliore secondo euristica grezza.
    valid_actions: tensore (K,80) CUDA oppure lista di vettori 80-dim.
    """
    best_score = -1e9
    if torch.is_tensor(valid_actions):
        best = valid_actions[0]
        K = valid_actions.size(0)
        for i in range(K):
            a = valid_actions[i]
            played = a[:40].reshape(10, 4)
            captured = a[40:].reshape(10, 4)
            # Remain on CUDA: compute argmax with torch and unravel indices
            flat_idx_t = torch.argmax(played.reshape(-1))
            pr = (flat_idx_t // 4).cpu().item()  # Converti solo il risultato finale
            ps = (flat_idx_t % 4).cpu().item()
            played_card = (pr + 1, ['denari', 'coppe', 'spade', 'bastoni'][ps])
            captured_cards = []
            for r in range(10):
                for s in range(4):
                    if captured[r, s] > 0:
                        captured_cards.append((r + 1, ['denari', 'coppe', 'spade', 'bastoni'][s]))
            score = score_heuristic_action(played_card, captured_cards, table_after=captured_cards and [] or [played_card])
            if score > best_score:
                best_score = score
                best = a
        return best
    # Fallback list path (tests/UI)
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



