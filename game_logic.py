# game_logic.py

from state import initialize_game
from actions import decode_action
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

def update_game_state(game_state, action_id, current_player):
    squad_id = 0 if current_player in [0,2] else 1

    hand = game_state["hands"][current_player]
    if not hand:
        final_breakdown = compute_final_score_breakdown(game_state)
        final_reward = compute_final_reward_from_breakdown(final_breakdown)
        return game_state, [final_reward[0], final_reward[1]], True, {"final_score": {0: final_breakdown[0]["total"], 1: final_breakdown[1]["total"]}, "score_breakdown": final_breakdown}

    hand_index, subset_ids = decode_action(action_id)
    hand_index %= len(hand)
    played_card = hand.pop(hand_index)

    table = game_state["table"]
    chosen_cards = []
    for i in sorted(subset_ids):
        if i < len(table):
            chosen_cards.append(table[i])

    sum_chosen = sum(c[0] for c in chosen_cards)
    capture_type = "no_capture"
    scopa_flag = False

    if sum_chosen == played_card[0]:
        # Cattura
        for i in sorted(subset_ids, reverse=True):
            if i < len(table):
                table.pop(i)
        game_state["captured_squads"][squad_id].extend(chosen_cards)
        game_state["captured_squads"][squad_id].append(played_card)
        if len(table) == 0:
            scopa_flag = True
        capture_type = "capture"
    else:
        # Nessuna cattura: la carta viene messa sul tavolo
        table.append(played_card)
        capture_type = "no_capture"

    # Verifica se scopa è valida (se scopa_flag e ci sono ancora carte nelle mani degli altri)
    cards_left = sum(len(game_state["hands"][p]) for p in range(4))
    if scopa_flag and cards_left > 0:
        capture_type = "scopa"
    # se scopa_flag e cards_left==0, era l'ultima giocata => scopa annullata

    move = {
        "player": current_player,
        "played_card": played_card,
        "capture_type": capture_type,
        "captured_cards": chosen_cards
    }
    game_state["history"].append(move)

    done = all(len(game_state["hands"][p]) == 0 for p in range(4))
    if done:
        # Se ci sono carte rimaste sul tavolo, assegnale alla squadra dell'ultima presa
        if game_state["table"]:
            last_capturing_team = None
            # Cerca nell'ultima parte della history la mossa di presa
            for m in reversed(game_state["history"]):
                if m["capture_type"] in ["capture", "scopa"]:
                    last_capturing_team = 0 if m["player"] in [0,2] else 1
                    break
            if last_capturing_team is not None:
                game_state["captured_squads"][last_capturing_team].extend(game_state["table"])
                game_state["table"].clear()

        final_breakdown = compute_final_score_breakdown(game_state)
        final_reward = compute_final_reward_from_breakdown(final_breakdown)
        return game_state, [final_reward[0], final_reward[1]], True, {"final_score": {0: final_breakdown[0]["total"], 1: final_breakdown[1]["total"]}, "score_breakdown": final_breakdown}
    else:
        return game_state, [0.0, 0.0], False, {"last_move": move}
