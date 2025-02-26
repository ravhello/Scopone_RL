from state import initialize_game
from actions import decode_action

def update_game_state(game_state, action_id, current_player):
    """
    Aggiorna lo stato (catture, scopa, ecc.) senza gestire reward o done.
    Restituisce sempre (game_state, [0.0, 0.0], info).
    """
    squad_id = 0 if current_player in [0, 2] else 1
    hand = game_state["hands"][current_player]
    if not hand:
        info = {"note": "empty hand"}
        return game_state, [0.0, 0.0], info

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
        for i in sorted(subset_ids, reverse=True):
            if i < len(table):
                table.pop(i)
        game_state["captured_squads"][squad_id].extend(chosen_cards)
        game_state["captured_squads"][squad_id].append(played_card)
        if len(table) == 0:
            scopa_flag = True
        capture_type = "capture"
    else:
        table.append(played_card)
        capture_type = "no_capture"
    cards_left = sum(len(game_state["hands"][p]) for p in range(4))
    if scopa_flag and cards_left > 0:
        capture_type = "scopa"
    move = {
        "player": current_player,
        "played_card": played_card,
        "capture_type": capture_type,
        "captured_cards": chosen_cards
    }
    game_state["history"].append(move)
    return game_state, [0.0, 0.0], {"last_move": move}
