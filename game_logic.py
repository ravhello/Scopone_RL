# game_logic.py

from state import initialize_game
from actions import decode_action
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

def update_game_state(game_state, action_id, current_player):
    """
    Applica l'azione al game_state, aggiornandone mani, tavolo, catture, ecc.
    1) Decodifica (hand_index, subset_table)
    2) Gioca la carta e, se la somma dei rank coincide, cattura.
    3) Se tavolo vuoto e non è l'ultima carta => scopa, altrimenti no.
    4) Se tutti i giocatori hanno 0 carte => done=True e calcolo punteggi + reward differenziale.
    5) Ritorna (game_state, [r0,r1], done, info).
       - [r0,r1] = [0,0] se la partita non è finita.
       - [r0,r1] = finalReward se finita.
    """
    squad_id = 0 if current_player in [0,2] else 1
    other_squad = 1 - squad_id

    hand = game_state["hands"][current_player]
    if not hand:
        # Nessuna carta => potrebbe essere finita? 
        # Ma se lo state è inconsistente, restituiamo done=True
        final_score = compute_final_score(game_state)
        final_reward = compute_final_reward(final_score)
        # Ritorno reward = [final_reward[0], final_reward[1]]
        return game_state, [final_reward[0], final_reward[1]], True, {"message":"No cards left."}

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
        # Nessuna cattura
        table.append(played_card)
        capture_type = "no_capture"

    # Controlliamo se scopa è valida (se scopa_flag e restano carte in mano agli altri).
    cards_left = sum(len(game_state["hands"][p]) for p in range(4))
    if scopa_flag and cards_left > 0:
        capture_type = "scopa"
    # se scopa_flag e cards_left==0 => ultima carta => scopa annullata

    move = {
        "player": current_player,
        "played_card": played_card,
        "capture_type": capture_type,
        "captured_cards": chosen_cards
    }
    game_state["history"].append(move)

    done = all(len(game_state["hands"][p])==0 for p in range(4))
    if done:
        breakdown = compute_final_score_breakdown(game_state)
        final_reward = compute_final_reward_from_breakdown(breakdown)
        # Ritorniamo un array di 2 reward: r0 e r1
        info = {"final_score": {0: breakdown[0]["total"], 1: breakdown[1]["total"]},
                "score_breakdown": breakdown}
        return game_state, [final_reward[0], final_reward[1]], True, info
    else:
        # Non è finita => reward = [0,0]
        return game_state, [0.0, 0.0], False, {"last_move": move}
