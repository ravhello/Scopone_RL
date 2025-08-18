# game_logic.py

import torch
from actions import decode_action_ids
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

def update_game_state(game_state, action_id, current_player, rules=None):
    """
    Ora accetta un'azione in formato matrice (vettore a 80 dimensioni)
    e aggiorna lo stato del gioco di conseguenza.
    """
    squad_id = 0 if current_player in [0,2] else 1

    hand = game_state["hands"][current_player]
    if not hand:
        final_breakdown = compute_final_score_breakdown(game_state, rules=rules)
        final_reward = compute_final_reward_from_breakdown(final_breakdown)
        return game_state, [final_reward[0], final_reward[1]], True, {"final_score": {0: final_breakdown[0]["total"], 1: final_breakdown[1]["total"]}, "score_breakdown": final_breakdown}

    # Helpers per conversioni (ID/tuple) e operazioni su CUDA
    device = torch.device('cuda')
    def to_id(x):
        if isinstance(x, int):
            return int(x)
        r, s = x
        suit_to_col = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
        return (r - 1) * 4 + suit_to_col[s]

    # Decodifica l'azione (sempre in ID)
    played_id, captured_ids = decode_action_ids(action_id)
    # Enforce ID-only
    table = game_state["table"]
    if len(hand) > 0 and not isinstance(hand[0], int):
        game_state["hands"][current_player] = [to_id(c) for c in hand]
        hand = game_state["hands"][current_player]
    if len(table) > 0 and not isinstance(table[0], int):
        game_state["table"] = [to_id(c) for c in table]
        table = game_state["table"]
    game_state["captured_squads"][0] = [to_id(c) for c in game_state["captured_squads"][0]]
    game_state["captured_squads"][1] = [to_id(c) for c in game_state["captured_squads"][1]]
    played_card = int(played_id)
    cards_to_capture = [int(cid) for cid in captured_ids]
    
    # Verifica presenza carta nella mano usando mask su CUDA
    hand_ids_t = torch.as_tensor([to_id(c) for c in hand], dtype=torch.long, device=device)
    if hand_ids_t.numel() == 0 or int((hand_ids_t == int(played_id)).any().item()) == 0:
        raise ValueError(f"La carta {played_card} non è nella mano del giocatore {current_player}")
    
    # Rimuovi la carta dalla mano
    try:
        hand.remove(played_card)
    except ValueError:
        pass

    # Verifica che le carte da catturare siano sul tavolo (mask su CUDA)
    table_ids_t = torch.as_tensor([to_id(c) for c in table], dtype=torch.long, device=device)
    if len(cards_to_capture) > 0:
        caps_t = torch.as_tensor([to_id(c) for c in cards_to_capture], dtype=torch.long, device=device)
        if table_ids_t.numel() == 0 or int((caps_t.unsqueeze(1) == table_ids_t.unsqueeze(0)).all(dim=1).sum().item()) != len(cards_to_capture):
            raise ValueError("Una o più carte da catturare non sono sul tavolo")

    capture_type = "no_capture"
    scopa_flag = False

    if cards_to_capture:
        # Cattura
        for card in cards_to_capture:
            table.remove(card)
        
        # Stato ID-only
        to_add = list(cards_to_capture)
        to_add.append(int(played_id))
        game_state["captured_squads"][squad_id].extend(to_add)
        
        if len(table) == 0:
            scopa_flag = True
        
        capture_type = "capture"
    else:
        # Nessuna cattura: la carta viene messa sul tavolo
        table.append(int(played_card))
        capture_type = "no_capture"

    # Verifica se scopa è valida (se scopa_flag e ci sono ancora carte nelle mani degli altri)
    cards_left = int(torch.as_tensor([len(game_state["hands"][p]) for p in range(4)], dtype=torch.long, device=device).sum().item())
    if scopa_flag and cards_left > 0:
        capture_type = "scopa"
    # se scopa_flag e cards_left==0, era l'ultima giocata => scopa annullata

    move = {
        "player": current_player,
        "played_card": played_card,
        "capture_type": capture_type,
        "captured_cards": cards_to_capture
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

        final_breakdown = compute_final_score_breakdown(game_state, rules=rules)
        final_reward = compute_final_reward_from_breakdown(final_breakdown)
        return game_state, [final_reward[0], final_reward[1]], True, {"final_score": {0: final_breakdown[0]["total"], 1: final_breakdown[1]["total"]}, "score_breakdown": final_breakdown}
    else:
        return game_state, [0.0, 0.0], False, {"last_move": move}