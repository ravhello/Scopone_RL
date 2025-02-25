from actions import decode_action_id
from state import create_deck, SUITS, RANKS

def compute_final_score_breakdown_from_replay(game_state):
    """
    Rigioca completamente la partita sulla base di history (che contiene
    'played_card' e 'choice_index' per ogni mossa) e calcola la valutazione finale.
    Restituisce un breakdown simile a:
      {
         0: {..., "total": ...},
         1: {..., "total": ...}
      }
    """
    result = reconstruct_entire_game(game_state)
    return result["breakdown"]

def reconstruct_entire_game(game_state):
    """
    Ricostruisce l'intera sequenza di mosse basandosi su:
       - le mani iniziali (già in game_state)
       - la table iniziale (vuota)
       - la history, che contiene solo (played_card, choice_index) in quell'ordine
         e l'ordine di turno è 0,1,2,3,0,1,2,3,... di volta in volta.

    Torna un dizionario con:
      {
        "captured_squads": {0: [...], 1: [...]},
        "breakdown": {
            0: {"carte":..., "denari":..., "settebello":..., "scope":..., "primiera":..., "total":...},
            1: {...}
        },
        "rewards": (rew0, rew1),
        "log": [stringhe descrittive di ogni mossa]
      }

    N.B.: L'ultima presa e le eventuali scope vengono calcolate in modo identico
          a una rigiocata “vera”.
    """
    # Copie locali (deep) delle mani e del tavolo, perché le manipoleremo
    import copy
    hands_local = copy.deepcopy(game_state["hands"])
    table_local = copy.deepcopy(game_state["table"])  # Dovrebbe essere vuoto a inizio

    # Qui salveremo le catture
    captured_squads = {0: [], 1: []}
    # Log descrittivo delle mosse
    replay_log = []

    num_moves = len(game_state["history"])
    current_player = 0
    last_capture_team = None

    for i, move in enumerate(game_state["history"]):
        # 'move' ha "played_card" e "choice_index"
        played_card = move["played_card"]
        choice_index = move["choice_index"]
        squad_id = 0 if current_player in [0,2] else 1

        # Cerchiamo tutte le azioni valide attuali (con la mano attuale)
        valid_actions = _replay_get_valid_actions(hands_local[current_player], table_local)
        # Ordiniamo DECRESCENTE e prendiamo l'azione in base a choice_index
        valid_actions_sorted = sorted(valid_actions, reverse=True)

        if choice_index < 0 or choice_index >= len(valid_actions_sorted):
            # Errore di coerenza
            replay_log.append(f"ERRORE: choice_index fuori range {choice_index}")
            break

        chosen_action = valid_actions_sorted[choice_index]
        # Decodifichiamo chosen_action
        h_i, subset = decode_action_id(chosen_action, hands_local[current_player])
        actual_card_in_hand = hands_local[current_player][h_i]

        # Verifichiamo che 'actual_card_in_hand' coincida con 'move["played_card"]'
        if actual_card_in_hand != played_card:
            replay_log.append(f"ERRORE: la carta in mano {actual_card_in_hand} non coincide con {played_card}")
            break

        # Ora applichiamo la mossa
        # 1) Rimuoviamo la carta dalla mano
        used_card = hands_local[current_player].pop(h_i)

        # 2) Verifichiamo se c'è cattura
        chosen_cards = []
        for idx in sorted(subset):
            if idx < len(table_local):
                chosen_cards.append(table_local[idx])
        sum_chosen = sum(c[0] for c in chosen_cards)

        capture_type = "no_capture"
        scopa_flag = False

        if sum_chosen == used_card[0]:
            # C'è cattura
            capture_type = "capture"
            # Rimuoviamo dal tavolo
            for idx in sorted(subset, reverse=True):
                table_local.pop(idx)
            # Mettiamo le carte catturate nel corrispondente team
            captured_squads[squad_id].extend(chosen_cards)
            captured_squads[squad_id].append(used_card)
            # Controllo scopa
            if len(table_local) == 0:
                # E' scopa
                capture_type = "scopa"
                scopa_flag = True
            last_capture_team = squad_id
        else:
            # Buttiamo la carta a terra
            table_local.append(used_card)

        # Logghiamo
        replay_log.append(f"Mossa {i}: Player {current_player} gioca {used_card}, choice_index={choice_index}, "
                          f"{capture_type} con {chosen_cards}")

        # Avanti prossimo giocatore
        current_player = (current_player + 1) % 4

    # Finita la rigiocata di tutte le mosse
    # Se ci sono carte rimaste sul tavolo, vanno all'ultima squadra che ha preso
    if len(table_local) > 0 and last_capture_team is not None:
        captured_squads[last_capture_team].extend(table_local)
        replay_log.append(f"Le ultime carte sul tavolo vanno al team {last_capture_team}: {table_local}")
        table_local.clear()

    # A questo punto calcoliamo i punteggi
    breakdown = _compute_final_score(captured_squads, replay_log)
    diff = breakdown[0]["total"] - breakdown[1]["total"]
    rew_0 = diff * 10
    rew_1 = -diff * 10

    return {
        "captured_squads": captured_squads,
        "breakdown": breakdown,
        "rewards": (rew_0, rew_1),
        "log": replay_log
    }


def _replay_get_valid_actions(hand, table):
    """
    Durante il replay, rigeneriamo la logica di get_valid_actions
    basandoci su 'hand' e 'table' (senza l'intero game_state).
    """
    import itertools
    valid = []
    for h_i, card in enumerate(hand):
        rank = card[0]
        # 1) cattura diretta se esiste
        same_rank_indices = [i for i,t_c in enumerate(table) if t_c[0] == rank]
        if same_rank_indices:
            action_id = encode_action(h_i, same_rank_indices)
            valid.append(action_id)
        else:
            # 2) combinazioni di somma
            sum_options = []
            idx_range = range(len(table))
            for size_ in range(1, len(table)+1):
                for subset in itertools.combinations(idx_range, size_):
                    chosen_cards = [table[x] for x in subset]
                    if sum(c[0] for c in chosen_cards) == rank:
                        sum_options.append(subset)
            if sum_options:
                for subset in sum_options:
                    action_id = encode_action(h_i, subset)
                    valid.append(action_id)
            else:
                # 3) butta la carta
                action_id = encode_action(h_i, ())
                valid.append(action_id)
    return valid

def encode_action(hand_index, subset_indices):
    """
    Stessa identica codifica di actions.py
    """
    action_id = (hand_index & 0xF)
    bitmask = 0
    for s in subset_indices:
        bitmask |= (1 << s)
    action_id |= (bitmask << 4)
    return action_id

def _compute_final_score(captured_squads, replay_log):
    """
    Calcola i punti finali (carte, denari, settebello, primiera, scope)
    e restituisce breakdown = {0: {...}, 1: {...}}
    """
    # Conteggio scope
    # Nella nostra ricostruzione (replay_log) ci sono righe tipo:
    # "Mossa i: Player x gioca (rank,suit), choice_index=..., capture con [ .. ]"
    # se compare 'scopa' nel testo, incrementiamo. (Oppure si poteva salvare un flag.)
    scope0 = 0
    scope1 = 0
    for line in replay_log:
        if "scopa" in line:
            # scopa -> prendi player
            # es: "Mossa 3: Player 0 gioca (7, 'denari'), choice_index=.., scopa con [(1, 'coppe'),..]"
            import re
            # cerco "Player <num>" e scopro se 0,1,2,3
            match = re.search(r"Player (\d+)", line)
            if match:
                pl = int(match.group(1))
                squad_id = 0 if pl in [0,2] else 1
                if squad_id == 0:
                    scope0 += 1
                else:
                    scope1 += 1

    # Contiamo le carte
    c0 = len(captured_squads[0])
    c1 = len(captured_squads[1])
    pt_c0, pt_c1 = (1,0) if c0>c1 else (0,1) if c1>c0 else (0,0)

    # Denari
    den0 = sum(1 for c in captured_squads[0] if c[1] == 'denari')
    den1 = sum(1 for c in captured_squads[1] if c[1] == 'denari')
    pt_d0, pt_d1 = (1,0) if den0>den1 else (0,1) if den1>den0 else (0,0)

    # Settebello
    sb0 = 1 if (7,'denari') in captured_squads[0] else 0
    sb1 = 1 if (7,'denari') in captured_squads[1] else 0

    # Primiera
    val_map = {1:16,2:12,3:13,4:14,5:15,6:18,7:21,8:10,9:10,10:10}
    best0 = {"denari":0,"coppe":0,"spade":0,"bastoni":0}
    best1 = {"denari":0,"coppe":0,"spade":0,"bastoni":0}
    for (r,s) in captured_squads[0]:
        v = val_map[r]
        if v>best0[s]:
            best0[s] = v
    for (r,s) in captured_squads[1]:
        v = val_map[r]
        if v>best1[s]:
            best1[s] = v
    prim0 = sum(best0.values())
    prim1 = sum(best1.values())
    pt_p0, pt_p1 = (1,0) if prim0>prim1 else (0,1) if prim1>prim0 else (0,0)

    total0 = pt_c0 + pt_d0 + sb0 + pt_p0 + scope0
    total1 = pt_c1 + pt_d1 + sb1 + pt_p1 + scope1

    breakdown = {
      0: {
         "carte": pt_c0,
         "denari": pt_d0,
         "settebello": sb0,
         "primiera": pt_p0,
         "scope": scope0,
         "total": total0
      },
      1: {
         "carte": pt_c1,
         "denari": pt_d1,
         "settebello": sb1,
         "primiera": pt_p1,
         "scope": scope1,
         "total": total1
      }
    }
    return breakdown
