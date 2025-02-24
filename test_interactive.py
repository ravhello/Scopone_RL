# test_interactive.py

import numpy as np
from environment import ScoponeEnvMA

# Mappa di conversione rank/suit in simboli brevi per visualizzazione
RANK_SYMBOLS = {
    1: 'A',   # Asso
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',   # spesso fante
    9: '9',   # cavallo
    10:'T'    # re
}

SUIT_SYMBOLS = {
    'denari':  'D',
    'coppe':   'C',
    'spade':   'S',
    'bastoni': 'B'
}

def format_card(card):
    """
    card è una tupla (rank, suit). Restituisce una stringa tipo '7D' (7 denari), 'AS' (asso spade), ecc.
    """
    rank, suit = card
    r_s = RANK_SYMBOLS.get(rank, str(rank))
    s_s = SUIT_SYMBOLS.get(suit, '?')
    return f"{r_s}{s_s}"

def partial_visual_state(env):
    """
    Restituisce una rappresentazione testuale del game_state, 
    come se fossimo 'env.current_player':
      - La mano del giocatore corrente
      - Le mani degli altri giocatori mascherate
      - Le carte sul tavolo
      - Le catture di ogni squadra
      - Eventuali ultime mosse di cronologia
    """
    gs = env.game_state
    cp = env.current_player

    lines = []
    lines.append(f"=== TURNO GIOCATORE {cp} ===")

    # Mostra le mani di tutti, mascherando quelle degli altri
    for p in range(4):
        if p == cp:
            hand_str = " ".join(format_card(c) for c in gs["hands"][p])
            lines.append(f"Mano p{p} (visibile): {hand_str if hand_str else '[vuota]'}")
        else:
            # mascheriamo
            hidden_count = len(gs["hands"][p])
            if hidden_count > 0:
                lines.append(f"Mano p{p} (nascosta): " + "[ ??? ] "*hidden_count)
            else:
                lines.append(f"Mano p{p} (nascosta): [vuota]")

    # Tavolo
    if gs["table"]:
        table_str = " ".join(format_card(c) for c in gs["table"])
    else:
        table_str = "[vuoto]"
    lines.append(f"Tavolo: {table_str}")

    # Catture di squadra
    squad0_str = " ".join(format_card(c) for c in gs["captured_squads"][0])
    squad1_str = " ".join(format_card(c) for c in gs["captured_squads"][1])
    lines.append(f"Catture Squadra0: {squad0_str if squad0_str else '[nessuna]'}")
    lines.append(f"Catture Squadra1: {squad1_str if squad1_str else '[nessuna]'}")

    # Ultime mosse della history (per esempio le ultime 2)
    last_n = 2
    if gs["history"]:
        lines.append("Ultime mosse:")
        for move in gs["history"][-last_n:]:
            pl = move["player"]
            pc = format_card(move["played_card"])
            ctype = move["capture_type"]
            cc = " ".join(format_card(c) for c in move["captured_cards"])
            lines.append(f" - Giocatore {pl} ha giocato {pc}, cattura={ctype}, carte_catturate=[{cc}]")
    else:
        lines.append("Nessuna mossa effettuata finora.")

    return "\n".join(lines)


def test_interactive():
    """
    Script di test interattivo con un'interfaccia testuale più visuale:
     - Mostra stato parziale
     - Elenca azioni valide
     - Chiede input per selezionare un'azione
     - Esegue step finché non termina la partita
    """

    env = ScoponeEnvMA()
    env.reset()

    done = False
    while not done:
        cp = env.current_player
        # Stampa stato parziale
        print()
        print("*"*60)
        print(partial_visual_state(env))
        print("*"*60)

        # Azioni valide
        valid_actions = env.get_valid_actions()
        print(f"\nAzioni valide (tot: {len(valid_actions)}):")
        for i, act_id in enumerate(valid_actions):
            print(f"[{i}] => azioneID={act_id}")

        # Input
        chosen_idx_str = input("Seleziona indice azione (o 'q' per uscire): ")
        if chosen_idx_str.strip().lower() == 'q':
            print("Uscita forzata dall'utente.")
            return
        try:
            chosen_idx = int(chosen_idx_str)
            if chosen_idx < 0 or chosen_idx >= len(valid_actions):
                print("Indice non valido, riprova.")
                continue
        except ValueError:
            print("Input non valido, riprova.")
            continue

        action = valid_actions[chosen_idx]
        # Step
        obs_next, reward, done, info = env.step(action)

        if done:
            team_rewards = info.get("team_rewards", [0.0, 0.0])
            print("\n=== PARTITA TERMINATA ===")
            print(f"Team Rewards finali: {team_rewards}")
            # Stampiamo lo stato finale (completo) a scopo di debug
            print(partial_visual_state(env))
            break

    print("Fine del test interattivo.")


if __name__ == "__main__":
    test_interactive()
