import numpy as np
import gym
from gym import spaces

from state import initialize_game
from observation import encode_state_for_player
from actions import get_valid_actions, decode_action, MAX_ACTIONS
from game_logic import compute_final_score_breakdown_from_replay, decode_action_id

class ScoponeEnvMA(gym.Env):
    """
    Multi-Agent Env a 4 giocatori:
      - Turni: 0 -> 1 -> 2 -> 3 -> 0 -> ...
      - Ognuno vede SOLO la propria mano.
      - Nessuna reward intermedia (ritorniamo [0,0] a ogni mossa).
      - A partita conclusa, restituiamo (osservazione_fittizia, 0, done=True, info)
        con info["team_rewards"] = [r0, r1].
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        # Observation space = 3764 dimensioni
        self.observation_space = spaces.Box(low=0, high=1, shape=(3764,), dtype=np.float32)
        # Action space = Discrete(2048)
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        
        # Stato base
        self.game_state = None
        self.done = False
        self.current_player = 0
        self.rewards = [0,0]  # ultima reward calcolata

        self.reset()
        
    def get_valid_actions(self):
        return get_valid_actions(
            game_state=self.game_state,
            current_player=self.current_player
        )
    
    def reset(self):
        self.game_state = initialize_game()
        self.done = False
        self.current_player = 0
        self.rewards = [0,0]
        return self._get_observation(self.current_player)

    def step(self, action):
        """
        In step():
          - Verifichiamo che la partita non sia finita.
          - Calcoliamo il 'choice_index' in base all'ordine *decrescente* delle azioni valide.
          - Decodifichiamo la carta giocata.
          - Salviamo in history: (played_card, choice_index).
          - Rimuoviamo la carta dalla mano e, se serve, la mettiamo a terra.
          - Controlliamo se la partita è terminata.
        """
        if self.done:
            raise ValueError("Partita gia' finita, non puoi fare step aggiuntivo.")

        valids = self.get_valid_actions()
        if action not in valids:
            raise ValueError(f"Azione {action} non valida. Valide: {valids}")

        # 1) Calcolo del choice_index
        #    Ordinando le azioni in modo DECRESCENTE come da richiesta:
        sorted_valids = sorted(valids, reverse=True)
        choice_index = sorted_valids.index(action)

        # 2) Decodifichiamo per sapere che carta stiamo giocando (solo per salvare in history)
        hand_index, subset_ids = decode_action_id(action, self.game_state["hands"][self.current_player])
        played_card = self.game_state["hands"][self.current_player][hand_index]

        # 3) Salviamo in history
        move = {
            "played_card": played_card,
            "choice_index": choice_index
        }
        self.game_state["history"].append(move)

        # 4) Rimuoviamo la carta dalla mano e, se c'è cattura, leviamo le carte dal tavolo
        self._apply_action(action)

        # 5) Verifichiamo se la partita è finita
        done = all(len(self.game_state["hands"][p]) == 0 for p in range(4))
        if done:
            # Se ci sono carte sul tavolo, l'assegnazione al team dell'ultima presa
            # avverrà in fase di "replay" (nella ricostruzione).
            final_breakdown = compute_final_score_breakdown_from_replay(self.game_state)
            # Estraggo i totali
            r0 = final_breakdown[0]["total"]
            r1 = final_breakdown[1]["total"]
            # Assegno la reward
            diff = r0 - r1
            rew_0 = diff * 10
            rew_1 = -diff * 10
            self.rewards = [rew_0, rew_1]

            obs_final = np.zeros(3764, dtype=np.float32)
            self.done = True
            info = {
                "team_rewards": [rew_0, rew_1],
                "score_breakdown": final_breakdown
            }
            return obs_final, 0.0, True, info
        else:
            # Non finita
            self.done = False
            self.rewards = [0.0, 0.0]
            self.current_player = (self.current_player + 1) % 4
            next_obs = self._get_observation(self.current_player)
            return next_obs, 0.0, False, {}

    def _apply_action(self, action):
        """
        Elimina la carta giocata dalla mano corrente.
        Esegue la cattura se appropriata (rimuovendo carte dal tavolo).
        Tuttavia, non salviamo nulla su 'captured_squads' nel game_state.
        (Le catture vere e proprie vengono ricostruite a posteriori.)
        """
        cp = self.current_player
        hand = self.game_state["hands"][cp]
        table = self.game_state["table"]

        # decode
        hand_index, subset_ids = decode_action_id(action, hand)
        played_card = hand.pop(hand_index)

        # Sommiamo i rank sul tavolo
        chosen_cards = []
        for i in sorted(subset_ids):
            if i < len(table):
                chosen_cards.append(table[i])
        sum_chosen = sum(c[0] for c in chosen_cards)

        # Se c'è cattura (sum_chosen == played_card[0]), rimuoviamo le carte dal tavolo
        if sum_chosen == played_card[0]:
            for i in sorted(subset_ids, reverse=True):
                if i < len(table):
                    table.pop(i)
            # Non salviamo la cattura da nessuna parte
        else:
            # Buttiamo la carta sul tavolo
            table.append(played_card)

    def _get_observation(self, player_id):
        return encode_state_for_player(self.game_state, player_id)

    def render(self, mode="human"):
        """
        Ricostruiamo l'intera partita (replay) per scoprire:
         - Chi ha preso cosa
         - Quante scope sono state fatte
         - Chi si è preso le ultime carte sul tavolo
         - Punteggio finale

        Poi stampiamo un resoconto dettagliato.
        """
        from game_logic import reconstruct_entire_game
        final_info = reconstruct_entire_game(self.game_state)

        print("===== SCOPONE, stato attuale (ricostruito) =====")
        print(f"Mani residue attuali: (non significative a fine partita) ")
        for p in range(4):
            print(f"  Giocatore {p}: {self.game_state['hands'][p]}")

        print(f"\n== REPLAY COMPLETO ==")
        for line in final_info["log"]:
            print(line)

        print("\n== ESITO FINALE ==")
        print(f"Carte catturate Team0: {final_info['captured_squads'][0]}")
        print(f"Carte catturate Team1: {final_info['captured_squads'][1]}")
        print(f"Scope Team0: {final_info['breakdown'][0]['scope']}")
        print(f"Scope Team1: {final_info['breakdown'][1]['scope']}")
        print(f"Totale Team0: {final_info['breakdown'][0]['total']}")
        print(f"Totale Team1: {final_info['breakdown'][1]['total']}")
        print(f"Ricompense finali: Team0={final_info['rewards'][0]}, Team1={final_info['rewards'][1]}")

    def close(self):
        pass
