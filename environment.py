# environment.py

import numpy as np
import gym
from gym import spaces

from state import initialize_game
from observation import encode_state_for_player
from actions import get_valid_actions, MAX_ACTIONS
from game_logic import update_game_state

class ScoponeEnvMA(gym.Env):
    """
    Multi-Agent Env a 4 giocatori:
      - Turni: 0 -> 1 -> 2 -> 3 -> 0 -> ...
      - Ognuno vede SOLO la propria mano.
      - Nessuna reward intermedia (ritorniamo [0,0] a ogni mossa).
      - A partita conclusa, restituiamo ([0]*3764, [r0,r1], done=True, info)
        dove r0 e r1 sono le reward dei due team.
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
        if self.done:
            raise ValueError("Partita gia' finita, non puoi fare step aggiuntivo.")

        # Valida
        valids = get_valid_actions(self.game_state, self.current_player)
        if action not in valids:
            raise ValueError(f"Azione {action} non valida. Valide: {valids}")

        # update game
        self.game_state, rw_array, done, info = update_game_state(
            self.game_state, action, self.current_player
        )
        self.done = done
        self.rewards = rw_array  # [r0, r1]

        if done:
            # Partita finita => restituiamo la ricompensa per il team che ha mosso
            obs_final = np.zeros(3764, dtype=np.float32)

            # In info mettiamo anche i punteggi di entrambi i team
            info["team_rewards"] = [rw_array[0], rw_array[1]]

            # Calcoliamo quale team ha mosso
            team_id = 0 if self.current_player in [0,2] else 1
            # Prendiamo la ricompensa finale per quell'agente
            final_reward = rw_array[team_id]

            return obs_final, final_reward, True, info
        else:
            # Non finita => ricompensa 0.0
            self.current_player = (self.current_player + 1) % 4
            next_obs = self._get_observation(self.current_player)
            return next_obs, 0.0, False, info

    def _get_observation(self, player_id):
        return encode_state_for_player(self.game_state, player_id)

    def render(self, mode="human"):
        print("===== SCOPONE, stato attuale =====")
        print("Current player:", self.current_player)
        print("Mani (DEBUG):")
        for p in range(4):
            print(f"  Giocatore {p}: {self.game_state['hands'][p]}")
        print("Tavolo:", self.game_state["table"])
        print("Catture Team0:", self.game_state["captured_squads"][0])
        print("Catture Team1:", self.game_state["captured_squads"][1])
        print("History:")
        for move in self.game_state["history"]:
            print(move)

    def close(self):
        pass
