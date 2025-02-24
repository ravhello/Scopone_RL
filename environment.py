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
        # Action space = Discrete(512)
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
            # Partita finita => reward finale [r0, r1]
            # Gym in singolo canale di reward => forziamo un'uscita custom.
            # Per compatibilita', ritorniamo (obs_fittizio, <scalar_reward>, done, info)
            # e in info mettiamo "team_rewards".
            # Oppure ritorniamo come "reward=0" e "info['team_rewards']=[r0,r1]".
            # Qui facciamo la 2a opzione, cosi' l'utente puo' leggerli.
            obs_final = np.zeros(3764, dtype=np.float32)
            info["team_rewards"] = [rw_array[0], rw_array[1]]
            return obs_final, 0.0, True, info
        else:
            # Non finita => reward=[0,0]
            # Passa al prossimo player
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
