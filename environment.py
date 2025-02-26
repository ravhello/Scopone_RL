import numpy as np
import gym
from gym import spaces

from state import initialize_game
from observation import encode_state_for_player
from actions import get_valid_actions, MAX_ACTIONS
from game_logic import update_game_state

class ScoponeEnvMA(gym.Env):
    """
    Ambiente semplificato per Scopone.
    Non gestisce lo stato "done" né la reward finale;
    lo step restituisce sempre reward 0.0.
    Le 40 mosse totali sono gestite esternamente.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(3764,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.game_state = None
        self.current_player = 0
        self.reset()

    def get_valid_actions(self):
        return get_valid_actions(self.game_state, self.current_player)

    def reset(self):
        self.game_state = initialize_game()
        self.current_player = 0
        return self._get_observation(self.current_player)

    def step(self, action):
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            raise ValueError(f"Azione {action} non valida. Valide: {valid_actions}")
        # Aggiorna lo stato, ignorando reward/done
        self.game_state, _, info = update_game_state(self.game_state, action, self.current_player)
        self.current_player = (self.current_player + 1) % 4
        next_obs = self._get_observation(self.current_player)
        return next_obs, 0.0, info

    def _get_observation(self, player_id):
        return encode_state_for_player(self.game_state, player_id)

    def render(self, mode="human"):
        print("=== Stato Ambiente Scopone ===")
        print("Current player:", self.current_player)
        print("Game state:", self.game_state)

    def close(self):
        pass
