# environment.py (per osservazione avanzata)

import numpy as np
import gym
from gym import spaces

from state import initialize_game
from observation import encode_state_for_player
from actions import get_valid_actions, encode_action, decode_action
from line_profiler import LineProfiler, profile, global_profiler

class ScoponeEnvMA(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Observation space con la rappresentazione avanzata e lo storico migliorato
        self.observation_space = spaces.Box(low=0, high=1, shape=(10823,), dtype=np.float32)
        
        # Action space usando la rappresentazione a matrice (80 dim)
        self.action_space = spaces.MultiBinary(80)
        
        # Stato base
        self.game_state = None
        self.done = False
        self.current_player = 0
        self.rewards = [0,0]
        
        self.reset()
    
    @profile
    def get_valid_actions(self):
        # Usiamo la funzione che restituisce azioni nel formato matrice
        return get_valid_actions(
            game_state=self.game_state,
            current_player=self.current_player
        )
    
    def step(self, action_vec):
        """
        Esegue un'azione nel formato matrice.
        
        Args:
            action_vec: Array di 80 bit
        """
        if self.done:
            raise ValueError("Partita già finita: non puoi fare altri step.")
        
        # Decodifica l'azione
        played_card, cards_to_capture = decode_action(action_vec)
        
        # Verifica validità dell'azione
        current_player = self.current_player
        hand = self.game_state["hands"][current_player]
        table = self.game_state["table"]
        
        if played_card not in hand:
            raise ValueError(f"La carta {played_card} non è nella mano del giocatore {current_player}.")
        
        # Verifica che le carte da catturare siano sul tavolo
        for c in cards_to_capture:
            if c not in table:
                raise ValueError(f"La carta {c} non si trova sul tavolo; cattura non valida.")
        
        # Verifica regole di cattura
        rank, suit = played_card
        
        # Controlla se c'è una carta di rank uguale sul tavolo
        same_rank_cards = [tc for tc in table if tc[0] == rank]
        
        if same_rank_cards:
            # Se esistono carte di rank uguale, la cattura diretta è obbligatoria
            if set(cards_to_capture) != set(same_rank_cards):
                raise ValueError(f"Quando esistono carte di rank uguale, devi catturarle tutte.")
        elif cards_to_capture:
            # Se non ci sono carte di rank uguale ma si cerca di catturare, la somma deve essere corretta
            sum_chosen = sum(c[0] for c in cards_to_capture)
            if sum_chosen != rank:
                raise ValueError(f"La somma delle carte catturate ({sum_chosen}) deve essere uguale al rank ({rank}).")
        
        # Esegue l'azione
        capture_type = "no_capture"
        
        # Rimuovi la carta giocata dalla mano
        hand.remove(played_card)
        
        if cards_to_capture:
            # Cattura carte
            for c in cards_to_capture:
                table.remove(c)
            
            # Aggiungi le carte catturate e la carta giocata alla squadra
            squad_id = 0 if current_player in [0, 2] else 1
            self.game_state["captured_squads"][squad_id].extend(cards_to_capture)
            self.game_state["captured_squads"][squad_id].append(played_card)
            
            # Verifica scopa
            if len(table) == 0:
                # Verifica che non sia l'ultima giocata
                cards_left = sum(len(self.game_state["hands"][p]) for p in range(4))
                if cards_left > 0:
                    capture_type = "scopa"
                else:
                    capture_type = "capture"
            else:
                capture_type = "capture"
        else:
            # Nessuna cattura: la carta va sul tavolo
            table.append(played_card)
        
        # Aggiorna history
        move_info = {
            "player": current_player,
            "played_card": played_card,
            "capture_type": capture_type,
            "captured_cards": cards_to_capture
        }
        self.game_state["history"].append(move_info)
        
        # Verifica se la partita è finita
        done = all(len(self.game_state["hands"][p]) == 0 for p in range(4))
        self.done = done
        
        if done:
            # Assegna le carte rimaste sul tavolo alla squadra dell'ultima presa
            if self.game_state["table"]:
                last_capturing_team = None
                for m in reversed(self.game_state["history"]):
                    if m["capture_type"] in ["capture", "scopa"]:
                        last_capturing_team = 0 if m["player"] in [0, 2] else 1
                        break
                
                if last_capturing_team is not None:
                    self.game_state["captured_squads"][last_capturing_team].extend(self.game_state["table"])
                    self.game_state["table"].clear()
            
            # Calcolo punteggio finale
            from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown
            final_breakdown = compute_final_score_breakdown(self.game_state)
            final_reward = compute_final_reward_from_breakdown(final_breakdown)
            
            info = {
                "score_breakdown": final_breakdown,
                "team_rewards": [final_reward[0], final_reward[1]]
            }
            
            # Reset dell'osservazione per lo stato terminale
            obs_final = np.zeros_like(self._get_observation(self.current_player))
            
            # Restituisci la ricompensa finale per il team del giocatore corrente
            current_team = 0 if current_player in [0, 2] else 1
            return obs_final, final_reward[current_team], True, info
        else:
            # Passa al prossimo giocatore
            self.current_player = (self.current_player + 1) % 4
            next_obs = self._get_observation(self.current_player)
            return next_obs, 0.0, False, {"last_move": move_info}
    
    @profile
    def _get_observation(self, player_id):
        """
        Gets the observation for the specified player.
        
        Args:
            player_id: The ID of the player (0-3)
            
        Returns:
            Observation vector as a numpy array
        """
        return encode_state_for_player(self.game_state, player_id)
    
    def reset(self):
        """
        Resets the environment to a new initial state.
        
        Returns:
            Initial observation
        """
        self.game_state = initialize_game()
        self.done = False
        self.current_player = 0
        self.rewards = [0, 0]
        return self._get_observation(self.current_player)