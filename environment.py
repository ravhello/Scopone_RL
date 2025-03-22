# environment.py - VERSIONE OTTIMIZZATA PER PERFORMANCE
import numpy as np
import torch
import gym
from gym import spaces
import time
from functools import lru_cache

from state import initialize_game
from observation import encode_state_for_player
from actions import get_valid_actions, encode_action, decode_action
from line_profiler import LineProfiler, profile, global_profiler

# Recupera il device globale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScoponeEnvMA(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Observation space con la rappresentazione avanzata
        self.observation_space = spaces.Box(low=0, high=1, shape=(10823,), dtype=np.float32)
        
        # Action space
        self.action_space = spaces.MultiBinary(80)
        
        # OTTIMIZZAZIONE: Migliore struttura dati per la cache delle azioni valide
        self._valid_actions_cache = {}
        self._last_state_hash = None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # OTTIMIZZAZIONE: Cache per encode_state_for_player
        self._observation_cache = {}
        
        # Stato del gioco
        self.game_state = None
        self.done = False
        self.current_player = 0
        self.rewards = [0,0]
        
        # Contatori per diagnostica prestazioni
        self._get_obs_time = 0
        self._get_valid_actions_time = 0
        self._step_time = 0
        self._step_count = 0
        
        self.reset()
    
    #@profile
    def get_valid_actions(self):
        """Versione ad alte prestazioni con cache efficiente"""
        start_time = time.time()
        
        # OTTIMIZZAZIONE: Calcola un hash più efficiente dello stato
        # Usiamo solo le carte rilevanti (mano del giocatore e tavolo)
        current_hash = hash(str(sorted(self.game_state["hands"][self.current_player])) + 
                          str(sorted(self.game_state["table"])))
        
        # Verifica cache
        if current_hash == self._last_state_hash and current_hash in self._valid_actions_cache:
            self._cache_hits += 1
            valid_actions = self._valid_actions_cache[current_hash]
        else:
            # Cache miss
            self._cache_misses += 1
            
            # Calcola le azioni valide
            valid_actions = get_valid_actions(
                game_state=self.game_state,
                current_player=self.current_player
            )
            
            # Aggiorna la cache
            self._valid_actions_cache = {current_hash: valid_actions}  # Mantieni la cache piccola
            self._last_state_hash = current_hash
        
        # Aggiorna il tempo di esecuzione
        self._get_valid_actions_time += time.time() - start_time
        
        return valid_actions
    
    def step(self, action_vec):
        """
        Versione ottimizzata di step per ridurre i trasferimenti CPU-GPU
        e migliorare le prestazioni complessive.
        """
        step_start_time = time.time()
        self._step_count += 1
        
        if self.done:
            raise ValueError("Partita già finita: non puoi fare altri step.")
        
        # Decodifica l'azione - nessun trasferimento CPU-GPU qui
        played_card, cards_to_capture = decode_action(action_vec)
        
        # Verifica validità (come prima)
        current_player = self.current_player
        hand = self.game_state["hands"][current_player]
        table = self.game_state["table"]
        
        if played_card not in hand:
            raise ValueError(f"La carta {played_card} non è nella mano del giocatore {current_player}.")
        
        # Verifica carte da catturare
        for c in cards_to_capture:
            if c not in table:
                raise ValueError(f"La carta {c} non si trova sul tavolo; cattura non valida.")
        
        # Verifica regole di cattura
        rank, suit = played_card
        same_rank_cards = [tc for tc in table if tc[0] == rank]
        
        if same_rank_cards:
            # Cattura diretta obbligatoria
            if set(cards_to_capture) != set(same_rank_cards):
                raise ValueError(f"Quando esistono carte di rank uguale, devi catturarle tutte.")
        elif cards_to_capture:
            # Verifica somma
            sum_chosen = sum(c[0] for c in cards_to_capture)
            if sum_chosen != rank:
                raise ValueError(f"La somma delle carte catturate ({sum_chosen}) deve essere uguale al rank ({rank}).")
        
        # OTTIMIZZAZIONE: Esegui l'azione in modo più efficiente
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
        
        # OTTIMIZZAZIONE: Invalida la cache delle osservazioni
        self._observation_cache = {}
        
        # Verifica se la partita è finita
        done = all(len(self.game_state["hands"][p]) == 0 for p in range(4))
        self.done = done
        
        if done:
            # Assegna le carte rimaste sul tavolo
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
            
            # OTTIMIZZAZIONE: Crea uno stato final di zeri senza GPU
            # No need to call get_observation for final state - just return zeros
            obs_shape = self._get_observation(self.current_player).shape
            obs_final = np.zeros(obs_shape, dtype=np.float32)
            
            # Aggiorna il tempo di esecuzione
            self._step_time += time.time() - step_start_time
            
            # Restituisci la ricompensa finale per il team del giocatore corrente
            current_team = 0 if current_player in [0, 2] else 1
            return obs_final, final_reward[current_team], True, info
        else:
            # Passa al prossimo giocatore
            self.current_player = (self.current_player + 1) % 4
            
            # OTTIMIZZAZIONE: Invalida la cache delle azioni valide
            self._last_state_hash = None
            
            # Ottieni la prossima osservazione
            next_obs = self._get_observation(self.current_player)
            
            # Aggiorna il tempo di esecuzione
            self._step_time += time.time() - step_start_time
            
            return next_obs, 0.0, False, {"last_move": move_info}
    
    #@profile
    def _get_observation(self, player_id):
        """
        Versione ottimizzata per performance che utilizza caching aggressivo
        """
        start_time = time.time()
        
        # OTTIMIZZAZIONE: Crea una chiave di cache efficiente
        # Usa solo le informazioni rilevanti
        cache_key = (
            player_id,
            self.current_player,
            str(sorted(self.game_state["hands"][player_id])),
            str(sorted(self.game_state["table"])),
            len(self.game_state["history"]),
            str(sorted(self.game_state["captured_squads"][0])),
            str(sorted(self.game_state["captured_squads"][1]))
        )
        
        # Verifica la cache
        if cache_key in self._observation_cache:
            result = self._observation_cache[cache_key]
        else:
            # Calcola l'osservazione
            result = encode_state_for_player(self.game_state, player_id)
            
            # Salva in cache
            self._observation_cache[cache_key] = result
            
            # Limita la dimensione della cache
            if len(self._observation_cache) > 100:
                # Rimuovi 50 elementi casuali
                import random
                keys_to_remove = random.sample(list(self._observation_cache.keys()), 50)
                for key in keys_to_remove:
                    del self._observation_cache[key]
        
        # Aggiorna il tempo di esecuzione
        self._get_obs_time += time.time() - start_time
        
        return result
    
    def reset(self):
        """Versione ottimizzata di reset"""
        # Reimposta lo stato del gioco
        self.game_state = initialize_game()
        self.done = False
        self.current_player = 0
        self.rewards = [0, 0]
        
        # Reset delle cache
        self._valid_actions_cache = {}
        self._last_state_hash = None
        self._observation_cache = {}
        
        # Ottieni l'osservazione iniziale
        return self._get_observation(self.current_player)
    
    def print_stats(self):
        """Stampa statistiche di performance"""
        if self._step_count > 0:
            print(f"Environment stats after {self._step_count} steps:")
            print(f"  Avg. _get_observation time: {self._get_obs_time / self._step_count * 1000:.2f} ms")
            print(f"  Avg. get_valid_actions time: {self._get_valid_actions_time / self._step_count * 1000:.2f} ms")
            print(f"  Avg. step time: {self._step_time / self._step_count * 1000:.2f} ms")
            
            total_cache = self._cache_hits + self._cache_misses
            if total_cache > 0:
                hit_rate = self._cache_hits / total_cache * 100
                print(f"  Action cache hit rate: {hit_rate:.1f}% ({self._cache_hits}/{total_cache})")