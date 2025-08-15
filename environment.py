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
    def __init__(self, rules=None):
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
        
        # Regole/varianti opzionali della partita
        # Esempi di chiavi supportate:
        #  - asso_piglia_tutto: bool
        #  - scopa_on_asso_piglia_tutto: bool
        #  - scopa_on_last_capture: bool
        #  - re_bello: bool
        #  - napola: bool
        #  - napola_scoring: "fixed3" | "length"
        #  - max_consecutive_scope: int | None (limite per team)
        self.rules = rules or {}
        
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

            # Variante: Asso piglia tutto → aggiungi azioni extra (asso cattura tutto il tavolo)
            try:
                if self.rules.get("asso_piglia_tutto", False):
                    hand_cards = self.game_state["hands"].get(self.current_player, [])
                    table_cards = self.game_state.get("table", [])
                    if table_cards:
                        for card in hand_cards:
                            if card[0] == 1:  # Asso
                                extra = encode_action(card, list(table_cards))
                                # Evita duplicati
                                exists = False
                                for v in valid_actions:
                                    try:
                                        # v è un np.array
                                        if (extra == v).all():
                                            exists = True
                                            break
                                    except Exception:
                                        pass
                                if not exists:
                                    valid_actions.append(extra)

                    # Nuova regola: posabilità dell'asso piglia tutto
                    # Se posabile è consentito, aggiungi anche l'azione di "posa" (nessuna presa)
                    ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
                    ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
                    if ap_posabile:
                        for card in hand_cards:
                            if card[0] == 1:
                                can_place_now = True if not ap_only_empty else (len(table_cards) == 0)
                                if can_place_now:
                                    # Verifica se esiste già l'azione di posa (cattura vuota)
                                    place_vec = encode_action(card, [])
                                    exists = False
                                    for v in valid_actions:
                                        try:
                                            if (place_vec == v).all():
                                                exists = True
                                                break
                                        except Exception:
                                            pass
                                    if not exists:
                                        valid_actions.append(place_vec)

                    # Se AP è attivo ma la posa non è consentita nelle condizioni attuali,
                    # rimuovi eventuali azioni di "posa" dell'asso (no-capture).
                    # Eccezione importante: a tavolo vuoto manteniamo l'azione "asso + []"
                    # per rappresentare (lato UI) la presa forzata su tavolo vuoto.
                    allow_place_now = ap_posabile and (not ap_only_empty or (len(table_cards) == 0))
                    if not allow_place_now and len(table_cards) > 0:
                        filtered = []
                        for v in valid_actions:
                            keep = True
                            try:
                                pc, cc = decode_action(v)
                                if pc[0] == 1 and len(cc) == 0:
                                    # è una posa di asso → vietata ora
                                    keep = False
                            except ValueError:
                                # Azione non decodificabile -> scartala
                                keep = False
                            if keep:
                                filtered.append(v)
                        valid_actions = filtered
            except Exception:
                # In caso di qualunque errore, non interrompere il flusso
                pass
            
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
        try:
            played_card, cards_to_capture = decode_action(action_vec)
        except ValueError as e:
            raise ValueError(f"Azione non valida: {e}")
        
        # Verifica validità (come prima)
        current_player = self.current_player
        hand = self.game_state["hands"][current_player]
        table = self.game_state["table"]
        # Snapshot length of table before modifications for scopa logic nuances
        pre_table_len = len(table)
        
        if played_card not in hand:
            raise ValueError(f"La carta {played_card} non è nella mano del giocatore {current_player}.")
        
        # Verifica carte da catturare
        for c in cards_to_capture:
            if c not in table:
                raise ValueError(f"La carta {c} non si trova sul tavolo; cattura non valida.")

        # Applicazione nuova regola AP posabilità: forza presa totale se non è consentito posare
        rank, suit = played_card
        ap_enabled = bool(self.rules.get("asso_piglia_tutto", False))
        ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
        ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
        forced_ace_capture_on_empty = False
        # Override one-shot: la UI può impostare questo flag per auto-presa su tavolo vuoto anche se posabile è ON
        force_self_capture_once = bool(self.rules.get("force_ace_self_capture_on_empty_once", False))
        if ap_enabled and rank == 1:
            can_place_now = ap_posabile and (not ap_only_empty or (ap_only_empty and len(table) == 0))
            if (not can_place_now and len(cards_to_capture) == 0) or (force_self_capture_once and len(table) == 0 and len(cards_to_capture) == 0):
                if len(table) > 0:
                    # Forza presa di tutto il tavolo
                    cards_to_capture = list(table)
                else:
                    # Tavolo vuoto: la posa è vietata, tratta come cattura forzata per scopa
                    forced_ace_capture_on_empty = True
                # Consuma l'override one-shot, se presente
                if force_self_capture_once:
                    try:
                        self.rules["force_ace_self_capture_on_empty_once"] = False
                    except Exception:
                        pass
        
        # Verifica regole di cattura
        same_rank_cards = [tc for tc in table if tc[0] == rank]
        
        if same_rank_cards:
            # Eccezione: Asso piglia tutto permette di ignorare la regola della presa diretta
            ace_take_all = (rank == 1 and self.rules.get("asso_piglia_tutto", False) 
                            and set(cards_to_capture) == set(table))
            if not ace_take_all:
                # Nuova regola: devi catturare UNA carta di pari rank (non una combinazione né sottoinsiemi di somma)
                if not (len(cards_to_capture) == 1 and cards_to_capture[0] in same_rank_cards):
                    raise ValueError("Quando esistono carte di rank uguale, devi catturarne una (non una combinazione).")
        elif cards_to_capture:
            # Verifica somma
            # Eccezione: Asso piglia tutto
            if not (rank == 1 and self.rules.get("asso_piglia_tutto", False) and set(cards_to_capture) == set(table)):
                sum_chosen = sum(c[0] for c in cards_to_capture)
                if sum_chosen != rank:
                    raise ValueError(f"La somma delle carte catturate ({sum_chosen}) deve essere uguale al rank ({rank}).")
        
        # OTTIMIZZAZIONE: Esegui l'azione in modo più efficiente
        capture_type = "no_capture"
        
        # Rimuovi la carta giocata dalla mano
        hand.remove(played_card)
        
        if forced_ace_capture_on_empty:
            # Cattura forzata su tavolo vuoto: conta come scopa (o cattura se disabilitata via opzione)
            squad_id = 0 if current_player in [0, 2] else 1
            self.game_state["captured_squads"][squad_id].append(played_card)
            # Scopa se non è l'ultima carta giocata
            cards_left = sum(len(self.game_state["hands"][p]) for p in range(4))
            if cards_left > 0:
                capture_type = "scopa"
            else:
                capture_type = "scopa" if self.rules.get("scopa_on_last_capture", False) else "capture"
        elif cards_to_capture:
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
                    # Scopa all'ultima presa: opzionale
                    capture_type = "scopa" if self.rules.get("scopa_on_last_capture", False) else "capture"
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
        # Eccezione: Asso piglia tutto non conta scopa (a meno di opzione esplicita)
        # Tuttavia, se sul tavolo c'era solo un asso ed è stato preso (anche usando AP),
        # deve comunque contare come scopa come nella presa diretta normale.
        if (move_info["capture_type"] == "scopa" and rank == 1 and 
            self.rules.get("asso_piglia_tutto", False) and not self.rules.get("scopa_on_asso_piglia_tutto", False)):
            try:
                # Se prima della mossa c'era una sola carta sul tavolo ed era un asso,
                # manteniamo la scopa.
                single_ace_sweep = (pre_table_len == 1 and len(cards_to_capture) == 1 and cards_to_capture[0][0] == 1)
            except Exception:
                single_ace_sweep = False
            if not single_ace_sweep:
                move_info["capture_type"] = "capture"

        # Limite scope consecutive per team
        if move_info["capture_type"] == "scopa":
            limit = self.rules.get("max_consecutive_scope")
            if isinstance(limit, int) and limit > 0:
                team_id = 0 if current_player in [0, 2] else 1
                consecutive = 0
                for m in reversed(self.game_state["history"]):
                    if m.get("capture_type") == "scopa":
                        prev_team = 0 if m.get("player") in [0, 2] else 1
                        if prev_team == team_id:
                            consecutive += 1
                        else:
                            break
                    else:
                        break
                if consecutive >= limit:
                    move_info["capture_type"] = "capture"

        self.game_state["history"].append(move_info)
        
        # OTTIMIZZAZIONE: Invalida la cache delle osservazioni
        self._observation_cache = {}
        
        # Verifica se la partita è finita
        done = all(len(self.game_state["hands"][p]) == 0 for p in range(4))
        self.done = done
        
        if done:
            # Assegna le carte rimaste sul tavolo (opzionale)
            if self.game_state["table"]:
                if self.rules.get("last_cards_to_dealer", True):
                    last_capturing_team = None
                    for m in reversed(self.game_state["history"]):
                        if m["capture_type"] in ["capture", "scopa"]:
                            last_capturing_team = 0 if m["player"] in [0, 2] else 1
                            break
                    
                    if last_capturing_team is not None:
                        self.game_state["captured_squads"][last_capturing_team].extend(self.game_state["table"])
                # In ogni caso svuota il tavolo a fine mano
                self.game_state["table"].clear()
            
            # Calcolo punteggio finale
            from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown
            final_breakdown = compute_final_score_breakdown(self.game_state, rules=self.rules)
            final_reward = compute_final_reward_from_breakdown(final_breakdown)
            
            info = {
                "score_breakdown": final_breakdown,
                "team_rewards": [final_reward[0], final_reward[1]]
            }
            
            # OTTIMIZZAZIONE: Crea uno stato finale di zeri senza GPU
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
    
    def reset(self, starting_player=None):
        """Versione ottimizzata di reset"""
        # Reimposta lo stato del gioco rispettando le regole/varianti
        self.game_state = initialize_game(rules=self.rules)
        self.done = False
        self.current_player = starting_player if starting_player is not None else 0
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