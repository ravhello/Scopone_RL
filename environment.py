# environment.py - VERSIONE OTTIMIZZATA PER PERFORMANCE
import torch
import gym
from gym import spaces
import time
from functools import lru_cache
from collections import OrderedDict

from state import initialize_game
from observation import encode_state_for_player, encode_state_compact_for_player
from actions import get_valid_actions, encode_action, decode_action, decode_action_ids
# line_profiler opzionale: disabilitato in test/produzione

# Recupera il device globale (forzato a CUDA)
device = torch.device("cuda")

def _suit_to_int(suit):
    if suit == 'denari':
        return 0
    if suit == 'coppe':
        return 1
    if suit == 'spade':
        return 2
    return 3  # 'bastoni'

def _card_to_id(card):
    # Accetta sia tuple (rank, suit) sia ID int
    if isinstance(card, int):
        return int(card)
    rank, suit = card
    return (rank - 1) * 4 + _suit_to_int(suit)

def _id_to_card(cid):
    # Legacy helper (evitare nuovo uso): mantieni per compat GUI
    rank = cid // 4 + 1
    suit = ['denari', 'coppe', 'spade', 'bastoni'][cid % 4]
    return (rank, suit)

def _ids_to_bitset(ids):
    bits = 0
    for cid in ids:
        bits |= (1 << cid)
    return bits

class ScoponeEnvMA(gym.Env):
    def __init__(self, rules=None, use_compact_obs: bool = False, k_history: int = 12):
        super().__init__()
        
        # Config osservazione
        self.use_compact_obs = bool(use_compact_obs)
        self.k_history = int(k_history)
        if self.use_compact_obs:
            # Dimensione compatta: 43 + 40 + 82 + 61*k + 40 + 120 + 8 + 2 + 1 + 2 + 1 + 10 + 150
            obs_dim = 43 + 40 + 82 + 61 * self.k_history + 40 + 120 + 8 + 2 + 1 + 2 + 1 + 10 + 150
        else:
            obs_dim = 10823
        # Observation space con la rappresentazione selezionata
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,))
        
        # Action space
        self.action_space = spaces.MultiBinary(80)
        
        # OTTIMIZZAZIONE: Migliore struttura dati per la cache delle azioni valide (LRU)
        self._valid_actions_cache = OrderedDict()
        self._last_state_hash = None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # OTTIMIZZAZIONE: Cache per encode_state_for_player (LRU)
        self._observation_cache = OrderedDict()
        self._cache_capacity = 128
        
        # Stato del gioco
        self.game_state = None  # usa rappresentazione a tuple (rank,suit) per compat, ma prevedi futura migrazione a ID/bitset
        self.done = False
        self.current_player = 0
        self.rewards = [0,0]
        # Cache ID/bitset (parziale)
        self._use_id_cache = True
        self._use_bitset = True
        self._hands_ids = {p: [] for p in range(4)}
        self._table_ids = []
        self._hands_bits = {p: 0 for p in range(4)}
        self._table_bits = 0
        
        # Regole/varianti opzionali della partita
        # Nota: negli script con modalità di default, "asso_piglia_tutto" è disattivato
        # e quindi ignorato (cioè non viene applicata la variante AP).
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
    def _rebuild_id_caches(self):
        if not self._use_id_cache:
            return
        for p in range(4):
            ids = [_card_to_id(c) for c in self.game_state['hands'][p]]
            self._hands_ids[p] = ids
            self._hands_bits[p] = _ids_to_bitset(ids) if self._use_bitset else 0
        tids = [_card_to_id(c) for c in self.game_state['table']]
        self._table_ids = tids
        self._table_bits = _ids_to_bitset(tids) if self._use_bitset else 0

    
    def clone(self):
        """Crea una copia profonda dell'ambiente per simulazione/ricerca."""
        import copy
        cloned = ScoponeEnvMA(rules=copy.deepcopy(self.rules), use_compact_obs=self.use_compact_obs, k_history=self.k_history)
        cloned.game_state = copy.deepcopy(self.game_state)
        cloned.done = self.done
        cloned.current_player = self.current_player
        cloned.rewards = list(self.rewards)
        # reset cache per sicurezza
        cloned._valid_actions_cache = OrderedDict()
        cloned._observation_cache = OrderedDict()
        cloned._last_state_hash = None
        return cloned
    #@profile
    def get_valid_actions(self):
        """Versione ad alte prestazioni con cache efficiente"""
        start_time = time.time()
        
        # OTTIMIZZAZIONE: Calcola una chiave cache efficiente basata su ID numerici
        # usa cache ID se disponibile
        if isinstance(self.game_state["hands"][self.current_player][0], int):
            hand_ids = tuple(sorted(self.game_state["hands"][self.current_player]))
            table_ids = tuple(sorted(self.game_state["table"]))
        else:
            if self._use_id_cache and self._hands_ids[self.current_player] is not None:
                hand_ids = tuple(sorted(self._hands_ids[self.current_player]))
                table_ids = tuple(sorted(self._table_ids))
            else:
                hand_ids = tuple(sorted(_card_to_id(c) for c in self.game_state["hands"][self.current_player]))
                table_ids = tuple(sorted(_card_to_id(c) for c in self.game_state["table"]))
        current_hash = (hand_ids, table_ids)
        
        # Verifica cache
        if current_hash == self._last_state_hash and current_hash in self._valid_actions_cache:
            self._cache_hits += 1
            valid_actions = self._valid_actions_cache[current_hash]
            # LRU refresh
            self._valid_actions_cache.move_to_end(current_hash)
        else:
            # Cache miss
            self._cache_misses += 1
            
            # Calcola le azioni valide (fast path con ID/bitset)
            try:
                valid_actions = []
                table_ids_list = list(table_ids)

                # helper: trova subset che sommano a target
                from actions import find_sum_subsets_ids, encode_action

                for pid in hand_ids:
                    prank = pid // 4 + 1
                    # pari-rank sul tavolo
                    direct_ids = [cid for cid in table_ids_list if (cid // 4 + 1) == prank]
                    if direct_ids:
                        for did in direct_ids:
                            valid_actions.append(encode_action(pid, [did]))
                    else:
                        # somma
                        subs = find_sum_subsets_ids(table_ids_list, prank)
                        if subs:
                            for sub in subs:
                                valid_actions.append(encode_action(pid, list(sub)))
                        else:
                            # scarto
                            valid_actions.append(encode_action(pid, []))

                # Variante: Asso piglia tutto
                if self.rules.get("asso_piglia_tutto", False) and len(table_ids_list) > 0:
                    for pid in hand_ids:
                        if (pid // 4 + 1) == 1:
                            extra = encode_action(pid, list(table_ids_list))
                            valid_actions.append(extra)
            except Exception:
                # Fallback alla versione originale
                valid_actions = get_valid_actions(
                    game_state=self.game_state,
                    current_player=self.current_player
                )

            # Variante: Asso piglia tutto → aggiungi azioni extra (asso cattura tutto il tavolo)
            try:
                if self.rules.get("asso_piglia_tutto", False):
                    # Chiavi compatte per deduplicare in O(1) evitando decode per confronto
                    def _action_key(vec):
                        try:
                            import torch as _torch
                            if _torch.is_tensor(vec):
                                nz = _torch.nonzero(vec > 0, as_tuple=False).flatten().tolist()
                                return tuple(int(i) for i in nz)
                            else:
                                nz = torch.nonzero(torch.as_tensor(vec, dtype=torch.float32), as_tuple=False).flatten()
                                return tuple(int(i.item()) for i in nz)
                        except Exception:
                            return None
                    existing_keys = set()
                    for v in valid_actions:
                        k = _action_key(v)
                        if k is not None:
                            existing_keys.add(k)

                    hand_cards = self.game_state["hands"].get(self.current_player, [])
                    table_cards = self.game_state.get("table", [])
                    if table_cards:
                        for card in hand_cards:
                            cid = _card_to_id(card)
                            if ((cid // 4) + 1) == 1:  # Asso
                                extra = encode_action(cid, list(map(_card_to_id, table_cards)))
                                # Evita duplicati con chiave rapida
                                extra_key = _action_key(extra)
                                if extra_key is None or extra_key not in existing_keys:
                                    valid_actions.append(extra)
                                    if extra_key is not None:
                                        existing_keys.add(extra_key)

                    # Nuova regola: posabilità dell'asso piglia tutto
                    # Se posabile è consentito, aggiungi anche l'azione di "posa" (nessuna presa)
                    ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
                    ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
                    if ap_posabile:
                        for card in hand_cards:
                            cid = _card_to_id(card)
                            if ((cid // 4) + 1) == 1:
                                can_place_now = True if not ap_only_empty else (len(table_cards) == 0)
                                if can_place_now:
                                    # Verifica se esiste già l'azione di posa (cattura vuota)
                                    place_vec = encode_action(cid, [])
                                    place_key = _action_key(place_vec)
                                    if place_key is None or place_key not in existing_keys:
                                        valid_actions.append(place_vec)
                                        if place_key is not None:
                                            existing_keys.add(place_key)

                    # Se AP è attivo ma la posa non è consentita nelle condizioni attuali,
                    # rimuovi eventuali azioni di "posa" dell'asso (no-capture).
                    # Eccezione importante: a tavolo vuoto manteniamo l'azione "asso + []"
                    # per rappresentare (lato UI) la presa forzata su tavolo vuoto.
                    allow_place_now = ap_posabile and (not ap_only_empty or (len(table_cards) == 0))
                    if not allow_place_now and len(table_cards) > 0:
                        filtered = []
                        for v in valid_actions:
                            try:
                                import torch as _torch
                                if _torch.is_tensor(v):
                                    played = v[:40].reshape(10, 4)
                                    captured = v[40:]
                                    is_ace_play = bool(played[0, :].any().item())
                                    is_no_capture = not bool(captured.any().item())
                                else:
                                    played = v[:40].reshape(10, 4)
                                    captured = v[40:]
                                    is_ace_play = played[0, :].any()
                                    is_no_capture = not torch.any(captured)
                                if is_ace_play and is_no_capture:
                                    continue
                            except Exception:
                                # Non scartare se non interpretabile
                                pass
                            filtered.append(v)
                        valid_actions = filtered
            except Exception:
                # In caso di qualunque errore, non interrompere il flusso
                pass
            
            # Aggiorna la cache
            # LRU insert
            self._valid_actions_cache[current_hash] = valid_actions
            self._valid_actions_cache.move_to_end(current_hash)
            while len(self._valid_actions_cache) > self._cache_capacity:
                self._valid_actions_cache.popitem(last=False)
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
        
        # Validazione forma vettore azione (deve essere 80)
        try:
            import torch as _torch
            if hasattr(action_vec, 'shape') and getattr(action_vec, 'ndim', 1) >= 1:
                act_len = int(action_vec.numel() if _torch.is_tensor(action_vec) else _torch.as_tensor(action_vec).numel())
            else:
                act_len = len(action_vec)
        except Exception:
            act_len = None
        if act_len != 80:
            raise ValueError(f"Formato vettore azione non valido: atteso 80, ricevuto {act_len}")

        # Decodifica l'azione in ID
        pid, cap_ids = decode_action_ids(action_vec)
        
        # Verifica validità (come prima)
        current_player = self.current_player
        hand = self.game_state["hands"][current_player]
        table = self.game_state["table"]
        pre_table_len = len(table)
        if pid not in [_card_to_id(c) for c in hand] and pid not in hand:
            raise ValueError(f"La carta {pid} non è nella mano del giocatore {current_player}.")
        
        # Verifica carte da catturare (ID)
        if self._use_bitset and self._use_id_cache:
            for cid in cap_ids:
                if (self._table_bits >> cid) & 1 == 0:
                    raise ValueError(f"La carta {cid} non si trova sul tavolo; cattura non valida.")
        else:
            table_set = set(table)
            for cid in cap_ids:
                if cid not in table_set:
                    raise ValueError(f"La carta {cid} non si trova sul tavolo; cattura non valida.")

        # Applicazione nuova regola AP posabilità: forza presa totale se non è consentito posare
        rank = pid // 4 + 1
        ap_enabled = bool(self.rules.get("asso_piglia_tutto", False))
        ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
        ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
        forced_ace_capture_on_empty = False
        # Override one-shot: la UI può impostare questo flag per auto-presa su tavolo vuoto anche se posabile è ON
        force_self_capture_once = bool(self.rules.get("force_ace_self_capture_on_empty_once", False))
        if ap_enabled and rank == 1:
            can_place_now = ap_posabile and (not ap_only_empty or (ap_only_empty and len(table) == 0))
            if (not can_place_now and len(cap_ids) == 0) or (force_self_capture_once and len(table) == 0 and len(cap_ids) == 0):
                if len(table) > 0:
                    # Forza presa di tutto il tavolo
                    cap_ids = [_card_to_id(c) for c in table]
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
        # Calcoli su ID per coerenza
        table_ids_current = list(self._table_ids) if self._use_id_cache else [_card_to_id(c) for c in table]
        same_rank_ids = [cid for cid in table_ids_current if ((cid // 4) + 1) == rank]
        if same_rank_ids:
            # Eccezione: Asso piglia tutto permette di ignorare la regola della presa diretta
            ace_take_all = (rank == 1 and self.rules.get("asso_piglia_tutto", False)
                            and set(cap_ids) == set(table_ids_current))
            if not ace_take_all:
                # Devi catturare UNA carta di pari rank
                if not (len(cap_ids) == 1 and (cap_ids[0] in same_rank_ids)):
                    raise ValueError("Quando esistono carte di rank uguale, devi catturarne una (non una combinazione).")
        elif cap_ids:
            # Verifica somma
            # Eccezione: Asso piglia tutto
            if not (rank == 1 and self.rules.get("asso_piglia_tutto", False) and set(cap_ids) == set(table_ids_current)):
                sum_chosen = sum(((cid // 4) + 1) for cid in cap_ids)
                if sum_chosen != rank:
                    raise ValueError(f"La somma delle carte catturate ({sum_chosen}) deve essere uguale al rank ({rank}).")
        
        # OTTIMIZZAZIONE: Esegui l'azione in modo più efficiente
        capture_type = "no_capture"
        
        # Rimuovi la carta giocata dalla mano (ID)
        try:
            hand.remove(pid)
        except ValueError:
            pass
        # Aggiorna cache ID/bitset
        if self._use_id_cache:
            try:
                self._hands_ids[current_player].remove(pid)
            except Exception:
                pass
            if self._use_bitset:
                self._hands_bits[current_player] &= ~(1 << pid)
        
        if forced_ace_capture_on_empty:
            # Cattura forzata su tavolo vuoto: conta come scopa (o cattura se disabilitata via opzione)
            squad_id = 0 if current_player in [0, 2] else 1
            self.game_state["captured_squads"][squad_id].append(pid)
            # Scopa se non è l'ultima carta giocata
            cards_left = sum(len(self.game_state["hands"][p]) for p in range(4))
            if cards_left > 0:
                capture_type = "scopa"
            else:
                capture_type = "scopa" if self.rules.get("scopa_on_last_capture", False) else "capture"
        elif cap_ids:
            # Cattura carte
            for cid in cap_ids:
                try:
                    table.remove(cid)
                except Exception:
                    try:
                        table.remove(_id_to_card(cid))
                    except Exception:
                        pass
                if self._use_id_cache:
                    try:
                        self._table_ids.remove(cid)
                    except Exception:
                        pass
                    if self._use_bitset:
                        self._table_bits &= ~(1 << cid)
            
            # Aggiungi le carte catturate e la carta giocata alla squadra
            squad_id = 0 if current_player in [0, 2] else 1
            self.game_state["captured_squads"][squad_id].extend(cap_ids)
            self.game_state["captured_squads"][squad_id].append(pid)
            
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
            table.append(pid)
            if self._use_id_cache:
                self._table_ids.append(pid)
                if self._use_bitset:
                    self._table_bits |= (1 << pid)
        
        # Calcola reward shaping opzionale
        shaped_reward = 0.0
        if capture_type == "scopa" and bool(self.rules.get("shape_scopa", False)):
            try:
                shaped_reward = float(self.rules.get("scopa_reward", 0.1))
            except Exception:
                shaped_reward = 0.1

        # Aggiorna history
        move_info = {
            "player": current_player,
            "played_card": pid,
            "capture_type": capture_type,
            "captured_cards": list(cap_ids)
        }
        # Eccezione: Asso piglia tutto non conta scopa (a meno di opzione esplicita)
        # Tuttavia, se sul tavolo c'era solo un asso ed è stato preso (anche usando AP),
        # deve comunque contare come scopa come nella presa diretta normale.
        if (move_info["capture_type"] == "scopa" and rank == 1 and 
            self.rules.get("asso_piglia_tutto", False) and not self.rules.get("scopa_on_asso_piglia_tutto", False)):
            try:
                # Se prima della mossa c'era una sola carta sul tavolo ed era un asso,
                # manteniamo la scopa.
                single_ace_sweep = (pre_table_len == 1 and len(cap_ids) == 1 and ((cap_ids[0] // 4) + 1) == 1)
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
        
        # OTTIMIZZAZIONE: Invalida la cache delle osservazioni (LRU)
        self._observation_cache = OrderedDict()
        
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
                # reset cache table
                if self._use_id_cache:
                    self._table_ids.clear()
                    self._table_bits = 0
            
            # Calcolo punteggio finale
            from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown
            final_breakdown = compute_final_score_breakdown(self.game_state, rules=self.rules)
            final_reward = compute_final_reward_from_breakdown(final_breakdown)
            
            info = {
                "score_breakdown": final_breakdown,
                "team_rewards": [final_reward[0], final_reward[1]]
            }
            
            # OTTIMIZZAZIONE: Crea uno stato finale di zeri senza ricalcolo (rimane tensor su CUDA)
            obs_final = torch.zeros(self.observation_space.shape, dtype=torch.float32, device=device)
            
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
            
            return next_obs, shaped_reward, False, {"last_move": move_info}
    
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
            tuple(sorted(_card_to_id(c) for c in self.game_state["hands"][player_id])),
            tuple(sorted(_card_to_id(c) for c in self.game_state["table"])),
            len(self.game_state["history"]),
            tuple(sorted(_card_to_id(c) for c in self.game_state["captured_squads"][0])),
            tuple(sorted(_card_to_id(c) for c in self.game_state["captured_squads"][1])),
            self.use_compact_obs,
            self.k_history
        )
        
        # Verifica la cache
        if cache_key in self._observation_cache:
            result = self._observation_cache[cache_key]
            # LRU refresh
            self._observation_cache.move_to_end(cache_key)
        else:
            # Calcola l'osservazione
            if self.use_compact_obs:
                result = encode_state_compact_for_player(self.game_state, player_id, k_history=self.k_history)
            else:
                result = encode_state_for_player(self.game_state, player_id)
            
            # Salva in cache (LRU)
            self._observation_cache[cache_key] = result
            self._observation_cache.move_to_end(cache_key)
            while len(self._observation_cache) > self._cache_capacity:
                self._observation_cache.popitem(last=False)
        
        # Aggiorna il tempo di esecuzione
        self._get_obs_time += time.time() - start_time
        
        return result
    
    def reset(self, starting_player=None):
        """Versione ottimizzata di reset"""
        # Reimposta lo stato del gioco rispettando le regole/varianti
        self.game_state = initialize_game(rules=self.rules)
        # Mantieni lo stato in ID senza riconversioni
        self.done = False
        self.current_player = starting_player if starting_player is not None else 0
        self.rewards = [0, 0]
        
        # Reset delle cache (LRU)
        self._valid_actions_cache = OrderedDict()
        self._last_state_hash = None
        self._observation_cache = OrderedDict()
        # Ricostruisci cache ID/bitset
        try:
            # se in stato ID
            if isinstance(self.game_state['hands'][0][0], int):
                # popola cache con ID direttamente
                for p in range(4):
                    self._hands_ids[p] = list(self.game_state['hands'][p])
                    self._hands_bits[p] = _ids_to_bitset(self._hands_ids[p]) if self._use_bitset else 0
                self._table_ids = list(self.game_state['table'])
                self._table_bits = _ids_to_bitset(self._table_ids) if self._use_bitset else 0
            else:
                self._rebuild_id_caches()
        except Exception:
            self._rebuild_id_caches()
        
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