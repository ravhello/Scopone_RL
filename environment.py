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
    if isinstance(card, int):
        return int(card)
    rank, suit = card
    return (rank - 1) * 4 + _suit_to_int(suit)

def _id_to_card(cid):
    # Legacy helper (evitare nuovo uso)
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
        # GPU mirrors: int64 bitsets on CUDA to keep state resident on device
        self._hands_bits_t = torch.zeros(4, dtype=torch.int64, device=device)
        self._table_bits_t = torch.zeros((), dtype=torch.int64, device=device)
        self._captured_bits_t = torch.zeros(2, dtype=torch.int64, device=device)
        self._one_i64 = torch.tensor(1, dtype=torch.int64, device=device)
        
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
            # sync CUDA mirror
            try:
                self._hands_bits_t[p] = torch.as_tensor(self._hands_bits[p], dtype=torch.int64, device=device)
            except Exception:
                self._hands_bits_t[p] = torch.tensor(int(self._hands_bits[p]), dtype=torch.int64, device=device)
        tids = [_card_to_id(c) for c in self.game_state['table']]
        self._table_ids = tids
        self._table_bits = _ids_to_bitset(tids) if self._use_bitset else 0
        try:
            self._table_bits_t = torch.as_tensor(self._table_bits, dtype=torch.int64, device=device)
        except Exception:
            self._table_bits_t = torch.tensor(int(self._table_bits), dtype=torch.int64, device=device)
        # captured squads mirror
        try:
            bits0 = _ids_to_bitset([_card_to_id(c) for c in self.game_state['captured_squads'][0]])
            bits1 = _ids_to_bitset([_card_to_id(c) for c in self.game_state['captured_squads'][1]])
        except Exception:
            bits0, bits1 = 0, 0
        self._captured_bits_t[0] = torch.as_tensor(bits0, dtype=torch.int64, device=device)
        self._captured_bits_t[1] = torch.as_tensor(bits1, dtype=torch.int64, device=device)

    
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
        """Calcola azioni valide interamente su GPU usando i mirror a bitset CUDA."""
        start_time = time.time()
        from observation import RANK_OF_ID as _RANK_OF_ID
        from actions import encode_action

        # Estrai ID in mano e sul tavolo da bitset tensor
        ids = torch.arange(40, device=device, dtype=torch.int64)
        hand_mask = ((self._hands_bits_t[self.current_player] >> ids) & 1).bool()
        table_mask = ((self._table_bits_t >> ids) & 1).bool()
        hand_ids_t = ids[hand_mask]
        table_ids_t = ids[table_mask]

        valid_actions = []

        if hand_ids_t.numel() == 0:
            self._get_valid_actions_time += time.time() - start_time
            return valid_actions

        for pid_t in hand_ids_t:
            pid = int(pid_t.item())
            prank = int(_RANK_OF_ID[pid].item())

            # Pari-rank sul tavolo
            if table_ids_t.numel() > 0:
                table_ranks = _RANK_OF_ID[table_ids_t].to(torch.int64)
                direct_sel = (table_ranks == prank)
                direct_ids_t = table_ids_t[direct_sel]
            else:
                direct_ids_t = torch.empty(0, dtype=torch.int64, device=device)

            if direct_ids_t.numel() > 0:
                for did_t in direct_ids_t:
                    valid_actions.append(encode_action(pid, [int(did_t.item())]))
            else:
                # Somma: trova tutti i subset con somma prank (GPU)
                n = int(table_ids_t.numel())
                if n > 0:
                    pos = torch.arange(n, device=device, dtype=torch.int64)
                    ranks_k = _RANK_OF_ID[table_ids_t].to(torch.int64)
                    masks = torch.arange(1, 1 << n, device=device, dtype=torch.int64)
                    sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.int64)
                    sums = (sel * ranks_k.unsqueeze(0)).sum(dim=1)
                    good = (sums == prank).nonzero(as_tuple=False).flatten()
                    if good.numel() > 0:
                        for gi in good:
                            subset_ids = table_ids_t[((masks[gi].unsqueeze(0) >> pos) & 1).bool()].tolist()
                            valid_actions.append(encode_action(pid, subset_ids))
                    else:
                        # Scarto
                        valid_actions.append(encode_action(pid, []))
                else:
                    # Tavolo vuoto: solo scarto
                    valid_actions.append(encode_action(pid, []))

            # Variante: Asso piglia tutto (aggiungi cattura completa)
            if self.rules.get("asso_piglia_tutto", False) and table_ids_t.numel() > 0 and prank == 1:
                valid_actions.append(encode_action(pid, table_ids_t.tolist()))
            # Ace place action if allowed
            ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
            ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
            if prank == 1 and ap_posabile and (not ap_only_empty or table_ids_t.numel() == 0):
                valid_actions.append(encode_action(pid, []))

        # Post-filtri AP: rimuovi posa asso non consentita se richiesto
        ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
        ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
        if not (ap_posabile and (not ap_only_empty or table_ids_t.numel() == 0)) and table_ids_t.numel() > 0:
            filtered = []
            for v in valid_actions:
                try:
                    played = v[:40].reshape(10, 4)
                    captured = v[40:]
                    is_ace_play = bool(played[0, :].any().item())
                    is_no_capture = not bool(captured.any().item())
                    if is_ace_play and is_no_capture:
                        continue
                except Exception:
                    pass
                filtered.append(v)
            valid_actions = filtered

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
        
        # Verifica carte da catturare (ID) interamente via bitset CUDA mirror
        for cid in cap_ids:
            present = int(((self._table_bits_t >> int(cid)) & self._one_i64).item())
            if present == 0:
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
        
        # Verifica regole di cattura su GPU
        ids = torch.arange(40, device=device, dtype=torch.int64)
        table_ids_t = ids[((self._table_bits_t >> ids) & 1).bool()]
        from observation import RANK_OF_ID as _RANK_OF_ID
        same_rank_ids_t = table_ids_t[(_RANK_OF_ID[table_ids_t].to(torch.int64) == int(rank))]
        same_rank_ids = same_rank_ids_t.tolist()
        if same_rank_ids:
            # Eccezione: Asso piglia tutto permette di ignorare la regola della presa diretta
            ace_take_all = (rank == 1 and self.rules.get("asso_piglia_tutto", False)
                            and set(cap_ids) == set(table_ids_t.tolist()))
            if not ace_take_all:
                # Devi catturare UNA carta di pari rank
                if not (len(cap_ids) == 1 and (cap_ids[0] in same_rank_ids)):
                    raise ValueError("Quando esistono carte di rank uguale, devi catturarne una (non una combinazione).")
        elif cap_ids:
            # Verifica somma
            # Eccezione: Asso piglia tutto
            if not (rank == 1 and self.rules.get("asso_piglia_tutto", False) and set(cap_ids) == set(table_ids_t.tolist())):
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
                # tensor mirror
                try:
                    mask = (self._one_i64 << int(pid))
                    self._hands_bits_t[current_player] = self._hands_bits_t[current_player] & (~mask)
                except Exception:
                    pass
        
        if forced_ace_capture_on_empty:
            # Cattura forzata su tavolo vuoto: conta come scopa (o cattura se disabilitata via opzione)
            squad_id = 0 if current_player in [0, 2] else 1
            self.game_state["captured_squads"][squad_id].append(pid)
            try:
                self._captured_bits_t[squad_id] = self._captured_bits_t[squad_id] | (self._one_i64 << int(pid))
            except Exception:
                pass
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
                        try:
                            self._table_bits_t = self._table_bits_t & (~(self._one_i64 << int(cid)))
                        except Exception:
                            pass
            
            # Aggiungi le carte catturate e la carta giocata alla squadra
            squad_id = 0 if current_player in [0, 2] else 1
            self.game_state["captured_squads"][squad_id].extend(cap_ids)
            self.game_state["captured_squads"][squad_id].append(pid)
            try:
                take_mask = (self._one_i64 << int(pid))
                for cid in cap_ids:
                    take_mask = take_mask | (self._one_i64 << int(cid))
                self._captured_bits_t[squad_id] = self._captured_bits_t[squad_id] | take_mask
            except Exception:
                pass
            
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
                    try:
                        self._table_bits_t = self._table_bits_t | (self._one_i64 << int(pid))
                    except Exception:
                        pass
        
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
                        try:
                            add_mask = torch.zeros((), dtype=torch.int64, device=device)
                            for cid in self.game_state["table"]:
                                add_mask = add_mask | (self._one_i64 << int(_card_to_id(cid) if not isinstance(cid, int) else int(cid)))
                            self._captured_bits_t[last_capturing_team] = self._captured_bits_t[last_capturing_team] | add_mask
                        except Exception:
                            pass
                # In ogni caso svuota il tavolo a fine mano
                self.game_state["table"].clear()
                # reset cache table
                if self._use_id_cache:
                    self._table_ids.clear()
                    self._table_bits = 0
                    try:
                        self._table_bits_t = torch.zeros((), dtype=torch.int64, device=device)
                    except Exception:
                        pass
            
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
                    try:
                        self._hands_bits_t[p] = torch.as_tensor(self._hands_bits[p], dtype=torch.int64, device=device)
                    except Exception:
                        self._hands_bits_t[p] = torch.tensor(int(self._hands_bits[p]), dtype=torch.int64, device=device)
                self._table_ids = list(self.game_state['table'])
                self._table_bits = _ids_to_bitset(self._table_ids) if self._use_bitset else 0
                try:
                    self._table_bits_t = torch.as_tensor(self._table_bits, dtype=torch.int64, device=device)
                except Exception:
                    self._table_bits_t = torch.tensor(int(self._table_bits), dtype=torch.int64, device=device)
            else:
                self._rebuild_id_caches()
        except Exception:
            self._rebuild_id_caches()
        # reset captured bits tensors
        try:
            bits0 = _ids_to_bitset([_card_to_id(c) for c in self.game_state['captured_squads'][0]])
            bits1 = _ids_to_bitset([_card_to_id(c) for c in self.game_state['captured_squads'][1]])
        except Exception:
            bits0, bits1 = 0, 0
        self._captured_bits_t[0] = torch.as_tensor(bits0, dtype=torch.int64, device=device)
        self._captured_bits_t[1] = torch.as_tensor(bits1, dtype=torch.int64, device=device)
        
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