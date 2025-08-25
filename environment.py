# environment.py - VERSIONE OTTIMIZZATA PER PERFORMANCE
import torch
import os
from utils.device import get_env_device
import gym
from gym import spaces
import time
from collections import OrderedDict

from state import initialize_game
from actions import decode_action_ids
# line_profiler opzionale: disabilitato in test/produzione

# Per l'ambiente usiamo la CPU per evitare micro-kernel su GPU.
# Può essere forzato impostando ENV_DEVICE, ma di default resta CPU.
device = get_env_device()

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

def _ids_to_bitset(ids):
    bits = 0
    for cid in ids:
        bits |= (1 << cid)
    return bits

class ScoponeEnvMA(gym.Env):
    def __init__(self, rules=None, use_compact_obs: bool = False, k_history: int = 39):
        super().__init__()
        
        # Config osservazione
        # Forza la modalità compatta: la legacy 10823 è deprecata
        self.use_compact_obs = True if use_compact_obs is None else bool(use_compact_obs) or True
        self.k_history = int(k_history)
        if self.use_compact_obs:
            # Dimensione compatta dinamica in base ai flag OBS_INCLUDE_*:
            # Base fisse: 43 (mani) + 40 (tavolo) + 82 (catture) + 61*k (history) + 40 (missing)
            #            + 120 (inferred) + 8 (primiera) + 2 (denari) + 1 (settebello) + 2 (score)
            #            + 1 (table sum) + 10 (table possible sums) + 2 (scopa counts)
            include_rank = os.environ.get('OBS_INCLUDE_RANK_PROBS', '0') == '1'
            include_scopa = os.environ.get('OBS_INCLUDE_SCOPA_PROBS', '0') == '1'
            include_inferred = os.environ.get('OBS_INCLUDE_INFERRED', '0') == '1'
            # Part fisse comuni (senza inferred): 43+40+82 + 61*k + 40 + 8 + 2 + 1 + 2 + 1 + 10 + 2 + 30
            fixed = 43 + 40 + 82 + 61 * self.k_history + 40 + 8 + 2 + 1 + 2 + 1 + 10 + 2 + 30
            base = fixed + (120 if include_inferred else 0)
            obs_dim = base + (10 if include_scopa else 0) + (150 if include_rank else 0)
        else:
            # Non dovrebbe mai accadere: legacy disattivata
            raise NotImplementedError("La rappresentazione legacy 10823-dim è deprecata: usa use_compact_obs=True")
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
        # Cache DP per subset-sum sul tavolo (chiave: (table_bits, rank))
        self._subset_sum_cache = {}
        
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
        # Mirrors (on env device; default CPU). We keep API compatible.
        self._hands_bits_t = torch.zeros(4, dtype=torch.int64, device=device)
        self._table_bits_t = torch.zeros((), dtype=torch.int64, device=device)
        self._captured_bits_t = torch.zeros(2, dtype=torch.int64, device=device)
        self._one_i64 = torch.tensor(1, dtype=torch.int64, device=device)
        # Precompute id range to avoid per-call allocations
        self._id_range = torch.arange(40, dtype=torch.int64, device=device)
        
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
        # Ricostruisci le cache ID/bitset per allinearle allo stato copiato
        try:
            cloned._rebuild_id_caches()
        except Exception:
            # fallback: tenta un reset e poi riallinea
            try:
                cloned.reset(starting_player=self.current_player)
                cloned.game_state = copy.deepcopy(self.game_state)
                cloned._rebuild_id_caches()
            except Exception:
                pass
        return cloned
    #@profile
    def get_valid_actions(self):
        """Calcola azioni valide su CPU usando bitset; evita micro-kernel su GPU."""
        start_time = time.time()
        # RANK_OF_ID è definito per CUDA; per CPU calcoliamo al volo
        from actions import encode_action_from_ids_gpu

        # Chiave di cache LRU basata su (giocatore, mano, tavolo, regole AP)
        try:
            ap_enabled = bool(self.rules.get("asso_piglia_tutto", False))
            ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
            ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
            state_key = (
                int(self.current_player),
                int(self._hands_bits_t[self.current_player].item()),
                int(self._table_bits_t.item()),
                ap_enabled, ap_posabile, ap_only_empty,
            )
            cached = self._valid_actions_cache.get(state_key)
            if cached is not None:
                self._cache_hits += 1
                self._get_valid_actions_time += time.time() - start_time
                # Ritorna una copia superficiale per evitare side-effect sulla lista
                return list(cached)
        except Exception:
            state_key = None

        # Estrai ID in mano e sul tavolo da bitset tensor (CPU tensors)
        ids = self._id_range
        hand_mask = ((self._hands_bits_t[self.current_player] >> ids) & 1).bool()
        table_mask = ((self._table_bits_t >> ids) & 1).bool()
        hand_ids_t = ids[hand_mask]
        table_ids_t = ids[table_mask]

        valid_actions = []

        if hand_ids_t.numel() == 0:
            self._get_valid_actions_time += time.time() - start_time
            return valid_actions

        # Hoist invariants out of per-hand loop
        encode_fn = encode_action_from_ids_gpu
        if table_ids_t.numel() > 0:
            table_ranks = (table_ids_t // 4 + 1).to(torch.int64)
        else:
            table_ranks = None

        for pid_t in hand_ids_t:
            prank_t = (pid_t // 4 + 1).to(torch.int64)

            # Pari-rank sul tavolo (use hoisted table_ranks if available)
            if table_ranks is not None:
                direct_sel = (table_ranks == prank_t)
                direct_ids_t = table_ids_t[direct_sel]
            else:
                direct_ids_t = torch.empty(0, dtype=torch.int64, device=device)

            if direct_ids_t.numel() > 0:
                for did_t in direct_ids_t:
                    valid_actions.append(encode_fn(pid_t, did_t.view(1)))
            else:
                # Somma: trova subset con somma pari a prank (DP su somma-rank, n piccolo, rank<=10)
                n = int(table_ids_t.numel())
                if n > 0:
                    table_ids = table_ids_t.tolist()
                    target = int(prank_t.item())
                    # Cache per (table_bits, target)
                    try:
                        subset_key = (int(self._table_bits_t.item()), target)
                    except Exception:
                        subset_key = None
                    masks_list = None
                    if subset_key is not None:
                        masks_list = self._subset_sum_cache.get(subset_key)
                    if masks_list is None:
                        # Via vettoriale: calcola tutti i sottoinsiemi validi con Torch
                        from actions import find_sum_subsets_ids as _find_sum_subsets_ids
                        subsets = _find_sum_subsets_ids(table_ids, target)
                        # Converte liste di ID in bitmask locali (rispetto a table_ids)
                        idx_map = {cid: j for j, cid in enumerate(table_ids)}
                        masks_list = []
                        for subset in subsets:
                            bm = 0
                            for cid in subset:
                                j = idx_map.get(int(cid))
                                if j is not None:
                                    bm |= (1 << j)
                            masks_list.append(bm)
                        if subset_key is not None:
                            self._subset_sum_cache[subset_key] = masks_list
                    if masks_list:
                        for bm in masks_list:
                            sel_ids = [table_ids[j] for j in range(n) if ((bm >> j) & 1) == 1]
                            cap_ids_t = torch.as_tensor(sel_ids, dtype=torch.long, device=device)
                            valid_actions.append(encode_fn(pid_t, cap_ids_t))
                    else:
                        # Scarto (nessuna combinazione valida)
                        valid_actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))
                else:
                    # Tavolo vuoto: solo scarto
                    valid_actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))

            # Variante: Asso piglia tutto (aggiungi cattura completa)
            if self.rules.get("asso_piglia_tutto", False) and table_ids_t.numel() > 0 and int(prank_t.item()) == 1:
                valid_actions.append(encode_action_from_ids_gpu(pid_t, table_ids_t))
            # Ace place action if allowed
            ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
            ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
            if int(prank_t.item()) == 1 and ap_posabile and (not ap_only_empty or table_ids_t.numel() == 0):
                valid_actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))

        # Post-filtri AP: rimuovi posa asso non consentita solo se AP è attivo
        ap_enabled = bool(self.rules.get("asso_piglia_tutto", False))
        ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
        ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
        if ap_enabled and not (ap_posabile and (not ap_only_empty or table_ids_t.numel() == 0)) and table_ids_t.numel() > 0:
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

        # In condizioni corrette, se il giocatore ha carte in mano, deve esistere almeno un'azione valida
        if hand_ids_t.numel() > 0 and len(valid_actions) == 0:
            raise RuntimeError(f"No valid actions for player {self.current_player} (hand_ids={hand_ids_t.tolist()}, table_ids={table_ids_t.tolist()}, rules={self.rules})")

        # Aggiorna cache LRU
        try:
            if state_key is not None:
                self._cache_misses += 1
                self._valid_actions_cache[state_key] = tuple(valid_actions)
                self._valid_actions_cache.move_to_end(state_key)
                while len(self._valid_actions_cache) > self._cache_capacity:
                    self._valid_actions_cache.popitem(last=False)
        except Exception:
            pass

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
        
        # Verifica carte da catturare (ID) via bitset CPU (evita sync GPU)
        for cid in cap_ids:
            try:
                present = (self._table_bits >> int(cid)) & 1
            except Exception:
                try:
                    table_id_set = {_card_to_id(c) for c in table}
                except Exception:
                    table_id_set = set()
                present = 1 if int(cid) in table_id_set else 0
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
        
        # Verifica regole di cattura via bitset su CPU
        # Usa solo strutture CPU per evitare kernel micro e sync
        try:
            table_ids_list = list(self._table_ids) if self._use_id_cache else [_card_to_id(c) for c in table]
        except Exception:
            table_ids_list = [_card_to_id(c) for c in table]
        same_rank_ids = [i for i in table_ids_list if (i // 4 + 1) == int(rank)]
        if same_rank_ids:
            # Eccezione: Asso piglia tutto permette di ignorare la regola della presa diretta
            ace_take_all = (rank == 1 and self.rules.get("asso_piglia_tutto", False)
                            and set(cap_ids) == set(table_ids_list))
            if not ace_take_all:
                # Devi catturare UNA carta di pari rank
                if not (len(cap_ids) == 1 and (cap_ids[0] in same_rank_ids)):
                    raise ValueError("Quando esistono carte di rank uguale, devi catturarne una (non una combinazione).")
        elif cap_ids:
            # Verifica somma
            # Eccezione: Asso piglia tutto
            if not (rank == 1 and self.rules.get("asso_piglia_tutto", False) and set(cap_ids) == set(table_ids_list)):
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
            # Cattura forzata su tavolo vuoto con AP
            squad_id = 0 if current_player in [0, 2] else 1
            self.game_state["captured_squads"][squad_id].append(pid)
            try:
                self._captured_bits_t[squad_id] = self._captured_bits_t[squad_id] | (self._one_i64 << int(pid))
            except Exception:
                pass
            # Regole scopa su AP a tavolo vuoto
            ap_scopa_on = bool(self.rules.get("scopa_on_asso_piglia_tutto", False))
            cards_left = sum(len(self.game_state["hands"][p]) for p in range(4))
            if ap_scopa_on:
                # Se AP ha scopa attiva, conta sempre scopa (anche all'ultima presa)
                capture_type = "scopa"
            else:
                # AP con scopa disattivata: mai scopa a tavolo vuoto
                capture_type = "capture"
        elif cap_ids:
            # Cattura carte
            for cid in cap_ids:
                try:
                    table.remove(cid)
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
            # Demote salvo eccezione: presa DIRETTA di un asso su asso (unica carta prima della mossa)
            try:
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
                # Conta solo le scope consecutive nelle giocate della STESSA squadra
                for m in reversed(self.game_state["history"]):
                    prev_team = 0 if m.get("player") in [0, 2] else 1
                    if prev_team != team_id:
                        # mosse dell'altra squadra non interrompono né incrementano la serie
                        continue
                    if m.get("capture_type") == "scopa":
                        consecutive += 1
                    else:
                        # una giocata senza scopa della stessa squadra interrompe la serie
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
            # Non è consentito terminare una mano senza alcuna presa effettuata
            if not any(m.get("capture_type") in ("capture", "scopa") for m in self.game_state["history"]):
                raise ValueError("La mano non può terminare senza alcuna presa.")
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
                from observation import encode_state_compact_for_player_fast
                # esponi mirrors per fast-path GPU
                self.game_state['_hands_bits_t'] = self._hands_bits_t
                self.game_state['_table_bits_t'] = self._table_bits_t
                self.game_state['_captured_bits_t'] = self._captured_bits_t
                result = encode_state_compact_for_player_fast(self.game_state, player_id, k_history=self.k_history)
            
            # Salva in cache (LRU)
            try:
                # Ensure observations returned on CPU to avoid GPU micro-kernels in env
                result_cpu = result.detach().to('cpu') if torch.is_tensor(result) else result
            except Exception:
                result_cpu = result
            self._observation_cache[cache_key] = result_cpu
            self._observation_cache.move_to_end(cache_key)
            while len(self._observation_cache) > self._cache_capacity:
                self._observation_cache.popitem(last=False)
        
        # Aggiorna il tempo di esecuzione
        self._get_obs_time += time.time() - start_time
        
        # Always return CPU tensor
        try:
            return result.detach().to('cpu') if torch.is_tensor(result) else result
        except Exception:
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