# environment.py - VERSIONE OTTIMIZZATA PER PERFORMANCE
import torch
import gym
from gym import spaces
import time
from functools import lru_cache
from collections import OrderedDict

from state import initialize_game
from observation import encode_state_compact_for_player_fast
from actions import decode_action_ids
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
        
        # Stato del gioco (GPU-only)
        self.game_state = None
        self.done = False
        self.current_player = 0
        # ricompense cumulative (CUDA)
        self.rewards_t = torch.zeros(2, dtype=torch.float32, device=device)
        # GPU mirrors: int64 bitsets on CUDA to keep state resident on device
        self._hands_bits_t = torch.zeros(4, dtype=torch.int64, device=device)
        self._table_bits_t = torch.zeros((), dtype=torch.int64, device=device)
        self._captured_bits_t = torch.zeros(2, dtype=torch.int64, device=device)
        self._one_i64 = torch.tensor(1, dtype=torch.int64, device=device)
        # Track remaining hand cards to avoid GPU->CPU sync for done flag
        self._remaining_hand_cards = 0
        self._last_capturing_team_t = torch.tensor(-1, dtype=torch.long, device=device)
        # history tensor (max 40 moves). fields: player_id(1), capture_type(1), played_id(1), captured_mask(40)
        self._max_moves = 40
        self._history_len_t = torch.zeros((), dtype=torch.long, device=device)
        self._history_player_t = torch.full((self._max_moves,), -1, dtype=torch.long, device=device)
        self._history_capture_type_t = torch.zeros((self._max_moves,), dtype=torch.long, device=device)
        self._history_played_t = torch.full((self._max_moves,), -1, dtype=torch.long, device=device)
        self._history_captured_mask_t = torch.zeros((self._max_moves, 40), dtype=torch.float32, device=device)
        
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
    def _set_game_state_from_deal(self, deal_state_dict):
        """Inizializza i mirror bitset GPU a partire dallo stato iniziale (liste ID) e crea il game_state GPU-only."""
        # hands
        for p in range(4):
            lst = deal_state_dict['hands'][p]
            if not lst:
                self._hands_bits_t[p] = torch.zeros((), dtype=torch.int64, device=device)
            else:
                ids = torch.as_tensor(lst, dtype=torch.long, device=device)
                shifts = (self._one_i64 << ids)
                self._hands_bits_t[p] = shifts.sum()
        # table
        tbl = deal_state_dict['table']
        if not tbl:
            self._table_bits_t = torch.zeros((), dtype=torch.int64, device=device)
        else:
            ids = torch.as_tensor(tbl, dtype=torch.long, device=device)
            self._table_bits_t = (self._one_i64 << ids).sum()
        # captured squads
        for t in [0, 1]:
            lst = deal_state_dict['captured_squads'][t]
            if not lst:
                self._captured_bits_t[t] = torch.zeros((), dtype=torch.int64, device=device)
            else:
                ids = torch.as_tensor(lst, dtype=torch.long, device=device)
                self._captured_bits_t[t] = (self._one_i64 << ids).sum()
        # remaining cards
        self._remaining_hand_cards = sum(len(deal_state_dict['hands'][p]) for p in range(4))
        # history tensors
        self._history_len_t.zero_()
        self._history_player_t.fill_(-1)
        self._history_capture_type_t.zero_()
        self._history_played_t.fill_(-1)
        self._history_captured_mask_t.zero_()
        # build GPU-only game_state dict for observation/rewards
        self.game_state = {
            '_hands_bits_t': self._hands_bits_t,
            '_table_bits_t': self._table_bits_t,
            '_captured_bits_t': self._captured_bits_t,
            'history_len_t': self._history_len_t,
            'history_player_t': self._history_player_t,
            'history_capture_type_t': self._history_capture_type_t,
            'history_played_t': self._history_played_t,
            'history_captured_mask_t': self._history_captured_mask_t,
        }

    
    def clone(self):
        """Clona lo stato GPU dell'ambiente (solo campi necessari per MCTS)."""
        cloned = ScoponeEnvMA(rules=dict(self.rules), use_compact_obs=True, k_history=self.k_history)
        cloned.done = bool(self.done)
        cloned.current_player = int(self.current_player)
        cloned.rewards_t = self.rewards_t.clone()
        cloned._hands_bits_t = self._hands_bits_t.clone()
        cloned._table_bits_t = self._table_bits_t.clone()
        cloned._captured_bits_t = self._captured_bits_t.clone()
        cloned._one_i64 = self._one_i64
        cloned._remaining_hand_cards = int(self._remaining_hand_cards)
        cloned._last_capturing_team_t = self._last_capturing_team_t.clone()
        cloned._history_len_t = self._history_len_t.clone()
        cloned._history_player_t = self._history_player_t.clone()
        cloned._history_capture_type_t = self._history_capture_type_t.clone()
        cloned._history_played_t = self._history_played_t.clone()
        cloned._history_captured_mask_t = self._history_captured_mask_t.clone()
        cloned.game_state = {
            '_hands_bits_t': cloned._hands_bits_t,
            '_table_bits_t': cloned._table_bits_t,
            '_captured_bits_t': cloned._captured_bits_t,
            'history_len_t': cloned._history_len_t,
            'history_player_t': cloned._history_player_t,
            'history_capture_type_t': cloned._history_capture_type_t,
            'history_played_t': cloned._history_played_t,
            'history_captured_mask_t': cloned._history_captured_mask_t,
        }
        return cloned

    def apply_determinization_gpu(self, owner_masks: torch.Tensor, observer_id: int):
        """
        Applica una determinizzazione usando maschere GPU (3,40) booleane per i tre giocatori non osservatori
        nell'ordine others=[(obs+1)%4,(obs+2)%4,(obs+3)%4]. Aggiorna solo i mirror GPU (_hands_bits_t) e cache ID.
        """
        device_local = self._hands_bits_t.device
        masks = owner_masks.to(device_local, dtype=torch.bool)
        ids = torch.arange(40, device=device_local, dtype=torch.int64)
        others = [(observer_id + 1) % 4, (observer_id + 2) % 4, (observer_id + 3) % 4]
        for i, pid in enumerate(others):
            # ricostruisci bitset via somma delle potenze di due selezionate
            shifts = (self._one_i64 << ids)
            bitset = shifts[masks[i]].sum()
            self._hands_bits_t[pid] = bitset
            # Niente mirror/Sync su CPU in modalità GPU-only
    #@profile
    def get_valid_actions(self):
        """Calcola azioni valide interamente su GPU usando i mirror a bitset CUDA."""
        start_time = time.time()
        from observation import RANK_OF_ID as _RANK_OF_ID
        from actions import encode_action_from_ids_gpu

        # Estrai ID in mano e sul tavolo da bitset tensor
        ids = torch.arange(40, device=device, dtype=torch.int64)
        hand_mask = ((self._hands_bits_t[self.current_player] >> ids) & 1).bool()
        table_mask = ((self._table_bits_t >> ids) & 1).bool()
        hand_ids_t = ids[hand_mask]
        table_ids_t = ids[table_mask]

        valid_actions = []

        if hand_ids_t.numel() == 0:
            self._get_valid_actions_time += time.time() - start_time
            return torch.zeros((0, 80), dtype=torch.float32, device=device)

        for pid_t in hand_ids_t:
            # resta su GPU
            prank_t = _RANK_OF_ID[pid_t].to(torch.int64)

            # Pari-rank sul tavolo
            if table_ids_t.numel() > 0:
                table_ranks = _RANK_OF_ID[table_ids_t].to(torch.int64)
                direct_sel = (table_ranks == prank_t)
                direct_ids_t = table_ids_t[direct_sel]
            else:
                direct_ids_t = torch.empty(0, dtype=torch.int64, device=device)

            if direct_ids_t.numel() > 0:
                for did_t in direct_ids_t:
                    valid_actions.append(encode_action_from_ids_gpu(pid_t, did_t.view(1)))
            else:
                # Somma: trova tutti i subset con somma prank (GPU)
                n = int(table_ids_t.numel())
                if n > 0:
                    pos = torch.arange(n, device=device, dtype=torch.int64)
                    ranks_k = _RANK_OF_ID[table_ids_t].to(torch.int64)
                    masks = torch.arange(1, 1 << n, device=device, dtype=torch.int64)
                    sel = ((masks.unsqueeze(1) >> pos) & 1).to(torch.int64)
                    sums = (sel * ranks_k.unsqueeze(0)).sum(dim=1)
                    good = (sums == prank_t).nonzero(as_tuple=False).flatten()
                    if good.numel() > 0:
                        for gi in good:
                            cap_ids_t = table_ids_t[((masks[gi].unsqueeze(0) >> pos) & 1).bool()]
                            valid_actions.append(encode_action_from_ids_gpu(pid_t, cap_ids_t))
                    else:
                        # Scarto
                        valid_actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))
                else:
                    # Tavolo vuoto: solo scarto
                    valid_actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))

            # Variante: Asso piglia tutto (aggiungi cattura completa)
            if self.rules.get("asso_piglia_tutto", False) and table_ids_t.numel() > 0 and (prank_t == 1):
                valid_actions.append(encode_action_from_ids_gpu(pid_t, table_ids_t))
            # Ace place action if allowed
            ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
            ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
            if (prank_t == 1) and ap_posabile and (not ap_only_empty or table_ids_t.numel() == 0):
                valid_actions.append(encode_action_from_ids_gpu(pid_t, torch.empty(0, dtype=torch.long, device=device)))

        # Post-filtri AP: rimuovi posa asso non consentita se richiesto (GPU-vectorized)
        ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
        ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
        if not (ap_posabile and (not ap_only_empty or table_ids_t.numel() == 0)) and table_ids_t.numel() > 0 and len(valid_actions) > 0:
            try:
                stack = torch.stack(valid_actions, dim=0).to(device=device, dtype=torch.float32)
                played_all = stack[:, :40].reshape(-1, 10, 4)
                captured_all = stack[:, 40:]
                ace_play = played_all[:, 0, :].any(dim=1)
                no_capture = ~captured_all.any(dim=1)
                keep_mask = ~(ace_play & no_capture)
                stack = stack[keep_mask]
                valid_actions = [row for row in stack]
            except Exception:
                pass

        self._get_valid_actions_time += time.time() - start_time
        if len(valid_actions) == 0:
            return torch.zeros((0, 80), dtype=torch.float32, device=device)
        return torch.stack(valid_actions, dim=0)
    
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

        # Decodifica l'azione in ID (GPU-native, nessuna sync)
        try:
            from actions import decode_action_ids_torch as _decode_ids_torch
            pid_t, cap_ids_t = _decode_ids_torch(action_vec)
        except Exception:
            pid_t, cap_ids_t = decode_action_ids(action_vec)
        
        # Verifica validità (come prima)
        current_player = self.current_player
        # Verifica appartenenza carta alla mano via bitset GPU
        hand_has_pid = ((self._hands_bits_t[current_player] & (self._one_i64 << pid_t)) != 0)
        if not hand_has_pid.ne(0).is_nonzero():
            raise ValueError(f"La carta richiesta non è nella mano del giocatore {current_player}.")
        
        # Verifica carte da catturare (ID) interamente via bitset CUDA mirror (vectorized)
        if cap_ids_t.numel() > 0:
            present_mask = (((self._table_bits_t >> cap_ids_t) & self._one_i64) != 0)
            if not present_mask.all().is_nonzero():
                raise ValueError("Cattura non valida: alcune carte non sono sul tavolo.")

        # Applicazione nuova regola AP posabilità: forza presa totale se non è consentito posare
        rank_t = pid_t // 4 + 1
        ap_enabled = bool(self.rules.get("asso_piglia_tutto", False))
        ap_posabile = bool(self.rules.get("asso_piglia_tutto_posabile", False))
        ap_only_empty = bool(self.rules.get("asso_piglia_tutto_posabile_only_empty", False))
        forced_ace_capture_on_empty = False
        # Override one-shot: la UI può impostare questo flag per auto-presa su tavolo vuoto anche se posabile è ON
        force_self_capture_once = bool(self.rules.get("force_ace_self_capture_on_empty_once", False))
        if ap_enabled:
            # Usa IDS_CUDA precomputato per evitare arange ripetuti
            from observation import IDS_CUDA as _IDS_CUDA
            table_ids_t = _IDS_CUDA[((self._table_bits_t >> _IDS_CUDA) & 1).to(torch.bool)]
            num_table_cards = int(table_ids_t.numel())
            if (rank_t == 1).is_nonzero():
                can_place_now = ap_posabile and (not ap_only_empty or (ap_only_empty and num_table_cards == 0))
                if (not can_place_now and cap_ids_t.numel() == 0) or (force_self_capture_once and num_table_cards == 0 and cap_ids_t.numel() == 0):
                    if num_table_cards > 0:
                    # Forza presa di tutto il tavolo
                        cap_ids_t = table_ids_t
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
        from observation import IDS_CUDA as _IDS_CUDA
        table_ids_t = _IDS_CUDA[((self._table_bits_t >> _IDS_CUDA) & 1).to(torch.bool)]
        from observation import RANK_OF_ID as _RANK_OF_ID
        same_rank_mask = (_RANK_OF_ID[table_ids_t].to(torch.int64) == rank_t)
        same_rank_ids_t = table_ids_t[same_rank_mask]
        if same_rank_ids_t.numel() > 0:
            # Eccezione: Asso piglia tutto permette di ignorare la regola della presa diretta
            if not (((rank_t == 1) & torch.tensor(self.rules.get("asso_piglia_tutto", False), device=device)).is_nonzero()):
                # Devi catturare UNA carta di pari rank
                cond_single = (cap_ids_t.numel() == 1)
                cond_in = (torch.isin(cap_ids_t.view(-1), same_rank_ids_t).any() if cap_ids_t.numel() > 0 else torch.tensor(False, device=device))
                if not ((cond_single and cond_in.is_nonzero())):
                    raise ValueError("Quando esistono carte di rank uguale, devi catturarne una (non una combinazione).")
        elif cap_ids_t.numel() > 0:
            # Verifica somma
            # Eccezione: Asso piglia tutto
            if not ((rank_t == 1) and self.rules.get("asso_piglia_tutto", False)
                    and (cap_ids_t.numel() == table_ids_t.numel())
                    and torch.equal(torch.sort(cap_ids_t)[0], torch.sort(table_ids_t)[0])):
                sum_chosen_ok = (((cap_ids_t // 4) + 1).sum() == rank_t)
                if not sum_chosen_ok.is_nonzero():
                    raise ValueError("La somma delle carte catturate deve essere uguale al rank della carta giocata.")
        
        # OTTIMIZZAZIONE: Esegui l'azione in modo più efficiente
        capture_type = "no_capture"
        
        # Rimuovi la carta giocata dalla mano (ID)
        # Aggiorna bitset mano giocatore (GPU-only)
        mask_played = (self._one_i64 << pid_t)
        self._hands_bits_t[current_player] = self._hands_bits_t[current_player] & (~mask_played)
        # decrementa contatore carte
        self._remaining_hand_cards = max(0, self._remaining_hand_cards - 1)
        
        if forced_ace_capture_on_empty:
            # Cattura forzata su tavolo vuoto: conta come scopa (o cattura se disabilitata via opzione)
            squad_id = 0 if current_player in [0, 2] else 1
            self._captured_bits_t[squad_id] = self._captured_bits_t[squad_id] | mask_played
            # Scopa se non è l'ultima carta giocata
            if self._remaining_hand_cards > 0:
                capture_type = "scopa"
            else:
                capture_type = "scopa" if self.rules.get("scopa_on_last_capture", False) else "capture"
            self._last_capturing_team_t = torch.tensor(squad_id, dtype=torch.long, device=device)
        elif cap_ids_t.numel() > 0:
            # Cattura carte
            cap_mask = (self._one_i64 << cap_ids_t).sum() if cap_ids_t.numel() > 0 else torch.zeros((), dtype=torch.int64, device=device)
            self._table_bits_t = self._table_bits_t & (~cap_mask)
            # Aggiungi le carte catturate e la carta giocata alla squadra
            squad_id = 0 if current_player in [0, 2] else 1
            take_mask = mask_played | cap_mask
            self._captured_bits_t[squad_id] = self._captured_bits_t[squad_id] | take_mask
            
            # Verifica scopa
            if (self._table_bits_t == 0).is_nonzero():
                # Verifica che non sia l'ultima giocata
                if self._remaining_hand_cards > 0:
                    capture_type = "scopa"
                else:
                    # Scopa all'ultima presa: opzionale
                    capture_type = "scopa" if self.rules.get("scopa_on_last_capture", False) else "capture"
            else:
                capture_type = "capture"
            self._last_capturing_team_t = torch.tensor(squad_id, dtype=torch.long, device=device)
        else:
            # Nessuna cattura: la carta va sul tavolo
            self._table_bits_t = self._table_bits_t | mask_played
        
        # Calcola reward shaping opzionale
        shaped_reward_t = torch.zeros((), dtype=torch.float32, device=device)
        if capture_type == "scopa" and bool(self.rules.get("shape_scopa", False)):
            try:
                shaped_reward_t = torch.tensor(self.rules.get("scopa_reward", 0.1), dtype=torch.float32, device=device)
            except Exception:
                shaped_reward_t = torch.tensor(0.1, dtype=torch.float32, device=device)

        # Aggiorna history GPU-only
        ct_map = {"no_capture": 0, "capture": 1, "scopa": 2}
        idx = self._history_len_t.long() if torch.is_tensor(self._history_len_t) else 0
        if idx < self._max_moves:
            self._history_player_t[idx] = torch.tensor(current_player, dtype=torch.long, device=device)
            self._history_capture_type_t[idx] = torch.tensor(ct_map.get(capture_type, 0), dtype=torch.long, device=device)
            self._history_played_t[idx] = pid_t.to(dtype=torch.long)
            cap_mask_vec = torch.zeros(40, dtype=torch.float32, device=device)
            if cap_ids_t.numel() > 0:
                rows = (cap_ids_t // 4).clamp_(0, 9)
                cols = (cap_ids_t % 4).clamp_(0, 3)
                cap_mask_mat = cap_mask_vec.view(10, 4)
                cap_mask_mat[rows, cols] = 1.0
                cap_mask_vec = cap_mask_mat.reshape(-1)
            self._history_captured_mask_t[idx] = cap_mask_vec
            self._history_len_t += 1
        # Nota: comportamento speciale "scopa_on_asso_piglia_tutto" non necessario in GPU-only

        # Limite scope consecutive per team: contiamo usando history tensors
        if capture_type == "scopa":
            limit = self.rules.get("max_consecutive_scope")
            if isinstance(limit, int) and limit > 0:
                team_id = 0 if current_player in [0, 2] else 1
                consec = 0
                hlen = self._history_len_t if torch.is_tensor(self._history_len_t) else 0
                # Usa operazioni vettoriali invece di loop Python  
                if hlen > 0:
                    # Maschera per scope (tipo 2)
                    scopa_mask = (self._history_capture_type_t[:hlen] == 2)
                    players = self._history_player_t[:hlen]
                    # Calcola team per ogni player: 0 se player in [0,2], 1 altrimenti
                    teams = torch.where((players == 0) | (players == 2), 
                                      torch.tensor(0, device=device),
                                      torch.tensor(1, device=device))
                    # Trova scope consecutive del team corrente
                    team_scope_mask = scopa_mask & (teams == team_id)
                    # Conta consecutive partendo dalla fine
                    for j in range(hlen-1, -1, -1):
                        if team_scope_mask[j]:
                            consec += 1
                        elif scopa_mask[j]:  # Scopa di altro team
                            break
                if consec >= limit:
                    self._history_capture_type_t[hlen-1] = torch.tensor(1, dtype=torch.long, device=device)
        
        # OTTIMIZZAZIONE: Invalida la cache delle osservazioni (LRU)
        self._observation_cache = OrderedDict()
        
        # Verifica se la partita è finita
        done = (self._remaining_hand_cards == 0)
        self.done = done
        
        if done:
            # Assegna le carte rimaste sul tavolo (opzionale)
            table_non_empty = (self._table_bits_t != 0).is_nonzero()
            if table_non_empty:
                if self.rules.get("last_cards_to_dealer", True):
                    if (self._last_capturing_team_t >= 0).any():
                        lid = self._last_capturing_team_t.long()
                        self._captured_bits_t[lid] = self._captured_bits_t[lid] | self._table_bits_t
                # svuota tavolo
                self._table_bits_t = torch.zeros((), dtype=torch.int64, device=device)
            
            # Calcolo punteggio finale completamente su GPU
            from rewards import compute_final_score_breakdown_torch, compute_final_team_rewards_torch
            # Espone mirror bitset nel game_state per funzioni rewards/obs
            try:
                self.game_state['_hands_bits_t'] = self._hands_bits_t
                self.game_state['_table_bits_t'] = self._table_bits_t
                self.game_state['_captured_bits_t'] = self._captured_bits_t
            except Exception:
                pass
            final_breakdown_t = compute_final_score_breakdown_torch(self.game_state, rules=self.rules)
            team_rewards_t = compute_final_team_rewards_torch(self.game_state, rules=self.rules)  # (2,) CUDA
            info = {
                "score_breakdown_t": final_breakdown_t,
                "team_rewards_t": team_rewards_t,
            }
            
            # Osservazione finale (zeros)
            obs_final = torch.zeros(self.observation_space.shape, dtype=torch.float32, device=device)
            
            # Aggiorna il tempo di esecuzione
            self._step_time += time.time() - step_start_time
            
            # Restituisci la ricompensa finale per il team del giocatore corrente
            current_team = 0 if current_player in [0, 2] else 1
            return obs_final, team_rewards_t[current_team], True, info
        else:
            # Passa al prossimo giocatore
            self.current_player = (self.current_player + 1) % 4
            
            # OTTIMIZZAZIONE: Invalida la cache delle azioni valide
            self._last_state_hash = None
            
            # Ottieni la prossima osservazione
            next_obs = self._get_observation(self.current_player)
            
            # Aggiorna il tempo di esecuzione
            self._step_time += time.time() - step_start_time
            
            return next_obs, shaped_reward_t, False, {"last_move_t": torch.tensor(1 if capture_type == 'scopa' else 0, device=device)}
    
    #@profile
    def _get_observation(self, player_id):
        """Osservazione compatta GPU-only, nessuna cache CPU."""
        start_time = time.time()
        # Esponi mirror per fast-path
        self.game_state['_hands_bits_t'] = self._hands_bits_t
        self.game_state['_table_bits_t'] = self._table_bits_t
        self.game_state['_captured_bits_t'] = self._captured_bits_t
        self.game_state['history_len_t'] = self._history_len_t
        self.game_state['history_player_t'] = self._history_player_t
        self.game_state['history_capture_type_t'] = self._history_capture_type_t
        self.game_state['history_played_t'] = self._history_played_t
        self.game_state['history_captured_mask_t'] = self._history_captured_mask_t
        obs = encode_state_compact_for_player_fast(self.game_state, player_id, k_history=self.k_history)
        self._get_obs_time += time.time() - start_time
        return obs

    # ==== GPU helper for GUI (no CPU lists) ====
    def get_table_vec(self) -> torch.Tensor:
        """Ritorna vettore (40,) CUDA con 1 dove la carta è sul tavolo."""
        ids = torch.arange(40, device=device, dtype=torch.int64)
        return (((self._table_bits_t >> ids) & 1).to(torch.float32))

    def get_hand_vec(self, player_id: int) -> torch.Tensor:
        """Ritorna vettore (40,) CUDA con 1 dove la carta è in mano al player."""
        ids = torch.arange(40, device=device, dtype=torch.int64)
        return (((self._hands_bits_t[player_id] >> ids) & 1).to(torch.float32))
    
    def reset(self, starting_player=None):
        """Versione ottimizzata di reset"""
        # Reimposta lo stato del gioco rispettando le regole/varianti (usa solo bitset)
        deal_state = initialize_game(rules=self.rules)
        self.done = False
        self.current_player = starting_player if starting_player is not None else 0
        self.rewards_t.zero_()
        self._valid_actions_cache = OrderedDict()
        self._last_state_hash = None
        self._observation_cache = OrderedDict()
        self._set_game_state_from_deal(deal_state)
        
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




