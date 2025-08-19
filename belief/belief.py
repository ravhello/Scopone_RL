import torch
from typing import List, Dict, Tuple
import random


def card_to_id(card) -> int:
    if isinstance(card, int):
        return int(card)
    suit_to_int = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
    rank, suit = card
    return (rank - 1) * 4 + suit_to_int[suit]


def id_to_card(cid: int) -> Tuple[int, str]:
    suits = ['denari', 'coppe', 'spade', 'bastoni']
    rank = cid // 4 + 1
    suit = suits[cid % 4]
    return (rank, suit)


class BeliefState:
    """
    Particle filter semplificato per mani nascoste (ID carte 0..39).
    Mantiene assegnazioni per i 3 giocatori non osservatori, coerenti con stato pubblico corrente.
    """
    def __init__(self, game_state: Dict, observer_id: int, num_particles: int = 256, seed: int = 42, ess_frac: float = 0.5):
        self.rng = random.Random(seed)
        self.observer = observer_id
        self.num_particles = num_particles
        self.ess_frac = float(ess_frac)
        # Particelle su GPU: owner[p, i, cid] = True se nella particella p la carta cid (0..39)
        # è assegnata al giocatore others[i] (i=0..2 rispetto all'osservatore)
        self.owner = torch.zeros((num_particles, 3, 40), dtype=torch.bool, device=torch.device('cuda'))
        self.weights = torch.ones(num_particles, dtype=torch.float32, device=torch.device('cuda')) / float(num_particles)
        # vincoli permanenti derivati dalla history: rank che un player non può avere
        self._forbidden_ranks_by_pid = {pid: set() for pid in range(4)}
        # conteggi per suit/rank giocati per-player
        self._played_ids_by_pid = {pid: set() for pid in range(4)}
        self._init_from_state(game_state)

    def _visible_mask_and_counts(self, game_state: Dict):
        device = self.weights.device
        hands_bits_t = game_state['_hands_bits_t']
        table_bits_t = game_state['_table_bits_t']
        captured_bits_t = game_state['_captured_bits_t']
        ids = torch.arange(40, device=device, dtype=torch.int64)
        visible_bits = hands_bits_t[self.observer] | table_bits_t | captured_bits_t[0] | captured_bits_t[1]
        vis_mask = (((visible_bits >> ids) & 1).to(torch.bool))
        counts = torch.zeros(4, dtype=torch.long, device=device)
        for pid in range(4):
            counts[pid] = (((hands_bits_t[pid] >> ids) & 1).to(torch.float32).sum()).to(torch.long)
        return vis_mask, counts

    def _init_from_state(self, game_state: Dict):
        device = self.weights.device
        vis_mask, counts_all = self._visible_mask_and_counts(game_state)
        unknown_ids = torch.nonzero(~vis_mask, as_tuple=False).flatten()
        others = torch.tensor([(self.observer + 1) % 4, (self.observer + 2) % 4, (self.observer + 3) % 4], device=device, dtype=torch.long)
        counts_3_t = counts_all.index_select(0, others).clamp_min(0)
        # reset owner
        self.owner.zero_()
        U = unknown_ids.numel()
        if U > 0:
            pos = torch.arange(U, device=device, dtype=torch.long)
            cum = torch.cumsum(counts_3_t, dim=0)
            cum = torch.minimum(cum, torch.tensor(U, device=device, dtype=cum.dtype))
            for p in range(self.num_particles):
                perm = torch.randperm(U, device=device)
                g0 = pos < cum[0]
                g1 = (pos >= cum[0]) & (pos < cum[1])
                g2 = (pos >= cum[1]) & (pos < cum[2])
                if g0.any():
                    self.owner[p, 0, unknown_ids[perm[g0]]] = True
                if g1.any():
                    self.owner[p, 1, unknown_ids[perm[g1]]] = True
                if g2.any():
                    self.owner[p, 2, unknown_ids[perm[g2]]] = True
        self.weights.fill_(1.0 / float(self.num_particles))

    def _unknown_ids_and_counts(self, game_state: Dict):
        vis_mask, counts_all = self._visible_mask_and_counts(game_state)
        unknown_ids = torch.nonzero(~vis_mask, as_tuple=False).flatten()
        others = torch.tensor([(self.observer + 1) % 4, (self.observer + 2) % 4, (self.observer + 3) % 4], device=self.weights.device)
        active_unknown = (counts_all.index_select(0, others) > 0).to(torch.float32).sum()
        return unknown_ids, None, None, active_unknown, torch.tensor(0.0, device=self.weights.device), None, None

    def _per_player_exhausted_suits(self, game_state: Dict) -> Dict[int, List[bool]]:
        """Deduce per-player i semi impossibili considerando history e pubblico (grezzo, conservativo)."""
        suits_total = [10, 10, 10, 10]  # 10 carte per suit
        public_counts = [0, 0, 0, 0]
        for c in game_state.get('table', []):
            public_counts[card_to_id(c) % 4] += 1
        cs = game_state.get('captured_squads', None)
        if cs is not None:
            try:
                for c in cs[0] + cs[1]:
                    public_counts[card_to_id(c) % 4] += 1
            except Exception:
                try:
                    for c in cs.get(0, []) + cs.get(1, []):
                        public_counts[card_to_id(c) % 4] += 1
                except Exception:
                    pass
        per_pid = {}
        for pid in range(4):
            played_counts = [0, 0, 0, 0]
            for cid in self._played_ids_by_pid.get(pid, set()):
                played_counts[cid % 4] += 1
            # visibile nella mano osservatore
            if pid == self.observer:
                for c in game_state['hands'][self.observer]:
                    played_counts[card_to_id(c) % 4] += 1
            exhausted = []
            for s in range(4):
                exhausted.append(played_counts[s] + public_counts[s] >= suits_total[s])
            per_pid[pid] = exhausted
        return per_pid

    def effective_sample_size(self) -> torch.Tensor:
        w = self.weights
        return 1.0 / (w.pow(2).sum())

    def resample_if_needed(self, ess_threshold: float = None):
        if ess_threshold is None:
            ess_threshold = max(1.0, self.ess_frac * self.num_particles)
        ess = self.effective_sample_size()
        thr = torch.tensor(float(ess_threshold), device=self.weights.device, dtype=self.weights.dtype)
        if (ess < thr):
            idx = torch.multinomial(self.weights, num_samples=self.num_particles, replacement=True)
            self.owner = self.owner.index_select(0, idx)
            self.weights.fill_(1.0 / float(self.num_particles))

    def update_with_move(self, move: Dict, game_state: Dict, rules: Dict = None, ess_threshold: float = None):
        """
        Aggiorna i pesi filtrando le particelle incoerenti con la mossa osservata.
        Criteri minimi:
          - la carta giocata deve appartenere alla mano del giocatore corrente nella particella
          - le carte catturate devono essere sul tavolo pubblico (già garantito dallo stato)
        """
        player = int(move['player'])
        # move fields possono essere ID; uniformiamo a IDs
        played = card_to_id(move['played_card'])
        captured_ids = [card_to_id(c) for c in move.get('captured_cards', [])]
        # Legalità di base con regole scopa/scopone
        capture_type = move.get('capture_type', 'capture' if captured_ids else 'no_capture')
        rank_played = (played // 4) + 1
        sum_captured = sum(((cid // 4) + 1) for cid in captured_ids) if captured_ids else 0
        ap_enabled = bool((rules or {}).get('asso_piglia_tutto', False))

        # Legal move predicate: o presa diretta (una sola carta con rank uguale), o somma (somma == rank), o AP
        def _move_is_legal_under_rules() -> bool:
            if not captured_ids:
                return True
            # asso piglia tutto (tavolo intero): se AP abilitato e played è Asso e captured non vuoto
            if ap_enabled and rank_played == 1:
                return True
            # presa diretta
            if len(captured_ids) == 1 and ((captured_ids[0] // 4) + 1) == rank_played:
                return True
            # somma
            if sum_captured == rank_played:
                return True
            return False

        if not _move_is_legal_under_rules():
            # mossa illegale rispetto a regole: non aggiorniamo (evita distruzione del belief)
            return

        new_weights = torch.zeros_like(self.weights)
        # no table reconstruction in GPU-only mode

        # calcola cap e impossibilità globali
        _, suit_rem, rank_rem, active_unknown, progress, suit_possible, rank_possible = self._unknown_ids_and_counts(game_state)
        import math
        extra_margin = max(0, int(math.ceil(2.0 * (1.0 - progress))))
        suit_cap = []
        rank_cap = []
        for c in suit_rem:
            suit_cap.append(0 if c <= 0 else int(math.ceil(c / active_unknown)) + extra_margin)
        for c in rank_rem:
            rank_cap.append(0 if c <= 0 else int(math.ceil(c / active_unknown)) + extra_margin)
        # skip complex constraints in GPU-only

        # update pesi con controllo presenza carta giocata
        if player != self.observer:
            others = [(self.observer + 1) % 4, (self.observer + 2) % 4, (self.observer + 3) % 4]
            m = others.index(player) if player in others else -1
            if m >= 0:
                present = self.owner[:, m, played]
                new_weights = torch.where(present, self.weights, torch.zeros_like(self.weights))
                # rimuovi la carta dalla mano assegnata in tutte le particelle
                self.owner[:, m, played] = False
            else:
                new_weights = self.weights.clone()
        else:
            new_weights = self.weights.clone()
        # (vincoli aggiuntivi complessi omessi per ora; pesi già filtrati dalla presenza della carta)
        s = new_weights.sum()
        if (s > 0):
            self.weights = new_weights / s
        # aggiorna vincoli persistenti: se esiste almeno una carta dello stesso rank sul tavolo prima della mossa
        # e il player non ha effettuato presa diretta (e non AP taking-all), allora quel rank è vietato per il player
        # skip forbidden rank deductions in GPU-only
        # aggiorna storico carte giocate per-player
        try:
            self._played_ids_by_pid[player].add(played)
        except Exception:
            pass
        # resampling adattivo
        # simple resample every update (GPU-only)
        idx = torch.multinomial(self.weights, num_samples=self.num_particles, replacement=True)
        self.owner = self.owner.index_select(0, idx)
        self.weights.fill_(1.0 / float(self.num_particles))

    def sample_determinization(self, num: int):
        raise NotImplementedError("Use sample_determinization_gpu in GPU-only mode")

    def sample_determinization_gpu(self, num: int) -> torch.Tensor:
        """
        GPU-native: ritorna maschere boolean (num, 3, 40) su CUDA per le determinizzazioni campionate.
        L'ordine dei 3 giocatori è others = [(observer+1)%4, (observer+2)%4, (observer+3)%4].
        """
        idx = torch.multinomial(self.weights, num_samples=int(num), replacement=True)
        return self.owner.index_select(0, idx)

    def belief_summary(self, game_state: Dict, player_id: int):
        """
        Ritorna un vettore 120-dim (3 * 40) di marginali per carta∈{partner, opp1, opp2}.
        Ordine giocatori: cicla su pid!=player_id in ordine [ (player_id+1)%4, (player_id+2)%4, (player_id+3)%4 ].
        """
        # Somma pesata delle particelle su GPU
        w = self.weights.view(-1, 1, 1).to(dtype=torch.float32)
        marg = (self.owner.float() * w).sum(dim=0)  # (3,40)
        return marg.reshape(-1)



