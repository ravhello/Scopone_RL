from tests.torch_np import np
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
        self.particles: List[Dict[int, List[int]]] = []  # ogni particella: {player_id: [card_ids, ...]}
        self.weights = np.ones(num_particles, dtype=np.float32) / num_particles
        # vincoli permanenti derivati dalla history: rank che un player non può avere
        self._forbidden_ranks_by_pid = {pid: set() for pid in range(4)}
        # conteggi per suit/rank giocati per-player
        self._played_ids_by_pid = {pid: set() for pid in range(4)}
        self._init_from_state(game_state)

    def _visible_ids(self, game_state: Dict) -> Tuple[set, Dict[int, int]]:
        """Restituisce insieme di carte visibili e conteggi richiesti per ogni giocatore restante."""
        visible = set()
        counts = {}
        # carte visibili: mano osservatore, tavolo, captured di entrambi
        for c in game_state['hands'][self.observer]:
            visible.add(card_to_id(c))
        for c in game_state['table']:
            visible.add(card_to_id(c))
        for c in game_state['captured_squads'][0] + game_state['captured_squads'][1]:
            visible.add(card_to_id(c))
        # conteggio per ogni altro player = len(mano corrente)
        for pid in range(4):
            if pid == self.observer:
                continue
            counts[pid] = len(game_state['hands'][pid])
        return visible, counts

    def _init_from_state(self, game_state: Dict):
        visible, counts = self._visible_ids(game_state)
        all_cards = set(range(40))
        unknown = list(all_cards - visible)
        # Vincoli storici: carte già giocate per player
        played_by_pid = {pid: set() for pid in range(4)}
        try:
            for mv in game_state.get('history', []):
                pid = int(mv.get('player'))
                cid = card_to_id(mv.get('played_card'))
                played_by_pid[pid].add(cid)
                self._played_ids_by_pid[pid].add(cid)
        except Exception:
            pass
        for _ in range(self.num_particles):
            pool = list(unknown)
            self.rng.shuffle(pool)
            assignment = {}
            for pid in counts:
                k = counts[pid]
                forb_ids = played_by_pid.get(pid, set())
                # applica anche vincoli su rank vietati (se già presenti da update incrementali)
                forb_ranks = self._forbidden_ranks_by_pid.get(pid, set())
                allowed = [c for c in pool if (c not in forb_ids) and ((c // 4 + 1) not in forb_ranks)]
                take = allowed[:k]
                if len(take) < k:
                    rest = [c for c in pool if c not in take]
                    take += rest[:(k - len(take))]
                assignment[pid] = list(take)
                pool = [c for c in pool if c not in take]
            self.particles.append(assignment)
        self.weights[:] = 1.0 / self.num_particles

    def _unknown_ids_and_counts(self, game_state: Dict):
        visible, counts = self._visible_ids(game_state)
        unknown = [cid for cid in range(40) if cid not in visible]
        # residui per suit e rank tra le carte sconosciute
        suit_rem = [0, 0, 0, 0]
        rank_rem = [0] * 10
        for cid in unknown:
            suit_rem[cid % 4] += 1
            rank_rem[cid // 4] += 1
        # numero di giocatori non osservatori con carte in mano
        active_unknown = sum(1 for pid in range(4) if pid != self.observer and counts.get(pid, 0) > 0)
        # progresso della mano: ~ carte giocate / 40
        try:
            progress = min(1.0, max(0.0, len(game_state.get('history', [])) / 40.0))
        except Exception:
            progress = 0.0
        # rende impossibili suit/rank completamente esauriti
        suit_possible = [suit_rem[s] > 0 for s in range(4)]
        rank_possible = [rank_rem[r] > 0 for r in range(10)]
        return unknown, suit_rem, rank_rem, max(1, active_unknown), progress, suit_possible, rank_possible

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

    def effective_sample_size(self) -> float:
        w = self.weights
        return 1.0 / np.sum(np.square(w))

    def resample_if_needed(self, ess_threshold: float = None):
        if ess_threshold is None:
            ess_threshold = max(1.0, self.ess_frac * self.num_particles)
        if self.effective_sample_size() < ess_threshold:
            idx = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=self.weights)
            self.particles = [self.particles[i].copy() for i in idx]
            self.weights[:] = 1.0 / self.num_particles

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

        new_weights = np.zeros_like(self.weights)
        # ricostruisci tavolo pre-mossa
        table_after = [card_to_id(c) for c in game_state.get('table', [])]
        if captured_ids:
            table_before = table_after + captured_ids
        else:
            # no-capture: played è stato aggiunto al tavolo -> rimuovilo
            tmp = list(table_after)
            if played in tmp:
                tmp.remove(played)
            table_before = tmp
        direct_same_rank = [cid for cid in table_before if ((cid // 4) + 1) == rank_played]

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
        # deduzione di semi esauriti per-player
        exhausted_by_pid = self._per_player_exhausted_suits(game_state)

        for i, part in enumerate(self.particles):
            ok = True
            if player != self.observer:
                hand_ids = list(part.get(player, []))
                if played not in hand_ids:
                    ok = False
                else:
                    # simula rimozione dalla mano
                    hand_ids.remove(played)
                    part[player] = hand_ids
            # vincoli di legalità aggiuntivi sulla tavolo pre-mossa
            if ok and direct_same_rank and not (ap_enabled and rank_played == 1 and set(captured_ids) == set(table_before)):
                if not (len(captured_ids) == 1 and captured_ids[0] in direct_same_rank):
                    ok = False
            if ok and (not direct_same_rank) and captured_ids:
                # subset-sum rigoroso
                if not set(captured_ids).issubset(set(table_before)):
                    ok = False
                elif sum(((cid // 4) + 1) for cid in captured_ids) != rank_played and not (ap_enabled and rank_played == 1 and set(captured_ids) == set(table_before)):
                    ok = False
            # coerenza con public state
            if ok:
                public_ids = set(card_to_id(c) for c in game_state.get('table', []))
                cs = game_state.get('captured_squads', None)
                try:
                    public_ids |= set(card_to_id(c) for c in cs[0] + cs[1])
                except Exception:
                    try:
                        public_ids |= set(card_to_id(c) for c in cs.get(0, []) + cs.get(1, []))
                    except Exception:
                        pass
                for pid, h in part.items():
                    if any(cid in public_ids for cid in h):
                        ok = False
                        break
            # vincoli storici per player
            if ok:
                try:
                    hist_by_pid = {pid: set() for pid in range(4)}
                    for mv in game_state.get('history', []):
                        pid0 = int(mv.get('player'))
                        hist_by_pid[pid0].add(card_to_id(mv.get('played_card')))
                    for pid, h in part.items():
                        if pid == self.observer:
                            continue
                        forb = hist_by_pid.get(pid, set())
                        if any(cid in forb for cid in h):
                            ok = False
                            break
                except Exception:
                    pass
            # vincolo persistente su rank proibiti emersi dalla history
            if ok and self._forbidden_ranks_by_pid.get(player):
                forb_ranks = self._forbidden_ranks_by_pid.get(player, set())
                if any(((cid // 4) + 1) in forb_ranks for cid in part.get(player, [])):
                    ok = False
            # vincoli addizionali: conteggi residui per suit/rank per-player e suit/rank impossibili
            if ok:
                for pid, h in part.items():
                    if pid == self.observer:
                        continue
                    suit_counts = [0, 0, 0, 0]
                    rank_counts = [0] * 10
                    for cid in h:
                        suit_counts[cid % 4] += 1
                        rank_counts[cid // 4] += 1
                    # impossibili globali
                    if any((not suit_possible[s] and suit_counts[s] > 0) for s in range(4)):
                        ok = False
                        break
                    if any((not rank_possible[r] and rank_counts[r] > 0) for r in range(10)):
                        ok = False
                        break
                    # per-player suits esauriti
                    exhausted = exhausted_by_pid.get(pid, [False, False, False, False])
                    if any(exhausted[s] and suit_counts[s] > 0 for s in range(4)):
                        ok = False
                        break
                    # cap su residui
                    if any(suit_counts[s] > suit_cap[s] for s in range(4)):
                        ok = False
                        break
                    if any(rank_counts[r] > rank_cap[r] for r in range(10)):
                        ok = False
                        break
            new_weights[i] = 1.0 if ok else 0.0
        s = new_weights.sum()
        if s > 0:
            self.weights = new_weights / s
        # aggiorna vincoli persistenti: se esiste almeno una carta dello stesso rank sul tavolo prima della mossa
        # e il player non ha effettuato presa diretta (e non AP taking-all), allora quel rank è vietato per il player
        try:
            if direct_same_rank and not (ap_enabled and rank_played == 1 and set(captured_ids) == set(table_before)):
                took_direct = (len(captured_ids) == 1 and captured_ids[0] in direct_same_rank)
                if not took_direct:
                    self._forbidden_ranks_by_pid[player].add(rank_played)
        except Exception:
            pass
        # aggiorna storico carte giocate per-player
        try:
            self._played_ids_by_pid[player].add(played)
        except Exception:
            pass
        # resampling adattivo
        if ess_threshold is None:
            vis, _ = self._visible_ids(game_state)
            unknown_frac = max(0.0, (40 - len(vis)) / 40.0)
            dyn = self.num_particles * max(0.3, min(0.8, unknown_frac))
            self.resample_if_needed(ess_threshold=dyn)
        else:
            self.resample_if_needed(ess_threshold=ess_threshold)

    def sample_determinization(self, num: int) -> List[Dict[int, List[int]]]:
        """Estrae 'num' assegnazioni (mani avversarie) in base ai pesi attuali."""
        idx = np.random.choice(self.num_particles, size=num, replace=True, p=self.weights)
        return [self.particles[i] for i in idx]

    def belief_summary(self, game_state: Dict, player_id: int):
        """
        Ritorna un vettore 120-dim (3 * 40) di marginali per carta∈{partner, opp1, opp2}.
        Ordine giocatori: cicla su pid!=player_id in ordine [ (player_id+1)%4, (player_id+2)%4, (player_id+3)%4 ].
        """
        others = [(player_id + 1) % 4, (player_id + 2) % 4, (player_id + 3) % 4]
        marg = np.zeros((3, 40), dtype=np.float32)
        # Le carte visibili hanno marginale 0 per tutti (sono già assegnate a osservatore/tavolo/captured)
        # Conta fra le particelle la presenza di ciascuna carta nelle mani di ciascun altro giocatore
        for i, pid in enumerate(others):
            counts = np.zeros(40, dtype=np.float32)
            for w, part in zip(self.weights, self.particles):
                for cid in part.get(pid, []):
                    counts[cid] += w
            marg[i] = counts.astype(np.float32)
        return torch.as_tensor(marg.reshape(-1), dtype=torch.float32, device=torch.device('cuda'))  # 120



