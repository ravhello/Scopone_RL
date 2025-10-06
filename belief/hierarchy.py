"""Hierarchical belief inference for Scopone.

This module provides deterministic, game-theoretic belief computations up to
level-3 nesting for perfect-recall players. The implementation relies on a
combinatorial assignment dynamic-programming solver that enumerates all hidden
card allocations consistent with the public information available to each
player. Probabilities are derived exactly (within floating point precision)
from log-counts of feasible assignments.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "compute_level1",
    "compute_level2",
    "compute_level3",
    "compute_belief_hierarchy",
    "combine_level_probs",
]

ALL_CARD_IDS: Tuple[int, ...] = tuple(range(40))
SUIT_TO_INT: Dict[str, int] = {"denari": 0, "coppe": 1, "spade": 2, "bastoni": 3}


def _card_to_id(card) -> int:
    if isinstance(card, int):
        return int(card)
    if not isinstance(card, (tuple, list)) or len(card) != 2:
        raise ValueError(f"Unsupported card format: {card!r}")
    rank, suit = card
    return (int(rank) - 1) * 4 + SUIT_TO_INT[str(suit)]


def _hand_to_ids(hand: Iterable) -> List[int]:
    return [_card_to_id(c) for c in hand]


def _get_team(player_id: int) -> int:
    return 0 if player_id in (0, 2) else 1


def _ensure_long_tensor(val, device=None) -> torch.Tensor:
    if torch.is_tensor(val):
        return val.to(device=device, dtype=torch.int64)
    return torch.as_tensor(val, dtype=torch.int64, device=device)


def _played_bits(game_state: Dict, player_id: int) -> int:
    played = game_state.get("_played_bits_by_player_t")
    if torch.is_tensor(played):
        return int(played[player_id].item())
    if isinstance(played, dict) and player_id in played:
        val = played[player_id]
        if torch.is_tensor(val):
            return int(val.item())
        return int(val)
    return 0


def _visible_cards_for_player(game_state: Dict, player_id: int, *, override_hand: Optional[Sequence[int]] = None) -> List[int]:
    visible: List[int] = []
    if override_hand is not None:
        visible.extend(int(cid) for cid in override_hand)
    else:
        visible.extend(_hand_to_ids(game_state.get("hands", {}).get(player_id, [])))
    visible.extend(_card_to_id(c) for c in game_state.get("table", []))
    captured = game_state.get("captured_squads", {})
    if isinstance(captured, dict):
        visible.extend(_card_to_id(c) for c in captured.get(0, []))
        visible.extend(_card_to_id(c) for c in captured.get(1, []))
    elif isinstance(captured, (list, tuple)) and len(captured) == 2:
        visible.extend(_card_to_id(c) for c in captured[0])
        visible.extend(_card_to_id(c) for c in captured[1])
    return sorted(set(visible))


def _others_in_play_order(perspective: int) -> List[int]:
    return [((perspective + i) % 4) for i in range(1, 4)]


def _build_allowed_matrix(
    game_state: Dict,
    perspective: int,
    unknown_cards: Sequence[int],
    others: Sequence[int],
) -> List[Tuple[bool, bool, bool]]:
    played_bits_per_player = [
        _played_bits(game_state, pid) for pid in range(4)
    ]
    allowed: List[Tuple[bool, bool, bool]] = []
    for cid in unknown_cards:
        row = []
        for pid in others:
            played_bits = played_bits_per_player[pid]
            if played_bits & (1 << cid):
                row.append(False)
            else:
                row.append(True)
        allowed.append(tuple(row))
    return allowed


def _logsumexp_pair(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _logsumexp(values: Iterable[float]) -> float:
    it = iter(values)
    try:
        first = next(it)
    except StopIteration:
        return -math.inf
    total = first
    for v in it:
        total = _logsumexp_pair(total, v)
    return total


@dataclass
class AssignmentDP:
    """Dynamic-programming helper for counting/sampling card allocations."""

    allowed: List[Tuple[bool, bool, bool]]
    counts: Tuple[int, int, int]

    def __post_init__(self) -> None:
        self.n = len(self.allowed)
        self.k0, self.k1, self.k2 = self.counts
        self._build_tables()

    def _build_tables(self) -> None:
        n, k0, k1, k2 = self.n, self.k0, self.k1, self.k2
        # Forward log-counts: F[t][a][b]
        self.logF = [[[-math.inf] * (k1 + 1) for _ in range(k0 + 1)] for _ in range(n + 1)]
        self.logF[0][0][0] = 0.0
        for t in range(n):
            allow_t = self.allowed[t]
            for a in range(k0 + 1):
                for b in range(k1 + 1):
                    c = t - a - b
                    if c < 0 or c > k2:
                        continue
                    cur = self.logF[t][a][b]
                    if cur == -math.inf:
                        continue
                    # assign to player 0
                    if allow_t[0] and a + 1 <= k0:
                        self.logF[t + 1][a + 1][b] = _logsumexp_pair(self.logF[t + 1][a + 1][b], cur)
                    # assign to player 1
                    if allow_t[1] and b + 1 <= k1:
                        self.logF[t + 1][a][b + 1] = _logsumexp_pair(self.logF[t + 1][a][b + 1], cur)
                    # assign to player 2
                    if allow_t[2] and c + 1 <= k2:
                        self.logF[t + 1][a][b] = _logsumexp_pair(self.logF[t + 1][a][b], cur)

        # Backward log-counts: B[t][r0][r1] with remaining counts
        self.logB = [[[-math.inf] * (k1 + 1) for _ in range(k0 + 1)] for _ in range(n + 1)]
        self.logB[n][0][0] = 0.0
        for t in range(n - 1, -1, -1):
            allow_t = self.allowed[t]
            for r0 in range(k0 + 1):
                for r1 in range(k1 + 1):
                    remaining_cards = n - t
                    r2 = remaining_cards - r0 - r1
                    if r2 < 0 or r2 > self.k2:
                        self.logB[t][r0][r1] = -math.inf
                        continue
                    options = []
                    # assign to player 0
                    if allow_t[0] and r0 > 0:
                        options.append(self.logB[t + 1][r0 - 1][r1])
                    # assign to player 1
                    if allow_t[1] and r1 > 0:
                        options.append(self.logB[t + 1][r0][r1 - 1])
                    # assign to player 2
                    if allow_t[2] and r2 > 0:
                        options.append(self.logB[t + 1][r0][r1])
                    self.logB[t][r0][r1] = _logsumexp(options)

        self.total_log = self.logF[n][k0][k1]

    def total_assignments(self) -> float:
        if self.total_log == -math.inf:
            return 0.0
        return math.exp(self.total_log)

    def card_player_logmass(self, index: int, player_slot: int) -> float:
        if self.total_log == -math.inf:
            return -math.inf
        k0, k1, k2 = self.k0, self.k1, self.k2
        log_mass = -math.inf
        allow = self.allowed[index]
        if not allow[player_slot]:
            return -math.inf
        for a in range(k0 + 1):
            for b in range(k1 + 1):
                c = index - a - b
                if c < 0 or c > k2:
                    continue
                prefix = self.logF[index][a][b]
                if prefix == -math.inf:
                    continue
                if player_slot == 0:
                    if a >= k0:
                        continue
                    rem0 = k0 - (a + 1)
                    rem1 = k1 - b
                elif player_slot == 1:
                    if b >= k1:
                        continue
                    rem0 = k0 - a
                    rem1 = k1 - (b + 1)
                else:  # player_slot == 2
                    assigned_p2 = c
                    if assigned_p2 >= k2:
                        continue
                    rem0 = k0 - a
                    rem1 = k1 - b
                if rem0 < 0 or rem1 < 0:
                    continue
                suffix = self.logB[index + 1][rem0][rem1]
                if suffix == -math.inf:
                    continue
                log_mass = _logsumexp_pair(log_mass, prefix + suffix)
        return log_mass

    def card_player_probabilities(self) -> List[Tuple[float, float, float]]:
        probs: List[Tuple[float, float, float]] = []
        if self.total_log == -math.inf:
            return [(0.0, 0.0, 0.0) for _ in range(self.n)]
        for i in range(self.n):
            lp0 = self.card_player_logmass(i, 0)
            lp1 = self.card_player_logmass(i, 1)
            lp2 = self.card_player_logmass(i, 2)
            row = []
            for lp in (lp0, lp1, lp2):
                if lp == -math.inf:
                    row.append(0.0)
                else:
                    row.append(math.exp(lp - self.total_log))
            # normalise small numerical drift
            s = sum(row)
            if s > 1e-9:
                row = [max(0.0, min(1.0, r / s)) for r in row]
            probs.append(tuple(row))
        return probs

    def sample_assignment(self, rng: random.Random) -> List[int]:
        """Sample a consistent assignment uniformly at random."""
        if self.total_log == -math.inf:
            raise RuntimeError("No feasible assignments to sample from")
        assignment = []
        rem0, rem1, rem2 = self.k0, self.k1, self.k2
        for t in range(self.n):
            allow_t = self.allowed[t]
            options: List[Tuple[int, float]] = []
            total_state = self.logB[t][rem0][rem1]
            if total_state == -math.inf:
                raise RuntimeError("Inconsistent DP state during sampling")
            if allow_t[0] and rem0 > 0:
                options.append((0, self.logB[t + 1][rem0 - 1][rem1]))
            if allow_t[1] and rem1 > 0:
                options.append((1, self.logB[t + 1][rem0][rem1 - 1]))
            if allow_t[2] and rem2 > 0:
                options.append((2, self.logB[t + 1][rem0][rem1]))
            if not options:
                raise RuntimeError("No legal assignment choices at step")
            weights = []
            for _, logw in options:
                if logw == -math.inf:
                    weights.append(0.0)
                else:
                    weights.append(math.exp(logw - total_state))
            total_w = sum(weights)
            if total_w <= 0:
                # fallback to uniform among options
                weights = [1.0 / len(options)] * len(options)
            else:
                weights = [w / total_w for w in weights]
            r = rng.random()
            cumulative = 0.0
            chosen_idx = len(options) - 1
            for idx_opt, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx_opt
                    break
            player_choice = options[chosen_idx][0]
            assignment.append(player_choice)
            if player_choice == 0:
                rem0 -= 1
            elif player_choice == 1:
                rem1 -= 1
            else:
                rem2 -= 1
        return assignment


def compute_level1(game_state: Dict, perspective: int, *, hand_override: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, List[int]]:
    """Compute first-order belief (current player's belief on others' hands).

    Returns a tensor shaped (3, 40) and the ordered list of other players.
    """
    others = _others_in_play_order(perspective)
    visible_ids = _visible_cards_for_player(game_state, perspective, override_hand=hand_override)
    visible_set = set(visible_ids)
    unknown_cards = [cid for cid in ALL_CARD_IDS if cid not in visible_set]
    if not unknown_cards:
        probs = torch.zeros((3, 40), dtype=torch.float32)
        return probs, others

    counts = tuple(len(game_state.get("hands", {}).get(pid, [])) for pid in others)
    allowed = _build_allowed_matrix(game_state, perspective, unknown_cards, others)
    dp = AssignmentDP(allowed, counts)
    card_probs = dp.card_player_probabilities()
    probs = torch.zeros((3, 40), dtype=torch.float32)
    for idx_card, cid in enumerate(unknown_cards):
        p0, p1, p2 = card_probs[idx_card]
        probs[0, cid] = float(p0)
        probs[1, cid] = float(p1)
        probs[2, cid] = float(p2)
    return probs, others


def compute_level2(game_state: Dict, perspective: int) -> torch.Tensor:
    """Compute second-order belief: others' belief about the perspective's hand."""
    level2 = torch.zeros((3, 40), dtype=torch.float32)
    _, others = compute_level1(game_state, perspective)
    for idx, opp in enumerate(others):
        probs_opp, others_from_opp = compute_level1(game_state, opp)
        if perspective not in others_from_opp:
            continue
        opp_idx = others_from_opp.index(perspective)
        level2[idx] = probs_opp[opp_idx]
    return level2


def compute_level3(
    game_state: Dict,
    perspective: int,
    *,
    num_samples: int = 32,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Approximate third-order belief via Monte Carlo expectation.

    For each opponent, we sample hands consistent with their information,
    compute the perspective player's level-1 belief under those sampled hands,
    and average the resulting distributions over the opponent in question.
    """
    rng = rng or random.Random()
    level3 = torch.zeros((3, 40), dtype=torch.float32)
    level1_persp, others = compute_level1(game_state, perspective)
    for idx, opp in enumerate(others):
        probs_opp, others_from_opp = compute_level1(game_state, opp)
        if perspective not in others_from_opp:
            # opponent already terminal; fallback to perspective's level1
            level3[idx] = level1_persp[idx]
            continue
        opp_perspective_index = others_from_opp.index(perspective)
        # Build DP for opponent view
        visible_ids = _visible_cards_for_player(game_state, opp)
        visible_set = set(visible_ids)
        unknown_cards = [cid for cid in ALL_CARD_IDS if cid not in visible_set]
        if not unknown_cards:
            level3[idx] = level1_persp[idx]
            continue
        counts = tuple(len(game_state.get("hands", {}).get(pid, [])) for pid in others_from_opp)
        allowed = _build_allowed_matrix(game_state, opp, unknown_cards, others_from_opp)
        dp = AssignmentDP(allowed, counts)
        total_assignments = dp.total_assignments()
        if total_assignments == 0.0:
            level3[idx] = level1_persp[idx]
            continue
        accum = torch.zeros(40, dtype=torch.float32)
        effective_samples = 0
        for _ in range(max(1, num_samples)):
            try:
                assignment = dp.sample_assignment(rng)
            except RuntimeError:
                break
            sample_hand = [
                unknown_cards[pos]
                for pos, slot in enumerate(assignment)
                if slot == opp_perspective_index
            ]
            required_size = len(game_state.get("hands", {}).get(perspective, []))
            if len(sample_hand) != required_size:
                continue
            probs_sample, others_from_persp = compute_level1(
                game_state,
                perspective,
                hand_override=sample_hand,
            )
            if opp not in others_from_persp:
                continue
            opp_idx_in_persp = others_from_persp.index(opp)
            accum += probs_sample[opp_idx_in_persp]
            effective_samples += 1
        if effective_samples == 0:
            level3[idx] = level1_persp[idx]
        else:
            level3[idx] = accum / float(effective_samples)
    return level3


def compute_belief_hierarchy(
    game_state: Dict,
    perspective: int,
    *,
    num_samples_level3: int = 32,
    rng: Optional[random.Random] = None,
) -> Dict[str, torch.Tensor]:
    """Return hierarchical beliefs (levels 1-3) for the selected player."""
    lvl1, _ = compute_level1(game_state, perspective)
    lvl2 = compute_level2(game_state, perspective)
    lvl3 = compute_level3(game_state, perspective, num_samples=num_samples_level3, rng=rng)
    return {
        "level1": lvl1,
        "level2": lvl2,
        "level3": lvl3,
    }


def combine_level_probs(level1: torch.Tensor, level3: torch.Tensor, alpha: float = 0.65) -> torch.Tensor:
    """Blend level-1 and level-3 beliefs via geometric interpolation."""
    eps = 1e-9
    alpha = float(max(0.0, min(1.0, alpha)))
    lvl1 = level1.clamp_min(eps)
    lvl3 = level3.clamp_min(eps)
    blend = torch.pow(lvl1, alpha) * torch.pow(lvl3, 1.0 - alpha)
    row_sums = blend.sum(dim=1, keepdim=True).clamp_min(eps)
    return blend / row_sums


def sample_determinization(env, alpha: float = 0.65, noise_scale: float = 0.0):
    """Return a deterministic assignment of unknown cards based on hierarchical beliefs."""
    env._attach_state_views()
    current = env.current_player
    hierarchy = compute_belief_hierarchy(env.game_state, current)
    blend = combine_level_probs(hierarchy['level1'], hierarchy['level3'], alpha=alpha)
    probs = blend.detach().cpu().numpy()
    suit_map = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}

    def _card_id(card) -> int:
        if isinstance(card, int):
            return int(card)
        rank, suit = card
        return int((int(rank) - 1) * 4 + suit_map[str(suit)])

    visible = set(_card_id(c) for c in env.game_state['hands'][current])
    visible.update(_card_id(c) for c in env.game_state['table'])
    captured = env.game_state['captured_squads']
    visible.update(_card_id(c) for c in captured[0])
    visible.update(_card_id(c) for c in captured[1])

    unknown_ids = [cid for cid in range(40) if cid not in visible]
    others = [(current + 1) % 4, (current + 2) % 4, (current + 3) % 4]
    caps = [len(env.game_state['hands'][pid]) for pid in others]
    n = len(unknown_ids)
    if n == 0:
        return {pid: [] for pid in others}
    total_caps = sum(caps)
    if total_caps != n:
        caps = caps[:]
        if total_caps > 0:
            scale = n / float(total_caps)
            caps = [max(0, int(round(c * scale))) for c in caps]
        else:
            base = n // 3
            rem = n - 3 * base
            caps = [base, base, base]
            for i in range(rem):
                caps[i] += 1
        diff = n - sum(caps)
        for i in range(abs(diff)):
            idx = i % 3
            caps[idx] += 1 if diff > 0 else -1
        caps = [max(0, c) for c in caps]
    cap0, cap1, cap2 = caps

    import math
    import numpy as _np

    eps = 1e-9
    costs = []
    for cid in unknown_ids:
        pc = probs[:, cid]
        s = float(pc.sum())
        if s <= eps:
            card_probs = _np.full(3, 1.0 / 3.0, dtype=_np.float32)
        else:
            card_probs = (pc / s).astype(_np.float32)
        c = [-math.log(max(eps, float(card_probs[i]))) for i in range(3)]
        if noise_scale > 0:
            g = _np.random.gumbel(size=3) * noise_scale
            c = [c[i] + float(g[i]) for i in range(3)]
        costs.append(c)

    INF = 1e12
    dp = [[[INF] * (cap1 + 1) for _ in range(cap0 + 1)] for __ in range(n + 1)]
    bk = [[[-1] * (cap1 + 1) for _ in range(cap0 + 1)] for __ in range(n + 1)]
    dp[0][0][0] = 0.0
    for t in range(n):
        c0, c1, c2 = costs[t]
        for a in range(0, min(t, cap0) + 1):
            for b in range(0, min(t - a, cap1) + 1):
                cur = dp[t][a][b]
                if cur >= INF:
                    continue
                if a + 1 <= cap0 and dp[t + 1][a + 1][b] > cur + c0:
                    dp[t + 1][a + 1][b] = cur + c0
                    bk[t + 1][a + 1][b] = 0
                if b + 1 <= cap1 and dp[t + 1][a][b + 1] > cur + c1:
                    dp[t + 1][a][b + 1] = cur + c1
                    bk[t + 1][a][b + 1] = 1
                assigned2 = t - a - b
                if assigned2 + 1 <= cap2 and dp[t + 1][a][b] > cur + c2:
                    dp[t + 1][a][b] = cur + c2
                    bk[t + 1][a][b] = 2
    if dp[n][cap0][cap1] >= INF:
        return None
    det = {pid: [] for pid in others}
    a, b = cap0, cap1
    for t in range(n, 0, -1):
        choice = bk[t][a][b]
        cid = unknown_ids[t - 1]
        if choice == 0:
            det[others[0]].append(cid)
            a -= 1
        elif choice == 1:
            det[others[1]].append(cid)
            b -= 1
        else:
            det[others[2]].append(cid)
    return det
