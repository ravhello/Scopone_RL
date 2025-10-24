import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from environment import ScoponeEnvMA
from rewards import compute_final_score_breakdown
from utils.device import get_compute_device
from utils.fallback import notify_fallback

_DEFAULT_MCTS_MOVE_TIMEOUT_S = 120.0


class MCTSTimeoutError(RuntimeError):
    def __init__(self, timeout_s: float, elapsed_s: float):
        super().__init__(f"IS-MCTS move exceeded timeout of {timeout_s:.2f}s (elapsed {elapsed_s:.2f}s)")
        self.timeout_s = float(timeout_s)
        self.elapsed_s = float(elapsed_s)


class _MctsMoveTimer:
    __slots__ = ('timeout_s', '_start', '_deadline')

    def __init__(self, timeout_s: Optional[float]):
        self._start = time.perf_counter()
        if timeout_s is None or timeout_s <= 0:
            self.timeout_s = None
            self._deadline = None
        else:
            self.timeout_s = float(timeout_s)
            self._deadline = self._start + self.timeout_s

    def check(self) -> None:
        if self._deadline is None:
            return
        now = time.perf_counter()
        if now >= self._deadline:
            elapsed = now - self._start
            raise MCTSTimeoutError(self.timeout_s, elapsed)

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def remaining(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.perf_counter())


def _resolve_move_timeout_s() -> Optional[float]:
    raw = os.environ.get('SCOPONE_MCTS_MOVE_TIMEOUT_S', '') or ''
    raw = raw.strip()
    if not raw:
        return _DEFAULT_MCTS_MOVE_TIMEOUT_S
    lowered = raw.lower()
    if lowered in ('off', 'none', 'no', 'false', '0'):
        return None
    try:
        val = float(raw)
        if val <= 0:
            return None
        return float(val)
    except ValueError:
        return _DEFAULT_MCTS_MOVE_TIMEOUT_S


class ISMCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action  # azione che ha portato a questo nodo (80-dim np.array)
        self.children: List[ISMCTSNode] = []
        self.N = 0  # visite
        self.W = 0.0  # somma valori (pesi inclusi)
        self.Q = 0.0  # valore medio
        self.P = 0.0  # prior (dalla policy)
        self.weight = 0.0  # somma pesi determinizzazioni

    def ucb_score(self, c_puct: float = 1.0) -> float:
        if self.parent is None:
            return float('-inf')
        prior = self.P
        total_N = max(1, self.parent.N)
        return self.Q + c_puct * prior * math.sqrt(total_N) / (1 + self.N)


def _extract_determinization(det_result: Any) -> Tuple[Optional[Dict[int, Iterable[int]]], Optional[float]]:
    if det_result is None:
        return None, None
    if isinstance(det_result, tuple):
        assignments = det_result[0]
        logp = det_result[1] if len(det_result) > 1 else None
    elif isinstance(det_result, dict):
        if all(isinstance(k, int) for k in det_result.keys()):
            assignments = det_result
            logp = None
        else:
            assignments = det_result.get('assignments') or det_result.get('hands')
            logp = det_result.get('logp')
    else:
        assignments = det_result
        logp = None
    if assignments is not None:
        assignments = {int(pid): list(cards) for pid, cards in assignments.items()}
    logp = float(logp) if logp is not None else None
    return assignments, logp


def _normalize_log_weights(log_weights: Sequence[float], count: int) -> Tuple[List[float], float, float]:
    if count <= 0:
        return [], 0.0, 0.0
    if not log_weights:
        return [0.0] * count, (1.0 if count > 1 else 0.0), 0.0
    finite_logs = [lw for lw in log_weights if math.isfinite(lw)]
    if not finite_logs:
        return [0.0] * count, (1.0 if count > 1 else 0.0), 0.0

    weights: List[float] = []
    total_mass = 0.0
    for lw in log_weights:
        if math.isfinite(lw):
            w = math.exp(lw)
            if not math.isfinite(w):
                w = 0.0
        else:
            w = 0.0
        w = float(max(0.0, w))
        weights.append(w)
        total_mass += w

    entropy = 0.0
    if total_mass > 0.0:
        norm_w = [w / total_mass for w in weights]
        for w in norm_w:
            if w > 1e-12:
                entropy -= w * math.log(w)
        if count > 1:
            entropy /= math.log(count)
        else:
            entropy = 0.0
    else:
        entropy = 1.0 if count > 1 else 0.0

    return weights, float(entropy), float(total_mass)


def _apply_determinization(env: ScoponeEnvMA, assignment: Optional[Dict[int, Iterable[int]]]) -> None:
    if not assignment:
        return
    hands = env.game_state['hands']
    for pid, cards in assignment.items():
        hands[int(pid)] = list(cards)
    env._rebuild_id_caches()


def _final_score_diff(env: ScoponeEnvMA) -> float:
    breakdown = compute_final_score_breakdown(env.game_state, env.rules)
    return float(breakdown[0]['total'] - breakdown[1]['total'])


def _remaining_moves(env: ScoponeEnvMA) -> int:
    hands = env.game_state.get('hands', None)
    if isinstance(hands, (list, tuple)):
        return int(sum(len(h) for h in hands))
    if isinstance(hands, dict):
        return int(sum(len(h) for h in hands.values()))
    return 0


def _state_signature(env: ScoponeEnvMA) -> Tuple[Any, ...]:
    table_bits = int(env._table_bits) if hasattr(env, '_table_bits') else int(env._table_bits_t.item())
    if hasattr(env, '_hands_bits'):
        hands_bits_ref = env._hands_bits
        hands_bits = tuple(int(hands_bits_ref[pid]) for pid in range(4))
    else:
        hands_bits = tuple(int(env._hands_bits_t[pid].item()) for pid in range(4))
    captured_bits = tuple(int(env._captured_bits_t[i].item()) for i in range(2))
    history_len = len(env.game_state.get('history', [])) if isinstance(env.game_state, dict) else 0
    return (int(env.current_player), table_bits, hands_bits, captured_bits, history_len, bool(env.done))


def _ensure_tensor_action(action: Any) -> torch.Tensor:
    if torch.is_tensor(action):
        return action
    return torch.as_tensor(action, dtype=torch.float32)


def _solve_to_terminal(env: ScoponeEnvMA,
                       cache: Dict[Tuple[Any, ...], Tuple[float, int]]) -> Tuple[float, int]:
    if env.done:
        return _final_score_diff(env), 1
    key = _state_signature(env)
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        return cached
    legals = env.get_valid_actions()
    if torch.is_tensor(legals):
        legals_list = list(legals.unbind(0))
    else:
        legals_list = list(legals)
    if not legals_list:
        val = _final_score_diff(env)
        out = (val, 1)
        if cache is not None:
            cache[key] = out
        return out
    team_sign = 1.0 if env.current_player in (0, 2) else -1.0
    best_val = float('-inf') if team_sign > 0 else float('inf')
    leaves_acc = 0
    for act in legals_list:
        child = env.clone()
        child.step(_ensure_tensor_action(act))
        val, leaves = _solve_to_terminal(child, cache)
        leaves_acc += leaves
        if team_sign > 0:
            if val > best_val:
                best_val = val
        else:
            if val < best_val:
                best_val = val
    if not math.isfinite(best_val):
        best_val = _final_score_diff(env)
    out = (best_val, leaves_acc)
    if cache is not None:
        cache[key] = out
    return out


def _auto_exact_temperature(values: Sequence[float], root_sign: float) -> float:
    if len(values) <= 1:
        return 0.0
    signed = [root_sign * v for v in values]
    best = max(signed)
    remaining = [x for x in signed if x != best]
    if not remaining:
        return 0.0
    second = max(remaining)
    gap = best - second
    if gap >= 3.0:
        return 0.0
    if gap >= 1.5:
        return 0.05
    if gap >= 0.75:
        return 0.1
    return 0.2


def _try_exact_evaluation(root_env: ScoponeEnvMA,
                          legals: Sequence[Any],
                          det_pack: Sequence[Dict[str, Any]],
                          root_player: int) -> Optional[Tuple[torch.Tensor, List[float]]]:
    if not det_pack or not legals:
        return None
    legals_tensors = [_ensure_tensor_action(a) for a in legals]
    caches = [dict() for _ in det_pack]
    values = [0.0 for _ in legals_tensors]
    weights = [max(0.0, float(det.get('weight', 0.0))) for det in det_pack]
    sum_w = sum(weights)
    if not math.isfinite(sum_w) or sum_w <= 0:
        return None
    for det_idx, det in enumerate(det_pack):
        base_env = root_env.clone()
        _apply_determinization(base_env, det.get('assignment'))
        cache = caches[det_idx]
        for idx, act in enumerate(legals_tensors):
            sim_env = base_env.clone()
            sim_env.step(act)
            if sim_env.done:
                val = _final_score_diff(sim_env)
            else:
                val, _ = _solve_to_terminal(sim_env, cache)
            values[idx] += weights[det_idx] * val
    if sum_w > 0:
        values = [v / sum_w for v in values]
    root_sign = 1.0 if root_player in (0, 2) else -1.0
    best_idx = 0
    best_score = float('-inf')
    for idx, val in enumerate(values):
        score = root_sign * val
        if score > best_score:
            best_score = score
            best_idx = idx
    temp = _auto_exact_temperature(values, root_sign)
    if temp > 1e-6:
        logits = [root_sign * v / temp for v in values]
        max_logit = max(logits)
        exp_vals = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_vals)
        if total > 0:
            probs = [ev / total for ev in exp_vals]
        else:
            probs = [0.0] * len(values)
            probs[best_idx] = 1.0
    else:
        probs = [0.0] * len(values)
        probs[best_idx] = 1.0
    return legals_tensors[best_idx], probs


def _allocate_simulation_counts(total: int, weights: Sequence[float]) -> List[int]:
    n = len(weights)
    if n == 0 or total <= 0:
        return [0] * n
    if not any(w > 0 for w in weights):
        weights = [1.0 / n] * n
    scaled = [w * total for w in weights]
    counts = [0] * n
    for _ in range(total):
        idx = max(range(n), key=lambda i: (scaled[i] - counts[i]))
        counts[idx] += 1
    return counts


def run_is_mcts(env: ScoponeEnvMA,
                policy_fn,
                value_fn,
                num_simulations: int = 200,
                c_puct: float = 1.0,
                belief: object = None,
                num_determinization: int = 1,
                root_temperature: float = 0.0,
                prior_smooth_eps: float = 0.0,
                robust_child: bool = True,
                root_dirichlet_alpha: float = 0.0,
                root_dirichlet_eps: float = 0.0,
                return_stats: bool = False,
                belief_sampler=None,
                exact_only: bool = False):
    if int(num_simulations) < 0:
        raise ValueError("IS-MCTS: num_simulations must be >= 0")
    if int(num_determinization) <= 0:
        raise ValueError("IS-MCTS: num_determinization must be >= 1")
    if float(c_puct) < 0:
        raise ValueError("IS-MCTS: c_puct must be >= 0")
    if float(root_temperature) < 0:
        raise ValueError("IS-MCTS: root_temperature must be >= 0")
    if float(prior_smooth_eps) < 0 or float(prior_smooth_eps) > 1:
        raise ValueError("IS-MCTS: prior_smooth_eps must be in [0,1]")
    if float(root_dirichlet_eps) < 0 or float(root_dirichlet_eps) > 1:
        raise ValueError("IS-MCTS: root_dirichlet_eps must be in [0,1]")
    if float(root_dirichlet_alpha) < 0:
        raise ValueError("IS-MCTS: root_dirichlet_alpha must be >= 0")

    depth_limit_env = os.environ.get('SCOPONE_MCTS_MAX_DEPTH', '').strip()
    if depth_limit_env:
        try:
            depth_limit = max(0, int(depth_limit_env))
        except ValueError:
            depth_limit = 0
    else:
        depth_limit = 0

    move_timeout_s = _resolve_move_timeout_s()
    move_timer = _MctsMoveTimer(move_timeout_s) if move_timeout_s is not None else None
    timer_check = move_timer.check if move_timer is not None else None
    if timer_check:
        timer_check()

    root_env = env.clone()
    obs = root_env._get_observation(root_env.current_player)
    legals = root_env.get_valid_actions()
    is_empty = (hasattr(legals, 'numel') and legals.numel() == 0) or (hasattr(legals, '__len__') and len(legals) == 0)
    if is_empty:
        raise ValueError("No legal actions for IS-MCTS")

    def action_key(vec):
        import torch as _torch
        if _torch.is_tensor(vec):
            nz = _torch.nonzero(vec > 0, as_tuple=False).flatten().tolist()
            return tuple(int(i) for i in nz)
        else:
            return tuple(i for i, v in enumerate(vec) if v > 0)

    if torch.is_tensor(legals):
        legals_seq = list(legals.unbind(0))
    else:
        legals_seq = list(legals)
    if not legals_seq:
        raise ValueError("No legal actions for IS-MCTS")

    try:
        priors = policy_fn(obs, legals)
    except Exception as e:
        raise RuntimeError("IS-MCTS: policy_fn failed to produce priors") from e

    import numpy as _np
    if isinstance(priors, _np.ndarray):
        priors_len = len(priors)
        if priors_len != len(legals_seq):
            raise RuntimeError(f"IS-MCTS: priors length {priors_len} != num legals {len(legals_seq)}")
        if not _np.isfinite(priors).all():
            raise RuntimeError("IS-MCTS: priors contain non-finite values")
        if prior_smooth_eps > 0 and priors_len > 1:
            priors = (1 - prior_smooth_eps) * priors + prior_smooth_eps * (1.0 / priors_len)
        if root_dirichlet_eps > 0 and priors_len > 1 and root_dirichlet_alpha > 0:
            noise = _np.random.dirichlet([root_dirichlet_alpha] * priors_len)
            priors = (1 - root_dirichlet_eps) * priors + root_dirichlet_eps * noise
    else:
        if not torch.is_tensor(priors):
            pri_dev = get_compute_device()
            priors = torch.as_tensor(priors, dtype=torch.float32, device=pri_dev)
        device = priors.device
        priors_len = int(priors.numel())
        if priors_len != len(legals_seq):
            raise RuntimeError(f"IS-MCTS: priors length {priors_len} != num legals {len(legals_seq)}")
        if not torch.isfinite(priors).all():
            raise RuntimeError("IS-MCTS: priors contain non-finite values")
        if prior_smooth_eps > 0 and priors_len > 1:
            priors = (1.0 - prior_smooth_eps) * priors + prior_smooth_eps * (1.0 / priors_len)
        if root_dirichlet_eps > 0 and priors_len > 1 and root_dirichlet_alpha > 0:
            alpha = torch.full((priors_len,), float(root_dirichlet_alpha), device=device, dtype=priors.dtype)
            noise = torch.distributions.Dirichlet(alpha).sample()
            priors = (1.0 - root_dirichlet_eps) * priors + root_dirichlet_eps * noise
    root = ISMCTSNode(parent=None, action=None)
    root.N = 0
    node_cache: Dict[Tuple[Any, ...], ISMCTSNode] = {}
    pk_root = (int(root_env.current_player), tuple(sorted(root_env.game_state.get('table', []))))
    prior_list = priors.tolist() if hasattr(priors, 'tolist') else list(priors)
    ak_order = [action_key(a) for a in legals_seq]
    if torch.is_tensor(legals):
        legals_seq = list(legals.unbind(0))
    else:
        legals_seq = list(legals)
    if not legals_seq:
        raise ValueError("No legal actions for IS-MCTS")
    if isinstance(priors, _np.ndarray):
        if priors_len != len(legals_seq):
            raise RuntimeError(f"IS-MCTS: priors length {priors_len} != num legals {len(legals_seq)}")
    else:
        if priors_len != len(legals_seq):
            raise RuntimeError(f"IS-MCTS: priors length {priors_len} != num legals {len(legals_seq)}")
    ak_order = [action_key(a) for a in legals_seq]

    def _select_greedy_from_priors():
        if not legals_seq:
            raise RuntimeError("IS-MCTS: greedy fallback requested with no legal actions")
        if isinstance(priors, _np.ndarray):
            best_idx = int(_np.argmax(priors))
        else:
            priors_tensor = priors if torch.is_tensor(priors) else torch.as_tensor(priors, dtype=torch.float32)
            best_idx = int(torch.argmax(priors_tensor).item())
        best_idx = max(0, min(best_idx, len(legals_seq) - 1))
        best_action = legals_seq[best_idx]
        if return_stats:
            probs_np = _np.zeros(len(legals_seq), dtype=_np.float32)
            probs_np[best_idx] = 1.0
            return best_action, probs_np
        return best_action

    for a, p in zip(legals_seq, prior_list):
        ak = action_key(a)
        key = (pk_root, ak)
        if key in node_cache:
            child = node_cache[key]
            child.P = float(p)
            if child not in root.children:
                root.children.append(child)
        else:
            child = ISMCTSNode(parent=root, action=a)
            child.P = float(p)
            root.children.append(child)
            node_cache[key] = child

    det_outputs = []
    det_log_weights = []
    if callable(belief_sampler):
        for _ in range(int(num_determinization)):
            det = belief_sampler(root_env)
            assignments, logp = _extract_determinization(det)
            det_outputs.append({'assignment': assignments, 'logp': logp})
            det_log_weights.append(logp if logp is not None else 0.0)
    if not det_outputs:
        det_outputs = [{'assignment': None, 'logp': 0.0}]
        det_log_weights = [0.0]
    det_weights, entropy, mass_observed = _normalize_log_weights(det_log_weights, len(det_outputs))
    for det, w in zip(det_outputs, det_weights):
        det['weight'] = w

    cover_frac_env = os.environ.get('SCOPONE_MCTS_EXACT_COVER_FRAC', None)
    try:
        cover_frac = float(str(cover_frac_env).strip()) if cover_frac_env is not None else 0.0
    except (ValueError, TypeError):
        cover_frac = 0.0
    if cover_frac > 1.0:
        cover_frac = cover_frac / 100.0
    if cover_frac < 0.0:
        cover_frac = 0.0
    cover_frac = min(1.0, cover_frac)
    auto_exact_by_coverage = False
    if cover_frac > 0.0 and det_outputs:
        observed_mass = max(0.0, min(1.0, mass_observed))
        if observed_mass >= cover_frac:
            auto_exact_by_coverage = True

    remaining_moves_root = _remaining_moves(root_env)
    should_try_exact = bool(det_outputs) and auto_exact_by_coverage
    if should_try_exact:
        exact_candidate = _try_exact_evaluation(root_env,
                                                legals_seq,
                                                det_outputs,
                                                int(root_env.current_player))
        if exact_candidate is not None:
            best_action, probs = exact_candidate
            if return_stats:
                probs_t = torch.tensor(probs, dtype=torch.float32, device=get_compute_device())
                ch_keys = [action_key(ch.action) for ch in root.children]
                probs_np = probs_t.detach().cpu().numpy()
                agg = {}
                for k, p in zip(ch_keys, probs_np):
                    agg[k] = float(agg.get(k, 0.0) + float(p))
                import numpy as _np
                if ak_order:
                    p_vec = _np.asarray([agg.get(k, 0.0) for k in ak_order], dtype=_np.float32)
                    s = float(p_vec.sum())
                    if s > 0:
                        p_vec = p_vec / s
                else:
                    p_vec = probs_np
            return best_action, p_vec
            return best_action
    if exact_only and not should_try_exact:
        return _select_greedy_from_priors()

    try:
        if int(num_simulations) <= 0:
            best = max(root.children, key=(lambda n: n.P))
            if return_stats:
                import numpy as _np
                probs_np = _np.zeros(len(ak_order), dtype=_np.float32)
                if ak_order:
                    target_idx = ak_order.index(action_key(best.action))
                    probs_np[target_idx] = 1.0
                return best.action, probs_np
            return best.action

        entropy_factor = max(0.0, 1.0 - entropy)
        progress_factor = 0.0
        if remaining_moves_root > 0:
            progress_factor = max(0.0, 1.0 - min(1.0, remaining_moves_root / 40.0))
        scale = 1.0 + 0.75 * entropy_factor + 0.25 * progress_factor
        if remaining_moves_root <= 12:
            scale += 0.15
        scale = min(scale, 2.0)
        sim_budget = max(1, int(round(int(num_simulations) * scale)))
        det_counts = _allocate_simulation_counts(sim_budget, [det['weight'] for det in det_outputs])

        def simulate_with_det(det_assignment: Optional[Dict[int, Iterable[int]]], det_weight: float) -> None:
            nonlocal node_cache
            if timer_check:
                timer_check()
            sim_env = root_env.clone()
            _apply_determinization(sim_env, det_assignment)
            node = root
            path = [node]
            reached_depth_cap = False
            while True:
                if timer_check:
                    timer_check()
                if depth_limit > 0 and (len(path) - 1) >= depth_limit:
                    reached_depth_cap = True
                    break
                if not node.children:
                    break
                legals_cur = sim_env.get_valid_actions()
                if torch.is_tensor(legals_cur):
                    legals_list = list(legals_cur.unbind(0))
                else:
                    legals_list = list(legals_cur)
                legal_keys = set(action_key(a) for a in legals_list)
                legal_children = [ch for ch in node.children if action_key(ch.action) in legal_keys or ch.action is None]
                if not legal_children:
                    break
                node = max(legal_children, key=lambda n: n.ucb_score(c_puct))
                path.append(node)
                if node.action is not None:
                    _, _, done_flag, _ = sim_env.step(_ensure_tensor_action(node.action))
                    if done_flag:
                        break
            if (not sim_env.done) and (not reached_depth_cap):
                obs_s = sim_env._get_observation(sim_env.current_player)
                legals_s = sim_env.get_valid_actions()
                if torch.is_tensor(legals_s):
                    legals_seq = list(legals_s.unbind(0))
                else:
                    legals_seq = list(legals_s)
                if legals_seq:
                    if timer_check:
                        timer_check()
                    try:
                        priors_s = policy_fn(obs_s, legals_s)
                    except Exception as e:
                        raise RuntimeError("IS-MCTS: policy_fn failed during expansion priors") from e
                    if isinstance(priors_s, _np.ndarray):
                        if prior_smooth_eps > 0 and len(priors_s) > 1:
                            priors_s = (1 - prior_smooth_eps) * priors_s + prior_smooth_eps * (1.0 / len(priors_s))
                        visits_here = max(1, node.N)
                        k_allow = min(len(legals_seq), int(3.0 * (visits_here ** 0.5)) + 1)
                        top_idx = _np.argsort(-priors_s)[:k_allow]
                        legals_sel = [legals_seq[i] for i in top_idx]
                        priors_sel = priors_s[top_idx]
                    else:
                        if not torch.is_tensor(priors_s):
                            pri_dev2 = get_compute_device()
                            priors_s = torch.as_tensor(priors_s, dtype=torch.float32, device=pri_dev2)
                        if prior_smooth_eps > 0 and priors_s.numel() > 1:
                            priors_s = (1.0 - prior_smooth_eps) * priors_s + prior_smooth_eps * (1.0 / priors_s.numel())
                        visits_here = max(1, node.N)
                        k_allow = min(int(priors_s.numel()), int(3.0 * (visits_here ** 0.5)) + 1)
                        top_idx = torch.argsort(priors_s, descending=True)[:k_allow]
                        legals_sel = [legals_seq[int(i)] for i in top_idx.tolist()]
                        priors_sel = priors_s[top_idx].detach().cpu().numpy()
                    cur = sim_env.current_player
                    table_ids = tuple(sorted(sim_env.game_state.get('table', [])))
                    cs = sim_env.game_state.get('captured_squads', [[], []])
                    if isinstance(cs, dict):
                        cs0 = tuple(sorted(cs.get(0, [])))
                        cs1 = tuple(sorted(cs.get(1, [])))
                    else:
                        cs0 = tuple(sorted(cs[0]))
                        cs1 = tuple(sorted(cs[1]))
                    pk = (cur, table_ids, cs0, cs1)
                    for a, p in zip(legals_sel, (priors_sel.tolist() if hasattr(priors_sel, 'tolist') else priors_sel)):
                        ak = action_key(a)
                        key = (pk, ak)
                        if key in node_cache:
                            ch = node_cache[key]
                            ch.P = float(p)
                            if ch not in node.children:
                                node.children.append(ch)
                        else:
                            ch = ISMCTSNode(parent=node, action=a)
                            ch.P = float(p)
                            node.children.append(ch)
                            node_cache[key] = ch
            if sim_env.done:
                v = _final_score_diff(sim_env)
            else:
                if timer_check:
                    timer_check()
                obs_v = sim_env._get_observation(sim_env.current_player)
                v = float(value_fn(obs_v, sim_env))
            weight = max(det_weight, 1e-6)
            for n in reversed(path):
                n.N += 1
                n.W += weight * v
                n.weight += weight
                if n.weight > 0:
                    n.Q = n.W / n.weight
                else:
                    n.Q = n.W / max(1, n.N)

        for det, count in zip(det_outputs, det_counts):
            for _ in range(count):
                if timer_check:
                    timer_check()
                simulate_with_det(det.get('assignment'), float(det.get('weight', 1.0)))

        device = get_compute_device()
        if timer_check:
            timer_check()
        if root_temperature and root_temperature > 1e-6:
            visits_t = torch.tensor([ch.N for ch in root.children], dtype=torch.float32, device=device)
            logits = torch.pow(visits_t + 1e-9, 1.0 / float(root_temperature))
            probs_t = logits / torch.clamp_min(logits.sum(), 1e-9)
            probs_t = probs_t.nan_to_num(0.0)
            probs_t = torch.clamp(probs_t, min=0.0)
            s = probs_t.sum()
            if not torch.isfinite(s) or s <= 0:
                raise RuntimeError("IS-MCTS: invalid root selection probabilities (NaN/Inf or zero-sum)")
            probs_t = probs_t / s
            idx = int(torch.multinomial(probs_t, num_samples=1).item())
            if return_stats:
                import numpy as _np
                ch_keys = [action_key(ch.action) for ch in root.children]
                probs_np = probs_t.detach().cpu().numpy()
                agg = {}
                for k, p in zip(ch_keys, probs_np):
                    agg[k] = float(agg.get(k, 0.0) + float(p))
                if ak_order:
                    p_vec = _np.asarray([agg.get(k, 0.0) for k in ak_order], dtype=_np.float32)
                    s = float(p_vec.sum())
                    if s > 0:
                        p_vec = p_vec / s
                else:
                    p_vec = probs_np
                return root.children[idx].action, p_vec
            return root.children[idx].action
        else:
            best = max(root.children, key=(lambda n: n.N) if robust_child else (lambda n: n.Q))
            if return_stats:
                visits_t = torch.tensor([ch.N for ch in root.children], dtype=torch.float32, device=device)
                probs_t = visits_t / torch.clamp_min(visits_t.sum(), 1e-9)
                import numpy as _np
                ch_keys = [action_key(ch.action) for ch in root.children]
                probs_np = probs_t.detach().cpu().numpy()
                agg = {}
                for k, p in zip(ch_keys, probs_np):
                    agg[k] = float(agg.get(k, 0.0) + float(p))
                if ak_order:
                    p_vec = _np.asarray([agg.get(k, 0.0) for k in ak_order], dtype=_np.float32)
                    s = float(p_vec.sum())
                    if s > 0:
                        p_vec = p_vec / s
                else:
                    p_vec = probs_np
                return best.action, p_vec
            return best.action
    except MCTSTimeoutError as exc:
        elapsed = exc.elapsed_s
        limit = exc.timeout_s
        timer_elapsed = move_timer.elapsed() if move_timer is not None else None
        print(f"[WARN][is_mcts] move timed out after {elapsed:.2f}s (limit {limit:.2f}s); falling back to greedy policy.", flush=True)
        ctx_parts = [
            f"player={root_env.current_player}",
            f"sims_requested={num_simulations}",
            f"dets={num_determinization}",
            f"exact_only={exact_only}",
            f"remaining_moves={remaining_moves_root}",
            f"belief_sampler={'yes' if callable(belief_sampler) else 'no'}",
            f"belief_arg={'yes' if belief is not None else 'no'}",
        ]
        if timer_elapsed is not None:
            ctx_parts.append(f"timer_elapsed={timer_elapsed:.2f}s")
        try:
            total_visits = sum(ch.N for ch in root.children)
            ctx_parts.append(f"completed_sims={total_visits}")
        except Exception:
            pass
        print(f"[WARN][is_mcts] context: " + ", ".join(ctx_parts), flush=True)
        try:
            priors_repr = priors.tolist() if hasattr(priors, 'tolist') else list(priors)
            print(f"[WARN][is_mcts] priors={priors_repr}", flush=True)
        except Exception as pri_exc:
            print(f"[WARN][is_mcts] failed to serialize priors: {pri_exc}", flush=True)
        print(f"[WARN][is_mcts] legal_keys={ak_order}", flush=True)
        try:
            child_stats = []
            for ch in root.children:
                child_stats.append({
                    'key': action_key(ch.action) if ch.action is not None else None,
                    'N': ch.N,
                    'Q': ch.Q,
                    'P': ch.P,
                    'W': ch.W,
                    'weight': ch.weight,
                })
            print(f"[WARN][is_mcts] root_children_stats={child_stats}", flush=True)
        except Exception as child_exc:
            print(f"[WARN][is_mcts] failed to serialize root children: {child_exc}", flush=True)
        try:
            from pprint import pformat
            state_str = pformat(root_env.game_state, width=120)
            print("[WARN][is_mcts] game_state snapshot:", flush=True)
            print(state_str, flush=True)
        except Exception as state_exc:
            print(f"[WARN][is_mcts] failed to format game_state: {state_exc}", flush=True)
        return _select_greedy_from_priors()
