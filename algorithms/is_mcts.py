import math
import random
from typing import List, Tuple

from environment import ScoponeEnvMA
from utils.fallback import notify_fallback
import os
import torch
from utils.device import get_compute_device
# BeliefState legacy rimosso: MCTS usa SOLO belief_sampler neurale; nessun fallback a filtri particellari


class ISMCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action  # azione che ha portato a questo nodo (80-dim np.array)
        self.children: List[ISMCTSNode] = []
        self.N = 0  # visite
        self.W = 0.0  # somma valori
        self.Q = 0.0  # valore medio
        self.P = 0.0  # prior (dalla policy)

    def ucb_score(self, c_puct: float = 1.0) -> float:
        if self.parent is None:
            return float('-inf')
        prior = self.P
        total_N = max(1, self.parent.N)
        return self.Q + c_puct * prior * math.sqrt(total_N) / (1 + self.N)


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
                belief_sampler=None):
    """
    IS-MCTS con determinizzazioni multiple e PUCT.
    - prior_smooth_eps: smoothing dei prior con (1-eps)*p + eps/|A|
    - root_temperature: softmax su N^(1/T) per la scelta finale (0 => argmax N)
    """
    # Validate input hyperparameters
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

    root_env = env.clone()
    obs = root_env._get_observation(root_env.current_player)
    legals = root_env.get_valid_actions()
    # Support both Tensor (A,80) and list outputs without ambiguous truthiness
    is_empty = (hasattr(legals, 'numel') and legals.numel() == 0) or (hasattr(legals, '__len__') and len(legals) == 0)
    if is_empty:
        raise ValueError("No legal actions for IS-MCTS")

    # Helpers per chiavi
    def action_key(vec):
        import torch as _torch
        import numpy as _np
        if _torch.is_tensor(vec):
            nz = _torch.nonzero(vec > 0, as_tuple=False).flatten().tolist()
            return tuple(int(i) for i in nz)
        else:
            return tuple(_np.flatnonzero(vec).tolist())
    def public_key(state_env: ScoponeEnvMA):
        cur = state_env.current_player
        table_ids = tuple(sorted(state_env.game_state.get('table', [])))
        return (cur, table_ids)

    # Ordine chiavi azioni alla radice (per aggregazione stabile delle visite)
    ak_order = []
    ak_order = [action_key(a) for a in legals]

    # Prior iniziali (supporta tensori torch su CUDA)
    try:
        priors = policy_fn(obs, legals)
    except Exception as e:
        raise RuntimeError("IS-MCTS: policy_fn failed to produce priors") from e
    import numpy as _np
    if isinstance(priors, _np.ndarray):
        priors_len = len(priors)
        if priors_len != len(legals):
            raise RuntimeError(f"IS-MCTS: priors length {priors_len} != num legals {len(legals)}")
        if not _np.isfinite(priors).all():
            raise RuntimeError("IS-MCTS: priors contain non-finite values")
        if prior_smooth_eps > 0 and priors_len > 1:
            priors = (1 - prior_smooth_eps) * priors + prior_smooth_eps * (1.0 / priors_len)
        if root_dirichlet_eps > 0 and priors_len > 1 and root_dirichlet_alpha > 0:
            noise = _np.random.dirichlet([root_dirichlet_alpha] * priors_len)
            priors = (1 - root_dirichlet_eps) * priors + root_dirichlet_eps * noise
    else:
        # Assume torch.Tensor path
        import os as _os
        if not torch.is_tensor(priors):
            from utils.device import get_compute_device as _get_compute_device
            pri_dev = _get_compute_device()
            priors = torch.as_tensor(priors, dtype=torch.float32, device=pri_dev)
        device = priors.device
        priors_len = int(priors.numel())
        if priors_len != len(legals):
            raise RuntimeError(f"IS-MCTS: priors length {priors_len} != num legals {len(legals)}")
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
    node_cache = {}
    pk_root = public_key(root_env)
    prior_list = priors.tolist() if hasattr(priors, 'tolist') else list(priors)
    for a, p in zip(legals, prior_list):
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

    for _ in range(num_simulations):
        # Esegui una o più determinizzazioni
        for _det in range(max(1, num_determinization)):
            sim_env = root_env.clone()
            # Determinizzazione: SOLO tramite belief_sampler neurale se fornito
            if callable(belief_sampler):
                det = belief_sampler(sim_env)
                if isinstance(det, dict):
                    for pid, ids in det.items():
                        sim_env.game_state['hands'][pid] = list(ids)
                    sim_env._rebuild_id_caches()
            # Selection (ensure chosen child action is legal under current determinization)
            node = root
            while True:
                if not node.children:
                    break
                # Get current legals and legal keys
                legals_cur = sim_env.get_valid_actions()
                legal_keys = set(action_key(a) for a in legals_cur)
                # Filter children to only those legal in this determinization
                legal_children = [ch for ch in node.children if action_key(ch.action) in legal_keys or ch.action is None]
                if not legal_children:
                    # Stop selection and go to expansion with current legals
                    break
                node = max(legal_children, key=lambda n: n.ucb_score(c_puct))
                if node.action is not None:
                    _, _, done, _ = sim_env.step(node.action)
                    if done:
                        break

            # Expansion
            if not sim_env.done:
                obs_s = sim_env._get_observation(sim_env.current_player)
                legals_s = sim_env.get_valid_actions()
                has_legals_s = (hasattr(legals_s, 'numel') and legals_s.numel() > 0) or (hasattr(legals_s, '__len__') and len(legals_s) > 0)
                if has_legals_s:
                    try:
                        priors_s = policy_fn(obs_s, legals_s)
                    except Exception as e:
                        raise RuntimeError("IS-MCTS: policy_fn failed during expansion priors") from e
                    if isinstance(priors_s, _np.ndarray):
                        if prior_smooth_eps > 0 and len(priors_s) > 1:
                            priors_s = (1 - prior_smooth_eps) * priors_s + prior_smooth_eps * (1.0 / len(priors_s))
                        # Progressive widening: seleziona top-K in base alle visite del nodo
                        visits_here = max(1, node.N)
                        k_allow = min(len(legals_s), int(3.0 * (visits_here ** 0.5)) + 1)
                        top_idx = _np.argsort(-priors_s)[:k_allow]
                        legals_sel = [legals_s[i] for i in top_idx]
                        priors_sel = priors_s[top_idx]
                    else:
                        if not torch.is_tensor(priors_s):
                            from utils.device import get_compute_device as _get_compute_device
                            pri_dev2 = _get_compute_device()
                            priors_s = torch.as_tensor(priors_s, dtype=torch.float32, device=pri_dev2)
                        if prior_smooth_eps > 0 and priors_s.numel() > 1:
                            priors_s = (1.0 - prior_smooth_eps) * priors_s + prior_smooth_eps * (1.0 / priors_s.numel())
                        visits_here = max(1, node.N)
                        k_allow = min(int(priors_s.numel()), int(3.0 * (visits_here ** 0.5)) + 1)
                        top_idx = torch.argsort(priors_s, descending=True)[:k_allow]
                        legals_sel = [legals_s[int(i)] for i in top_idx.tolist()]
                        priors_sel = priors_s[top_idx].detach().cpu().numpy()
                    # Public key arricchita
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
            # Evaluation
            if sim_env.done:
                v = 0.0
            else:
                obs_v = sim_env._get_observation(sim_env.current_player)
                # Pass anche l'env corrente al value_fn per consentire seat/belief
                v = float(value_fn(obs_v, sim_env))
            # Backup
            while node is not None:
                node.N += 1
                node.W += v
                node.Q = node.W / node.N
                node = node.parent

    # Scelta finale: robust child o soft a seconda di temperature (preferisci torch su CUDA)
    device = get_compute_device()
    if root_temperature and root_temperature > 1e-6:
        visits_t = torch.tensor([ch.N for ch in root.children], dtype=torch.float32, device=device)
        logits = torch.pow(visits_t + 1e-9, 1.0 / float(root_temperature))
        probs_t = logits / torch.clamp_min(logits.sum(), 1e-9)
        # sanitize
        probs_t = probs_t.nan_to_num(0.0)
        probs_t = torch.clamp(probs_t, min=0.0)
        s = probs_t.sum()
        if not torch.isfinite(s) or s <= 0:
            raise RuntimeError("IS-MCTS: invalid root selection probabilities (NaN/Inf or zero-sum)")
        else:
            probs_t = probs_t / s
        idx = int(torch.multinomial(probs_t, num_samples=1).item())
        if return_stats:
            # Aggrega probabilità per ciascuna azione legale alla radice
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