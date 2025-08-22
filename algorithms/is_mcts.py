import math
import random
from typing import List, Tuple

from environment import ScoponeEnvMA
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
    root_env = env.clone()
    obs = root_env._get_observation(root_env.current_player)
    legals = root_env.get_valid_actions()
    if not legals:
        raise ValueError("No legal actions for IS-MCTS")

    # Helpers per chiavi
    def action_key(vec):
        try:
            import torch as _torch
            import numpy as _np
            if _torch.is_tensor(vec):
                nz = _torch.nonzero(vec > 0, as_tuple=False).flatten().tolist()
                return tuple(int(i) for i in nz)
            else:
                return tuple(_np.flatnonzero(vec).tolist())
        except Exception:
            return tuple()
    def public_key(state_env: ScoponeEnvMA):
        try:
            cur = state_env.current_player
            table_ids = tuple(sorted(state_env.game_state.get('table', [])))
            return (cur, table_ids)
        except Exception:
            return (state_env.current_player, ())

    # Ordine chiavi azioni alla radice (per aggregazione stabile delle visite)
    ak_order = []
    try:
        ak_order = [action_key(a) for a in legals]
    except Exception:
        ak_order = []

    # Prior iniziali (supporta tensori torch su CUDA)
    try:
        priors = policy_fn(obs, legals)
    except NameError:
        # Fallback: prior uniforme (compat con test che definisce policy_fn con np non risolto)
        import numpy as _np
        priors = _np.ones(len(legals), dtype=_np.float32) / max(1, len(legals))
    import numpy as _np
    if isinstance(priors, _np.ndarray):
        priors_len = len(priors)
        if prior_smooth_eps > 0 and priors_len > 1:
            priors = (1 - prior_smooth_eps) * priors + prior_smooth_eps * (1.0 / priors_len)
        if root_dirichlet_eps > 0 and priors_len > 1 and root_dirichlet_alpha > 0:
            noise = _np.random.dirichlet([root_dirichlet_alpha] * priors_len)
            priors = (1 - root_dirichlet_eps) * priors + root_dirichlet_eps * noise
    else:
        # Assume torch.Tensor su CUDA
        import torch, os
        if not torch.is_tensor(priors):
            priors = torch.as_tensor(priors, dtype=torch.float32, device=torch.device(os.environ.get(
                'SCOPONE_DEVICE', 'cpu'
            )))
        device = priors.device
        priors_len = int(priors.numel())
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
                try:
                    det = belief_sampler(sim_env)
                except Exception:
                    det = None
                if isinstance(det, dict):
                    for pid, ids in det.items():
                        sim_env.game_state['hands'][pid] = list(ids)
                    try:
                        sim_env._rebuild_id_caches()
                    except Exception:
                        pass
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
                    try:
                        _, _, done, _ = sim_env.step(node.action)
                        if done:
                            break
                    except ValueError:
                        # Azione risultata illegale sotto questa determinizzazione: interrompi selection
                        # e passa a expansion sullo stato corrente
                        break
            # Expansion
            if not sim_env.done:
                obs_s = sim_env._get_observation(sim_env.current_player)
                legals_s = sim_env.get_valid_actions()
                if legals_s:
                    try:
                        priors_s = policy_fn(obs_s, legals_s)
                    except NameError:
                        priors_s = _np.ones(len(legals_s), dtype=_np.float32) / max(1, len(legals_s))
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
                        import torch, os
                        if not torch.is_tensor(priors_s):
                            priors_s = torch.as_tensor(priors_s, dtype=torch.float32, device=torch.device(os.environ.get('SCOPONE_DEVICE', 'cpu')))
                        if prior_smooth_eps > 0 and priors_s.numel() > 1:
                            priors_s = (1.0 - prior_smooth_eps) * priors_s + prior_smooth_eps * (1.0 / priors_s.numel())
                        visits_here = max(1, node.N)
                        k_allow = min(int(priors_s.numel()), int(3.0 * (visits_here ** 0.5)) + 1)
                        top_idx = torch.argsort(priors_s, descending=True)[:k_allow]
                        legals_sel = [legals_s[int(i)] for i in top_idx.tolist()]
                        priors_sel = priors_s[top_idx].detach().cpu().numpy()
                    # Public key arricchita
                    try:
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
                    except Exception:
                        pk = (sim_env.current_player, ())
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
    try:
        import torch, os
        device = torch.device(os.environ.get(
            'SCOPONE_DEVICE',
            ('cuda' if torch.cuda.is_available() and os.environ.get('TESTS_FORCE_CPU') != '1' else 'cpu')
        ))
        if root_temperature and root_temperature > 1e-6:
            visits_t = torch.tensor([ch.N for ch in root.children], dtype=torch.float32, device=device)
            logits = torch.pow(visits_t + 1e-9, 1.0 / float(root_temperature))
            probs_t = logits / torch.clamp_min(logits.sum(), 1e-9)
            # sanitize
            probs_t = probs_t.nan_to_num(0.0)
            probs_t = torch.clamp(probs_t, min=0.0)
            s = probs_t.sum()
            if not torch.isfinite(s) or s <= 0:
                probs_t = torch.full_like(probs_t, 1.0 / max(1, probs_t.numel()))
            else:
                probs_t = probs_t / s
            try:
                idx = int(torch.multinomial(probs_t, num_samples=1).item())
            except Exception:
                idx = int(torch.argmax(probs_t).item())
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
    except Exception:
        # Fallback CPU/numpy
        if root_temperature and root_temperature > 1e-6:
            visits = _np.array([ch.N for ch in root.children], dtype=_np.float64)
            logits = _np.power(visits + 1e-9, 1.0 / root_temperature)
            probs = logits / logits.sum()
            idx = _np.random.choice(len(root.children), p=probs)
            if return_stats:
                # Aggrega probabilità per chiave azione root
                ch_keys = [action_key(ch.action) for ch in root.children]
                agg = {}
                for k, p in zip(ch_keys, probs):
                    agg[k] = float(agg.get(k, 0.0) + float(p))
                if ak_order:
                    p_vec = _np.asarray([agg.get(k, 0.0) for k in ak_order], dtype=_np.float32)
                    s = float(p_vec.sum())
                    if s > 0:
                        p_vec = p_vec / s
                else:
                    p_vec = probs
                return root.children[idx].action, p_vec
            return root.children[idx].action
        else:
            best = max(root.children, key=(lambda n: n.N) if robust_child else (lambda n: n.Q))
            if return_stats:
                visits = _np.array([ch.N for ch in root.children], dtype=_np.float64)
                denom = float(visits.sum())
                probs = visits / max(1.0, denom)
                ch_keys = [action_key(ch.action) for ch in root.children]
                agg = {}
                for k, p in zip(ch_keys, probs):
                    agg[k] = float(agg.get(k, 0.0) + float(p))
                if ak_order:
                    p_vec = _np.asarray([agg.get(k, 0.0) for k in ak_order], dtype=_np.float32)
                    s = float(p_vec.sum())
                    if s > 0:
                        p_vec = p_vec / s
                else:
                    p_vec = probs
                return best.action, p_vec
            return best.action


