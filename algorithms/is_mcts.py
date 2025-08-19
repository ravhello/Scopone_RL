import math
import random
from typing import List, Tuple

from environment import ScoponeEnvMA
import torch
from belief.belief import BeliefState


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
                belief: BeliefState = None,
                num_determinization: int = 1,
                root_temperature: float = 0.0,
                prior_smooth_eps: float = 0.0,
                robust_child: bool = True,
                root_dirichlet_alpha: float = 0.0,
                root_dirichlet_eps: float = 0.0,
                return_stats: bool = False):
    """
    IS-MCTS con determinizzazioni multiple e PUCT.
    - prior_smooth_eps: smoothing dei prior con (1-eps)*p + eps/|A|
    - root_temperature: softmax su N^(1/T) per la scelta finale (0 => argmax N)
    """
    root_env = env.clone()
    obs = root_env._get_observation(root_env.current_player)
    legals = root_env.get_valid_actions()
    if (torch.is_tensor(legals) and legals.size(0) == 0) or (not torch.is_tensor(legals) and not legals):
        raise ValueError("No legal actions for IS-MCTS")

    # Helpers per chiavi
    def action_key(vec):
        try:
            if torch.is_tensor(vec):
                nz = torch.nonzero(vec > 0, as_tuple=False).flatten()
                # map indices to Python only for dict keys (UI/tests path)
                return tuple([int(i) for i in nz.detach().tolist()])
            else:
                import numpy as _np
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

    # Prior iniziali (supporta tensori torch su CUDA)
    try:
        priors = policy_fn(obs, legals)
    except NameError:
        K = (legals.size(0) if torch.is_tensor(legals) else len(legals))
        priors = torch.ones(K, dtype=torch.float32, device=torch.device('cuda')) / max(1, K)
    if not torch.is_tensor(priors):
        priors = torch.as_tensor(priors, dtype=torch.float32, device=torch.device('cuda'))
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
    if torch.is_tensor(legals):
        K = legals.size(0)
        for i in range(K):
            a = legals[i]
            p = float(priors[i])
            ak = action_key(a)
            key = (pk_root, ak)
            if key in node_cache:
                child = node_cache[key]
                child.P = p
                if child not in root.children:
                    root.children.append(child)
            else:
                child = ISMCTSNode(parent=root, action=a)
                child.P = p
                root.children.append(child)
                node_cache[key] = child
    else:
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
            # Determinizzazione dal belief (se presente): assegna mani agli altri giocatori
            if belief is not None:
                try:
                    masks = belief.sample_determinization_gpu(1)[0]  # (3,40) bool CUDA
                    sim_env.apply_determinization_gpu(masks, belief.observer)
                except Exception:
                    pass
            # Selection (ensure chosen child action is legal under current determinization)
            node = root
            while True:
                if not node.children:
                    break
                # Get current legals and legal keys
                legals_cur = sim_env.get_valid_actions()
                if torch.is_tensor(legals_cur):
                    legal_keys = set(action_key(legals_cur[i]) for i in range(legals_cur.size(0)))
                else:
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
                if (torch.is_tensor(legals_s) and legals_s.size(0) > 0) or (not torch.is_tensor(legals_s) and legals_s):
                    try:
                        priors_s = policy_fn(obs_s, legals_s)
                    except NameError:
                        K = (legals_s.size(0) if torch.is_tensor(legals_s) else len(legals_s))
                        priors_s = torch.ones(K, dtype=torch.float32, device=torch.device('cuda')) / max(1, K)
                    if not torch.is_tensor(priors_s):
                        priors_s = torch.as_tensor(priors_s, dtype=torch.float32, device=torch.device('cuda'))
                    if prior_smooth_eps > 0 and priors_s.numel() > 1:
                        priors_s = (1.0 - prior_smooth_eps) * priors_s + prior_smooth_eps * (1.0 / priors_s.numel())
                    pk = public_key(sim_env)
                    if torch.is_tensor(legals_s):
                        for i in range(legals_s.size(0)):
                            a = legals_s[i]
                            p = float(priors_s[i])
                            ak = action_key(a)
                            key = (pk, ak)
                            if key in node_cache:
                                ch = node_cache[key]
                                ch.P = p
                                if ch not in node.children:
                                    node.children.append(ch)
                            else:
                                ch = ISMCTSNode(parent=node, action=a)
                                ch.P = p
                                node.children.append(ch)
                                node_cache[key] = ch
                    else:
                        for a, p in zip(legals_s, priors_s.tolist()):
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
                v_t = value_fn(obs_v)
                v = v_t
            # Backup
            while node is not None:
                node.N += 1
                node.W += float(v)
                node.Q = node.W / node.N
                node = node.parent

    # Scelta finale: robust child o soft a seconda di temperature (preferisci torch su CUDA)
    try:
        import torch
        device = torch.device('cuda')
        if root_temperature and root_temperature > 1e-6:
            visits_t = torch.tensor([ch.N for ch in root.children], dtype=torch.float32, device=device)
            logits = torch.pow(visits_t + 1e-9, 1.0 / float(root_temperature))
            probs_t = logits / torch.clamp_min(logits.sum(), 1e-9)
            idx_t = torch.multinomial(probs_t, num_samples=1).to(dtype=torch.long).squeeze(0)
            # Evita conversioni CPU: mantieni idx_t tensor e usa direttamente
            idx = int(idx_t)
            if return_stats:
                # Ritorna anche le probabilità come tensore su CUDA in modalità GPU-only
                return root.children[idx].action, probs_t
            return root.children[idx].action
        else:
            best = max(root.children, key=(lambda n: n.N) if robust_child else (lambda n: n.Q))
            if return_stats:
                visits_t = torch.tensor([ch.N for ch in root.children], dtype=torch.float32, device=device)
                probs_t = visits_t / torch.clamp_min(visits_t.sum(), 1e-9)
                return best.action, probs_t
            return best.action
    except Exception:
        # GPU-only mode: do not fall back to CPU/numpy; re-raise for visibility
        raise


