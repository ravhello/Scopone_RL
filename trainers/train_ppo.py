import torch
from tqdm import tqdm
from typing import Dict, List, Callable, Optional, Tuple
import os
import time
import sys
import multiprocessing as mp

# Ensure project root is on sys.path when running as script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
 
from selfplay.league import League
from models.action_conditioned import ActionConditionedActor
from utils.seed import set_global_seeds
from evaluation.eval import evaluate_pair_actors

import torch.optim as optim

_DEVICE_STR = os.environ.get(
    'SCOPONE_DEVICE',
    ('cuda' if torch.cuda.is_available() and os.environ.get('TESTS_FORCE_CPU') != '1' else 'cpu')
)
device = torch.device(_DEVICE_STR)
# Global perf flags
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def _seat_vec_for(cp: int) -> torch.Tensor:
    v = torch.zeros(6, dtype=torch.float32)
    v[cp] = 1.0
    v[4] = 1.0 if cp in [0, 2] else 0.0
    v[5] = 1.0 if cp in [1, 3] else 0.0
    return v


def _env_worker(worker_id: int,
                cfg: Dict,
                request_q: mp.Queue,
                action_q: mp.Queue,
                episode_q: mp.Queue):
    env = ScoponeEnvMA(rules=cfg.get('rules', {'shape_scopa': False}),
                       use_compact_obs=cfg.get('use_compact_obs', True),
                       k_history=int(cfg.get('k_history', 39)))
    episodes_per_env = int(cfg.get('episodes_per_env', 1))
    send_legals = bool(cfg.get('send_legals', True))
    use_mcts = bool(cfg.get('use_mcts', False))
    train_both_teams = bool(cfg.get('train_both_teams', False))
    main_seats = cfg.get('main_seats', None)
    # MCTS config
    mcts_sims = int(cfg.get('mcts_sims', 128))
    mcts_dets = int(cfg.get('mcts_dets', 4))
    mcts_c_puct = float(cfg.get('mcts_c_puct', 1.0))
    mcts_root_temp = float(cfg.get('mcts_root_temp', 0.0))
    mcts_prior_smooth_eps = float(cfg.get('mcts_prior_smooth_eps', 0.0))
    mcts_dirichlet_alpha = float(cfg.get('mcts_dirichlet_alpha', 0.25))
    mcts_dirichlet_eps = float(cfg.get('mcts_dirichlet_eps', 0.25))
    mcts_progress_start = float(cfg.get('mcts_progress_start', 0.25))
    mcts_progress_full = float(cfg.get('mcts_progress_full', 0.75))
    mcts_min_sims = int(cfg.get('mcts_min_sims', 0))

    for ep in range(episodes_per_env):
        env.reset()
        obs_list, next_obs_list = [], []
        act_list = []
        rew_list, done_list = [], []
        seat_team_list = []
        belief_sum_list = []
        legals_list, legals_offset, legals_count = [], [], []
        chosen_index_list = []
        mcts_policy_list = []
        mcts_weight_list = []
        others_hands_list = []

        done = False
        info = {}
        while not done:
            obs = env._get_observation(env.current_player)
            legal = env.get_valid_actions()
            if not legal:
                break
            cp = env.current_player
            seat_vec = _seat_vec_for(cp)
            is_main = True if train_both_teams else ((main_seats is None and cp in [0, 2]) or (main_seats is not None and cp in main_seats))
            if is_main and use_mcts and len(legal) > 0:
                # Build helper RPCs to master for batched scoring
                if send_legals:
                    leg_serial = [ (x.tolist() if torch.is_tensor(x) else list(x)) for x in legal ]
                else:
                    leg_serial = []
                def policy_fn_mcts(_obs, _legals):
                    request_q.put({
                        'type': 'score_policy',
                        'wid': worker_id,
                        'obs': (_obs.tolist() if torch.is_tensor(_obs) else list(_obs)),
                        'legals': [ (y.tolist() if torch.is_tensor(y) else list(y)) for y in _legals ],
                        'seat': seat_vec.tolist(),
                    })
                    resp = action_q.get()
                    pri = resp.get('priors', None)
                    import numpy as _np
                    return _np.asarray(pri, dtype=_np.float32) if pri is not None else _np.ones((len(_legals),), dtype=_np.float32) / max(1, len(_legals))
                def value_fn_mcts(_obs, _env):
                    s_vec = _seat_vec_for(_env.current_player)
                    request_q.put({
                        'type': 'score_value',
                        'wid': worker_id,
                        'obs': (_obs.tolist() if torch.is_tensor(_obs) else list(_obs)),
                        'seat': s_vec.tolist(),
                    })
                    resp = action_q.get()
                    return float(resp.get('value', 0.0))
                def belief_sampler_neural(_env):
                    try:
                        o_cur = _env._get_observation(_env.current_player)
                        s_vec = _seat_vec_for(_env.current_player)
                        request_q.put({
                            'type': 'score_belief',
                            'wid': worker_id,
                            'obs': (o_cur.tolist() if torch.is_tensor(o_cur) else list(o_cur)),
                            'seat': s_vec.tolist(),
                        })
                        resp = action_q.get()
                        probs_flat = resp.get('belief_probs', None)
                        if probs_flat is None:
                            return None
                        import numpy as _np
                        probs = _np.asarray(probs_flat, dtype=_np.float32).reshape(3, 40)
                        # visible mask from obs on worker side
                        if torch.is_tensor(o_cur):
                            o_t = o_cur.detach().to('cpu', dtype=torch.float32).unsqueeze(0)
                        else:
                            o_t = torch.as_tensor(o_cur, dtype=torch.float32).unsqueeze(0)
                        hand_table = o_t[:, :83]
                        hand_mask = hand_table[:, :40] > 0.5
                        table_mask = hand_table[:, 43:83] > 0.5
                        captured = o_t[:, 83:165]
                        cap0_mask = captured[:, :40] > 0.5
                        cap1_mask = captured[:, 40:80] > 0.5
                        vis = (hand_mask | table_mask | cap0_mask | cap1_mask).squeeze(0).numpy().astype(bool)
                        unknown_ids = [cid for cid in range(40) if not vis[cid]]
                        others = [(_env.current_player + 1) % 4, (_env.current_player + 2) % 4, (_env.current_player + 3) % 4]
                        counts = {pid: len(_env.game_state['hands'][pid]) for pid in others}
                        caps = [int(counts.get(pid, 0)) for pid in others]
                        n = len(unknown_ids)
                        if sum(caps) != n:
                            caps[2] = max(0, n - caps[0] - caps[1])
                            if sum(caps) != n:
                                base = n // 3
                                rem = n - 3 * base
                                caps = [base, base, base]
                                for i in range(rem):
                                    caps[i] += 1
                        import numpy as _np
                        noise_scale = float(os.environ.get('DET_NOISE', '0.0'))
                        costs = []
                        for cid in unknown_ids:
                            pc = probs[:, cid]
                            ps = pc / max(1e-12, pc.sum())
                            c = [-_np.log(max(1e-12, ps[i])) for i in range(3)]
                            if noise_scale > 0:
                                u = _np.random.uniform(1e-9, 1.0-1e-9, size=3)
                                g = -_np.log(-_np.log(u)) * noise_scale
                                c = [c[i] + float(g[i]) for i in range(3)]
                            costs.append(c)
                        INF = 1e12
                        cap0, cap1, cap2 = caps
                        dp = [[[INF]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                        bk = [[[-1]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                        dp[0][0][0] = 0.0
                        for t in range(n):
                            c0, c1, c2 = costs[t]
                            for a in range(0, min(t, cap0)+1):
                                for b in range(0, min(t-a, cap1)+1):
                                    cur = dp[t][a][b]
                                    if cur >= INF:
                                        continue
                                    if a+1 <= cap0 and dp[t+1][a+1][b] > cur + c0:
                                        dp[t+1][a+1][b] = cur + c0
                                        bk[t+1][a+1][b] = 0
                                    if b+1 <= cap1 and dp[t+1][a][b+1] > cur + c1:
                                        dp[t+1][a][b+1] = cur + c1
                                        bk[t+1][a][b+1] = 1
                                    assigned2 = t - a - b
                                    if assigned2 + 1 <= cap2 and dp[t+1][a][b] > cur + c2:
                                        dp[t+1][a][b] = cur + c2
                                        bk[t+1][a][b] = 2
                        if dp[n][cap0][cap1] >= INF:
                            return None
                        det = {pid: [] for pid in others}
                        a, b = cap0, cap1
                        for t in range(n, 0, -1):
                            choice = bk[t][a][b]
                            cid = unknown_ids[t-1]
                            if choice == 0:
                                det[others[0]].append(cid)
                                a -= 1
                            elif choice == 1:
                                det[others[1]].append(cid)
                                b -= 1
                            else:
                                det[others[2]].append(cid)
                        return det
                    except Exception:
                        return None
                # Progress-based scaling (uniform with single-env)
                try:
                    progress = float(min(1.0, max(0.0, len(env.game_state.get('history', [])) / 40.0)))
                except Exception:
                    progress = 0.0
                denom = max(1e-6, (mcts_progress_full - mcts_progress_start))
                alpha = min(1.0, max(0.0, (progress - mcts_progress_start) / denom))
                # Permetti 0 simulazioni se mcts_min_sims==0
                base_min = int(mcts_min_sims) if (mcts_min_sims is not None and int(mcts_min_sims) >= 0) else 0
                sims_scaled = int(max(base_min, round(mcts_sims * (0.25 + 0.75 * alpha))))
                root_temp_dyn = float(mcts_root_temp) if float(mcts_root_temp) > 0 else float(max(0.0, 1.0 - alpha))

                if sims_scaled <= 0:
                    # Fallback: usa selezione dal master (GPU) come nel ramo non-MCTS
                    if send_legals:
                        leg_serial = [ (x.tolist() if torch.is_tensor(x) else list(x)) for x in legal ]
                    else:
                        leg_serial = []
                    request_q.put({
                        'type': 'step',
                        'wid': worker_id,
                        'obs': (obs.tolist() if torch.is_tensor(obs) else list(obs)),
                        'legals': leg_serial,
                        'seat': seat_vec.tolist(),
                    })
                    resp = action_q.get()
                    idx = int(resp.get('idx', 0))
                    idx = max(0, min(idx, len(legal) - 1))
                    act_t = legal[idx] if torch.is_tensor(legal[idx]) else torch.as_tensor(legal[idx], dtype=torch.float32)
                    next_obs, rew, done, info = env.step(act_t)
                    # No distillation target
                    mcts_policy_list.extend([0.0] * len(legal))
                    mcts_weight_list.append(0.0)
                else:
                    from algorithms.is_mcts import run_is_mcts
                    mcts_action, mcts_visits = run_is_mcts(env,
                        policy_fn=policy_fn_mcts,
                        value_fn=value_fn_mcts,
                        num_simulations=int(sims_scaled),
                        c_puct=float(mcts_c_puct),
                        belief=None,
                        num_determinization=int(mcts_dets),
                        root_temperature=root_temp_dyn,
                        prior_smooth_eps=float(mcts_prior_smooth_eps if mcts_prior_smooth_eps > 0 else 0.1),
                        robust_child=True,
                        root_dirichlet_alpha=float(mcts_dirichlet_alpha),
                        root_dirichlet_eps=float(mcts_dirichlet_eps if mcts_dirichlet_eps > 0 else 0.25),
                        return_stats=True,
                        belief_sampler=belief_sampler_neural)
                    chosen_act = mcts_action if torch.is_tensor(mcts_action) else torch.as_tensor(mcts_action, dtype=torch.float32)
                    def _act_key(x_t: torch.Tensor):
                        xt = x_t if torch.is_tensor(x_t) else torch.as_tensor(x_t, dtype=torch.float32)
                        return tuple(torch.nonzero(xt > 0.5, as_tuple=False).flatten().tolist())
                    key_target = _act_key(chosen_act)
                    idx = 0
                    for i_a, a in enumerate(legal):
                        try:
                            if _act_key(a) == key_target:
                                idx = int(i_a)
                                break
                        except Exception:
                            continue
                    act_t = legal[idx] if torch.is_tensor(legal[idx]) else torch.as_tensor(legal[idx], dtype=torch.float32)
                    next_obs, rew, done, info = env.step(act_t)
                    # Distillation targets
                    try:
                        mcts_probs = torch.as_tensor(mcts_visits, dtype=torch.float32)
                        ssum = float(mcts_probs.sum().item())
                        if ssum > 0:
                            mcts_probs = mcts_probs / ssum
                    except Exception:
                        mcts_probs = torch.full((len(legal),), 0.0, dtype=torch.float32)
                    mcts_policy_list.extend((mcts_probs.tolist() if hasattr(mcts_probs, 'tolist') else list(mcts_probs)))
                    mcts_weight_list.append(1.0)
            else:
                # Request action selection from master (GPU)
                if send_legals:
                    leg_serial = [ (x.tolist() if torch.is_tensor(x) else list(x)) for x in legal ]
                else:
                    leg_serial = []
                request_q.put({
                    'type': 'step',
                    'wid': worker_id,
                    'obs': (obs.tolist() if torch.is_tensor(obs) else list(obs)),
                    'legals': leg_serial,
                    'seat': seat_vec.tolist(),
                })
                resp = action_q.get()
                idx = int(resp.get('idx', 0))
                idx = max(0, min(idx, len(legal) - 1))
                act_t = legal[idx] if torch.is_tensor(legal[idx]) else torch.as_tensor(legal[idx], dtype=torch.float32)
                next_obs, rew, done, info = env.step(act_t)
                # No distillation target
                mcts_policy_list.extend([0.0] * len(legal))
                mcts_weight_list.append(0.0)

            obs_list.append(torch.as_tensor(obs, dtype=torch.float32) if not torch.is_tensor(obs) else obs.clone().detach().to('cpu', dtype=torch.float32))
            next_obs_list.append(torch.as_tensor(next_obs, dtype=torch.float32) if not torch.is_tensor(next_obs) else next_obs.clone().detach().to('cpu', dtype=torch.float32))
            act_list.append(act_t.clone().detach().to('cpu', dtype=torch.float32))
            rew_list.append(float(rew))
            done_list.append(bool(done))
            seat_team_list.append(seat_vec.clone().detach().to('cpu'))
            # belief summary disabled in workers by default (zeros placeholder)
            belief_sum_list.append(torch.zeros(120, dtype=torch.float32))
            legals_offset.append(len(legals_list))
            legals_count.append(len(legal))
            if send_legals:
                legals_list.extend([torch.as_tensor(x, dtype=torch.float32) for x in legal])
            else:
                # Fallback: store only chosen action as the single legal to keep API compatible
                legals_list.append(act_t.clone().detach().to('cpu', dtype=torch.float32))
            chosen_index_list.append(int(idx))
            # Others' hands supervision target (3x40)
            try:
                hands = env.game_state.get('hands', None)
                if hands is not None:
                    others = [ (cp + 1) % 4, (cp + 2) % 4, (cp + 3) % 4 ]
                    target = torch.zeros((3,40), dtype=torch.float32)
                    for i,pid in enumerate(others):
                        for c in hands[pid]:
                            if isinstance(c, int):
                                cid = c
                            else:
                                r, s = c
                                suit_to_int = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
                                cid = int((r - 1) * 4 + suit_to_int[s])
                            target[i, int(cid)] = 1.0
                    others_hands_list.append(target)
                else:
                    others_hands_list.append(torch.zeros((3,40), dtype=torch.float32))
            except Exception:
                others_hands_list.append(torch.zeros((3,40), dtype=torch.float32))

        # Episode payload back to master
        episode_q.put({
            'wid': worker_id,
            'obs': [x.tolist() for x in obs_list],
            'next_obs': [x.tolist() for x in next_obs_list],
            'act': [x.tolist() for x in act_list],
            'rew': rew_list,
            'done': done_list,
            'seat': [x.tolist() for x in seat_team_list],
            'belief_summary': [x.tolist() for x in belief_sum_list],
            'legals': [x.tolist() for x in legals_list],
            'leg_off': legals_offset,
            'leg_cnt': legals_count,
            'chosen_idx': chosen_index_list,
            'team_rewards': (info.get('team_rewards', [0.0, 0.0]) if isinstance(info, dict) else [0.0, 0.0]),
            'mcts_policy': mcts_policy_list,
            'mcts_weight': mcts_weight_list,
            'others_hands': [x.tolist() for x in others_hands_list],
        })


def _batched_select_indices(agent: ActionConditionedPPO,
                            reqs: List[Dict]) -> List[Tuple[int, int]]:
    # Returns list of (wid, idx) per req
    if len(reqs) == 0:
        return []
    # Stack CPU tensors then move once to CUDA
    obs_cpu = torch.stack([torch.as_tensor(r['obs'], dtype=torch.float32) for r in reqs], dim=0)
    seat_cpu = torch.stack([torch.as_tensor(r['seat'], dtype=torch.float32) for r in reqs], dim=0)
    # Flatten legals and build offsets
    flat_legals = []
    offs, cnts = [], []
    for r in reqs:
        offs.append(len(flat_legals))
        cnt = len(r['legals'])
        cnts.append(cnt)
        if cnt > 0:
            flat_legals.extend(r['legals'])
    if len(flat_legals) == 0:
        # No legals, return zeros
        return [(int(reqs[i]['wid']), 0) for i in range(len(reqs))]
    leg_cpu = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in flat_legals], dim=0)
    with torch.no_grad():
        o_t = obs_cpu.pin_memory().to(device=device, non_blocking=True)
        s_t = seat_cpu.pin_memory().to(device=device, non_blocking=True)
        leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
        state_proj = agent.actor.compute_state_proj(o_t, s_t)  # (B,64)
        a_emb_all = agent.actor.action_enc(leg_t)             # (M,64)
        B = o_t.size(0)
        cnts_t = torch.as_tensor(cnts, dtype=torch.long, device=device)
        max_cnt = int(cnts_t.max().item()) if B > 0 else 0
        pos = torch.arange(max_cnt, device=device, dtype=torch.long) if max_cnt > 0 else None
        out_idx = []
        if max_cnt > 0:
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts_t.unsqueeze(1)
            offs_t = torch.as_tensor(offs, dtype=torch.long, device=device)
            abs_idx = (offs_t.unsqueeze(1) + rel_pos_2d)[mask]
            sample_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            a_emb_mb = a_emb_all[abs_idx]
            legal_scores = (a_emb_mb * state_proj[sample_idx]).sum(dim=1)
            padded = torch.full((B, max_cnt), -1e9, device=device, dtype=legal_scores.dtype)
            padded[mask] = legal_scores
            logits = padded
            probs = torch.softmax(logits, dim=1)
            # Keep only legal positions; set others to zero
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            # Select only rows with at least one legal action for sampling
            valid_rows = (cnts_t > 0)
            sel = torch.zeros((B,), dtype=torch.long, device=device)
            if bool(valid_rows.any()):
                pv = probs[valid_rows]
                mv = mask[valid_rows]
                # Sanitize probabilities: remove NaN/Inf and negatives
                pv = pv.nan_to_num(0.0)
                pv = torch.clamp(pv, min=0.0)
                # Row-wise fix for zero-sum rows: fall back to uniform over allowed
                rs = pv.sum(dim=1, keepdim=True)
                bad_rows = (~torch.isfinite(rs)) | (rs <= 0)
                if bool(bad_rows.any()):
                    allowed = mv.to(pv.dtype)
                    counts = allowed.sum(dim=1, keepdim=True).clamp_min(1.0)
                    uniform = allowed / counts
                    pv = torch.where(bad_rows, uniform, pv)
                # Final guard: ensure strictly positive row sums
                rs = pv.sum(dim=1, keepdim=True)
                need_uniform = (rs <= 0)
                if bool(need_uniform.any()):
                    allowed = mv.to(pv.dtype)
                    counts = allowed.sum(dim=1, keepdim=True).clamp_min(1.0)
                    pv = torch.where(need_uniform, allowed / counts, pv)
                # Sample for exploration; fallback to argmax if still problematic
                try:
                    sub = torch.multinomial(pv, num_samples=1).squeeze(1)
                except Exception:
                    sub = torch.argmax(pv, dim=1)
                # Write back selections to full batch
                sel[valid_rows] = sub
            # rows with no legal actions keep sel=0
            sel_cpu = sel.detach().to('cpu')
            for i in range(B):
                out_idx.append((int(reqs[i]['wid']), int(sel_cpu[i].item())))
        else:
            out_idx = [(int(reqs[i]['wid']), 0) for i in range(B)]
    return out_idx

def _batched_service(agent: ActionConditionedPPO, reqs: List[Dict]) -> List[Dict]:
    """Service batched policy/value/belief scoring requests.
    Returns a list of dicts with results in the same order as input reqs.
    """
    if len(reqs) == 0:
        return []
    # Group by type
    obs_cpu = torch.stack([torch.as_tensor(r['obs'], dtype=torch.float32) for r in reqs], dim=0)
    seat_cpu = torch.stack([torch.as_tensor(r.get('seat', [0,0,0,0,0,0]), dtype=torch.float32) for r in reqs], dim=0)
    # Prepare CUDA once
    with torch.no_grad():
        o_t = obs_cpu.pin_memory().to(device=device, non_blocking=True)
        s_t = seat_cpu.pin_memory().to(device=device, non_blocking=True)
        state_proj = agent.actor.compute_state_proj(o_t, s_t)
    results: List[Dict] = []
    cursor = 0
    for r in reqs:
        rtype = r.get('type')
        if rtype == 'score_policy':
            legals = r.get('legals', [])
            if len(legals) == 0:
                results.append({'priors': []})
                cursor += 1
                continue
            leg_t = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in legals], dim=0).pin_memory().to(device=device, non_blocking=True)
            with torch.no_grad():
                scores = agent.actor.action_enc(leg_t) @ state_proj[cursor].unsqueeze(1)
                scores = scores.squeeze(1)
                priors = torch.softmax(scores, dim=0).detach().cpu().tolist()
            results.append({'priors': priors})
            cursor += 1
        elif rtype == 'score_value':
            with torch.no_grad():
                val = agent.critic(o_t[cursor].unsqueeze(0), s_t[cursor].unsqueeze(0)).squeeze(0)
            results.append({'value': float(val.detach().cpu().item())})
            cursor += 1
        elif rtype == 'score_belief':
            with torch.no_grad():
                # Compute belief probs (120) with masking inside model
                logits = agent.actor.belief_net(agent.actor.state_enc(o_t[cursor].unsqueeze(0), s_t[cursor].unsqueeze(0)))
                hand_table = o_t[cursor:cursor+1, :83]
                hand_mask = hand_table[:, :40] > 0.5
                table_mask = hand_table[:, 43:83] > 0.5
                captured = o_t[cursor:cursor+1, 83:165]
                cap0_mask = captured[:, :40] > 0.5
                cap1_mask = captured[:, 40:80] > 0.5
                visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)
                probs_flat = agent.actor.belief_net.probs(logits, visible_mask).squeeze(0).detach().cpu().tolist()
            results.append({'belief_probs': probs_flat})
            cursor += 1
        else:
            results.append({})
            cursor += 1
    return results
def _compute_per_seat_diagnostics(agent: ActionConditionedPPO, batch: Dict) -> Dict[str, torch.Tensor]:
    """Calcola approx_kl, entropia e clip_frac per gruppi seat 0/2 vs 1/3.

    Tutte le operazioni restano su GPU; ritorna tensori 0-D su device.
    """
    obs = batch['obs'] if torch.is_tensor(batch['obs']) else torch.as_tensor(batch['obs'], dtype=torch.float32)
    obs = obs.to(device=device)
    seat = batch.get('seat_team', None)
    if seat is None:
        seat = torch.zeros((obs.size(0), 6), dtype=torch.float32)
    elif not torch.is_tensor(seat):
        seat = torch.as_tensor(seat, dtype=torch.float32)
    seat = seat.to(device=device)
    legals = batch['legals'] if torch.is_tensor(batch['legals']) else torch.as_tensor(batch['legals'], dtype=torch.float32)
    offs = batch['legals_offset'] if torch.is_tensor(batch['legals_offset']) else torch.as_tensor(batch['legals_offset'], dtype=torch.long)
    cnts = batch['legals_count'] if torch.is_tensor(batch['legals_count']) else torch.as_tensor(batch['legals_count'], dtype=torch.long)
    chosen_idx = batch['chosen_index'] if torch.is_tensor(batch['chosen_index']) else torch.as_tensor(batch['chosen_index'], dtype=torch.long)
    old_logp = torch.as_tensor(batch['old_logp'], dtype=torch.float32)
    legals = legals.to(device=device)
    offs = offs.to(device=device)
    cnts = cnts.to(device=device)
    chosen_idx = chosen_idx.to(device=device)
    old_logp = old_logp.to(device=device)

    approx_kl = torch.zeros(obs.size(0), dtype=torch.float32, device=device)
    entropy = torch.zeros(obs.size(0), dtype=torch.float32, device=device)
    with torch.no_grad():
        B = obs.size(0)
        # Compute state projection (B,64) con belief neurale interno
        state_proj = agent.actor.compute_state_proj(obs, seat)
        # Action embeddings for all legals in the batch
        a_emb_global = agent.actor.action_enc(legals)  # (M_all,64)
        max_cnt = int(cnts.max().item()) if B > 0 else 0
        if max_cnt > 0:
            pos = torch.arange(max_cnt, device=device, dtype=torch.long)
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts.unsqueeze(1)
            sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            abs_idx_2d = offs.unsqueeze(1) + rel_pos_2d
            abs_idx = abs_idx_2d[mask]
            a_emb_mb = a_emb_global[abs_idx]
            legal_scores = (a_emb_mb * state_proj[sample_idx_per_legal]).sum(dim=1)
            padded = torch.full((B, max_cnt), -float('inf'), device=device, dtype=legal_scores.dtype)
            padded[mask] = legal_scores
        else:
            padded = torch.full((B, 0), -float('inf'), device=device, dtype=state_proj.dtype)
        logp_group = torch.log_softmax(padded, dim=1)
        probs_group = torch.softmax(padded, dim=1)
        entropy[:] = (-(probs_group * logp_group).sum(dim=1))
        chosen_clamped = torch.minimum(chosen_idx, (cnts - 1).clamp_min(0)) if max_cnt > 0 else chosen_idx
        new_logp_chosen = logp_group[torch.arange(B, device=device), chosen_clamped]
        approx_kl[:] = (old_logp - new_logp_chosen).abs()
    # clip fraction per-sample sul chosen
    # ricalcolo ratio corretto: new_logp_chosen - old_logp
    ratio = torch.exp(new_logp_chosen - old_logp)
    clip_low = 1.0 - agent.clip_ratio
    clip_high = 1.0 + agent.clip_ratio
    clipped_mask = (ratio < clip_low) | (ratio > clip_high)

    seats = torch.argmax(seat[:, :4], dim=1)
    mask_02 = (seats == 0) | (seats == 2)
    mask_13 = (seats == 1) | (seats == 3)
    out: Dict[str, torch.Tensor] = {}
    # Evita branching su CPU: usa conteggi e where per gestire gruppi vuoti
    one = torch.tensor(1.0, device=device, dtype=torch.float32)
    count_02 = mask_02.float().sum()
    count_13 = mask_13.float().sum()
    sum_kl_02 = (approx_kl * mask_02.float()).sum()
    sum_kl_13 = (approx_kl * mask_13.float()).sum()
    sum_en_02 = (entropy * mask_02.float()).sum()
    sum_en_13 = (entropy * mask_13.float()).sum()
    sum_cf_02 = (clipped_mask.float() * mask_02.float()).sum()
    sum_cf_13 = (clipped_mask.float() * mask_13.float()).sum()
    mean_kl_02 = torch.where(count_02 > 0, sum_kl_02 / count_02, torch.zeros((), device=device, dtype=torch.float32))
    mean_kl_13 = torch.where(count_13 > 0, sum_kl_13 / count_13, torch.zeros((), device=device, dtype=torch.float32))
    mean_en_02 = torch.where(count_02 > 0, sum_en_02 / count_02, torch.zeros((), device=device, dtype=torch.float32))
    mean_en_13 = torch.where(count_13 > 0, sum_en_13 / count_13, torch.zeros((), device=device, dtype=torch.float32))
    mean_cf_02 = torch.where(count_02 > 0, sum_cf_02 / count_02, torch.zeros((), device=device, dtype=torch.float32))
    mean_cf_13 = torch.where(count_13 > 0, sum_cf_13 / count_13, torch.zeros((), device=device, dtype=torch.float32))
    out['by_seat/kl_02'] = mean_kl_02
    out['by_seat/kl_13'] = mean_kl_13
    out['by_seat/entropy_02'] = mean_en_02
    out['by_seat/entropy_13'] = mean_en_13
    out['by_seat/clip_frac_02'] = mean_cf_02
    out['by_seat/clip_frac_13'] = mean_cf_13
    return out


def _load_frozen_actor(ckpt_path: str, obs_dim: int) -> ActionConditionedActor:
    actor = ActionConditionedActor(obs_dim=obs_dim)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'actor' in ckpt:
            actor.load_state_dict(ckpt['actor'])
        # else: leave randomly init
    except Exception:
        pass
    actor.eval()
    return actor


def collect_trajectory(env: ScoponeEnvMA, agent: ActionConditionedPPO, horizon: int = 128,
                       gamma: float = 0.99, lam: float = 0.95,
                       partner_actor: ActionConditionedActor = None,
                       opponent_actor: ActionConditionedActor = None,
                       main_seats: List[int] = None,
                       belief_particles: int = 512, belief_ess_frac: float = 0.5,
                       episodes: int = None, final_reward_only: bool = True,
                       use_mcts: bool = True,
                       mcts_sims: int = 128, mcts_dets: int = 4, mcts_c_puct: float = 1.0,
                       mcts_root_temp: float = 0.0, mcts_prior_smooth_eps: float = 0.1,
                       mcts_dirichlet_alpha: float = 0.25, mcts_dirichlet_eps: float = 0.25,
                       mcts_train_factor: float = 1.0,
                       mcts_progress_start: float = 0.25,
                       mcts_progress_full: float = 0.75,
                       mcts_min_sims: int = 0,
                       train_both_teams: bool = False) -> Dict:
    obs_list, next_obs_list = [], []
    act_list = []
    rew_list, done_list = [], []
    legals_list, legals_offset, legals_count = [], [], []
    chosen_index_t_list = []
    seat_team_list = []
    belief_sum_list = []
    mcts_policy_flat = []  # concatenazione delle distribuzioni visite MCTS per ciascun sample main
    mcts_weight_list = []  # 1.0 se MCTS usato per il sample, altrimenti 0.0
    # supervision per belief aux: per ogni sample, vettore (3,40) one-hot di mani reali altrui
    others_hands_targets = []

    
    routing_log = []  # (player_id, source)

    steps = 0
    if final_reward_only:
        # Raccogli per episodi completi: se non specificato, usa multipli di 40 derivati da horizon
        episodes = (max(1, horizon // 40) if episodes is None else max(1, int(episodes)))
        episodes_done = 0
    # Tracciamento delle slice episodio e dei team rewards per reward flat
    current_ep_start_idx = 0
    ep_slices: List[Tuple[int, int]] = []
    ep_team_rewards: List[List[float]] = []
    while True:
        if env.done:
            env.reset()
            current_ep_start_idx = len(obs_list)
            

        # All env logic on CPU
        obs = env._get_observation(env.current_player)
        legal = env.get_valid_actions()
        if not legal:
            break

        cp = env.current_player
        seat_vec = torch.zeros(6, dtype=torch.float32, device='cpu')
        seat_vec[cp] = 1.0
        seat_vec[4] = 1.0 if cp in [0, 2] else 0.0
        seat_vec[5] = 1.0 if cp in [1, 3] else 0.0

        # Selezione azione: se train_both_teams è True, tutti i seat sono "main"
        is_main = True if train_both_teams else ((main_seats is None and cp in [0, 2]) or (main_seats is not None and cp in main_seats))
        if is_main:
            # Belief summary per il giocatore corrente ad ogni sua mossa
            # belief neurale: usa actor per predire marginali; mantieni tensor 120-d su CPU per batch
            o_cpu = obs.clone().detach().to('cpu', dtype=torch.float32) if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32, device='cpu')
            s_cpu = seat_vec.clone().detach().to('cpu', dtype=torch.float32)
            o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
            s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
            with torch.no_grad():
                state_feat = agent.actor.state_enc(o_t, s_t)
                logits = agent.actor.belief_net(state_feat)
                # visible mask da obs
                hand_table = o_t[:, :83]
                hand_mask = hand_table[:, :40] > 0.5
                table_mask = hand_table[:, 43:83] > 0.5
                captured = o_t[:, 83:165]
                cap0_mask = captured[:, :40] > 0.5
                cap1_mask = captured[:, 40:80] > 0.5
                visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)
                probs_flat = agent.actor.belief_net.probs(logits, visible_mask)
            bsum = probs_flat.squeeze(0).detach().to('cpu')
            # MCTS sempre attivo (stile AlphaZero): poche simulazioni sempre, scala con il progresso della mano
            try:
                progress = float(min(1.0, max(0.0, len(env.game_state.get('history', [])) / 40.0)))
            except Exception:
                progress = 0.0
            use_mcts_cur = bool(use_mcts and len(legal) > 0)
            if use_mcts_cur:
                # Se il fattore di training è 0, disabilita MCTS del tutto (nessun minimo forzato)
                if mcts_train_factor is not None and float(mcts_train_factor) <= 0.0:
                    sims_scaled = 0
                    use_mcts_cur = False
                else:
                    # scala simulazioni in base al progresso
                    denom = max(1e-6, (mcts_progress_full - mcts_progress_start))
                    alpha = min(1.0, max(0.0, (progress - mcts_progress_start) / denom))
                    sims_scaled = int(round(mcts_sims * (0.25 + 0.75 * alpha) * float(mcts_train_factor)))
                    use_mcts_cur = sims_scaled > 0
            if use_mcts_cur:
                # MCTS con determinizzazione dal belief del giocatore corrente
                from algorithms.is_mcts import run_is_mcts
                import numpy as _np
                # Policy: usa l'actor per generare prior sui legali
                def policy_fn_mcts(_obs, _legals):
                    # scorri direttamente con actor: (B=1) logits su legali
                    o_cpu = _obs if torch.is_tensor(_obs) else torch.as_tensor(_obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    leg_cpu = torch.stack([
                        (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
                    for x in _legals], dim=0)
                    o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
                    # actor forward usa belief neurale interno
                    with torch.no_grad():
                        logits = agent.actor(o_t, leg_t)
                        priors = torch.softmax(logits, dim=0).detach().cpu().numpy()
                    return priors
                # Value: usa il critic con belief neurale interno
                def value_fn_mcts(_obs, _env):
                    # Prepara seat_team
                    o_cpu = _obs if torch.is_tensor(_obs) else torch.as_tensor(_obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    # seat vector
                    seat_vec = torch.zeros(6, dtype=torch.float32)
                    cp_loc = _env.current_player
                    seat_vec[cp_loc] = 1.0
                    seat_vec[4] = 1.0 if cp_loc in [0, 2] else 0.0
                    seat_vec[5] = 1.0 if cp_loc in [1, 3] else 0.0
                    # To CUDA
                    o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    s_t = seat_vec.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    with torch.no_grad():
                        v = agent.critic(o_t, s_t)
                    return float(v.squeeze(0).detach().cpu().item())
                # Belief determinization sampler dal BeliefNet: campiona assignment coerenti
                def belief_sampler_neural(_env):
                    try:
                        # Costruisci marginali 3x40 dal BeliefNet
                        obs_cur = _env._get_observation(_env.current_player)
                        o_cpu = obs_cur if torch.is_tensor(obs_cur) else torch.as_tensor(obs_cur, dtype=torch.float32)
                        if torch.is_tensor(o_cpu):
                            o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                        s_cpu = _seat_vec_for(_env.current_player).detach().to('cpu', dtype=torch.float32)
                        o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        with torch.no_grad():
                            state_feat = agent.actor.state_enc(o_t, s_t)
                            logits = agent.actor.belief_net(state_feat)  # (1,120)
                            # Visibilità dall'osservazione
                            hand_table = o_t[:, :83]
                            hand_mask = hand_table[:, :40] > 0.5
                            table_mask = hand_table[:, 43:83] > 0.5
                            captured = o_t[:, 83:165]
                            cap0_mask = captured[:, :40] > 0.5
                            cap1_mask = captured[:, 40:80] > 0.5
                            visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)  # (1,40)
                            probs_flat = agent.actor.belief_net.probs(logits, visible_mask)  # (1,120)
                        probs = probs_flat.view(3, 40).detach().cpu().numpy()  # (3,40)
                        vis = visible_mask.squeeze(0).detach().cpu().numpy().astype(bool)
                        unknown_ids = [cid for cid in range(40) if not vis[cid]]
                        # Capacità (conteggi mano) correnti degli altri giocatori
                        others = [(_env.current_player + 1) % 4, (_env.current_player + 2) % 4, (_env.current_player + 3) % 4]
                        counts = {pid: len(_env.game_state['hands'][pid]) for pid in others}
                        caps = [int(counts.get(pid, 0)) for pid in others]
                        n = len(unknown_ids)
                        # riallinea capacità se necessario
                        if sum(caps) != n:
                            caps[2] = max(0, n - caps[0] - caps[1])
                            if sum(caps) != n:
                                # fallback uniforme
                                base = n // 3
                                rem = n - 3 * base
                                caps = [base, base, base]
                                for i in range(rem):
                                    caps[i] += 1
                        # Costi = -log p con piccolo rumore per diversità
                        noise_scale = float(os.environ.get('DET_NOISE', '0.0'))
                        costs = []  # shape (n,3)
                        for cid in unknown_ids:
                            pc = probs[:, cid]
                            ps = pc / max(1e-12, pc.sum())
                            c = [-_np.log(max(1e-12, ps[i])) for i in range(3)]
                            if noise_scale > 0:
                                u = _np.random.uniform(1e-9, 1.0-1e-9, size=3)
                                g = -_np.log(-_np.log(u)) * noise_scale
                                c = [c[i] + float(g[i]) for i in range(3)]
                            costs.append(c)
                        # DP ottimo per 3 giocatori con capacità note
                        INF = 1e12
                        cap0, cap1, cap2 = caps
                        dp = [[[INF]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                        bk = [[[-1]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                        dp[0][0][0] = 0.0
                        for t in range(n):
                            c0, c1, c2 = costs[t]
                            for a in range(0, min(t, cap0)+1):
                                for b in range(0, min(t-a, cap1)+1):
                                    cur = dp[t][a][b]
                                    if cur >= INF: 
                                        continue
                                    # assign to player 0
                                    if a+1 <= cap0:
                                        if dp[t+1][a+1][b] > cur + c0:
                                            dp[t+1][a+1][b] = cur + c0
                                            bk[t+1][a+1][b] = 0
                                    # assign to player 1
                                    if b+1 <= cap1:
                                        if dp[t+1][a][b+1] > cur + c1:
                                            dp[t+1][a][b+1] = cur + c1
                                            bk[t+1][a][b+1] = 1
                                    # assign to player 2 (implicit count)
                                    assigned2 = t - a - b
                                    if assigned2 + 1 <= cap2:
                                        if dp[t+1][a][b] > cur + c2:
                                            dp[t+1][a][b] = cur + c2
                                            bk[t+1][a][b] = 2
                        if dp[n][cap0][cap1] >= INF:
                            # fallback: ritorna None per triggerare greedy di backup
                            return None
                        # Ricostruisci percorso
                        det = {pid: [] for pid in others}
                        a, b = cap0, cap1
                        for t in range(n, 0, -1):
                            choice = bk[t][a][b]
                            cid = unknown_ids[t-1]
                            if choice == 0:
                                det[others[0]].append(cid)
                                a -= 1
                            elif choice == 1:
                                det[others[1]].append(cid)
                                b -= 1
                            else:
                                det[others[2]].append(cid)
                        return det
                    except Exception:
                        return None
                # temperatura radice dinamica: alta a inizio mano, bassa verso la fine
                root_temp_dyn = float(mcts_root_temp) if float(mcts_root_temp) > 0 else float(max(0.0, 1.0 - alpha))
                mcts_action, mcts_visits = run_is_mcts(env,
                                          policy_fn=policy_fn_mcts,
                                          value_fn=value_fn_mcts,
                                          num_simulations=int(sims_scaled),
                                          c_puct=float(mcts_c_puct),
                                          belief=None,
                                          num_determinization=int(mcts_dets),
                                          root_temperature=root_temp_dyn,
                                          prior_smooth_eps=float(mcts_prior_smooth_eps if mcts_prior_smooth_eps > 0 else 0.1),
                                          robust_child=True,
                                          root_dirichlet_alpha=float(mcts_dirichlet_alpha),
                                          root_dirichlet_eps=float(mcts_dirichlet_eps if mcts_dirichlet_eps > 0 else 0.25),
                                          return_stats=True,
                                          belief_sampler=belief_sampler_neural)
                chosen_act = mcts_action if torch.is_tensor(mcts_action) else torch.as_tensor(mcts_action, dtype=torch.float32)
                # trova indice dell'azione scelta tra i legali con key robusto (posizioni non-zero)
                def _act_key(x_t: torch.Tensor):
                    xt = x_t if torch.is_tensor(x_t) else torch.as_tensor(x_t, dtype=torch.float32)
                    return tuple(torch.nonzero(xt > 0.5, as_tuple=False).flatten().tolist())
                key_target = _act_key(chosen_act)
                idx_t = None
                for i_a, a in enumerate(legal):
                    try:
                        if _act_key(a) == key_target:
                            idx_t = torch.tensor(i_a, dtype=torch.long)
                            break
                    except Exception:
                        continue
                if idx_t is None:
                    idx_t = torch.tensor(0, dtype=torch.long)
                next_obs, rew, done, info = env.step(chosen_act)
                routing_log.append((cp, 'mcts'))
                # registra target distillazione per questo sample (ordine legali corrente)
                try:
                    mcts_probs = torch.as_tensor(mcts_visits, dtype=torch.float32)
                    # normalizza in caso di degenerazione
                    s = float(mcts_probs.sum().item())
                    if s > 0:
                        mcts_probs = mcts_probs / s
                except Exception:
                    mcts_probs = torch.full((len(legal),), 0.0, dtype=torch.float32)
                mcts_policy_flat.extend((mcts_probs.tolist() if hasattr(mcts_probs, 'tolist') else list(mcts_probs)))
                mcts_weight_list.append(1.0)
            else:
                chosen_act, _logp, idx_t = agent.select_action(obs, legal, seat_vec)
                next_obs, rew, done, info = env.step(chosen_act)
                routing_log.append((cp, 'main'))
                # Nessuna distillazione per questo sample
                mcts_policy_flat.extend([0.0] * len(legal))
                mcts_weight_list.append(0.0)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            act_list.append(chosen_act)
            rew_list.append(rew)
            done_list.append(done)
            seat_team_list.append(seat_vec)
            belief_sum_list.append(bsum)
            legals_offset.append(len(legals_list))
            legals_count.append(len(legal))
            chosen_index_t_list.append(idx_t)
            legals_list.extend(legal)
            # costruisci target mani reali altrui (3x40) da stato env
            try:
                hands = env.game_state.get('hands', None)
                if hands is not None:
                    others = [ (cp + 1) % 4, (cp + 2) % 4, (cp + 3) % 4 ]
                    target = torch.zeros((3,40), dtype=torch.float32)
                    for i,pid in enumerate(others):
                        for c in hands[pid]:
                            # env hands should be IDs; if tuple, convert inline
                            if isinstance(c, int):
                                cid = c
                            else:
                                r, s = c
                                suit_to_int = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
                                cid = int((r - 1) * 4 + suit_to_int[s])
                            target[i, int(cid)] = 1.0
                    others_hands_targets.append(target)
                else:
                    others_hands_targets.append(torch.zeros((3,40), dtype=torch.float32))
            except Exception:
                others_hands_targets.append(torch.zeros((3,40), dtype=torch.float32))
        else:
            # partner congelato sui seat del compagno; opponent sugli avversari
            is_partner_seat = (cp in [0, 2] and (main_seats == [1, 3])) or (cp in [1, 3] and (main_seats == [0, 2]))
            frozen = partner_actor if (is_partner_seat and partner_actor is not None) else opponent_actor
            if frozen is not None:
                with torch.no_grad():
                    # Use GPU for frozen actor scoring but keep env data on CPU
                    o_cpu = obs.clone().detach().to('cpu', dtype=torch.float32) if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32, device='cpu')
                    leg_cpu = torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device='cpu') for x in legal], dim=0)
                    o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
                    logits = frozen(o_t, leg_t)
                    idx_t = torch.argmax(logits).to('cpu')
                    act = leg_cpu[idx_t]
            else:
                idx_t = torch.randint(len(legal), (1,), device='cpu').squeeze(0)
                leg_t = torch.stack([
                    x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device='cpu')
                for x in legal], dim=0)
                act = leg_t[idx_t]
            next_obs, rew, done, info = env.step(act)
            routing_log.append((cp, 'partner' if is_partner_seat else 'opponent'))
        

        steps += 1
        # Condizioni di uscita: per episodi o per passi
        if final_reward_only:
            if done:
                # Registra i confini dell'episodio corrente e i team rewards
                ep_slices.append((current_ep_start_idx, len(obs_list)))
                try:
                    tr = info.get('team_rewards', [0.0, 0.0]) if isinstance(info, dict) else [0.0, 0.0]
                except Exception:
                    tr = [0.0, 0.0]
                ep_team_rewards.append(tr)
                if episodes_done >= (episodes - 1):
                    break
                else:
                    episodes_done += 1
                    current_ep_start_idx = len(obs_list)
                    continue
        else:
            if steps >= horizon:
                break

    # CTDE: stima V(next) vettorizzata su GPU
    next_val_t = None
    if len(next_obs_list) > 0:
        with torch.no_grad():
            next_obs_t = torch.stack([no if torch.is_tensor(no) else torch.as_tensor(no, dtype=torch.float32) for no in next_obs_list], dim=0).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            s_all = torch.stack(seat_team_list, dim=0).pin_memory().to(device=device, non_blocking=True)
            b_all = torch.stack(belief_sum_list, dim=0).pin_memory().to(device=device, non_blocking=True)
            done_mask_bool = torch.as_tensor([bool(d) for d in done_list], dtype=torch.bool, device=device)
            next_val_t = agent.critic(next_obs_t, s_all)
            next_val_t = torch.where(done_mask_bool, torch.zeros_like(next_val_t), next_val_t)

    # Compute V(obs) in batch su GPU e GAE
    T = len(rew_list)
    # Costruisci reward flat ±1 per tutte le transizioni dell'episodio, per entrambi i team
    if T > 0 and len(ep_slices) > 0:
        flat_rew = [0.0] * T
        for i_ep, (s, e) in enumerate(ep_slices):
            tr = ep_team_rewards[i_ep] if i_ep < len(ep_team_rewards) else [0.0, 0.0]
            # Determina la reward per team 0 e team 1 (proporzionale al risultato finale)
            try:
                t0 = float(tr[0])
            except Exception:
                # Supporta struttura dict {0:pos,1:neg}
                t0 = float(tr.get(0, 0.0)) if isinstance(tr, dict) else 0.0
            try:
                t1 = float(tr[1])
            except Exception:
                t1 = float(tr.get(1, -t0)) if isinstance(tr, dict) else -t0
            for i in range(s, e):
                st = seat_team_list[i]
                try:
                    team0_flag = bool(float(st[4]) > 0.5)
                except Exception:
                    # fallback: deduci dal seat index 0/2 vs 1/3
                    try:
                        seat_idx = int(torch.argmax(st[:4]).item()) if torch.is_tensor(st) else int(max(range(4), key=lambda k: st[k]))
                    except Exception:
                        seat_idx = 0
                    team0_flag = seat_idx in [0, 2]
                flat_rew[i] = t0 if team0_flag else t1
        rew_t = torch.as_tensor(flat_rew, dtype=torch.float32, device=device)
    else:
        rew_t = torch.as_tensor(rew_list, dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    done_mask = torch.as_tensor([0.0 if not d else 1.0 for d in done_list], dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    if T > 0:
        with torch.no_grad():
            o_all = torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o, dtype=torch.float32) for o in obs_list], dim=0).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            s_all = torch.stack(seat_team_list, dim=0).pin_memory().to(device=device, non_blocking=True)
            val_t = agent.critic(o_all, s_all)
            nval_t = next_val_t if next_val_t is not None else torch.zeros_like(val_t)
    else:
        val_t = torch.zeros((0,), dtype=torch.float32, device=device)
        nval_t = torch.zeros((0,), dtype=torch.float32, device=device)
    adv_vec = torch.zeros_like(rew_t)
    gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    for t in reversed(range(T)):
        delta = rew_t[t] + gamma * nval_t[t] - val_t[t]
        gae = delta + gamma * lam * (1.0 - done_mask[t]) * gae
        adv_vec[t] = gae
    ret_vec = adv_vec + val_t

    # Keep batch entirely as torch tensors on CUDA
    # Build CPU tensors first, then pin and transfer as a batch later in update
    obs_cpu = torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o, dtype=torch.float32) for o in obs_list], dim=0) if len(obs_list)>0 else torch.zeros((0, env.observation_space.shape[0]), dtype=torch.float32)
    act_cpu = torch.stack([a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32) for a in act_list], dim=0) if len(act_list)>0 else torch.zeros((0, 80), dtype=torch.float32)
    legals_cpu = torch.stack([l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.float32) for l in legals_list], dim=0) if legals_list else torch.zeros((0, 80), dtype=torch.float32)
    seat_team_cpu = torch.stack(seat_team_list, dim=0) if len(seat_team_list)>0 else torch.zeros((0,6), dtype=torch.float32)
    belief_sum_cpu = torch.stack(belief_sum_list, dim=0) if len(belief_sum_list)>0 else torch.zeros((0,120), dtype=torch.float32)
    legals_offset_cpu = torch.as_tensor(legals_offset, dtype=torch.long) if len(legals_offset)>0 else torch.zeros((0,), dtype=torch.long)
    legals_count_cpu = torch.as_tensor(legals_count, dtype=torch.long) if len(legals_count)>0 else torch.zeros((0,), dtype=torch.long)
    chosen_index_cpu = (torch.stack(chosen_index_t_list, dim=0).to(dtype=torch.long) if len(chosen_index_t_list)>0 else torch.zeros((0,), dtype=torch.long))
    mcts_policy_cpu = torch.as_tensor(mcts_policy_flat, dtype=torch.float32) if mcts_policy_flat else torch.zeros((0,), dtype=torch.float32)
    mcts_weight_cpu = torch.as_tensor(mcts_weight_list, dtype=torch.float32) if mcts_weight_list else torch.zeros((0,), dtype=torch.float32)
    others_hands_cpu = torch.stack(others_hands_targets, dim=0) if len(others_hands_targets)>0 else torch.zeros((0,3,40), dtype=torch.float32)
    # Sanitizza lunghezze: policy_flat deve avere somma(cnts) elementi e weight deve avere len(obs)
    total_legals = int(legals_count_cpu.sum().item()) if len(legals_count_cpu) > 0 else 0
    if mcts_policy_cpu.numel() != total_legals:
        if mcts_policy_cpu.numel() > total_legals:
            mcts_policy_cpu = mcts_policy_cpu[:total_legals]
        else:
            pad = torch.zeros((total_legals - mcts_policy_cpu.numel(),), dtype=torch.float32)
            mcts_policy_cpu = torch.cat([mcts_policy_cpu, pad], dim=0)
    total_samples = len(obs_list)
    if mcts_weight_cpu.numel() != total_samples:
        if mcts_weight_cpu.numel() > total_samples:
            mcts_weight_cpu = mcts_weight_cpu[:total_samples]
        else:
            padw = torch.zeros((total_samples - mcts_weight_cpu.numel(),), dtype=torch.float32)
            mcts_weight_cpu = torch.cat([mcts_weight_cpu, padw], dim=0)
    # Keep rewards/done on GPU for GAE already computed; store CPU too for logging if needed
    ret_t = ret_vec
    adv_t = adv_vec
    rew_t = rew_t
    done_t = torch.as_tensor(done_list, dtype=torch.bool, device=device) if len(done_list)>0 else torch.zeros((0,), dtype=torch.bool, device=device)

    # Calcola old_logp in batch per evitare sincronizzazioni step-by-step (transfer to CUDA once)
    if obs_cpu.size(0) > 0:
        with torch.no_grad():
            # Ensure source are CPU tensors before pinning
            def to_pinned(x):
                return (x.detach().to('cpu') if torch.is_tensor(x) else torch.as_tensor(x)).pin_memory()
            obs_t = to_pinned(obs_cpu).to(device=device, dtype=torch.float32, non_blocking=True)
            seat_team_t = to_pinned(seat_team_cpu).to(device=device, non_blocking=True)
            legals_t = to_pinned(legals_cpu).to(device=device, non_blocking=True)
            legals_offset_t = to_pinned(legals_offset_cpu).to(device=device, non_blocking=True)
            legals_count_t = to_pinned(legals_count_cpu).to(device=device, non_blocking=True)
            chosen_index_t = to_pinned(chosen_index_cpu).to(device=device, non_blocking=True)
            # belief summary may be absent
            # belief non più richiesto per calcolare old_logp
            # Two-stage old_logp
            B = obs_t.size(0)
            max_cnt = int(legals_count_t.max().item()) if B > 0 else 0
            if max_cnt > 0:
                state_proj = agent.actor.compute_state_proj(obs_t, seat_team_t)  # (B,64)
                card_emb = agent.actor.card_emb_play.to(device)
                card_logits_all = torch.matmul(state_proj, card_emb.t())  # (B,40)
                pos = torch.arange(max_cnt, device=device, dtype=torch.long)
                rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
                mask = rel_pos_2d < legals_count_t.unsqueeze(1)
                abs_idx = (legals_offset_t.unsqueeze(1) + rel_pos_2d)[mask]
                sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
                legals_mb = legals_t[abs_idx].contiguous()
                played_ids_mb = torch.argmax(legals_mb[:, :40], dim=1)
                # card logp masked per sample
                card_mask = torch.zeros((B, 40), dtype=torch.bool, device=device)
                card_mask[sample_idx_per_legal, played_ids_mb] = True
                masked_card_logits = card_logits_all.masked_fill(~card_mask, float('-inf'))
                logp_cards = torch.log_softmax(masked_card_logits, dim=1)
                chosen_clamped = torch.minimum(chosen_index_t, (legals_count_t - 1).clamp_min(0))
                chosen_abs = (legals_offset_t + chosen_clamped)
                total_legals = legals_t.size(0)
                pos_map = torch.full((total_legals,), -1, dtype=torch.long, device=device)
                pos_map[abs_idx] = torch.arange(abs_idx.numel(), device=device, dtype=torch.long)
                chosen_pos = pos_map[chosen_abs]
                played_ids_all = torch.argmax(legals_t[:, :40], dim=1)
                chosen_card_ids = played_ids_all[chosen_abs]
                logp_card = logp_cards[torch.arange(B, device=device), chosen_card_ids]
                # capture
                a_emb_mb = agent.actor.action_enc(legals_mb)
                cap_logits = (a_emb_mb * state_proj[sample_idx_per_legal]).sum(dim=1)
                group_ids = sample_idx_per_legal * 40 + played_ids_mb
                num_groups = B * 40
                group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
                try:
                    group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
                except Exception:
                    tmp = torch.zeros_like(group_max)
                    tmp.index_copy_(0, group_ids, cap_logits)
                    group_max = torch.maximum(group_max, tmp)
                gmax_per_legal = group_max[group_ids]
                exp_shifted = torch.exp(cap_logits - gmax_per_legal)
                group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
                group_sum.index_add_(0, group_ids, exp_shifted)
                lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
                logp_cap_per_legal = cap_logits - lse_per_legal
                logp_cap = logp_cap_per_legal[chosen_pos]
                old_logp_t = (logp_card + logp_cap)
            else:
                old_logp_t = torch.zeros((B,), dtype=torch.float32, device=device)
    else:
        old_logp_t = torch.zeros((0,), dtype=torch.float32, device=device)

    # Package CPU copies; transfer inside update to minimize H2D events
    batch = {
        'obs': obs_cpu,
        'act': act_cpu,
        'old_logp': old_logp_t.detach().to('cpu'),
        'ret': ret_t.detach().to('cpu'),
        'adv': adv_t.detach().to('cpu'),
        'rew': rew_t,
        'done': done_t,
        'seat_team': seat_team_cpu,
        'belief_summary': belief_sum_cpu,
        'legals': legals_cpu,
        'legals_offset': legals_offset_cpu,
        'legals_count': legals_count_cpu,
        'chosen_index': chosen_index_cpu,
        'mcts_policy': mcts_policy_cpu,
        'mcts_weight': mcts_weight_cpu,
        'others_hands': others_hands_cpu,
        'routing_log': routing_log,
    }
    return batch


def collect_trajectory_parallel(agent: ActionConditionedPPO,
                                num_envs: int = 32,
                                episodes_total_hint: int = 8,
                                use_compact_obs: bool = True,
                                k_history: int = 39,
                                gamma: float = 1.0,
                                lam: float = 0.95,
                                use_mcts: bool = True,
                                train_both_teams: bool = True,
                                main_seats: List[int] = None,
                                mcts_sims: int = 128,
                                mcts_dets: int = 4,
                                mcts_c_puct: float = 1.0,
                                mcts_root_temp: float = 0.0,
                                mcts_prior_smooth_eps: float = 0.1,
                                mcts_dirichlet_alpha: float = 0.25,
                                mcts_dirichlet_eps: float = 0.25,
                                mcts_progress_start: float = 0.25,
                                mcts_progress_full: float = 0.75,
                                mcts_min_sims: int = 0) -> Dict:
    # Use 'spawn' on CUDA to avoid fork+CUDA deadlocks; fallback to 'fork' otherwise
    try:
        start_method = 'spawn' if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'fork'
    except Exception:
        start_method = 'fork'
    ctx = mp.get_context(start_method)
    request_q = ctx.Queue(maxsize=num_envs * 4)
    episode_q = ctx.Queue(maxsize=num_envs * 2)
    action_queues = [ctx.Queue(maxsize=2) for _ in range(num_envs)]
    # Distribute episodes roughly equally
    episodes_per_env = max(1, int((episodes_total_hint + num_envs - 1) // num_envs))
    workers = []
    cfg = {
        'rules': {'shape_scopa': False},
        'use_compact_obs': bool(use_compact_obs),
        'k_history': int(k_history),
        'episodes_per_env': episodes_per_env,
        'send_legals': True,
        'use_mcts': bool(use_mcts),
        'train_both_teams': bool(train_both_teams),
        'main_seats': main_seats if main_seats is not None else [0,2],
        'mcts_sims': int(mcts_sims),
        'mcts_dets': int(mcts_dets),
        'mcts_c_puct': float(mcts_c_puct),
        'mcts_root_temp': float(mcts_root_temp),
        'mcts_prior_smooth_eps': float(mcts_prior_smooth_eps),
        'mcts_dirichlet_alpha': float(mcts_dirichlet_alpha),
        'mcts_dirichlet_eps': float(mcts_dirichlet_eps),
        'mcts_progress_start': float(mcts_progress_start),
        'mcts_progress_full': float(mcts_progress_full),
        'mcts_min_sims': int(mcts_min_sims),
    }
    for wid in range(num_envs):
        p = ctx.Process(target=_env_worker, args=(wid, cfg, request_q, action_queues[wid], episode_q), daemon=True)
        p.start()
        workers.append(p)

    episodes_received = 0
    episodes_payloads = []
    # Main loop: service action requests and collect episodes
    while episodes_received < num_envs * episodes_per_env:
        # Gather a micro-batch of requests
        reqs = []
        try:
            r0 = request_q.get(timeout=0.05)
            # Always enqueue the first request regardless of type ('step', 'score_*', etc.)
            reqs.append(r0)
        except Exception:
            pass
        # Drain more without blocking (collect both 'step' and other request types)
        for _ in range(max(1, num_envs)):
            try:
                r = request_q.get_nowait()
                reqs.append(r)
            except Exception:
                break
        # Process batch on GPU
        if len(reqs) > 0:
            # Split by type: 'step' vs scoring
            step_reqs = [r for r in reqs if r.get('type') == 'step']
            other_reqs = [r for r in reqs if r.get('type') != 'step']
            if len(step_reqs) > 0:
                sel = _batched_select_indices(agent, step_reqs)
                for (wid, idx) in sel:
                    try:
                        action_queues[wid].put({'idx': int(idx)}, block=False)
                    except Exception:
                        action_queues[wid].put({'idx': int(idx)})
            if len(other_reqs) > 0:
                outs = _batched_service(agent, other_reqs)
                for r, out in zip(other_reqs, outs):
                    wid = int(r.get('wid', 0))
                    try:
                        action_queues[wid].put(out, block=False)
                    except Exception:
                        action_queues[wid].put(out)
        # Drain any completed episodes
        while True:
            try:
                ep = episode_q.get_nowait()
                episodes_payloads.append(ep)
                episodes_received += 1
            except Exception:
                break

    # Join workers
    for p in workers:
        try:
            p.join(timeout=0.1)
        except Exception:
            pass

    # Build batch CPU tensors from payloads
    obs_cpu = []
    act_cpu = []
    next_obs_cpu = []
    rew_list = []
    done_list = []
    seat_cpu = []
    belief_cpu = []
    legals_cpu = []
    leg_off = []
    leg_cnt = []
    chosen_idx = []
    for ep in episodes_payloads:
        obs_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['obs']]
        next_obs_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['next_obs']]
        act_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['act']]
        seat_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['seat']]
        # Determina reward episodio dal team 0/1 (proporzionale al risultato)
        tr = ep.get('team_rewards', [0.0, 0.0])
        try:
            t0 = float(tr[0])
        except Exception:
            t0 = float(tr.get(0, 0.0)) if isinstance(tr, dict) else 0.0
        try:
            t1 = float(tr[1])
        except Exception:
            t1 = float(tr.get(1, -t0)) if isinstance(tr, dict) else -t0
        rew_ep = []
        for st in seat_ep:
            team0_flag = bool(float(st[4]) > 0.5)
            if not team0_flag:
                try:
                    seat_idx = int(torch.argmax(st[:4]).item())
                except Exception:
                    seat_idx = 0
                team0_flag = seat_idx in [0, 2]
            rew_ep.append(t0 if team0_flag else t1)

        obs_cpu.extend(obs_ep)
        next_obs_cpu.extend(next_obs_ep)
        act_cpu.extend(act_ep)
        rew_list.extend(rew_ep)
        done_list.extend(ep['done'])
        seat_cpu.extend(seat_ep)
        belief_cpu.extend([torch.as_tensor(x, dtype=torch.float32) for x in ep['belief_summary']])
        # Adjust offsets by current length of legals_cpu
        base = len(legals_cpu)
        leg_off.extend([base + off for off in ep['leg_off']])
        leg_cnt.extend(ep['leg_cnt'])
        legals_cpu.extend([torch.as_tensor(x, dtype=torch.float32) for x in ep['legals']])
        chosen_idx.extend(ep['chosen_idx'])
        # distill/belief aux
        if 'mcts_policy' in ep:
            # adjust offsets later after concat
            pass

    # Stack to tensors
    obs_cpu_t = torch.stack(obs_cpu, dim=0) if len(obs_cpu) > 0 else torch.zeros((0, 1), dtype=torch.float32)
    next_obs_cpu_t = torch.stack(next_obs_cpu, dim=0) if len(next_obs_cpu) > 0 else torch.zeros((0, 1), dtype=torch.float32)
    act_cpu_t = torch.stack(act_cpu, dim=0) if len(act_cpu) > 0 else torch.zeros((0, 80), dtype=torch.float32)
    seat_cpu_t = torch.stack(seat_cpu, dim=0) if len(seat_cpu) > 0 else torch.zeros((0, 6), dtype=torch.float32)
    belief_cpu_t = torch.stack(belief_cpu, dim=0) if len(belief_cpu) > 0 else torch.zeros((0, 120), dtype=torch.float32)
    legals_cpu_t = torch.stack(legals_cpu, dim=0) if len(legals_cpu) > 0 else torch.zeros((0, 80), dtype=torch.float32)
    leg_off_t = torch.as_tensor(leg_off, dtype=torch.long)
    leg_cnt_t = torch.as_tensor(leg_cnt, dtype=torch.long)
    chosen_idx_t = torch.as_tensor(chosen_idx, dtype=torch.long)
    # Distill/belief aux targets from episodes
    mcts_policy_flat = []
    mcts_weight = []
    others_hands = []
    base = 0
    for ep in episodes_payloads:
        cnts_ep = ep['leg_cnt']
        if 'mcts_policy' in ep:
            mcts_policy_flat.extend(ep['mcts_policy'])
            mcts_weight.extend(ep.get('mcts_weight', [0.0]*len(ep['obs'])))
        if 'others_hands' in ep:
            for oh in ep['others_hands']:
                others_hands.append(torch.as_tensor(oh, dtype=torch.float32))
        base += len(cnts_ep)
    mcts_policy_t = torch.as_tensor(mcts_policy_flat, dtype=torch.float32)
    mcts_weight_t = torch.as_tensor(mcts_weight, dtype=torch.float32)
    others_hands_t = torch.stack(others_hands, dim=0) if len(others_hands)>0 else torch.zeros((0,3,40), dtype=torch.float32)

    # Compute values and advantages on GPU similar to collect_trajectory
    rew_t = torch.as_tensor(rew_list, dtype=torch.float32, device=device) if len(rew_list) > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
    done_mask = torch.as_tensor([0.0 if not d else 1.0 for d in done_list], dtype=torch.float32, device=device) if len(done_list) > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
    if len(rew_list) > 0:
        with torch.no_grad():
            o_all = obs_cpu_t.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            s_all = seat_cpu_t.pin_memory().to(device=device, non_blocking=True)
            b_all = belief_cpu_t.pin_memory().to(device=device, non_blocking=True)
            val_t = agent.critic(o_all, s_all)
            n_all = next_obs_cpu_t.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            nval_t = agent.critic(n_all, s_all)
            nval_t = torch.where(done_mask.bool(), torch.zeros_like(nval_t), nval_t)
    else:
        val_t = torch.zeros((0,), dtype=torch.float32, device=device)
        nval_t = torch.zeros((0,), dtype=torch.float32, device=device)
    adv_vec = torch.zeros_like(rew_t)
    gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    T = int(rew_t.numel())
    for t in reversed(range(T)):
        delta = rew_t[t] + nval_t[t] * gamma - val_t[t]
        gae = delta + gamma * lam * (1.0 - done_mask[t]) * gae
        adv_vec[t] = gae
    ret_vec = adv_vec + val_t

    # Compute old_logp in batch on GPU (factored card + capture scheme)
    if obs_cpu_t.size(0) > 0:
        with torch.no_grad():
            def to_pinned(x):
                return (x.detach().to('cpu') if torch.is_tensor(x) else torch.as_tensor(x)).pin_memory()
            obs_t = to_pinned(obs_cpu_t).to(device=device, dtype=torch.float32, non_blocking=True)
            seat_t = to_pinned(seat_cpu_t).to(device=device, non_blocking=True)
            leg_t = to_pinned(legals_cpu_t).to(device=device, non_blocking=True)
            offs = leg_off_t.pin_memory().to(device=device)
            cnts = leg_cnt_t.pin_memory().to(device=device)
            B = obs_t.size(0)
            max_cnt = int(cnts.max().item()) if B > 0 else 0
            if max_cnt > 0:
                # State projection and card logits
                state_proj = agent.actor.compute_state_proj(obs_t, seat_t)  # (B,64)
                card_emb = agent.actor.card_emb_play.to(device)
                card_logits_all = torch.matmul(state_proj, card_emb.t())  # (B,40)
                pos = torch.arange(max_cnt, device=device, dtype=torch.long)
                rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
                mask = rel_pos_2d < cnts.unsqueeze(1)
                abs_idx = (offs.unsqueeze(1) + rel_pos_2d)[mask]
                sample_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
                legals_mb = leg_t[abs_idx].contiguous()                   # (M_mb,80)
                played_ids_mb = torch.argmax(legals_mb[:, :40], dim=1)    # (M_mb)
                # Mask per-carta per sample
                card_mask = torch.zeros((B, 40), dtype=torch.bool, device=device)
                card_mask[sample_idx, played_ids_mb] = True
                masked_card_logits = card_logits_all.masked_fill(~card_mask, float('-inf'))
                logp_cards = torch.log_softmax(masked_card_logits, dim=1)  # (B,40)
                # Map chosen indices to absolute and card ids
                chosen_clamped = torch.minimum(chosen_idx_t.pin_memory().to(device=device), (cnts - 1).clamp_min(0))
                chosen_abs = (offs + chosen_clamped)
                total_legals = leg_t.size(0)
                pos_map = torch.full((total_legals,), -1, dtype=torch.long, device=device)
                pos_map[abs_idx] = torch.arange(abs_idx.numel(), device=device, dtype=torch.long)
                chosen_pos = pos_map[chosen_abs]
                played_ids_all = torch.argmax(leg_t[:, :40], dim=1)
                chosen_card_ids = played_ids_all[chosen_abs]
                logp_card = logp_cards[torch.arange(B, device=device), chosen_card_ids]
                # Capture log-softmax within card groups
                a_emb_mb = agent.actor.action_enc(legals_mb)               # (M_mb,64)
                cap_logits = (a_emb_mb * state_proj[sample_idx]).sum(dim=1)
                group_ids = sample_idx * 40 + played_ids_mb
                num_groups = B * 40
                group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
                try:
                    group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
                except Exception:
                    tmp = torch.zeros_like(group_max)
                    tmp.index_copy_(0, group_ids, cap_logits)
                    group_max = torch.maximum(group_max, tmp)
                gmax_per_legal = group_max[group_ids]
                exp_shifted = torch.exp(cap_logits - gmax_per_legal)
                group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
                group_sum.index_add_(0, group_ids, exp_shifted)
                lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
                logp_cap_per_legal = cap_logits - lse_per_legal
                logp_cap = logp_cap_per_legal[chosen_pos]
                old_logp_t = (logp_card + logp_cap)
            else:
                old_logp_t = torch.zeros((B,), dtype=torch.float32, device=device)
    else:
        old_logp_t = torch.zeros((0,), dtype=torch.float32, device=device)

    batch = {
        'obs': obs_cpu_t,
        'act': act_cpu_t,
        'old_logp': old_logp_t.detach().to('cpu'),
        'ret': ret_vec.detach().to('cpu'),
        'adv': adv_vec.detach().to('cpu'),
        'rew': rew_t,
        'done': torch.as_tensor(done_list, dtype=torch.bool, device=device) if len(done_list)>0 else torch.zeros((0,), dtype=torch.bool, device=device),
        'seat_team': seat_cpu_t,
        'belief_summary': belief_cpu_t,
        'legals': legals_cpu_t,
        'legals_offset': leg_off_t,
        'legals_count': leg_cnt_t,
        'chosen_index': chosen_idx_t,
        'mcts_policy': mcts_policy_t,
        'mcts_weight': mcts_weight_t,
        'others_hands': others_hands_t,
        'routing_log': [('parallel', 'main')],
    }
    return batch


def train_ppo(num_iterations: int = 1000, horizon: int = 256, save_every: int = 200, ckpt_path: str = 'checkpoints/ppo_ac.pth', use_compact_obs: bool = True, k_history: int = 39, seed: int = 0,
              entropy_schedule_type: str = 'linear', eval_every: int = 0, eval_games: int = 10, belief_particles: int = 512, belief_ess_frac: float = 0.5,
              mcts_in_eval: bool = True, mcts_sims: int = 128, mcts_dets: int = 4, mcts_c_puct: float = 1.0, mcts_root_temp: float = 0.0,
              mcts_prior_smooth_eps: float = 0.0, mcts_dirichlet_alpha: float = 0.25, mcts_dirichlet_eps: float = 0.25,
              num_envs: int = 32,
              on_iter_end: Optional[Callable[[int], None]] = None):
    set_global_seeds(seed)
    # Disattiva reward shaping intermedio: solo reward finale
    env = ScoponeEnvMA(rules={'shape_scopa': False}, use_compact_obs=use_compact_obs, k_history=k_history)
    obs_dim = env.observation_space.shape[0]
    agent = ActionConditionedPPO(obs_dim=obs_dim)

    # Cosine annealing LR schedulers
    actor_sched = optim.lr_scheduler.CosineAnnealingLR(agent.opt_actor, T_max=max(1, num_iterations))
    critic_sched = optim.lr_scheduler.CosineAnnealingLR(agent.opt_critic, T_max=max(1, num_iterations))
    agent.add_lr_schedulers(actor_sched, critic_sched)

    # entropy schedules
    def entropy_schedule_linear(step: int, start: float = 0.01, end: float = 0.001, decay_steps: int = 100000):
        if step >= decay_steps:
            return end
        frac = (decay_steps - step) / decay_steps
        return end + (start - end) * frac

    def entropy_schedule_cosine(step: int, start: float = 0.01, end: float = 0.001, period: int = 100000):
        import math
        t = min(step, period)
        cos = (1 + math.cos(math.pi * t / period)) / 2.0
        return end + (start - end) * cos

    if entropy_schedule_type == 'cosine':
        agent.set_entropy_schedule(lambda s: entropy_schedule_cosine(s))
    else:
        agent.set_entropy_schedule(lambda s: entropy_schedule_linear(s))

    writer = None
    if os.environ.get('SCOPONE_DISABLE_TB', '0') != '1':
        try:
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
            writer = _SummaryWriter(log_dir='runs/ppo_ac')
        except Exception:
            writer = None
    if writer is not None:
        # Spiega le metriche chiave in TensorBoard
        writer.add_text('help/metrics',
                        '\n'.join([
                            'train/loss_pi: Surrogate loss PPO (policy). Negativo = miglioramento dell\'obiettivo.',
                            'train/loss_v: Value loss (MSE). Alto = critico lontano dai return.',
                            'train/entropy: Entropia media della policy (azioni legali). Più alta = più esplorazione.',
                            'train/approx_kl: KL media (stima) tra policy nuova e vecchia (per early stop).',
                            'train/avg_kl: KL media aggregata su tutti i minibatch dell\'update.',
                            'train/clip_frac: Frazione di campioni con rapporto clipppato (|r-1| > ε).',
                            'train/avg_clip_frac: Media del clip_frac sui minibatch.',
                            'train/grad_norm_actor / train/grad_norm_critic: Norma L2 dei gradienti (diagnostica stabilità).',
                            'train/lr_actor / train/lr_critic: Learning rate correnti (post scheduler).',
                            'train/episode_time_s: Tempo per iterazione (raccolta + update).',
                            'train/avg_return: Return medio del batch raccolto (proxy qualità corrente).',
                            'by_seat/ret_02, by_seat/ret_13: Return medio per gruppo di posti (0/2 vs 1/3).',
                            'by_seat/kl_02, by_seat/kl_13: KL media per gruppo di posti.',
                            'by_seat/entropy_02, by_seat/entropy_13: Entropia media per gruppo di posti.',
                            'by_seat/clip_frac_02, by_seat/clip_frac_13: Frazione di clipping per gruppo.',
                            'league/mini_eval_wr: Win-rate nella mini-valutazione vs checkpoint precedente.',
                            'league/elo_current / league/elo_previous: Elo nel league per corrente/precedente.',
                        ]), 0)
    league = League(base_dir='checkpoints/league')

    partner_actor = None
    opponent_actor = None
    # alterna il main actor tra seat 0/2 e 1/3 per episodi
    even_main_seats = [0, 2]
    odd_main_seats = [1, 3]

    best_return = -1e9
    best_ckpt_path = ckpt_path.replace('.pth', '_best.pth')
    best_wr = -1e9
    best_wr_ckpt_path = ckpt_path.replace('.pth', '_bestwr.pth')

    for it in tqdm(range(num_iterations), desc="PPO iterations"):
        t0 = time.time()
        try:
            p_ckpt, o_ckpt = league.sample_pair()
            # Simple local cache to avoid reloading frozen actors every iter
            if not hasattr(league, '_frozen_cache'):
                league._frozen_cache = {}
            if p_ckpt and os.path.isfile(p_ckpt):
                partner_actor = league._frozen_cache.get(p_ckpt)
                if partner_actor is None:
                    partner_actor = _load_frozen_actor(p_ckpt, obs_dim)
                    league._frozen_cache[p_ckpt] = partner_actor
            if o_ckpt and os.path.isfile(o_ckpt):
                opponent_actor = league._frozen_cache.get(o_ckpt)
                if opponent_actor is None:
                    opponent_actor = _load_frozen_actor(o_ckpt, obs_dim)
                    league._frozen_cache[o_ckpt] = opponent_actor
        except Exception:
            partner_actor = None
        main_seats = even_main_seats if (it % 2 == 0) else odd_main_seats
        use_parallel = (num_envs is not None and int(num_envs) > 1)
        if use_parallel:
            episodes_hint = max(1, horizon // 40)
            batch = collect_trajectory_parallel(agent,
                                                num_envs=int(num_envs),
                                                episodes_total_hint=episodes_hint,
                                                use_compact_obs=use_compact_obs,
                                                k_history=k_history,
                                                gamma=1.0,
                                                lam=0.95,
                                                use_mcts=True,
                                                train_both_teams=True,
                                                main_seats=main_seats,
                                                mcts_sims=mcts_sims,
                                                mcts_dets=mcts_dets,
                                                mcts_c_puct=mcts_c_puct,
                                                mcts_root_temp=mcts_root_temp,
                                                mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                                                mcts_dirichlet_eps=mcts_dirichlet_eps)
        else:
            # Strategia MCTS: warmup senza MCTS per le prime iterazioni, poi scala con il progresso mano
            mcts_train_factor = 0.0 if it < 500 else 1.0
            batch = collect_trajectory(env, agent, horizon=horizon, partner_actor=partner_actor, opponent_actor=opponent_actor, main_seats=main_seats,
                                       belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                       episodes=None, final_reward_only=True,
                                       use_mcts=True,
                                       mcts_sims=mcts_sims, mcts_dets=mcts_dets, mcts_c_puct=mcts_c_puct,
                                       mcts_root_temp=mcts_root_temp, mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                       mcts_dirichlet_alpha=mcts_dirichlet_alpha, mcts_dirichlet_eps=mcts_dirichlet_eps,
                                       mcts_train_factor=mcts_train_factor,
                                       mcts_progress_start=0.25, mcts_progress_full=0.75,
                                       mcts_min_sims=0,
                                       train_both_teams=True,
                                       gamma=1.0,
                                       lam=0.95)
        if len(batch['obs']) == 0:
            continue
        # normalizza vantaggi completamente su GPU (no sync)
        adv = batch['adv']
        if adv.numel() > 0:
            mean = adv.mean()
            std = adv.std()
            std = torch.clamp(std, min=1e-8)
            batch['adv'] = (adv - mean) / std
        info = agent.update(batch, epochs=4, minibatch_size=256)
        dt = time.time() - t0

        # proxy per best: media return del batch
        # All device tensors; compute small stats without moving large arrays
        if len(batch['ret']):
            avg_return = float(batch['ret'].mean().detach().cpu().item())
        else:
            avg_return = 0.0
        if avg_return > best_return:
            best_return = avg_return
            try:
                os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
            except Exception:
                pass
            agent.save(best_ckpt_path)

        # mini-eval periodica e Elo update
        if eval_every and (it + 1) % eval_every == 0 and len(league.history) >= 1:
            cur_tmp = ckpt_path.replace('.pth', f'_tmp_it{it+1}.pth')
            agent.save(cur_tmp)
            league.register(cur_tmp)
            prev_ckpt = league.history[-2] if len(league.history) >= 2 else None
            if prev_ckpt is not None:
                mcts_cfg = None
                if mcts_in_eval:
                    mcts_cfg = {
                        'sims': mcts_sims,
                        'dets': mcts_dets,
                        'c_puct': mcts_c_puct,
                        'root_temp': mcts_root_temp,
                        'prior_smooth_eps': mcts_prior_smooth_eps,
                        'root_dirichlet_alpha': mcts_dirichlet_alpha,
                        'root_dirichlet_eps': mcts_dirichlet_eps,
                        'robust_child': True,
                    }
                wr, _ = evaluate_pair_actors(cur_tmp, prev_ckpt, games=eval_games, use_compact_obs=use_compact_obs, k_history=k_history,
                                             mcts=mcts_cfg, belief_particles=(belief_particles if mcts_in_eval else 0), belief_ess_frac=belief_ess_frac)
                league.update_elo(cur_tmp, prev_ckpt, wr)
                if writer is not None:
                    writer.add_scalar('league/mini_eval_wr', wr, it)
                    writer.add_scalar('league/elo_current', league.elo.get(cur_tmp, 1000.0), it)
                    writer.add_scalar('league/elo_previous', league.elo.get(prev_ckpt, 1000.0), it)
                # salva checkpoint best wr con soglia/CI (Wilson interval)
                import math
                n = max(1, eval_games)
                p = wr
                z = 1.96
                denom = 1 + z*z/n
                center = p + z*z/(2*n)
                margin = z*math.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
                lower = (center - margin) / denom
                improved = wr > best_wr
                meets_threshold = lower >= 0.5  # lower bound sopra il 50%
                if improved or meets_threshold:
                    best_wr = max(best_wr, wr)
                    try:
                        os.makedirs(os.path.dirname(best_wr_ckpt_path), exist_ok=True)
                    except Exception:
                        pass
                    agent.save(best_wr_ckpt_path)
                if wr > best_wr:
                    best_wr = wr
                    try:
                        os.makedirs(os.path.dirname(best_wr_ckpt_path), exist_ok=True)
                    except Exception:
                        pass
                    agent.save(best_wr_ckpt_path)

        if it % 50 == 0:
            def _to_float(x):
                return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
            pretty = {k: round(_to_float(v), 4) for k, v in info.items()}
            print(pretty)
        if writer is not None:
            def _to_float(x):
                return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
            for k, v in info.items():
                writer.add_scalar(f'train/{k}', _to_float(v), it)
            writer.add_scalar('train/episode_time_s', dt, it)
            writer.add_scalar('train/avg_return', avg_return, it)
            writer.add_text('train/main_seats', str(main_seats), it)
            # Log by seat completamente su GPU; singola sync per tutto il blocco
            try:
                seats_t = torch.argmax(batch['seat_team'][:, :4], dim=1)
                ret_t = batch['ret']
                mask_02_t = (seats_t == 0) | (seats_t == 2)
                mask_13_t = (seats_t == 1) | (seats_t == 3)
                cnt_02 = mask_02_t.float().sum()
                cnt_13 = mask_13_t.float().sum()
                sum_ret_02 = (ret_t * mask_02_t.float()).sum()
                sum_ret_13 = (ret_t * mask_13_t.float()).sum()
                mean_ret_02 = torch.where(cnt_02 > 0, sum_ret_02 / cnt_02, torch.zeros((), device=device, dtype=torch.float32))
                mean_ret_13 = torch.where(cnt_13 > 0, sum_ret_13 / cnt_13, torch.zeros((), device=device, dtype=torch.float32))
                diag = _compute_per_seat_diagnostics(agent, batch)
                # Prepara chiavi e valori da sincronizzare in una volta
                keys = [
                    'by_seat/ret_02', 'by_seat/ret_13',
                    'by_seat/kl_02', 'by_seat/kl_13',
                    'by_seat/entropy_02', 'by_seat/entropy_13',
                    'by_seat/clip_frac_02', 'by_seat/clip_frac_13'
                ]
                vals = [
                    mean_ret_02.to(torch.float32),
                    mean_ret_13.to(torch.float32),
                    diag['by_seat/kl_02'].to(torch.float32),
                    diag['by_seat/kl_13'].to(torch.float32),
                    diag['by_seat/entropy_02'].to(torch.float32),
                    diag['by_seat/entropy_13'].to(torch.float32),
                    diag['by_seat/clip_frac_02'].to(torch.float32),
                    diag['by_seat/clip_frac_13'].to(torch.float32),
                ]
                stacked = torch.stack(vals)
                numbers = stacked.detach().cpu().tolist()  # unica sincronizzazione CPU
                for key, num in zip(keys, numbers):
                    writer.add_scalar(key, float(num), it)
            except Exception:
                pass
        if (it + 1) % save_every == 0:
            try:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            except Exception:
                pass
            agent.save(ckpt_path)
            try:
                league.register(ckpt_path)
            except Exception:
                pass
        # Optional profiler or external hook per-iteration
        if on_iter_end is not None:
            try:
                on_iter_end(it)
            except Exception:
                pass
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO Action-Conditioned for Scopone')
    parser.add_argument('--iters', type=int, default=2000, help='Number of PPO iterations')
    parser.add_argument('--horizon', type=int, default=256, help='Rollout horizon (steps) per iteration; con solo reward finale raccoglie ~horizon//40 episodi')
    parser.add_argument('--save-every', type=int, default=200, help='Save checkpoint every N iterations')
    parser.add_argument('--ckpt', type=str, default='checkpoints/ppo_ac.pth', help='Checkpoint path')
    parser.add_argument('--compact', action='store_true', help='Use compact observation (recommended)')
    parser.add_argument('--k-history', type=int, default=39, help='Number of recent moves in compact history')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--entropy-schedule', type=str, default='linear', choices=['linear','cosine'], help='Entropy schedule type')
    parser.add_argument('--eval-every', type=int, default=0, help='Run mini-eval every N iters (0=off)')
    parser.add_argument('--eval-games', type=int, default=10, help='Games per mini-eval')
    parser.add_argument('--belief-particles', type=int, default=512, help='Belief particles for trainer')
    parser.add_argument('--belief-ess-frac', type=float, default=0.5, help='Belief ESS fraction for trainer')
    parser.add_argument('--mcts-eval', action='store_true', help='Use MCTS in mini-eval')
    parser.add_argument('--mcts-train', action='store_true', default=True, help='Use MCTS during training action selection for main seats')
    parser.add_argument('--train-both-teams', action='store_true', help='Train both teams simultaneously (all seats are main)')
    parser.add_argument('--mcts-sims', type=int, default=128)
    parser.add_argument('--mcts-dets', type=int, default=4)
    parser.add_argument('--mcts-c-puct', type=float, default=1.0)
    parser.add_argument('--mcts-root-temp', type=float, default=0.0)
    parser.add_argument('--mcts-prior-smooth-eps', type=float, default=0.0)
    parser.add_argument('--mcts-dirichlet-alpha', type=float, default=0.25)
    parser.add_argument('--mcts-dirichlet-eps', type=float, default=0.25)
    parser.add_argument('--num-envs', type=int, default=32, help='Number of parallel env workers (>=1). 1 disables parallel mode')
    args = parser.parse_args()
    train_ppo(num_iterations=args.iters, horizon=args.horizon, save_every=args.save_every, ckpt_path=args.ckpt,
              use_compact_obs=args.compact, k_history=args.k_history, seed=args.seed,
              entropy_schedule_type=args.entropy_schedule, eval_every=args.eval_every, eval_games=args.eval_games,
              belief_particles=args.belief_particles, belief_ess_frac=args.belief_ess_frac,
              mcts_in_eval=args.mcts_eval, mcts_sims=args.mcts_sims, mcts_dets=args.mcts_dets, mcts_c_puct=args.mcts_c_puct,
              mcts_root_temp=args.mcts_root_temp, mcts_prior_smooth_eps=args.mcts_prior_smooth_eps,
              mcts_dirichlet_alpha=args.mcts_dirichlet_alpha, mcts_dirichlet_eps=args.mcts_dirichlet_eps,
              num_envs=args.num_envs)


