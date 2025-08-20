import torch
from tqdm import tqdm
from typing import Dict, List, Callable, Optional
import os
import time
import sys
from tests.torch_np import np

# Ensure project root is on sys.path when running as script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from belief.belief import BeliefState
from selfplay.league import League
from models.action_conditioned import ActionConditionedActor
from utils.seed import set_global_seeds
from evaluation.eval import evaluate_pair_actors

import torch.optim as optim

device = torch.device("cuda")
# Global perf flags
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
def _compute_per_seat_diagnostics(agent: ActionConditionedPPO, batch: Dict) -> Dict[str, torch.Tensor]:
    """Calcola approx_kl, entropia e clip_frac per gruppi seat 0/2 vs 1/3.

    Tutte le operazioni restano su GPU; ritorna tensori 0-D su device.
    """
    obs = batch['obs'] if torch.is_tensor(batch['obs']) else torch.as_tensor(batch['obs'], dtype=torch.float32, device=device)
    seat = batch.get('seat_team', None)
    if seat is None:
        seat = torch.zeros((obs.size(0), 6), dtype=torch.float32, device=device)
    elif not torch.is_tensor(seat):
        seat = torch.as_tensor(seat, dtype=torch.float32, device=device)
    legals = batch['legals'] if torch.is_tensor(batch['legals']) else torch.as_tensor(batch['legals'], dtype=torch.float32, device=device)
    offs = batch['legals_offset'] if torch.is_tensor(batch['legals_offset']) else torch.as_tensor(batch['legals_offset'], dtype=torch.long, device=device)
    cnts = batch['legals_count'] if torch.is_tensor(batch['legals_count']) else torch.as_tensor(batch['legals_count'], dtype=torch.long, device=device)
    chosen_idx = batch['chosen_index'] if torch.is_tensor(batch['chosen_index']) else torch.as_tensor(batch['chosen_index'], dtype=torch.long, device=device)
    old_logp = torch.as_tensor(batch['old_logp'], dtype=torch.float32, device=device)

    approx_kl = torch.zeros(obs.size(0), dtype=torch.float32, device=device)
    entropy = torch.zeros(obs.size(0), dtype=torch.float32, device=device)
    with torch.no_grad():
        B = obs.size(0)
        raw_logits = agent.actor(obs, None, seat)  # (B, 80) or (80,)
        if raw_logits.dim() == 1:
            raw_logits = raw_logits.unsqueeze(0)
        max_cnt = int(cnts.max().item()) if B > 0 else 0
        if max_cnt > 0:
            pos = torch.arange(max_cnt, device=device, dtype=torch.long)
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts.unsqueeze(1)
            rel_pos = rel_pos_2d[mask]
            sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            abs_idx_2d = offs.unsqueeze(1) + rel_pos_2d
            abs_idx = abs_idx_2d[mask]
            legals_mb = legals[abs_idx].contiguous()
            legal_scores = (legals_mb * raw_logits[sample_idx_per_legal]).sum(dim=1)
            padded = torch.full((B, max_cnt), -float('inf'), device=device, dtype=legal_scores.dtype)
            padded[mask] = legal_scores
        else:
            padded = torch.full((B, 0), -float('inf'), device=device, dtype=raw_logits.dtype)
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
                       mcts_sims: int = 128, mcts_dets: int = 1, mcts_c_puct: float = 1.0,
                       mcts_root_temp: float = 0.0, mcts_prior_smooth_eps: float = 0.0,
                       mcts_dirichlet_alpha: float = 0.0, mcts_dirichlet_eps: float = 0.0,
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

    use_belief = bool(belief_particles and belief_particles > 0)
    # Mantieni un belief per ciascun giocatore; aggiorna ad ogni mossa
    belief_by_pid = ({p: BeliefState(env.game_state, observer_id=p, num_particles=belief_particles, ess_frac=belief_ess_frac)
                      for p in range(4)} if use_belief else {})
    routing_log = []  # (player_id, source)

    steps = 0
    if final_reward_only:
        # Raccogli per episodi completi: se non specificato, usa multipli di 40 derivati da horizon
        episodes = (max(1, horizon // 40) if episodes is None else max(1, int(episodes)))
        episodes_done = 0
    while True:
        if env.done:
            env.reset()
            # Reinizializza i belief per ogni giocatore a inizio mano
            if use_belief:
                belief_by_pid = {p: BeliefState(env.game_state, observer_id=p, num_particles=belief_particles, ess_frac=belief_ess_frac)
                                 for p in range(4)}

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
            if use_belief:
                bsum_np = belief_by_pid[cp].belief_summary(env.game_state, cp)
                bsum = torch.as_tensor(bsum_np, dtype=torch.float32, device='cpu')
            else:
                bsum = torch.zeros(120, dtype=torch.float32, device='cpu')
            # Condizione dinamica per MCTS: scala con progresso della mano e fattore di training
            # Progresso ~ mosse giocate / 40
            try:
                progress = float(min(1.0, max(0.0, len(env.game_state.get('history', [])) / 40.0)))
            except Exception:
                progress = 0.0
            use_mcts_cur = bool(use_mcts and (mcts_train_factor > 0.0) and (progress >= mcts_progress_start))
            if use_mcts_cur:
                # scala simulazioni in base a (progress - start)/(full - start) e al fattore di training
                denom = max(1e-6, (mcts_progress_full - mcts_progress_start))
                alpha = min(1.0, max(0.0, (progress - mcts_progress_start) / denom))
                sims_scaled = int(max(mcts_min_sims, round(mcts_sims * alpha * mcts_train_factor)))
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
                    with torch.no_grad():
                        logits = agent.actor(o_t, leg_t, None)
                        priors = torch.softmax(logits, dim=0).detach().cpu().numpy()
                    return priors
                # Value: usa il critic (ignora belief per velocità; la rete attuale non usa il canale belief direttamente)
                def value_fn_mcts(_obs):
                    o_cpu = _obs if torch.is_tensor(_obs) else torch.as_tensor(_obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    with torch.no_grad():
                        v = agent.critic(o_t)
                    return float(v.squeeze(0).detach().cpu().item())
                mcts_action = run_is_mcts(env,
                                          policy_fn=policy_fn_mcts,
                                          value_fn=value_fn_mcts,
                                          num_simulations=int(sims_scaled),
                                          c_puct=float(mcts_c_puct),
                                          belief=(belief_by_pid[cp] if use_belief else None),
                                          num_determinization=int(mcts_dets),
                                          root_temperature=float(mcts_root_temp),
                                          prior_smooth_eps=float(mcts_prior_smooth_eps),
                                          robust_child=True,
                                          root_dirichlet_alpha=float(mcts_dirichlet_alpha),
                                          root_dirichlet_eps=float(mcts_dirichlet_eps),
                                          return_stats=False)
                chosen_act = mcts_action if torch.is_tensor(mcts_action) else torch.as_tensor(mcts_action, dtype=torch.float32)
                # trova indice dell'azione scelta tra i legali
                idx_t = None
                for i_a, a in enumerate(legal):
                    try:
                        if torch.is_tensor(a) and torch.is_tensor(chosen_act):
                            if bool(torch.all(a == chosen_act).item()):
                                idx_t = torch.tensor(i_a, dtype=torch.long)
                                break
                        else:
                            if _np.array_equal(_np.asarray(a), _np.asarray(chosen_act.detach().cpu().numpy())):
                                idx_t = torch.tensor(i_a, dtype=torch.long)
                                break
                    except Exception:
                        continue
                if idx_t is None:
                    idx_t = torch.tensor(0, dtype=torch.long)
                next_obs, rew, done, info = env.step(chosen_act)
                routing_log.append((cp, 'mcts'))
            else:
                chosen_act, _logp, idx_t = agent.select_action(obs, legal, seat_vec, bsum)
                next_obs, rew, done, info = env.step(chosen_act)
                routing_log.append((cp, 'main'))

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
        # Aggiorna i belief di TUTTI i giocatori ad ogni mossa osservata
        if use_belief:
            try:
                last_move = env.game_state['history'][-1]
                for p in range(4):
                    belief_by_pid[p].update_with_move(last_move, env.game_state, env.rules,
                                                      ess_threshold=0.5 * belief_by_pid[p].num_particles)
            except Exception:
                pass

        steps += 1
        # Condizioni di uscita: per episodi o per passi
        if final_reward_only:
            if done:
                if episodes_done >= (episodes - 1):
                    break
                else:
                    episodes_done += 1
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
            next_val_t = agent.critic(next_obs_t, s_all, b_all)
            next_val_t = torch.where(done_mask_bool, torch.zeros_like(next_val_t), next_val_t)

    # Compute V(obs) in batch su GPU e GAE
    T = len(rew_list)
    rew_t = torch.as_tensor(rew_list, dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    done_mask = torch.as_tensor([0.0 if not d else 1.0 for d in done_list], dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    if T > 0:
        with torch.no_grad():
            o_all = torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o, dtype=torch.float32) for o in obs_list], dim=0).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            s_all = torch.stack(seat_team_list, dim=0).pin_memory().to(device=device, non_blocking=True)
            b_all = torch.stack(belief_sum_list, dim=0).pin_memory().to(device=device, non_blocking=True)
            val_t = agent.critic(o_all, s_all, b_all)
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
            raw_logits = agent.actor(obs_t, None, seat_team_t)
            if raw_logits.dim() == 1:
                raw_logits = raw_logits.unsqueeze(0)
            B = obs_t.size(0)
            max_cnt = int(legals_count_t.max().item()) if B > 0 else 0
            if max_cnt > 0:
                pos = torch.arange(max_cnt, device=device, dtype=torch.long)
                rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
                mask = rel_pos_2d < legals_count_t.unsqueeze(1)
                sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
                abs_idx = (legals_offset_t.unsqueeze(1) + rel_pos_2d)[mask]
                legals_mb = legals_t[abs_idx].contiguous()
                legal_scores = (legals_mb * raw_logits[sample_idx_per_legal]).sum(dim=1)
                padded = torch.full((B, max_cnt), -float('inf'), device=device, dtype=legal_scores.dtype)
                padded[mask] = legal_scores
                logp_group = torch.log_softmax(padded, dim=1)
                chosen_clamped = torch.minimum(chosen_index_t, (legals_count_t - 1).clamp_min(0))
                old_logp_t = logp_group[torch.arange(B, device=device), chosen_clamped]
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
        'routing_log': routing_log,
    }
    return batch


def train_ppo(num_iterations: int = 1000, horizon: int = 256, save_every: int = 200, ckpt_path: str = 'checkpoints/ppo_ac.pth', use_compact_obs: bool = True, k_history: int = 39, seed: int = 0,
              entropy_schedule_type: str = 'linear', eval_every: int = 0, eval_games: int = 10, belief_particles: int = 512, belief_ess_frac: float = 0.5,
              mcts_in_eval: bool = False, mcts_sims: int = 128, mcts_dets: int = 4, mcts_c_puct: float = 1.0, mcts_root_temp: float = 0.0,
              mcts_prior_smooth_eps: float = 0.0, mcts_dirichlet_alpha: float = 0.25, mcts_dirichlet_eps: float = 0.25,
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

    writer = SummaryWriter(log_dir='runs/ppo_ac') if TB_AVAILABLE else None
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
        # Strategia MCTS: warmup senza MCTS per le prime iterazioni, poi scala con il progresso mano
        mcts_train_factor = 0.0 if it < 500 else 1.0  # warmup 50 iterazioni
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
                                   train_both_teams=False)
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
    args = parser.parse_args()
    train_ppo(num_iterations=args.iters, horizon=args.horizon, save_every=args.save_every, ckpt_path=args.ckpt,
              use_compact_obs=args.compact, k_history=args.k_history, seed=args.seed,
              entropy_schedule_type=args.entropy_schedule, eval_every=args.eval_every, eval_games=args.eval_games,
              belief_particles=args.belief_particles, belief_ess_frac=args.belief_ess_frac,
              mcts_in_eval=args.mcts_eval, mcts_sims=args.mcts_sims, mcts_dets=args.mcts_dets, mcts_c_puct=args.mcts_c_puct,
              mcts_root_temp=args.mcts_root_temp, mcts_prior_smooth_eps=args.mcts_prior_smooth_eps,
              mcts_dirichlet_alpha=args.mcts_dirichlet_alpha, mcts_dirichlet_eps=args.mcts_dirichlet_eps)


