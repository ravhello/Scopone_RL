import torch
from tqdm import tqdm
from typing import Dict, List
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
def _compute_per_seat_diagnostics(agent: ActionConditionedPPO, batch: Dict) -> Dict[str, float]:
    """Calcola approx_kl ed entropia per gruppo seat 0/2 vs 1/3 sul batch corrente."""
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
        for i in range(obs.size(0)):
            start = int(offs[i])
            end = start + int(cnts[i])
            legal_i = legals[start:end]
            logits_i = agent.actor(obs[i], legal_i, seat[i])
            logp_i = torch.log_softmax(logits_i, dim=0)
            probs_i = torch.softmax(logits_i, dim=0)
            entropy[i] = -(probs_i * logp_i).sum()
            approx_kl[i] = (old_logp[i] - logp_i[int(chosen_idx[i])]).abs()
    # clip fraction per-sample sul chosen
    ratio = torch.exp(-approx_kl)  # solo se old_logp - new_logp = KL approx, ma usiamo direttamente logp chosen
    # ricalcoliamo ratio in modo corretto: new_logp_chosen - old_logp
    new_logp_chosen = torch.zeros_like(old_logp)
    with torch.no_grad():
        for i in range(obs.size(0)):
            start = int(offs[i])
            end = start + int(cnts[i])
            legal_i = legals[start:end]
            logits_i = agent.actor(obs[i], legal_i, seat[i])
            logp_i = torch.log_softmax(logits_i, dim=0)
            new_logp_chosen[i] = logp_i[int(chosen_idx[i])]
    ratio = torch.exp(new_logp_chosen - old_logp)
    clip_low = 1.0 - agent.clip_ratio
    clip_high = 1.0 + agent.clip_ratio
    clipped_mask = (ratio < clip_low) | (ratio > clip_high)

    seats = torch.argmax(seat[:, :4], dim=1)
    mask_02 = (seats == 0) | (seats == 2)
    mask_13 = (seats == 1) | (seats == 3)
    out = {}
    if mask_02.any():
        out['by_seat/kl_02'] = float(approx_kl[mask_02].mean().item())
        out['by_seat/entropy_02'] = float(entropy[mask_02].mean().item())
        out['by_seat/clip_frac_02'] = float(clipped_mask[mask_02].float().mean().item())
    if mask_13.any():
        out['by_seat/kl_13'] = float(approx_kl[mask_13].mean().item())
        out['by_seat/entropy_13'] = float(entropy[mask_13].mean().item())
        out['by_seat/clip_frac_13'] = float(clipped_mask[mask_13].float().mean().item())
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
                       belief_particles: int = 256, belief_ess_frac: float = 0.5) -> Dict:
    obs_list, next_obs_list = [], []
    act_list, logp_list = [], []
    rew_list, done_list = [], []
    val_list, next_val_list = [], []
    legals_list, legals_offset, legals_count, chosen_index = [], [], [], []
    seat_team_list = []
    belief_sum_list = []

    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=belief_particles, ess_frac=belief_ess_frac)
    routing_log = []  # (player_id, source)

    steps = 0
    while steps < horizon:
        if env.done:
            env.reset()

        obs = env._get_observation(env.current_player)
        legal = env.get_valid_actions()
        if not legal:
            break

        cp = env.current_player
        seat_vec = torch.zeros(6, dtype=torch.float32, device=device)
        seat_vec[cp] = 1.0
        seat_vec[4] = 1.0 if cp in [0, 2] else 0.0
        seat_vec[5] = 1.0 if cp in [1, 3] else 0.0
        bsum_np = belief.belief_summary(env.game_state, cp)
        bsum = torch.as_tensor(bsum_np, dtype=torch.float32, device=device)

        # Selezione azione in base ai main_seats (default: [0,2])
        is_main = (main_seats is None and cp in [0, 2]) or (main_seats is not None and cp in main_seats)
        if is_main:
            with torch.no_grad():
                o_t = obs.clone().detach().to(device=device, dtype=torch.float32).unsqueeze(0) if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                s_t = seat_vec.unsqueeze(0)
                b_t = bsum.unsqueeze(0)
                v = float(agent.critic(o_t, s_t, b_t))
            act, logp, idx = agent.select_action(obs, legal, seat_vec, bsum)
            next_obs, rew, done, info = env.step(act)
            routing_log.append((cp, 'main'))

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            act_list.append(act)
            logp_list.append(logp)
            rew_list.append(rew)
            done_list.append(done)
            val_list.append(v)
            seat_team_list.append(seat_vec)
            belief_sum_list.append(bsum)
            legals_offset.append(len(legals_list))
            legals_count.append(len(legal))
            chosen_index.append(idx)
            legals_list.extend(legal)
        else:
            # partner congelato sui seat del compagno; opponent sugli avversari
            is_partner_seat = (cp in [0, 2] and (main_seats == [1, 3])) or (cp in [1, 3] and (main_seats == [0, 2]))
            frozen = partner_actor if (is_partner_seat and partner_actor is not None) else opponent_actor
            if frozen is not None:
                with torch.no_grad():
                    o_t = obs.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32, device=device)
                    if len(legal) > 0 and torch.is_tensor(legal[0]):
                        leg_t = torch.stack(legal).to(device=device, dtype=torch.float32)
                    else:
                        leg_t = torch.stack([
                            x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device=device)
                        for x in legal], dim=0)
                    logits = frozen(o_t.unsqueeze(0), leg_t)
                    idx = int(torch.argmax(logits).item())
                act = legal[idx]
            else:
                idx = int(torch.randint(len(legal), (1,), device=device).item())
                act = legal[idx]
            next_obs, rew, done, info = env.step(act)
            routing_log.append((cp, 'partner' if is_partner_seat else 'opponent'))

        try:
            last_move = env.game_state['history'][-1]
            belief.update_with_move(last_move, env.game_state, env.rules, ess_threshold=0.5 * belief.num_particles)
        except Exception:
            pass

        steps += 1

    # CTDE: valuta V(next) con seat_team e belief_summary
    next_val_list = []
    if len(next_obs_list) > 0:
        with torch.no_grad():
            for i in range(len(next_obs_list)):
                if done_list[i]:
                    next_val_list.append(0.0)
                else:
                    no = next_obs_list[i]
                    o_t = no.clone().detach().to(device=device, dtype=torch.float32).unsqueeze(0) if torch.is_tensor(no) else torch.as_tensor(no, dtype=torch.float32, device=device).unsqueeze(0)
                    s_t = seat_team_list[i].unsqueeze(0)
                    b_t = belief_sum_list[i].unsqueeze(0)
                    nv = float(agent.critic(o_t, s_t, b_t))
                    next_val_list.append(float(nv))

    # Compute GAE on GPU
    T = len(rew_list)
    rew_t = torch.as_tensor(rew_list, dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    done_mask = torch.as_tensor([0.0 if not d else 1.0 for d in done_list], dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    val_t = torch.as_tensor(val_list, dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    nval_t = torch.as_tensor(next_val_list, dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    adv_vec = torch.zeros_like(rew_t)
    gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    for t in reversed(range(T)):
        delta = rew_t[t] + gamma * nval_t[t] - val_t[t]
        gae = delta + gamma * lam * (1.0 - done_mask[t]) * gae
        adv_vec[t] = gae
    ret_vec = adv_vec + val_t

    # Keep batch entirely as torch tensors on CUDA
    obs_t = torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o, dtype=torch.float32, device=device) for o in obs_list], dim=0) if len(obs_list)>0 else torch.zeros((0, env.observation_space.shape[0]), dtype=torch.float32, device=device)
    act_t = torch.stack([a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32, device=device) for a in act_list], dim=0) if len(act_list)>0 else torch.zeros((0, 80), dtype=torch.float32, device=device)
    legals_t = torch.stack([l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.float32, device=device) for l in legals_list], dim=0) if legals_list else torch.zeros((0, 80), dtype=torch.float32, device=device)
    seat_team_t = torch.stack(seat_team_list, dim=0) if len(seat_team_list)>0 else torch.zeros((0,6), dtype=torch.float32, device=device)
    belief_sum_t = torch.stack(belief_sum_list, dim=0) if len(belief_sum_list)>0 else torch.zeros((0,120), dtype=torch.float32, device=device)
    legals_offset_t = torch.as_tensor(legals_offset, dtype=torch.long, device=device) if len(legals_offset)>0 else torch.zeros((0,), dtype=torch.long, device=device)
    legals_count_t = torch.as_tensor(legals_count, dtype=torch.long, device=device) if len(legals_count)>0 else torch.zeros((0,), dtype=torch.long, device=device)
    chosen_index_t = torch.as_tensor(chosen_index, dtype=torch.long, device=device) if len(chosen_index)>0 else torch.zeros((0,), dtype=torch.long, device=device)
    ret_t = ret_vec
    adv_t = adv_vec
    rew_t = rew_t
    done_t = torch.as_tensor(done_list, dtype=torch.bool, device=device) if len(done_list)>0 else torch.zeros((0,), dtype=torch.bool, device=device)

    batch = {
        'obs': obs_t,
        'act': act_t,
        'old_logp': torch.as_tensor(logp_list, dtype=torch.float32, device=device) if len(logp_list)>0 else torch.zeros((0,), dtype=torch.float32, device=device),
        'ret': ret_t,
        'adv': adv_t,
        'rew': rew_t,
        'done': done_t,
        'seat_team': seat_team_t,
        'belief_summary': belief_sum_t,
        'legals': legals_t,
        'legals_offset': legals_offset_t,
        'legals_count': legals_count_t,
        'chosen_index': chosen_index_t,
        'routing_log': routing_log,
    }
    return batch


def train_ppo(num_iterations: int = 1000, horizon: int = 256, save_every: int = 200, ckpt_path: str = 'checkpoints/ppo_ac.pth', use_compact_obs: bool = True, k_history: int = 12, seed: int = 0,
              entropy_schedule_type: str = 'linear', eval_every: int = 0, eval_games: int = 10, belief_particles: int = 256, belief_ess_frac: float = 0.5,
              mcts_in_eval: bool = False, mcts_sims: int = 128, mcts_dets: int = 4, mcts_c_puct: float = 1.0, mcts_root_temp: float = 0.0,
              mcts_prior_smooth_eps: float = 0.0, mcts_dirichlet_alpha: float = 0.25, mcts_dirichlet_eps: float = 0.25):
    set_global_seeds(seed)
    env = ScoponeEnvMA(use_compact_obs=use_compact_obs, k_history=k_history)
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
            if p_ckpt and os.path.isfile(p_ckpt):
                partner_actor = _load_frozen_actor(p_ckpt, obs_dim)
            if o_ckpt and os.path.isfile(o_ckpt):
                opponent_actor = _load_frozen_actor(o_ckpt, obs_dim)
        except Exception:
            partner_actor = None
        main_seats = even_main_seats if (it % 2 == 0) else odd_main_seats
        batch = collect_trajectory(env, agent, horizon=horizon, partner_actor=partner_actor, opponent_actor=opponent_actor, main_seats=main_seats,
                                   belief_particles=belief_particles, belief_ess_frac=belief_ess_frac)
        if len(batch['obs']) == 0:
            continue
        # normalizza vantaggi
        adv = batch['adv']
        if adv.std() > 1e-8:
            batch['adv'] = (adv - adv.mean()) / (adv.std() + 1e-8)
        info = agent.update(batch, epochs=4, minibatch_size=256)
        dt = time.time() - t0

        # proxy per best: media return del batch
        avg_return = float(np.mean(batch['ret'])) if len(batch['ret']) else 0.0
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
            print({k: round(v, 4) for k, v in info.items()})
        if writer is not None:
            for k, v in info.items():
                writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/episode_time_s', dt, it)
            writer.add_scalar('train/avg_return', avg_return, it)
            writer.add_text('train/main_seats', str(main_seats), it)
            # Log by seat group (0/2 vs 1/3) using batch returns as proxy
            try:
                seats = np.argmax(batch['seat_team'][:, :4], axis=1)
                ret_arr = batch['ret']
                mask_02 = np.isin(seats, [0, 2])
                mask_13 = np.isin(seats, [1, 3])
                if mask_02.any():
                    writer.add_scalar('by_seat/ret_02', float(ret_arr[mask_02].mean()), it)
                if mask_13.any():
                    writer.add_scalar('by_seat/ret_13', float(ret_arr[mask_13].mean()), it)
                # diagnostica per-seat: KL/entropy
                diag = _compute_per_seat_diagnostics(agent, batch)
                for k, v in diag.items():
                    writer.add_scalar(k, v, it)
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
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO Action-Conditioned for Scopone')
    parser.add_argument('--iters', type=int, default=2000, help='Number of PPO iterations')
    parser.add_argument('--horizon', type=int, default=256, help='Rollout horizon (steps) per iteration')
    parser.add_argument('--save-every', type=int, default=200, help='Save checkpoint every N iterations')
    parser.add_argument('--ckpt', type=str, default='checkpoints/ppo_ac.pth', help='Checkpoint path')
    parser.add_argument('--compact', action='store_true', help='Use compact observation (recommended)')
    parser.add_argument('--k-history', type=int, default=12, help='Number of recent moves in compact history')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--entropy-schedule', type=str, default='linear', choices=['linear','cosine'], help='Entropy schedule type')
    parser.add_argument('--eval-every', type=int, default=0, help='Run mini-eval every N iters (0=off)')
    parser.add_argument('--eval-games', type=int, default=10, help='Games per mini-eval')
    parser.add_argument('--belief-particles', type=int, default=256, help='Belief particles for trainer')
    parser.add_argument('--belief-ess-frac', type=float, default=0.5, help='Belief ESS fraction for trainer')
    parser.add_argument('--mcts-eval', action='store_true', help='Use MCTS in mini-eval')
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


