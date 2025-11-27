"""
Confronta la differenza di score (team A - team B) tra raccolta seriale e parallela,
allineando i game index: per ogni game calcola diff_serial[i] - diff_parallel[i].

Esegue 2000 game per ciascun collector (horizon=20, episodes=1, main seats 0/2 con opponent congelato).
Salva il grafico in reward_trend_diff.png.
"""

import os
import sys
import random
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

# Ensure project root on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory, collect_trajectory_parallel


def _setup_env_defaults() -> None:
    os.environ.setdefault('SCOPONE_DISABLE_SAVE', '1')
    os.environ.setdefault('SCOPONE_DISABLE_EVAL', '1')
    os.environ.setdefault('SCOPONE_LEAGUE_REFRESH', '0')
    os.environ.setdefault('SCOPONE_MINIBATCH_ALIGN', '0')
    os.environ.setdefault('SCOPONE_MINIBATCH', '0')
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    os.environ.setdefault('TQDM_DISABLE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')


def _serial_run(num_games: int, seed: int) -> List[float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    opponent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0]).actor  # frozen bootstrap

    diffs: List[float] = []
    for _ in trange(num_games, desc="Serial", leave=False):
        batch = collect_trajectory(
            env,
            agent,
            horizon=20,
            use_mcts=False,
            mcts_sims=0,
            mcts_dets=0,
            train_both_teams=False,
            partner_actor=agent.actor,  # seats 0/2
            opponent_actor=opponent,    # seats 1/3 frozen
            alternate_main_seats=False,
            episodes=1,
            seed=seed,
        )
        diff = env.rewards[0] - env.rewards[1]
        diffs.append(float(diff))
        if batch['obs'].numel() > 0:
            agent.update(batch, epochs=1, minibatch_size=20)
    return diffs


def _parallel_run(num_games: int, seed: int, num_envs: int = 32) -> List[float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    obs_dim = ScoponeEnvMA(k_history=4).observation_space.shape[0]
    agent = ActionConditionedPPO(obs_dim=obs_dim)
    opponent = ActionConditionedPPO(obs_dim=obs_dim).actor  # frozen bootstrap

    # Raccogli tutte le partite in una sola chiamata per evitare il respawn continuo dei worker
    batch = collect_trajectory_parallel(
        agent,
        num_envs=num_envs,
        episodes_total_hint=num_games,
        k_history=4,
        gamma=1.0,
        lam=1.0,
        use_mcts=False,
        train_both_teams=False,
        main_seats=[0, 2],
        mcts_sims=0,
        mcts_dets=0,
        mcts_c_puct=1.0,
        mcts_root_temp=0.0,
        mcts_prior_smooth_eps=0.0,
        mcts_dirichlet_alpha=0.25,
        mcts_dirichlet_eps=0.0,
        mcts_min_sims=0,
        mcts_train_factor=1.0,
        seed=seed,
        show_progress_env=False,
        tqdm_base_pos=0,
        frozen_actor=opponent,
        frozen_non_main=True,
        alternate_main_seats=False,
    )

    if 'logp' not in batch:
        raise RuntimeError("collect_trajectory_parallel batch missing logp; check trainer output")
    logp_full = batch['logp']

    diffs: List[float] = []
    done = batch['done']
    leg_off = batch['legals_offset']
    leg_cnt = batch['legals_count']
    legals = batch['legals']
    start = 0
    for end_idx, done_flag in enumerate(done):
        if not bool(done_flag.item()):
            continue
        end = end_idx + 1
        # calcola diff per la singola partita usando reward aggregati per team
        seats = batch['seat_team'][start:end]
        rew_ep = batch['rew'][start:end]
        team0_mask = seats[:, 4] > 0.5
        team1_mask = seats[:, 5] > 0.5
        if team0_mask.any() and team1_mask.any():
            diff = float(rew_ep[team0_mask].sum().item() - rew_ep[team1_mask].sum().item())
        else:
            diff = 0.0
        diffs.append(diff)
        # costruisci sotto-batch per aggiornare dopo ogni partita
        leg_start = int(leg_off[start].item())
        leg_end = int(leg_off[end - 1].item() + leg_cnt[end - 1].item())
        ep_batch = {
            'obs': batch['obs'][start:end],
            'act': batch['act'][start:end],
            'ret': batch['ret'][start:end],
            'adv': batch['adv'][start:end],
            'rew': rew_ep,
            'done': batch['done'][start:end],
            'seat_team': seats,
            'legals': legals[leg_start:leg_end],
            'legals_offset': leg_off[start:end] - leg_off[start],
            'legals_count': leg_cnt[start:end],
            'chosen_index': batch['chosen_index'][start:end],
        }
        if 'mcts_policy' in batch:
            ep_batch['mcts_policy'] = batch['mcts_policy'][leg_start:leg_end]
        if 'mcts_weight' in batch:
            ep_batch['mcts_weight'] = batch['mcts_weight'][start:end]
        if 'others_hands' in batch:
            ep_batch['others_hands'] = batch['others_hands'][start:end]
        ep_batch['logp'] = logp_full[start:end]
        if ep_batch['obs'].numel() > 0:
            agent.update(ep_batch, epochs=1, minibatch_size=20)
        start = end
    return diffs


def main(num_games: int = 2000, seed: int = 999, num_envs_parallel: int = 32) -> None:
    _setup_env_defaults()
    serial_diffs = _serial_run(num_games, seed)
    parallel_diffs = _parallel_run(num_games, seed, num_envs=num_envs_parallel)

    n = min(len(serial_diffs), len(parallel_diffs))
    serial_diffs = serial_diffs[:n]
    parallel_diffs = parallel_diffs[:n]
    # Cumulative averages per game index
    serial_avg = [sum(serial_diffs[:i+1]) / float(i+1) for i in range(n)]
    parallel_avg = [sum(parallel_diffs[:i+1]) / float(i+1) for i in range(n)]
    delta_avg = [sa - pa for sa, pa in zip(serial_avg, parallel_avg)]

    plt.figure(figsize=(12, 6))
    plt.plot(serial_avg, label='Serial avg diff (A-B)', alpha=0.7)
    plt.plot(parallel_avg, label='Parallel avg diff (A-B)', alpha=0.7)
    plt.plot(delta_avg, label='Serial - Parallel (avg)', alpha=0.9)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Game')
    plt.ylabel('Avg score diff')
    plt.title('Serial vs Parallel cumulative average score difference')
    plt.legend()
    plt.tight_layout()
    out_path = 'reward_trend_diff.png'
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
