"""
Confronta la differenza di score (team A - team B) tra raccolta seriale e parallela,
allineando i game index: per ogni game calcola diff_serial[i] - diff_parallel[i].

Esegue 400 game per ciascun collector (horizon=20, episodes=1, main seats 0/2 con opponent congelato).
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
        # Usa i team_rewards episodici per avere la stessa sorgente del parallelo
        tr = batch.get('episode_team_rewards', None)
        if tr is not None and hasattr(tr, 'numel') and tr.numel() >= 2:
            diff = float(tr[0][0].item() - tr[0][1].item())
        else:
            diff = env.rewards[0] - env.rewards[1]
        diffs.append(diff)
        if batch['obs'].numel() > 0:
            agent.update(batch, epochs=1, minibatch_size=20)
    return diffs


def _parallel_run(num_games: int, seed: int) -> List[float]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    obs_dim = ScoponeEnvMA(k_history=4).observation_space.shape[0]
    agent = ActionConditionedPPO(obs_dim=obs_dim)
    opponent = ActionConditionedPPO(obs_dim=obs_dim).actor  # frozen bootstrap

    diffs: List[float] = []
    for _ in trange(num_games, desc="Parallel", leave=False):
        batch = collect_trajectory_parallel(
            agent,
            num_envs=1,
            episodes_total_hint=1,
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
        team_rewards = batch.get('episode_team_rewards', None)
        if team_rewards is not None and team_rewards.numel() >= 2:
            diff = float(team_rewards[0][0].item() - team_rewards[0][1].item())
        else:
            seats = batch['seat_team']
            rew_ep = batch['rew']
            team0_mask = seats[:, 4] > 0.5
            team1_mask = seats[:, 5] > 0.5
            diff = float(rew_ep[team0_mask].sum().item() - rew_ep[team1_mask].sum().item())
        diffs.append(diff)
        # aggiorna per episodio con minibatch costante per coerenza con il seriale
        if batch['obs'].numel() > 0:
            agent.update(batch, epochs=1, minibatch_size=20)
    return diffs


def main(num_games: int = 400, seed: int = 999, num_envs_parallel: int = 32) -> None:
    _setup_env_defaults()
    serial_diffs = _serial_run(num_games, seed)
    # num_envs_parallel non usato: _parallel_run raccoglie 1 env per allinearsi ai test
    parallel_diffs = _parallel_run(num_games, seed)

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
