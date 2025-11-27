"""
Plot the score-difference trend using the parallel collector (num_envs=1, frozen opponent).
This mirrors the regression test that checks improvement over repeated games in parallel mode.

Outputs: reward_trend_parallel.png in the current directory.
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory_parallel


def main():
    # Base env settings (CPU only, no eval/save)
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

    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    obs_dim = ScoponeEnvMA(k_history=4).observation_space.shape[0]
    agent = ActionConditionedPPO(obs_dim=obs_dim)
    opponent = ActionConditionedPPO(obs_dim=obs_dim).actor  # frozen bootstrap opponent

    num_games = 2000  # keep runtime reasonable
    diffs = []
    for _ in range(num_games):
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
        seats = batch['seat_team']
        ret = batch['ret']
        team0_mask = seats[:, 4] > 0.5
        team1_mask = seats[:, 5] > 0.5
        if team0_mask.any() and team1_mask.any():
            diff = float(ret[team0_mask].mean().item() - ret[team1_mask].mean().item())
        else:
            diff = 0.0
        diffs.append(diff)
        if batch['obs'].numel() > 0:
            agent.update(batch, epochs=1, minibatch_size=20)

    plt.figure(figsize=(10, 4))
    plt.plot(diffs, label='score diff (A-B) parallel')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Game')
    plt.ylabel('Score diff')
    plt.title('Team A score difference over repeated games (parallel collector)')
    plt.legend()
    plt.tight_layout()
    out_path = 'reward_trend_parallel.png'
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
