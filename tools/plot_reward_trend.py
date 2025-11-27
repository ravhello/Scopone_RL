"""
Utility: plot trend of score difference (team A - team B) over repeated games.
It reuses the same setup of the regression test:
- Team A seats: 0/2 (agent trained and updated each game)
- Team B seats: 1/3 (frozen random bootstrap)
- Horizon: 20, episodes=1 (20 transitions per update)

Outputs: reward_trend.png in the current directory.
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
from trainers.train_ppo import collect_trajectory


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

    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)

    env = ScoponeEnvMA(k_history=4)
    agentA = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    opponent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0]).actor  # frozen bootstrap

    num_games = 2000  # keep reasonable runtime
    diffs = []
    for _ in range(num_games):
        batch = collect_trajectory(
            env,
            agentA,
            horizon=20,
            use_mcts=False,
            mcts_sims=0,
            mcts_dets=0,
            train_both_teams=False,
            partner_actor=agentA.actor,  # seats 0/2
            opponent_actor=opponent,      # seats 1/3 frozen
            alternate_main_seats=False,
            episodes=1,
        )
        diff = env.rewards[0] - env.rewards[1]
        diffs.append(float(diff))
        if batch['obs'].numel() > 0:
            agentA.update(batch, epochs=1, minibatch_size=20)

    plt.figure(figsize=(10, 4))
    plt.plot(diffs, label='score diff (A-B)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Game')
    plt.ylabel('Score diff')
    plt.title('Team A score difference over repeated games')
    plt.legend()
    plt.tight_layout()
    out_path = 'reward_trend.png'
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
