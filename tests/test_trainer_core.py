import os

import torch

from algorithms.ppo_ac import ActionConditionedPPO
from environment import ScoponeEnvMA
from trainers.train_ppo import collect_trajectory, train_ppo


def test_collect_trajectory_without_mcts_fallback_runs():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # MCTS disattivato: esercita solo la policy fallback
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        mcts_c_puct=1.0,
        mcts_min_sims=0,
        train_both_teams=False,
    )
    assert 'obs' in batch and batch['obs'].shape[0] > 0
    assert 'chosen_index' in batch and batch['chosen_index'].shape[0] == batch['obs'].shape[0]


def test_collect_trajectory_marks_done_on_episode_boundaries():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    os.environ.setdefault('SCOPONE_MINIBATCH', '20')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=1,
        train_both_teams=False,
        final_reward_only=True,
    )
    done = batch['done']
    assert done.dtype == torch.bool  # type: ignore[attr-defined]
    # horizon=40 with train_both_teams=False collects two 20-step episodes
    assert int(done.sum().item()) == 2


def test_train_ppo_single_env_one_iter_smoke():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    # Verify one iteration completes in single-env mode (exercise non-parallel path)
    ran = {'called': False}

    def on_end(_):
        ran['called'] = True

    train_ppo(
        num_iterations=1,
        horizon=40,
        k_history=4,
        num_envs=1,
        mcts_sims=0,
        mcts_dets=0,
        on_iter_end=on_end,
    )
    assert ran['called'] is True


