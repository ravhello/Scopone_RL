from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory, train_ppo


def test_collect_trajectory_without_mcts_fallback_runs():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # Force MCTS off via factor=0.0 to exercise fallback action selection
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=True,
        mcts_sims=8,
        mcts_dets=1,
        mcts_c_puct=1.0,
        mcts_train_factor=0.0,
        mcts_min_sims=0,
        train_both_teams=False,
    )
    assert 'obs' in batch and batch['obs'].shape[0] > 0
    assert 'chosen_index' in batch and batch['chosen_index'].shape[0] == batch['obs'].shape[0]


def test_train_ppo_single_env_one_iter_smoke():
    # Verify one iteration completes in single-env mode (exercise non-parallel path)
    ran = {'called': False}

    def on_end(_):
        ran['called'] = True

    train_ppo(
        num_iterations=1,
        horizon=40,
        use_compact_obs=True,
        k_history=4,
        num_envs=1,
        mcts_sims=8,
        mcts_dets=1,
        on_iter_end=on_end,
    )
    assert ran['called'] is True


