from trainers.train_ppo import collect_trajectory_parallel
from algorithms.ppo_ac import ActionConditionedPPO
from environment import ScoponeEnvMA


def test_collect_trajectory_parallel_minimal_runs():
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory_parallel(
        agent,
        num_envs=2,
        episodes_total_hint=1,
        k_history=4,
        use_mcts=True,
        mcts_sims=8,
        mcts_dets=1,
        mcts_c_puct=1.0,
        mcts_root_temp=0.0,
        mcts_prior_smooth_eps=0.0,
        mcts_dirichlet_alpha=0.25,
        mcts_dirichlet_eps=0.25,
    )
    assert 'obs' in batch
    assert batch['obs'].shape[0] > 0
    assert 'chosen_index' in batch
    assert batch['chosen_index'].shape[0] == batch['obs'].shape[0]



