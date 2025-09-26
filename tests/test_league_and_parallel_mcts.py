import os
import tempfile
from selfplay.league import League
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory_parallel


def test_league_register_sample_and_update_elo(tmp_path=None):
    if tmp_path is None:
        tmp_dir = tempfile.mkdtemp()
    else:
        tmp_dir = str(tmp_path)
    base = os.path.join(tmp_dir, 'league')
    league = League(base_dir=base)
    # make two tiny checkpoints
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    ck1 = os.path.join(tmp_dir, 'a1.pth')
    ck2 = os.path.join(tmp_dir, 'a2.pth')
    agent.save(ck1)
    agent.save(ck2)
    league.register(ck1)
    league.register(ck2)
    p, o = league.sample_pair()
    assert p in league.history and o in league.history
    league.update_elo_from_diff(ck1, ck2, avg_point_diff_a=2.0)
    assert ck1 in league.elo and ck2 in league.elo


def test_parallel_mcts_high_dets_minimal_runs():
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory_parallel(
        agent,
        num_envs=2,
        episodes_total_hint=1,
        k_history=4,
        use_mcts=True,
        mcts_sims=1,
        mcts_dets=1,
        mcts_c_puct=1.0,
        mcts_root_temp=0.0,
    )
    assert 'obs' in batch and batch['obs'].shape[0] > 0
    assert 'mcts_weight' in batch and batch['mcts_weight'].shape[0] == batch['obs'].shape[0]


