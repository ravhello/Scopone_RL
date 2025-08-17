from tests.torch_np import np
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory


def test_collect_trajectory_with_belief_summary():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory(env, agent, horizon=8)
    assert 'belief_summary' in batch and batch['belief_summary'].shape[0] == batch['obs'].shape[0]


def test_next_value_ctde_inputs_present():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory(env, agent, horizon=8)
    assert 'seat_team' in batch and batch['seat_team'].shape[0] == batch['obs'].shape[0]
    assert batch['belief_summary'].shape[0] == batch['obs'].shape[0]
    # GAE calcolato con next_vals CTDE coerenti -> dimensioni combaciano
    assert batch['ret'].shape[0] == batch['obs'].shape[0]
    assert batch['adv'].shape[0] == batch['obs'].shape[0]
    # semantica base: se done[t] allora next_val[t]==0 (implicito nel calcolo GAE)
    assert 'done' in batch and 'rew' in batch


def test_partner_opponent_routing_basic():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # frozen actors (random weights but eval mode)
    batch = collect_trajectory(env, agent, horizon=8)
    assert batch['obs'].shape[0] > 0
    # verify routing log contains labels and player ids
    assert 'routing_log' in batch
    assert len(batch['routing_log']) >= 1
    pids = [pid for pid, _ in batch['routing_log']]
    sources = [src for _, src in batch['routing_log']]
    assert all(src in ('main', 'partner', 'opponent') for src in sources)
    assert all(isinstance(pid, int) for pid in pids)
    # se main_seats=[0,2], i seat 1/3 devono comparire nel routing come opponent o partner
    if any(pid in [1,3] for pid in pids):
        assert any(src in ('partner','opponent') for (pid, src) in batch['routing_log'] if pid in [1,3])

