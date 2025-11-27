import os
import pytest
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory
from conftest import collect_batch


@pytest.mark.parametrize('collector_kind', ['serial', 'parallel'])
def test_collect_trajectory_with_belief_summary(collector_kind):
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=40)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    if collector_kind == 'serial':
        batch = collect_trajectory(env, agent, horizon=40, use_mcts=False, mcts_sims=0, mcts_dets=0)
    else:
        batch = collect_batch('parallel', agent, env=env, episodes_total_hint=1, k_history=40, use_mcts=False, mcts_sims=0, mcts_dets=0, train_both_teams=False, main_seats=[0, 2], alternate_main_seats=False)
    assert 'belief_summary' in batch and batch['belief_summary'].shape[0] == batch['obs'].shape[0]


@pytest.mark.parametrize('collector_kind', ['serial', 'parallel'])
def test_next_value_ctde_inputs_present(collector_kind):
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=40)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    if collector_kind == 'serial':
        batch = collect_trajectory(env, agent, horizon=40, use_mcts=False, mcts_sims=0, mcts_dets=0)
    else:
        batch = collect_batch('parallel', agent, env=env, episodes_total_hint=1, k_history=40, use_mcts=False, mcts_sims=0, mcts_dets=0, train_both_teams=False, main_seats=[0, 2], alternate_main_seats=False)
    assert 'seat_team' in batch and batch['seat_team'].shape[0] == batch['obs'].shape[0]
    assert batch['belief_summary'].shape[0] == batch['obs'].shape[0]
    # GAE calcolato con next_vals CTDE coerenti -> dimensioni combaciano
    assert batch['ret'].shape[0] == batch['obs'].shape[0]
    assert batch['adv'].shape[0] == batch['obs'].shape[0]
    # semantica base: se done[t] allora next_val[t]==0 (implicito nel calcolo GAE)
    assert 'done' in batch and 'rew' in batch


@pytest.mark.parametrize('collector_kind', ['serial', 'parallel'])
def test_partner_opponent_routing_basic(collector_kind):
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=40)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # frozen actors (random weights but eval mode)
    if collector_kind == 'serial':
        batch = collect_trajectory(env, agent, horizon=40, use_mcts=False, mcts_sims=0, mcts_dets=0)
    else:
        batch = collect_batch('parallel', agent, env=env, episodes_total_hint=1, k_history=40, use_mcts=False, mcts_sims=0, mcts_dets=0, train_both_teams=False, main_seats=[0, 2], alternate_main_seats=False)
    assert batch['obs'].shape[0] > 0
    # verify routing log contains labels and player ids
    assert 'routing_log' in batch
    assert len(batch['routing_log']) >= 1
    pids = [pid for pid, _ in batch['routing_log']]
    sources = [src for _, src in batch['routing_log']]
    # new core may label MCTS-driven steps explicitly as 'mcts'
    assert all(src in ('main', 'partner', 'opponent', 'mcts') for src in sources)
    assert all(isinstance(pid, int) for pid in pids)
    # se main_seats=[0,2], i seat 1/3 devono comparire nel routing come opponent o partner
    if any(pid in [1,3] for pid in pids):
        assert any(src in ('partner','opponent') for (pid, src) in batch['routing_log'] if pid in [1,3])

