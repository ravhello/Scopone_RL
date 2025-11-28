import random

import numpy as np
import pytest
import torch

from algorithms.ppo_ac import ActionConditionedPPO
from environment import ScoponeEnvMA
import trainers.train_ppo as train_mod
from utils.seed import set_global_seeds


def test_normalize_adv_tensor_scales_and_handles_constants():
    adv = torch.tensor([1.0, -1.0, 3.0, -3.0], dtype=torch.float32)
    out = train_mod._normalize_adv_tensor('unit', adv.clone())
    assert torch.isfinite(out).all()
    assert out.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert out.std(unbiased=False).item() == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    const = torch.full((4,), 2.0, dtype=torch.float32)
    const_out = train_mod._normalize_adv_tensor('const', const)
    assert torch.allclose(const_out, torch.zeros_like(const_out), atol=1e-6)
    empty = torch.tensor([], dtype=torch.float32)
    empty_out = train_mod._normalize_adv_tensor('empty', empty)
    assert empty_out.numel() == 0


def test_require_team_rewards_validation_and_conversion():
    base_info = {
        'team_rewards': [1.0, -1.0],
        'score_breakdown': {0: {'total': 5.0}, 1: {'total': 3.0}},
    }
    ok = train_mod._require_team_rewards(base_info, "ctx")
    assert ok == [1.0, -1.0]
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards({'score_breakdown': base_info['score_breakdown']}, "ctx")
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards({'team_rewards': [1.0], 'score_breakdown': base_info['score_breakdown']}, "ctx")
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards({'team_rewards': [float('nan'), 0.0], 'score_breakdown': base_info['score_breakdown']}, "ctx")
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards(
            {'team_rewards': [0.5, 0.5], 'score_breakdown': {0: {'total': 0.0}, 1: {'total': 0.0}}},
            "ctx")


def test_serial_seed_roundtrip_restores_outer_rng_state():
    train_mod._SERIAL_RNG_STATE.clear()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    outer_py = random.getstate()
    outer_np = np.random.get_state()
    outer_torch = torch.get_rng_state()

    token = train_mod._serial_seed_enter(99)
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1)
    train_mod._serial_seed_exit(token)

    r_copy = random.Random()
    r_copy.setstate(outer_py)
    expected_py = r_copy.random()
    rs_copy = np.random.RandomState()
    rs_copy.set_state(outer_np)
    expected_np = rs_copy.rand()
    g = torch.Generator()
    g.set_state(outer_torch)
    expected_torch = float(torch.rand(1, generator=g).item())

    actual_py = random.random()
    actual_np = np.random.rand()
    actual_torch = float(torch.rand(1).item())

    assert actual_py == pytest.approx(expected_py)
    assert actual_np == pytest.approx(expected_np)
    assert actual_torch == pytest.approx(expected_torch, rel=1e-6, abs=1e-6)


def test_serial_seed_caches_progression_between_calls():
    train_mod._SERIAL_RNG_STATE.clear()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    token1 = train_mod._serial_seed_enter(7)
    first_py = random.random()
    train_mod._serial_seed_exit(token1)
    stored_state = train_mod._SERIAL_RNG_STATE[7]['py']

    token2 = train_mod._serial_seed_enter(7)
    second_py = random.random()
    train_mod._serial_seed_exit(token2)

    r = random.Random()
    r.setstate(stored_state)
    expected_second = r.random()
    assert first_py != second_py
    assert second_py == pytest.approx(expected_second)


def test_collect_trajectory_parallel_validates_arguments():
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    with pytest.raises(ValueError):
        train_mod.collect_trajectory_parallel(agent, num_envs=0)
    with pytest.raises(ValueError):
        train_mod.collect_trajectory_parallel(agent, episodes_total_hint=0)
    with pytest.raises(ValueError):
        train_mod.collect_trajectory_parallel(agent, k_history=0)
    with pytest.raises(ValueError):
        train_mod.collect_trajectory_parallel(agent, gamma=1.1)


def test_collect_trajectory_serial_reproducible_with_seed():
    seed_val = 123
    train_mod._SERIAL_RNG_STATE.clear()
    set_global_seeds(seed_val)
    env1 = ScoponeEnvMA(k_history=4)
    agent1 = ActionConditionedPPO(obs_dim=env1.observation_space.shape[0])
    batch1 = train_mod.collect_trajectory(
        env1,
        agent1,
        horizon=20,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
        partner_actor=agent1.actor,
        opponent_actor=agent1.actor,
        alternate_main_seats=False,
        episodes=1,
        seed=seed_val,
    )
    train_mod._SERIAL_RNG_STATE.clear()
    set_global_seeds(seed_val)
    env2 = ScoponeEnvMA(k_history=4)
    agent2 = ActionConditionedPPO(obs_dim=env2.observation_space.shape[0])
    batch2 = train_mod.collect_trajectory(
        env2,
        agent2,
        horizon=20,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
        partner_actor=agent2.actor,
        opponent_actor=agent2.actor,
        alternate_main_seats=False,
        episodes=1,
        seed=seed_val,
    )
    for key in ['obs', 'legals', 'chosen_index', 'rew']:
        assert torch.allclose(batch1[key], batch2[key]), f"Mismatch on {key} with same seed (serial)"


def test_collect_trajectory_parallel_reproducible_with_seed():
    seed_val = 321
    set_global_seeds(seed_val)
    agent1 = ActionConditionedPPO(obs_dim=ScoponeEnvMA(k_history=4).observation_space.shape[0])
    batch1 = train_mod.collect_trajectory_parallel(
        agent1,
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
        seed=seed_val,
        show_progress_env=False,
        tqdm_base_pos=0,
        frozen_actor=agent1.actor,
        frozen_non_main=True,
        alternate_main_seats=False,
    )
    set_global_seeds(seed_val)
    agent2 = ActionConditionedPPO(obs_dim=ScoponeEnvMA(k_history=4).observation_space.shape[0])
    batch2 = train_mod.collect_trajectory_parallel(
        agent2,
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
        seed=seed_val,
        show_progress_env=False,
        tqdm_base_pos=0,
        frozen_actor=agent2.actor,
        frozen_non_main=True,
        alternate_main_seats=False,
    )
    for key in ['obs', 'legals', 'chosen_index', 'rew']:
        assert torch.allclose(batch1[key], batch2[key]), f"Parallel mismatch on {key} with same seed"


def test_serial_and_parallel_deterministic_with_same_seed():
    seed_val = 123
    env_serial = ScoponeEnvMA(k_history=4)
    agent_serial = ActionConditionedPPO(obs_dim=env_serial.observation_space.shape[0])
    batch_serial = train_mod.collect_trajectory(
        env_serial,
        agent_serial,
        horizon=20,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
        partner_actor=agent_serial.actor,
        opponent_actor=agent_serial.actor,
        alternate_main_seats=False,
        episodes=1,
        seed=seed_val,
    )
    agent_parallel = ActionConditionedPPO(obs_dim=env_serial.observation_space.shape[0])
    batch_parallel = train_mod.collect_trajectory_parallel(
        agent_parallel,
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
        seed=seed_val,
        show_progress_env=False,
        tqdm_base_pos=0,
        frozen_actor=agent_parallel.actor,
        frozen_non_main=True,
        alternate_main_seats=False,
    )
    # Con lo stesso seed e un solo episodio, obs e legals devono coincidere
    assert batch_serial['obs'].shape == batch_parallel['obs'].shape
    assert torch.allclose(batch_serial['obs'], batch_parallel['obs'])
    assert torch.allclose(batch_serial['legals'], batch_parallel['legals'])
    assert torch.allclose(batch_serial['chosen_index'], batch_parallel['chosen_index'])
    assert torch.allclose(batch_serial['rew'], batch_parallel['rew'])
