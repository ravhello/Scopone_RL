import random

import numpy as np
import pytest
import torch

from algorithms.ppo_ac import ActionConditionedPPO
from environment import ScoponeEnvMA
import trainers.train_ppo as train_mod


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
    ok = train_mod._require_team_rewards({'team_rewards': [1.0, -1.0]}, "ctx")
    assert ok == [1.0, -1.0]
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards({}, "ctx")
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards({'team_rewards': [1.0]}, "ctx")
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards({'team_rewards': [float('nan'), 0.0]}, "ctx")


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
