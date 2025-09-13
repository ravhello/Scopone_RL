import torch
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import _compute_per_seat_diagnostics


def test_compute_per_seat_diagnostics_shapes():
    import os
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # create a tiny batch via collect_trajectory
    from trainers.train_ppo import collect_trajectory
    batch = collect_trajectory(env, agent, horizon=40, use_mcts=False)
    # ensure minimal fields
    assert 'obs' in batch and 'legals' in batch
    out = _compute_per_seat_diagnostics(agent, batch)
    # expected keys present
    for key in (
        'by_seat/kl_02','by_seat/kl_13','by_seat/entropy_02','by_seat/entropy_13','by_seat/clip_frac_02','by_seat/clip_frac_13'):
        assert key in out
        assert isinstance(out[key], torch.Tensor)


