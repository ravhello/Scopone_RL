import os
from environment import ScoponeEnvMA
from algorithms.is_mcts import run_is_mcts
import numpy as np

def test_is_mcts_runs():
    # Stabilize device/compile
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    obs = env._get_observation(env.current_player)
    legals = env.get_valid_actions()
    # after conftest monkeypatch, legals is a list
    assert len(legals) > 0

    def policy_fn(o, leg):
        # prior uniforme
        return np.ones(len(leg), dtype=np.float32) / len(leg)

    def value_fn(o, _env=None):
        return 0.0

    action = run_is_mcts(env, policy_fn, value_fn, num_simulations=8, c_puct=1.0)
    assert action is not None

def test_is_mcts_selection_and_noise_flags():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    def policy_fn(o, leg):
        return np.ones(len(leg), dtype=np.float32) / len(leg)
    def value_fn(o, _env=None):
        return 0.0
    # robust child
    a1 = run_is_mcts(env, policy_fn, value_fn, num_simulations=6, c_puct=1.0, num_determinization=2, robust_child=True)
    assert a1 is not None
    # max-Q con temperatura e smoothing e dirichlet
    a2 = run_is_mcts(env, policy_fn, value_fn, num_simulations=6, c_puct=1.0, num_determinization=2,
                     robust_child=False, root_temperature=0.7, prior_smooth_eps=0.1, root_dirichlet_alpha=0.3, root_dirichlet_eps=0.25)
    assert a2 is not None

def test_is_mcts_with_belief_and_flags():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    def policy_fn(o, leg):
        return np.ones(len(leg), dtype=np.float32) / max(1, len(leg))
    def value_fn(o, _env=None):
        return 0.0
    action = run_is_mcts(env, policy_fn, value_fn, num_simulations=8, c_puct=1.2, num_determinization=2,
                         root_temperature=0.5, prior_smooth_eps=0.1, robust_child=False, root_dirichlet_alpha=0.3, root_dirichlet_eps=0.25)
    assert action is not None

