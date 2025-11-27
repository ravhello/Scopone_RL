# Keep this file minimal. GPU-dependent tests will self-skip via decorators.
import pytest  # noqa: F401
import os
import warnings

# Silence noisy third-party deprecation chatter that does not impact behavior.
warnings.filterwarnings(
    "ignore",
    message=r"Type google\.protobuf\.pyext\._message\..*PyType_Spec",
    category=DeprecationWarning,
)
# Extra guard: silence the same warning via module regex in case message changes slightly.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*google\.protobuf\.pyext\._message",
)


@pytest.fixture(autouse=True, scope="session")
def force_cpu_for_tests():
    # Force CPU during test session unless explicitly overridden
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('OBS_DEVICE', 'cpu')
    os.environ.setdefault('ACTIONS_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('REW_DEVICE', 'cpu')
    yield


def collect_batch(collector_kind, agent, *, env=None, seed=None, **kwargs):
    """
    Utility to gather a small batch either via serial collect_trajectory or parallel collect_trajectory_parallel
    with consistent default arguments for tests.
    """
    import trainers.train_ppo as train_mod
    if collector_kind == 'parallel':
        return train_mod.collect_trajectory_parallel(
            agent,
            num_envs=1,
            episodes_total_hint=int(kwargs.get('episodes_total_hint', 1)),
            k_history=int(kwargs.get('k_history', 4)),
            gamma=float(kwargs.get('gamma', 1.0)),
            lam=float(kwargs.get('lam', 1.0)),
            use_mcts=bool(kwargs.get('use_mcts', False)),
            train_both_teams=bool(kwargs.get('train_both_teams', False)),
            main_seats=kwargs.get('main_seats', [0, 2]),
            mcts_sims=int(kwargs.get('mcts_sims', 0)),
            mcts_dets=int(kwargs.get('mcts_dets', 0)),
            mcts_c_puct=float(kwargs.get('mcts_c_puct', 1.0)),
            mcts_root_temp=float(kwargs.get('mcts_root_temp', 0.0)),
            mcts_prior_smooth_eps=float(kwargs.get('mcts_prior_smooth_eps', 0.0)),
            mcts_dirichlet_alpha=float(kwargs.get('mcts_dirichlet_alpha', 0.25)),
            mcts_dirichlet_eps=float(kwargs.get('mcts_dirichlet_eps', 0.0)),
            mcts_min_sims=int(kwargs.get('mcts_min_sims', 0)),
            mcts_train_factor=float(kwargs.get('mcts_train_factor', 1.0)),
            seed=seed,
            show_progress_env=False,
            tqdm_base_pos=0,
            frozen_actor=kwargs.get('frozen_actor', kwargs.get('opponent_actor')),
            frozen_non_main=bool(kwargs.get('frozen_non_main', False)),
            alternate_main_seats=bool(kwargs.get('alternate_main_seats', False)),
        )
    else:
        assert env is not None, "env is required for serial collection"
        return train_mod.collect_trajectory(
            env,
            agent,
            horizon=int(kwargs.get('horizon', 40)),
            use_mcts=bool(kwargs.get('use_mcts', False)),
            mcts_sims=int(kwargs.get('mcts_sims', 0)),
            mcts_dets=int(kwargs.get('mcts_dets', 0)),
            mcts_c_puct=float(kwargs.get('mcts_c_puct', 1.0)),
            mcts_root_temp=float(kwargs.get('mcts_root_temp', 0.0)),
            mcts_prior_smooth_eps=float(kwargs.get('mcts_prior_smooth_eps', 0.0)),
            mcts_dirichlet_alpha=float(kwargs.get('mcts_dirichlet_alpha', 0.25)),
            mcts_dirichlet_eps=float(kwargs.get('mcts_dirichlet_eps', 0.0)),
            mcts_train_factor=float(kwargs.get('mcts_train_factor', 1.0)),
            mcts_min_sims=int(kwargs.get('mcts_min_sims', 0)),
            train_both_teams=bool(kwargs.get('train_both_teams', False)),
            partner_actor=kwargs.get('partner_actor'),
            opponent_actor=kwargs.get('opponent_actor'),
            alternate_main_seats=bool(kwargs.get('alternate_main_seats', False)),
            episodes=kwargs.get('episodes', None),
            final_reward_only=kwargs.get('final_reward_only', True),
            seed=seed,
        )
