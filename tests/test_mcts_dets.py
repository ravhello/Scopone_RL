from tests.torch_np import np
from environment import ScoponeEnvMA
from algorithms.is_mcts import run_is_mcts
from belief.belief import BeliefState
from utils.seed import set_global_seeds


def test_mcts_multiple_dets_runs():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=8)
    obs = env._get_observation(env.current_player)
    legals = env.get_valid_actions()
    assert len(legals) > 0
    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=64)
    def policy_fn(o, leg):
        return np.ones(len(leg), dtype=np.float32) / len(leg)
    def value_fn(o):
        return 0.0
    action = run_is_mcts(env, policy_fn, value_fn, num_simulations=8, c_puct=1.0, belief=belief, num_determinization=4,
                         root_temperature=0.6, prior_smooth_eps=0.1, robust_child=False, root_dirichlet_alpha=0.3, root_dirichlet_eps=0.25)[0]
    assert action is not None


def test_mcts_robust_vs_maxq_and_noise():
    set_global_seeds(123)
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=64)
    def policy_fn(o, leg):
        return np.ones(len(leg), dtype=np.float32) / len(leg)
    def value_fn(o):
        return 0.0
    a1, p1 = run_is_mcts(env, policy_fn, value_fn, num_simulations=10, c_puct=1.0, belief=belief, num_determinization=2, robust_child=True, return_stats=True)
    a2, p2 = run_is_mcts(env, policy_fn, value_fn, num_simulations=10, c_puct=1.0, belief=belief, num_determinization=2,
                     robust_child=False, root_temperature=0.7, prior_smooth_eps=0.1, root_dirichlet_alpha=0.3, root_dirichlet_eps=0.25, return_stats=True)
    assert a1 is not None and a2 is not None
    # distribuzioni di visite/prob dovrebbero essere validhe
    from tests.torch_np import np
    p1_np = p1
    p2_np = p2
    assert abs(np.sum(p1_np) - 1.0) < 1e-6
    assert abs(np.sum(p2_np) - 1.0) < 1e-6
    # con parametri diversi le distribuzioni dovrebbero differire
    assert not np.allclose(p1_np, p2_np)


def test_mcts_temperature_effect_on_entropy():
    set_global_seeds(321)
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=64)
    def policy_fn(o, leg):
        return np.ones(len(leg), dtype=np.float32) / len(leg)
    def value_fn(o):
        return 0.0
    _, p_cold = run_is_mcts(env, policy_fn, value_fn, num_simulations=12, c_puct=1.0, belief=belief, num_determinization=2,
                             robust_child=False, root_temperature=0.2, return_stats=True)
    _, p_hot = run_is_mcts(env, policy_fn, value_fn, num_simulations=12, c_puct=1.0, belief=belief, num_determinization=2,
                            robust_child=False, root_temperature=1.2, return_stats=True)
    # entropia con temperatura più alta dovrebbe essere >=
    p_cold_t = np.to_tensor(p_cold, dtype=np.float32)
    p_hot_t = np.to_tensor(p_hot, dtype=np.float32)
    H_cold = -np.sum(p_cold_t * np.log(p_cold_t + 1e-12))
    H_hot = -np.sum(p_hot_t * np.log(p_hot_t + 1e-12))
    assert float(H_hot) >= float(H_cold) - 1e-6

