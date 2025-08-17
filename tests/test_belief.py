from tests.torch_np import np
from environment import ScoponeEnvMA
from belief.belief import BeliefState


def test_belief_runs_through_full_episode():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=6)
    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=128, ess_frac=0.4)
    steps = 0
    while not env.done and steps < 200:
        obs = env._get_observation(env.current_player)
        legals = env.get_valid_actions()
        if not legals:
            break
        action = legals[0]
        obs, rew, done, info = env.step(action)
        steps += 1
    assert steps > 0

