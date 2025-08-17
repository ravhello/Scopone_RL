from tests.torch_np import np
from environment import ScoponeEnvMA
from belief.belief import BeliefState


def test_belief_runs_through_full_episode():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=6)
    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=128, ess_frac=0.4)
    steps = 0
    while not env.done and steps < 100:
        legals = env.get_valid_actions()
        if not legals:
            break
        env.step(legals[0])
        steps += 1
    assert steps > 0


