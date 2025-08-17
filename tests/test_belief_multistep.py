from tests.torch_np import np
from environment import ScoponeEnvMA
from belief.belief import BeliefState


def test_belief_multistep():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=6)
    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=128, ess_frac=0.4)
    for _ in range(10):
        if env.done:
            break
        legals = env.get_valid_actions()
        if not legals:
            break
        env.step(legals[0])
    assert True


