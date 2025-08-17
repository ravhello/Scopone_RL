from tests.torch_np import np
from environment import ScoponeEnvMA
from belief.belief import BeliefState


def test_belief_legal_moves():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=6)
    belief = BeliefState(env.game_state, observer_id=env.current_player, num_particles=64)
    legals = env.get_valid_actions()
    assert len(legals) > 0

