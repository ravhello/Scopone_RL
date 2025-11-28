import torch
from environment import ScoponeEnvMA


def test_env_reset_and_shapes():
    env = ScoponeEnvMA(k_history=4)
    obs0 = env._get_observation(env.current_player)
    assert obs0.shape[0] == env.observation_space.shape[0]
    legals = env.get_valid_actions()
    # legals può essere Tensor (A,80) o lista di vettori: verifica solo che non sia vuoto
    assert len(legals) > 0


def test_env_step_with_first_legal():
    env = ScoponeEnvMA(k_history=4)
    legals = env.get_valid_actions()
    a0 = legals[0]
    next_obs, rew, done, info = env.step(a0)
    assert isinstance(next_obs, torch.Tensor)
    assert next_obs.shape[0] == env.observation_space.shape[0]
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_env_reaches_done_eventually():
    env = ScoponeEnvMA(k_history=4)
    steps = 0
    while not env.done and steps < 200:
        legals = env.get_valid_actions()
        assert len(legals) > 0
        env.step(legals[0])
        steps += 1
    assert env.done is True, f"environment never reached done after {steps} steps"
    assert steps == 40, f"expected exactly 40 steps to finish a hand, got {steps}"


