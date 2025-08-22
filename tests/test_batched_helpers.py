import torch
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import _batched_select_indices, _batched_service


def _seat_vec_for(cp: int) -> torch.Tensor:
    v = torch.zeros(6, dtype=torch.float32)
    v[cp] = 1.0
    v[4] = 1.0 if cp in [0, 2] else 0.0
    v[5] = 1.0 if cp in [1, 3] else 0.0
    return v


def test_batched_select_indices_basic():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    obs = env._get_observation(env.current_player)
    legals = env.get_valid_actions()
    assert len(legals) > 0
    legals_small = legals[: min(3, len(legals))]
    cp = env.current_player
    seat = _seat_vec_for(cp)
    reqs = [
        {
            'type': 'step',
            'wid': 0,
            'obs': (obs.tolist() if torch.is_tensor(obs) else list(obs)),
            'legals': [x.tolist() for x in legals_small],
            'seat': seat.tolist(),
        },
        {
            'type': 'step',
            'wid': 1,
            'obs': (obs.tolist() if torch.is_tensor(obs) else list(obs)),
            'legals': [x.tolist() for x in legals_small],
            'seat': seat.tolist(),
        },
    ]
    sel = _batched_select_indices(agent, reqs)
    assert len(sel) == len(reqs)
    for wid, idx in sel:
        assert isinstance(wid, int)
        assert isinstance(idx, int)
        assert 0 <= idx < len(legals_small)


def test_batched_service_policy_value_belief():
    env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    obs = env._get_observation(env.current_player)
    legals = env.get_valid_actions()
    assert len(legals) > 0
    legals_small = legals[: min(3, len(legals))]
    cp = env.current_player
    seat = _seat_vec_for(cp)
    reqs = [
        {
            'type': 'score_policy',
            'wid': 0,
            'obs': (obs.tolist() if torch.is_tensor(obs) else list(obs)),
            'legals': [x.tolist() for x in legals_small],
            'seat': seat.tolist(),
        },
        {
            'type': 'score_value',
            'wid': 0,
            'obs': (obs.tolist() if torch.is_tensor(obs) else list(obs)),
            'seat': seat.tolist(),
        },
        {
            'type': 'score_belief',
            'wid': 0,
            'obs': (obs.tolist() if torch.is_tensor(obs) else list(obs)),
            'seat': seat.tolist(),
        },
    ]
    outs = _batched_service(agent, reqs)
    assert len(outs) == len(reqs)
    assert 'priors' in outs[0] and isinstance(outs[0]['priors'], list) and len(outs[0]['priors']) == len(legals_small)
    assert 'value' in outs[1]
    assert 'belief_probs' in outs[2] and isinstance(outs[2]['belief_probs'], list) and len(outs[2]['belief_probs']) == 120


