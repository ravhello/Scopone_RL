import os
import os
import threading
import pytest
import torch

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory
from actions import encode_action_from_ids_tensor, decode_action_ids, encode_action_hash
from conftest import collect_batch


def _run_parallel_with_timeout(fn, timeout_s: float = 30.0):
    """
    Esegue fn in un thread separato con timeout esplicito per evitare blocchi silenziosi
    nel collector parallelo; se scade il timeout, fallisce con un messaggio chiaro.
    """
    out = {}
    err = []

    def _wrap():
        try:
            out['val'] = fn()
        except BaseException as exc:
            err.append(exc)

    t = threading.Thread(target=_wrap, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        pytest.fail(f"collect_batch(parallel) non ha terminato entro {timeout_s}s; abilita SCOPONE_PAR_DEBUG=1 per maggiori log")
    if err:
        raise err[0]
    if 'val' not in out:
        pytest.fail("collect_batch(parallel) thread terminato senza produrre un valore di ritorno")
    return out['val']


def _make_agent_and_batch(horizon: int = 40, collector_kind: str = 'serial'):
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    if collector_kind == 'serial':
        batch = collect_trajectory(
            env,
            agent,
            horizon=horizon,
            use_mcts=False,
            mcts_sims=0,
            mcts_dets=0,
            train_both_teams=False,
        )
    else:
        os.environ.setdefault('SCOPONE_PAR_DEBUG', '1')
        os.environ.setdefault('SCOPONE_PPO_DEBUG', '1')
        batch = _run_parallel_with_timeout(
            lambda: collect_batch(
                'parallel',
                agent,
                env=env,
                episodes_total_hint=1,
                k_history=4,
                use_mcts=False,
                mcts_sims=0,
                mcts_dets=0,
                train_both_teams=False,
                main_seats=[0, 2],
                alternate_main_seats=False,
            ),
            timeout_s=40.0,
        )
    return agent, batch


@pytest.mark.parametrize('collector_kind', ['serial', 'parallel'])
def test_update_changes_parameters_and_metrics_are_finite(collector_kind):
    agent, batch = _make_agent_and_batch(horizon=40, collector_kind=collector_kind)
    # Snapshot weights before update
    params_before = [p.detach().clone() for p in list(agent.actor.parameters()) + list(agent.critic.parameters())]
    # Run a single update step on the collected batch
    info = agent.update(batch, epochs=1, minibatch_size=max(1, batch['obs'].size(0)))
    # Metrics should be finite scalars
    for v in info.values():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all()
    # At least one parameter should have changed
    params_after = list(agent.actor.parameters()) + list(agent.critic.parameters())
    changed = any(not torch.equal(p0, p1) for p0, p1 in zip(params_before, params_after))
    assert changed, "Update did not modify any model parameters"


def test_encode_action_hash_is_consistent_and_distinguishes_captures():
    # Base action
    played = torch.tensor(7, dtype=torch.long)
    captured = torch.tensor([2, 9], dtype=torch.long)
    vec = encode_action_from_ids_tensor(played, captured)
    code = encode_action_hash(vec)
    # Round-trip through decode/encode preserves the hash
    p2, cap2 = decode_action_ids(vec)
    vec2 = encode_action_from_ids_tensor(torch.tensor(p2), torch.tensor(cap2, dtype=torch.long))
    assert code == encode_action_hash(vec2)
    # Different capture sets should yield a different hash
    vec_alt = encode_action_from_ids_tensor(played, torch.tensor([3, 9], dtype=torch.long))
    assert code != encode_action_hash(vec_alt)
