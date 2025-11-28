import torch
import os
import threading
import pytest
from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import _compute_per_seat_diagnostics, collect_trajectory
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


@pytest.mark.parametrize('collector_kind', ['serial', 'parallel'])
def test_compute_per_seat_diagnostics_shapes(collector_kind):
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # create a tiny batch via collect_trajectory
    if collector_kind == 'serial':
        batch = collect_trajectory(env, agent, horizon=40, use_mcts=False)
    else:
        os.environ.setdefault('SCOPONE_PAR_DEBUG', '1')
        os.environ.setdefault('SCOPONE_PPO_DEBUG', '1')
        batch = _run_parallel_with_timeout(
            lambda: collect_batch('parallel', agent, env=env, episodes_total_hint=1, k_history=4, use_mcts=False, mcts_sims=0, mcts_dets=0, train_both_teams=False, main_seats=[0, 2], alternate_main_seats=False),
            timeout_s=40.0,
        )
    # ensure minimal fields
    assert 'obs' in batch and 'legals' in batch
    out = _compute_per_seat_diagnostics(agent, batch)
    # expected keys present
    for key in (
        'by_seat/kl_02','by_seat/kl_13','by_seat/entropy_02','by_seat/entropy_13','by_seat/clip_frac_02','by_seat/clip_frac_13'):
        assert key in out
        assert isinstance(out[key], torch.Tensor)


