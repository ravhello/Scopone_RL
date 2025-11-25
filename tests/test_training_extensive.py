import os
import torch

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory, train_ppo


def _set_base_env(monkeypatch):
    monkeypatch.setenv('SCOPONE_DISABLE_SAVE', '1')
    monkeypatch.setenv('SCOPONE_DISABLE_EVAL', '1')
    monkeypatch.setenv('SCOPONE_LEAGUE_REFRESH', '0')
    monkeypatch.setenv('SCOPONE_MINIBATCH_ALIGN', '0')
    monkeypatch.setenv('SCOPONE_MINIBATCH', '0')
    monkeypatch.setenv('SCOPONE_DEVICE', 'cpu')
    monkeypatch.setenv('ENV_DEVICE', 'cpu')
    monkeypatch.setenv('SCOPONE_TORCH_COMPILE', '0')
    monkeypatch.setenv('TQDM_DISABLE', '1')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')


def _make_env_agent(monkeypatch):
    _set_base_env(monkeypatch)
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    return env, agent


def test_collect_trajectory_consistency(monkeypatch):
    env, agent = _make_env_agent(monkeypatch)
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
    )
    B = int(batch['obs'].size(0))
    assert B > 0
    # Shapes must align
    assert batch['act'].shape[0] == B
    assert batch['legals_count'].shape[0] == B
    assert batch['chosen_index'].shape[0] == B
    # legals_count must sum to total legals rows
    total_legals = int(batch['legals_count'].sum().item())
    assert batch['legals'].shape[0] == total_legals
    # chosen_index must be within bounds per row
    counts = batch['legals_count']
    idx = batch['chosen_index']
    assert torch.all(idx < counts.clamp_min(1))
    # Advantage/returns finite
    assert torch.isfinite(batch['adv']).all()
    assert torch.isfinite(batch['ret']).all()
    # Routing log should have entries
    assert len(batch.get('routing_log', [])) > 0


def test_update_metrics_and_ranges(monkeypatch):
    env, agent = _make_env_agent(monkeypatch)
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
    )
    info = agent.update(batch, epochs=1, minibatch_size=max(1, batch['obs'].size(0)))
    # Metrics should be finite scalars
    for k, v in info.items():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all(), f"Metric {k} has non-finite values"
    # Sanity: approx_kl should be finite (può essere leggermente negativo per numerica)
    akl = info.get('approx_kl', torch.tensor(0.0))
    if torch.is_tensor(akl):
        assert torch.isfinite(akl).all()
    clip_frac = info.get('clip_frac', torch.tensor(0.0))
    if torch.is_tensor(clip_frac):
        assert 0.0 <= float(clip_frac) <= 1.0 + 1e-3
    # Value loss non negativa per default MSE
    if 'loss_v' in info:
        assert float(info['loss_v']) >= 0.0


class _FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))

    def add_text(self, *args, **kwargs):
        return None

    def close(self):
        return None


def test_train_loop_writes_train_scalars(monkeypatch):
    _set_base_env(monkeypatch)
    writer = _FakeWriter()
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', lambda log_dir=None: writer, raising=True)
    ended = []

    def on_iter_end(it):
        ended.append(int(it))

    train_ppo(
        num_iterations=2,
        horizon=40,
        k_history=4,
        num_envs=1,
        mcts_sims=0,
        mcts_dets=0,
        eval_every=0,
        save_every=100,
        on_iter_end=on_iter_end,
    )
    # Ensure both iterations ran
    assert ended == [0, 1]
    # Ensure at least one train scalar per step
    logged_steps = sorted({s for (t, _v, s) in writer.scalars if t.startswith('train/')})
    assert logged_steps == [0, 1]


def test_resume_checkpoint_skips_empty(monkeypatch, tmp_path):
    _set_base_env(monkeypatch)
    # Crea un checkpoint vuoto per simulare file corrotto
    ckpt_path = tmp_path / "empty.pth"
    ckpt_path.write_bytes(b"")
    monkeypatch.setenv('SCOPONE_DISABLE_SAVE', '1')
    monkeypatch.setenv('SCOPONE_DISABLE_EVAL', '1')
    monkeypatch.setenv('SCOPONE_LEAGUE_REFRESH', '0')
    # Esegui train_ppo: deve ignorare il checkpoint vuoto e non lanciare eccezioni
    train_ppo(
        num_iterations=1,
        horizon=40,
        k_history=4,
        num_envs=1,
        mcts_sims=0,
        mcts_dets=0,
        eval_every=0,
        save_every=100,
        ckpt_path=str(ckpt_path),
    )


def test_collect_trajectory_strict_chosen_index(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.setenv('SCOPONE_STRICT_CHECKS', '1')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # Deve passare senza sollevare mismatch di hash
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
    )
    assert batch['obs'].size(0) > 0


def test_collect_trajectory_train_both_teams_covers_all_seats(monkeypatch):
    env, agent = _make_env_agent(monkeypatch)
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=True,
    )
    seats = batch['seat_team']
    assert seats.shape[0] > 0
    # Ogni riga è one-hot sui primi 4 posti
    seat_ids = torch.argmax(seats[:, :4], dim=1).tolist()
    assert set(seat_ids) == {0, 1, 2, 3}


def test_actor_critic_forward_are_finite(monkeypatch):
    env, agent = _make_env_agent(monkeypatch)
    obs = env._get_observation(env.current_player).unsqueeze(0).to(dtype=torch.float32)
    seat = torch.zeros((1, 6), dtype=torch.float32)
    seat[0, 0] = 1.0
    legals = env.get_valid_actions()
    legals_tensor = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in legals], dim=0)
    with torch.no_grad():
        sp = agent.actor.compute_state_proj(obs, seat)
        logits = agent.actor(obs, legals_tensor, seat)
        val = agent.critic.forward(obs, seat)
    assert torch.isfinite(sp).all()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(val).all()
