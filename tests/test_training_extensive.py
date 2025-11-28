import os
import torch
import pytest
import numpy as np
import threading
from typing import Dict

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from trainers.train_ppo import collect_trajectory, collect_trajectory_parallel, train_ppo
from models.action_conditioned import ActionConditionedActor
from utils.seed import set_global_seeds


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
        pytest.fail(f"collect_trajectory_parallel non ha terminato entro {timeout_s}s; abilita SCOPONE_PAR_DEBUG=1 per maggiori log")
    if err:
        raise err[0]
    return out.get('val')


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


def _seat_vec_for(seat_idx: int) -> torch.Tensor:
    v = torch.zeros(6, dtype=torch.float32)
    v[seat_idx] = 1.0
    v[4] = 1.0 if seat_idx in (0, 2) else 0.0
    v[5] = 1.0 if seat_idx in (1, 3) else 0.0
    return v


def _eval_agent_vs_random(agent: ActionConditionedPPO, games: int = 4) -> float:
    env = ScoponeEnvMA(k_history=4)
    wins = 0
    for _ in range(games):
        env.reset()
        done = False
        while not done:
            legals = env.get_valid_actions()
            if env.current_player in (0, 2):
                obs = env._get_observation(env.current_player)
                seat_vec = _seat_vec_for(env.current_player)
                act, _, _ = agent.select_action(obs, legals, seat_vec)
            else:
                # random opponent/partner
                if isinstance(legals, list):
                    act = legals[0] if len(legals) == 1 else legals[torch.randint(len(legals), (1,)).item()]
                else:
                    idx = torch.randint(legals.size(0), (1,)).item()
                    act = legals[idx]
            _, _r, done, _info = env.step(act)
        if env.rewards[0] > env.rewards[1]:
            wins += 1
    return wins / float(games) if games > 0 else 0.0


def _make_batch_with_dummy_rewards(monkeypatch, team0_reward: float = 1.0, team1_reward: float = -1.0):
    _set_base_env(monkeypatch)
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
    )
    # Override rewards in batch to controlled values for testing normalize/reduce logic
    rew = batch['rew']
    seats = batch['seat_team']
    team0_mask = seats[:, 4] > 0.5
    team1_mask = seats[:, 5] > 0.5
    rew = rew.clone()
    rew[team0_mask] = team0_reward
    rew[team1_mask] = team1_reward
    batch['rew'] = rew
    # Recompute ret/adv trivially for sanity (no bootstrap)
    batch['adv'] = torch.zeros_like(rew)
    batch['ret'] = rew.clone()
    return agent, batch


def _check_flat_rewards_for_batch(batch):
    rew = batch['rew']
    seats = batch['seat_team']
    team0_mask = seats[:, 4] > 0.5
    team1_mask = seats[:, 5] > 0.5
    if team0_mask.any():
        r0 = rew[team0_mask]
        assert torch.allclose(r0, r0[0])
    if team1_mask.any():
        r1 = rew[team1_mask]
        assert torch.allclose(r1, r1[0])


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
    # Flag team0 su seat pari, team1 su dispari
    for sid, row in zip(seat_ids, seats):
        if sid in (0, 2):
            assert row[4] > 0.5 and row[5] < 0.5
        else:
            assert row[5] > 0.5 and row[4] < 0.5


def test_flat_reward_constant_within_episode(monkeypatch):
    env, agent = _make_env_agent(monkeypatch)
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
        final_reward_only=True,
    )
    rew = batch['rew']
    done = batch['done']
    seats = batch['seat_team']
    if rew.numel() == 0:
        return
    start = 0
    for i, d in enumerate(done.tolist()):
        if d or i == len(done) - 1:
            end = i + 1
            seg_rew = rew[start:end]
            seg_seats = seats[start:end]
            team_flags = (seg_seats[:, 4] > 0.5).to(torch.bool)
            r_team0 = seg_rew[team_flags]
            r_team1 = seg_rew[~team_flags]
            if r_team0.numel() > 0:
                assert torch.allclose(r_team0, r_team0[0])
            if r_team1.numel() > 0:
                assert torch.allclose(r_team1, r_team1[0])
            start = end


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


def test_train_logging_avg_return_matches_batch(monkeypatch):
    _set_base_env(monkeypatch)
    captured = {}
    writer = _FakeWriter()
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', lambda log_dir=None: writer, raising=True)
    import trainers.train_ppo as train_mod
    orig_collect = train_mod.collect_trajectory

    def wrapped_collect(*args, **kwargs):
        batch = orig_collect(*args, **kwargs)
        captured['batch'] = batch
        return batch

    monkeypatch.setattr(train_mod, 'collect_trajectory', wrapped_collect, raising=True)
    train_mod.train_ppo(
        num_iterations=1,
        horizon=40,
        k_history=4,
        num_envs=1,
        mcts_sims=0,
        mcts_dets=0,
        eval_every=0,
        save_every=100,
    )
    assert 'batch' in captured
    mean_ret = float(captured['batch']['ret'].mean().item()) if captured['batch']['ret'].numel() > 0 else 0.0
    logged = [v for (t, v, s) in writer.scalars if t == 'train/avg_return' and s == 0]
    assert logged, "Missing train/avg_return log"
    assert abs(logged[0] - mean_ret) < 1e-6


def test_short_training_does_not_degrade_vs_random(monkeypatch):
    # Verifica che poche iterazioni non peggiorino drasticamente vs random baseline
    _set_base_env(monkeypatch)
    # Fissa seed per ridurre flakiness
    torch.manual_seed(123)
    agent = ActionConditionedPPO(obs_dim=ScoponeEnvMA(k_history=4).observation_space.shape[0])
    base_wr = _eval_agent_vs_random(agent, games=6)
    train_ppo(
        num_iterations=3,
        horizon=40,
        k_history=4,
        num_envs=1,
        mcts_sims=0,
        mcts_dets=0,
        eval_every=0,
        save_every=100,
    )
    improved_wr = _eval_agent_vs_random(agent, games=6)
    # Consenti fluttuazioni ampie date poche partite, evita solo degrado massiccio
    assert improved_wr >= base_wr - 0.5


def test_return_matches_env_rewards(monkeypatch):
    env, agent = _make_env_agent(monkeypatch)
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
        final_reward_only=True,
    )
    if batch['obs'].numel() == 0:
        return
    # Ricostruisci reward finale per team dal campo rew e dalla maschera seat_team
    rew = batch['rew']
    seats = batch['seat_team']
    team0_mask = seats[:, 4] > 0.5
    team1_mask = seats[:, 5] > 0.5
    # Reward finale dovrebbe essere costante per team nell'episodio
    r0 = rew[team0_mask]
    r1 = rew[team1_mask]
    if r0.numel() > 0 and r1.numel() > 0:
        assert torch.allclose(r0, r0[0])
        assert torch.allclose(r1, r1[0])


def test_env_rewards_updated_on_done(monkeypatch):
    _set_base_env(monkeypatch)
    env = ScoponeEnvMA(k_history=4)
    # Forza una mano minima: giocatori 0 e 2 contro 1 e 3, tavolo vuoto, una carta per team0 che prende tutto
    env.reset()
    env.current_player = 0
    env.game_state["hands"] = {0: [0], 1: [], 2: [], 3: []}  # carta 0 a P0
    env.game_state["table"] = []
    env.game_state["captured_squads"] = {0: [1], 1: []}  # simula almeno una presa precedente
    env.game_state["history"] = [{"player": 0, "played_card": 1, "capture_type": "capture", "captured_cards": []}]
    env._rebuild_id_caches()
    import torch
    from actions import encode_action_from_ids_tensor
    act = encode_action_from_ids_tensor(torch.tensor(0, dtype=torch.long), torch.tensor([], dtype=torch.long))
    _obs, r, done, info = env.step(act)
    assert done is True
    # env.rewards dovrebbe riflettere info['team_rewards']
    if isinstance(info, dict) and 'team_rewards' in info:
        t0, t1 = info['team_rewards']
        assert env.rewards[0] == t0
        assert env.rewards[1] == t1


def test_iterative_avg_return_monotonicity(monkeypatch):
    _set_base_env(monkeypatch)
    torch.manual_seed(321)
    agent = ActionConditionedPPO(obs_dim=ScoponeEnvMA(k_history=4).observation_space.shape[0])
    wrs = []
    # Eval baseline
    wrs.append(_eval_agent_vs_random(agent, games=6))
    # Esegui 5 iterazioni brevi, registrando win-rate dopo ciascuna
    for _ in range(2):
        train_ppo(
            num_iterations=2,
            horizon=40,
            k_history=4,
            num_envs=1,
            mcts_sims=0,
            mcts_dets=0,
            eval_every=0,
            save_every=100,
        )
        wrs.append(_eval_agent_vs_random(agent, games=6))
    # Non deve scendere drasticamente (tolleriamo -0.5 rispetto al migliore raggiunto)
    best = max(wrs)
    assert wrs[-1] >= best - 0.5


def test_warm_start_zero_skips_resume_load(monkeypatch, tmp_path):
    _set_base_env(monkeypatch)
    monkeypatch.setenv('SCOPONE_WARM_START', '0')
    ckpt_path = tmp_path / "dummy.pth"
    ckpt_path.write_bytes(b"123")  # non-empty file to tempt resume
    load_called = {'flag': False}

    # Spy ActionConditionedPPO.load
    import algorithms.ppo_ac as ppo_mod
    orig_load = ppo_mod.ActionConditionedPPO.load
    def spy_load(self, path, map_location=None):
        load_called['flag'] = True
        return orig_load(self, path, map_location=map_location)
    monkeypatch.setattr(ppo_mod.ActionConditionedPPO, 'load', spy_load, raising=True)

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
    assert load_called['flag'] is False


def _assert_batches_close(b_serial, b_parallel, *, check_values: bool = True) -> bool:
    fields = [
        'obs', 'act', 'ret', 'adv', 'rew', 'done',
        'legals', 'legals_offset', 'legals_count',
        'chosen_index', 'seat_team',
    ]
    diffs = []
    for f in fields:
        t1 = b_serial[f]
        t2 = b_parallel[f]
        if t1.shape != t2.shape:
            diffs.append(f"{f}: shape {tuple(t1.shape)} vs {tuple(t2.shape)}")
            continue
        if check_values and not torch.allclose(t1, t2, atol=1e-6, rtol=1e-5):
            delta = (t1 - t2).abs()
            max_d = float(delta.max().item()) if delta.numel() > 0 else 0.0
            mean_d = float(delta.to(dtype=torch.float32).mean().item()) if delta.numel() > 0 else 0.0
            diffs.append(f"{f}: max_diff={max_d} mean_diff={mean_d}")
    if diffs:
        print("[serial-vs-parallel mismatch]")
        for d in diffs:
            print(d)
        return False
    return True


def _serial_trend(seed_val: int, opponent_actor: ActionConditionedActor, games: int = 200) -> Dict[str, float]:
    set_global_seeds(seed_val)
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    diffs, ret_means, adv_means = [], [], []
    initial_state = None
    for _ in range(games):
        batch = collect_trajectory(
            env,
            agent,
            horizon=20,
            use_mcts=False,
            mcts_sims=0,
            mcts_dets=0,
            train_both_teams=False,
            partner_actor=agent.actor,  # seat 0/2
            opponent_actor=opponent_actor,      # seat 1/3 frozen
            alternate_main_seats=False,
            episodes=1,
        )
        assert batch['obs'].size(0) == 20
        _check_flat_rewards_for_batch(batch)
        if initial_state is None:
            initial_state = torch.clone(env._hands_bits_t)
        else:
            assert torch.equal(env._hands_bits_t, initial_state)
        diff = env.rewards[0] - env.rewards[1]
        diffs.append(float(diff))
        ret_means.append(float(batch['ret'].mean().item()) if batch['ret'].numel() > 0 else 0.0)
        adv_means.append(float(batch['adv'].mean().item()) if batch['adv'].numel() > 0 else 0.0)
        if batch['obs'].numel() > 0:
            agent.update(batch, epochs=1, minibatch_size=20)
    return {
        'diffs': diffs,
        'ret_means': ret_means,
        'adv_means': adv_means,
    }


def _parallel_trend(seed_val: int, opponent_actor: ActionConditionedActor, games: int = 200) -> Dict[str, float]:
    set_global_seeds(seed_val)
    agent = ActionConditionedPPO(obs_dim=ScoponeEnvMA(k_history=4).observation_space.shape[0])
    diffs, ret_means, adv_means = [], [], []
    # Usa chunk_size=1 per avvicinarsi al percorso seriale (un episodio per raccolta)
    chunk_size = 1
    remaining = games

    def _slice_episode_batch(full_batch, start_idx: int, end_idx: int):
        """Estrai un sotto-batch per singolo episodio, riallineando legals e offset."""
        leg_off_t = full_batch['legals_offset']
        leg_cnt_t = full_batch['legals_count']
        if leg_off_t.numel() > 0:
            leg_start = int(leg_off_t[start_idx].item())
        else:
            leg_start = 0
        if end_idx > start_idx and leg_cnt_t.numel() > 0:
            last_off = int(leg_off_t[end_idx - 1].item())
            last_cnt = int(leg_cnt_t[end_idx - 1].item())
            leg_end = last_off + last_cnt
        else:
            leg_end = leg_start
        def _maybe_slice(key):
            v = full_batch.get(key, None)
            if v is None:
                return v
            if isinstance(v, list):
                return v[start_idx:end_idx]
            return v[start_idx:end_idx]
        ep_idx = max(0, episodes_in_batch - 1)
        ep = {
            'obs': full_batch['obs'][start_idx:end_idx],
            'act': full_batch['act'][start_idx:end_idx],
            'ret': full_batch['ret'][start_idx:end_idx],
            'adv': full_batch['adv'][start_idx:end_idx],
            'rew': full_batch['rew'][start_idx:end_idx],
            'done': full_batch['done'][start_idx:end_idx],
            'seat_team': full_batch['seat_team'][start_idx:end_idx],
            'belief_summary': full_batch['belief_summary'][start_idx:end_idx],
            'legals': full_batch['legals'][leg_start:leg_end],
            'legals_offset': full_batch['legals_offset'][start_idx:end_idx] - leg_start,
            'legals_count': full_batch['legals_count'][start_idx:end_idx],
            'chosen_index': full_batch['chosen_index'][start_idx:end_idx],
            'mcts_weight': full_batch.get('mcts_weight', torch.zeros((0,)))[start_idx:end_idx],
            'others_hands': full_batch.get('others_hands', torch.zeros((0, 3, 40)))[start_idx:end_idx],
            'routing_log': _maybe_slice('routing_log'),
        }
        if 'mcts_policy' in full_batch:
            ep['mcts_policy'] = full_batch['mcts_policy'][leg_start:leg_end]
        if 'logp' in full_batch:
            ep['logp'] = full_batch['logp'][start_idx:end_idx]
        if 'episode_scores' in full_batch:
            ep['episode_scores'] = full_batch['episode_scores'][ep_idx:ep_idx + 1]
        if 'episode_team_rewards' in full_batch:
            ep['episode_team_rewards'] = full_batch['episode_team_rewards'][ep_idx:ep_idx + 1]
        return ep

    while remaining > 0:
        run_eps = min(chunk_size, remaining)
        batch = _run_parallel_with_timeout(
            lambda: collect_trajectory_parallel(
                agent,
                num_envs=1,
                episodes_total_hint=run_eps,
                k_history=4,
                gamma=1.0,
                lam=1.0,
                use_mcts=False,
                train_both_teams=False,
                main_seats=[0, 2],
                mcts_sims=0,
                mcts_dets=0,
                mcts_c_puct=1.0,
                mcts_root_temp=0.0,
                mcts_prior_smooth_eps=0.0,
                mcts_dirichlet_alpha=0.25,
                mcts_dirichlet_eps=0.0,
                mcts_min_sims=0,
                mcts_train_factor=1.0,
                seed=seed_val,
                show_progress_env=False,
                tqdm_base_pos=0,
                frozen_actor=opponent_actor,
                frozen_non_main=True,
                alternate_main_seats=False,
            ),
            timeout_s=120.0,
        )
        done_flags = batch['done'].tolist()
        start = 0
        episodes_in_batch = 0
        for i, d in enumerate(done_flags):
            if not d:
                continue
            end = i + 1
            episodes_in_batch += 1
            ep_batch = _slice_episode_batch(batch, start, end)
            seats = ep_batch['seat_team']
            ret = ep_batch['ret']
            team_rewards = ep_batch.get('episode_team_rewards', None)
            if team_rewards is not None and team_rewards.numel() >= 2:
                diff = float(team_rewards[0][0].item() - team_rewards[0][1].item())
            else:
                team0_mask = seats[:, 4] > 0.5
                team1_mask = seats[:, 5] > 0.5
                if team0_mask.any() and team1_mask.any():
                    diff = float(ret[team0_mask].mean().item() - ret[team1_mask].mean().item())
                else:
                    diff = 0.0
            diffs.append(diff)
            ret_means.append(float(ret.mean().item()) if ret.numel() > 0 else 0.0)
            adv_means.append(float(ep_batch['adv'].mean().item()) if ep_batch['adv'].numel() > 0 else 0.0)
            if ep_batch['obs'].numel() > 0:
                agent.update(ep_batch, epochs=1, minibatch_size=20)
            start = end
            if len(diffs) >= games:
                break
        remaining = max(0, games - len(diffs))
    return {
        'diffs': diffs,
        'ret_means': ret_means,
        'adv_means': adv_means,
    }


def test_collectors_parallel_matches_serial_shapes(monkeypatch):
    _set_base_env(monkeypatch)
    seed_val = 999
    set_global_seeds(seed_val)
    opponent_base_env = ScoponeEnvMA(k_history=4)
    opponent = ActionConditionedPPO(obs_dim=opponent_base_env.observation_space.shape[0]).actor

    set_global_seeds(seed_val)
    serial_env = ScoponeEnvMA(k_history=4)
    serial_agent = ActionConditionedPPO(obs_dim=serial_env.observation_space.shape[0])
    batch_serial = collect_trajectory(
        serial_env,
        serial_agent,
        horizon=20,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
        partner_actor=serial_agent.actor,
        opponent_actor=opponent,
        alternate_main_seats=False,
        episodes=1,
        seed=seed_val,
    )

    set_global_seeds(seed_val)
    parallel_agent = ActionConditionedPPO(obs_dim=opponent_base_env.observation_space.shape[0])
    batch_parallel = _run_parallel_with_timeout(
        lambda: collect_trajectory_parallel(
            parallel_agent,
            num_envs=1,
            episodes_total_hint=1,
            k_history=4,
            gamma=1.0,
            lam=1.0,
            use_mcts=False,
            train_both_teams=False,
            main_seats=[0, 2],
            mcts_sims=0,
            mcts_dets=0,
            mcts_c_puct=1.0,
            mcts_root_temp=0.0,
            mcts_prior_smooth_eps=0.0,
            mcts_dirichlet_alpha=0.25,
            mcts_dirichlet_eps=0.0,
            mcts_min_sims=0,
            mcts_train_factor=1.0,
            seed=seed_val,
            show_progress_env=False,
            tqdm_base_pos=0,
            frozen_actor=opponent,
            frozen_non_main=True,
            alternate_main_seats=False,
        ),
        timeout_s=60.0,
    )

    # Richiede coerenza delle shape e dei valori
    assert _assert_batches_close(batch_serial, batch_parallel, check_values=False)
    assert _assert_batches_close(batch_serial, batch_parallel, check_values=True)


def test_serial_repeated_games_improve_score_diff(monkeypatch):
    _set_base_env(monkeypatch)
    seed_val = 999
    set_global_seeds(seed_val)
    opponent = ActionConditionedPPO(obs_dim=ScoponeEnvMA(k_history=4).observation_space.shape[0]).actor
    stats = _serial_trend(seed_val, opponent, games=200)
    diffs = stats['diffs']
    ret_means = stats['ret_means']
    adv_means = stats['adv_means']
    first_avg = sum(diffs[:10]) / 10.0
    last_avg = sum(diffs[-10:]) / 10.0
    print(f"[serial trend] first_avg={first_avg:.4f} last_avg={last_avg:.4f}")
    assert last_avg > first_avg, f"Serial score diff did not improve: first_avg={first_avg}, last_avg={last_avg}"
    assert all(np.isfinite(ret_means))
    assert all(np.isfinite(adv_means))


def test_parallel_repeated_games_improve_score_diff(monkeypatch):
    _set_base_env(monkeypatch)
    seed_val = 999
    set_global_seeds(seed_val)
    opponent = ActionConditionedPPO(obs_dim=ScoponeEnvMA(k_history=4).observation_space.shape[0]).actor
    stats = _parallel_trend(seed_val, opponent, games=200)
    diffs = stats['diffs']
    ret_means = stats['ret_means']
    adv_means = stats['adv_means']
    first_avg = sum(diffs[:10]) / 10.0 if len(diffs) >= 10 else 0.0
    last_avg = sum(diffs[-10:]) / 10.0 if len(diffs) >= 10 else 0.0
    print(f"[parallel trend] first_avg={first_avg:.4f} last_avg={last_avg:.4f}")
    assert last_avg > first_avg, f"Parallel score diff did not improve: first_avg={first_avg}, last_avg={last_avg}"
    assert all(np.isfinite(ret_means))
    assert all(np.isfinite(adv_means))


def test_update_respects_overridden_rewards(monkeypatch):
    agent, batch = _make_batch_with_dummy_rewards(monkeypatch, team0_reward=2.0, team1_reward=-2.0)
    info = agent.update(batch, epochs=1, minibatch_size=max(1, batch['obs'].size(0)))
    # Loss terms devono essere finiti
    for k, v in info.items():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all(), f"Non-finite metric {k}"
    # Ret medi coerenti con override
    ret = batch['ret']
    seats = batch['seat_team']
    t0_mask = seats[:, 4] > 0.5
    t1_mask = seats[:, 5] > 0.5
    mean_t0 = float(ret[t0_mask].mean().item()) if t0_mask.any() else 0.0
    mean_t1 = float(ret[t1_mask].mean().item()) if t1_mask.any() else 0.0
    assert mean_t0 > 0 and mean_t1 < 0


def test_avg_return_logging_with_mixed_rewards(monkeypatch):
    _set_base_env(monkeypatch)
    writer = _FakeWriter()
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', lambda log_dir=None: writer, raising=True)
    agent, batch = _make_batch_with_dummy_rewards(monkeypatch, team0_reward=1.5, team1_reward=-0.5)
    # Monkeypatch collect_trajectory to return our batch
    import trainers.train_ppo as train_mod
    monkeypatch.setattr(train_mod, 'collect_trajectory', lambda *args, **kwargs: batch, raising=True)
    train_mod.train_ppo(
        num_iterations=1,
        horizon=40,
        k_history=4,
        num_envs=1,
        mcts_sims=0,
        mcts_dets=0,
        eval_every=0,
        save_every=100,
    )
    mean_ret = float(batch['ret'].mean().item()) if batch['ret'].numel() > 0 else 0.0
    logged = [v for (t, v, s) in writer.scalars if t == 'train/avg_return' and s == 0]
    assert logged, "Missing train/avg_return log"
    assert abs(logged[0] - mean_ret) < 1e-6


def test_replay_single_trajectory_increases_chosen_prob(monkeypatch):
    _set_base_env(monkeypatch)
    torch.manual_seed(42)
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
    )
    if batch['obs'].numel() == 0:
        return
    # Calcola probazione dell'azione scelta per il primo sample
    idx0 = 0
    cnt0 = int(batch['legals_count'][idx0].item())
    off0 = int(batch['legals_offset'][idx0].item())
    legals0 = batch['legals'][off0:off0 + cnt0]
    seat0 = batch['seat_team'][idx0:idx0 + 1]
    obs0 = batch['obs'][idx0:idx0 + 1]
    with torch.no_grad():
        logits0 = agent.actor(obs0, legals0, seat0)
        probs0 = torch.softmax(logits0.squeeze(0), dim=0)
    chosen_idx0 = int(batch['chosen_index'][idx0].item())
    prob_before = float(probs0[chosen_idx0].item()) if probs0.numel() > chosen_idx0 else 0.0

    # Rinforza il batch ripetendolo e imponendo advantaggio positivo
    batch['adv'] = torch.ones_like(batch['ret'])
    batch['ret'] = torch.ones_like(batch['ret'])
    for _ in range(3):
        agent.update(batch, epochs=1, minibatch_size=max(1, batch['obs'].size(0)))

    with torch.no_grad():
        logits1 = agent.actor(obs0, legals0, seat0)
        probs1 = torch.softmax(logits1.squeeze(0), dim=0)
    prob_after = float(probs1[chosen_idx0].item()) if probs1.numel() > chosen_idx0 else 0.0
    # Non deve peggiorare significativamente
    assert prob_after >= prob_before - 0.05
