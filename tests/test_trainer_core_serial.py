import os

import torch

from algorithms.ppo_ac import ActionConditionedPPO
from environment import ScoponeEnvMA
import trainers.train_ppo as train_mod
from trainers.train_ppo import collect_trajectory, train_ppo


def test_collect_trajectory_without_mcts_fallback_runs():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    # MCTS disattivato: esercita solo la policy fallback
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        mcts_c_puct=1.0,
        mcts_min_sims=0,
        train_both_teams=False,
    )
    assert 'obs' in batch and batch['obs'].shape[0] > 0
    assert 'chosen_index' in batch and batch['chosen_index'].shape[0] == batch['obs'].shape[0]


def test_collect_trajectory_marks_done_on_episode_boundaries():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    os.environ.setdefault('SCOPONE_MINIBATCH', '20')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=1,
        train_both_teams=False,
        final_reward_only=True,
    )
    done = batch['done']
    assert done.dtype == torch.bool  # type: ignore[attr-defined]
    # horizon=40 with train_both_teams=False collects two 20-step episodes
    assert int(done.sum().item()) == 2


def test_train_ppo_single_env_one_iter_smoke():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    # Verify one iteration completes in single-env mode (exercise non-parallel path)
    ran = {'called': False}

    def on_end(_):
        ran['called'] = True

    train_ppo(
        num_iterations=1,
        horizon=40,
        k_history=4,
        num_envs=1,
        mcts_sims=0,
        mcts_dets=0,
        on_iter_end=on_end,
    )
    assert ran['called'] is True


def test_collect_trajectory_serial_has_rewards_and_logp():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
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
        main_seats=[0, 2],
        alternate_main_seats=False,
        episodes=10,
    )
    assert 'logp' in batch, "missing logp in serial batch"
    assert batch['logp'].shape[0] == batch['obs'].shape[0]
    assert 'rew' in batch and batch['rew'].shape[0] == batch['obs'].shape[0]
    assert 'done' in batch and batch['done'].shape[0] == batch['obs'].shape[0]
    seats = batch['seat_team']
    per_ep = []
    start = 0
    for i, d in enumerate(batch['done']):
        if bool(d.item()):
            end = i + 1
            seats_ep = seats[start:end]
            rew_ep = batch['rew'][start:end]
            t0 = rew_ep[seats_ep[:, 4] > 0.5].sum()
            t1 = rew_ep[seats_ep[:, 5] > 0.5].sum()
            per_ep.append((float(t0.item()), float(t1.item())))
            start = end
    assert per_ep, "no episodes found in serial batch"
    # train_both_teams=False: batch contiene solo main seat; richiediamo almeno un episodio con reward non nullo
    assert any((abs(t0) > 0 or abs(t1) > 0) for (t0, t1) in per_ep), f"all episode rewards are zero: {per_ep}"


def test_collect_trajectory_serial_main_only_rew_matches_team_rewards():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
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
        main_seats=[0, 2],
        alternate_main_seats=False,
        episodes=5,
    )
    team_rewards = batch['episode_team_rewards']
    assert team_rewards.shape[0] > 0
    done = batch['done']
    start = 0
    ep_idx = 0
    for i, d in enumerate(done):
        if not bool(d.item()):
            continue
        end = i + 1
        rew_ep = batch['rew'][start:end]
        unique = torch.unique(rew_ep)
        assert unique.numel() == 1, f"unexpected multiple reward values in main-only episode: {unique.tolist()}"
        t0 = float(team_rewards[ep_idx][0].item())
        assert torch.allclose(unique[0], torch.tensor(t0, dtype=rew_ep.dtype)), f"per-step rewards {unique[0]} differ from team_rewards[0]={t0}"
        start = end
        ep_idx += 1
    assert ep_idx == team_rewards.shape[0]


def test_collect_trajectory_serial_main_only_rew_matches_team_rewards_alternate():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
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
        main_seats=[0, 2],
        alternate_main_seats=True,
        episodes=5,
    )
    team_rewards = batch['episode_team_rewards']
    assert team_rewards.shape[0] > 0
    done = batch['done']
    start = 0
    ep_idx = 0
    for i, d in enumerate(done):
        if not bool(d.item()):
            continue
        end = i + 1
        rew_ep = batch['rew'][start:end]
        seats_ep = batch['seat_team'][start:end]
        team0_mask = seats_ep[:, 4] > 0.5
        team1_mask = seats_ep[:, 5] > 0.5
        team_flag = 0 if bool(team0_mask.any()) else 1
        per_step = rew_ep[team0_mask | team1_mask]
        unique = torch.unique(per_step)
        assert unique.numel() == 1, f"unexpected multiple reward values in main-only episode: {unique.tolist()}"
        t_val = float(team_rewards[ep_idx][team_flag].item())
        assert torch.allclose(unique[0], torch.tensor(t_val, dtype=per_step.dtype)), f"per-step rewards {unique[0]} differ from team_rewards[{team_flag}]={t_val}"
        start = end
        ep_idx += 1
    assert ep_idx == team_rewards.shape[0]


def test_collect_trajectory_serial_both_teams_rewards_balanced():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory(
        env,
        agent,
        horizon=40,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=True,
        main_seats=[0, 2],
        alternate_main_seats=False,
        episodes=10,
    )
    assert 'rew' in batch and 'done' in batch
    seats = batch['seat_team']
    per_ep = []
    start = 0
    for i, d in enumerate(batch['done']):
        if bool(d.item()):
            end = i + 1
            seats_ep = seats[start:end]
            rew_ep = batch['rew'][start:end]
            t0 = rew_ep[seats_ep[:, 4] > 0.5].sum()
            t1 = rew_ep[seats_ep[:, 5] > 0.5].sum()
            per_ep.append((float(t0.item()), float(t1.item())))
            # zero-sum se entrambi i team presenti
            assert abs(float((t0 + t1).item())) < 1e-5, f"episode rewards not zero-sum: t0={t0}, t1={t1}"
            start = end
    assert per_ep, "no episodes found in serial batch"
    assert any((abs(t0) > 0 or abs(t1) > 0) for (t0, t1) in per_ep), f"all episode rewards are zero: {per_ep}"


def test_collect_trajectory_score_sum_positive(monkeypatch):
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    score_sums = []
    original = train_mod._require_team_rewards

    def _wrapped(info, context):
        out = original(info, context)
        sb = info.get('score_breakdown', {})
        if isinstance(sb, dict) and (0 in sb or '0' in sb) and (1 in sb or '1' in sb):
            t0 = sb.get(0, sb.get('0'))
            t1 = sb.get(1, sb.get('1'))
            t0_total = float(t0.get('total', t0)) if isinstance(t0, dict) else float(t0)
            t1_total = float(t1.get('total', t1)) if isinstance(t1, dict) else float(t1)
            score_sums.append(t0_total + t1_total)
        return out

    monkeypatch.setattr(train_mod, "_require_team_rewards", _wrapped)
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
    assert batch['obs'].shape[0] > 0
    assert score_sums and all(s > 0 for s in score_sums)


