import os
import torch
import pytest

from trainers.train_ppo import collect_trajectory_parallel
import trainers.train_ppo as train_mod
from algorithms.ppo_ac import ActionConditionedPPO
from environment import ScoponeEnvMA


def _per_episode_sums(x: torch.Tensor, done: torch.Tensor):
    sums = []
    start = 0
    for i, d in enumerate(done):
        if bool(d.item()):
            sums.append(float(x[start:i + 1].sum().item()))
            start = i + 1
    return sums


def test_collect_trajectory_parallel_minimal_runs():
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory_parallel(
        agent,
        num_envs=2,
        episodes_total_hint=1,
        k_history=4,
        use_mcts=True,
        mcts_sims=8,
        mcts_dets=1,
        mcts_c_puct=1.0,
        mcts_root_temp=0.0,
        mcts_prior_smooth_eps=0.0,
        mcts_dirichlet_alpha=0.25,
        mcts_dirichlet_eps=0.25,
    )
    assert 'obs' in batch
    assert batch['obs'].shape[0] > 0
    assert 'chosen_index' in batch
    assert batch['chosen_index'].shape[0] == batch['obs'].shape[0]


def test_collect_trajectory_parallel_has_rewards_and_logp():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory_parallel(
        agent,
        num_envs=1,
        episodes_total_hint=10,
        k_history=4,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=False,
        main_seats=[0, 2],
        mcts_prior_smooth_eps=0.0,
        mcts_dirichlet_eps=0.0,
        mcts_min_sims=0,
        show_progress_env=False,
        tqdm_base_pos=0,
    )
    assert 'logp' in batch, "missing logp in parallel batch"
    assert batch['logp'].shape[0] == batch['obs'].shape[0]
    assert 'rew' in batch and batch['rew'].shape[0] == batch['obs'].shape[0]
    assert 'done' in batch and batch['done'].shape[0] == batch['obs'].shape[0]
    assert 'episode_scores' in batch, "missing episode_scores in parallel batch"
    scores = batch['episode_scores']
    scores_list = scores.tolist() if torch.is_tensor(scores) else list(scores)
    assert 'episode_team_rewards' in batch, "missing episode_team_rewards in parallel batch"
    team_rewards = batch['episode_team_rewards']
    team_rewards_list = team_rewards.tolist() if torch.is_tensor(team_rewards) else list(team_rewards)
    # train_both_teams=False: batch contiene solo main seat; richiedi almeno un episodio con reward non nullo
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
    assert per_ep, "no episodes found in parallel batch"
    assert len(scores_list) == len(per_ep) == len(team_rewards_list)
    print(f"[debug] parallel main-only episode score totals (score0,score1): {scores_list}")
    print(f"[debug] parallel main-only episode team_rewards (r0,r1): {team_rewards_list}")
    print(f"[debug] parallel main-only per-episode totals (t0,t1): {per_ep}")
    assert len(per_ep) == 10
    assert int(batch['done'].sum().item()) == len(per_ep)
    assert any((abs(t0) > 0 or abs(t1) > 0) for (t0, t1) in per_ep), f"all episode rewards are zero: {per_ep}"


def test_collect_trajectory_parallel_both_teams_rewards_balanced():
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
    env = ScoponeEnvMA(k_history=4)
    agent = ActionConditionedPPO(obs_dim=env.observation_space.shape[0])
    batch = collect_trajectory_parallel(
        agent,
        num_envs=1,
        episodes_total_hint=10,
        k_history=4,
        use_mcts=False,
        mcts_sims=0,
        mcts_dets=0,
        train_both_teams=True,
        main_seats=[0, 2],
        mcts_prior_smooth_eps=0.0,
        mcts_dirichlet_eps=0.0,
        mcts_min_sims=0,
        show_progress_env=False,
        tqdm_base_pos=0,
    )
    assert 'rew' in batch and 'done' in batch
    assert 'episode_scores' in batch, "missing episode_scores in parallel batch"
    scores = batch['episode_scores']
    scores_list = scores.tolist() if torch.is_tensor(scores) else list(scores)
    assert 'episode_team_rewards' in batch, "missing episode_team_rewards in parallel batch"
    team_rewards = batch['episode_team_rewards']
    team_rewards_list = team_rewards.tolist() if torch.is_tensor(team_rewards) else list(team_rewards)
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
    assert per_ep, "no episodes found in parallel batch (train_both_teams=True)"
    assert len(scores_list) == len(per_ep) == len(team_rewards_list)
    print(f"[debug] parallel both-teams episode score totals (score0,score1): {scores_list}")
    print(f"[debug] parallel both-teams episode team_rewards (r0,r1): {team_rewards_list}")
    print(f"[debug] parallel both-teams per-episode totals (t0,t1,sum): {[(t0, t1, t0 + t1) for (t0, t1) in per_ep]}")
    assert len(per_ep) == 10
    assert int(batch['done'].sum().item()) == len(per_ep)
    # zero-sum per episodio quando raccolti entrambi i team
    assert all(abs(t0 + t1) < 1e-5 for (t0, t1) in per_ep), f"episode rewards not zero-sum: {per_ep}"
    assert all((abs(t0) > 0 or abs(t1) > 0) for (t0, t1) in per_ep), f"all episode rewards are zero: {per_ep}"


def test_require_team_rewards_rejects_non_positive_score_sum_parallel():
    bad_info = {
        'team_rewards': [1.0, -1.0],
        'score_breakdown': {0: {'total': 0.0}, 1: {'total': 0.0}},
    }
    with pytest.raises(RuntimeError):
        train_mod._require_team_rewards(bad_info, "collect_trajectory_parallel")
