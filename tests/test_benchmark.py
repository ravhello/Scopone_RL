from tools.benchmark_ac import run_benchmark


def test_benchmark_runs_no_mcts():
    df, summary = run_benchmark(games=2, use_mcts=False, sims=8, dets=2, compact=True, k_history=4, ckpt_path='', seed=123)
    assert 'win_rate_team0' in summary
    assert df.shape[0] == 2


def test_benchmark_runs_with_mcts():
    df, summary = run_benchmark(games=2, use_mcts=True, sims=8, dets=2, compact=True, k_history=4, ckpt_path='', seed=123,
                                c_puct=1.0, root_temp=0.5, prior_smooth_eps=0.1, belief_particles=32, belief_ess_frac=0.4, robust_child=False,
                                root_dirichlet_alpha=0.3, root_dirichlet_eps=0.25)
    assert 'win_rate_team0' in summary
    assert df.shape[0] == 2

