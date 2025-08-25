import os


class FakeSummaryWriter:
    """
    Minimal fake SummaryWriter that captures add_scalar/add_text calls.
    Stores scalars grouped by step to validate per-iteration logging.
    """

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []  # list of (tag, value, step)
        self.texts = []    # list of (tag, text, step or None)
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((str(tag), float(value) if value is not None else None, int(step)))

    def add_text(self, tag, text, step=None):
        self.texts.append((str(tag), str(text), (None if step is None else int(step))))

    def close(self):
        self.closed = True


def _run_short_training(monkeypatch, num_iterations=3):
    """
    Run a very short PPO training on CPU with TB enabled and a fake writer.
    Returns: (writer, ended_steps)
    """
    # Ensure TB is enabled and execution is fast and deterministic enough for tests
    monkeypatch.setenv('SCOPONE_DISABLE_TB', '0')
    try:
        import torch
        if torch.cuda.is_available():
            monkeypatch.delenv('TESTS_FORCE_CPU', raising=False)
            monkeypatch.setenv('SCOPONE_DEVICE', 'cuda')
        else:
            monkeypatch.setenv('TESTS_FORCE_CPU', '1')
    except Exception:
        monkeypatch.setenv('TESTS_FORCE_CPU', '1')
    monkeypatch.setenv('ENV_DEVICE', 'cpu')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')

    # Patch SummaryWriter to our fake
    writer = FakeSummaryWriter(log_dir='runs/test')
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', lambda log_dir=None: writer, raising=True)

    from trainers.train_ppo import train_ppo

    ended_steps = []
    def on_iter_end(it):
        ended_steps.append(int(it))

    # Keep the run minimal to avoid long test times
    train_ppo(
        num_iterations=num_iterations,
        horizon=40,
        use_compact_obs=True,
        k_history=12,
        num_envs=1,
        mcts_sims=0,
        mcts_sims_eval=0,
        eval_every=0,
        mcts_in_eval=False,
        on_iter_end=on_iter_end,
    )

    return writer, ended_steps


def test_tb_writes_scalars_each_iteration(monkeypatch):
    """
    Validates that for a short training, at least one 'train/*' scalar is written
    for every iteration step that completes.
    """
    writer, ended_steps = _run_short_training(monkeypatch, num_iterations=3)

    # Steps that received at least one 'train/' scalar
    logged_steps = sorted({step for (tag, _val, step) in writer.scalars if tag.startswith('train/')})

    # We expect all completed iterations to have some 'train/' logs
    assert ended_steps == [0, 1, 2], f"Unexpected ended steps: {ended_steps}"
    assert logged_steps == [0, 1, 2], (
        f"Missing TensorBoard train/* logs for some iterations. Logged steps: {logged_steps};\n"
        f"All scalars: {writer.scalars}"
    )


def test_tb_behavior_when_collect_trajectory_errors(monkeypatch):
    """
    Injects an error on the second call to collect_trajectory to confirm that
    logging stops accordingly and to surface where the failure occurs.
    """
    monkeypatch.setenv('SCOPONE_DISABLE_TB', '0')
    try:
        import torch
        if torch.cuda.is_available():
            monkeypatch.delenv('TESTS_FORCE_CPU', raising=False)
            monkeypatch.setenv('SCOPONE_DEVICE', 'cuda')
        else:
            monkeypatch.setenv('TESTS_FORCE_CPU', '1')
    except Exception:
        monkeypatch.setenv('TESTS_FORCE_CPU', '1')
    monkeypatch.setenv('ENV_DEVICE', 'cpu')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')

    writer = FakeSummaryWriter(log_dir='runs/test')
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', lambda log_dir=None: writer, raising=True)

    import trainers.train_ppo as train_mod

    call_counter = {'n': 0}
    orig_collect = train_mod.collect_trajectory

    def flaky_collect(*args, **kwargs):
        call_counter['n'] += 1
        # Raise on the second collection (iteration 1, zero-based)
        if call_counter['n'] >= 2:
            raise RuntimeError('Injected failure in collect_trajectory')
        return orig_collect(*args, **kwargs)

    monkeypatch.setattr(train_mod, 'collect_trajectory', flaky_collect, raising=True)

    ended_steps = []
    def on_iter_end(it):
        ended_steps.append(int(it))

    error_raised = False
    try:
        train_mod.train_ppo(
            num_iterations=3,
            horizon=40,
            use_compact_obs=True,
            k_history=12,
            num_envs=1,
            mcts_sims=0,
            mcts_sims_eval=0,
            eval_every=0,
            mcts_in_eval=False,
            on_iter_end=on_iter_end,
        )
    except RuntimeError as e:
        error_raised = True
        # Provide context for debugging when this test fails
        assert 'Injected failure in collect_trajectory' in str(e)

    # Depending on internal exception handling, either the exception propagates
    # or the loop stops early. In both cases, ensure that only the first step
    # produced logs and later steps did not.
    logged_steps = sorted({step for (tag, _val, step) in writer.scalars if tag.startswith('train/')})
    assert logged_steps == [0], (
        f"Expected only first iteration to be logged due to injected error. Logged steps: {logged_steps};\n"
        f"All scalars: {writer.scalars}; Ended steps: {ended_steps}; Error raised: {error_raised}"
    )



