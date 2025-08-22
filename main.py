import os

# Silence TensorFlow/absl noise before any heavy imports
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # hide INFO/WARNING/ERROR from TF C++ logs
os.environ.setdefault('ABSL_LOGGING_MIN_LOG_LEVEL', '3')  # absl logs: only FATAL
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # disable oneDNN custom ops info spam
os.environ.setdefault('SCOPONE_DISABLE_TB', '1')  # default: don't import tensorboard

# Also configure absl handler levels for any TF/XLA components
try:
    import absl.logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.FATAL)
    _absl_logging.use_absl_handler()
    import logging as _py_logging
    _py_logging.getLogger('absl').setLevel(_py_logging.FATAL)
except Exception:
    pass

import torch
from utils.device import get_compute_device
from trainers.train_ppo import train_ppo

# Minimal entrypoint: launch PPO training only
if __name__ == "__main__":
    device = get_compute_device()
    print(f"Using device: {device}")
    train_ppo(num_iterations=10, horizon=256, use_compact_obs=True, k_history=39,
              num_envs=1,
              mcts_sims=0,
              mcts_sims_eval=4,
              eval_every=0,
              mcts_in_eval=True)


