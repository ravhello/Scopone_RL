import os
import torch
from trainers.train_ppo import train_ppo

# Minimal entrypoint: launch PPO training only
if __name__ == "__main__":
    device = torch.device(os.environ.get(
        'SCOPONE_DEVICE',
        ('cuda' if torch.cuda.is_available() and os.environ.get('TESTS_FORCE_CPU') != '1' else 'cpu')
    ))
    print(f"Using device: {device}")
    train_ppo(num_iterations=10, horizon=256, use_compact_obs=True, k_history=39, num_envs=1, mcts_sims=0) #, eval_every=5) , mcts_in_eval=False)


