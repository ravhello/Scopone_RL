#!/usr/bin/env python
# profile_train.py - Standalone profiling script

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os
import time
from tqdm import tqdm
import gc
import fnmatch
from pathlib import Path

# PyTorch profiler imports
import torch.profiler as prof
from torch.profiler import profile, record_function, ProfilerActivity

# Import your custom modules
from environment import ScoponeEnvMA
from state import initialize_game
from observation import encode_state_for_player
from actions import get_valid_actions, encode_action, decode_action
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU configuration
if torch.cuda.is_available():
    # Optimize for CUDA performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # For newer GPUs (Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Increase cache allocation size
    torch.cuda.empty_cache()
    torch.cuda.memory.set_per_process_memory_fraction(0.95)

# Training parameters
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
BATCH_SIZE = 128
REPLAY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
CHECKPOINT_PATH = "checkpoints/scopone_checkpoint"

# Create profile directory
PROFILE_DIR = "./profile_results"
Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)


class EpisodicReplayBuffer:
    def __init__(self, capacity=20):
        self.episodes = collections.deque(maxlen=capacity)
        self.current_episode = []
        
    def start_episode(self):
        self.current_episode = []
        
    def add_transition(self, transition):
        self.current_episode.append(transition)
        
    def end_episode(self):
        if self.current_episode:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
        
    def sample_episode(self):
        if not self.episodes:
            return []
        return random.choice(self.episodes)
    
    def sample_batch(self, batch_size):
        all_transitions = []
        for episode in self.episodes:
            all_transitions.extend(episode)
        
        if len(all_transitions) < batch_size:
            return all_transitions
        
        batch = random.sample(all_transitions, batch_size)
        obs, actions, rewards, next_obs, dones, next_valids = zip(*batch)
        return (np.array(obs), actions, rewards, np.array(next_obs), dones, next_valids)
    
    def __len__(self):
        return sum(len(episode) for episode in self.episodes) + len(self.current_episode)
    
    def get_all_episodes(self):
        return list(self.episodes)
    
    def get_previous_episodes(self):
        if len(self.episodes) <= 1:
            return []
        return list(self.episodes)[:-1]


class QNetwork(nn.Module):
    def __init__(self, obs_dim=10823, action_dim=80):
        super().__init__()
        # Feature extractor optimized for advanced representation
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Additional sections
        self.hand_table_processor = nn.Sequential(
            nn.Linear(83, 64),
            nn.ReLU()
        )
        
        self.captured_processor = nn.Sequential(
            nn.Linear(82, 64),
            nn.ReLU()
        )
        
        self.stats_processor = nn.Sequential(
            nn.Linear(334, 64),
            nn.ReLU()
        )
        
        self.history_processor = nn.Sequential(
            nn.Linear(10320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(128 + 64*4, 256),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(256, action_dim)
        
        # Move model to GPU
        self.to(device)
        
        # Set CUDA options for performance
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        # Ensure input is on GPU
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.device != device:
            x = x.to(device)
                
        import torch.nn.functional as F
        # Use inplace ReLU to save memory
        x1 = F.relu(self.backbone[0](x), inplace=True)
        x2 = F.relu(self.backbone[2](x1), inplace=True)
        x3 = F.relu(self.backbone[4](x2), inplace=True)
        backbone_features = F.relu(self.backbone[6](x3), inplace=True)
        
        # Split input into semantic sections
        hand_table = x[:, :83]
        captured = x[:, 83:165]
        history = x[:, 169:10489]
        stats = x[:, 10489:]
        
        # Process each section - inplace version
        hand_table_features = F.relu(self.hand_table_processor[0](hand_table), inplace=True)
        captured_features = F.relu(self.captured_processor[0](captured), inplace=True)
        history_features = F.relu(self.history_processor[0](history), inplace=True)
        history_features = F.relu(self.history_processor[2](history_features), inplace=True)
        stats_features = F.relu(self.stats_processor[0](stats), inplace=True)
        
        # Combine all features
        combined = torch.cat([
            backbone_features,
            hand_table_features,
            captured_features,
            history_features,
            stats_features
        ], dim=1)
        
        # Process combined features - inplace version
        final_features = F.relu(self.combiner[0](combined), inplace=True)
        
        # Calculate action values
        action_values = self.action_head(final_features)
        
        return action_values


class DQNAgent:
    def __init__(self, team_id):
        self.team_id = team_id
        self.online_qnet = QNetwork()
        self.target_qnet = QNetwork()
        self.sync_target()
        
        self.optimizer = optim.Adam(self.online_qnet.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.train_steps = 0
        self.episodic_buffer = EpisodicReplayBuffer()
        
        # GPU optimization
        torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    def pick_action(self, obs, valid_actions, env):
        if not valid_actions:
            print("\n[DEBUG] No valid actions! Current state:")
            print("  Current player:", env.current_player)
            print("  Table:", env.game_state["table"])
            for p in range(4):
                print(f"  Hand p{p}:", env.game_state["hands"][p])
            print("  History:", env.game_state["history"])
            raise ValueError("No valid actions (valid_actions=[]).")
        
        # Epsilon-greedy: choose random action with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # OPTIMIZATION: Convert all inputs to GPU tensors in a single operation
            # and reuse pre-allocated buffer if available
            if hasattr(self, 'valid_actions_buffer') and len(valid_actions) <= self.valid_actions_buffer.size(0):
                valid_actions_t = self.valid_actions_buffer[:len(valid_actions)]
                for i, va in enumerate(valid_actions):
                    if isinstance(va, np.ndarray):
                        valid_actions_t[i].copy_(torch.tensor(va, device=device))
                    else:
                        valid_actions_t[i].copy_(va)
            else:
                # Create buffer if it doesn't exist
                if not hasattr(self, 'valid_actions_buffer') or len(valid_actions) > self.valid_actions_buffer.size(0):
                    self.valid_actions_buffer = torch.zeros((max(100, len(valid_actions)), 80), 
                                                        dtype=torch.float32, device=device)
                valid_actions_t = torch.tensor(np.stack(valid_actions), 
                                            dtype=torch.float32, device=device)
                
            with torch.no_grad():
                # OPTIMIZATION: Reuse observation buffer if possible
                if hasattr(self, 'obs_buffer'):
                    obs_t = self.obs_buffer
                    if isinstance(obs, np.ndarray):
                        obs_t.copy_(torch.tensor(obs, device=device).unsqueeze(0))
                    else:
                        obs_t.copy_(obs.unsqueeze(0))
                else:
                    self.obs_buffer = torch.zeros((1, len(obs)), dtype=torch.float32, device=device)
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                # OPTIMIZATION: Use mixed precision for faster inference
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    action_values = self.online_qnet(obs_t)
                    q_values = torch.sum(action_values.view(1, 1, 80) * valid_actions_t.view(-1, 1, 80), dim=2).squeeze()
                
                best_action_idx = torch.argmax(q_values).item()
            
            return valid_actions[best_action_idx]
    
    def train_episodic_monte_carlo(self, specific_episode=None):
        # Determine which episodes to process
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            return  # No episodes available
        
        # OPTIMIZATION: Use pre-allocated buffers if possible
        max_transitions = sum(len(episode) for episode in episodes_to_process)
        
        # Create or resize buffers if needed
        if not hasattr(self, 'train_obs_buffer') or max_transitions > self.train_obs_buffer.size(0):
            self.train_obs_buffer = torch.zeros((max_transitions, 10823), dtype=torch.float32, device=device)
            self.train_actions_buffer = torch.zeros((max_transitions, 80), dtype=torch.float32, device=device)
            self.train_returns_buffer = torch.zeros(max_transitions, dtype=torch.float32, device=device)
        
        # Fill buffers efficiently
        idx = 0
        for episode in episodes_to_process:
            if not episode:
                continue
                    
            # Get final reward from last transition
            final_reward = episode[-1][2] if episode else 0.0
            
            for obs, action, _, _, _, _ in episode:
                # Copy directly to buffer to avoid intermediate tensor creation
                if isinstance(obs, np.ndarray):
                    self.train_obs_buffer[idx].copy_(torch.tensor(obs, device=device))
                else:
                    self.train_obs_buffer[idx].copy_(obs)
                    
                if isinstance(action, np.ndarray):
                    self.train_actions_buffer[idx].copy_(torch.tensor(action, device=device))
                else:
                    self.train_actions_buffer[idx].copy_(action)
                    
                self.train_returns_buffer[idx] = final_reward
                idx += 1
        
        if idx == 0:
            return  # No transitions to process
        
        # Use buffer slices for training
        all_obs_t = self.train_obs_buffer[:idx]
        all_actions_t = self.train_actions_buffer[:idx]
        all_returns_t = self.train_returns_buffer[:idx]
        
        # Increase batch_size to better utilize GPU
        batch_size = min(512, idx)
        num_batches = (idx + batch_size - 1) // batch_size
        
        # Reduce target sync frequency
        sync_counter = 0
        sync_frequency = 10
        
        # Track metrics for diagnostics
        total_loss = 0.0
        batch_count = 0
        
        # OPTIMIZATION: Use mixed precision more efficiently with float16
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, idx)
                
                # Take slices of tensors already on GPU (avoid copies)
                batch_obs_t = all_obs_t[start_idx:end_idx]
                batch_actions_t = all_actions_t[start_idx:end_idx]
                batch_returns_t = all_returns_t[start_idx:end_idx]
                
                # OPTIMIZATION: Zero gradients - use set_to_none=True for better memory efficiency
                self.optimizer.zero_grad(set_to_none=True)
                
                # OPTIMIZATION: Forward pass with kernel fusion where possible
                q_values = self.online_qnet(batch_obs_t)
                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                
                # OPTIMIZATION: Loss with mixed precision - use reduction='mean' for numerical stability
                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                
                # Track loss for diagnostics
                total_loss += loss.item()
                batch_count += 1
                
                # OPTIMIZATION: Backward and optimizer step with gradient scaling for mixed precision
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # OPTIMIZATION: Clip gradient with moderate norm for training stability
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    
                    # OPTIMIZATION: Optimizer step with scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    self.optimizer.step()
                
                # Update epsilon after each batch to advance training
                self.update_epsilon()
                
                # OPTIMIZATION: Sync target network periodically (not every batch)
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    self.sync_target()
                    sync_counter = 0
                    
                # OPTIMIZATION: Release GPU memory periodically
                if batch_idx % 10 == 0 and batch_idx > 0:
                    torch.cuda.empty_cache()
    
    def store_episode_transition(self, transition):
        # Add to current episode
        self.episodic_buffer.add_transition(transition)
    
    def end_episode(self):
        # End current episode WITHOUT training
        # Training must be called explicitly after this method
        self.episodic_buffer.end_episode()
            
    def start_episode(self):
        self.episodic_buffer.start_episode()
    
    def sync_target(self):
        self.target_qnet.load_state_dict(self.online_qnet.state_dict())

    def maybe_sync_target(self):
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.sync_target()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, EPSILON_START - (self.train_steps / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
        self.train_steps += 1

    def save_checkpoint(self, filename):
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[DQNAgent] Created checkpoint directory: {directory}")
        
        try:
            torch.save({
                "online_state_dict": self.online_qnet.state_dict(),
                "target_state_dict": self.target_qnet.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps
            }, filename)
            print(f"[DQNAgent] Checkpoint saved: {filename}")
        except Exception as e:
            print(f"[DQNAgent] ERROR saving checkpoint {filename}: {e}")

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename, map_location=device)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Checkpoint loaded from {filename}")


def find_latest_checkpoint(base_path, team_id):
    """Find the latest checkpoint for a specific team"""
    dir_path = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    
    # First check if standard checkpoint exists
    standard_ckpt = f"{base_path}_team{team_id}.pth"
    if os.path.isfile(standard_ckpt):
        return standard_ckpt
        
    # Otherwise search for checkpoints with episode number
    if os.path.exists(dir_path):
        pattern = f"{base_name}_team{team_id}_ep*.pth"
        matching_files = [f for f in os.listdir(dir_path) if fnmatch.fnmatch(f, pattern)]
        
        if matching_files:
            matching_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]), reverse=True)
            return os.path.join(dir_path, matching_files[0])
    
    return None


def save_profile_summary(profiler, name="profile_summary"):
    """Save detailed summary of profiling results"""
    import pandas as pd
    
    # Export key averages to CSV
    events = profiler.key_averages()
    data = []
    
    for evt in events:
        # Skip tiny operations that pollute the data
        if evt.cpu_time_total < 10 and evt.cuda_time_total < 10:
            continue
            
        data.append({
            'Name': evt.key,
            'CPU Time (μs)': evt.cpu_time_total,
            'CUDA Time (μs)': evt.cuda_time_total,
            'Self CPU Time (μs)': evt.self_cpu_time_total,
            'Self CUDA Time (μs)': evt.self_cuda_time_total,
            'CPU Memory Used (B)': evt.cpu_memory_usage,
            'CUDA Memory Used (B)': evt.cuda_memory_usage,
            'Calls': evt.count
        })
    
    df = pd.DataFrame(data)
    df.to_csv(f"{PROFILE_DIR}/{name}.csv", index=False)
    
    # Calculate CPU/GPU time breakdown
    total_cpu_time = sum(evt.cpu_time_total for evt in events)
    total_cuda_time = sum(evt.cuda_time_total for evt in events)
    
    # Find memcpy operations (data transfers)
    memcpy_ops = [evt for evt in events if 'memcpy' in evt.key.lower()]
    memcpy_time = sum(evt.cuda_time_total for evt in memcpy_ops)
    
    # Save summary text file
    with open(f"{PROFILE_DIR}/{name}_overview.txt", 'w') as f:
        f.write("PROFILING SUMMARY\n")
        f.write("================\n\n")
        f.write(f"Total CPU Time: {total_cpu_time/1000:.2f} ms\n")
        f.write(f"Total CUDA Time: {total_cuda_time/1000:.2f} ms\n")
        f.write(f"Data Transfer Time: {memcpy_time/1000:.2f} ms ({memcpy_time/total_cuda_time*100 if total_cuda_time else 0:.1f}% of CUDA time)\n\n")
        
        f.write("TOP CPU OPERATIONS\n")
        f.write("=================\n")
        cpu_events = sorted(events, key=lambda x: x.self_cpu_time_total, reverse=True)[:20]
        for i, evt in enumerate(cpu_events):
            f.write(f"{i+1}. {evt.key.split('/')[-1]} - {evt.self_cpu_time_total/1000:.2f} ms ({evt.count} calls)\n")
        
        f.write("\nTOP CUDA OPERATIONS\n")
        f.write("==================\n")
        cuda_events = sorted(events, key=lambda x: x.self_cuda_time_total, reverse=True)[:20]
        for i, evt in enumerate(cuda_events):
            f.write(f"{i+1}. {evt.key.split('/')[-1]} - {evt.self_cuda_time_total/1000:.2f} ms ({evt.count} calls)\n")
    
    print(f"Saved profile summary to {PROFILE_DIR}/{name}.csv")
    print(f"Saved profile overview to {PROFILE_DIR}/{name}_overview.txt")
    
    return df


def profiled_train_agents(num_episodes=200):
    """
    Profiled version of the training function, limited to the specified number of episodes
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_dir):
        print(f"Creating checkpoint directory: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    
    # Create the agents
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Look for existing checkpoints
    print(f"Looking for latest checkpoints...")
    
    # Team 0
    team0_ckpt = find_latest_checkpoint(CHECKPOINT_PATH, 0)
    if team0_ckpt:
        try:
            print(f"Found checkpoint for team 0: {team0_ckpt}")
            agent_team0.load_checkpoint(team0_ckpt)
        except Exception as e:
            print(f"ERROR loading team 0 checkpoint: {e}")
    else:
        print(f"No checkpoint found for team 0")
    
    # Team 1
    team1_ckpt = find_latest_checkpoint(CHECKPOINT_PATH, 1)
    if team1_ckpt:
        try:
            print(f"Found checkpoint for team 1: {team1_ckpt}")
            agent_team1.load_checkpoint(team1_ckpt)
        except Exception as e:
            print(f"ERROR loading team 1 checkpoint: {e}")
    else:
        print(f"No checkpoint found for team 1")

    # Control variables
    first_player = 0
    global_step = 0
    
    # Performance monitoring
    episode_times = []
    train_times = []
    inference_times = []
    
    # Create profiler
    profiler_activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(ProfilerActivity.CUDA)
    
    # Set up the profiler
    profiler = profile(
        activities=profiler_activities,
        schedule=prof.schedule(wait=1, warmup=1, active=num_episodes-2, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler.__enter__()
    
    # Progress bar
    pbar = tqdm(total=num_episodes, desc="Profiling episodes")
    
    # Main loop for episodes
    for ep in range(num_episodes):
        episode_start_time = time.time()
        
        # Update progress
        pbar.set_description(f"Episode {ep+1}/{num_episodes} (Player {first_player})")
        pbar.update(1)
        
        # Create environment and initialize
        with record_function("Environment_Setup"):
            env = ScoponeEnvMA()
            env.current_player = first_player

        # Initialize episode buffers
        with record_function("Initialize_Episode"):
            agent_team0.start_episode()
            agent_team1.start_episode()

        # Initial state
        with record_function("Initial_Observation"):
            obs_current = env._get_observation(env.current_player)
            done = False
        
        # Count transitions per team
        team0_transitions = 0
        team1_transitions = 0

        # Game loop
        with record_function("Game_Loop"):
            inference_start = time.time()
            step_counter = 0
            
            while not done:
                step_counter += 1
                with record_function(f"Game_Step_{step_counter}"):
                    cp = env.current_player
                    team_id = 0 if cp in [0,2] else 1
                    agent = agent_team0 if team_id==0 else agent_team1
                    
                    # Get valid actions
                    with record_function("Get_Valid_Actions"):
                        valid_acts = env.get_valid_actions()
                    
                    if not valid_acts:
                        break
                    
                    # Choose action
                    with record_function("Pick_Action"):
                        action = agent.pick_action(obs_current, valid_acts, env)
                    
                    # Execute action
                    with record_function("Environment_Step"):
                        next_obs, reward, done, info = env.step(action)
                    
                    # Store transition
                    with record_function("Store_Transition"):
                        next_valid = env.get_valid_actions() if not done else []
                        transition = (obs_current, action, reward, next_obs, done, next_valid)
                        
                        if team_id == 0:
                            agent_team0.store_episode_transition(transition)
                            team0_transitions += 1
                        else:
                            agent_team1.store_episode_transition(transition)
                            team1_transitions += 1
                    
                    global_step += 1
                    
                    # Prepare for next iteration
                    obs_current = next_obs
        
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        # End episodes
        with record_function("End_Episodes"):
            agent_team0.end_episode()
            agent_team1.end_episode()
        
        # Get final rewards
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
        
        # Training
        with record_function("Training"):
            train_start_time = time.time()
            
            # Team 0 training
            with record_function("Team0_Training"):
                if agent_team0.episodic_buffer.episodes:
                    last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
                    if last_episode_team0:
                        agent_team0.train_episodic_monte_carlo()
            
            # Team 1 training
            with record_function("Team1_Training"):
                if agent_team1.episodic_buffer.episodes:
                    last_episode_team1 = agent_team1.episodic_buffer.episodes[-1]
                    if last_episode_team1:
                        agent_team1.train_episodic_monte_carlo()
            
            # Target network sync
            if global_step % TARGET_UPDATE_FREQ == 0:
                with record_function("Sync_Target_Networks"):
                    agent_team0.sync_target()
                    agent_team1.sync_target()
            
            train_time = time.time() - train_start_time
            train_times.append(train_time)
        
        # Memory cleanup
        with record_function("Memory_Cleanup"):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Episode time
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        
        # Save checkpoint periodically
        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            with record_function("Save_Checkpoints"):
                agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_profile_team0_ep{ep+1}.pth")
                agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_profile_team1_ep{ep+1}.pth")
        
        # Step the profiler
        profiler.step()
        
        # Next player
        first_player = (first_player + 1) % 4
    
    # End profiler
    profiler.__exit__(None, None, None)
    
    # Close progress bar
    pbar.close()
    
    # Save profiler results
    try:
        profiler.export_chrome_trace(f"{PROFILE_DIR}/trace.json")
        print(f"Exported Chrome trace to {PROFILE_DIR}/trace.json")
        print("You can view this trace in Chrome by navigating to chrome://tracing")
    except Exception as e:
        print(f"Error exporting trace: {e}")
    
    # Save tabular summary
    save_profile_summary(profiler)
    
    # Generate performance report
    avg_episode_time = sum(episode_times) / len(episode_times)
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_train_time = sum(train_times) / len(train_times)
    
    print("\n=== Performance Report ===")
    print(f"Average episode time: {avg_episode_time:.3f}s")
    print(f"Average inference time: {avg_inference_time:.3f}s ({avg_inference_time/avg_episode_time*100:.1f}% of episode)")
    print(f"Average training time: {avg_train_time:.3f}s ({avg_train_time/avg_episode_time*100:.1f}% of episode)")
    
    # Save performance metrics
    with open(f"{PROFILE_DIR}/performance_metrics.txt", "w") as f:
        f.write("PERFORMANCE METRICS\n")
        f.write("===================\n\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Total steps: {global_step}\n\n")
        
        f.write(f"Average episode time: {avg_episode_time:.3f}s\n")
        f.write(f"Average inference time: {avg_inference_time:.3f}s ({avg_inference_time/avg_episode_time*100:.1f}% of episode)\n")
        f.write(f"Average training time: {avg_train_time:.3f}s ({avg_train_time/avg_episode_time*100:.1f}% of episode)\n\n")
        
        f.write("GPU MEMORY USAGE\n")
        f.write("===============\n")
        if torch.cuda.is_available():
            f.write(f"Peak memory allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB\n")
            f.write(f"Peak memory reserved: {torch.cuda.max_memory_reserved()/1024**2:.1f}MB\n")
            f.write(f"Current memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB\n")
            f.write(f"Current memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB\n")
        else:
            f.write("No GPU available\n")
    
    print(f"Saved performance metrics to {PROFILE_DIR}/performance_metrics.txt")
    
    return profiler


def create_visualization(profile_dir=PROFILE_DIR):
    """Create visualizations from profiling data"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Check if visualization dependencies are available
    try:
        import pandas as pd
        import matplotlib
        import seaborn
    except ImportError:
        print("Visualization libraries not available. Install pandas, matplotlib, and seaborn.")
        return
    
    # Check if profile data exists
    csv_path = f"{profile_dir}/profile_summary.csv"
    if not os.path.exists(csv_path):
        print(f"Profile data not found at {csv_path}")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create visualizations directory
    viz_dir = f"{profile_dir}/visualizations"
    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid", font_scale=1.1)
    
    # 1. Top CPU Operations
    plt.figure(figsize=(14, 8))
    top_cpu = df.sort_values('Self CPU Time (μs)', ascending=False).head(15)
    top_cpu['Name'] = top_cpu['Name'].str.split('/').str[-1]  # Simplify names
    sns.barplot(x='Self CPU Time (μs)', y='Name', data=top_cpu, palette='Blues_d')
    plt.title('Top 15 Operations by CPU Time', fontsize=16)
    plt.xlabel('Self CPU Time (μs)', fontsize=14)
    plt.ylabel('Operation', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/top_cpu_operations.png", dpi=300)
    plt.close()
    
    # 2. Top CUDA Operations
    plt.figure(figsize=(14, 8))
    top_cuda = df.sort_values('Self CUDA Time (μs)', ascending=False).head(15)
    top_cuda['Name'] = top_cuda['Name'].str.split('/').str[-1]  # Simplify names
    sns.barplot(x='Self CUDA Time (μs)', y='Name', data=top_cuda, palette='Reds_d')
    plt.title('Top 15 Operations by CUDA Time', fontsize=16)
    plt.xlabel('Self CUDA Time (μs)', fontsize=14)
    plt.ylabel('Operation', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/top_cuda_operations.png", dpi=300)
    plt.close()
    
    # 3. Memory Usage by Operation
    plt.figure(figsize=(14, 8))
    top_mem = df.sort_values('CUDA Memory Used (B)', ascending=False).head(15)
    top_mem['Name'] = top_mem['Name'].str.split('/').str[-1]  # Simplify names
    top_mem['CUDA Memory (MB)'] = top_mem['CUDA Memory Used (B)'] / (1024 * 1024)
    sns.barplot(x='CUDA Memory (MB)', y='Name', data=top_mem, palette='Greens_d')
    plt.title('Top 15 Operations by CUDA Memory Usage', fontsize=16)
    plt.xlabel('CUDA Memory (MB)', fontsize=14)
    plt.ylabel('Operation', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/top_memory_operations.png", dpi=300)
    plt.close()
    
    # 4. CPU vs CUDA Time Comparison
    plt.figure(figsize=(14, 8))
    comp_df = df.sort_values('Self CUDA Time (μs)', ascending=False).head(10).copy()
    comp_df['Name'] = comp_df['Name'].str.split('/').str[-1]  # Simplify names
    comp_df['Self CPU Time (ms)'] = comp_df['Self CPU Time (μs)'] / 1000
    comp_df['Self CUDA Time (ms)'] = comp_df['Self CUDA Time (μs)'] / 1000
    
    # Reshape for grouped bar chart
    comp_melted = pd.melt(comp_df, 
                          id_vars=['Name'], 
                          value_vars=['Self CPU Time (ms)', 'Self CUDA Time (ms)'],
                          var_name='Timing Type', 
                          value_name='Time (ms)')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Name', y='Time (ms)', hue='Timing Type', data=comp_melted, palette='Set1')
    plt.title('CPU vs CUDA Time for Top Operations', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Operation', fontsize=14)
    plt.ylabel('Time (ms)', fontsize=14)
    plt.legend(title='', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/cpu_vs_cuda_comparison.png", dpi=300)
    plt.close()
    
    # 5. Most Called Operations
    plt.figure(figsize=(14, 8))
    top_calls = df.sort_values('Calls', ascending=False).head(15)
    top_calls['Name'] = top_calls['Name'].str.split('/').str[-1]  # Simplify names
    sns.barplot(x='Calls', y='Name', data=top_calls, palette='Purples_d')
    plt.title('Top 15 Most Frequently Called Operations', fontsize=16)
    plt.xlabel('Number of Calls', fontsize=14)
    plt.ylabel('Operation', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/most_called_operations.png", dpi=300)
    plt.close()
    
    print(f"Created visualizations in {viz_dir}")


if __name__ == "__main__":
    # Number of episodes to profile
    NUM_EPISODES = 200
    
    print(f"Starting profiled training for {NUM_EPISODES} episodes...")
    
    # Run profiled training
    start_time = time.time()
    profile_result = profiled_train_agents(NUM_EPISODES)
    total_time = time.time() - start_time
    
    print(f"Profiling completed in {total_time:.2f} seconds")
    print(f"Results saved to {PROFILE_DIR}")
    
    # Create visualizations
    try:
        print("Creating visualizations...")
        create_visualization()
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print("Done!")