# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os
import time
from tqdm import tqdm

# Import PyTorch Profiler for Chrome trace generation
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Advanced GPU Configuration
if torch.cuda.is_available():
    # Optimize for CUDA performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # For newer GPUs (Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Increase cache allocation size
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'memory'):
        torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use up to 95% of available memory

from environment import ScoponeEnvMA
from actions import decode_action

# Network and training parameters
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000    # total training steps to go from 1.0 to 0.01
BATCH_SIZE = 128          # mini-batch size
REPLAY_SIZE = 10000      # maximum replay buffer capacity
TARGET_UPDATE_FREQ = 1000  # how often to synchronize target network
CHECKPOINT_PATH = "checkpoints/scopone_checkpoint"

############################################################
# 1) Define EpisodicReplayBuffer class
############################################################
class EpisodicReplayBuffer:
    def __init__(self, capacity=20):  # Store up to 20 complete episodes
        self.episodes = collections.deque(maxlen=capacity)
        self.current_episode = []  # Current episode under construction
        
    def start_episode(self):
        """Start a new episode"""
        self.current_episode = []
        
    def add_transition(self, transition):
        """Add a transition to the current episode"""
        self.current_episode.append(transition)
        
    def end_episode(self):
        """End the current episode and add it to the buffer"""
        if self.current_episode:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
        
    def sample_episode(self):
        """Randomly sample a complete episode"""
        if not self.episodes:
            return []
        return random.choice(self.episodes)
    
    def sample_batch(self, batch_size):
        """Randomly sample a batch of transitions from all episodes"""
        all_transitions = []
        for episode in self.episodes:
            all_transitions.extend(episode)
        
        if len(all_transitions) < batch_size:
            return all_transitions
        
        batch = random.sample(all_transitions, batch_size)
        obs, actions, rewards, next_obs, dones, next_valids = zip(*batch)
        return (np.array(obs), actions, rewards, np.array(next_obs), dones, next_valids)
    
    def __len__(self):
        """Return the total number of transitions in all episodes"""
        return sum(len(episode) for episode in self.episodes) + len(self.current_episode)
    
    def get_all_episodes(self):
        """Return all episodes"""
        return list(self.episodes)
    
    def get_previous_episodes(self):
        """
        Returns all episodes except the most recent one.
        Useful for training on past episodes.
        """
        if len(self.episodes) <= 1:
            return []
        return list(self.episodes)[:-1]

############################################################
# 2) QNetwork neural network
############################################################
class QNetwork(nn.Module):
    def __init__(self, obs_dim=10823, action_dim=80):  # Updated from 10793 to 10823
        super().__init__()
        # Optimized feature extractor for advanced representation
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
            nn.Linear(334, 64),  # Updated from 304 to 334 (+30 for new values)
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
        
        # Move entire model to GPU at initialization
        self.to(device)
        
        # Set CUDA options for performance
        if torch.cuda.is_available():
            # Enable TF32 on Ampere GPUs for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Optimization for fixed input sizes
            torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        # Ensure input is on GPU - optimized
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
        
        # Process each section - in-place version
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
        
        # Process combined features - in-place version
        final_features = F.relu(self.combiner[0](combined), inplace=True)
        
        # Calculate action values
        action_values = self.action_head(final_features)
        
        return action_values

############################################################
# 3) DQNAgent with target network + replay
############################################################

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
        
        # Additions for GPU optimization
        torch.backends.cudnn.benchmark = True  # Optimization for fixed input sizes
        self.scaler = torch.amp.GradScaler('cuda')  # For mixed precision training
    
    def pick_action(self, obs, valid_actions, env):
        """GPU-optimized epsilon-greedy"""
        if not valid_actions:
            print("\n[DEBUG] No valid actions! Current state:")
            print("  Current player:", env.current_player)
            print("  Table:", env.game_state["table"])
            for p in range(4):
                print(f"  Hand p{p}:", env.game_state["hands"][p])
            print("  History:", env.game_state["history"])
            raise ValueError("No valid actions (valid_actions=[]).")
        
        # Epsilon-greedy: choose random action with probability epsilon
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
                    
                # OPTIMIZATION: Use mixed precision to speed up inference
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    action_values = self.online_qnet(obs_t)
                    q_values = torch.sum(action_values.view(1, 1, 80) * valid_actions_t.view(-1, 1, 80), dim=2).squeeze()
                
                best_action_idx = torch.argmax(q_values).item()
            
            return valid_actions[best_action_idx]
    
    def train_episodic_monte_carlo(self, specific_episode=None):
        """
        Optimized version with pre-allocated buffers and mixed precision.
        """
        # Determine which episodes to process
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            return  # No episodes available
        
        # OPTIMIZATION: Use pre-allocated buffers if possible
        max_transitions = sum(len(episode) for episode in episodes_to_process)
        
        # Create or resize buffers if necessary
        if not hasattr(self, 'train_obs_buffer') or max_transitions > self.train_obs_buffer.size(0):
            self.train_obs_buffer = torch.zeros((max_transitions, 10823), dtype=torch.float32, device=device)
            self.train_actions_buffer = torch.zeros((max_transitions, 80), dtype=torch.float32, device=device)
            self.train_returns_buffer = torch.zeros(max_transitions, dtype=torch.float32, device=device)
        
        # Fill buffers efficiently
        idx = 0
        for episode in episodes_to_process:
            if not episode:
                continue
                    
            # Get final reward from last transition of episode
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
        batch_size = min(512, idx)  # Optimized size for modern GPUs
        num_batches = (idx + batch_size - 1) // batch_size
        
        # Reduce sync_target frequency
        sync_counter = 0
        sync_frequency = 10  # Sync every 10 batches instead of every batch
        
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
                self.scaler.scale(loss).backward()
                
                # OPTIMIZATION: Clip gradients with moderate norm for training stability
                torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                
                # OPTIMIZATION: Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update epsilon after each batch to advance training
                self.update_epsilon()
                
                # OPTIMIZATION: Sync target network periodically (not every batch)
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    self.sync_target()
                    sync_counter = 0
    
    def store_episode_transition(self, transition):
        """
        Store a transition in the episodic buffer.
        """
        # Add to current episode
        self.episodic_buffer.add_transition(transition)
    
    def end_episode(self):
        """
        End the current episode WITHOUT training.
        Training must be explicitly called after this method.
        """
        self.episodic_buffer.end_episode()
        # Note: automatic training removed here
            
    def start_episode(self):
        """Start a new episode."""
        self.episodic_buffer.start_episode()
    
    def sync_target(self):
        """Synchronizes the target network weights with the online network weights."""
        self.target_qnet.load_state_dict(self.online_qnet.state_dict())

    def maybe_sync_target(self):
        """Syncs the target network with the online network every TARGET_UPDATE_FREQ steps."""
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.sync_target()

    def update_epsilon(self):
        """Updates epsilon for epsilon-greedy exploration based on training steps."""
        self.epsilon = max(EPSILON_END, EPSILON_START - (self.train_steps / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
        self.train_steps += 1

    def save_checkpoint(self, filename):
        """Save checkpoint ensuring directory exists"""
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
        """Load checkpoint considering GPU"""
        ckpt = torch.load(filename, map_location=device)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Checkpoint loaded from {filename}")

############################################################
# 4) Multi-agent training with profiling
############################################################

def train_agents(num_episodes=10):
    """
    Executes multi-agent episodic training with PyTorch profiling.
    Each episode will be profiled separately for better visualization.
    """
    # Create traces directory if it doesn't exist
    trace_dir = "traces"
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)
        print(f"Created trace directory: {trace_dir}")

    # Optimal GPU configuration
    if torch.cuda.is_available():
        # Settings to maximize throughput
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Aggressive memory management
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
        
        # Set allocator to reduce fragmentation
        torch.cuda.memory_stats()

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_dir):
        print(f"Creating checkpoint directory: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    
    # Create agents
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Utility function to find the most recent checkpoint
    def find_latest_checkpoint(base_path, team_id):
        """Find the most recent checkpoint for a specific team"""
        dir_path = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        
        # First check if checkpoint without episode number exists
        standard_ckpt = f"{base_path}_team{team_id}.pth"
        if os.path.isfile(standard_ckpt):
            return standard_ckpt
            
        # Otherwise look for checkpoints with episode number
        if os.path.exists(dir_path):
            # Pattern for checkpoints with episode number
            import fnmatch
            pattern = f"{base_name}_team{team_id}_ep*.pth"
            matching_files = [f for f in os.listdir(dir_path) if fnmatch.fnmatch(f, pattern)]
            
            if matching_files:
                # Extract episode number and sort by highest number
                matching_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]), reverse=True)
                return os.path.join(dir_path, matching_files[0])
        
        return None
    
    # Load checkpoints with improved logic
    print(f"Looking for most recent checkpoints for teams...")
    
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
    
    # OPTIMIZATION: Pre-allocate tensor buffers to avoid repeated allocations
    # Card buffer
    card_buffer = {}
    for suit in ['denari', 'coppe', 'spade', 'bastoni']:
        for rank in range(1, 11):
            card_buffer[(rank, suit)] = torch.zeros(80, dtype=torch.float32, device=device)
    
    # Training batch buffers
    max_transitions = 40  # Expected maximum transitions in an episode
    team0_obs_buffer = torch.zeros((max_transitions, 10823), dtype=torch.float32, device=device)
    team0_actions_buffer = torch.zeros((max_transitions, 80), dtype=torch.float32, device=device)
    team0_rewards_buffer = torch.zeros(max_transitions, dtype=torch.float32, device=device)
    
    team1_obs_buffer = torch.zeros_like(team0_obs_buffer)
    team1_actions_buffer = torch.zeros_like(team0_actions_buffer)
    team1_rewards_buffer = torch.zeros_like(team0_rewards_buffer)

    # Handle process-level profiler setup
    profiler_activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(ProfilerActivity.CUDA)

    # Define schedule for profiling
    # - wait: 1 step warmup for each episode
    # - active: profile for 3 steps
    # - repeat: repeat (only needed for the stepping sequence for one episode)
    # - skip: skip the rest of the steps that are not profiled
    prof_schedule = schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1
    )

    # Create a combined file for all episode traces
    combined_trace_path = os.path.join(trace_dir, "combined_trace.json")
    # Initialize an empty list to store all trace data
    all_trace_data = []
    
    # Create a progress bar
    pbar = tqdm(total=num_episodes, desc="Training episodes")
    
    # Main loop for episodes
    for ep in range(num_episodes):
        # Update progress bar with description that includes player info
        pbar.set_description(f"Episode {ep+1}/{num_episodes} (Player {first_player})")
        pbar.update(1)
        
        # Only profile the first 10 episodes
        do_profile = ep < 10
        
        if do_profile:
            # Set up a fresh profiler for this episode
            prof = profile(
                activities=profiler_activities, 
                schedule=prof_schedule,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True
            )
            prof.__enter__()
        
        episode_start_time = time.time()
        
        # Create environment and initialize
        env = ScoponeEnvMA()
        env.current_player = first_player

        # Initialize episodic buffers
        agent_team0.start_episode()
        agent_team1.start_episode()

        # Initial state
        done = False
        obs_current = env._get_observation(env.current_player)
        
        # OPTIMIZATION: Ensure obs_current is a numpy array
        if torch.is_tensor(obs_current):
            obs_current = obs_current.cpu().numpy()
            
        # Transition counts for each team
        team0_transitions = 0
        team1_transitions = 0

        # Main game loop
        inference_start = time.time()
        step_counter = 0
        
        while not done:
            # Record current player and team
            cp = env.current_player
            team_id = 0 if cp in [0,2] else 1
            agent = agent_team0 if team_id==0 else agent_team1

            # Get valid actions
            with record_function("get_valid_actions"):
                valid_acts = env.get_valid_actions()
            
            if not valid_acts:
                break
            
            # Action selection
            with record_function("pick_action"):
                # OPTIMIZATION: Efficient tensor conversion
                if len(valid_acts) > 0:
                    if isinstance(valid_acts[0], np.ndarray):
                        valid_acts_t = torch.tensor(np.stack(valid_acts), dtype=torch.float32, device=device)
                        # Optimized action selection
                        action = agent.pick_action(obs_current, valid_acts, env)
                    else:
                        # Fallback if valid_acts not already converted
                        action = agent.pick_action(obs_current, valid_acts, env)
            
            # Execute action on environment
            with record_function("environment_step"):
                next_obs, reward, done, info = env.step(action)
            
            # Ensure next_obs is numpy array
            if torch.is_tensor(next_obs):
                next_obs = next_obs.cpu().numpy()
                
            global_step += 1

            # Prepare transition
            next_valid = env.get_valid_actions() if not done else []
            transition = (obs_current, action, reward, next_obs, done, next_valid)
            
            # Store transition in buffer of current player's agent
            if team_id == 0:
                agent_team0.store_episode_transition(transition)
                team0_transitions += 1
            else:
                agent_team1.store_episode_transition(transition)
                team1_transitions += 1
            
            # Prepare for next iteration
            obs_current = next_obs
            
            # Step the profiler to move to next profiling window
            if do_profile:
                prof.step()
            
            step_counter += 1
        
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        # End episodes and prepare for training
        agent_team0.end_episode()
        agent_team1.end_episode()
        
        # Get final rewards
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
        
        # TRAINING AT THE END OF EPISODE
        with record_function("training_phase"):
            train_start_time = time.time()
            
            # Team 0 training
            with record_function("team0_training"):
                if agent_team0.episodic_buffer.episodes:
                    last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
                    if last_episode_team0:
                        # Extract episode data
                        all_obs0, all_actions0, _, _, _, _ = zip(*last_episode_team0)
                        
                        # OPTIMIZATION: Reuse pre-allocated buffers
                        ep_len = len(all_obs0)
                        if ep_len > max_transitions:
                            # Resize buffers if necessary
                            team0_obs_buffer.resize_(ep_len, 10823)
                            team0_actions_buffer.resize_(ep_len, 80)
                            team0_rewards_buffer.resize_(ep_len)
                        
                        # Batch data transfer to GPU
                        for i, (obs, action) in enumerate(zip(all_obs0, all_actions0)):
                            # Optimized direct conversion
                            if i < ep_len:
                                if isinstance(obs, np.ndarray):
                                    team0_obs_buffer[i].copy_(torch.tensor(obs, device=device))
                                else:
                                    team0_obs_buffer[i].copy_(obs)
                                    
                                if isinstance(action, np.ndarray):
                                    team0_actions_buffer[i].copy_(torch.tensor(action, device=device))
                                else:
                                    team0_actions_buffer[i].copy_(action)
                                    
                                team0_rewards_buffer[i] = team0_reward
                        
                        # Final batch with slicing
                        team0_batch = (
                            team0_obs_buffer[:ep_len], 
                            team0_actions_buffer[:ep_len], 
                            team0_rewards_buffer[:ep_len]
                        )
                        
                        # Team 0 training
                        # Get batch with optimal size
                        team0_obs_t, team0_actions_t, team0_rewards_t = team0_batch
                        
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            # Process in larger batches to better use GPU
                            batch_size = min(512, len(team0_obs_t))
                            num_batches = (len(team0_obs_t) + batch_size - 1) // batch_size
                            
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_size
                                end_idx = min(start_idx + batch_size, len(team0_obs_t))
                                
                                # Slices of tensors already on GPU
                                batch_obs_t = team0_obs_t[start_idx:end_idx]
                                batch_actions_t = team0_actions_t[start_idx:end_idx]
                                batch_returns_t = team0_rewards_t[start_idx:end_idx]
                                
                                # Efficient zero gradients
                                agent_team0.optimizer.zero_grad(set_to_none=True)
                                
                                # Optimized forward pass
                                q_values = agent_team0.online_qnet(batch_obs_t)
                                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                                
                                # Loss with numerical stability
                                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                                
                                # Backward with scaling
                                agent_team0.scaler.scale(loss).backward()
                                
                                # Gradient clipping for stability
                                torch.nn.utils.clip_grad_norm_(agent_team0.online_qnet.parameters(), max_norm=10.0)
                                
                                # Step with scaling
                                agent_team0.scaler.step(agent_team0.optimizer)
                                agent_team0.scaler.update()
                                
                                # Update epsilon
                                agent_team0.update_epsilon()
                                
                                # Step the profiler for training
                                if do_profile:
                                    prof.step()
            
            # Team 1 training
            with record_function("team1_training"):
                if agent_team1.episodic_buffer.episodes:
                    last_episode_team1 = agent_team1.episodic_buffer.episodes[-1]
                    if last_episode_team1:
                        all_obs1, all_actions1, _, _, _, _ = zip(*last_episode_team1)
                        
                        ep_len = len(all_obs1)
                        if ep_len > max_transitions:
                            team1_obs_buffer.resize_(ep_len, 10823)
                            team1_actions_buffer.resize_(ep_len, 80)
                            team1_rewards_buffer.resize_(ep_len)
                        
                        for i, (obs, action) in enumerate(zip(all_obs1, all_actions1)):
                            if i < ep_len:
                                if isinstance(obs, np.ndarray):
                                    team1_obs_buffer[i].copy_(torch.tensor(obs, device=device))
                                else:
                                    team1_obs_buffer[i].copy_(obs)
                                    
                                if isinstance(action, np.ndarray):
                                    team1_actions_buffer[i].copy_(torch.tensor(action, device=device))
                                else:
                                    team1_actions_buffer[i].copy_(action)
                                    
                                team1_rewards_buffer[i] = team1_reward
                        
                        team1_batch = (
                            team1_obs_buffer[:ep_len], 
                            team1_actions_buffer[:ep_len], 
                            team1_rewards_buffer[:ep_len]
                        )
                        
                        # Team 1 training
                        team1_obs_t, team1_actions_t, team1_rewards_t = team1_batch
                        
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            batch_size = min(512, len(team1_obs_t))
                            num_batches = (len(team1_obs_t) + batch_size - 1) // batch_size
                            
                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_size
                                end_idx = min(start_idx + batch_size, len(team1_obs_t))
                                
                                batch_obs_t = team1_obs_t[start_idx:end_idx]
                                batch_actions_t = team1_actions_t[start_idx:end_idx]
                                batch_returns_t = team1_rewards_t[start_idx:end_idx]
                                
                                agent_team1.optimizer.zero_grad(set_to_none=True)
                                
                                q_values = agent_team1.online_qnet(batch_obs_t)
                                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                                
                                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                                
                                agent_team1.scaler.scale(loss).backward()
                                torch.nn.utils.clip_grad_norm_(agent_team1.online_qnet.parameters(), max_norm=10.0)
                                agent_team1.scaler.step(agent_team1.optimizer)
                                agent_team1.scaler.update()
                                
                                agent_team1.update_epsilon()
                                
                                # Step the profiler for training
                                if do_profile:
                                    prof.step()
            
            # Target network synchronization
            if global_step % TARGET_UPDATE_FREQ == 0:
                with record_function("sync_target_networks"):
                    agent_team0.sync_target()
                    agent_team1.sync_target()
        
        # Track training time
        train_time = time.time() - train_start_time
        train_times.append(train_time)
        
        # Total episode time
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        
        # Close profiler for this episode and save trace
        if do_profile:
            prof.__exit__(None, None, None)
            
            # Export the profile trace to a file
            episode_trace_path = os.path.join(trace_dir, f"episode_{ep}_trace.json")
            prof.export_chrome_trace(episode_trace_path)
            print(f"Profile trace for episode {ep} saved to: {episode_trace_path}")
        
        # Save periodic checkpoints
        if (ep + 1) % 1000 == 0 or ep == num_episodes - 1:
            # Save with episode number to track progression
            agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_team0_ep{ep+1}.pth")
            agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_team1_ep{ep+1}.pth")
            
            # Also save without episode number for easy resumption
            agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_team0.pth")
            agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_team1.pth")
        
        # Prepare for next episode
        first_player = (first_player + 1) % 4
    
    # Close progress bar
    pbar.close()
    
    # Create a simple HTML file that combines all the trace files for easy viewing
    html_file_path = os.path.join(trace_dir, "view_traces.html")
    with open(html_file_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Traces</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1 { color: #333; }
                ul { list-style-type: none; padding: 0; }
                li { margin-bottom: 10px; }
                a { text-decoration: none; color: #0066cc; }
                a:hover { text-decoration: underline; }
                .button { 
                    display: inline-block; 
                    background-color: #4CAF50; 
                    color: white; 
                    padding: 10px 15px; 
                    margin: 5px 0; 
                    border: none; 
                    border-radius: 4px;
                    cursor: pointer;
                }
                .button:hover { background-color: #45a049; }
            </style>
        </head>
        <body>
            <h1>Training Traces</h1>
            <p>Click on a trace file to view it in Chrome's trace viewer: <code>chrome://tracing</code></p>
            <p>
                <a class="button" href="chrome://tracing" target="_blank">Open Chrome Tracing</a>
            </p>
            <ul>
        """)
        
        # List all the trace files
        trace_files = [f for f in os.listdir(trace_dir) if f.endswith('_trace.json')]
        for trace_file in sorted(trace_files):
            f.write(f'<li><a href="{trace_file}" download>{trace_file}</a></li>\n')
        
        f.write("""
            </ul>
            <script>
                document.querySelectorAll('a[download]').forEach(link => {
                    link.addEventListener('click', function(e) {
                        e.preventDefault();
                        const filename = this.getAttribute('href');
                        const fullPath = location.href.substring(0, location.href.lastIndexOf('/') + 1) + filename;
                        
                        // Create instructions for the user
                        alert('Download the file, then open chrome://tracing in a new tab and load the downloaded file.');
                        
                        // Trigger download
                        const a = document.createElement('a');
                        a.href = filename;
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    });
                });
            </script>
        </body>
        </html>
        """)
    
    print(f"Created HTML viewer for traces: {html_file_path}")
    print(f"To view traces, open {html_file_path} in your browser")

    # Verify checkpoints were saved
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if os.path.exists(checkpoint_dir):
        print("\nCheckpoints found in directory:")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        for file in checkpoint_files:
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # size in MB
            print(f" - {file} ({file_size:.2f} MB)")
    else:
        print(f"\nWARNING: Checkpoint directory {checkpoint_dir} does not exist!")
    
    # Final statistics report
    print("\n=== Training complete ===")
    print("\nPerformance summary:")
    print(f"Average time per episode: {sum(episode_times)/len(episode_times):.2f}s")
    print(f"Average time for training: {sum(train_times)/len(train_times):.2f}s")
    print(f"Average time for inference: {sum(inference_times)/len(inference_times):.2f}s")
    print(f"Percentage of time in training: {sum(train_times)/sum(episode_times)*100:.1f}%")
    print(f"Percentage of time in inference: {sum(inference_times)/sum(episode_times)*100:.1f}%")
    
    # Print final epsilons
    print(f"Final epsilon team 0: {agent_team0.epsilon:.4f}")
    print(f"Final epsilon team 1: {agent_team1.epsilon:.4f}")


if __name__ == "__main__":
    # Run training with specified number of episodes
    train_agents(num_episodes=2000)