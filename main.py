import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os

from environment import ScoponeEnvMA
from actions import encode_action, get_valid_actions, MAX_ACTIONS
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

# Parametri di rete e training
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000  
BATCH_SIZE = 32
REPLAY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
CHECKPOINT_PATH = "scopone_checkpoint"

############################################################
# 1) ReplayBuffer
############################################################
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, transition):
        # transition = (obs, action, reward, next_obs, valid_next)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, valid_next = zip(*batch)
        return (list(obs), list(actions), list(rewards), list(next_obs), list(valid_next))

    def __len__(self):
        return len(self.buffer)

############################################################
# 2) QNetwork
############################################################
class QNetwork(nn.Module):
    def __init__(self, obs_dim=3764, act_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, act_dim)
        )
    def forward(self, x):
        return self.net(x)

############################################################
# 3) DQNAgent
############################################################
class DQNAgent:
    def __init__(self, team_id):
        self.team_id = team_id
        # Definiamo il device (GPU se disponibile, altrimenti CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_qnet = QNetwork().to(self.device)
        self.target_qnet = QNetwork().to(self.device)
        self.sync_target()
        self.optimizer = optim.Adam(self.online_qnet.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.train_steps = 0
        self.replay_buffer = ReplayBuffer()

    def sync_target(self):
        self.target_qnet.load_state_dict(self.online_qnet.state_dict())

    def pick_action(self, obs, valid_actions, env):
        if not valid_actions:
            print("\n[DEBUG] Nessuna azione valida per il player", env.current_player)
            raise ValueError("No valid actions!")
        # Se il campione è già un tensore (in ambiente GPU) usiamo unsqueeze
        if isinstance(obs, torch.Tensor):
            obs_t = obs.unsqueeze(0)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_qnet(obs_t)[0]
        best_a = None
        best_q = -1e9
        for a in valid_actions:
            # Qui per ogni indice di azione, estraiamo il valore (seppur con .item(), l'overhead è minimo)
            val = q_values[a].item()
            if val > best_q:
                best_q = val
                best_a = a
        return best_a

    def update_epsilon(self):
        self.train_steps += 1
        ratio = max(0, (EPSILON_DECAY - self.train_steps) / EPSILON_DECAY)
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * ratio

    def store_transition(self, transition):
        # transition = (obs, action, reward, next_obs, valid_next)
        obs, action, reward, next_obs, valid_next = transition
        # Converto in tensori sul device se non lo sono già
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(next_obs):
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        self.replay_buffer.push((obs, action, reward, next_obs, valid_next))

    def can_train(self):
        return len(self.replay_buffer) >= BATCH_SIZE

    def train_step(self):
        if not self.can_train():
            return
        obs, actions, rewards, next_obs, valid_next = self.replay_buffer.sample(BATCH_SIZE)
        # Poiché abbiamo già salvato tensori su GPU, possiamo usare torch.stack
        obs_t = torch.stack(obs)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.stack(next_obs)
        q_values = self.online_qnet(obs_t)
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_all = self.target_qnet(next_obs_t)
        targets = []
        for i in range(BATCH_SIZE):
            valid_acts = valid_next[i]
            if valid_acts:
                max_q_next = max(q_next_all[i, a].item() for a in valid_acts)
            else:
                max_q_next = 0.0
            t = rewards_t[i] + GAMMA * max_q_next
            targets.append(t)
        targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_sa, targets_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def maybe_sync_target(self):
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.sync_target()

    def save_checkpoint(self, filename):
        torch.save({
            "online_state_dict": self.online_qnet.state_dict(),
            "target_state_dict": self.target_qnet.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps
        }, filename)
        print(f"[DQNAgent] Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename, map_location=self.device)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Checkpoint loaded from {filename}")

############################################################
# 4) Multi-agent training con Monte Carlo
############################################################
def train_agents(num_episodes=10):
    # Impostiamo il profiler per monitorare sia attività CPU che GPU.
    with torch.profiler.profile(
         activities=[
             torch.profiler.ProfilerActivity.CPU,
             torch.profiler.ProfilerActivity.CUDA
         ],
         schedule=torch.profiler.schedule(
             wait=1,    # salta il primo step
             warmup=1,  # step di riscaldamento
             active=3   # registra per 3 step
         ),
         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
         record_shapes=True,
         profile_memory=True,
         with_stack=True
    ) as profiler:
        
        agent_team0 = DQNAgent(team_id=0)
        agent_team1 = DQNAgent(team_id=1)
        if os.path.isfile(CHECKPOINT_PATH + "_team0.pth"):
            agent_team0.load_checkpoint(CHECKPOINT_PATH + "_team0.pth")
        if os.path.isfile(CHECKPOINT_PATH + "_team1.pth"):
            agent_team1.load_checkpoint(CHECKPOINT_PATH + "_team1.pth")
        first_player = 0

        for ep in range(num_episodes):
            print(f"\n=== Episodio {ep+1}, inizia player {first_player} ===")
            env = ScoponeEnvMA()
            env.current_player = first_player

            episode_transitions = {0: [], 1: []}
            for move in range(40):
                cp = env.current_player
                team_id = 0 if cp in [0, 2] else 1
                agent = agent_team0 if team_id == 0 else agent_team1
                obs_current = env._get_observation(cp)
                valid_actions = env.get_valid_actions()
                action = agent.pick_action(obs_current, valid_actions, env)
                next_obs, rew, info = env.step(action)
                next_valid = env.get_valid_actions()
                transition = (obs_current, action, 0.0, next_obs, next_valid)
                episode_transitions[team_id].append(transition)
                agent.store_transition(transition)
                agent.train_step()
                agent.update_epsilon()
                agent.maybe_sync_target()

            breakdown = compute_final_score_breakdown(env.game_state)
            final_reward = compute_final_reward_from_breakdown(breakdown)
            r0, r1 = final_reward[0], final_reward[1]
            print(f"Team Rewards finali: [r0={r0}, r1={r1}]")

            for team_id, transitions in episode_transitions.items():
                team_reward = r0 if team_id == 0 else r1
                for trans in transitions:
                    updated_trans = (trans[0], trans[1], team_reward, trans[3], trans[4])
                    if team_id == 0:
                        agent_team0.store_transition(updated_trans)
                    else:
                        agent_team1.store_transition(updated_trans)

            for _ in range(50):
                agent_team0.train_step()
                agent_team1.train_step()
                agent_team0.update_epsilon()
                agent_team1.update_epsilon()
                agent_team0.maybe_sync_target()
                agent_team1.maybe_sync_target()

            first_player = (first_player + 1) % 4

            # Ogni episodio, segnala un passo al profiler
            profiler.step()

        agent_team0.save_checkpoint(CHECKPOINT_PATH + "_team0.pth")
        agent_team1.save_checkpoint(CHECKPOINT_PATH + "_team1.pth")
        print("=== Fine training DQN multi-agent ===")

if __name__ == "__main__":
    train_agents(num_episodes=10)
