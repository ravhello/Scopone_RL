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
TAU = 0.01  # Tasso di aggiornamento
CHECKPOINT_PATH = "scopone_checkpoint"

############################################################
# 1) PrioritizedReplayBuffer
############################################################
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.size = 0
        self.write = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s):
        idx = 0
        while idx < self.capacity - 1:  # finché non raggiungiamo una foglia
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.sum_tree = SumTree(capacity)

    def push(self, transition):
        # Assegna la massima priorità per le nuove transizioni
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:]) if self.sum_tree.size > 0 else 1.0
        self.sum_tree.add(max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        indices = []
        samples = []
        epsilon = 1e-6  # piccolo epsilon per evitare divisioni per zero
        total = self.sum_tree.size
        total_priority = self.sum_tree.total_priority
        # Se il buffer è vuoto o la somma totale è zero, ritorna None
        if total == 0 or total_priority == 0:
            return None
        segment = total_priority / batch_size

        for i in range(batch_size):
            while True:
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                idx, priority, data = self.sum_tree.get_leaf(s)
                # Se il campione è valido, lo accettiamo
                if data is not None:
                    indices.append(idx)
                    samples.append(data)
                    break
                # In caso contrario, ripeti il campionamento per questa porzione

        # Calcola i pesi per l'importanza sampling
        priorities = np.array([self.sum_tree.tree[idx] for idx in indices])
        sampling_probabilities = priorities / total_priority
        weights = (total * sampling_probabilities + epsilon) ** (-beta)
        weights = weights / weights.max()

        # Decomponi i campioni in tuple (obs, action, reward, next_obs, valid_next)
        obs, actions, rewards, next_obs, valid_next = zip(*samples)
        return (list(obs), list(actions), list(rewards), list(next_obs), list(valid_next), indices, torch.tensor(weights, dtype=torch.float32))

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        td_errors_np = torch.abs(td_errors).detach().cpu().numpy() + epsilon
        for idx, error in zip(indices, td_errors_np):
            self.sum_tree.update(idx, error ** self.alpha)

    def __len__(self):
        return self.sum_tree.size


############################################################
# 2) QNetwork
############################################################
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim=3764, act_dim=2048):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        # branch valore
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # branch vantaggio
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )

    def forward(self, x):
        x = self.feature_layer(x)
        V = self.value_stream(x)              # [batch_size, 1]
        A = self.advantage_stream(x)         # [batch_size, act_dim]

        # Q(s,a) = V(s) + A(s,a) - max(A(s,a'))
        A_mean = A.mean(dim=1, keepdim=True) # potresti usare anche max invece di mean
        Q = V + (A - A_mean)
        return Q

    def masked_forward(self, x, valid_actions):
        q_values = self.forward(x)
        valid_actions = valid_actions.to(x.device)
        q_valid = q_values.index_select(1, valid_actions)
        return q_valid


############################################################
# 3) DQNAgent
############################################################
class DQNAgent:
    def __init__(self, team_id):
        self.team_id = team_id
        # Definiamo il device (GPU se disponibile, altrimenti CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_qnet = DuelingQNetwork().to(self.device)
        self.target_qnet = DuelingQNetwork().to(self.device)
        self.sync_target()
        self.optimizer = optim.Adam(self.online_qnet.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.train_steps = 0
        # Utilizzo del nuovo PrioritizedReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer()

    def sync_target(self):
        for target_param, online_param in zip(self.target_qnet.parameters(), self.online_qnet.parameters()):
            target_param.data.copy_(TAU * online_param.data + (1 - TAU) * target_param.data)

    def pick_action(self, obs, valid_actions, env):
        if valid_actions.numel() == 0:
            print("\n[DEBUG] Nessuna azione valida per il player", env.current_player)
            raise ValueError("No valid actions!")
        obs_t = obs.clone().detach().to(self.device).unsqueeze(0)

        with torch.no_grad():
            q_valid = self.online_qnet.masked_forward(obs_t, valid_actions)

        best_idx = torch.argmax(q_valid).item()
        # Ritorna l'azione come intero
        return valid_actions[best_idx].item()

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

        sample_result = self.replay_buffer.sample(BATCH_SIZE)
        if sample_result is None:
            return
        obs, actions, rewards, next_obs, valid_next, indices, weights = sample_result

        obs_t = torch.stack(obs)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.stack(next_obs)

        # Calcola Q(s, a)
        q_values = self.online_qnet(obs_t)
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # -- [ INIZIO Double DQN ] --
        with torch.no_grad():
            # 1) Ricava Q-values dalla rete ONLINE per determinare l'argmax a'
            q_next_online = self.online_qnet(next_obs_t)

            # Creazione mask per le azioni valide
            batch_size, num_actions = q_next_online.shape
            mask = torch.zeros((batch_size, num_actions), dtype=torch.bool, device=self.device)
            batch_indices = []
            action_indices = []
            for i, valid in enumerate(valid_next):
                if valid.numel() > 0:
                    batch_indices.append(torch.full((valid.size(0),), i, dtype=torch.long, device=self.device))
                    action_indices.append(valid)
            if batch_indices:
                batch_indices = torch.cat(batch_indices)
                action_indices = torch.cat(action_indices)
                mask[batch_indices, action_indices] = True

            # Metti a -inf le azioni non valide
            q_next_online_masked = q_next_online.masked_fill(~mask, float('-inf'))
            # argmax a' su rete online
            max_actions_next = q_next_online_masked.argmax(dim=1)

            # 2) Ora calcoli Q(s', a') usando la rete TARGET
            q_next_target_all = self.target_qnet(next_obs_t)
            # Applica la stessa maschera
            q_next_target_all_masked = q_next_target_all.masked_fill(~mask, float('-inf'))

            # Ottieni i Q-value con le azioni scelte dalla rete online
            max_q_next = q_next_target_all_masked.gather(1, max_actions_next.unsqueeze(1)).squeeze(1)
            # Se era -inf (nessuna azione valida), rimpiazza con 0
            max_q_next = torch.where(max_q_next == float('-inf'), torch.zeros_like(max_q_next), max_q_next)

        targets_t = rewards_t + GAMMA * max_q_next
        # -- [ FINE Double DQN ] --

        # Calcolo loss (meglio SmoothL1Loss)
        loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss
        loss_per_sample = loss_fn(q_sa, targets_t)
        loss = (loss_per_sample * weights.to(self.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Aggiornamento soft rete target
        self.sync_target()

        # Aggiorna le priorità
        td_errors = q_sa - targets_t
        self.replay_buffer.update_priorities(indices, td_errors)

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
    with torch.profiler.profile(
         activities=[
             torch.profiler.ProfilerActivity.CPU,
             torch.profiler.ProfilerActivity.CUDA
         ],
         schedule=torch.profiler.schedule(
             wait=1,      # ignora il primo step
             warmup=1,    # step di warmup
             active=2000  # registra per 2000 step (sufficientemente alto per il tuo loop)
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
                with torch.profiler.record_function("Scelta dell'azione"):
                    cp = env.current_player
                    team_id = 0 if cp in [0, 2] else 1
                    agent = agent_team0 if team_id == 0 else agent_team1
                    obs_current = env._get_observation(cp)
                    valid_actions = env.get_valid_actions()
                    action = agent.pick_action(obs_current, valid_actions, env)

                with torch.profiler.record_function("Step dell'ambiente"):
                    next_obs, rew, info = env.step(action)
                    next_valid = env.get_valid_actions()

                with torch.profiler.record_function("Memorizzazione transizione"):
                    transition = (obs_current, action, 0.0, next_obs, next_valid)
                    episode_transitions[team_id].append(transition)
                    agent.store_transition(transition)

                with torch.profiler.record_function("Training dell'agente"):
                    agent.train_step()
                    agent.update_epsilon()

            with torch.profiler.record_function("Calcolo Reward finale"):
                breakdown = compute_final_score_breakdown(env.game_state)
                final_reward = compute_final_reward_from_breakdown(breakdown)
                r0, r1 = final_reward[0], final_reward[1]

            print(f"Team Rewards finali: [r0={r0}, r1={r1}]")

            with torch.profiler.record_function("Aggiornamento Replay Buffer"):
                for team_id, transitions in episode_transitions.items():
                    team_reward = r0 if team_id == 0 else r1
                    for trans in transitions:
                        updated_trans = (trans[0], trans[1], team_reward, trans[3], trans[4])
                        if team_id == 0:
                            agent_team0.store_transition(updated_trans)
                        else:
                            agent_team1.store_transition(updated_trans)

            with torch.profiler.record_function("Episodi extra di training"):
                for _ in range(50):
                    agent_team0.train_step()
                    agent_team1.train_step()
                    agent_team0.update_epsilon()
                    agent_team1.update_epsilon()

            first_player = (first_player + 1) % 4

            # Chiamata a profiler.step() al termine di ogni episodio
            profiler.step()

        # Ora, mentre siamo ancora nel contesto 'with', estraiamo e stampiamo i dati
        records = profiler.key_averages()
        table_data = []
        for record in records:
            total_time = record.self_cpu_time_total + record.self_cuda_time_total
            table_data.append((record.key, total_time, record.count))
        table_data.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Function Name':30s} | {'Total Time (us)':>15s} | {'Call Count':>10s}")
        print("-" * 60)
        for key, total_time, count in table_data:
            print(f"{key:30s} | {total_time:15.0f} | {count:10d}")
        
        agent_team0.save_checkpoint(CHECKPOINT_PATH + "_team0.pth")
        agent_team1.save_checkpoint(CHECKPOINT_PATH + "_team1.pth")
        print("=== Fine training DQN multi-agent ===")


if __name__ == "__main__":
    train_agents(num_episodes=10)
