# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os

from environment import ScoponeEnvMA
from actions import encode_action, get_valid_actions, MAX_ACTIONS

# Parametri di rete e training
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000    # passi totali di training per passare da 1.0 a 0.01
BATCH_SIZE = 32          # dimensione mini-batch
REPLAY_SIZE = 10000      # capacità massima del replay buffer
TARGET_UPDATE_FREQ = 1000  # ogni quanti step sincronizzi la rete target
CHECKPOINT_PATH = "scopone_checkpoint"

############################################################
# 1) Definiamo una classe ReplayBuffer
############################################################

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, transition):
        """
        Aggiunge una transizione: (obs, action, reward, next_obs, done, valid_actions_next)
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Estrae in modo casuale un mini-batch di transizioni.
        """
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, next_valids = zip(*batch)
        return (np.array(obs), actions, rewards, np.array(next_obs), dones, next_valids)

    def __len__(self):
        return len(self.buffer)

############################################################
# 2) Rete neurale QNetwork
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
# 3) DQNAgent con target network + replay
############################################################

class DQNAgent:
    def __init__(self, team_id):
        self.team_id = team_id
        self.online_qnet = QNetwork()
        self.target_qnet = QNetwork()
        self.sync_target()  # inizialmente copia i pesi

        self.optimizer = optim.Adam(self.online_qnet.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.train_steps = 0
        self.replay_buffer = ReplayBuffer()

    def sync_target(self):
        """
        Copia i pesi dalla rete online (online_qnet) alla rete target (target_qnet).
        """
        self.target_qnet.load_state_dict(self.online_qnet.state_dict())

    def pick_action(self, obs, valid_actions, env):
        """
        Epsilon-greedy su Q(s,·). Se valid_actions è vuota, solleva eccezione (nessuna azione possibile).
        """
        if not valid_actions:
            # Stampa debug di environment
            print("\n[DEBUG] Nessuna azione valida! Stato attuale:")
            print("  Current player:", env.current_player)
            print("  Tavolo:", env.game_state["table"])
            for p in range(4):
                print(f"  Mano p{p}:", env.game_state["hands"][p])
            print("  History:", env.game_state["history"])
            raise ValueError("Nessuna azione valida (valid_actions=[]).")

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape=[1,3764]
            with torch.no_grad():
                q_values = self.online_qnet(obs_t)[0]  # shape [2048]
            best_a = None
            best_q = -1e9
            for a in valid_actions:
                val = q_values[a].item()
                if val > best_q:
                    best_q = val
                    best_a = a
            return best_a

    def update_epsilon(self):
        """
        Aggiorna epsilon step-by-step, scendendo linearmente da EPSILON_START a EPSILON_END entro EPSILON_DECAY step.
        """
        self.train_steps += 1
        ratio = max(0, (EPSILON_DECAY - self.train_steps) / EPSILON_DECAY)
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END)*ratio

    def store_transition(self, transition):
        """
        Inserisce la transizione nel replay buffer: (obs, action, reward, next_obs, done, valid_next)
        """
        self.replay_buffer.push(transition)

    def can_train(self):
        """
        Verifica se il buffer ha abbastanza transizioni per un mini-batch.
        """
        return len(self.replay_buffer) >= BATCH_SIZE

    def train_step(self):
        """
        Esegue un update di Q-learning su un mini-batch dal replay buffer.
        """
        if not self.can_train():
            return

        obs, actions, rewards, next_obs, dones, next_valids = self.replay_buffer.sample(BATCH_SIZE)

        obs_t = torch.tensor(obs, dtype=torch.float32)       # [B,3764]
        actions_t = torch.tensor(actions, dtype=torch.long)  # [B]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)  # [B]
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32) # [B,3764]
        dones_t = torch.tensor(dones, dtype=torch.float32)    # [B]

        # Q(s, a)
        q_values = self.online_qnet(obs_t)  # [B,2048]
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # [B]

        # costruiamo il target
        with torch.no_grad():
            q_next_all = self.target_qnet(next_obs_t)  # [B,2048]

        targets = []
        for i in range(BATCH_SIZE):
            if dones_t[i] > 0.5:
                t = rewards_t[i]
            else:
                valid_acts = next_valids[i]  # array di azioni
                max_q_next = max(q_next_all[i, a].item() for a in valid_acts) if valid_acts else 0.0
                t = rewards_t[i] + GAMMA * max_q_next
            targets.append(t)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_sa, targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def maybe_sync_target(self):
        """
        Se self.train_steps è multiplo di TARGET_UPDATE_FREQ, sincronizza la rete target.
        """
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.sync_target()

    def save_checkpoint(self, filename):
        torch.save({
            "online_state_dict": self.online_qnet.state_dict(),
            "target_state_dict": self.target_qnet.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps
        }, filename)
        print(f"[DQNAgent] Checkpoint salvato: {filename}")

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Checkpoint caricato da {filename}")

############################################################
# 4) Multi-agent training
############################################################

def train_agents(num_episodes=10):
    """
    Esegue un training multi-agent "nativo" (Team0 e Team1) con:
      - Replay Buffer
      - Target Network
      - Nessuna reward intermedia
    """
    # Creiamo 2 agenti
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Caricamento checkpoint se esistono
    if os.path.isfile(CHECKPOINT_PATH+"_team0.pth"):
        agent_team0.load_checkpoint(CHECKPOINT_PATH+"_team0.pth")
    if os.path.isfile(CHECKPOINT_PATH+"_team1.pth"):
        agent_team1.load_checkpoint(CHECKPOINT_PATH+"_team1.pth")

    first_player = 0
    global_step = 0  

    for ep in range(num_episodes):
        print(f"\n=== Episodio {ep+1}, inizia player {first_player} ===")
        env = ScoponeEnvMA()
        env.current_player = first_player

        done = False
        obs_current = env._get_observation(env.current_player)

        while not done:
            cp = env.current_player
            team_id = 0 if cp in [0,2] else 1
            agent = agent_team0 if team_id==0 else agent_team1

            valid_acts = env.get_valid_actions()
            action = agent.pick_action(obs_current, valid_acts, env)

            next_obs, reward_scalar, done, info = env.step(action)
            global_step += 1

            # reward=0 durante l'episodio
            if not done:
                next_player = env.current_player
                next_valid = env.get_valid_actions()
            else:
                next_valid = []

            agent.store_transition( (obs_current, action, 0.0, next_obs, done, next_valid) )

            agent.train_step()
            agent.update_epsilon()
            agent.maybe_sync_target()

            obs_current = next_obs

        # Fine partita
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            print(f"Team Rewards finali: {team_rewards}")

        first_player = (first_player + 1)%4

    # Salviamo i checkpoint finali
    agent_team0.save_checkpoint(CHECKPOINT_PATH+"_team0.pth")
    agent_team1.save_checkpoint(CHECKPOINT_PATH+"_team1.pth")
    print("=== Fine training con DQN multi-agent + Replay + Target Net ===")

if __name__ == "__main__":
    train_agents(num_episodes=2000)
