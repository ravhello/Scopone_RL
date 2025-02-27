import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from environment import ScoponeEnvMA
from actions import MAX_ACTIONS
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

# Parametri di rete e training
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
BATCH_SIZE = 32
REPLAY_SIZE = 10000
TAU = 0.01  # Tasso di aggiornamento per il soft update
CHECKPOINT_PATH = "scopone_checkpoint"
WARMUP_STEPS = 2000  # prima di questa soglia l'agente non si allena


############################################################
# 1) PrioritizedReplayBuffer (uguale all'originale)
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
        while idx < self.capacity - 1:
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
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:]) if self.sum_tree.size > 0 else 1.0
        self.sum_tree.add(max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        indices = []
        samples = []
        epsilon = 1e-6
        total = self.sum_tree.size
        total_priority = self.sum_tree.total_priority
        if total == 0 or total_priority == 0:
            return None
        segment = total_priority / batch_size

        for i in range(batch_size):
            while True:
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                idx, priority, data = self.sum_tree.get_leaf(s)
                if data is not None:
                    indices.append(idx)
                    samples.append(data)
                    break

        priorities = np.array([self.sum_tree.tree[idx] for idx in indices])
        sampling_probabilities = priorities / total_priority
        weights = (total * sampling_probabilities + epsilon) ** (-beta)
        weights = weights / weights.max()

        obs, actions, rewards, next_obs, valid_next = zip(*samples)
        return (list(obs), list(actions), list(rewards), list(next_obs), list(valid_next),
                indices, torch.tensor(weights, dtype=torch.float32))

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        td_errors_np = torch.abs(td_errors).detach().cpu().numpy() + epsilon
        for idx, error in zip(indices, td_errors_np):
            self.sum_tree.update(idx, error ** self.alpha)

    def __len__(self):
        return self.sum_tree.size


############################################################
# 2) DuelingQNetwork (implementazione "reale")
############################################################
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim=3764, hidden_dim=512):
        """
        Emette un tensore [batch_size, MAX_ACTIONS].
        Nel dueling DQN classico:
          Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a'))
        """
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim),
            nn.ReLU()
        )
        # Stream che produce V(s) di dimensione [batch_size, 1]
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Stream che produce A(s,a) di dimensione [batch_size, MAX_ACTIONS]
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, MAX_ACTIONS)
        )

    def forward(self, state):
        """
        state: [batch_size, obs_dim]
        Ritorna: [batch_size, MAX_ACTIONS]
        """
        feat = self.feature_layer(state)              # [B, hidden_dim]
        val = self.value_stream(feat)                 # [B, 1]
        adv = self.adv_stream(feat)                   # [B, MAX_ACTIONS]
        adv_mean = adv.mean(dim=1, keepdim=True)      # [B, 1]

        # Q(s,a) per ciascun a
        # shape finale: [B, MAX_ACTIONS]
        q_all = val + adv - adv_mean
        return q_all


############################################################
# 3) DQNAgent
############################################################
class DQNAgent:
    def __init__(self, team_id, replay_buffer):
        self.replay_buffer = replay_buffer
        self.team_id = team_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inizializza la rete "dueling" e la rete target
        self.online_qnet = DuelingQNetwork().to(self.device)
        self.target_qnet = DuelingQNetwork().to(self.device)
        self._soft_update(tau=1.0)  # Copia completa iniziale

        self.optimizer = optim.Adam(self.online_qnet.parameters(), lr=LR)

        self.epsilon = EPSILON_START
        self.train_steps = 0

    def _soft_update(self, tau=TAU):
        """
        Aggiorna i parametri della rete target con un soft update di fattore tau.
        target_param = tau * online_param + (1-tau) * target_param
        """
        for target_param, online_param in zip(self.target_qnet.parameters(), self.online_qnet.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def pick_action(self, obs, valid_actions, env=None):
        """
        Sceglie l'azione epsilon-greedy tra le sole valid_actions.
        valid_actions: torch.tensor di dimensione [n_valid], contenente gli indici di azione validi
        """
        if valid_actions.numel() == 0:
            print("[DEBUG] Nessuna azione valida!")
            raise ValueError("No valid actions.")

        # Con probabilità epsilon, scelta casuale
        if random.random() < self.epsilon:
            idx = random.randint(0, valid_actions.numel() - 1)
            return valid_actions[idx].item()
        else:
            # Altrimenti, calcolo Q(s) = [1, MAX_ACTIONS], prendo i Q per le sole azioni valide e faccio argmax
            obs_t = obs.unsqueeze(0).to(self.device)  # [1, obs_dim]
            with torch.no_grad():
                q_all = self.online_qnet(obs_t)       # [1, MAX_ACTIONS]
            q_all = q_all[0]                          # [MAX_ACTIONS]

            # Filtra i Q-values sulle azioni valide
            q_valid = q_all[valid_actions]            # [n_valid]
            best_idx_local = torch.argmax(q_valid).item()
            action_chosen = valid_actions[best_idx_local].item()
            return action_chosen

    def update_epsilon(self):
        """
        Decade epsilon linearmente secondo EPSILON_DECAY, fino a EPSILON_END.
        """
        self.train_steps += 1
        ratio = max(0, (EPSILON_DECAY - self.train_steps) / EPSILON_DECAY)
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * ratio

    def store_transition(self, transition):
        """
        Aggiunge transizione al replay buffer, convertendo in tensori se necessario.
        transition = (obs, action, reward, next_obs, valid_next)
        """
        obs, action, reward, next_obs, valid_next = transition
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(next_obs):
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        self.replay_buffer.push((obs, action, reward, next_obs, valid_next))

    def can_train(self):
        """
        Ritorna True se abbiamo abbastanza campioni nel buffer
        E se abbiamo superato il warm-up di WARMUP_STEPS
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return False
        if self.train_steps < WARMUP_STEPS:
            return False
        return True

    def train_step(self):
        """
        Estrae un batch di transizioni, calcola la loss e aggiorna la rete online.
        Applica il Double DQN e aggiorna la rete target con soft update.
        """
        if not self.can_train():
            return

        batch = self.replay_buffer.sample(BATCH_SIZE)
        if batch is None:
            return

        obs, actions, rewards, next_obs, valid_next, indices, weights = batch

        obs_t = torch.stack(obs).to(self.device)               # [B, obs_dim]
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)  # [B]
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.stack(next_obs).to(self.device)     # [B, obs_dim]
        weights_t = weights.to(self.device)                    # [B]

        # 1) Calcolo Q(s,a) online: otteniamo [B, MAX_ACTIONS], poi gather sulle azioni eseguite
        q_all_online = self.online_qnet(obs_t)                 # [B, MAX_ACTIONS]
        q_sa = q_all_online.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # [B]

        # 2) Double DQN per il next state
        with torch.no_grad():
            # 2.a) Troviamo azioni argmax con la rete online
            q_all_next_online = self.online_qnet(next_obs_t)   # [B, MAX_ACTIONS]

            # valid_next[i] => azioni valide per la i-esima transizione
            # Dobbiamo estrarre l'argmax *tra le azioni valide* per ciascun sample
            best_actions_list = []
            for i in range(BATCH_SIZE):
                va = valid_next[i]  # tensore di azioni valide
                if va.numel() == 0:
                    # Se non ci sono azioni valide, forziamo "azione 0" e q_next=0
                    best_actions_list.append(torch.tensor(0, device=self.device))
                else:
                    q_valid = q_all_next_online[i][va]  # [n_valid]
                    best_idx_local = torch.argmax(q_valid).item()
                    best_action = va[best_idx_local]
                    best_actions_list.append(best_action)

            best_actions_t = torch.stack(best_actions_list)     # [B]

            # 2.b) Valutiamo le Q con la rete target e prendiamo Q(s', best_action)
            q_all_next_target = self.target_qnet(next_obs_t)    # [B, MAX_ACTIONS]
            # q_next_target_vals => [B]
            q_next_target_vals = q_all_next_target.gather(1, best_actions_t.unsqueeze(1)).squeeze(1)

            # Se non c'è alcuna azione valida (va.numel()==0), settiamo q=0
            # Creiamo maschera booleana
            mask_no_action = torch.tensor(
                [1 if valid_next[i].numel() == 0 else 0 for i in range(BATCH_SIZE)],
                device=self.device, dtype=torch.bool
            )
            q_next_target_vals = torch.where(mask_no_action,
                                             torch.zeros_like(q_next_target_vals),
                                             q_next_target_vals)

            # 2.c) Target
            targets_t = rewards_t + GAMMA * q_next_target_vals  # [B]

        # 3) Loss con SmoothL1 (Huber) e PESI di importanza
        loss_fn = nn.SmoothL1Loss(reduction='none')
        loss_per_sample = loss_fn(q_sa, targets_t)   # [B]
        loss = (loss_per_sample * weights_t).mean()

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4) Soft update della rete target
        self._soft_update()

        # 5) Aggiornamento delle priorità basato su TD-error
        td_errors = q_sa - targets_t  # [B]
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
def train_agents(num_episodes=50):
    """
    Esegue un training multi-agente (due team) giocando partite di 40 mosse.
    Al termine di ogni partita, assegna la ricompensa finale e fa training intensivo.
    """
    from environment import ScoponeEnvMA  # Import qui per chiarezza
    shared_buffer = PrioritizedReplayBuffer(capacity=REPLAY_SIZE)

    # Due agenti (team 0 e team 1)
    agent_team0 = DQNAgent(team_id=0, replay_buffer=shared_buffer)
    agent_team1 = DQNAgent(team_id=1, replay_buffer=shared_buffer)

    # Caricamento eventuale checkpoint
    if os.path.isfile(CHECKPOINT_PATH + "_team0.pth"):
        agent_team0.load_checkpoint(CHECKPOINT_PATH + "_team0.pth")
    if os.path.isfile(CHECKPOINT_PATH + "_team1.pth"):
        agent_team1.load_checkpoint(CHECKPOINT_PATH + "_team1.pth")

    first_player = 0
    BATCH_SIZE_EPISODES = 10
    TRAIN_STEPS_AFTER_BATCH = 200

    for ep in range(num_episodes):
        print(f"\n=== Episodio {ep+1}, inizia player {first_player} ===")
        env = ScoponeEnvMA()
        env.current_player = first_player

        episode_transitions = {0: [], 1: []}

        # Simuliamo 40 mosse (una partita)
        for move in range(40):
            cp = env.current_player
            team_id = 0 if cp in [0, 2] else 1
            agent = agent_team0 if team_id == 0 else agent_team1

            obs_current = env._get_observation(cp)
            valid_actions = env.get_valid_actions()  # tensore di azioni valide

            action = agent.pick_action(obs_current, valid_actions)

            next_obs, _, info = env.step(action)
            next_valid = env.get_valid_actions()

            # Reward intermedia = 0. Assegneremo il punteggio totale a fine episodio
            transition = (obs_current, action, 0.0, next_obs, next_valid)
            episode_transitions[team_id].append(transition)

        # Calcolo punteggio finale
        breakdown = compute_final_score_breakdown(env.game_state)
        final_reward = compute_final_reward_from_breakdown(breakdown)
        r0, r1 = final_reward[0], final_reward[1]
        print(f"Team Rewards finali: [r0={r0}, r1={r1}]")

        # Assegniamo la reward a tutte le transizioni accumulate e salviamo nel buffer
        for tid, transitions in episode_transitions.items():
            team_reward = r0 if tid == 0 else r1
            ag = agent_team0 if tid == 0 else agent_team1
            for trans in transitions:
                obs_, act_, _, nxt_obs_, nxt_val_ = trans
                updated_trans = (obs_, act_, team_reward, nxt_obs_, nxt_val_)
                ag.store_transition(updated_trans)

        # Cambiamo il primo giocatore per la partita successiva
        first_player = (first_player + 1) % 4

        # Training intensivo ogni 10 partite
        if (ep + 1) % BATCH_SIZE_EPISODES == 0:
            print(f"[Episodio {ep+1}] Inizio blocco di training intensivo...")
            for _ in range(TRAIN_STEPS_AFTER_BATCH):
                agent_team0.train_step()
                agent_team1.train_step()
                agent_team0.update_epsilon()
                agent_team1.update_epsilon()
            print(f"[Episodio {ep+1}] Fine blocco di training intensivo.")

    # Salvataggio checkpoint finali
    agent_team0.save_checkpoint(CHECKPOINT_PATH + "_team0.pth")
    agent_team1.save_checkpoint(CHECKPOINT_PATH + "_team1.pth")
    print("=== Fine training DQN multi-agent ===")


if __name__ == "__main__":
    train_agents(num_episodes=10)
