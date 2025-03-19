# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os

from environment import ScoponeEnvMA

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
    

class EpisodicReplayBuffer:
    def __init__(self, capacity=100):  # Memorizza fino a 100 episodi completi
        self.episodes = collections.deque(maxlen=capacity)
        self.current_episode = []  # Episodio corrente in costruzione
        
    def start_episode(self):
        """Inizia un nuovo episodio"""
        self.current_episode = []
        
    def add_transition(self, transition):
        """Aggiunge una transizione all'episodio corrente"""
        self.current_episode.append(transition)
        
    def end_episode(self):
        """Conclude l'episodio corrente e lo aggiunge al buffer"""
        if self.current_episode:
            self.episodes.append(list(self.current_episode))
            self.current_episode = []
        
    def sample_episode(self):
        """Estrae casualmente un episodio completo"""
        if not self.episodes:
            return []
        return random.choice(self.episodes)
    
    def sample_batch(self, batch_size):
        """Estrae casualmente un batch di transizioni da tutti gli episodi"""
        all_transitions = []
        for episode in self.episodes:
            all_transitions.extend(episode)
        
        if len(all_transitions) < batch_size:
            return all_transitions
        
        batch = random.sample(all_transitions, batch_size)
        obs, actions, rewards, next_obs, dones, next_valids = zip(*batch)
        return (np.array(obs), actions, rewards, np.array(next_obs), dones, next_valids)
    
    def __len__(self):
        """Ritorna il numero totale di transizioni in tutti gli episodi"""
        return sum(len(episode) for episode in self.episodes) + len(self.current_episode)
    
    def get_all_episodes(self):
        """Restituisce tutti gli episodi"""
        return list(self.episodes)

############################################################
# 2) Rete neurale QNetwork
############################################################

class QNetwork(nn.Module):
    def __init__(self, obs_dim=4484, action_dim=154):
        super().__init__()
        # Feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Singola testa per valutare l'intera azione
        self.action_head = nn.Linear(256, action_dim)
    
    def forward(self, x):
        features = self.backbone(x)
        action_values = self.action_head(features)
        return action_values

############################################################
# 3) DQNAgent con target network + replay
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
        
        # Mantieni il ReplayBuffer standard
        self.replay_buffer = ReplayBuffer()
        
        # Aggiungi il nuovo EpisodicReplayBuffer
        self.episodic_buffer = EpisodicReplayBuffer()
        
        # Flags per controllare quali strategie di apprendimento utilizzare
        self.use_dqn = True             # DQN standard 
        self.use_monte_carlo = True     # Monte Carlo
        self.use_episodic = True        # Episodic Learning
    
    def pick_action(self, obs, valid_actions, env):
        """
        Epsilon-greedy con singola testa che valuta l'azione completa.
        """
        if not valid_actions:
            # Debug in caso di nessuna azione valida
            print("\n[DEBUG] Nessuna azione valida! Stato attuale:")
            print("  Current player:", env.current_player)
            print("  Tavolo:", env.game_state["table"])
            for p in range(4):
                print(f"  Mano p{p}:", env.game_state["hands"][p])
            print("  History:", env.game_state["history"])
            raise ValueError("Nessuna azione valida (valid_actions=[]).")
        
        # Epsilon-greedy: scegli un'azione casuale con probabilità epsilon
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Ottieni i valori Q per ogni azione
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_values = self.online_qnet(obs_t)
            
            best_action = None
            best_q_value = float('-inf')
            
            for action in valid_actions:
                # Converti l'azione in tensor
                action_t = torch.tensor(action, dtype=torch.float32)
                
                # Calcola il valore Q come prodotto scalare
                q_value = torch.sum(action_values[0] * action_t).item()
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action
    
    def train_step(self):
        """
        Esegue un passo di training con singola testa (DQN standard).
        """
        if not self.can_train():
            return
        
        obs, actions, rewards, next_obs, dones, next_valids = self.replay_buffer.sample(BATCH_SIZE)
        
        obs_t = torch.tensor(obs, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        
        # Calcola i Q-values per lo stato corrente
        current_q_values = self.online_qnet(obs_t)
        
        # Calcola i Q-values per le azioni scelte
        q_values = []
        for i, action in enumerate(actions):
            action_t = torch.tensor(action, dtype=torch.float32)
            q_value = torch.sum(current_q_values[i] * action_t)
            q_values.append(q_value)
        
        q_values = torch.stack(q_values)
        
        # Calcola i target Q-values usando la target network
        with torch.no_grad():
            target_q_values = self.target_qnet(next_obs_t)
        
        # Calcola i target per ciascun esempio nel batch
        targets = []
        for i in range(len(obs)):
            if dones_t[i] > 0.5:
                # Se lo stato è terminale, il target è semplicemente la reward
                targets.append(rewards_t[i])
            else:
                # Altrimenti, calcola il massimo Q-value delle azioni valide nello stato successivo
                valid_acts = next_valids[i]
                if not valid_acts:
                    max_q_next = 0.0
                else:
                    max_q_next = float('-inf')
                    for action in valid_acts:
                        action_t = torch.tensor(action, dtype=torch.float32)
                        q_next = torch.sum(target_q_values[i] * action_t).item()
                        max_q_next = max(max_q_next, q_next)
                    
                targets.append(rewards_t[i] + GAMMA * max_q_next)
        
        targets_t = torch.tensor(targets, dtype=torch.float32)
        
        # Calcola la loss e aggiorna i pesi
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_values, targets_t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Aggiorna epsilon e la target network se necessario
        self.update_epsilon()
        self.maybe_sync_target()
    
    def train_monte_carlo(self, episode=None):
        """
        Applica l'apprendimento Monte Carlo con pesi uniformi per tutte le transizioni.
        """
        if episode is None and self.episodic_buffer.episodes:
            episode = self.episodic_buffer.episodes[-1]
        
        if not episode:
            return
        
        # Ottieni la reward finale dall'ultima transizione
        final_reward = episode[-1][2] if episode else 0.0
        
        # Invece di applicare il decadimento gamma,
        # assegna la stessa reward finale a tutte le transizioni
        returns = [final_reward for _ in range(len(episode))]
        
        # Aggiorna i Q-values per ogni transizione con lo stesso peso
        for i, ((obs, action, _, _, _, _), G) in enumerate(zip(episode, returns)):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_t = torch.tensor(action, dtype=torch.float32)
            
            # Calcola Q-value corrente
            self.optimizer.zero_grad()
            q_values = self.online_qnet(obs_t)
            q_value = torch.sum(q_values[0] * action_t)
            
            # Target è il rendimento finale uguale per tutte le transizioni
            target = torch.tensor([G], dtype=torch.float32)
            
            # Calcola loss e aggiorna
            loss = nn.MSELoss()(q_value.unsqueeze(0), target)
            loss.backward()
            self.optimizer.step()
            
            # Aggiorna counters
            self.update_epsilon()
            self.maybe_sync_target()
    
    def train_episodic(self):
        """
        Esegue un passo di training usando tutti gli episodi disponibili.
        """
        if not self.episodic_buffer.episodes:
            return
        
        # Usa tutti gli episodi senza campionamento
        all_episodes = self.episodic_buffer.get_all_episodes()
        
        for episode in all_episodes:
            # Calcola i rendimenti per ogni transizione nell'episodio
            G = 0
            returns = []
            for _, _, reward, _, _, _ in reversed(episode):
                G = reward + GAMMA * G
                returns.insert(0, G)
            
            # Aggiorna i Q-values per ogni transizione con il suo rendimento
            for i, ((obs, action, _, _, _, _), G) in enumerate(zip(episode, returns)):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_t = torch.tensor(action, dtype=torch.float32)
                
                # Calcola Q-value corrente
                self.optimizer.zero_grad()
                q_values = self.online_qnet(obs_t)
                q_value = torch.sum(q_values[0] * action_t)
                
                # Target è il rendimento dell'episodio
                target = torch.tensor([G], dtype=torch.float32)
                
                # Calcola loss e aggiorna
                loss = nn.MSELoss()(q_value.unsqueeze(0), target)
                loss.backward()
                self.optimizer.step()
                
                # Aggiorna counters
                self.update_epsilon()
                self.maybe_sync_target()
    
    def train_combined(self):
        """
        Combina le diverse strategie di training in un unico metodo.
        """
        # DQN standard
        if self.use_dqn and self.can_train():
            self.train_step()
        
        # Episodic learning - trains on all episodes
        if self.use_episodic and self.episodic_buffer.episodes:
            self.train_episodic()
    
    def start_episode(self):
        """Inizia un nuovo episodio."""
        self.episodic_buffer.start_episode()
    
    def end_episode(self):
        """
        Termina l'episodio corrente e applica Monte Carlo se abilitato.
        """
        self.episodic_buffer.end_episode()
        
        # Applica Monte Carlo se abilitato
        if self.use_monte_carlo and self.episodic_buffer.episodes:
            self.train_monte_carlo()
    
    def store_transition(self, transition):
        """Memorizza nel replay buffer standard (compatibilità)."""
        self.replay_buffer.push(transition)
    
    def store_episode_transition(self, transition):
        """
        Memorizza una transizione sia nell'episodic buffer che nel buffer standard.
        """
        # Aggiungi all'episodio corrente
        self.episodic_buffer.add_transition(transition)
        
        # Mantieni compatibilità con il buffer standard
        self.replay_buffer.push(transition)
    
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

    def can_train(self):
        """Checks if there are enough samples in the replay buffer to train."""
        return len(self.replay_buffer) >= BATCH_SIZE

    def save_checkpoint(self, filename):
        """Saves the agent's state to a checkpoint file."""
        torch.save({
            "online_state_dict": self.online_qnet.state_dict(),
            "target_state_dict": self.target_qnet.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps
        }, filename)
        print(f"[DQNAgent] Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Loads the agent's state from a checkpoint file."""
        ckpt = torch.load(filename)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Checkpoint loaded from {filename}")

############################################################
# 4) Multi-agent training
############################################################

def train_agents(num_episodes=10):
    """
    Esegue un training multi-agent combinando DQN, Monte Carlo ed Episodic.
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

        # Iniziamo un nuovo episodio per entrambi gli agenti
        agent_team0.start_episode()
        agent_team1.start_episode()
        
        # Tracciamo tutti i transitions per applicare Monte Carlo alla fine
        team0_transitions = []
        team1_transitions = []

        done = False
        obs_current = env._get_observation(env.current_player)

        while not done:
            cp = env.current_player
            team_id = 0 if cp in [0,2] else 1
            agent = agent_team0 if team_id==0 else agent_team1

            valid_acts = env.get_valid_actions()
            if not valid_acts:
                break
                
            action = agent.pick_action(obs_current, valid_acts, env)

            next_obs, reward, done, info = env.step(action)
            global_step += 1

            # Gestione della reward finale
            if done and "team_rewards" in info:
                team_rewards = info["team_rewards"]
                
                # Aggiorniamo le reward per entrambi i team
                team0_reward = team_rewards[0]
                team1_reward = team_rewards[1]
                
                # Creiamo transizioni finali per entrambi i team
                team0_obs = env._get_observation(0)
                team1_obs = env._get_observation(1)
                
                # Transizione finale per Team0
                team0_final_transition = (
                    team0_obs, np.zeros_like(action), team0_reward,
                    np.zeros_like(team0_obs), True, []
                )
                
                # Transizione finale per Team1
                team1_final_transition = (
                    team1_obs, np.zeros_like(action), team1_reward,
                    np.zeros_like(team1_obs), True, []
                )
                
                # Aggiungiamo le transizioni finali alle rispettive liste
                team0_transitions.append(team0_final_transition)
                team1_transitions.append(team1_final_transition)
                
                # Memorizzazione delle transizioni finali
                agent_team0.store_episode_transition(team0_final_transition)
                agent_team1.store_episode_transition(team1_final_transition)
            else:
                # Reward normale durante il gioco
                next_valid = env.get_valid_actions() if not done else []
                
                # Crea la transizione
                transition = (obs_current, action, reward, next_obs, done, next_valid)
                
                # Memorizza la transizione nell'agente del giocatore corrente
                if team_id == 0:
                    team0_transitions.append(transition)
                    agent_team0.store_episode_transition(transition)
                else:
                    team1_transitions.append(transition)
                    agent_team1.store_episode_transition(transition)
            
            # Training step standard (verrà usato solo se enabled)
            agent.train_combined()
            
            obs_current = next_obs

        # Fine partita, terminiamo gli episodi
        agent_team0.end_episode()
        agent_team1.end_episode()
        
        # Applicazione esplicita di Monte Carlo per entrambi i team
        if team0_transitions:
            monte_carlo_update_team(agent_team0, team0_transitions)
        if team1_transitions:
            monte_carlo_update_team(agent_team1, team1_transitions)
        
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            print(f"Team Rewards finali: {team_rewards}")

        first_player = (first_player + 1) % 4

    # Salviamo i checkpoint finali
    agent_team0.save_checkpoint(CHECKPOINT_PATH+"_team0.pth")
    agent_team1.save_checkpoint(CHECKPOINT_PATH+"_team1.pth")
    print("=== Fine training con strategie combinate DQN + Monte Carlo + Episodic ===")

def monte_carlo_update_team(agent, transitions):
    """
    Aggiorna i Q-values usando Monte Carlo con pesi uniformi per tutte le transizioni.
    """
    if not transitions:
        return
    
    # Estrai la reward finale dall'ultima transizione
    final_reward = transitions[-1][2] if transitions else 0.0
    
    # Usa lo stesso valore di reward per tutte le transizioni
    returns = [final_reward] * len(transitions)
    
    # Aggiorna i Q-values per ciascuna transizione
    for i, ((obs, action, _, _, _, _), G) in enumerate(zip(transitions, returns)):
        # Converti osservazione e azione in tensori
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32)
        
        # Calcola il valore Q corrente
        agent.optimizer.zero_grad()
        current_q_values = agent.online_qnet(obs_t)
        q_value = torch.sum(current_q_values[0] * action_t)
        
        # Target è il rendimento finale uguale per tutte le transizioni
        target = torch.tensor([G], dtype=torch.float32)
        
        # Calcola la loss e aggiorna i pesi
        loss = nn.MSELoss()(q_value.unsqueeze(0), target)
        loss.backward()
        agent.optimizer.step()
        
        # Aggiorna target periodicamente
        if i % 10 == 0:  # Ogni 10 aggiornamenti
            agent.sync_target()

if __name__ == "__main__":
    train_agents(num_episodes=2000)