# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os

# Configurazione GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from environment import ScoponeEnvMA

# Parametri di rete e training
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000    # passi totali di training per passare da 1.0 a 0.01
BATCH_SIZE = 128          # dimensione mini-batch
REPLAY_SIZE = 10000      # capacità massima del replay buffer
TARGET_UPDATE_FREQ = 1000  # ogni quanti step sincronizzi la rete target
CHECKPOINT_PATH = "scopone_checkpoint"

############################################################
# 1) Definiamo una classe EpisodicReplayBuffer
############################################################
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
    def __init__(self, obs_dim=10823, action_dim=80):  # Aggiornato da 10793 a 10823
        super().__init__()
        # Feature extractor ottimizzato per la rappresentazione avanzata
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
        
        # Sezioni aggiuntive
        self.hand_table_processor = nn.Sequential(
            nn.Linear(83, 64),
            nn.ReLU()
        )
        
        self.captured_processor = nn.Sequential(
            nn.Linear(82, 64),
            nn.ReLU()
        )
        
        self.stats_processor = nn.Sequential(
            nn.Linear(334, 64),  # Aggiornato da 304 a 334 (+30 per i nuovi valori)
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
        
        # Sposta tutto il modello su GPU all'inizializzazione
        self.to(device)
    
    def forward(self, x):
        # Assicurati che l'input sia sulla GPU
        if x.device != next(self.parameters()).device:
            x = x.to(device)
            
        # Processa l'intero input attraverso il backbone
        backbone_features = self.backbone(x)
        
        # Divide l'input in sezioni semantiche
        hand_table = x[:, :83]
        captured = x[:, 83:165]
        history = x[:, 169:10489]
        stats = x[:, 10489:]  # L'indice non cambia, cambia solo la dimensione
        
        # Processa ogni sezione
        hand_table_features = self.hand_table_processor(hand_table)
        captured_features = self.captured_processor(captured)
        history_features = self.history_processor(history)
        stats_features = self.stats_processor(stats)
        
        # Combina tutte le features
        combined = torch.cat([
            backbone_features,
            hand_table_features,
            captured_features,
            history_features,
            stats_features
        ], dim=1)
        
        # Elabora le features combinate
        final_features = self.combiner(combined)
        
        # Calcola i valori delle azioni
        action_values = self.action_head(final_features)
        
        return action_values

############################################################
# 3) DQNAgent con target network + replay
############################################################

class DQNAgent:
    def __init__(self, team_id):
        self.team_id = team_id
        self.online_qnet = QNetwork()  # Già spostato su GPU nell'inizializzazione
        self.target_qnet = QNetwork()  # Già spostato su GPU nell'inizializzazione
        self.sync_target()
        
        self.optimizer = optim.Adam(self.online_qnet.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.train_steps = 0
        self.episodic_buffer = EpisodicReplayBuffer()
    
    def pick_action(self, obs, valid_actions, env):
        """Epsilon-greedy con gestione GPU"""
        if not valid_actions:
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
            # Ottieni i valori Q per ogni azione - spostato su GPU
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.online_qnet(obs_t)
            
            best_action = None
            best_q_value = float('-inf')
            
            for action in valid_actions:
                # Converti l'azione in tensor e spostala su GPU
                action_t = torch.tensor(action, dtype=torch.float32).to(device)
                
                # Calcola il valore Q come prodotto scalare
                q_value = torch.sum(action_values[0] * action_t).item()
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action
    
    def train_episodic_monte_carlo(self, specific_episode=None):
        """Addestramento Monte Carlo su episodi usando flat rewards, ottimizzato per GPU"""
        # Determina quali episodi processare
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            return  # Nessun episodio disponibile
        
        # Processa ogni episodio nel buffer
        for episode in episodes_to_process:
            if not episode:
                continue
                
            # Ottieni la reward finale dall'ultima transizione
            final_reward = episode[-1][2] if episode else 0.0
            
            # Assegna la stessa reward finale a tutte le transizioni (flat Monte Carlo)
            returns = [final_reward for _ in range(len(episode))]
            
            # Ottimizzazione: processa in batch anziché una alla volta
            batch_size = min(64, len(episode))
            num_batches = (len(episode) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(episode))
                
                batch_obs = []
                batch_actions = []
                batch_returns = []
                
                for i in range(start_idx, end_idx):
                    obs, action, _, _, _, _ = episode[i]
                    G = returns[i]
                    
                    batch_obs.append(obs)
                    batch_actions.append(action)
                    batch_returns.append(G)
                
                # Converti in tensor e sposta su GPU
                obs_t = torch.tensor(batch_obs, dtype=torch.float32).to(device)
                batch_actions_t = [torch.tensor(a, dtype=torch.float32).to(device) for a in batch_actions]
                returns_t = torch.tensor(batch_returns, dtype=torch.float32).to(device)
                
                # Calcola i Q-values correnti
                self.optimizer.zero_grad()
                q_values = self.online_qnet(obs_t)
                
                # Calcola i Q-values per le azioni scelte
                q_values_for_actions = []
                for i, action_t in enumerate(batch_actions_t):
                    q_value = torch.sum(q_values[i] * action_t)
                    q_values_for_actions.append(q_value)
                
                q_values_for_actions = torch.stack(q_values_for_actions)
                
                # Calcola loss e aggiorna
                loss = nn.MSELoss()(q_values_for_actions, returns_t)
                loss.backward()
                self.optimizer.step()
                
                # Aggiorna counters
                self.update_epsilon()
                self.maybe_sync_target()
    
    def train(self):
        """Addestramento su tutti gli episodi disponibili usando Monte Carlo flat rewards."""
        self.train_episodic_monte_carlo()
    
    def start_episode(self):
        """Inizia un nuovo episodio."""
        self.episodic_buffer.start_episode()
    
    def end_episode(self):
        """
        Termina l'episodio corrente e applica immediatamente Monte Carlo sull'ultimo episodio.
        """
        self.episodic_buffer.end_episode()
        
        # Applica Monte Carlo solo sull'ultimo episodio, se disponibile
        if self.episodic_buffer.episodes:
            self.train_episodic_monte_carlo(self.episodic_buffer.episodes[-1])
    
    def store_episode_transition(self, transition):
        """
        Memorizza una transizione nell'episodic buffer.
        """
        # Aggiungi all'episodio corrente
        self.episodic_buffer.add_transition(transition)
    
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
        """Salva il checkpoint tenendo conto della GPU"""
        torch.save({
            "online_state_dict": self.online_qnet.state_dict(),
            "target_state_dict": self.target_qnet.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps
        }, filename)
        print(f"[DQNAgent] Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Carica il checkpoint tenendo conto della GPU"""
        ckpt = torch.load(filename, map_location=device)
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
    Compatibile con la rappresentazione a matrice (80 dim).
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
                    team0_obs, np.zeros(80, dtype=np.float32), team0_reward,
                    np.zeros_like(team0_obs), True, []
                )
                
                # Transizione finale per Team1
                team1_final_transition = (
                    team1_obs, np.zeros(80, dtype=np.float32), team1_reward,
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
            agent.train()
            
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
    """Aggiornamento Monte Carlo ottimizzato per GPU"""
    if not transitions:
        return
    
    # Estrai la reward finale dall'ultima transizione
    final_reward = transitions[-1][2] if transitions else 0.0
    
    # Usa lo stesso valore di reward per tutte le transizioni
    returns = [final_reward] * len(transitions)
    
    # Elaborazione in batch anziché per singola transizione
    batch_size = min(64, len(transitions))
    num_batches = (len(transitions) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(transitions))
        
        batch_obs = []
        batch_actions = []
        batch_returns = []
        
        for i in range(start_idx, end_idx):
            obs, action, _, _, _, _ = transitions[i]
            G = returns[i]
            
            batch_obs.append(obs)
            batch_actions.append(action)
            batch_returns.append(G)
        
        # Converti in tensor e sposta su GPU
        obs_t = torch.tensor(batch_obs, dtype=torch.float32).to(device)
        batch_actions_t = [torch.tensor(a, dtype=torch.float32).to(device) for a in batch_actions]
        returns_t = torch.tensor(batch_returns, dtype=torch.float32).to(device)
        
        # Calcola i Q-values correnti
        agent.optimizer.zero_grad()
        q_values = agent.online_qnet(obs_t)
        
        # Calcola i Q-values per le azioni scelte
        q_values_for_actions = []
        for i, action_t in enumerate(batch_actions_t):
            q_value = torch.sum(q_values[i] * action_t)
            q_values_for_actions.append(q_value)
        
        q_values_for_actions = torch.stack(q_values_for_actions)
        
        # Calcola loss e aggiorna i pesi
        loss = nn.MSELoss()(q_values_for_actions, returns_t)
        loss.backward()
        agent.optimizer.step()
        
        # Aggiorna target periodicamente
        agent.sync_target()

if __name__ == "__main__":
    train_agents(num_episodes=400)