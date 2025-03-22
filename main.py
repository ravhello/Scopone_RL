# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os
from line_profiler import LineProfiler, profile, global_profiler
import time

# Configurazione GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from environment import ScoponeEnvMA

# Alla fine di ogni episodio
# Forza garbage collection
import gc
gc.collect()
if torch.cuda.is_available():
    # Più aggressivo nel liberare la memoria
    torch.cuda.empty_cache()
    # Stampa statistiche memoria
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    print(f"  GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

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
    def __init__(self, capacity=20):  # Memorizza fino a 20 episodi completi
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
    
    def get_previous_episodes(self):
        """
        Returns all episodes except the most recent one.
        Useful for training on past episodes.
        """
        if len(self.episodes) <= 1:
            return []
        return list(self.episodes)[:-1]

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
    
    @profile
    def forward(self, x):
        # Assicurati che l'input sia sulla GPU
        if x.device != next(self.parameters()).device:
            x = x.to(device)
            
        # Utilizza inplace operations dove possibile
        import torch.nn.functional as F
        
        # Processa l'intero input attraverso il backbone
        x1 = F.relu(self.backbone[0](x), inplace=True)
        x2 = F.relu(self.backbone[2](x1), inplace=True)
        x3 = F.relu(self.backbone[4](x2), inplace=True)
        backbone_features = F.relu(self.backbone[6](x3), inplace=True)
        
        # Divide l'input in sezioni semantiche
        hand_table = x[:, :83]
        captured = x[:, 83:165]
        history = x[:, 169:10489]
        stats = x[:, 10489:]
        
        # Processa ogni sezione
        hand_table_features = F.relu(self.hand_table_processor[0](hand_table), inplace=True)
        captured_features = F.relu(self.captured_processor[0](captured), inplace=True)
        history_features = F.relu(self.history_processor[0](history), inplace=True)
        history_features = F.relu(self.history_processor[2](history_features), inplace=True)
        stats_features = F.relu(self.stats_processor[0](stats), inplace=True)
        
        # Combina tutte le features
        combined = torch.cat([
            backbone_features,
            hand_table_features,
            captured_features,
            history_features,
            stats_features
        ], dim=1)
        
        # Elabora le features combinate
        final_features = F.relu(self.combiner[0](combined), inplace=True)
        
        # Calcola i valori delle azioni
        action_values = self.action_head(final_features)
        
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
        self.episodic_buffer = EpisodicReplayBuffer()
        
        # Aggiunte per ottimizzazione GPU
        torch.backends.cudnn.benchmark = True  # Ottimizzazione per dimensioni di input fisse
        self.scaler = torch.amp.GradScaler('cuda')  # Per mixed precision training
    
    @profile
    def pick_action(self, obs, valid_actions, env):
        """Epsilon-greedy ottimizzato per GPU"""
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
            # Crea un unico tensore per tutte le azioni valide
            valid_actions_np = np.stack(valid_actions)
            
            # Sposta tutto su GPU una sola volta
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            valid_actions_t = torch.tensor(valid_actions_np, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # Calcola i valori Q per l'osservazione
                action_values = self.online_qnet(obs_t)
                
                # Calcola i valori Q per tutte le azioni valide simultaneamente
                # [1, 80] x [n_actions, 80] -> [n_actions]
                q_values = torch.sum(action_values.view(1, 1, 80) * valid_actions_t.view(-1, 1, 80), dim=2).squeeze()
                
                # Trova l'azione con il valore Q più alto
                best_action_idx = torch.argmax(q_values).item()
            
            return valid_actions[best_action_idx]
    
    @profile
    def train_episodic_monte_carlo(self, specific_episode=None):
        """
        Addestramento Monte Carlo su episodi usando flat rewards con ottimizzazioni GPU.
        """
        # Determina quali episodi processare
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            return  # Nessun episodio disponibile
        
        # Prepara tutti i dati prima di processarli
        all_obs = []
        all_actions = []
        all_returns = []
        
        for episode in episodes_to_process:
            if not episode:
                continue
                
            # Ottieni la reward finale dall'ultima transizione dell'episodio
            final_reward = episode[-1][2] if episode else 0.0
            
            # Aggiungi tutte le transizioni, usando la stessa reward finale per tutte
            for obs, action, _, _, _, _ in episode:
                all_obs.append(obs)
                all_actions.append(action)
                all_returns.append(final_reward)
        
        if not all_obs:
            return  # Nessuna transizione da processare
        
        # Converti a numpy array una sola volta
        all_obs_np = np.stack(all_obs)
        all_actions_np = np.stack(all_actions)
        all_returns_np = np.array(all_returns)
        
        # Converti a tensor e sposta su GPU - una sola volta!
        all_obs_t = torch.tensor(all_obs_np, dtype=torch.float32, device=device)
        all_actions_t = torch.tensor(all_actions_np, dtype=torch.float32, device=device)
        all_returns_t = torch.tensor(all_returns_np, dtype=torch.float32, device=device)
        
        # Process in batch più grandi per sfruttare meglio la GPU
        batch_size = min(256, len(all_obs))
        num_batches = (len(all_obs) + batch_size - 1) // batch_size
        
        # Riduci la frequenza di sync_target
        sync_counter = 0
        sync_frequency = 10
        
        # Corretto: usa torch.amp.autocast invece di torch.cuda.amp.autocast
        with torch.amp.autocast(device_type='cuda'):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_obs))
                
                # Prendi slices dei tensori già sulla GPU
                batch_obs_t = all_obs_t[start_idx:end_idx]
                batch_actions_t = all_actions_t[start_idx:end_idx]
                batch_returns_t = all_returns_t[start_idx:end_idx]
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True è più veloce
                
                # Forward pass
                q_values = self.online_qnet(batch_obs_t)
                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                
                # Loss usando mixed precision
                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                
                # Backward e optimizer step con gradient scaling
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Aggiorna epsilon
                self.update_epsilon()
                
                # Sync target periodicamente
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    self.sync_target()
                    sync_counter = 0
        
        # Libera memoria
        del all_obs_t, all_actions_t, all_returns_t
        torch.cuda.empty_cache()
    
    @profile
    def store_episode_transition(self, transition):
        """
        Memorizza una transizione nell'episodic buffer.
        """
        # Aggiungi all'episodio corrente
        self.episodic_buffer.add_transition(transition)
    
    @profile
    def end_episode(self):
        """
        Termina l'episodio corrente SENZA training.
        Il training deve essere chiamato esplicitamente dopo questo metodo.
        """
        self.episodic_buffer.end_episode()
        # Nota: rimosso il training automatico qui
            
    @profile
    def start_episode(self):
        """Inizia un nuovo episodio."""
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

@profile
def train_agents(num_episodes=10):
    """
    Esegue un training multi-agent episodico.
    Il training avviene solo alla fine di ogni episodio, con reward flat per tutte le mosse.
    Ottimizzato per GPU.
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
    
    # Per monitorare i tempi di esecuzione
    episode_times = []
    train_times = []  # Tracciamento separato dei tempi di training

    for ep in range(num_episodes):
        episode_start_time = time.time()
        print(f"\n=== Episodio {ep+1}, inizia player {first_player} ===")
        
        # Crea l'ambiente di gioco
        env = ScoponeEnvMA()
        env.current_player = first_player

        # Iniziamo un nuovo episodio per entrambi gli agenti
        agent_team0.episodic_buffer.start_episode()
        agent_team1.episodic_buffer.start_episode()

        done = False
        obs_current = env._get_observation(env.current_player)

        # Loop principale della partita
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

            # Durante il gioco, le mosse hanno reward=0
            next_valid = env.get_valid_actions() if not done else []
            transition = (obs_current, action, reward, next_obs, done, next_valid)
            
            # Memorizza la transizione nel buffer dell'agente del giocatore corrente
            if team_id == 0:
                agent_team0.episodic_buffer.add_transition(transition)
            else:
                agent_team1.episodic_buffer.add_transition(transition)
            
            # Prepara per la prossima iterazione
            obs_current = next_obs

        # Fine partita - Termina gli episodi
        agent_team0.episodic_buffer.end_episode()
        agent_team1.episodic_buffer.end_episode()
        
        # Ottieni le reward finali
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
            print(f"Team Rewards finali: {team_rewards}")
        
        # TRAINING ALLA FINE DELL'EPISODIO - Ottimizzato per GPU
        print(f"Training alla fine dell'episodio {ep+1}...")
        train_start_time = time.time()
        
        # Team 0 training - pre-processamento batch
        if agent_team0.episodic_buffer.episodes:
            # Ottieni l'ultimo episodio
            last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
            
            if not last_episode_team0:
                print("  Nessuna transizione per team 0, skip training")
            else:
                print(f"  Training team 0 sull'ultimo episodio (reward={team0_reward})")
                
                # Usa zip per estrarre tutti i componenti in una volta sola
                all_obs, all_actions, _, all_next_obs, all_dones, all_next_valids = zip(*last_episode_team0)
                
                # Pre-processa i dati in batch
                all_obs_np = np.stack(all_obs)
                all_actions_np = np.stack(all_actions)
                
                # Crea episodio di training con reward flat
                training_episode_team0 = []
                for i in range(len(last_episode_team0)):
                    training_episode_team0.append((
                        all_obs[i], 
                        all_actions[i], 
                        team0_reward,  # Reward flat
                        all_next_obs[i],
                        all_dones[i],
                        all_next_valids[i]
                    ))
                
                # Training con batch pre-processati
                agent_team0.train_episodic_monte_carlo(training_episode_team0)
        
        # Team 1 training - stessa logica
        if agent_team1.episodic_buffer.episodes:
            last_episode_team1 = agent_team1.episodic_buffer.episodes[-1]
            
            if not last_episode_team1:
                print("  Nessuna transizione per team 1, skip training")
            else:
                print(f"  Training team 1 sull'ultimo episodio (reward={team1_reward})")
                
                # Estrazione efficiente dei dati
                all_obs, all_actions, _, all_next_obs, all_dones, all_next_valids = zip(*last_episode_team1)
                
                # Pre-processa i dati in batch
                all_obs_np = np.stack(all_obs)
                all_actions_np = np.stack(all_actions)
                
                # Crea episodio di training
                training_episode_team1 = []
                for i in range(len(last_episode_team1)):
                    training_episode_team1.append((
                        all_obs[i], 
                        all_actions[i], 
                        team1_reward,  # Reward flat
                        all_next_obs[i],
                        all_dones[i],
                        all_next_valids[i]
                    ))
                
                agent_team1.train_episodic_monte_carlo(training_episode_team1)
        
        # Training su episodi passati - stesso approccio
        if ep % 5 == 0 and len(agent_team0.episodic_buffer.episodes) > 3:
            print("  Training aggiuntivo su episodi passati per team 0")
            prev_episodes = agent_team0.episodic_buffer.get_previous_episodes()
            past_episodes = random.sample(prev_episodes, min(3, len(prev_episodes)))
            
            for episode in past_episodes:
                if episode and len(episode) > 0:
                    # Pre-processing batch anche per episodi passati
                    episode_obs, episode_actions, _, episode_next_obs, episode_dones, episode_next_valids = zip(*episode)
                    # Preprocessing numpy
                    _ = np.stack(episode_obs)
                    _ = np.stack(episode_actions)
                    agent_team0.train_episodic_monte_carlo(episode)
        
        if ep % 5 == 0 and len(agent_team1.episodic_buffer.episodes) > 3:
            print("  Training aggiuntivo su episodi passati per team 1")
            prev_episodes = agent_team1.episodic_buffer.get_previous_episodes()
            past_episodes = random.sample(prev_episodes, min(3, len(prev_episodes)))
            
            for episode in past_episodes:
                if episode and len(episode) > 0:
                    # Pre-processing batch anche per episodi passati
                    episode_obs, episode_actions, _, episode_next_obs, episode_dones, episode_next_valids = zip(*episode)
                    # Preprocessing numpy
                    _ = np.stack(episode_obs)
                    _ = np.stack(episode_actions)
                    agent_team1.train_episodic_monte_carlo(episode)
        
        # Traccia tempo di training
        train_time = time.time() - train_start_time
        train_times.append(train_time)
        print(f"  Training completato in {train_time:.2f} secondi")
        
        # Gestione memoria GPU migliorata
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Monitoraggio memoria GPU
            print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, "
                  f"{torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")
        
        # Calcola e registra il tempo dell'episodio
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        print(f"Episodio {ep+1} completato in {episode_time:.2f} secondi")
        
        # Prepara per il prossimo episodio
        first_player = (first_player + 1) % 4

    # Salviamo i checkpoint finali
    agent_team0.save_checkpoint(CHECKPOINT_PATH+"_team0.pth")
    agent_team1.save_checkpoint(CHECKPOINT_PATH+"_team1.pth")
    
    print("=== Fine training ===")
    print("\nTempi di esecuzione degli episodi:")
    for i, t in enumerate(episode_times):
        print(f"Episodio {i+1}: {t:.2f}s")
    print(f"Tempo medio per episodio: {sum(episode_times)/len(episode_times):.2f}s")
    print(f"Tempo medio per training: {sum(train_times)/len(train_times):.2f}s")
    print(f"Percentuale di tempo in training: {sum(train_times)/sum(episode_times)*100:.1f}%")


if __name__ == "__main__":
    # Esegui il training per pochi episodi per profilare
    train_agents(num_episodes=50000)
    
    # Stampa i risultati del profiling
    global_profiler.print_stats()
    
    # Genera un report dettagliato e salvalo su file
    report = global_profiler.generate_report("profiling_report.txt")