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

# Configurazione GPU avanzata
if torch.cuda.is_available():
    # Ottimizza per performance CUDA
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Per GPU più recenti (Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Aumenta la dimensione di allocazione cache
    torch.cuda.empty_cache()
    torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Usa fino al 95% della memoria disponibile

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
        
        # Imposta opzioni CUDA per performance
        if torch.cuda.is_available():
            # Abilita TF32 su Ampere GPUs per migliori performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Ottimizzazione per dimensioni di input fisse
            torch.backends.cudnn.benchmark = True
    
    @profile
    def forward(self, x):
        # Assicurati che l'input sia sulla GPU
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.device != device:
            x = x.to(device)
                
        import torch.nn.functional as F
        # Utilizzo di ReLU inplace per risparmiare memoria
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
            # MODIFICA: Converti tutti gli input in tensori GPU in un'unica operazione
            valid_actions_t = torch.tensor(np.stack(valid_actions), 
                                        dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # MODIFICA: Crea direttamente il tensore su GPU
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_values = self.online_qnet(obs_t)
                
                # MODIFICA: Operazione interamente su GPU senza conversioni intermedie
                q_values = torch.sum(action_values.view(1, 1, 80) * valid_actions_t.view(-1, 1, 80), dim=2).squeeze()
                best_action_idx = torch.argmax(q_values).item()
            
            return valid_actions[best_action_idx]
    
    @profile
    def train_episodic_monte_carlo(self, specific_episode=None):
        """
        Addestramento Monte Carlo su episodi usando flat rewards con ottimizzazioni GPU avanzate.
        """
        # Determina quali episodi processare
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            return  # Nessun episodio disponibile
        
        # Prepara tutti i dati prima di processarli sulla GPU
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
        
        # Converti tutti i dati in tensori GPU in un'unica operazione
        try:
            # Implementazione ottimizzata: stack diretto su GPU
            all_obs_t = torch.tensor(np.stack(all_obs), dtype=torch.float32, device=device)
            all_actions_t = torch.tensor(np.stack(all_actions), dtype=torch.float32, device=device)
            all_returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
        except RuntimeError as e:
            # Fallback in caso di out of memory: caricamento incrementale
            print(f"GPU memory issue in tensor creation: {e}")
            batch_size = 256
            num_batches = (len(all_obs) + batch_size - 1) // batch_size
            all_obs_t = []
            all_actions_t = []
            all_returns_t = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(all_obs))
                batch_obs = all_obs[start_idx:end_idx]
                batch_actions = all_actions[start_idx:end_idx]
                batch_returns = all_returns[start_idx:end_idx]
                
                all_obs_t.append(torch.tensor(np.stack(batch_obs), 
                                            dtype=torch.float32, device=device))
                all_actions_t.append(torch.tensor(np.stack(batch_actions), 
                                            dtype=torch.float32, device=device))
                all_returns_t.append(torch.tensor(batch_returns, 
                                            dtype=torch.float32, device=device))
            
            all_obs_t = torch.cat(all_obs_t)
            all_actions_t = torch.cat(all_actions_t)
            all_returns_t = torch.cat(all_returns_t)
        
        # Aumenta batch_size per sfruttare meglio la GPU
        batch_size = min(512, len(all_obs))  # Dimensione ottimizzata per GPU moderne
        num_batches = (len(all_obs_t) + batch_size - 1) // batch_size
        
        # Riduci la frequenza di sync_target
        sync_counter = 0
        sync_frequency = 10  # Sincronizza ogni 10 batch invece di ogni batch
        
        # Traccia le metriche per diagnostica
        total_loss = 0.0
        batch_count = 0
        
        # Usa mixed precision in modo più efficiente con float16
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_obs_t))
                
                # Prendi slices dei tensori già sulla GPU (evita copie)
                batch_obs_t = all_obs_t[start_idx:end_idx]
                batch_actions_t = all_actions_t[start_idx:end_idx]
                batch_returns_t = all_returns_t[start_idx:end_idx]
                
                # Zero gradients - usa set_to_none=True per maggiore efficienza di memoria
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass con kernel fusion dove possibile
                q_values = self.online_qnet(batch_obs_t)
                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                
                # Loss con mixed precision - usa reduction='mean' per stabilità numerica
                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                
                # Traccia la loss per diagnostica
                total_loss += loss.item()
                batch_count += 1
                
                # Backward e optimizer step con gradient scaling per mixed precision
                self.scaler.scale(loss).backward()
                
                # Clip gradient con una norma moderata per stabilità di training
                torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                
                # Optimizer step con scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Aggiorna epsilon dopo ogni batch per avanzare il training
                self.update_epsilon()
                
                # Sync target network periodicamente (non ad ogni batch)
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    self.sync_target()
                    sync_counter = 0
                    
                # Rilascia memoria GPU ogni 10 batch se necessario
                if batch_idx % 10 == 0 and batch_idx > 0:
                    torch.cuda.empty_cache()
        
        # Stampa la loss media per diagnostica
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"  Training: Avg loss = {avg_loss:.6f}, Epsilon = {self.epsilon:.4f}")
        
        # Libera memoria in modo più aggressivo al termine
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
    Esegue un training multi-agent episodico completamente ottimizzato per GPU.
    Il training avviene alla fine di ogni episodio, con reward flat per tutte le mosse.
    Implementa ottimizzazioni per ridurre drasticamente i trasferimenti CPU-GPU.
    """
    # Configurazione ottimale GPU
    if torch.cuda.is_available():
        # Impostazioni per massimizzare throughput
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Gestione aggressiva della memoria
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
        
        # Imposta allocator per ridurre frammentazione
        torch.cuda.memory_stats()

    # Crea gli agenti
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Carica checkpoint se esistono
    if os.path.isfile(CHECKPOINT_PATH+"_team0.pth"):
        agent_team0.load_checkpoint(CHECKPOINT_PATH+"_team0.pth")
    if os.path.isfile(CHECKPOINT_PATH+"_team1.pth"):
        agent_team1.load_checkpoint(CHECKPOINT_PATH+"_team1.pth")

    # Variabili di controllo
    first_player = 0
    global_step = 0
    
    # Monitoraggio prestazioni
    episode_times = []
    train_times = []
    inference_times = []
    
    # OTTIMIZZAZIONE: Pre-alloca buffer di tensori per evitare allocazioni ripetute
    # Buffer per carte
    card_buffer = {}
    for suit in ['denari', 'coppe', 'spade', 'bastoni']:
        for rank in range(1, 11):
            card_buffer[(rank, suit)] = torch.zeros(80, dtype=torch.float32, device=device)
    
    # Buffer per batch di training
    max_transitions = 40  # Numero massimo di transizioni atteso in un episodio
    team0_obs_buffer = torch.zeros((max_transitions, 10823), dtype=torch.float32, device=device)
    team0_actions_buffer = torch.zeros((max_transitions, 80), dtype=torch.float32, device=device)
    team0_rewards_buffer = torch.zeros(max_transitions, dtype=torch.float32, device=device)
    
    team1_obs_buffer = torch.zeros_like(team0_obs_buffer)
    team1_actions_buffer = torch.zeros_like(team0_actions_buffer)
    team1_rewards_buffer = torch.zeros_like(team0_rewards_buffer)
    
    # OTTIMIZZAZIONE: Profilo di memoria per training asincrono
    async_train_team0 = False
    async_train_team1 = False
    
    # Loop principale per episodi
    for ep in range(num_episodes):
        episode_start_time = time.time()
        print(f"\n=== Episodio {ep+1}/{num_episodes}, inizia player {first_player} ===")
        
        # Crea ambiente e inizializza
        env = ScoponeEnvMA()
        env.current_player = first_player

        # Inizializza buffer episodici
        agent_team0.start_episode()
        agent_team1.start_episode()

        # Stato iniziale
        done = False
        obs_current = env._get_observation(env.current_player)
        
        # OTTIMIZZAZIONE: Assicura che obs_current sia un array numpy
        if torch.is_tensor(obs_current):
            obs_current = obs_current.cpu().numpy()
            
        # Conteggi delle transition per ciascun team
        team0_transitions = 0
        team1_transitions = 0

        # Loop principale della partita
        inference_start = time.time()
        while not done:
            cp = env.current_player
            team_id = 0 if cp in [0,2] else 1
            agent = agent_team0 if team_id==0 else agent_team1

            # Ottieni azioni valide
            valid_acts = env.get_valid_actions()
            if not valid_acts:
                break
            
            # OTTIMIZZAZIONE: Conversione efficiente a tensori
            # Se valid_acts contiene già array numpy, converti una volta sola
            if len(valid_acts) > 0:
                if isinstance(valid_acts[0], np.ndarray):
                    valid_acts_t = torch.tensor(np.stack(valid_acts), dtype=torch.float32, device=device)
                    # Scelta azione ottimizzata
                    action = agent.pick_action(obs_current, valid_acts, env)
                else:
                    # Fallback se valid_acts non è già convertito
                    action = agent.pick_action(obs_current, valid_acts, env)
            
            # Esegui azione sull'ambiente
            next_obs, reward, done, info = env.step(action)
            
            # Assicura che next_obs sia numpy array
            if torch.is_tensor(next_obs):
                next_obs = next_obs.cpu().numpy()
                
            global_step += 1

            # Prepara transition
            next_valid = env.get_valid_actions() if not done else []
            transition = (obs_current, action, reward, next_obs, done, next_valid)
            
            # Memorizza la transizione nel buffer dell'agente del giocatore corrente
            if team_id == 0:
                agent_team0.store_episode_transition(transition)
                team0_transitions += 1
            else:
                agent_team1.store_episode_transition(transition)
                team1_transitions += 1
            
            # Prepara per la prossima iterazione
            obs_current = next_obs
            
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        # Memoria usata dopo inferenza
        if torch.cuda.is_available():
            print(f"  Memoria GPU dopo inferenza: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocata")

        # Termina episodi e prepara per training
        agent_team0.end_episode()
        agent_team1.end_episode()
        
        # Ottieni le reward finali
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
            print(f"  Team Rewards finali: {team_rewards}")
        
        # TRAINING ALLA FINE DELL'EPISODIO - Completamente ottimizzato per GPU
        print(f"  Training alla fine dell'episodio {ep+1}...")
        train_start_time = time.time()
        
        # OTTIMIZZAZIONE: Prepara batch direttamente su GPU
        # 1. Team 0 training batch
        team0_batch = None
        if agent_team0.episodic_buffer.episodes:
            last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
            if last_episode_team0:
                # Estrai dati episodio
                all_obs0, all_actions0, _, _, _, _ = zip(*last_episode_team0)
                
                # OTTIMIZZAZIONE: Riutilizza buffer pre-allocati
                ep_len = len(all_obs0)
                if ep_len > max_transitions:
                    # Ridimensiona buffer se necessario
                    team0_obs_buffer.resize_(ep_len, 10823)
                    team0_actions_buffer.resize_(ep_len, 80)
                    team0_rewards_buffer.resize_(ep_len)
                
                # Trasferimento dati su GPU in batch
                for i, (obs, action) in enumerate(zip(all_obs0, all_actions0)):
                    # Conversione diretta ottimizzata
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
                
                # Batch finale con slicing
                team0_batch = (
                    team0_obs_buffer[:ep_len], 
                    team0_actions_buffer[:ep_len], 
                    team0_rewards_buffer[:ep_len]
                )
        
        # 2. Team 1 training batch - stessa logica ottimizzata
        team1_batch = None
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
        
        # OTTIMIZZAZIONE: Training con Mixed Precision
        # Team 0 training con batch preparato
        if team0_batch:
            print(f"  Training team 0 sull'ultimo episodio (reward={team0_reward}, mosse={team0_transitions})")
            
            # Ottiene batch con dimensione ottimale
            team0_obs_t, team0_actions_t, team0_rewards_t = team0_batch
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Processa in batch più grandi per sfruttare meglio la GPU
                batch_size = min(512, len(team0_obs_t))
                num_batches = (len(team0_obs_t) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(team0_obs_t))
                    
                    # Slices dei tensori già sulla GPU
                    batch_obs_t = team0_obs_t[start_idx:end_idx]
                    batch_actions_t = team0_actions_t[start_idx:end_idx]
                    batch_returns_t = team0_rewards_t[start_idx:end_idx]
                    
                    # Zero gradients efficienti
                    agent_team0.optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass ottimizzato
                    q_values = agent_team0.online_qnet(batch_obs_t)
                    q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                    
                    # Loss con stabilità numerica
                    loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                    
                    # Backward con scaling
                    agent_team0.scaler.scale(loss).backward()
                    
                    # Gradient clipping per stabilità
                    torch.nn.utils.clip_grad_norm_(agent_team0.online_qnet.parameters(), max_norm=10.0)
                    
                    # Step con scaling
                    agent_team0.scaler.step(agent_team0.optimizer)
                    agent_team0.scaler.update()
                    
                    # Aggiorna epsilon
                    agent_team0.update_epsilon()
        
        # Team 1 training con uguale logica ottimizzata
        if team1_batch:
            print(f"  Training team 1 sull'ultimo episodio (reward={team1_reward}, mosse={team1_transitions})")
            
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
        
        # OTTIMIZZAZIONE: Sincronizzazione target network meno frequente
        if global_step % TARGET_UPDATE_FREQ == 0:
            agent_team0.sync_target()
            agent_team1.sync_target()
        
        # OTTIMIZZAZIONE: Training su episodi passati con riutilizzo memoria
        # Training su passi precedenti ogni 5 episodi
        if ep % 5 == 0 and ep > 0:
            # Team 0 training su episodi passati
            if len(agent_team0.episodic_buffer.episodes) > 3:
                print("  Training aggiuntivo su episodi passati per team 0")
                
                # Seleziona episodi precedenti
                prev_episodes = agent_team0.episodic_buffer.get_previous_episodes()
                past_episodes = random.sample(prev_episodes, min(3, len(prev_episodes)))
                
                # Count totale transizioni
                total_transitions = sum(len(episode) for episode in past_episodes)
                
                # Alloca/ridimensiona buffer se necessario
                if total_transitions > team0_obs_buffer.shape[0]:
                    team0_obs_buffer.resize_(total_transitions, 10823)
                    team0_actions_buffer.resize_(total_transitions, 80)
                    team0_rewards_buffer.resize_(total_transitions)
                
                # Prepara mega-batch
                idx = 0
                for episode in past_episodes:
                    if episode and len(episode) > 0:
                        # Reward dell'episodio 
                        episode_reward = episode[-1][2]
                        
                        for obs, action, _, _, _, _ in episode:
                            if idx < total_transitions:
                                # Trasferimento diretto su GPU
                                if isinstance(obs, np.ndarray):
                                    team0_obs_buffer[idx].copy_(torch.tensor(obs, device=device))
                                else:
                                    team0_obs_buffer[idx].copy_(obs)
                                    
                                if isinstance(action, np.ndarray):  
                                    team0_actions_buffer[idx].copy_(torch.tensor(action, device=device))
                                else:
                                    team0_actions_buffer[idx].copy_(action)
                                    
                                team0_rewards_buffer[idx] = episode_reward
                                idx += 1
                
                # Training su mega-batch
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    batch_size = min(512, idx)
                    num_batches = (idx + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, idx)
                        
                        batch_obs_t = team0_obs_buffer[start_idx:end_idx]
                        batch_actions_t = team0_actions_buffer[start_idx:end_idx]
                        batch_returns_t = team0_rewards_buffer[start_idx:end_idx]
                        
                        agent_team0.optimizer.zero_grad(set_to_none=True)
                        
                        q_values = agent_team0.online_qnet(batch_obs_t)
                        q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                        
                        loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                        
                        agent_team0.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(agent_team0.online_qnet.parameters(), max_norm=10.0)
                        agent_team0.scaler.step(agent_team0.optimizer)
                        agent_team0.scaler.update()
                        
                        agent_team0.update_epsilon()
            
            # Team 1 training con stessa logica ottimizzata
            if len(agent_team1.episodic_buffer.episodes) > 3:
                print("  Training aggiuntivo su episodi passati per team 1")
                
                prev_episodes = agent_team1.episodic_buffer.get_previous_episodes()
                past_episodes = random.sample(prev_episodes, min(3, len(prev_episodes)))
                
                total_transitions = sum(len(episode) for episode in past_episodes)
                
                if total_transitions > team1_obs_buffer.shape[0]:
                    team1_obs_buffer.resize_(total_transitions, 10823)
                    team1_actions_buffer.resize_(total_transitions, 80)
                    team1_rewards_buffer.resize_(total_transitions)
                
                idx = 0
                for episode in past_episodes:
                    if episode and len(episode) > 0:
                        episode_reward = episode[-1][2]
                        
                        for obs, action, _, _, _, _ in episode:
                            if idx < total_transitions:
                                if isinstance(obs, np.ndarray):
                                    team1_obs_buffer[idx].copy_(torch.tensor(obs, device=device))
                                else:
                                    team1_obs_buffer[idx].copy_(obs)
                                    
                                if isinstance(action, np.ndarray):
                                    team1_actions_buffer[idx].copy_(torch.tensor(action, device=device))
                                else:
                                    team1_actions_buffer[idx].copy_(action)
                                    
                                team1_rewards_buffer[idx] = episode_reward
                                idx += 1
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    batch_size = min(512, idx)
                    num_batches = (idx + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, idx)
                        
                        batch_obs_t = team1_obs_buffer[start_idx:end_idx]
                        batch_actions_t = team1_actions_buffer[start_idx:end_idx]
                        batch_returns_t = team1_rewards_buffer[start_idx:end_idx]
                        
                        agent_team1.optimizer.zero_grad(set_to_none=True)
                        
                        q_values = agent_team1.online_qnet(batch_obs_t)
                        q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                        
                        loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                        
                        agent_team1.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(agent_team1.online_qnet.parameters(), max_norm=10.0)
                        agent_team1.scaler.step(agent_team1.optimizer)
                        agent_team1.scaler.update()
                        
                        agent_team1.update_epsilon()
        
        # OTTIMIZZAZIONE: Pulizia memoria aggressiva
        # Rimuovi riferimenti espliciti
        team0_batch = None
        team1_batch = None
        
        # Forza garbage collection
        import gc
        gc.collect()
        
        # Libera memoria CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory_stats()  # Riorganizza allocazione
        
        # Traccia tempo di training
        train_time = time.time() - train_start_time
        train_times.append(train_time)
        print(f"  Training completato in {train_time:.2f} secondi")
        
        # Monitoraggio memoria GPU
        if torch.cuda.is_available():
            print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, "
                  f"{torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")
        
        # Tempo totale episodio
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        print(f"Episodio {ep+1} completato in {episode_time:.2f} secondi")
        
        # Salva checkpoint periodici
        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_team0_ep{ep+1}.pth")
            agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_team1_ep{ep+1}.pth")
        
        # Prepara per il prossimo episodio
        first_player = (first_player + 1) % 4

    # Salva checkpoint finali
    agent_team0.save_checkpoint(CHECKPOINT_PATH+"_team0.pth")
    agent_team1.save_checkpoint(CHECKPOINT_PATH+"_team1.pth")
    
    # Report statistiche finali
    print("\n=== Fine training ===")
    print("\nRiepilogo prestazioni:")
    print(f"Tempo medio per episodio: {sum(episode_times)/len(episode_times):.2f}s")
    print(f"Tempo medio per training: {sum(train_times)/len(train_times):.2f}s")
    print(f"Tempo medio per inferenza: {sum(inference_times)/len(inference_times):.2f}s")
    print(f"Percentuale di tempo in training: {sum(train_times)/sum(episode_times)*100:.1f}%")
    print(f"Percentuale di tempo in inferenza: {sum(inference_times)/sum(episode_times)*100:.1f}%")
    
    # Stampa epsilon finali
    print(f"Epsilon finale team 0: {agent_team0.epsilon:.4f}")
    print(f"Epsilon finale team 1: {agent_team1.epsilon:.4f}")


if __name__ == "__main__":
    # Esegui il training per pochi episodi per profilare
    train_agents(num_episodes=50000)
    
    # Stampa i risultati del profiling
    global_profiler.print_stats()
    
    # Genera un report dettagliato e salvalo su file
    report = global_profiler.generate_report("profiling_report.txt")