#!/usr/bin/env python
# tensorboard_profiler.py - Version of main.py with TensorBoard profiling

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

# Import PyTorch profiler with TensorBoard support
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

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
    torch.cuda.memory.set_per_process_memory_fraction(0.95)

from environment import ScoponeEnvMA

# Parametri di rete e training
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
BATCH_SIZE = 128
REPLAY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
CHECKPOINT_PATH = "checkpoints/scopone_checkpoint"

# Directory per i logs di TensorBoard
TENSORBOARD_LOG_DIR = "./tensorboard_logs"
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

############################################################
# 1) EpisodicReplayBuffer (unchanged)
############################################################
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

############################################################
# 2) QNetwork (unchanged)
############################################################
class QNetwork(nn.Module):
    def __init__(self, obs_dim=10823, action_dim=80):
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
        
        # Sposta tutto il modello su GPU all'inizializzazione
        self.to(device)
        
        # Imposta opzioni CUDA per performance
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
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
        
        # Processa ogni sezione - versione in-place
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
        
        # Elabora le features combinate - versione in-place
        final_features = F.relu(self.combiner[0](combined), inplace=True)
        
        # Calcola i valori delle azioni
        action_values = self.action_head(final_features)
        
        return action_values

############################################################
# 3) DQNAgent (unchanged)
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
        torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    def pick_action(self, obs, valid_actions, env):
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
            # OTTIMIZZAZIONE: Converti tutti gli input in tensori GPU in un'unica operazione
            # e riusa il buffer pre-allocato se disponibile
            if hasattr(self, 'valid_actions_buffer') and len(valid_actions) <= self.valid_actions_buffer.size(0):
                valid_actions_t = self.valid_actions_buffer[:len(valid_actions)]
                for i, va in enumerate(valid_actions):
                    if isinstance(va, np.ndarray):
                        valid_actions_t[i].copy_(torch.tensor(va, device=device))
                    else:
                        valid_actions_t[i].copy_(va)
            else:
                # Creazione del buffer se non esiste
                if not hasattr(self, 'valid_actions_buffer') or len(valid_actions) > self.valid_actions_buffer.size(0):
                    self.valid_actions_buffer = torch.zeros((max(100, len(valid_actions)), 80), 
                                                        dtype=torch.float32, device=device)
                valid_actions_t = torch.tensor(np.stack(valid_actions), 
                                            dtype=torch.float32, device=device)
                
            with torch.no_grad():
                # OTTIMIZZAZIONE: Riusa il buffer per observation se possibile
                if hasattr(self, 'obs_buffer'):
                    obs_t = self.obs_buffer
                    if isinstance(obs, np.ndarray):
                        obs_t.copy_(torch.tensor(obs, device=device).unsqueeze(0))
                    else:
                        obs_t.copy_(obs.unsqueeze(0))
                else:
                    self.obs_buffer = torch.zeros((1, len(obs)), dtype=torch.float32, device=device)
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                # OTTIMIZZAZIONE: Usa mixed precision per accelerare l'inferenza
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    action_values = self.online_qnet(obs_t)
                    q_values = torch.sum(action_values.view(1, 1, 80) * valid_actions_t.view(-1, 1, 80), dim=2).squeeze()
                
                best_action_idx = torch.argmax(q_values).item()
            
            return valid_actions[best_action_idx]
    
    def train_episodic_monte_carlo(self, specific_episode=None):
        # Determina quali episodi processare
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            return  # Nessun episodio disponibile
        
        # OTTIMIZZAZIONE: Usa buffer pre-allocati se possibile
        max_transitions = sum(len(episode) for episode in episodes_to_process)
        
        # Crea o ridimensiona i buffer se necessario
        if not hasattr(self, 'train_obs_buffer') or max_transitions > self.train_obs_buffer.size(0):
            self.train_obs_buffer = torch.zeros((max_transitions, 10823), dtype=torch.float32, device=device)
            self.train_actions_buffer = torch.zeros((max_transitions, 80), dtype=torch.float32, device=device)
            self.train_returns_buffer = torch.zeros(max_transitions, dtype=torch.float32, device=device)
        
        # Riempi i buffer in modo efficiente
        idx = 0
        for episode in episodes_to_process:
            if not episode:
                continue
                    
            # Ottieni la reward finale dall'ultima transizione dell'episodio
            final_reward = episode[-1][2] if episode else 0.0
            
            for obs, action, _, _, _, _ in episode:
                # Copia direttamente nel buffer per evitare creazioni di tensori intermedie
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
            return  # Nessuna transizione da processare
        
        # Usa slices dei buffer per il training
        all_obs_t = self.train_obs_buffer[:idx]
        all_actions_t = self.train_actions_buffer[:idx]
        all_returns_t = self.train_returns_buffer[:idx]
        
        # Aumenta batch_size per sfruttare meglio la GPU
        batch_size = min(512, idx)
        num_batches = (idx + batch_size - 1) // batch_size
        
        # Riduci la frequenza di sync_target
        sync_counter = 0
        sync_frequency = 10
        
        # Traccia le metriche per diagnostica
        total_loss = 0.0
        batch_count = 0
        
        # OTTIMIZZAZIONE: Usa mixed precision in modo più efficiente con float16
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, idx)
                
                # Prendi slices dei tensori già sulla GPU (evita copie)
                batch_obs_t = all_obs_t[start_idx:end_idx]
                batch_actions_t = all_actions_t[start_idx:end_idx]
                batch_returns_t = all_returns_t[start_idx:end_idx]
                
                # OTTIMIZZAZIONE: Zero gradients - usa set_to_none=True per maggiore efficienza di memoria
                self.optimizer.zero_grad(set_to_none=True)
                
                # OTTIMIZZAZIONE: Forward pass con kernel fusion dove possibile
                q_values = self.online_qnet(batch_obs_t)
                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                
                # OTTIMIZZAZIONE: Loss con mixed precision - usa reduction='mean' per stabilità numerica
                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                
                # Traccia la loss per diagnostica
                total_loss += loss.item()
                batch_count += 1
                
                # OTTIMIZZAZIONE: Backward e optimizer step con gradient scaling per mixed precision
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # OTTIMIZZAZIONE: Clip gradient con una norma moderata per stabilità di training
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    
                    # OTTIMIZZAZIONE: Optimizer step con scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    self.optimizer.step()
                
                # Aggiorna epsilon dopo ogni batch per avanzare il training
                self.update_epsilon()
                
                # OTTIMIZZAZIONE: Sync target network periodicamente (non ad ogni batch)
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    self.sync_target()
                    sync_counter = 0
                    
                # OTTIMIZZAZIONE: Rilascia memoria GPU periodicamente
                if batch_idx % 10 == 0 and batch_idx > 0:
                    torch.cuda.empty_cache()
    
    def store_episode_transition(self, transition):
        # Aggiungi all'episodio corrente
        self.episodic_buffer.add_transition(transition)
    
    def end_episode(self):
        # Termina l'episodio corrente SENZA training
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
        # Crea la directory se non esiste
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[DQNAgent] Creata directory per checkpoint: {directory}")
        
        try:
            torch.save({
                "online_state_dict": self.online_qnet.state_dict(),
                "target_state_dict": self.target_qnet.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps
            }, filename)
            print(f"[DQNAgent] Checkpoint salvato: {filename}")
        except Exception as e:
            print(f"[DQNAgent] ERRORE nel salvataggio del checkpoint {filename}: {e}")

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename, map_location=device)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Checkpoint caricato da {filename}")

############################################################
# 4) Modified train_agents function with TensorBoard profiling
############################################################
def train_agents_with_profiler(num_episodes=20, profile_steps=40):
    """
    Versione modificata di train_agents che utilizza il TensorBoard profiler.
    
    Args:
        num_episodes: Numero totale di episodi di training
        profile_steps: Numero di step da profilare
    """
    print(f"Training con profiler TensorBoard: {num_episodes} episodi, profiling per {profile_steps} step")
    
    # Configurazione ottimale GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
        torch.cuda.memory_stats()

    # Crea la directory dei checkpoint se non esiste
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_dir):
        print(f"Creazione directory per checkpoint: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    
    # Crea gli agenti
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Funzione di utilità per trovare il checkpoint più recente
    def find_latest_checkpoint(base_path, team_id):
        dir_path = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        
        standard_ckpt = f"{base_path}_team{team_id}.pth"
        if os.path.isfile(standard_ckpt):
            return standard_ckpt
            
        if os.path.exists(dir_path):
            import fnmatch
            pattern = f"{base_name}_team{team_id}_ep*.pth"
            matching_files = [f for f in os.listdir(dir_path) if fnmatch.fnmatch(f, pattern)]
            
            if matching_files:
                matching_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]), reverse=True)
                return os.path.join(dir_path, matching_files[0])
        
        return None
    
    # Carica checkpoint
    print(f"Cercando checkpoint più recenti per i team...")
    
    # Team 0
    team0_ckpt = find_latest_checkpoint(CHECKPOINT_PATH, 0)
    if team0_ckpt:
        try:
            print(f"Trovato checkpoint per team 0: {team0_ckpt}")
            agent_team0.load_checkpoint(team0_ckpt)
        except Exception as e:
            print(f"ERRORE nel caricamento del checkpoint team 0: {e}")
    else:
        print(f"Nessun checkpoint trovato per team 0")
    
    # Team 1
    team1_ckpt = find_latest_checkpoint(CHECKPOINT_PATH, 1)
    if team1_ckpt:
        try:
            print(f"Trovato checkpoint per team 1: {team1_ckpt}")
            agent_team1.load_checkpoint(team1_ckpt)
        except Exception as e:
            print(f"ERRORE nel caricamento del checkpoint team 1: {e}")
    else:
        print(f"Nessun checkpoint trovato per team 1")

    # Variabili di controllo
    first_player = 0
    global_step = 0
    
    # Monitoraggio prestazioni
    episode_times = []
    train_times = []
    inference_times = []
    
    # OTTIMIZZAZIONE: Pre-alloca buffer
    max_transitions = 40
    team0_obs_buffer = torch.zeros((max_transitions, 10823), dtype=torch.float32, device=device)
    team0_actions_buffer = torch.zeros((max_transitions, 80), dtype=torch.float32, device=device)
    team0_rewards_buffer = torch.zeros(max_transitions, dtype=torch.float32, device=device)
    
    team1_obs_buffer = torch.zeros_like(team0_obs_buffer)
    team1_actions_buffer = torch.zeros_like(team0_actions_buffer)
    team1_rewards_buffer = torch.zeros_like(team0_rewards_buffer)
    
    # Configurazione profiler con TensorBoard
    # Definiamo le attività da monitorare
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    # Definiamo uno scheduler ottimizzato per profiling step specifici
    # wait=0: inizia a profilare subito
    # warmup=1: fa un po' di warmup prima di registrare
    # active=profile_steps: registra per N steps
    # repeat=1: ripeti il ciclo una volta
    schedule = torch.profiler.schedule(
        wait=0,
        warmup=1,
        active=profile_steps,
        repeat=1)
    
    # Trace handler per TensorBoard
    tb_trace_handler = tensorboard_trace_handler(TENSORBOARD_LOG_DIR)
    
    # Inizializza il profiler
    with profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=tb_trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        # Loop principale per episodi (limitato dal profiler)
        steps_profiled = 0
        pbar = tqdm(total=num_episodes, desc="Training episodes")
        
        for ep in range(num_episodes):
            episode_start_time = time.time()
            
            # Update progress bar
            pbar.set_description(f"Episode {ep+1}/{num_episodes} (Player {first_player})")
            pbar.update(1)
            
            # Crea ambiente e inizializza
            with record_function("Environment_Setup"):
                env = ScoponeEnvMA()
                env.current_player = first_player

            # Inizializza buffer episodici
            with record_function("Initialize_Episode"):
                agent_team0.start_episode()
                agent_team1.start_episode()

            # Stato iniziale
            with record_function("Initial_State"):
                done = False
                obs_current = env._get_observation(env.current_player)
                
                # Assicura che obs_current sia un array numpy
                if torch.is_tensor(obs_current):
                    obs_current = obs_current.cpu().numpy()
            
            # Conteggi delle transition per ciascun team
            team0_transitions = 0
            team1_transitions = 0

            # Loop principale della partita
            inference_start = time.time()
            with record_function("Game_Loop"):
                while not done:
                    cp = env.current_player
                    team_id = 0 if cp in [0,2] else 1
                    agent = agent_team0 if team_id==0 else agent_team1

                    # Ottieni azioni valide
                    with record_function("Get_Valid_Actions"):
                        valid_acts = env.get_valid_actions()
                    
                    if not valid_acts:
                        break
                    
                    # Scelta azione
                    with record_function("Pick_Action"):
                        # Conversione efficiente a tensori
                        if len(valid_acts) > 0:
                            if isinstance(valid_acts[0], np.ndarray):
                                valid_acts_t = torch.tensor(np.stack(valid_acts), dtype=torch.float32, device=device)
                                action = agent.pick_action(obs_current, valid_acts, env)
                            else:
                                action = agent.pick_action(obs_current, valid_acts, env)
                    
                    # Esegui azione sull'ambiente
                    with record_function("Environment_Step"):
                        next_obs, reward, done, info = env.step(action)
                        
                        # Assicura che next_obs sia numpy array
                        if torch.is_tensor(next_obs):
                            next_obs = next_obs.cpu().numpy()
                    
                    # Memorizza transition
                    with record_function("Store_Transition"):
                        global_step += 1
                        next_valid = env.get_valid_actions() if not done else []
                        transition = (obs_current, action, reward, next_obs, done, next_valid)
                        
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

            # Termina episodi
            with record_function("End_Episodes"):
                agent_team0.end_episode()
                agent_team1.end_episode()
            
            # Ottieni le reward finali
            team0_reward = 0.0
            team1_reward = 0.0
            if "team_rewards" in info:
                team_rewards = info["team_rewards"]
                team0_reward = team_rewards[0]
                team1_reward = team_rewards[1]
            
            # TRAINING ALLA FINE DELL'EPISODIO
            with record_function("Training"):
                train_start_time = time.time()
                
                # Team 0 training batch
                with record_function("Team0_Training"):
                    if agent_team0.episodic_buffer.episodes:
                        last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
                        if last_episode_team0:
                            # Training team 0
                            agent_team0.train_episodic_monte_carlo()
                
                # Team 1 training
                with record_function("Team1_Training"):
                    if agent_team1.episodic_buffer.episodes:
                        last_episode_team1 = agent_team1.episodic_buffer.episodes[-1]
                        if last_episode_team1:
                            # Training team 1
                            agent_team1.train_episodic_monte_carlo()
            
                # Sync target networks
                if global_step % TARGET_UPDATE_FREQ == 0:
                    with record_function("Sync_Target"):
                        agent_team0.sync_target()
                        agent_team1.sync_target()
            
                train_time = time.time() - train_start_time
                train_times.append(train_time)
            
            # Pulizia memoria
            with record_function("Memory_Cleanup"):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Tempo totale episodio
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)
            
            # Salva checkpoint periodici
            if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
                with record_function("Save_Checkpoints"):
                    agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_profiler_team0_ep{ep+1}.pth")
                    agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_profiler_team1_ep{ep+1}.pth")
            
            # Avanza il profiler
            try:
                prof.step()
                steps_profiled += 1
                
                # Se abbiamo profilato abbastanza step, esci dal loop
                if steps_profiled >= profile_steps:
                    print(f"Completato profiling di {steps_profiled} step. Terminando il training.")
                    break
            except Exception as e:
                print(f"Errore durante prof.step(): {e}")
                # Continuiamo comunque il training anche se il profiler fallisce
            
            # Prepara per il prossimo episodio
            first_player = (first_player + 1) % 4
        
        pbar.close()
    
    # Profiler viene automaticamente chiuso alla fine del blocco 'with'
    
    # Report statistiche finali
    avg_episode_time = sum(episode_times) / len(episode_times) if episode_times else 0
    avg_train_time = sum(train_times) / len(train_times) if train_times else 0
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    
    print("\n=== Fine training ===")
    print("\nRiepilogo prestazioni:")
    print(f"Tempo medio per episodio: {avg_episode_time:.2f}s")
    print(f"Tempo medio per training: {avg_train_time:.2f}s")
    print(f"Tempo medio per inferenza: {avg_inference_time:.2f}s")
    
    if avg_episode_time > 0:
        print(f"Percentuale di tempo in training: {avg_train_time/avg_episode_time*100:.1f}%")
        print(f"Percentuale di tempo in inferenza: {avg_inference_time/avg_episode_time*100:.1f}%")
    
    # Informazioni su come visualizzare i risultati
    print(f"\nProfiling completato. Per visualizzare i risultati:")
    print(f"1. Installa TensorBoard se non l'hai già fatto:")
    print(f"   pip install tensorboard")
    print(f"2. Avvia TensorBoard con il comando:")
    print(f"   tensorboard --logdir={TENSORBOARD_LOG_DIR}")
    print(f"3. Apri il browser su http://localhost:6006")
    print(f"4. Vai alla scheda 'PYTORCH_PROFILER' per visualizzare i dettagli del profiling")

if __name__ == "__main__":
    # Esegui il training con profiler
    # Il numero di episodi è il massimo che verrà eseguito, ma il profiler potrebbe fermarsi prima
    # Il numero di step da profilare è configurabile in base alle esigenze
    train_agents_with_profiler(num_episodes=20, profile_steps=10)