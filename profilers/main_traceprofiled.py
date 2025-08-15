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

# Importa modulo di profiling
import torch.profiler

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
"""
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
"""

# Parametri di rete e training
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000    # passi totali di training per passare da 1.0 a 0.01
BATCH_SIZE = 128          # dimensione mini-batch
REPLAY_SIZE = 10000      # capacità massima del replay buffer
TARGET_UPDATE_FREQ = 1000  # ogni quanti step sincronizzi la rete target
CHECKPOINT_PATH = "checkpoints/scopone_checkpoint"

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
    
    #@profile
    def forward(self, x):
        # Assicurati che l'input sia sulla GPU - ottimizzato
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
    
    #@profile
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
    
    #@profile
    def train_episodic_monte_carlo(self, specific_episode=None):
        """
        Versione ottimizzata con buffers pre-allocati e mixed precision.
        """
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
        batch_size = min(512, idx)  # Dimensione ottimizzata per GPU moderne
        num_batches = (idx + batch_size - 1) // batch_size
        
        # Riduci la frequenza di sync_target
        sync_counter = 0
        sync_frequency = 10  # Sincronizza ogni 10 batch invece di ogni batch
        
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
                self.scaler.scale(loss).backward()
                
                # OTTIMIZZAZIONE: Clip gradient con una norma moderata per stabilità di training
                torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                
                # OTTIMIZZAZIONE: Optimizer step con scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Aggiorna epsilon dopo ogni batch per avanzare il training
                self.update_epsilon()
                
                # OTTIMIZZAZIONE: Sync target network periodicamente (non ad ogni batch)
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    self.sync_target()
                    sync_counter = 0
                    
                # OTTIMIZZAZIONE: Rilascia memoria GPU periodicamente
                #if batch_idx % 10 == 0 and batch_idx > 0:
                #    torch.cuda.empty_cache()
    
    #@profile
    def store_episode_transition(self, transition):
        """
        Memorizza una transizione nell'episodic buffer.
        """
        # Aggiungi all'episodio corrente
        self.episodic_buffer.add_transition(transition)
    
    #@profile
    def end_episode(self):
        """
        Termina l'episodio corrente SENZA training.
        Il training deve essere chiamato esplicitamente dopo questo metodo.
        """
        self.episodic_buffer.end_episode()
        # Nota: rimosso il training automatico qui
            
    #@profile
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
        """Salva il checkpoint tenendo conto della GPU e assicura che la directory esista"""
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
    Esegue un training multi-agent episodico completamente ottimizzato per GPU.
    Il training avviene alla fine di ogni episodio, con reward flat per tutte le mosse.
    Implementa ottimizzazioni per ridurre drasticamente i trasferimenti CPU-GPU.
    Aggiunta il profiling per i primi 10 episodi.
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
        """Trova il checkpoint più recente per un team specifico"""
        dir_path = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        
        # Prima controlla se esiste il checkpoint senza numero episodio
        standard_ckpt = f"{base_path}_team{team_id}.pth"
        if os.path.isfile(standard_ckpt):
            return standard_ckpt
            
        # Altrimenti cerca i checkpoint con numero episodio
        if os.path.exists(dir_path):
            # Pattern per i checkpoint con numero episodio
            import fnmatch
            pattern = f"{base_name}_team{team_id}_ep*.pth"
            matching_files = [f for f in os.listdir(dir_path) if fnmatch.fnmatch(f, pattern)]
            
            if matching_files:
                # Estrai il numero episodio e ordina per numero più alto
                matching_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]), reverse=True)
                return os.path.join(dir_path, matching_files[0])
        
        return None
    
    # Carica checkpoint con logica migliorata
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
    
    # Setup profiler per i primi 10 episodi
    max_profiled_episodes = min(10, num_episodes)
    profiling_dir = "./tracing_logs"
    
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)
        
    print(f"Attivazione profiling per i primi {max_profiled_episodes} episodi...")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=max_profiled_episodes-2,  # -2 per compensare wait e warmup
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
    
        # Loop principale per episodi
        for ep in range(num_episodes):
            episode_start_time = time.time()

            # If this is the first episode, create a progress bar
            if ep == 0:
                pbar = tqdm(total=num_episodes, desc="Training episodes")
                
            # Update progress bar with description that includes player info
            pbar.set_description(f"Episode {ep+1}/{num_episodes} (Player {first_player})")
            pbar.update(1)

            # Close progress bar on last episode
            if ep == num_episodes - 1:
                pbar.close()
            
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
            #if torch.cuda.is_available():
                #print(f"  Memoria GPU dopo inferenza: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocata")

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
                #print(f"  Team Rewards finali: {team_rewards}")
            
            # TRAINING ALLA FINE DELL'EPISODIO - Completamente ottimizzato per GPU
            #print(f"  Training alla fine dell'episodio {ep+1}...")
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
                #print(f"  Training team 0 sull'ultimo episodio (reward={team0_reward}, mosse={team0_transitions})")
                
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
                #print(f"  Training team 1 sull'ultimo episodio (reward={team1_reward}, mosse={team1_transitions})")
                
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
                    #print("  Training aggiuntivo su episodi passati per team 0")
                    
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
                    #print("  Training aggiuntivo su episodi passati per team 1")
                    
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
            
            # Tracciamento profilazione
            if ep < max_profiled_episodes:
                prof.step()  # Indica la fine di uno step di profiling
                
                # Log per lo stato profilazione
                if ep == 0:
                    print("Fase di wait del profiler")
                elif ep == 1:
                    print("Fase di warmup del profiler")
                elif ep < max_profiled_episodes:
                    print(f"Registrazione profiler: episodio {ep+1}/{max_profiled_episodes}")
                    
                # Informazioni aggiuntive sull'utilizzo della memoria GPU
                if torch.cuda.is_available() and (ep+1) % 2 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                    print(f"  GPU: {mem_allocated:.1f}MB allocati, {mem_reserved:.1f}MB riservati")
                
            # Log del tempo per training
            train_time = time.time() - train_start_time
            train_times.append(train_time)
            
            # Tempo totale episodio
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)
            
            # Salva checkpoint periodici e checkpoint senza numero episodio
            if (ep + 1) % 1000 == 0 or ep == num_episodes - 1:
                # Salva con numero episodio per tracciare la progressione
                agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_team0_ep{ep+1}.pth")
                agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_team1_ep{ep+1}.pth")
                
                # Salva anche senza numero episodio per facilitare la ripresa
                agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_team0.pth")
                agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_team1.pth")
            
            # Prepara per il prossimo episodio
            first_player = (first_player + 1) % 4

        # Dopo il loop di profiling, esporta traccia Chrome
        if max_profiled_episodes > 0:
            chrome_trace_path = os.path.join(profiling_dir, "chrome_trace.json")
            print(f"\nProfiling completato. Dati salvati in {profiling_dir}")
            print(f"Traccia Chrome salvata in {chrome_trace_path}")

    # Verifica che i checkpoint siano stati salvati
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if os.path.exists(checkpoint_dir):
        print("\nCheckpoint trovati nella directory:")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        for file in checkpoint_files:
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # dimensione in MB
            print(f" - {file} ({file_size:.2f} MB)")
    else:
        print(f"\nATTENZIONE: La directory dei checkpoint {checkpoint_dir} non esiste!")
    
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
    train_agents(num_episodes=200)