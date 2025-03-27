# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import os
#from line_profiler import LineProfiler, profile, global_profiler
import time
from tqdm import tqdm

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

# Parametri di rete e training
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000    # passi totali di training per passare da 1.0 a 0.01
# Nota: BATCH_SIZE non più utilizzato - processiamo tutti i dati in un'unica passata
REPLAY_SIZE = 10000      # capacità massima del replay buffer
TARGET_UPDATE_FREQ = 1000  # ogni quanti step sincronizzi la rete target (ora usato solo per sincronizzazione globale)
CHECKPOINT_PATH = "checkpoints/scopone_checkpoint"

# Configura la frammentazione della memoria CUDA per dataset grandi
if torch.cuda.is_available():
    # Prova ad allocare memoria in blocchi più grandi per ridurre la frammentazione
    torch.cuda.empty_cache()
    torch.cuda.memory_stats()

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
    def __init__(self, obs_dim=10823, action_dim=80):
        super().__init__()
        # Backbone potenziato con neuroni quadruplicati
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Sezioni aggiuntive potenziate
        self.hand_table_processor = nn.Sequential(
            nn.Linear(83, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.captured_processor = nn.Sequential(
            nn.Linear(82, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.stats_processor = nn.Sequential(
            nn.Linear(334, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.history_processor = nn.Sequential(
            nn.Linear(10320, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(256 + 128*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(512, action_dim)
        
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
        x4 = F.relu(self.backbone[6](x3), inplace=True)
        backbone_features = F.relu(self.backbone[8](x4), inplace=True)
        
        # Divide l'input in sezioni semantiche
        hand_table = x[:, :83]
        captured = x[:, 83:165]
        history = x[:, 169:10489]
        stats = x[:, 10489:]
        
        # Processa ogni sezione - versione in-place
        hand_table_f1 = F.relu(self.hand_table_processor[0](hand_table), inplace=True)
        hand_table_features = F.relu(self.hand_table_processor[2](hand_table_f1), inplace=True)
        
        captured_f1 = F.relu(self.captured_processor[0](captured), inplace=True)
        captured_features = F.relu(self.captured_processor[2](captured_f1), inplace=True)
        
        history_f1 = F.relu(self.history_processor[0](history), inplace=True)
        history_f2 = F.relu(self.history_processor[2](history_f1), inplace=True)
        history_features = F.relu(self.history_processor[4](history_f2), inplace=True)
        
        stats_f1 = F.relu(self.stats_processor[0](stats), inplace=True)
        stats_features = F.relu(self.stats_processor[2](stats_f1), inplace=True)
        
        # Combina tutte le features
        combined = torch.cat([
            backbone_features,
            hand_table_features,
            captured_features,
            history_features,
            stats_features
        ], dim=1)
        
        # Elabora le features combinate - versione in-place
        combined_f1 = F.relu(self.combiner[0](combined), inplace=True)
        final_features = F.relu(self.combiner[2](combined_f1), inplace=True)
        
        # Calcola i valori delle azioni
        action_values = self.action_head(final_features)
        
        return action_values

############################################################
# 3) DQNAgent con target network + replay
############################################################

class SelfPlayDQNAgent:
    def __init__(self):
        """
        Inizializza un singolo agente DQN per self-play ottimizzato.
        Una singola rete neurale, ma gestisce due team separati come nell'originale.
        """
        self.online_qnet = QNetwork()
        self.target_qnet = QNetwork()
        self.sync_target()
        
        self.optimizer = optim.Adam(self.online_qnet.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.train_steps = 0
        
        # Buffer separati per ogni TEAM (non per giocatore) - come nell'originale
        self.team0_buffer = EpisodicReplayBuffer()
        self.team1_buffer = EpisodicReplayBuffer()
        
        # Aggiunte per ottimizzazione GPU
        torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Buffer pre-allocati per ottimizzazione
        self.valid_actions_buffer = torch.zeros((100, 80), dtype=torch.float32, device=device)
        self.obs_buffer = torch.zeros((1, 10823), dtype=torch.float32, device=device)
        
        # Buffer per training - allocati una volta sola
        self.train_obs_buffer_team0 = torch.zeros((40, 10823), dtype=torch.float32, device=device)
        self.train_actions_buffer_team0 = torch.zeros((40, 80), dtype=torch.float32, device=device)
        self.train_returns_buffer_team0 = torch.zeros(40, dtype=torch.float32, device=device)
        
        self.train_obs_buffer_team1 = torch.zeros_like(self.train_obs_buffer_team0)
        self.train_actions_buffer_team1 = torch.zeros_like(self.train_actions_buffer_team0)
        self.train_returns_buffer_team1 = torch.zeros_like(self.train_returns_buffer_team0)
    
    def pick_action(self, obs, valid_actions, env, current_player):
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
            # OTTIMIZZAZIONE: Riusa buffer pre-allocati e minimizza trasferimenti
            if len(valid_actions) > self.valid_actions_buffer.size(0):
                self.valid_actions_buffer = torch.zeros((len(valid_actions), 80), 
                                                      dtype=torch.float32, device=device)
            
            valid_actions_t = self.valid_actions_buffer[:len(valid_actions)]
            for i, va in enumerate(valid_actions):
                if isinstance(va, np.ndarray):
                    valid_actions_t[i].copy_(torch.tensor(va, device=device))
                else:
                    valid_actions_t[i].copy_(va)
            
            with torch.no_grad():
                # Usa buffer pre-allocato per l'osservazione
                if isinstance(obs, np.ndarray):
                    self.obs_buffer.copy_(torch.tensor(obs, device=device).unsqueeze(0))
                else:
                    self.obs_buffer.copy_(obs.unsqueeze(0))
                
                # Mixed precision per inferenza
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    action_values = self.online_qnet(self.obs_buffer)
                    q_values = torch.sum(action_values.view(1, 1, 80) * valid_actions_t.view(-1, 1, 80), dim=2).squeeze()
                
                best_action_idx = torch.argmax(q_values).item()
            
            return valid_actions[best_action_idx]
    
    def train_on_episode(self, team_id):
        """
        Training ottimizzato per un singolo team, più simile all'implementazione originale.
        """
        buffer = self.team0_buffer if team_id == 0 else self.team1_buffer
        train_obs_buffer = self.train_obs_buffer_team0 if team_id == 0 else self.train_obs_buffer_team1
        train_actions_buffer = self.train_actions_buffer_team0 if team_id == 0 else self.train_actions_buffer_team1
        train_returns_buffer = self.train_returns_buffer_team0 if team_id == 0 else self.train_returns_buffer_team1
        
        # Salta se non ci sono episodi
        if not buffer.episodes:
            return False
        
        # Usa solo l'ultimo episodio per allenamento standard
        last_episode = buffer.episodes[-1]
        if not last_episode:
            return False
        
        # Ottieni reward finale
        final_reward = last_episode[-1][2]
        
        # OTTIMIZZAZIONE: Prepara il batch in modo efficiente
        ep_len = len(last_episode)
        if ep_len > train_obs_buffer.size(0):
            if team_id == 0:
                self.train_obs_buffer_team0 = torch.zeros((ep_len, 10823), dtype=torch.float32, device=device)
                self.train_actions_buffer_team0 = torch.zeros((ep_len, 80), dtype=torch.float32, device=device)
                self.train_returns_buffer_team0 = torch.zeros(ep_len, dtype=torch.float32, device=device)
                train_obs_buffer = self.train_obs_buffer_team0
                train_actions_buffer = self.train_actions_buffer_team0
                train_returns_buffer = self.train_returns_buffer_team0
            else:
                self.train_obs_buffer_team1 = torch.zeros((ep_len, 10823), dtype=torch.float32, device=device)
                self.train_actions_buffer_team1 = torch.zeros((ep_len, 80), dtype=torch.float32, device=device)
                self.train_returns_buffer_team1 = torch.zeros(ep_len, dtype=torch.float32, device=device)
                train_obs_buffer = self.train_obs_buffer_team1
                train_actions_buffer = self.train_actions_buffer_team1
                train_returns_buffer = self.train_returns_buffer_team1
        
        # Riempi i buffer
        all_obs, all_actions, _, _, _, _ = zip(*last_episode)
        
        for i, (obs, action) in enumerate(zip(all_obs, all_actions)):
            if isinstance(obs, np.ndarray):
                train_obs_buffer[i].copy_(torch.tensor(obs, device=device))
            else:
                train_obs_buffer[i].copy_(obs)
                
            if isinstance(action, np.ndarray):
                train_actions_buffer[i].copy_(torch.tensor(action, device=device))
            else:
                train_actions_buffer[i].copy_(action)
                
            train_returns_buffer[i] = final_reward
        
        # OTTIMIZZAZIONE: Training con Mixed Precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Reset gradients in modo efficiente
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass ottimizzato
            q_values = self.online_qnet(train_obs_buffer[:ep_len])
            q_values_for_actions = torch.sum(q_values * train_actions_buffer[:ep_len], dim=1)
            
            # Loss calculation
            loss = nn.MSELoss()(q_values_for_actions, train_returns_buffer[:ep_len])
            
            # Backward con scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
            
            # Step con scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Aggiorna epsilon una sola volta per episodio
            self.update_epsilon()
        
        return True
    
    def train_on_past_episodes(self, team_id):
        """
        Training efficiente su episodi passati per un team specifico.
        """
        buffer = self.team0_buffer if team_id == 0 else self.team1_buffer
        train_obs_buffer = self.train_obs_buffer_team0 if team_id == 0 else self.train_obs_buffer_team1
        train_actions_buffer = self.train_actions_buffer_team0 if team_id == 0 else self.train_actions_buffer_team1
        train_returns_buffer = self.train_returns_buffer_team0 if team_id == 0 else self.train_returns_buffer_team1
        
        # Verifica che ci siano abbastanza episodi
        if len(buffer.episodes) <= 3:
            return False
            
        # Seleziona episodi passati
        prev_episodes = buffer.get_previous_episodes()
        past_episodes = random.sample(prev_episodes, min(3, len(prev_episodes)))
        
        # Calcola numero totale di transizioni
        total_transitions = sum(len(episode) for episode in past_episodes)
        if total_transitions == 0:
            return False
            
        # Ridimensiona buffer se necessario
        if total_transitions > train_obs_buffer.size(0):
            if team_id == 0:
                self.train_obs_buffer_team0 = torch.zeros((total_transitions, 10823), dtype=torch.float32, device=device)
                self.train_actions_buffer_team0 = torch.zeros((total_transitions, 80), dtype=torch.float32, device=device)
                self.train_returns_buffer_team0 = torch.zeros(total_transitions, dtype=torch.float32, device=device)
                train_obs_buffer = self.train_obs_buffer_team0
                train_actions_buffer = self.train_actions_buffer_team0
                train_returns_buffer = self.train_returns_buffer_team0
            else:
                self.train_obs_buffer_team1 = torch.zeros((total_transitions, 10823), dtype=torch.float32, device=device)
                self.train_actions_buffer_team1 = torch.zeros((total_transitions, 80), dtype=torch.float32, device=device)
                self.train_returns_buffer_team1 = torch.zeros(total_transitions, dtype=torch.float32, device=device)
                train_obs_buffer = self.train_obs_buffer_team1
                train_actions_buffer = self.train_actions_buffer_team1
                train_returns_buffer = self.train_returns_buffer_team1
        
        # Prepara il mega-batch
        idx = 0
        for episode in past_episodes:
            if not episode or len(episode) == 0:
                continue
                
            # Reward dell'episodio
            episode_reward = episode[-1][2]
            
            for obs, action, _, _, _, _ in episode:
                # Trasferimento diretto su GPU
                if isinstance(obs, np.ndarray):
                    train_obs_buffer[idx].copy_(torch.tensor(obs, device=device))
                else:
                    train_obs_buffer[idx].copy_(obs)
                    
                if isinstance(action, np.ndarray):
                    train_actions_buffer[idx].copy_(torch.tensor(action, device=device))
                else:
                    train_actions_buffer[idx].copy_(action)
                    
                train_returns_buffer[idx] = episode_reward
                idx += 1
        
        if idx == 0:
            return False
            
        # Training in batch
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            self.optimizer.zero_grad(set_to_none=True)
            
            q_values = self.online_qnet(train_obs_buffer[:idx])
            q_values_for_actions = torch.sum(q_values * train_actions_buffer[:idx], dim=1)
            
            loss = nn.MSELoss()(q_values_for_actions, train_returns_buffer[:idx])
            
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.update_epsilon()
        
        return True
    
    def store_transition(self, transition, team_id):
        """
        Memorizza una transizione nel buffer del team specifico.
        """
        if team_id == 0:
            self.team0_buffer.add_transition(transition)
        else:
            self.team1_buffer.add_transition(transition)
    
    def start_episode(self):
        """Inizia un nuovo episodio per entrambi i team."""
        self.team0_buffer.start_episode()
        self.team1_buffer.start_episode()
    
    def end_episode(self):
        """Termina l'episodio corrente per entrambi i team."""
        self.team0_buffer.end_episode()
        self.team1_buffer.end_episode()
    
    def sync_target(self):
        """Sincronizza i pesi della rete target con quella online."""
        self.target_qnet.load_state_dict(self.online_qnet.state_dict())
    
    def update_epsilon(self):
        """Aggiorna epsilon per esplorazione."""
        self.epsilon = max(EPSILON_END, EPSILON_START - (self.train_steps / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
        self.train_steps += 1
    
    def save_checkpoint(self, filename):
        """Salva il checkpoint."""
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[SelfPlayDQNAgent] Creata directory per checkpoint: {directory}")
        
        try:
            torch.save({
                "online_state_dict": self.online_qnet.state_dict(),
                "target_state_dict": self.target_qnet.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps
            }, filename)
            print(f"[SelfPlayDQNAgent] Checkpoint salvato: {filename}")
        except Exception as e:
            print(f"[SelfPlayDQNAgent] ERRORE nel salvataggio del checkpoint {filename}: {e}")
    
    def load_checkpoint(self, filename):
        """Carica il checkpoint."""
        ckpt = torch.load(filename, map_location=device)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[SelfPlayDQNAgent] Checkpoint loaded from {filename}")
        
############################################################
# 4) Multi-agent training
############################################################

def train_selfplay_optimized(num_episodes=10):
    """
    Funzione di training self-play ottimizzata, strutturata come l'originale
    ma usando un singolo modello per entrambi i team.
    """
    # Configurazione GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
        torch.cuda.memory_stats()

    # Crea directory per checkpoint
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_dir):
        print(f"Creazione directory per checkpoint: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    
    # Crea un singolo agente self-play
    agent = SelfPlayDQNAgent()

    # Funzione per trovare checkpoint
    def find_latest_checkpoint(base_path):
        dir_path = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        
        # Prima controlla checkpoint standard
        standard_ckpt = f"{base_path}.pth"
        if os.path.isfile(standard_ckpt):
            return standard_ckpt
            
        # Poi checkpoint con numero episodio
        if os.path.exists(dir_path):
            import fnmatch
            pattern = f"{base_name}_ep*.pth"
            matching_files = [f for f in os.listdir(dir_path) if fnmatch.fnmatch(f, pattern)]
            
            if matching_files:
                matching_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]), reverse=True)
                return os.path.join(dir_path, matching_files[0])
        
        return None
    
    # Carica checkpoint se disponibile
    print(f"Cercando checkpoint più recente...")
    ckpt_path = find_latest_checkpoint(CHECKPOINT_PATH)
    
    if ckpt_path:
        try:
            print(f"Trovato checkpoint: {ckpt_path}")
            agent.load_checkpoint(ckpt_path)
        except Exception as e:
            print(f"ERRORE nel caricamento del checkpoint: {e}")
    else:
        print(f"Nessun checkpoint trovato, inizializzazione con pesi casuali")

    # Inizializza variabili di controllo
    first_player = 0
    global_step = 0
    
    # Monitoraggio prestazioni
    episode_times = []
    train_times = []
    inference_times = []
    
    # Preallocazione buffer per carte
    card_buffer = {}
    for suit in ['denari', 'coppe', 'spade', 'bastoni']:
        for rank in range(1, 11):
            card_buffer[(rank, suit)] = torch.zeros(80, dtype=torch.float32, device=device)
    
    # Loop principale episodi
    for ep in range(num_episodes):
        episode_start_time = time.time()

        # Progress bar
        if ep == 0:
            pbar = tqdm(total=num_episodes, desc="Training episodes")
        pbar.set_description(f"Episode {ep+1}/{num_episodes} (Player {first_player})")
        pbar.update(1)
        if ep == num_episodes - 1:
            pbar.close()
        
        # Inizializza ambiente
        env = ScoponeEnvMA()
        env.current_player = first_player

        # Inizializza buffer episodici
        agent.start_episode()

        # Stato iniziale
        done = False
        obs_current = env._get_observation(env.current_player)
        
        # Converti a numpy se necessario
        if torch.is_tensor(obs_current):
            obs_current = obs_current.cpu().numpy()
            
        # Contatori per team
        team0_transitions = 0
        team1_transitions = 0

        # Loop principale partita
        inference_start = time.time()
        while not done:
            current_player = env.current_player
            team_id = 0 if current_player in [0,2] else 1

            # Ottieni azioni valide
            valid_acts = env.get_valid_actions()
            if not valid_acts:
                break
            
            # Scelta azione ottimizzata
            if len(valid_acts) > 0:
                if isinstance(valid_acts[0], np.ndarray):
                    action = agent.pick_action(obs_current, valid_acts, env, current_player)
                else:
                    action = agent.pick_action(obs_current, valid_acts, env, current_player)
            
            # Esegui azione
            next_obs, reward, done, info = env.step(action)
            
            # Converti a numpy se necessario
            if torch.is_tensor(next_obs):
                next_obs = next_obs.cpu().numpy()
                
            global_step += 1

            # Prepara transition
            next_valid = env.get_valid_actions() if not done else []
            transition = (obs_current, action, reward, next_obs, done, next_valid)
            
            # Memorizza transition nel buffer del team corretto
            agent.store_transition(transition, team_id)
            
            # Aggiorna contatori
            if team_id == 0:
                team0_transitions += 1
            else:
                team1_transitions += 1
            
            # Prepara per prossima iterazione
            obs_current = next_obs
            
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        # Termina episodi
        agent.end_episode()
        
        # Ottieni reward finali
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
        
        # TRAINING ALLA FINE DELL'EPISODIO
        train_start_time = time.time()
        
        # Training per team 0
        team0_trained = agent.train_on_episode(0)
        
        # Training per team 1
        team1_trained = agent.train_on_episode(1)
        
        # Sincronizzazione target network
        if global_step % TARGET_UPDATE_FREQ == 0:
            agent.sync_target()
        
        # Training su episodi passati
        if ep % 5 == 0 and ep > 0:
            agent.train_on_past_episodes(0)  # Team 0
            agent.train_on_past_episodes(1)  # Team 1
        
        # Traccia tempo training
        train_time = time.time() - train_start_time
        train_times.append(train_time)
        
        # Tempo totale episodio
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        
        # Salva checkpoint
        if (ep + 1) % 5000 == 0 or ep == num_episodes - 1:
            agent.save_checkpoint(f"{CHECKPOINT_PATH}_ep{ep+1}.pth")
            agent.save_checkpoint(f"{CHECKPOINT_PATH}.pth")
        
        # Prepara per prossimo episodio
        first_player = (first_player + 1) % 4

    # Verifica checkpoint
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if os.path.exists(checkpoint_dir):
        print("\nCheckpoint trovati nella directory:")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        for file in checkpoint_files:
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f" - {file} ({file_size:.2f} MB)")
    else:
        print(f"\nATTENZIONE: La directory dei checkpoint {checkpoint_dir} non esiste!")
    
    # Report finale
    print("\n=== Fine training ===")
    print("\nRiepilogo prestazioni:")
    print(f"Tempo medio per episodio: {sum(episode_times)/len(episode_times):.2f}s")
    print(f"Tempo medio per training: {sum(train_times)/len(train_times):.2f}s")
    print(f"Tempo medio per inferenza: {sum(inference_times)/len(inference_times):.2f}s")
    print(f"Percentuale di tempo in training: {sum(train_times)/sum(episode_times)*100:.1f}%")
    print(f"Percentuale di tempo in inferenza: {sum(inference_times)/sum(episode_times)*100:.1f}%")
    
    # Epsilon finale
    print(f"Epsilon finale: {agent.epsilon:.4f}")

# Modifica per utilizzare la nuova versione ottimizzata 
if __name__ == "__main__":
    train_selfplay_optimized(num_episodes=500000)