#!/usr/bin/env python
# profile_timing_fixed.py - Profiler per tempi di calcolo e trasferimenti (versione corretta)

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
import fnmatch
from pathlib import Path
import traceback

# Import PyTorch profiler
import torch.profiler as prof
from torch.profiler import profile, record_function, ProfilerActivity

# Import your custom modules
from environment import ScoponeEnvMA
from state import initialize_game
from observation import encode_state_for_player
from actions import get_valid_actions, encode_action, decode_action
from rewards import compute_final_score_breakdown, compute_final_reward_from_breakdown

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo device: {device}")

# Configurazione GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

# Parametri di training
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
BATCH_SIZE = 128
REPLAY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
CHECKPOINT_PATH = "checkpoints/scopone_checkpoint"

# Crea directory per il profiling
PROFILE_DIR = "./profile_results"
Path(PROFILE_DIR).mkdir(parents=True, exist_ok=True)

# Regole di default per l'ambiente (modalità standard senza varianti)
DEFAULT_RULES = {
    'start_with_4_on_table': False,
    'asso_piglia_tutto': False,
    'scopa_on_asso_piglia_tutto': False,
    'asso_piglia_tutto_posabile': False,
    'asso_piglia_tutto_posabile_only_empty': False,
    'scopa_on_last_capture': False,
    're_bello': False,
    'napola': False,
    'napola_scoring': 'fixed3',
    'max_consecutive_scope': None,
    'last_cards_to_dealer': True,
}


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


class QNetwork(nn.Module):
    def __init__(self, obs_dim=10823, action_dim=80):
        super().__init__()
        # Feature extractor 
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
        
        # Sposta il modello su GPU
        self.to(device)
        
        # Opzioni CUDA per performance
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        # Assicura che l'input sia su GPU
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        elif x.device != device:
            x = x.to(device)
                
        import torch.nn.functional as F
        # Usa ReLU inplace per risparmiare memoria
        x1 = F.relu(self.backbone[0](x), inplace=True)
        x2 = F.relu(self.backbone[2](x1), inplace=True)
        x3 = F.relu(self.backbone[4](x2), inplace=True)
        backbone_features = F.relu(self.backbone[6](x3), inplace=True)
        
        # Dividi l'input in sezioni semantiche
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
        
        # Ottimizzazione GPU
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
            # Converti gli input in tensori GPU in un'unica operazione
            if hasattr(self, 'valid_actions_buffer') and len(valid_actions) <= self.valid_actions_buffer.size(0):
                valid_actions_t = self.valid_actions_buffer[:len(valid_actions)]
                for i, va in enumerate(valid_actions):
                    if isinstance(va, np.ndarray):
                        valid_actions_t[i].copy_(torch.tensor(va, device=device))
                    else:
                        valid_actions_t[i].copy_(va)
            else:
                if not hasattr(self, 'valid_actions_buffer') or len(valid_actions) > self.valid_actions_buffer.size(0):
                    self.valid_actions_buffer = torch.zeros((max(100, len(valid_actions)), 80), 
                                                        dtype=torch.float32, device=device)
                valid_actions_t = torch.tensor(np.stack(valid_actions), 
                                            dtype=torch.float32, device=device)
                
            with torch.no_grad():
                if hasattr(self, 'obs_buffer'):
                    obs_t = self.obs_buffer
                    if isinstance(obs, np.ndarray):
                        obs_t.copy_(torch.tensor(obs, device=device).unsqueeze(0))
                    else:
                        obs_t.copy_(obs.unsqueeze(0))
                else:
                    self.obs_buffer = torch.zeros((1, len(obs)), dtype=torch.float32, device=device)
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                # Usa mixed precision per accelerare l'inferenza
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
        
        # Usa buffer pre-allocati se possibile
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
        
        # Usa mixed precision in modo più efficiente con float16
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, idx)
                
                # Prendi slices dei tensori già sulla GPU (evita copie)
                batch_obs_t = all_obs_t[start_idx:end_idx]
                batch_actions_t = all_actions_t[start_idx:end_idx]
                batch_returns_t = all_returns_t[start_idx:end_idx]
                
                # Zero gradients - usa set_to_none=True per maggiore efficienza di memoria
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass con kernel fusion dove possibile
                q_values = self.online_qnet(batch_obs_t)
                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                
                # Loss con mixed precision
                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                
                # Traccia la loss per diagnostica
                total_loss += loss.item()
                batch_count += 1
                
                # Backward e optimizer step con gradient scaling per mixed precision
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # Clip gradient con una norma moderata per stabilità di training
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    
                    # Optimizer step con scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    self.optimizer.step()
                
                # Aggiorna epsilon dopo ogni batch per avanzare il training
                self.update_epsilon()
                
                # Sync target network periodicamente (non ad ogni batch)
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    self.sync_target()
                    sync_counter = 0
    
    def store_episode_transition(self, transition):
        # Aggiungi all'episodio corrente
        self.episodic_buffer.add_transition(transition)
    
    def end_episode(self):
        # Termina l'episodio corrente SENZA training
        # Il training deve essere chiamato esplicitamente dopo questo metodo
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


def find_latest_checkpoint(base_path, team_id):
    """Trova il checkpoint più recente per un team specifico"""
    dir_path = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    
    # Prima controlla se esiste il checkpoint standard
    standard_ckpt = f"{base_path}_team{team_id}.pth"
    if os.path.isfile(standard_ckpt):
        return standard_ckpt
        
    # Altrimenti cerca i checkpoint con numero episodio
    if os.path.exists(dir_path):
        pattern = f"{base_name}_team{team_id}_ep*.pth"
        matching_files = [f for f in os.listdir(dir_path) if fnmatch.fnmatch(f, pattern)]
        
        if matching_files:
            matching_files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]), reverse=True)
            return os.path.join(dir_path, matching_files[0])
    
    return None


def analyze_timing_results(profiler_data_file=f"{PROFILE_DIR}/raw_profiler_data.pt", name="timing_results"):
    """Analizza e salva i risultati del profiler focalizzandosi sui tempi"""
    try:
        # Carica i dati del profiler salvati
        profiler_data = torch.load(profiler_data_file)
        events = profiler_data.get('key_averages', [])
        
        if not events:
            print(f"Nessun dato di profiling valido trovato in {profiler_data_file}")
            return
            
        # Trova le operazioni di memcpy (trasferimenti CPU-GPU)
        memcpy_ops = [evt for evt in events if 'memcpy' in evt.key.lower()]
        
        # Calcola statistiche totali
        total_cpu_time = sum(evt.cpu_time_total for evt in events)
        total_cuda_time = sum(evt.cuda_time_total for evt in events)
        total_memcpy_time = sum(evt.cuda_time_total for evt in memcpy_ops)
        
        # Salva un report di testo
        with open(f"{PROFILE_DIR}/{name}.txt", 'w') as f:
            f.write("RIEPILOGO TEMPI DI ESECUZIONE\n")
            f.write("===========================\n\n")
            f.write(f"Tempo CPU totale: {total_cpu_time/1000:.2f} ms\n")
            f.write(f"Tempo CUDA totale: {total_cuda_time/1000:.2f} ms\n")
            f.write(f"Tempo trasferimenti CPU-GPU: {total_memcpy_time/1000:.2f} ms " +
                    f"({total_memcpy_time/total_cuda_time*100 if total_cuda_time else 0:.1f}% del tempo CUDA)\n\n")
            
            # Top operazioni CPU per tempo
            f.write("TOP 20 OPERAZIONI CPU (per tempo totale)\n")
            f.write("======================================\n")
            cpu_events = sorted(events, key=lambda x: x.cpu_time_total, reverse=True)[:20]
            for i, evt in enumerate(cpu_events):
                f.write(f"{i+1}. {evt.key.split('/')[-1]} - {evt.cpu_time_total/1000:.2f} ms " +
                        f"({evt.count} chiamate, {evt.cpu_time_total/evt.count/1000:.3f} ms/chiamata)\n")
            
            # Top operazioni CUDA per tempo
            f.write("\nTOP 20 OPERAZIONI CUDA (per tempo totale)\n")
            f.write("=======================================\n")
            cuda_events = sorted(events, key=lambda x: x.cuda_time_total, reverse=True)[:20]
            for i, evt in enumerate(cuda_events):
                f.write(f"{i+1}. {evt.key.split('/')[-1]} - {evt.cuda_time_total/1000:.2f} ms " +
                        f"({evt.count} chiamate, {evt.cuda_time_total/evt.count/1000:.3f} ms/chiamata)\n")
            
            # Operazioni di trasferimento dati
            f.write("\nOPERAZIONI DI TRASFERIMENTO CPU-GPU\n")
            f.write("==================================\n")
            if memcpy_ops:
                for i, evt in enumerate(memcpy_ops):
                    f.write(f"{i+1}. {evt.key.split('/')[-1]} - {evt.cuda_time_total/1000:.2f} ms " +
                            f"({evt.count} chiamate, {evt.cuda_time_total/evt.count/1000:.3f} ms/chiamata)\n")
            else:
                f.write("Nessuna operazione di trasferimento esplicita rilevata.\n")
        
        print(f"Analisi dei tempi salvata in {PROFILE_DIR}/{name}.txt")
        
        # Crea anche una versione tabellare dei dati completi
        import pandas as pd
        data = []
        for evt in events:
            # Salta operazioni con tempi minimi che inquinano i dati
            if evt.cpu_time_total < 10 and evt.cuda_time_total < 10:
                continue
                
            data.append({
                'Name': evt.key,
                'CPU Time (ms)': evt.cpu_time_total/1000,
                'CUDA Time (ms)': evt.cuda_time_total/1000,
                'Self CPU Time (ms)': evt.self_cpu_time_total/1000,
                'Self CUDA Time (ms)': evt.self_cuda_time_total/1000,
                'Calls': evt.count,
                'Avg CPU Time (ms)': evt.cpu_time_total/evt.count/1000 if evt.count else 0,
                'Avg CUDA Time (ms)': evt.cuda_time_total/evt.count/1000 if evt.count else 0,
                'Is Transfer': 'memcpy' in evt.key.lower()
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f"{PROFILE_DIR}/{name}.csv", index=False)
        
        print(f"Dati completi salvati in {PROFILE_DIR}/{name}.csv")
        return df
        
    except Exception as e:
        print(f"Errore durante l'analisi dei risultati: {e}")
        traceback.print_exc()
        return None


def profiled_train_agents(num_episodes=200):
    """
    Versione profilata della funzione di training, limitata al numero di episodi specificato
    """
    # Crea la directory dei checkpoint se non esiste
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_dir):
        print(f"Creazione directory per checkpoint: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    
    # Crea gli agenti
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Cerca i checkpoint esistenti
    print(f"Ricerca dei checkpoint più recenti...")
    
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
    
    # Crea profiler
    profiler_activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(ProfilerActivity.CUDA)
    
    # Configura il profiler con schedule più gestibile
    # Usa "active" più breve per evitare il problema dello stack overflow
    profiler = profile(
        activities=profiler_activities,
        # MODIFICATO: Divide il profiling in blocchi più piccoli per evitare lo stack overflow
        schedule=prof.schedule(wait=1, warmup=1, active=40, repeat=4),
        record_shapes=True,
        with_stack=True
    )
    
    # Record di dati per l'analisi finale
    all_profiling_data = []
    current_profiler = None
    
    # Barra di progresso
    pbar = tqdm(total=num_episodes, desc="Profilazione episodi")
    
    # Loop principale per episodi
    for ep in range(num_episodes):
        episode_start_time = time.time()
        
        # Gestisci avvio/terminazione del profiler
        if ep % 40 == 0:
            # Chiudi profiler precedente se esiste
            if current_profiler is not None:
                try:
                    current_profiler.__exit__(None, None, None)
                    all_profiling_data.append(current_profiler)
                except Exception as e:
                    print(f"Errore nel chiudere il profiler all'episodio {ep}: {e}")
            
            # Avvia un nuovo profiler
            current_profiler = profiler.__enter__()
        
        # Aggiorna progresso
        pbar.set_description(f"Episodio {ep+1}/{num_episodes} (Giocatore {first_player})")
        pbar.update(1)
        
        # Crea environment e inizializza
        with record_function("Environment_Setup"):
            env = ScoponeEnvMA(rules=DEFAULT_RULES)
            env.current_player = first_player

        # Inizializza i buffer degli episodi
        with record_function("Initialize_Episode"):
            agent_team0.start_episode()
            agent_team1.start_episode()

        # Stato iniziale
        with record_function("Initial_Observation"):
            obs_current = env._get_observation(env.current_player)
            done = False
        
        # Conteggio transizioni per team
        team0_transitions = 0
        team1_transitions = 0

        # Game loop
        with record_function("Game_Loop"):
            inference_start = time.time()
            step_counter = 0
            
            while not done:
                step_counter += 1
                with record_function(f"Game_Step_{step_counter}"):
                    cp = env.current_player
                    team_id = 0 if cp in [0,2] else 1
                    agent = agent_team0 if team_id==0 else agent_team1
                    
                    # Ottieni azioni valide
                    with record_function("Get_Valid_Actions"):
                        valid_acts = env.get_valid_actions()
                    
                    if not valid_acts:
                        break
                    
                    # Scegli azione
                    with record_function("Pick_Action"):
                        action = agent.pick_action(obs_current, valid_acts, env)
                    
                    # Esegui azione
                    with record_function("Environment_Step"):
                        next_obs, reward, done, info = env.step(action)
                    
                    # Memorizza transizione
                    with record_function("Store_Transition"):
                        next_valid = env.get_valid_actions() if not done else []
                        transition = (obs_current, action, reward, next_obs, done, next_valid)
                        
                        if team_id == 0:
                            agent_team0.store_episode_transition(transition)
                            team0_transitions += 1
                        else:
                            agent_team1.store_episode_transition(transition)
                            team1_transitions += 1
                    
                    global_step += 1
                    
                    # Prepara per la prossima iterazione
                    obs_current = next_obs
        
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        # Termina episodi
        with record_function("End_Episodes"):
            agent_team0.end_episode()
            agent_team1.end_episode()
        
        # Ottieni reward finali
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
        
        # Training
        with record_function("Training"):
            train_start_time = time.time()
            
            # Team 0 training
            with record_function("Team0_Training"):
                if agent_team0.episodic_buffer.episodes:
                    last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
                    if last_episode_team0:
                        agent_team0.train_episodic_monte_carlo()
            
            # Team 1 training
            with record_function("Team1_Training"):
                if agent_team1.episodic_buffer.episodes:
                    last_episode_team1 = agent_team1.episodic_buffer.episodes[-1]
                    if last_episode_team1:
                        agent_team1.train_episodic_monte_carlo()
            
            # Sincronizzazione network target
            if global_step % TARGET_UPDATE_FREQ == 0:
                with record_function("Sync_Target_Networks"):
                    agent_team0.sync_target()
                    agent_team1.sync_target()
            
            train_time = time.time() - train_start_time
            train_times.append(train_time)
        
        # Pulizia memoria
        with record_function("Memory_Cleanup"):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Tempo episodio
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        
        # Salva checkpoint periodicamente
        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            with record_function("Save_Checkpoints"):
                agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_profile_team0_ep{ep+1}.pth")
                agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_profile_team1_ep{ep+1}.pth")
        
        # Step profiler
        if current_profiler is not None:
            try:
                # Fai lo step del profiler
                profiler.step()
            except Exception as e:
                print(f"Errore durante lo step del profiler all'episodio {ep+1}: {e}")
                # Tenta di chiudere e ricrearlo
                try:
                    current_profiler.__exit__(None, None, None)
                except:
                    pass
                current_profiler = profiler.__enter__()
        
        # Cambia primo giocatore
        first_player = (first_player + 1) % 4
    
    # Chiudi l'ultimo profiler se ancora aperto
    if current_profiler is not None:
        try:
            current_profiler.__exit__(None, None, None)
            all_profiling_data.append(current_profiler)
        except Exception as e:
            print(f"Errore nel chiudere l'ultimo profiler: {e}")
    
    # Chiudi barra di progresso
    pbar.close()
    
    # Esporta tutti i dati di profiling in un file
    try:
        # Prendi l'ultimo profiler (che dovrebbe avere i dati più freschi)
        if all_profiling_data:
            last_profiler = all_profiling_data[-1]
            # Salva i dati grezzi del profiler
            torch.save({
                'key_averages': last_profiler.key_averages(),
                'events': last_profiler._events
            }, f"{PROFILE_DIR}/raw_profiler_data.pt")
            
            # Prova a esportare anche la traccia Chrome
            try:
                last_profiler.export_chrome_trace(f"{PROFILE_DIR}/trace.json")
                print(f"Esportato Chrome trace in {PROFILE_DIR}/trace.json")
                print("Puoi visualizzare questo trace in Chrome navigando a chrome://tracing")
            except Exception as e:
                print(f"Errore nell'esportazione del trace: {e}")
    except Exception as e:
        print(f"Errore nel salvare i dati del profiler: {e}")
    
    # Analizza e salva i risultati del timing
    analyze_timing_results()
    
    # Genera report prestazioni
    avg_episode_time = sum(episode_times) / len(episode_times)
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_train_time = sum(train_times) / len(train_times)
    
    print("\n=== Report Prestazioni ===")
    print(f"Tempo medio per episodio: {avg_episode_time:.3f}s")
    print(f"Tempo medio per inferenza: {avg_inference_time:.3f}s ({avg_inference_time/avg_episode_time*100:.1f}% dell'episodio)")
    print(f"Tempo medio per training: {avg_train_time:.3f}s ({avg_train_time/avg_episode_time*100:.1f}% dell'episodio)")
    
    # Salva metriche di performance
    with open(f"{PROFILE_DIR}/performance_metrics.txt", "w") as f:
        f.write("METRICHE DI PERFORMANCE\n")
        f.write("=====================\n\n")
        f.write(f"Episodi: {num_episodes}\n")
        f.write(f"Step totali: {global_step}\n\n")
        
        f.write(f"Tempo medio per episodio: {avg_episode_time:.3f}s\n")
        f.write(f"Tempo medio per inferenza: {avg_inference_time:.3f}s ({avg_inference_time/avg_episode_time*100:.1f}% dell'episodio)\n")
        f.write(f"Tempo medio per training: {avg_train_time:.3f}s ({avg_train_time/avg_episode_time*100:.1f}% dell'episodio)\n")
    
    print(f"Metriche di performance salvate in {PROFILE_DIR}/performance_metrics.txt")
    
    return all_profiling_data


if __name__ == "__main__":
    # Numero di episodi da profilare
    NUM_EPISODES = 10
    
    print(f"Avvio profilazione per {NUM_EPISODES} episodi...")
    
    # Esegui training profilato
    start_time = time.time()
    try:
        profile_result = profiled_train_agents(NUM_EPISODES)
    except Exception as e:
        print(f"Errore durante la profilazione: {e}")
        traceback.print_exc()
        
        # Prova comunque ad analizzare i dati salvati (se esistono)
        print("Tentativo di analisi dei dati di profiling disponibili...")
        if os.path.exists(f"{PROFILE_DIR}/raw_profiler_data.pt"):
            analyze_timing_results()
    
    total_time = time.time() - start_time
    
    print(f"Profilazione completata in {total_time:.2f} secondi")
    print(f"Risultati salvati in {PROFILE_DIR}")
    print("\nAnalisi completata!")