#!/usr/bin/env python
# simple_profiler.py - Profiler manuale per tempi di calcolo e trasferimenti

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
import pandas as pd
import json

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

# Classe ProfilerTimer per timing accurati e annidati
class ProfilerTimer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.timings = collections.defaultdict(float)
        self.counts = collections.defaultdict(int)
        self.current_section = None
        self.cuda_timings = collections.defaultdict(float)
        self.transfer_timings = collections.defaultdict(float)
        self.transfer_counts = collections.defaultdict(int)
        self.transfer_sizes = collections.defaultdict(list)
        self.active = False
        
    def start(self, section_name):
        # Prima terminare qualsiasi sezione attiva
        if self.current_section is not None:
            self.stop()
            
        # Avvia una nuova sezione
        self.current_section = section_name
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.time()
        self.active = True
        self.counts[section_name] += 1
        
    def stop(self):
        if not self.active or self.current_section is None:
            return
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - self.start_time
        self.timings[self.current_section] += elapsed
        
        # Resetta lo stato
        self.current_section = None
        self.active = False
        
    def record_cuda_time(self, section_name, duration):
        self.cuda_timings[section_name] += duration
        
    def record_transfer(self, direction, size_bytes, duration):
        """Registra un trasferimento CPU-GPU o GPU-CPU"""
        self.transfer_timings[direction] += duration
        self.transfer_counts[direction] += 1
        self.transfer_sizes[direction].append(size_bytes)
        
    def get_summary(self):
        """Restituisce un riepilogo delle misurazioni"""
        summary = {
            "cpu_timings": {k: v for k, v in self.timings.items()},
            "counts": {k: v for k, v in self.counts.items()},
            "cuda_timings": {k: v for k, v in self.cuda_timings.items()},
            "transfer_timings": {k: v for k, v in self.transfer_timings.items()},
            "transfer_counts": {k: v for k, v in self.transfer_counts.items()},
            "transfer_sizes": {k: sum(v) for k, v in self.transfer_sizes.items()}
        }
        return summary
    
    def save_summary(self, output_file):
        """Salva il riepilogo in un file JSON"""
        summary = self.get_summary()
        
        # Converti in formato più leggibile
        readable_summary = {
            "cpu_timings_ms": {k: v*1000 for k, v in summary["cpu_timings"].items()},
            "counts": summary["counts"],
            "cuda_timings_ms": {k: v*1000 for k, v in summary["cuda_timings"].items()},
            "transfer_timings_ms": {k: v*1000 for k, v in summary["transfer_timings"].items()},
            "transfer_counts": summary["transfer_counts"],
            "transfer_sizes_mb": {k: v/(1024*1024) for k, v in summary["transfer_sizes"].items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(readable_summary, f, indent=2)
        
        print(f"Riepilogo salvato in {output_file}")
        
    def print_summary(self):
        """Stampa un riepilogo delle misurazioni"""
        summary = self.get_summary()
        
        print("\n=== RIEPILOGO TIMING ===")
        print("\nTempi CPU:")
        for section, timing in sorted(summary["cpu_timings"].items(), key=lambda x: x[1], reverse=True):
            count = summary["counts"][section]
            avg = timing / count if count > 0 else 0
            print(f"  {section}: {timing*1000:.2f} ms totali, {count} chiamate, {avg*1000:.2f} ms/chiamata")
            
        print("\nTempi CUDA:")
        for section, timing in sorted(summary["cuda_timings"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {section}: {timing*1000:.2f} ms")
            
        print("\nTrasferimenti CPU-GPU:")
        for direction, timing in sorted(summary["transfer_timings"].items(), key=lambda x: x[1], reverse=True):
            count = summary["transfer_counts"][direction]
            size = summary["transfer_sizes"][direction] / (1024*1024)  # MB
            avg_time = timing / count if count > 0 else 0
            avg_size = size / count if count > 0 else 0
            print(f"  {direction}: {timing*1000:.2f} ms totali, {count} trasferimenti, {size:.2f} MB totali")
            print(f"    Media: {avg_time*1000:.2f} ms/trasferimento, {avg_size:.2f} MB/trasferimento")


# Versione modificata di torch.Tensor.to per monitorare i trasferimenti
original_to = torch.Tensor.to
profiler_timer = ProfilerTimer("global")

def patched_to(self, *args, **kwargs):
    """Versione modificata di tensor.to() che monitora i trasferimenti CPU-GPU"""
    # Determina se è un trasferimento CPU-GPU o viceversa
    source_device = self.device
    if len(args) > 0 and isinstance(args[0], (torch.device, str)) and args[0] != source_device:
        target_device = args[0]
    elif 'device' in kwargs and kwargs['device'] != source_device:
        target_device = kwargs['device']
    else:
        # Non è un cambio di dispositivo, usa la versione originale
        return original_to(self, *args, **kwargs)
    
    # Determina la direzione del trasferimento
    if source_device.type == 'cpu' and (str(target_device) == 'cuda' or str(target_device).startswith('cuda:')):
        direction = "CPU->GPU"
    elif (source_device.type == 'cuda' or str(source_device).startswith('cuda:')) and str(target_device) == 'cpu':
        direction = "GPU->CPU"
    else:
        # Altro tipo di trasferimento
        return original_to(self, *args, **kwargs)
    
    # Calcola la dimensione del tensore in bytes
    size_bytes = self.element_size() * self.nelement()
    
    # Sincronizza CUDA e registra il tempo di trasferimento
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    # Esegui il trasferimento
    result = original_to(self, *args, **kwargs)
    
    # Sincronizza e calcola il tempo trascorso
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time
    
    # Registra il trasferimento
    profiler_timer.record_transfer(direction, size_bytes, elapsed)
    
    return result

# Sostituisci il metodo originale con quello modificato per il profiling
torch.Tensor.to = patched_to

# Classi Model e Training Identiche a prima
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
        profiler_timer.start("QNetwork_forward")
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
        
        profiler_timer.stop()  # QNetwork_forward
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
        profiler_timer.start("pick_action")
        
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
            result = random.choice(valid_actions)
            profiler_timer.stop()  # pick_action
            return result
            
        # Converti gli input in tensori GPU in un'unica operazione
        profiler_timer.start("prepare_action_tensors")
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
        
        if hasattr(self, 'obs_buffer'):
            obs_t = self.obs_buffer
            if isinstance(obs, np.ndarray):
                obs_t.copy_(torch.tensor(obs, device=device).unsqueeze(0))
            else:
                obs_t.copy_(obs.unsqueeze(0))
        else:
            self.obs_buffer = torch.zeros((1, len(obs)), dtype=torch.float32, device=device)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        profiler_timer.stop()  # prepare_action_tensors
            
        with torch.no_grad():
            # Usa mixed precision per accelerare l'inferenza
            profiler_timer.start("inference")
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                action_values = self.online_qnet(obs_t)
                q_values = torch.sum(action_values.view(1, 1, 80) * valid_actions_t.view(-1, 1, 80), dim=2).squeeze()
            profiler_timer.stop()  # inference
            
            best_action_idx = torch.argmax(q_values).item()
        
        profiler_timer.stop()  # pick_action
        return valid_actions[best_action_idx]
    
    def train_episodic_monte_carlo(self, specific_episode=None):
        profiler_timer.start("train_episodic_monte_carlo")
        
        # Determina quali episodi processare
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            profiler_timer.stop()  # train_episodic_monte_carlo
            return  # Nessun episodio disponibile
        
        # Usa buffer pre-allocati se possibile
        max_transitions = sum(len(episode) for episode in episodes_to_process)
        
        # Crea o ridimensiona i buffer se necessario
        profiler_timer.start("prepare_training_buffers")
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
            profiler_timer.stop()  # prepare_training_buffers
            profiler_timer.stop()  # train_episodic_monte_carlo
            return  # Nessuna transizione da processare
        
        # Usa slices dei buffer per il training
        all_obs_t = self.train_obs_buffer[:idx]
        all_actions_t = self.train_actions_buffer[:idx]
        all_returns_t = self.train_returns_buffer[:idx]
        profiler_timer.stop()  # prepare_training_buffers
        
        # Aumenta batch_size per sfruttare meglio la GPU
        batch_size = min(512, idx)
        num_batches = (idx + batch_size - 1) // batch_size
        
        # Riduci la frequenza di sync_target
        sync_counter = 0
        sync_frequency = 10
        
        # Traccia le metriche per diagnostica
        total_loss = 0.0
        batch_count = 0
        
        # Training loop
        profiler_timer.start("training_loop")
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
                profiler_timer.start("optimizer_zero_grad")
                self.optimizer.zero_grad(set_to_none=True)
                profiler_timer.stop()  # optimizer_zero_grad
                
                # Forward pass con kernel fusion dove possibile
                profiler_timer.start("forward_pass")
                q_values = self.online_qnet(batch_obs_t)
                q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
                profiler_timer.stop()  # forward_pass
                
                # Loss con mixed precision
                profiler_timer.start("loss_calculation")
                loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
                profiler_timer.stop()  # loss_calculation
                
                # Traccia la loss per diagnostica
                total_loss += loss.item()
                batch_count += 1
                
                # Backward e optimizer step con gradient scaling per mixed precision
                if self.scaler:
                    profiler_timer.start("backward_pass")
                    self.scaler.scale(loss).backward()
                    profiler_timer.stop()  # backward_pass
                    
                    # Clip gradient con una norma moderata per stabilità di training
                    profiler_timer.start("gradient_clipping")
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    profiler_timer.stop()  # gradient_clipping
                    
                    # Optimizer step con scaling
                    profiler_timer.start("optimizer_step")
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    profiler_timer.stop()  # optimizer_step
                else:
                    profiler_timer.start("backward_pass")
                    loss.backward()
                    profiler_timer.stop()  # backward_pass
                    
                    profiler_timer.start("gradient_clipping")
                    torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                    profiler_timer.stop()  # gradient_clipping
                    
                    profiler_timer.start("optimizer_step")
                    self.optimizer.step()
                    profiler_timer.stop()  # optimizer_step
                
                # Aggiorna epsilon dopo ogni batch per avanzare il training
                self.update_epsilon()
                
                # Sync target network periodicamente (non ad ogni batch)
                sync_counter += 1
                if sync_counter >= sync_frequency:
                    profiler_timer.start("sync_target")
                    self.sync_target()
                    profiler_timer.stop()  # sync_target
                    sync_counter = 0
        profiler_timer.stop()  # training_loop
        
        profiler_timer.stop()  # train_episodic_monte_carlo
        return total_loss / batch_count if batch_count > 0 else 0.0
    
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
        profiler_timer.start("sync_target")
        self.target_qnet.load_state_dict(self.online_qnet.state_dict())
        profiler_timer.stop()  # sync_target

    def maybe_sync_target(self):
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.sync_target()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, EPSILON_START - (self.train_steps / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
        self.train_steps += 1

    def save_checkpoint(self, filename):
        profiler_timer.start("save_checkpoint")
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
        profiler_timer.stop()  # save_checkpoint

    def load_checkpoint(self, filename):
        profiler_timer.start("load_checkpoint")
        ckpt = torch.load(filename, map_location=device)
        self.online_qnet.load_state_dict(ckpt["online_state_dict"])
        self.target_qnet.load_state_dict(ckpt["target_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"[DQNAgent] Checkpoint caricato da {filename}")
        profiler_timer.stop()  # load_checkpoint


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


def profiled_train_agents(num_episodes=20):
    """
    Versione profilata della funzione di training, limitata al numero di episodi specificato
    """
    profiler_timer.start("profiled_train_agents")
    
    # Crea la directory dei checkpoint se non esiste
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if not os.path.exists(checkpoint_dir):
        print(f"Creazione directory per checkpoint: {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    
    # Crea gli agenti
    profiler_timer.start("create_agents")
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)
    profiler_timer.stop()  # create_agents

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
    
    # Barra di progresso
    pbar = tqdm(total=num_episodes, desc="Profilazione episodi")
    
    # Loop principale per episodi
    for ep in range(num_episodes):
        profiler_timer.start(f"episode_{ep}")
        episode_start_time = time.time()
        
        # Aggiorna progresso
        pbar.set_description(f"Episodio {ep+1}/{num_episodes} (Giocatore {first_player})")
        pbar.update(1)
        
        # Crea environment e inizializza
        profiler_timer.start("environment_setup")
        env = ScoponeEnvMA()
        env.current_player = first_player
        profiler_timer.stop()  # environment_setup

        # Inizializza i buffer degli episodi
        profiler_timer.start("initialize_episode")
        agent_team0.start_episode()
        agent_team1.start_episode()
        profiler_timer.stop()  # initialize_episode

        # Stato iniziale
        profiler_timer.start("initial_observation")
        obs_current = env._get_observation(env.current_player)
        done = False
        profiler_timer.stop()  # initial_observation
        
        # Conteggio transizioni per team
        team0_transitions = 0
        team1_transitions = 0

        # Game loop
        profiler_timer.start("game_loop")
        inference_start = time.time()
        step_counter = 0
        
        while not done:
            step_counter += 1
            profiler_timer.start(f"game_step_{step_counter}")
            
            cp = env.current_player
            team_id = 0 if cp in [0,2] else 1
            agent = agent_team0 if team_id==0 else agent_team1
            
            # Ottieni azioni valide
            profiler_timer.start("get_valid_actions")
            valid_acts = env.get_valid_actions()
            profiler_timer.stop()  # get_valid_actions
            
            if not valid_acts:
                profiler_timer.stop()  # game_step_{step_counter}
                break
            
            # Scegli azione
            profiler_timer.start("pick_action")
            action = agent.pick_action(obs_current, valid_acts, env)
            profiler_timer.stop()  # pick_action
            
            # Esegui azione
            profiler_timer.start("environment_step")
            next_obs, reward, done, info = env.step(action)
            profiler_timer.stop()  # environment_step
            
            # Memorizza transizione
            profiler_timer.start("store_transition")
            next_valid = env.get_valid_actions() if not done else []
            transition = (obs_current, action, reward, next_obs, done, next_valid)
            
            if team_id == 0:
                agent_team0.store_episode_transition(transition)
                team0_transitions += 1
            else:
                agent_team1.store_episode_transition(transition)
                team1_transitions += 1
            profiler_timer.stop()  # store_transition
            
            global_step += 1
            
            # Prepara per la prossima iterazione
            obs_current = next_obs
            
            profiler_timer.stop()  # game_step_{step_counter}
        
        profiler_timer.stop()  # game_loop
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        # Termina episodi
        profiler_timer.start("end_episodes")
        agent_team0.end_episode()
        agent_team1.end_episode()
        profiler_timer.stop()  # end_episodes
        
        # Ottieni reward finali
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
        
        # Training
        profiler_timer.start("training")
        train_start_time = time.time()
        
        # Team 0 training
        profiler_timer.start("team0_training")
        if agent_team0.episodic_buffer.episodes:
            last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
            if last_episode_team0:
                agent_team0.train_episodic_monte_carlo()
        profiler_timer.stop()  # team0_training
        
        # Team 1 training
        profiler_timer.start("team1_training")
        if agent_team1.episodic_buffer.episodes:
            last_episode_team1 = agent_team1.episodic_buffer.episodes[-1]
            if last_episode_team1:
                agent_team1.train_episodic_monte_carlo()
        profiler_timer.stop()  # team1_training
        
        # Sincronizzazione network target
        if global_step % TARGET_UPDATE_FREQ == 0:
            profiler_timer.start("sync_target_networks")
            agent_team0.sync_target()
            agent_team1.sync_target()
            profiler_timer.stop()  # sync_target_networks
        
        profiler_timer.stop()  # training
        train_time = time.time() - train_start_time
        train_times.append(train_time)
        
        # Pulizia memoria
        profiler_timer.start("memory_cleanup")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        profiler_timer.stop()  # memory_cleanup
        
        # Tempo episodio
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        
        # Salva checkpoint periodicamente
        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            profiler_timer.start("save_checkpoints")
            agent_team0.save_checkpoint(f"{CHECKPOINT_PATH}_profile_team0_ep{ep+1}.pth")
            agent_team1.save_checkpoint(f"{CHECKPOINT_PATH}_profile_team1_ep{ep+1}.pth")
            profiler_timer.stop()  # save_checkpoints
            
        profiler_timer.stop()  # episode_{ep}
        
        # Cambia primo giocatore
        first_player = (first_player + 1) % 4
    
    # Chiudi barra di progresso
    pbar.close()
    
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
    
    profiler_timer.stop()  # profiled_train_agents
    return profiler_timer


if __name__ == "__main__":
    # Numero di episodi da profilare
    NUM_EPISODES = 20
    
    print(f"Avvio profilazione per {NUM_EPISODES} episodi...")
    
    # Esegui training profilato
    start_time = time.time()
    profile_result = profiled_train_agents(NUM_EPISODES)
    total_time = time.time() - start_time
    
    # Stampa e salva i risultati
    profile_result.print_summary()
    profile_result.save_summary(f"{PROFILE_DIR}/timing_summary.json")
    
    # Crea report in formato tabellare
    summary = profile_result.get_summary()
    
    # CPU timing
    cpu_data = []
    for section, timing in sorted(summary["cpu_timings"].items(), key=lambda x: x[1], reverse=True):
        count = summary["counts"].get(section, 0)
        avg = timing / count if count > 0 else 0
        cpu_data.append({
            'Section': section,
            'Total Time (ms)': timing * 1000,
            'Calls': count,
            'Avg Time/Call (ms)': avg * 1000
        })
    
    cpu_df = pd.DataFrame(cpu_data)
    cpu_df.to_csv(f"{PROFILE_DIR}/cpu_timing.csv", index=False)
    
    # Trasferimenti
    transfer_data = []
    for direction, timing in sorted(summary["transfer_timings"].items(), key=lambda x: x[1], reverse=True):
        count = summary["transfer_counts"].get(direction, 0)
        size = summary["transfer_sizes"].get(direction, 0) / (1024*1024)  # MB
        avg_time = timing / count if count > 0 else 0
        avg_size = size / count if count > 0 else 0
        transfer_data.append({
            'Direction': direction,
            'Total Time (ms)': timing * 1000,
            'Transfers': count,
            'Total Size (MB)': size,
            'Avg Time/Transfer (ms)': avg_time * 1000,
            'Avg Size/Transfer (MB)': avg_size
        })
    
    transfer_df = pd.DataFrame(transfer_data)
    transfer_df.to_csv(f"{PROFILE_DIR}/transfers.csv", index=False)
    
    print(f"Profilazione completata in {total_time:.2f} secondi")
    print(f"Risultati salvati in {PROFILE_DIR}")
    
    # Ripristina il metodo originale torch.Tensor.to
    torch.Tensor.to = original_to
    print("\nAnalisi completata!")