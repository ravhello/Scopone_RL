#!/usr/bin/env python
# fixed_chrome_profiler.py - Versione corretta che genera un trace JSON valido

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
import webbrowser
import subprocess
import platform
import json
import re

# Import PyTorch profiler con funzioni minime
from torch.profiler import profile, record_function, ProfilerActivity

# Configurazione GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurazione GPU avanzata
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

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

# Directory per i logs e trace file
TRACE_DIR = "./traces"
os.makedirs(TRACE_DIR, exist_ok=True)
TRACE_FILE = os.path.join(TRACE_DIR, "scopone_trace.json")
TEMP_TRACE_FILE = os.path.join(TRACE_DIR, "temp_trace.json")

# Classi di base (EpisodicReplayBuffer, QNetwork, DQNAgent) - identiche all'originale
# [Omesse per brevità - sono le stesse del codice originale]

############################################################
# 1) EpisodicReplayBuffer
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
# 2) QNetwork
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
# 3) DQNAgent
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
        # Versione semplificata per minimizzare la dimensione del profiling
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
        
        # Training loop semplificato per ridurre dimensioni trace
        for batch_idx in range(1):  # Esegui solo 1 batch per il profiling
            start_idx = 0
            end_idx = min(batch_size, idx)
            
            batch_obs_t = all_obs_t[start_idx:end_idx]
            batch_actions_t = all_actions_t[start_idx:end_idx]
            batch_returns_t = all_returns_t[start_idx:end_idx]
            
            self.optimizer.zero_grad(set_to_none=True)
            
            q_values = self.online_qnet(batch_obs_t)
            q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
            
            loss = nn.MSELoss()(q_values_for_actions, batch_returns_t)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            self.update_epsilon()
    
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
# 4) Funzioni ausiliarie per salvare un trace JSON valido
############################################################
def sanitize_json_string(s):
    """
    Rimuove caratteri di controllo non validi da una stringa JSON
    """
    # Rimuovi caratteri di controllo tranne tab, newline e carriage return
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)

def sanitize_trace_file(input_file, output_file):
    """
    Sanitizza il file trace per assicurarsi che sia un JSON valido.
    Può gestire file di grandi dimensioni processandoli in blocchi.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                # Leggi l'inizio del file per determinare se è un JSON array
                start = f_in.read(2)
                f_in.seek(0)  # Torna all'inizio
                
                if start.startswith('['):
                    # È un array JSON, scrive l'apertura
                    f_out.write('[')
                    
                    # Leggi il file riga per riga, eccetto la prima e ultima parentesi quadra
                    content = f_in.read()[1:-1]  # Rimuovi [ all'inizio e ] alla fine
                    
                    # Sanitizza il contenuto
                    sanitized_content = sanitize_json_string(content)
                    
                    # Scrivi il contenuto sanitizzato
                    f_out.write(sanitized_content)
                    
                    # Chiudi l'array
                    f_out.write(']')
                else:
                    # È un oggetto JSON normale
                    content = f_in.read()
                    sanitized_content = sanitize_json_string(content)
                    f_out.write(sanitized_content)
                
        print(f"File trace sanitizzato salvato: {output_file}")
        return True
    except Exception as e:
        print(f"Errore durante la sanitizzazione del trace: {e}")
        return False

def open_chrome_with_trace(trace_file):
    """
    Apre Chrome con il chrome://tracing che carica automaticamente il trace file
    """
    trace_path = os.path.abspath(trace_file)
    
    # Metodo che funziona su Windows, macOS e Linux
    system = platform.system()
    
    if system == 'Windows':
        try:
            # Prima prova: avvia Chrome direttamente con l'URL chrome://tracing
            chrome_paths = [
                'C:/Program Files/Google/Chrome/Application/chrome.exe',
                'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
                os.path.expanduser("~") + '/AppData/Local/Google/Chrome/Application/chrome.exe',
            ]
            
            chrome_path = None
            for path in chrome_paths:
                if os.path.exists(path):
                    chrome_path = path
                    break
            
            if chrome_path:
                subprocess.run([chrome_path, "chrome://tracing"], shell=True)
            else:
                # Fallback al browser predefinito
                webbrowser.open("chrome://tracing")
        except Exception as e:
            print(f"Errore nell'apertura di Chrome: {e}")
            # Fallback: apri il browser predefinito
            webbrowser.open("chrome://tracing")
    
    elif system == 'Darwin':  # macOS
        # Su macOS, usa il comando open con l'app Chrome
        try:
            subprocess.run(["open", "-a", "Google Chrome", "chrome://tracing"])
        except Exception as e:
            print(f"Errore nell'apertura di Chrome: {e}")
            webbrowser.open("chrome://tracing")
    
    else:  # Linux
        # Su Linux, prova a usare il comando google-chrome
        try:
            subprocess.run(["google-chrome", "chrome://tracing"])
        except:
            # Fallback a chromium o al browser predefinito
            try:
                subprocess.run(["chromium-browser", "chrome://tracing"])
            except:
                webbrowser.open("chrome://tracing")
    
    print(f"\nTrace file generato: {trace_path}")
    print("1. Chrome dovrebbe aprirsi con chrome://tracing")
    print("2. Clicca sul pulsante 'Load' (in alto a sinistra)")
    print("3. Naviga e seleziona il file trace generato:")
    print(f"   {trace_path}")
    print("\nNota: Se Chrome non si apre automaticamente, copia l'URL chrome://tracing in Chrome e carica manualmente il file.")

############################################################
# 5) Funzione di training con profiler
############################################################
def train_with_chrome_trace(num_episodes=5):
    """
    Versione ridotta di train_agents che genera un trace file valido per Chrome
    """
    print(f"Profiling di {num_episodes} episodi, generando trace file per Chrome...")
    
    # Configurazione GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    # Crea agenti
    agent_team0 = DQNAgent(team_id=0)
    agent_team1 = DQNAgent(team_id=1)

    # Variabili di controllo
    first_player = 0
    
    # Imposta profiler con focus solo sulle attività rilevanti
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    # Usa un profiler semplice con meno opzioni per ridurre la dimensione del trace
    with profile(
        activities=activities,
        record_shapes=False,  # Disattivato per ridurre dimensione trace
        with_stack=False      # Disattivato per ridurre dimensione trace
    ) as prof:
        
        # Progress bar per gli episodi
        pbar = tqdm(total=num_episodes, desc="Episodi profilati")
        
        for ep in range(num_episodes):
            # Aggiorna progress bar
            pbar.set_description(f"Episodio {ep+1}/{num_episodes} (Giocatore {first_player})")
            pbar.update(1)
            
            # Crea environment
            with record_function("Environment_Setup"):
                env = ScoponeEnvMA()
                env.current_player = first_player

            # Inizializza episodi
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
            
            # Game loop
            with record_function("Game_Loop"):
                step_count = 0
                while not done and step_count < 10:  # Limita a 10 step per episodio
                    step_count += 1
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
                        action = agent.pick_action(obs_current, valid_acts, env)
                    
                    # Step ambiente
                    with record_function("Environment_Step"):
                        next_obs, reward, done, info = env.step(action)
                        
                        # Assicura che next_obs sia numpy array
                        if torch.is_tensor(next_obs):
                            next_obs = next_obs.cpu().numpy()
                    
                    # Memorizza transition
                    with record_function("Store_Transition"):
                        next_valid = env.get_valid_actions() if not done else []
                        transition = (obs_current, action, reward, next_obs, done, next_valid)
                        
                        if team_id == 0:
                            agent_team0.store_episode_transition(transition)
                        else:
                            agent_team1.store_episode_transition(transition)
                    
                    # Prepara per la prossima iterazione
                    obs_current = next_obs

            # Termina episodi
            with record_function("End_Episodes"):
                agent_team0.end_episode()
                agent_team1.end_episode()
            
            # Training semplificato (solo per test)
            with record_function("Training"):
                if ep > 0:  # Salta il training sul primo episodio
                    # Team 0 training
                    with record_function("Team0_Training"):
                        if agent_team0.episodic_buffer.episodes:
                            agent_team0.train_episodic_monte_carlo()
                    
                    # Team 1 training
                    with record_function("Team1_Training"):
                        if agent_team1.episodic_buffer.episodes:
                            agent_team1.train_episodic_monte_carlo()
            
            # Memoria cleanup
            with record_function("Memory_Cleanup"):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Prepara per il prossimo episodio
            first_player = (first_player + 1) % 4
        
        pbar.close()
    
    # Salva il trace file temporaneo
    try:
        prof.export_chrome_trace(TEMP_TRACE_FILE)
        print(f"\nTrace file temporaneo generato: {TEMP_TRACE_FILE}")
        
        # Sanitizza il file trace per assicurarsi che sia JSON valido
        sanitize_result = sanitize_trace_file(TEMP_TRACE_FILE, TRACE_FILE)
        
        if sanitize_result:
            # Pulizia: rimuovi il file temporaneo
            try:
                os.remove(TEMP_TRACE_FILE)
            except:
                pass
            
            # Apri Chrome con il trace viewer
            open_chrome_with_trace(TRACE_FILE)
        else:
            print("Errore nella sanitizzazione del trace. Prova a caricare il file temporaneo.")
            open_chrome_with_trace(TEMP_TRACE_FILE)
    
    except Exception as e:
        print(f"\nErrore nella generazione del trace file: {e}")
        return None
    
    return prof

if __name__ == "__main__":
    # Esegui profiling e genera il trace file
    # Usa pochi episodi e limita il numero di step per episodio per generare trace più piccoli
    train_with_chrome_trace(num_episodes=50)