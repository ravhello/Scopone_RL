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
    
    @profile
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
    
    @profile
    def train_episodic_monte_carlo(self, specific_episode=None):
        """
        Addestramento Monte Carlo su episodi usando flat rewards.
        
        - Se specific_episode è fornito, usa quell'episodio specifico
        - Altrimenti, usa tutti gli episodi nel buffer
        
        La reward dell'ultima transizione dell'episodio viene applicata a tutte
        le transizioni dell'episodio in modo flat (stesso valore per tutti).
        """
        # Determina quali episodi processare
        if specific_episode is not None:
            episodes_to_process = [specific_episode]
        elif self.episodic_buffer.episodes:
            episodes_to_process = self.episodic_buffer.get_all_episodes()
        else:
            return  # Nessun episodio disponibile
        
        # Prepara tutti i dati prima di processarli
        all_transitions = []
        all_returns = []
        
        for episode in episodes_to_process:
            if not episode:
                continue
                
            # Ottieni la reward finale dall'ultima transizione dell'episodio
            final_reward = episode[-1][2] if episode else 0.0
            
            # Aggiungi tutte le transizioni, usando la stessa reward finale per tutte
            for obs, action, _, _, _, _ in episode:
                all_transitions.append((obs, action))
                all_returns.append(final_reward)  # Reward flat - stessa reward per tutte le transizioni
        
        if not all_transitions:
            return  # Nessuna transizione da processare
        
        # Process in batch più grandi per sfruttare meglio la GPU
        batch_size = min(256, len(all_transitions))
        num_batches = (len(all_transitions) + batch_size - 1) // batch_size
        
        # Riduci la frequenza di sync_target
        sync_counter = 0
        sync_frequency = 10  # Esegui sync ogni 10 batch invece di ogni batch
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_transitions))
            
            # Prepara i dati per questo batch
            batch_data = all_transitions[start_idx:end_idx]
            batch_returns = all_returns[start_idx:end_idx]
            
            # Separa osservazioni e azioni
            batch_obs, batch_actions = zip(*batch_data)
            
            # Conversione efficiente in tensori
            batch_actions_np = np.stack(batch_actions).astype(np.float32)
            
            # Trasferimento a GPU
            obs_t = torch.tensor(batch_obs, dtype=torch.float32, device=device)
            batch_actions_t = torch.tensor(batch_actions_np, dtype=torch.float32, device=device)
            returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            q_values = self.online_qnet(obs_t)
            
            # Calcolo vettoriale dei Q-values
            q_values_for_actions = torch.sum(q_values * batch_actions_t, dim=1)
            
            # Loss e backward
            loss = nn.MSELoss()(q_values_for_actions, returns_t)
            loss.backward()
            
            # Clipping del gradiente per stabilità
            torch.nn.utils.clip_grad_norm_(self.online_qnet.parameters(), max_norm=10.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Aggiorna epsilon
            self.update_epsilon()
            
            # Sync target periodicamente
            sync_counter += 1
            if sync_counter >= sync_frequency:
                self.sync_target()
                sync_counter = 0
            
            # Libera memoria
            del obs_t, batch_actions_t, returns_t, q_values, q_values_for_actions, loss
        
        # Garbage collection finale
        import gc
        gc.collect()
        if torch.cuda.is_available():
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

            # Durante il gioco, le mosse hanno reward=0 (sarà aggiornata alla fine in train_episodic_monte_carlo)
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
        
        # Ottieni le reward finali, se disponibili
        team0_reward = 0.0
        team1_reward = 0.0
        if "team_rewards" in info:
            team_rewards = info["team_rewards"]
            team0_reward = team_rewards[0]
            team1_reward = team_rewards[1]
            print(f"Team Rewards finali: {team_rewards}")
        
        # TRAINING ALLA FINE DELL'EPISODIO
        print(f"Training alla fine dell'episodio {ep+1}...")
        
        # Train team 0 - la reward finale viene applicata a tutte le mosse internamente
        if agent_team0.episodic_buffer.episodes:
            # Ottieni l'ultimo episodio
            last_episode_team0 = agent_team0.episodic_buffer.episodes[-1]
            
            # Se non ci sono transizioni, non fare training
            if not last_episode_team0:
                print("  Nessuna transizione per team 0, skip training")
            else:
                # Crea una copia dell'episodio con la reward finale aggiornata per tutte le mosse
                # Questo è necessario perché vogliamo preservare le reward originali nell'episodio
                training_episode_team0 = []
                for obs, action, _, next_obs, done, next_valid in last_episode_team0:
                    # Sostituisci la reward con la reward finale del team (flat)
                    training_episode_team0.append((obs, action, team0_reward, next_obs, done, next_valid))
                
                print(f"  Training team 0 sull'ultimo episodio (reward={team0_reward})")
                # Usa l'episodio con reward aggiornate per il training
                agent_team0.train_episodic_monte_carlo(training_episode_team0)
        
        # Train team 1 - la reward finale viene applicata a tutte le mosse internamente
        if agent_team1.episodic_buffer.episodes:
            # Ottieni l'ultimo episodio
            last_episode_team1 = agent_team1.episodic_buffer.episodes[-1]
            
            # Se non ci sono transizioni, non fare training
            if not last_episode_team1:
                print("  Nessuna transizione per team 1, skip training")
            else:
                # Crea una copia dell'episodio con la reward finale aggiornata per tutte le mosse
                # Questo è necessario perché vogliamo preservare le reward originali nell'episodio
                training_episode_team1 = []
                for obs, action, _, next_obs, done, next_valid in last_episode_team1:
                    # Sostituisci la reward con la reward finale del team (flat)
                    training_episode_team1.append((obs, action, team1_reward, next_obs, done, next_valid))
                
                print(f"  Training team 1 sull'ultimo episodio (reward={team1_reward})")
                # Usa l'episodio con reward aggiornate per il training
                agent_team1.train_episodic_monte_carlo(training_episode_team1)
        
        # Occasionalmente, trainiamo anche su episodi precedenti
        if ep % 5 == 0 and len(agent_team0.episodic_buffer.episodes) > 3:
            print("  Training aggiuntivo su episodi passati per team 0")
            prev_episodes = agent_team0.episodic_buffer.get_previous_episodes()
            past_episodes = random.sample(prev_episodes, 
                                        min(3, len(prev_episodes)))
            for episode in past_episodes:
                if episode:  # Verifica che l'episodio non sia vuoto
                    agent_team0.train_episodic_monte_carlo(episode)
        
        if ep % 5 == 0 and len(agent_team1.episodic_buffer.episodes) > 3:
            print("  Training aggiuntivo su episodi passati per team 1")
            prev_episodes = agent_team1.episodic_buffer.get_previous_episodes()
            past_episodes = random.sample(prev_episodes, 
                                        min(3, len(prev_episodes)))
            for episode in past_episodes:
                if episode:  # Verifica che l'episodio non sia vuoto
                    agent_team1.train_episodic_monte_carlo(episode)
        
        # Forza garbage collection per ridurre il consumo di memoria
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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


if __name__ == "__main__":
    # Esegui il training per pochi episodi per profilare
    train_agents(num_episodes=20)
    
    # Stampa i risultati del profiling
    global_profiler.print_stats()
    
    # Genera un report dettagliato e salvalo su file
    report = global_profiler.generate_report("profiling_report.txt")