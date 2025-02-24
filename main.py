# main.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from environment import ScoponeEnvMA

# Parametri di rete e training
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 5000   # passi totali di training
BATCH_SIZE = 1         # Qui usiamo un update "step by step" (semplificato)
CHECKPOINT_PATH = "scopone_checkpoint.pth"

############################################################
# 1) Definiamo la rete neurale per Q-learning
############################################################

class QNetwork(nn.Module):
    """
    Rete neurale che apprende Q(s,a) data l'osservazione (3764 dimensioni).
    Siccome 512 azioni possibili, lo strato di output è di dimensione 512.
    Ma la maggior parte saranno "invalide" in ogni stato; scegliamo la massima Q fra quelle valide.
    """
    def __init__(self, obs_dim=3764, act_dim=512):
        super().__init__()
        # Esempio rete piccola: 2 layer
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )

    def forward(self, x):
        return self.net(x)  # shape: [batch_size, act_dim]

############################################################
# 2) Definizione di un Agente Q-learning
############################################################

class QAgent:
    def __init__(self, team_id):
        self.team_id = team_id
        self.qnet = QNetwork()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)
        self.epsilon = EPSILON_START
        self.train_steps = 0  # conta i passi totali di training

    def pick_action(self, obs, valid_actions):
        """
        Epsilon-greedy su Q(s,·), ma scegliamo un'azione tra quelle valide.
        """
        if random.random() < self.epsilon:
            # esplorazione
            return random.choice(valid_actions)
        else:
            # sfruttamento
            # calcoliamo Q per TUTTE le 512 azioni, poi filtriamo
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape [1, 3764]
                q_values = self.qnet(obs_t)  # shape [1, 512]
                q_values = q_values[0].cpu().numpy()  # shape [512]

            # scegliamo l'azione con Q max fra quelle valide
            best_act = None
            best_q = -1e9
            for a in valid_actions:
                if q_values[a] > best_q:
                    best_q = q_values[a]
                    best_act = a
            return best_act

    def update_epsilon(self):
        """
        Decade epsilon col passare dei training steps.
        """
        self.train_steps += 1
        self.epsilon = max(EPSILON_END, EPSILON_START - self.train_steps / EPSILON_DECAY)

    def train_step(self, obs, action, reward, next_obs, done, next_valid_actions):
        """
        Un singolo aggiornamento Q-learning:
         Q(s,a) = r + gamma * max_{a' validi}(Q(s',a'))  se not done
                  r                                     se done
        """
        # Converte in tensori
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)      # [1, 3764]
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)  # [1, 3764]
        action_t = torch.tensor([action], dtype=torch.long)              # [1]
        reward_t = torch.tensor([reward], dtype=torch.float32)           # [1]

        # Stima Q(s,a)
        q_values = self.qnet(obs_t)        # [1, 512]
        q_sa = q_values[0, action_t]       # Q(s, action)

        # Calcola il target
        if done:
            target = reward_t
        else:
            # Cerchiamo massima Q nel next state ma SOLO tra next_valid_actions
            with torch.no_grad():
                q_next = self.qnet(next_obs_t)[0]   # shape [512]
            max_q_next = max(q_next[a].item() for a in next_valid_actions)
            target = reward_t + GAMMA * max_q_next

        # MSE loss
        loss = (q_sa - target)**2

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, filename):
        """
        Salva i pesi del modello su disco.
        """
        torch.save({
            "model_state_dict": self.qnet.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps
        }, filename)
        print(f"Checkpoint salvato su {filename}")

    def load_checkpoint(self, filename):
        """
        Carica i pesi del modello.
        """
        ckpt = torch.load(filename)
        self.qnet.load_state_dict(ckpt["model_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.train_steps = ckpt["train_steps"]
        print(f"Checkpoint caricato da {filename}")

############################################################
# 3) Funzione train principale
############################################################

def train_agents(num_episodes=10):
    """
    Esegue un training multi-agent: i due agent imparano le rispettive Q-funzioni.
    Rotazione del first_player dopo ogni partita.
    Salva un checkpoint al termine.
    """
    # Creiamo i due agent (Team 0 e Team 1)
    agent_team0 = QAgent(team_id=0)
    agent_team1 = QAgent(team_id=1)

    first_player = 0

    # Ciclo sugli episodi
    for episode in range(num_episodes):
        print(f"\n=== Episodio {episode+1}, inizia player {first_player} ===")
        env = ScoponeEnvMA()
        env.current_player = first_player

        done = False
        obs_current = env._get_observation(env.current_player)  # osservazione del primo player
        # Usiamo un buffer temporaneo per memorizzare la transizione (s,a,r,s') di ciascun team
        # Poiché la reward finale è assegnata solo alla fine, accumuliamo reward = 0 finché non finisce
        # e poi assegniamo la differenza di punteggio a ciascun team.
        transitions_team0 = []
        transitions_team1 = []

        while not done:
            cp = env.current_player
            # Determiniamo quale team sta giocando
            team_id = 0 if cp in [0,2] else 1
            agent = agent_team0 if team_id==0 else agent_team1

            # Otteniamo le azioni valide
            valid_acts = env.get_valid_actions()
            # Scegliamo l'azione con epsilon-greedy
            action = agent.pick_action(obs_current, valid_acts)

            # Eseguiamo lo step
            next_obs, reward_scalar, done, info = env.step(action)
            # Qui, reward_scalar=0.0 se non done, info["team_rewards"] se done

            # Se la partita non è terminata, la reward immediata = 0
            # Mettiamo in transizione
            transitions_team0.append(None)  # placeholder
            transitions_team1.append(None)

            if team_id==0:
                transitions_team0[-1] = (obs_current, action, 0.0, next_obs, done)  
            else:
                transitions_team1[-1] = (obs_current, action, 0.0, next_obs, done)

            obs_next = next_obs

            if not done:
                # Passiamo al prossimo
                obs_current = obs_next
            else:
                # Partita terminata => calcoliamo le due reward finali
                team_rewards = info.get("team_rewards", [0.0, 0.0])  
                print(f"Team Rewards finali = {team_rewards}")
                # Assegniamo a tutti i passaggi di team0 la reward team_rewards[0],
                # a tutti i passaggi di team1 la reward team_rewards[1]
                # Poi aggiorniamo i Q-net
                update_agents_after_episode(agent_team0, transitions_team0, team_rewards[0], env)
                update_agents_after_episode(agent_team1, transitions_team1, team_rewards[1], env)

            # Decresce epsilon
            agent.update_epsilon()

        # Cambiamo first_player
        first_player = (first_player+1)%4

    # Al termine del training salviamo i pesi
    agent_team0.save_checkpoint(CHECKPOINT_PATH+"_team0")
    agent_team1.save_checkpoint(CHECKPOINT_PATH+"_team1")
    print("=== Training concluso e checkpoint salvati. ===")

def update_agents_after_episode(agent, transitions, final_reward, env):
    """
    Per ogni transizione (s, a, r=0, s'), aggiorniamo con r + final_reward. 
    Esempio semplificato: assegniamo TUTTO il reward finale all'ultima transizione,
    oppure lo assegniamo equamente a TUTTE le transizioni.
    Qui, per semplicità, lo assegniamo uguale a tutte le transizioni di quell'agente.
    """
    if not transitions:
        return

    # Se preferisci assegnare il reward finale all'ULTIMA transizione soltanto:
    #   transitions[-1] = (s, a, final_reward, s', done)
    # e le altre = 0
    # Qui facciamo "spread" = final_reward su tutte le transizioni.

    # Se la partita è durata N mosse per questo agente, distribuiamo final_reward su ognuna
    # (approccio molto naive).
    # Oppure potresti assegnare final_reward su TUTTE, generando un "monte" di training che
    # potrebbe avere un segnale più forte.
    # Scegliamo la seconda opzione, per semplicità.

    for i, tr in enumerate(transitions):
        if tr is None:
            continue
        obs, act, _, next_obs, done = tr
        # calcoliamo reward = final_reward
        # Il next_valid_actions lo otteniamo dal env, ma
        # se done==True, next_valid_actions non serve (è finale).
        if not done:
            env.game_state["current_player"] = -1  # hack: forziamo un player inesistente
            # Oppure creiamo un game_state fittizio
            # Visto che la partita e' conclusa, potremmo ipotizzare che next_valid_actions=[]
            # ma in Q-learning ci serve la massima Q tra le azioni valide del next state
            # In uno scenario + complesso creeresti un "copia" del game_state e imposteresti
            # "current_player" all' (cp+1)%4, ma la partita e' terminata, e' un edge case.
            next_valid_acts = env.get_valid_actions()  
        else:
            next_valid_acts = []

        agent.train_step(
            obs, act, final_reward, next_obs, done, next_valid_acts
        )


if __name__ == "__main__":
    train_agents(num_episodes=20)
