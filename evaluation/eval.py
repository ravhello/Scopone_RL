import os
import torch
from tests.torch_np import np
from tqdm import tqdm
from typing import Tuple
from environment import ScoponeEnvMA
from heuristics.baseline import pick_action_heuristic
from selfplay.league import League
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False
from models.action_conditioned import ActionConditionedActor, CentralValueNet
from algorithms.is_mcts import run_is_mcts
from belief.belief import BeliefState


def play_match(agent_fn_team0, agent_fn_team1, games: int = 50, use_compact_obs: bool = True, k_history: int = 12) -> Tuple[float, dict]:
    """
    Gioca N partite e ritorna win-rate team0 e breakdown medio dei punteggi.
    agent_fn_*: callable(env) -> action (usa env.get_valid_actions())
    """
    wins = 0
    breakdown_sum = {0: {'carte': 0, 'denari': 0, 'settebello': 0, 'primiera': 0, 'scope': 0, 'total': 0},
                     1: {'carte': 0, 'denari': 0, 'settebello': 0, 'primiera': 0, 'scope': 0, 'total': 0}}
    for _ in tqdm(range(games), desc='Eval matches'):
        env = ScoponeEnvMA(use_compact_obs=use_compact_obs, k_history=k_history)
        done = False
        info = {}
        while not done:
            legals = env.get_valid_actions()
            if not legals:
                break
            if env.current_player in [0, 2]:
                action = agent_fn_team0(env)
            else:
                action = agent_fn_team1(env)
            _, _, done, info = env.step(action)
        if 'score_breakdown' in info:
            bd = info['score_breakdown']
            for t in [0, 1]:
                for k in breakdown_sum[t].keys():
                    breakdown_sum[t][k] += bd[t].get(k, 0)
            if bd[0]['total'] > bd[1]['total']:
                wins += 1
        elif 'team_rewards' in info:
            tr = info['team_rewards']
            if tr[0] > tr[1]:
                wins += 1
    # medie
    for t in [0, 1]:
        for k in breakdown_sum[t].keys():
            breakdown_sum[t][k] /= games
    return wins / games, breakdown_sum


def series_to_points(win_func, target_points=11):
    """Gioca una serie a target_points (es. 11) e ritorna vincitore (0/1)."""
    s0 = 0
    s1 = 0
    while s0 < target_points and s1 < target_points:
        w = win_func()
        if w:
            s0 += 1
        else:
            s1 += 1
    return 0 if s0 >= target_points else 1


def eval_vs_baseline(games=50, use_compact_obs=True, k_history=12, log_tb=False):
    writer = SummaryWriter(log_dir='runs/eval') if (log_tb and TB_AVAILABLE) else None
    def agent_fn_team0(env):
        # actor placeholder: usa euristica come baseline anche per team0 se serve
        return pick_action_heuristic(env.get_valid_actions())
    def agent_fn_team1(env):
        return pick_action_heuristic(env.get_valid_actions())
    wr, bd = play_match(agent_fn_team0, agent_fn_team1, games, use_compact_obs, k_history)
    if writer is not None:
        writer.add_scalar('eval/win_rate_team0', wr, 0)
        writer.close()
    return wr, bd


def evaluate_pair_actors(env=None, ckpt_path=''):
    created_env = False
    if env is None:
        env = ScoponeEnvMA(use_compact_obs=True, k_history=4)
        created_env = True
    obs_dim = env.observation_space.shape[0]
    actor = ActionConditionedActor(obs_dim=obs_dim)
    critic = CentralValueNet(obs_dim=obs_dim)
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=torch.device('cuda'))
            if isinstance(state, dict):
                if 'actor' in state:
                    actor.load_state_dict(state['actor'])
                if 'critic' in state:
                    critic.load_state_dict(state.get('critic', {}))
        except Exception:
            pass
    if created_env:
        del env
    return actor, critic


def league_eval_and_update(league_dir='checkpoints/league', games=20, target_points=11):
    """Esegue sfide tra ultimi due checkpoint registrati e aggiorna Elo nel league."""
    league = League(base_dir=league_dir)
    if len(league.history) < 2:
        return
    a, b = league.history[-2], league.history[-1]
    # funzione di win tra a e b (semplificata: usa singola partita per punto)
    def win_a():
        env = ScoponeEnvMA(use_compact_obs=True, k_history=12)
        # random outcome placeholder: in integrazione reale caricare attori e giocare
        return np.random.rand() < 0.5
    winner = series_to_points(win_a, target_points)
    league.update_elo(a, b, 1.0 if winner == 0 else 0.0)
    return league.elo



