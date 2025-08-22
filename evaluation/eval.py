import os
import torch
from tqdm import tqdm
from typing import Tuple
from environment import ScoponeEnvMA
from heuristics.baseline import pick_action_heuristic
from selfplay.league import League
try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False
from models.action_conditioned import ActionConditionedActor
from utils.device import get_compute_device
from algorithms.is_mcts import run_is_mcts
# BeliefState legacy opzionale (non usato nello scenario corrente)

device = get_compute_device()
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

def play_match(agent_fn_team0, agent_fn_team1, games: int = 50, use_compact_obs: bool = True, k_history: int = 12) -> Tuple[float, dict]:
    """
    Gioca N partite e ritorna win-rate team0 e breakdown medio dei punteggi.
    agent_fn_*: callable(env) -> action (usa env.get_valid_actions())
    """
    device = torch.device(os.environ.get(
        'SCOPONE_DEVICE',
        ('cuda' if torch.cuda.is_available() and os.environ.get('TESTS_FORCE_CPU') != '1' else 'cpu')
    ))
    wins = 0
    breakdown_sum = {0: {'carte': 0.0, 'denari': 0.0, 'settebello': 0.0, 'primiera': 0.0, 'scope': 0.0, 'total': 0.0},
                     1: {'carte': 0.0, 'denari': 0.0, 'settebello': 0.0, 'primiera': 0.0, 'scope': 0.0, 'total': 0.0}}
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
                    breakdown_sum[t][k] += float(bd[t].get(k, 0))
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
    writer = None
    if log_tb and TB_AVAILABLE and os.environ.get('SCOPONE_DISABLE_TB', '0') != '1':
        try:
            writer = _SummaryWriter(log_dir='runs/eval')
        except Exception:
            writer = None
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


def evaluate_pair_actors(ckpt_a: str, ckpt_b: str, games: int = 10,
                         use_compact_obs: bool = True, k_history: int = 12,
                         mcts: dict = None,
                         belief_particles: int = 0, belief_ess_frac: float = 0.5):
    """
    Valuta due checkpoint (A vs B) giocando N partite. Ritorna win-rate di A e breakdown medio.
    - Se mcts è fornito, usa IS-MCTS con i parametri dati per la selezione.
    - belief_particles>0 abilita belief a particelle per prior MCTS.
    """
    # Primo env per determinare obs_dim
    env0 = ScoponeEnvMA(use_compact_obs=use_compact_obs, k_history=k_history)
    obs_dim = env0.observation_space.shape[0]
    del env0
    # Carica attori
    actor_a = ActionConditionedActor(obs_dim=obs_dim).to(device)
    actor_b = ActionConditionedActor(obs_dim=obs_dim).to(device)
    try:
        if ckpt_a and os.path.isfile(ckpt_a):
            st_a = torch.load(ckpt_a, map_location=device)
            if isinstance(st_a, dict) and 'actor' in st_a:
                actor_a.load_state_dict(st_a['actor'])
        if ckpt_b and os.path.isfile(ckpt_b):
            st_b = torch.load(ckpt_b, map_location=device)
            if isinstance(st_b, dict) and 'actor' in st_b:
                actor_b.load_state_dict(st_b['actor'])
    except Exception:
        pass
    actor_a.eval(); actor_b.eval()

    def make_agent_fn(actor_model):
        def _select(env: ScoponeEnvMA):
            legals = env.get_valid_actions()
            cp = env.current_player
            # seat/team vec
            seat_vec = torch.zeros(6, dtype=torch.float32, device='cpu')
            seat_vec[cp] = 1.0
            seat_vec[4] = 1.0 if cp in [0, 2] else 0.0
            seat_vec[5] = 1.0 if cp in [1, 3] else 0.0
            # belief
            bsum = None
            # MCTS path
            if mcts is not None and len(legals) > 1:
                belief_obj = None
                def policy_fn(obs, legal_list):
                    # compute scores with actor on GPU
                    o_cpu = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    leg_cpu = torch.stack([
                        (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
                    for x in legal_list], dim=0)
                    s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
                    o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
                    s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    with torch.no_grad():
                        logits = actor_model(o_t, leg_t, s_t)
                        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
                    return probs
                # belief sampler neurale
                def belief_sampler_neural(_env):
                    try:
                        obs_cur = _env._get_observation(_env.current_player)
                        o_cpu = obs_cur if torch.is_tensor(obs_cur) else torch.as_tensor(obs_cur, dtype=torch.float32)
                        if torch.is_tensor(o_cpu):
                            o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                        s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
                        o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        with torch.no_grad():
                            state_feat = actor_model.state_enc(o_t, s_t)
                            logits = actor_model.belief_net(state_feat)
                            hand_table = o_t[:, :83]
                            hand_mask = hand_table[:, :40] > 0.5
                            table_mask = hand_table[:, 43:83] > 0.5
                            captured = o_t[:, 83:165]
                            cap0_mask = captured[:, :40] > 0.5
                            cap1_mask = captured[:, 40:80] > 0.5
                            visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)
                            probs_flat = actor_model.belief_net.probs(logits, visible_mask)
                        probs = probs_flat.view(3, 40).detach().cpu().numpy()
                        det = {}
                        others = [(_env.current_player + 1) % 4, (_env.current_player + 2) % 4, (_env.current_player + 3) % 4]
                        for i, pid in enumerate(others):
                            det[pid] = []
                        vis = visible_mask.squeeze(0).detach().cpu().numpy().astype(bool)
                        unknown_ids = [cid for cid in range(40) if not vis[cid]]
                        counts = {pid: len(_env.game_state['hands'][pid]) for pid in others}
                        caps = [int(counts.get(pid, 0)) for pid in others]
                        n = len(unknown_ids)
                        if sum(caps) != n:
                            caps[2] = max(0, n - caps[0] - caps[1])
                        # semplice greedy per eval (si può allineare alla DP del trainer se serve)
                        for cid in unknown_ids:
                            pc = probs[:, cid]
                            ps = pc / max(1e-9, pc.sum())
                            j = int(torch.argmax(torch.tensor(ps)).item())
                            if caps[j] > 0:
                                det[others[j]].append(cid)
                                caps[j] -= 1
                        return det
                    except Exception:
                        return None
                action = run_is_mcts(
                    env,
                    policy_fn=policy_fn,
                    value_fn=lambda _o, _e: 0.0,  # solo policy-guided in eval rapida
                    num_simulations=int(mcts.get('sims', 128)),
                    c_puct=float(mcts.get('c_puct', 1.0)),
                    belief=None,
                    num_determinization=int(mcts.get('dets', 1)),
                    root_temperature=float(mcts.get('root_temp', 0.0)),
                    prior_smooth_eps=float(mcts.get('prior_smooth_eps', 0.0)),
                    robust_child=True,
                    root_dirichlet_alpha=float(mcts.get('root_dirichlet_alpha', 0.25)),
                    root_dirichlet_eps=float(mcts.get('root_dirichlet_eps', 0.25)),
                    belief_sampler=belief_sampler_neural
                )
                return action
            # Greedy actor selection con belief neurale
            obs = env._get_observation(cp)
            o_cpu = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
            if torch.is_tensor(o_cpu):
                o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
            leg_cpu = torch.stack([
                (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
            for x in legals], dim=0)
            s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
            o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
            leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
            s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
            with torch.no_grad():
                logits = actor_model(o_t, leg_t, s_t)
                idx = torch.argmax(logits).to('cpu')
            return leg_cpu[idx]
        return _select

    agent_fn_team0 = make_agent_fn(actor_a)
    agent_fn_team1 = make_agent_fn(actor_b)
    wr, bd = play_match(agent_fn_team0, agent_fn_team1, games=games, use_compact_obs=use_compact_obs, k_history=k_history)
    return wr, bd


def league_eval_and_update(league_dir='checkpoints/league', games=20, target_points=11):
    """Esegue sfide tra ultimi due checkpoint registrati e aggiorna Elo nel league in base al win-rate reale."""
    league = League(base_dir=league_dir)
    if len(league.history) < 2:
        return
    a, b = league.history[-2], league.history[-1]
    # Usa serie di partite per stimare il win-rate di A contro B
    wr_ab, _ = evaluate_pair_actors(a, b, games=games, use_compact_obs=True, k_history=12, mcts=None)
    # Aggiorna Elo usando il win-rate medio come risultato fra [0,1]
    league.update_elo(a, b, wr_ab)
    return league.elo



