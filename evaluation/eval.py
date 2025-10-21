import os
# Default compile-friendly settings (work on CPU and CUDA)
os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
os.environ.setdefault('SCOPONE_TORCH_COMPILE_MODE', 'max-autotune')
os.environ.setdefault('SCOPONE_TORCH_COMPILE_BACKEND', 'inductor')
os.environ.setdefault('SCOPONE_INDUCTOR_AUTOTUNE', '1')
import torch
import math
import random
import time
import platform
import multiprocessing as mp
from tqdm import tqdm
import shutil
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
from utils.compile import maybe_compile_module
from algorithms.is_mcts import run_is_mcts
# BeliefState legacy opzionale (non usato nello scenario corrente)

# Force evaluation on CPU across the project for consistency and to avoid GPU H2D overhead in unbatched eval
device = torch.device('cpu')

# Debug helper for parallel eval
_DEF_FALSE = ['0','false','no','off']
_DEF_TRUE = ['1','true','yes','on']

def _eval_debug_enabled() -> bool:
    try:
        return str(os.environ.get('SCOPONE_EVAL_DEBUG', '0')).strip().lower() in _DEF_TRUE
    except Exception:
        return False

def _dbg(msg: str):
    if _eval_debug_enabled():
        try:
            tqdm.write(f"[eval-debug] {msg}")
        except Exception:
            print(f"[eval-debug] {msg}", flush=True)

def _get_mp_ctx():
    # Decide safe start method for parallel eval
    meth = (os.environ.get('SCOPONE_EVAL_MP_START')
            or os.environ.get('SCOPONE_MP_START')
            or '').strip().lower()
    if meth in ('spawn', 'fork', 'forkserver'):
        _dbg(f"using mp start method from env: {meth}")
        return mp.get_context(meth)
    # Heuristics: prefer spawn on WSL2 or when CUDA is available; else forkserver
    try:
        is_wsl = ('WSL_INTEROP' in os.environ) or ('microsoft' in platform.release().lower())
    except Exception:
        is_wsl = False
    try:
        has_cuda = torch.cuda.is_available() or str(os.environ.get('SCOPONE_TRAIN_DEVICE','cpu')).startswith('cuda')
    except Exception:
        has_cuda = False
    if is_wsl or has_cuda:
        _dbg("using mp start method: spawn (WSL/CUDA heuristic)")
        return mp.get_context('spawn')
    _dbg("using mp start method: forkserver (default)")
    return mp.get_context('forkserver')

def _get_eval_pool_timeout_seconds():
    """
    Parse SCOPONE_EVAL_POOL_TIMEOUT_S from environment.
    Returns:
        float timeout in seconds if > 0; None to indicate 'wait indefinitely'.
    Accepted 'off' values: '0', 'none', 'inf', 'infinity', negatives.
    """
    try:
        raw = str(os.environ.get('SCOPONE_EVAL_POOL_TIMEOUT_S', '600')).strip().lower()
        if raw in ('none', 'inf', 'infinity'):
            return None
        val = float(raw)
        if val <= 0.0:
            return None
        return float(val)
    except Exception:
        return 600.0

def _true_env(val: str) -> bool:
    return str(val).strip().lower() in _DEF_TRUE

def _eval_progress_alpha(env, start: float, full: float) -> float:
    try:
        hist = env.game_state.get('history', []) if hasattr(env, 'game_state') and isinstance(env.game_state, dict) else []
        progress = float(min(1.0, max(0.0, (len(hist) if isinstance(hist, list) else 0) / 40.0)))
    except Exception:
        progress = 0.0
    denom = max(1e-6, float(full) - float(start))
    alpha = min(1.0, max(0.0, (progress - float(start)) / denom))
    return float(alpha)

def _eval_resolve_mcts_params(env, mcts_cfg: dict | None):
    """Return (sims_scaled, root_temp_dyn) for eval MCTS, mirroring training scaling.
    Uses env overrides when available:
      - SCOPONE_EVAL_MCTS_SCALING (default '1')
      - SCOPONE_EVAL_MCTS_PROGRESS_START (default '0.25')
      - SCOPONE_EVAL_MCTS_PROGRESS_FULL (default '0.75')
      - SCOPONE_EVAL_MCTS_MIN_SIMS (default '0')
      - SCOPONE_EVAL_MCTS_TRAIN_FACTOR (default '1.0')
    """
    if mcts_cfg is None:
        return 0, 0.0
    try:
        sims_def = int(mcts_cfg.get('sims', 128))
    except Exception:
        sims_def = 128
    scaling_on = _true_env(os.environ.get('SCOPONE_EVAL_MCTS_SCALING', '1'))
    try:
        ps = float(mcts_cfg.get('progress_start', os.environ.get('SCOPONE_EVAL_MCTS_PROGRESS_START', '0.25')))
        pf = float(mcts_cfg.get('progress_full', os.environ.get('SCOPONE_EVAL_MCTS_PROGRESS_FULL', '0.75')))
        min_sims = int(mcts_cfg.get('min_sims', os.environ.get('SCOPONE_EVAL_MCTS_MIN_SIMS', '0')))
        train_factor = float(mcts_cfg.get('train_factor', os.environ.get('SCOPONE_EVAL_MCTS_TRAIN_FACTOR', '1.0')))
    except Exception:
        ps, pf, min_sims, train_factor = 0.25, 0.75, 0, 1.0
    alpha = _eval_progress_alpha(env, ps, pf)
    if not scaling_on:
        sims_scaled = int(max(0, sims_def))
    else:
        import math as _math
        sims_base = int(_math.ceil(float(sims_def) * (0.25 + 0.75 * alpha)))
        sims_base = int(_math.ceil(float(sims_base) * float(train_factor)))
        sims_scaled = int(max(int(min_sims), sims_base))
        if int(sims_def) > 0 and sims_scaled <= 0:
            sims_scaled = 1
    # Dynamic root temperature like training: if root_temp<=0, use (1 - alpha)
    try:
        rt = float(mcts_cfg.get('root_temp', 0.0))
    except Exception:
        rt = 0.0
    root_temp_dyn = float(rt) if float(rt) > 0.0 else float(max(0.0, 1.0 - alpha))
    return int(sims_scaled), float(root_temp_dyn)

def _terminal_ncols() -> int:
    try:
        return max(60, int(shutil.get_terminal_size(fallback=(120, 20)).columns))
    except Exception:
        return 120

def _fit_desc(desc: str, reserve: int = 48) -> str:
    cols = _terminal_ncols()
    max_len = max(8, cols - int(reserve))
    if len(desc) <= max_len:
        return desc
    return desc[:max_len - 1] + '…'


def play_match(agent_fn_team0, agent_fn_team1, games: int = 50, k_history: int = 12,
               tqdm_desc: str = None, tqdm_position: int = 0, tqdm_disable: bool = False) -> Tuple[float, dict]:
    """
    Gioca N partite e ritorna diff media (punti actor0 − punti actor1) e breakdown medio dei punteggi.
    Il team che inizia è sorteggiato alla prima partita e poi alternato; inoltre, le posizioni
    dei team ai posti (0/2 vs 1/3) vengono invertite a ogni partita per equità.
    agent_fn_*: callable(env) -> action (usa env.get_valid_actions())
    """
    # Evaluation runs strictly on CPU (independent from training device)
    wins = 0
    actor_tot_sum = {0: 0.0, 1: 0.0}
    breakdown_sum = {0: {'carte': 0.0, 'denari': 0.0, 'settebello': 0.0, 'primiera': 0.0, 'scope': 0.0, 'total': 0.0},
                     1: {'carte': 0.0, 'denari': 0.0, 'settebello': 0.0, 'primiera': 0.0, 'scope': 0.0, 'total': 0.0}}
    def _to_float_scalar(x):
            return float(x)

    bar_desc = _fit_desc(tqdm_desc or 'Eval matches')
    _ncols = _terminal_ncols()
    _disable = bool(os.environ.get('TQDM_DISABLE','0') in ['1','true','yes','on']) or bool(tqdm_disable)
    # Estrai team iniziale che parte (0 o 1) e alterna ad ogni partita
    start_team = int(random.random() < 0.5)
    for gi in tqdm(range(games), desc=bar_desc, position=tqdm_position, dynamic_ncols=False, ncols=_ncols,
                  leave=True, disable=_disable, miniters=1, mininterval=0.05, smoothing=0):
        env = ScoponeEnvMA(k_history=k_history)
        invert_seats = (gi % 2) == 1  # inverti mappatura posti: (0,2)<->(1,3)
        team_start = start_team ^ (gi & 1)
        # Calcola starting_player coerente con inversione posti
        if not invert_seats:
            starting_player = (0 if team_start == 0 else 1)
        else:
            starting_player = (1 if team_start == 0 else 0)
        env.reset(starting_player=starting_player)
        done = False
        info = {}
        while not done:
            legals = env.get_valid_actions()
            # Robust emptiness check: supports list/tuple or tensor returns
            _empty = False
            if legals is None:
                _empty = True
            elif isinstance(legals, (list, tuple)):
                _empty = (len(legals) == 0)
            elif torch.is_tensor(legals):
                _empty = (legals.numel() == 0) or (legals.shape[0] == 0)
            else:
                _empty = (len(legals) == 0)
            if _empty:
                break
            # Se invert_seats, il team0 actor gioca sui posti 1/3 (parità invertita)
            if not invert_seats:
                if env.current_player in [0, 2]:
                    action = agent_fn_team0(env)
                else:
                    action = agent_fn_team1(env)
            else:
                if env.current_player in [0, 2]:
                    action = agent_fn_team1(env)
                else:
                    action = agent_fn_team0(env)
            _, _, done, info = env.step(action)
        if 'score_breakdown' in info:
            bd = info['score_breakdown']
            for t in [0, 1]:
                for k in breakdown_sum[t].keys():
                    breakdown_sum[t][k] += _to_float_scalar(bd[t].get(k, 0))
            _t0 = _to_float_scalar(bd[0].get('total', 0))
            _t1 = _to_float_scalar(bd[1].get('total', 0))
            if not invert_seats:
                actor_tot_sum[0] += _t0
                actor_tot_sum[1] += _t1
                if _t0 > _t1:
                    wins += 1
            else:
                actor_tot_sum[0] += _t1
                actor_tot_sum[1] += _t0
                if _t1 > _t0:
                    wins += 1
        elif 'team_rewards' in info:
            tr = info['team_rewards']
            _r0 = _to_float_scalar(tr[0])
            _r1 = _to_float_scalar(tr[1])
            if not invert_seats:
                actor_tot_sum[0] += _r0
                actor_tot_sum[1] += _r1
                if _r0 > _r1:
                    wins += 1
            else:
                actor_tot_sum[0] += _r1
                actor_tot_sum[1] += _r0
                if _r1 > _r0:
                    wins += 1
    # medie
    for t in [0, 1]:
        for k in breakdown_sum[t].keys():
            breakdown_sum[t][k] /= games
    wr = wins / games if games > 0 else 0.0
    breakdown_sum['meta'] = {'win_rate': float(wr)}
    diff_avg = (actor_tot_sum[0] - actor_tot_sum[1]) / float(games) if games > 0 else 0.0
    return float(diff_avg), breakdown_sum


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


def eval_vs_baseline(games=50, k_history=12, log_tb=False):
    writer = None
    if log_tb and TB_AVAILABLE and os.environ.get('SCOPONE_DISABLE_TB', '0') != '1':
        writer = _SummaryWriter(log_dir='runs/eval')
    def agent_fn_team0(env):
        # actor placeholder: usa euristica come baseline anche per team0 se serve
        return pick_action_heuristic(env.get_valid_actions())
    def agent_fn_team1(env):
        return pick_action_heuristic(env.get_valid_actions())
    wr, bd = play_match(agent_fn_team0, agent_fn_team1, games, k_history)
    if writer is not None:
        writer.add_scalar('eval/win_rate_team0', wr, 0)
        writer.close()
    return wr, bd


def evaluate_pair_actors(ckpt_a: str, ckpt_b: str, games: int = 10,
                         k_history: int = 12,
                         mcts: dict = None,
                         belief_particles: int = 0, belief_ess_frac: float = 0.5,
                         tqdm_desc: str = None, tqdm_position: int = 0, tqdm_disable: bool = False):
    """
    Valuta due checkpoint (A vs B) giocando N partite. Ritorna diff media (punti A − punti B) e breakdown medio.
    - Se mcts è fornito, usa IS-MCTS con i parametri dati per la selezione.
    - belief_particles>0 abilita belief a particelle per prior MCTS.
    """
    # Primo env per determinare obs_dim
    env0 = ScoponeEnvMA(k_history=k_history)
    obs_dim = env0.observation_space.shape[0]
    del env0
    # Carica attori
    actor_a = maybe_compile_module(ActionConditionedActor(obs_dim=obs_dim), name='ActionConditionedActor[eval_A]').to(device)
    actor_b = maybe_compile_module(ActionConditionedActor(obs_dim=obs_dim), name='ActionConditionedActor[eval_B]').to(device)
    if ckpt_a and os.path.isfile(ckpt_a):
        st_a = torch.load(ckpt_a, map_location=device)
        if isinstance(st_a, dict) and 'actor' in st_a:
            actor_a.load_state_dict(st_a['actor'])
    if ckpt_b and os.path.isfile(ckpt_b):
        st_b = torch.load(ckpt_b, map_location=device)
        if isinstance(st_b, dict) and 'actor' in st_b:
            actor_b.load_state_dict(st_b['actor'])
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
                    if device.type == 'cuda':
                        o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
                        s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    else:
                        o_t = o_cpu.unsqueeze(0).to(device=device)
                        leg_t = leg_cpu.to(device=device)
                        s_t = s_cpu.unsqueeze(0).to(device=device)
                    with torch.no_grad():
                        logits = actor_model(o_t, leg_t, s_t)
                        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
                    return probs
                # belief sampler neurale
                def belief_sampler_neural(_env):
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
                    log_prob = 0.0
                    for cid in unknown_ids:
                        pc = probs[:, cid]
                        ps = pc / max(1e-9, pc.sum())
                        j = int(torch.argmax(torch.tensor(ps)).item())
                        if caps[j] > 0:
                            det[others[j]].append(cid)
                            caps[j] -= 1
                            log_prob += math.log(max(1e-12, float(ps[j])))
                    return {'assignments': det, 'logp': log_prob}
                # Exact-only handling: skip prior MCTS unless near end
                def _remaining_moves_eval(_env):
                    _h = _env.game_state.get('hands', None)
                    if isinstance(_h, (list, tuple)):
                        return int(sum(len(x) for x in _h))
                    if isinstance(_h, dict):
                        return int(sum(len(x) for x in _h.values()))
                    return 0
                def _auto_thresh_eval(_rem):
                    if _rem <= 0:
                        return 0
                    _hh = int(round(_rem * 0.45))
                    return max(6, min(16, _hh))
                _exact_only = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_ONLY', '0')).strip().lower() in ['1','true','yes','on']
                _rem = _remaining_moves_eval(env)
                _ovr = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_MAX_MOVES', '')).strip()
                if _ovr:
                    try:
                        _thr = max(0, int(_ovr))
                    except ValueError:
                        _thr = _auto_thresh_eval(_rem)
                else:
                    _thr = _auto_thresh_eval(_rem)
                _near_end = (_thr > 0) and (_rem <= _thr)
                if _exact_only:
                    sims_scaled = 0
                    root_temp_dyn = 0.0
                else:
                    sims_scaled, root_temp_dyn = _eval_resolve_mcts_params(env, dict(mcts))
                # Ensure per-context depth/exact settings propagate to MCTS core
                if _exact_only:
                    os.environ['SCOPONE_MCTS_MAX_DEPTH'] = '0'
                else:
                    os.environ['SCOPONE_MCTS_MAX_DEPTH'] = str(os.environ.get('SCOPONE_EVAL_MCTS_MAX_DEPTH', '0'))
                os.environ['SCOPONE_MCTS_EXACT_MAX_MOVES'] = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_MAX_MOVES', ''))
                os.environ['SCOPONE_MCTS_EXACT_COVER_FRAC'] = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_COVER_FRAC', '0'))
                mc = dict(mcts)
                mc['sims'] = int(sims_scaled)
                mc['root_temp'] = float(root_temp_dyn)
                # dets: separate prior vs exact in eval
                _d_prior = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS_PRIOR', str(mc.get('dets', 1))))
                _d_exact = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS_EXACT', str(mc.get('dets', 1))))
                if _exact_only:
                    _d_eff = int(_d_exact)
                else:
                    _d_eff = int(_d_exact if _near_end else _d_prior)
                action = run_is_mcts(
                    env,
                    policy_fn=policy_fn,
                    value_fn=lambda _o, _e: 0.0,  # solo policy-guided in eval rapida
                    num_simulations=int(mc.get('sims', 128)),
                    c_puct=float(mc.get('c_puct', 1.0)),
                    belief=None,
                    num_determinization=int(_d_eff),
                    root_temperature=float(mc.get('root_temp', 0.0)),
                    prior_smooth_eps=float(mc.get('prior_smooth_eps', 0.0)),
                    robust_child=True,
                    root_dirichlet_alpha=float(mc.get('root_dirichlet_alpha', 0.25)),
                    root_dirichlet_eps=float(mc.get('root_dirichlet_eps', 0.25)),
                    belief_sampler=belief_sampler_neural,
                    exact_only=_exact_only
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
            if device.type == 'cuda':
                o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
                s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
            else:
                o_t = o_cpu.unsqueeze(0).to(device=device)
                leg_t = leg_cpu.to(device=device)
                s_t = s_cpu.unsqueeze(0).to(device=device)
            with torch.no_grad():
                logits = actor_model(o_t, leg_t, s_t)
                idx = torch.argmax(logits).to('cpu')
            return leg_cpu[idx]
        # If a global line-profiler decorator is provided, wrap the closure so time
        # is attributed inside the function rather than to its call sites.
        _LP = globals().get('LINE_PROFILE_DECORATOR', None)
        if _LP is not None:
            _select = _LP(_select)
        return _select

    t_eval_start = time.time()
    agent_fn_team0 = make_agent_fn(actor_a)
    agent_fn_team1 = make_agent_fn(actor_b)
    # Default description uses basename of checkpoints
    if tqdm_desc is None:
        a_name = os.path.basename(ckpt_a) if ckpt_a else 'A'
        b_name = os.path.basename(ckpt_b) if ckpt_b else 'B'
        tqdm_desc = f"Eval {a_name} vs {b_name}"
    # Valuta workers paralleli da env: SCOPONE_EVAL_WORKERS>1 abilita versione parallela
    num_workers_env = int(os.environ.get('SCOPONE_EVAL_WORKERS', '1'))
    if num_workers_env > 1:
        _dbg(f"parallel eval requested: workers={num_workers_env} games={int(games)} dets={(mcts or {}).get('dets', 1)} desc={tqdm_desc}")
        # Distribuisci le determinizzazioni tra i worker: dets_per_worker >= 1
        dets_total = int((mcts or {}).get('dets', 1))
        if dets_total > 1:
            dets_per_worker = max(1, dets_total // num_workers_env)
            dets_rem = dets_total - dets_per_worker * num_workers_env
            # Costruisci per-worker mcts con dets distribuiti quasi uniformemente
            mcts_base = dict(mcts or {})
            mcts_list = []
            for i in range(num_workers_env):
                d = dets_per_worker + (1 if i < dets_rem else 0)
                mc = dict(mcts_base)
                mc['dets'] = int(max(1, d))
                mcts_list.append(mc)
            _dbg(f"distributing dets across workers: total={dets_total} list={[mc['dets'] for mc in mcts_list]}")
            diff, bd = evaluate_pair_actors_parallel_dist(ckpt_a, ckpt_b, games=games, k_history=k_history,
                                                          mcts_list=mcts_list, belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                                          num_workers=num_workers_env,
                                                          tqdm_desc=tqdm_desc, tqdm_position=int(tqdm_position or 0), tqdm_disable=bool(tqdm_disable))
            elapsed = time.time() - t_eval_start
            try:
                tqdm.write(f"[eval] {tqdm_desc}: {int(games)} games in {elapsed:.2f}s (workers={num_workers_env}, dist-dets)")
            except Exception:
                print(f"[eval] {tqdm_desc}: {int(games)} games in {elapsed:.2f}s (workers={num_workers_env}, dist-dets)", flush=True)
            return diff, bd
        # Nessuna multi-dets: usa parallelo standard
        _dbg("starting parallel eval without multi-dets distribution")
        diff, bd = evaluate_pair_actors_parallel(ckpt_a, ckpt_b, games=games, k_history=k_history,
                                             mcts=mcts, belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                             num_workers=num_workers_env,
                                             tqdm_desc=tqdm_desc, tqdm_position=int(tqdm_position or 0), tqdm_disable=bool(tqdm_disable))
        elapsed = time.time() - t_eval_start
        try:
            tqdm.write(f"[eval] {tqdm_desc}: {int(games)} games in {elapsed:.2f}s (workers={num_workers_env})")
        except Exception:
            print(f"[eval] {tqdm_desc}: {int(games)} games in {elapsed:.2f}s (workers={num_workers_env})", flush=True)
        return diff, bd
    else:
        diff, bd = play_match(agent_fn_team0, agent_fn_team1, games=games, k_history=k_history,
                            tqdm_desc=tqdm_desc, tqdm_position=int(tqdm_position or 0), tqdm_disable=bool(tqdm_disable))
        elapsed = time.time() - t_eval_start
        try:
            tqdm.write(f"[eval] {tqdm_desc}: {int(games)} games in {elapsed:.2f}s (workers=1)")
        except Exception:
            print(f"[eval] {tqdm_desc}: {int(games)} games in {elapsed:.2f}s (workers=1)", flush=True)
        return diff, bd


def _eval_pair_chunk_worker(args):
    """Worker: esegue un sottoinsieme di partite e ritorna (wins_int, breakdown_sum_dict)."""
    # Args can be (wid, ckpt_a, ckpt_b, games, ...) or without wid
    if len(args) == 8:
        wid, ckpt_a, ckpt_b, games, k_history, mcts, belief_particles, belief_ess_frac = args
    else:
        wid = -1
        (ckpt_a, ckpt_b, games, k_history, mcts, belief_particles, belief_ess_frac) = args
    _dbg(f"worker[{wid}] start: games={int(games)} k_history={int(k_history)} dets={(mcts or {}).get('dets',1)}")
    # Limit CPU threads per worker
    try:
        wt = int(os.environ.get('SCOPONE_WORKER_THREADS', '1'))
        os.environ['OMP_NUM_THREADS'] = str(wt)
        os.environ['MKL_NUM_THREADS'] = str(wt)
        torch.set_num_threads(wt)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    # Ricrea attori in ciascun processo
    from models.action_conditioned import ActionConditionedActor
    from utils.compile import maybe_compile_module
    from environment import ScoponeEnvMA
    import os as _os
    _os.environ['TQDM_DISABLE'] = '1'
    env0 = ScoponeEnvMA(k_history=k_history)
    obs_dim = env0.observation_space.shape[0]
    del env0
    _dbg(f"worker[{wid}] loading actors …")
    actor_a = maybe_compile_module(ActionConditionedActor(obs_dim=obs_dim), name='ActionConditionedActor[eval_A]').to(device)
    actor_b = maybe_compile_module(ActionConditionedActor(obs_dim=obs_dim), name='ActionConditionedActor[eval_B]').to(device)
    if ckpt_a and os.path.isfile(ckpt_a):
        st_a = torch.load(ckpt_a, map_location=device)
        if isinstance(st_a, dict) and 'actor' in st_a:
            actor_a.load_state_dict(st_a['actor'])
    if ckpt_b and os.path.isfile(ckpt_b):
        st_b = torch.load(ckpt_b, map_location=device)
        if isinstance(st_b, dict) and 'actor' in st_b:
            actor_b.load_state_dict(st_b['actor'])
    actor_a.eval(); actor_b.eval()
    _dbg(f"worker[{wid}] actors ready; starting matches …")

    def make_agent_fn(actor_model):
        def _select(env: ScoponeEnvMA):
            legals = env.get_valid_actions()
            cp = env.current_player
            seat_vec = torch.zeros(6, dtype=torch.float32, device='cpu')
            seat_vec[cp] = 1.0
            seat_vec[4] = 1.0 if cp in [0, 2] else 0.0
            seat_vec[5] = 1.0 if cp in [1, 3] else 0.0
            if mcts is not None and len(legals) > 1:
                belief_obj = None
                def policy_fn(obs, legal_list):
                    o_cpu = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    leg_cpu = torch.stack([
                        (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
                    for x in legal_list], dim=0)
                    s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
                    o_t = o_cpu.unsqueeze(0).to(device=device)
                    leg_t = leg_cpu.to(device=device)
                    s_t = s_cpu.unsqueeze(0).to(device=device)
                    with torch.no_grad():
                        logits = actor_model(o_t, leg_t, s_t)
                        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
                    return probs
                def belief_sampler_neural(_env):
                    obs_cur = _env._get_observation(_env.current_player)
                    o_cpu = obs_cur if torch.is_tensor(obs_cur) else torch.as_tensor(obs_cur, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
                    o_t = o_cpu.unsqueeze(0).to(device=device)
                    s_t = s_cpu.unsqueeze(0).to(device=device)
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
                    log_prob = 0.0
                    for cid in unknown_ids:
                        pc = probs[:, cid]
                        ps = pc / max(1e-9, pc.sum())
                        j = int(torch.argmax(torch.tensor(ps)).item())
                        if caps[j] > 0:
                            det[others[j]].append(cid)
                            caps[j] -= 1
                            log_prob += math.log(max(1e-12, float(ps[j])))
                    return {'assignments': det, 'logp': log_prob}
                from algorithms.is_mcts import run_is_mcts
                # Exact-only handling: skip prior MCTS unless near end
                def _remaining_moves_eval2(_env):
                    _h = _env.game_state.get('hands', None)
                    if isinstance(_h, (list, tuple)):
                        return int(sum(len(x) for x in _h))
                    if isinstance(_h, dict):
                        return int(sum(len(x) for x in _h.values()))
                    return 0
                def _auto_thresh_eval2(_rem):
                    if _rem <= 0:
                        return 0
                    _hh = int(round(_rem * 0.45))
                    return max(6, min(16, _hh))
                _exact_only = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_ONLY', '0')).strip().lower() in ['1','true','yes','on']
                _rem = _remaining_moves_eval2(env)
                _ovr = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_MAX_MOVES', '')).strip()
                if _ovr:
                    try:
                        _thr = max(0, int(_ovr))
                    except ValueError:
                        _thr = _auto_thresh_eval2(_rem)
                else:
                    _thr = _auto_thresh_eval2(_rem)
                _near_end = (_thr > 0) and (_rem <= _thr)
                if _exact_only:
                    sims_scaled = 0
                    root_temp_dyn = 0.0
                else:
                    sims_scaled, root_temp_dyn = _eval_resolve_mcts_params(env, dict(mcts))
                # Ensure per-context depth/exact settings propagate to MCTS core
                if _exact_only:
                    os.environ['SCOPONE_MCTS_MAX_DEPTH'] = '0'
                else:
                    os.environ['SCOPONE_MCTS_MAX_DEPTH'] = str(os.environ.get('SCOPONE_EVAL_MCTS_MAX_DEPTH', '0'))
                os.environ['SCOPONE_MCTS_EXACT_MAX_MOVES'] = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_MAX_MOVES', ''))
                os.environ['SCOPONE_MCTS_EXACT_COVER_FRAC'] = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_COVER_FRAC', '0'))
                mc = dict(mcts)
                mc['sims'] = int(sims_scaled)
                mc['root_temp'] = float(root_temp_dyn)
                _d_prior = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS_PRIOR', str(mc.get('dets', 1))))
                _d_exact = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS_EXACT', str(mc.get('dets', 1))))
                if _exact_only:
                    _d_eff = int(_d_exact)
                else:
                    _d_eff = int(_d_exact if _near_end else _d_prior)
                action = run_is_mcts(
                    env,
                    policy_fn=policy_fn,
                    value_fn=lambda _o, _e: 0.0,
                    num_simulations=int(mc.get('sims', 128)),
                    c_puct=float(mc.get('c_puct', 1.0)),
                    belief=None,
                    num_determinization=int(_d_eff),
                    root_temperature=float(mc.get('root_temp', 0.0)),
                    prior_smooth_eps=float(mc.get('prior_smooth_eps', 0.0)),
                    robust_child=True,
                    root_dirichlet_alpha=float(mc.get('root_dirichlet_alpha', 0.25)),
                    root_dirichlet_eps=float(mc.get('root_dirichlet_eps', 0.25)),
                    belief_sampler=belief_sampler_neural,
                    exact_only=_exact_only
                )
                return action
            obs = env._get_observation(cp)
            o_cpu = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
            if torch.is_tensor(o_cpu):
                o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
            leg_cpu = torch.stack([
                (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
            for x in legals], dim=0)
            s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
            o_t = o_cpu.unsqueeze(0).to(device=device)
            leg_t = leg_cpu.to(device=device)
            s_t = s_cpu.unsqueeze(0).to(device=device)
            with torch.no_grad():
                logits = actor_model(o_t, leg_t, s_t)
                idx = torch.argmax(logits).to('cpu')
            return leg_cpu[idx]
        return _select

    agent_fn_team0 = make_agent_fn(actor_a)
    agent_fn_team1 = make_agent_fn(actor_b)
    diff, bd = play_match(agent_fn_team0, agent_fn_team1, games=games, k_history=k_history,
                        tqdm_desc=None, tqdm_position=0, tqdm_disable=True)
    # Converti breakdown medio in somma per aggregazione
    bd_sum = {0: {}, 1: {}}
    for t in [0, 1]:
        for k in bd[t].keys():
            bd_sum[t][k] = float(bd[t][k]) * float(games)
    wr = float((bd.get('meta') or {}).get('win_rate', 0.0))
    wins_int = int(round(wr * games))
    diff_sum = float(diff) * float(games)
    _dbg(f"worker[{wid}] done: wins={wins_int}/{int(games)} diff_sum={diff_sum:.2f}")
    return diff_sum, bd_sum, wins_int, int(games)


def evaluate_pair_actors_parallel(ckpt_a: str, ckpt_b: str, games: int = 10,
                                  k_history: int = 12,
                                  mcts: dict = None,
                                  belief_particles: int = 0, belief_ess_frac: float = 0.5,
                                  num_workers: int = 2,
                                  tqdm_desc: str = None, tqdm_position: int = 0, tqdm_disable: bool = True):
    """Esegue l'eval in parallelo su più processi e aggrega i risultati con progress condiviso."""
    num_workers = max(1, int(num_workers))
    if num_workers == 1 or games <= 1:
        return evaluate_pair_actors(ckpt_a, ckpt_b, games, k_history, mcts, belief_particles, belief_ess_frac, tqdm_desc, tqdm_position, tqdm_disable)
    # Suddividi i giochi in chunk
    base = games // num_workers
    rem = games % num_workers
    chunks = [base + (1 if i < rem else 0) for i in range(num_workers)]
    chunks = [c for c in chunks if c > 0]
    args_list = [(i, ckpt_a, ckpt_b, chunks[i], k_history, mcts, belief_particles, belief_ess_frac) for i in range(len(chunks))]
    _dbg(f"parallel setup: workers={num_workers} chunks={chunks} args={len(args_list)}")
    # Progress bar aggregata
    total_games = sum(chunks)
    pbar = None
    if not tqdm_disable:
        pbar = tqdm(total=total_games, desc=(tqdm_desc or 'Eval'), position=int(tqdm_position or 0), dynamic_ncols=True, leave=True)
    # Esegui i worker in modo che possiamo aggiornare il pbar al completamento
    ctx = _get_mp_ctx()
    results = []
    _dbg("creating pool and dispatching tasks …")
    with ctx.Pool(processes=len(args_list)) as pool:
        try:
            async_res = pool.imap_unordered(_eval_pair_chunk_worker, args_list)
            timeout_s = _get_eval_pool_timeout_seconds()
            for idx in range(len(args_list)):
                _dbg(f"waiting result {idx+1}/{len(args_list)} …")
                while True:
                    try:
                        item = async_res.next() if timeout_s is None else async_res.next(timeout=timeout_s)
                    except mp.TimeoutError as te:
                        # If timeout is disabled, keep waiting indefinitely
                        if timeout_s is None:
                            _dbg("no result yet; keep waiting …")
                            continue
                        raise
                    try:
                        diff_sum_i, bd_sum_i, wins_i, games_i = item
                    except Exception as e:
                        _dbg(f"received malformed result {item!r}: {e}")
                        break
                    if pbar is not None:
                        pbar.update(int(games_i))
                    results.append((diff_sum_i, bd_sum_i, wins_i, games_i))
                    break
        except mp.TimeoutError as te:
            _dbg("pool timeout while waiting for results; terminating pool …")
            pool.terminate()
            raise RuntimeError(f"Evaluation timeout: no result within SCOPONE_EVAL_POOL_TIMEOUT_S for {tqdm_desc or 'Eval'}") from te
        except KeyboardInterrupt:
            _dbg("KeyboardInterrupt: terminating pool …")
            pool.terminate()
            raise
        except Exception as e:
            _dbg(f"pool.imap_unordered raised: {type(e).__name__}: {e}")
            pool.terminate()
            raise
        else:
            _dbg("all chunks finished, closing pool …")
    if pbar is not None:
        pbar.close()
    _dbg(f"aggregating {len(results)} results")
    total_games = 0
    total_wins = 0
    total_diff_sum = 0.0
    agg = {0: {}, 1: {}}
    for diff_sum_i, bd_sum_i, wins_i, games_i in results:
        total_games += int(games_i)
        total_wins += int(wins_i)
        total_diff_sum += float(diff_sum_i)
        for t in [0, 1]:
            for k, v in bd_sum_i[t].items():
                agg[t][k] = agg[t].get(k, 0.0) + float(v)
    # Media finale breakdown
    if total_games <= 0:
        return 0.0, {0: {}, 1: {}}
    bd_avg = {0: {}, 1: {}}
    for t in [0, 1]:
        for k, v in agg[t].items():
            bd_avg[t][k] = (float(v) / float(total_games))
    wr = float(total_wins) / float(total_games)
    bd_avg['meta'] = {'win_rate': float(wr)}
    diff_avg = float(total_diff_sum) / float(total_games)
    return float(diff_avg), bd_avg


def _eval_pair_chunk_worker_dist(args):
    """Worker: come _eval_pair_chunk_worker ma con mcts per-worker (es. dets distribuiti)."""
    (ckpt_a, ckpt_b, games, k_history, mcts_per_worker, belief_particles, belief_ess_frac) = args
    from models.action_conditioned import ActionConditionedActor
    from utils.compile import maybe_compile_module
    from environment import ScoponeEnvMA
    import os as _os
    _os.environ['TQDM_DISABLE'] = '1'
    env0 = ScoponeEnvMA(k_history=k_history)
    obs_dim = env0.observation_space.shape[0]
    del env0
    actor_a = maybe_compile_module(ActionConditionedActor(obs_dim=obs_dim), name='ActionConditionedActor[eval_A]').to(device)
    actor_b = maybe_compile_module(ActionConditionedActor(obs_dim=obs_dim), name='ActionConditionedActor[eval_B]').to(device)
    if ckpt_a and os.path.isfile(ckpt_a):
        st_a = torch.load(ckpt_a, map_location=device)
        if isinstance(st_a, dict) and 'actor' in st_a:
            actor_a.load_state_dict(st_a['actor'])
    if ckpt_b and os.path.isfile(ckpt_b):
        st_b = torch.load(ckpt_b, map_location=device)
        if isinstance(st_b, dict) and 'actor' in st_b:
            actor_b.load_state_dict(st_b['actor'])
    actor_a.eval(); actor_b.eval()

    def make_agent_fn(actor_model, mcts_local):
        def _select(env: ScoponeEnvMA):
            legals = env.get_valid_actions()
            cp = env.current_player
            seat_vec = torch.zeros(6, dtype=torch.float32, device='cpu')
            seat_vec[cp] = 1.0
            seat_vec[4] = 1.0 if cp in [0, 2] else 0.0
            seat_vec[5] = 1.0 if cp in [1, 3] else 0.0
            if mcts_local is not None and len(legals) > 1:
                def policy_fn(obs, legal_list):
                    o_cpu = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    leg_cpu = torch.stack([
                        (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
                    for x in legal_list], dim=0)
                    s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
                    o_t = o_cpu.unsqueeze(0).to(device=device)
                    leg_t = leg_cpu.to(device=device)
                    s_t = s_cpu.unsqueeze(0).to(device=device)
                    with torch.no_grad():
                        logits = actor_model(o_t, leg_t, s_t)
                        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
                    return probs
                from algorithms.is_mcts import run_is_mcts
                _exact_only = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_ONLY', '0')).strip().lower() in ['1', 'true', 'yes', 'on']
                if _exact_only:
                    sims_scaled = 0
                    root_temp_dyn = 0.0
                    dets_eff = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS_EXACT', str(mcts_local.get('dets', 1))))
                else:
                    sims_scaled, root_temp_dyn = _eval_resolve_mcts_params(env, dict(mcts_local))
                    dets_eff = int(mcts_local.get('dets', 1))
                mc = dict(mcts_local)
                mc['sims'] = int(sims_scaled)
                mc['root_temp'] = float(root_temp_dyn)
                mc['dets'] = int(dets_eff)
                if _exact_only:
                    os.environ['SCOPONE_MCTS_MAX_DEPTH'] = '0'
                else:
                    os.environ['SCOPONE_MCTS_MAX_DEPTH'] = str(os.environ.get('SCOPONE_EVAL_MCTS_MAX_DEPTH', '0'))
                os.environ['SCOPONE_MCTS_EXACT_MAX_MOVES'] = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_MAX_MOVES', ''))
                os.environ['SCOPONE_MCTS_EXACT_COVER_FRAC'] = str(os.environ.get('SCOPONE_EVAL_MCTS_EXACT_COVER_FRAC', '0'))
                action = run_is_mcts(
                    env,
                    policy_fn=policy_fn,
                    value_fn=lambda _o, _e: 0.0,
                    num_simulations=int(mc.get('sims', 128)),
                    c_puct=float(mc.get('c_puct', 1.0)),
                    belief=None,
                    num_determinization=int(mc.get('dets', 1)),
                    root_temperature=float(mc.get('root_temp', 0.0)),
                    prior_smooth_eps=float(mc.get('prior_smooth_eps', 0.0)),
                    robust_child=True,
                    root_dirichlet_alpha=float(mc.get('root_dirichlet_alpha', 0.25)),
                    root_dirichlet_eps=float(mc.get('root_dirichlet_eps', 0.25)),
                    exact_only=_exact_only
                )
                return action
            obs = env._get_observation(cp)
            o_cpu = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
            if torch.is_tensor(o_cpu):
                o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
            leg_cpu = torch.stack([
                (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
            for x in legals], dim=0)
            s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
            o_t = o_cpu.unsqueeze(0).to(device=device)
            leg_t = leg_cpu.to(device=device)
            s_t = s_cpu.unsqueeze(0).to(device=device)
            with torch.no_grad():
                logits = actor_model(o_t, leg_t, s_t)
                idx = torch.argmax(logits).to('cpu')
            return leg_cpu[idx]
        return _select

    agent_fn_team0 = make_agent_fn(actor_a, mcts_per_worker)
    agent_fn_team1 = make_agent_fn(actor_b, mcts_per_worker)
    diff, bd = play_match(agent_fn_team0, agent_fn_team1, games=games, k_history=k_history,
                        tqdm_desc=None, tqdm_position=0, tqdm_disable=True)
    bd_sum = {0: {}, 1: {}}
    for t in [0, 1]:
        for k in bd[t].keys():
            bd_sum[t][k] = float(bd[t][k]) * float(games)
    wr = float((bd.get('meta') or {}).get('win_rate', 0.0))
    wins_int = int(round(wr * games))
    diff_sum = float(diff) * float(games)
    return diff_sum, bd_sum, wins_int, int(games)


def evaluate_pair_actors_parallel_dist(ckpt_a: str, ckpt_b: str, games: int = 10,
                                       k_history: int = 12,
                                       mcts_list: list = None,
                                       belief_particles: int = 0, belief_ess_frac: float = 0.5,
                                       num_workers: int = 2,
                                       tqdm_desc: str = None, tqdm_position: int = 0, tqdm_disable: bool = True):
    """Parallelo con MCTS per-worker (es. dets distribuiti) con progress condiviso."""
    num_workers = max(1, int(num_workers))
    if not mcts_list or len(mcts_list) == 0:
        return evaluate_pair_actors_parallel(ckpt_a, ckpt_b, games, k_history, None, belief_particles, belief_ess_frac, num_workers, tqdm_desc, tqdm_position, tqdm_disable)
    # Chunk giochi per worker
    base = games // num_workers
    rem = games % num_workers
    chunks = [base + (1 if i < rem else 0) for i in range(num_workers)]
    chunks = [c for c in chunks if c > 0]
    # Allinea mcts_list ai worker (se meno, ricicla gli ultimi; se più, tronca)
    mcts_eff = (mcts_list + mcts_list[-1:]*num_workers)[:num_workers]
    max_chunk = int(os.environ.get('SCOPONE_EVAL_MAX_GAMES_PER_CHUNK', '32'))
    if max_chunk <= 0:
        max_chunk = None
    args_list: list = []
    for i, games_i in enumerate(chunks):
        if games_i <= 0:
            continue
        cfg = mcts_eff[min(i, len(mcts_eff) - 1)] if len(mcts_eff) > 0 else None
        remaining = games_i
        while remaining > 0:
            take = remaining if max_chunk is None else min(remaining, max_chunk)
            args_list.append((ckpt_a, ckpt_b, int(take), k_history, cfg, belief_particles, belief_ess_frac))
            remaining -= take
    if len(args_list) == 0:
        return 0.0, {0: {}, 1: {}}
    _dbg(f"parallel-dist setup: workers={num_workers} chunks={chunks} tasks={len(args_list)} dets={[m.get('dets',1) for m in mcts_eff]} max_chunk={max_chunk if max_chunk is not None else 'inf'}")
    # Progress bar aggregata
    total_games = sum(chunks)
    pbar = None
    if not tqdm_disable:
        pbar = tqdm(total=total_games, desc=(tqdm_desc or 'Eval'), position=int(tqdm_position or 0), dynamic_ncols=True, leave=True)
    ctx = _get_mp_ctx()
    results = []
    pool_size = max(1, min(num_workers, len(args_list)))
    with ctx.Pool(processes=pool_size) as pool:
        try:
            async_res = pool.imap_unordered(_eval_pair_chunk_worker_dist, args_list)
            timeout_s = _get_eval_pool_timeout_seconds()
            for idx in range(len(args_list)):
                _dbg(f"waiting dist result {idx+1}/{len(args_list)} …")
                while True:
                    try:
                        item = async_res.next() if timeout_s is None else async_res.next(timeout=timeout_s)
                    except mp.TimeoutError as te:
                        if timeout_s is None:
                            _dbg("no dist result yet; keep waiting …")
                            continue
                        raise
                    try:
                        diff_sum_i, bd_sum_i, wins_i, games_i = item
                    except Exception as e:
                        _dbg(f"received malformed result {item!r}: {e}")
                        break
                    if pbar is not None:
                        pbar.update(int(games_i))
                    results.append((diff_sum_i, bd_sum_i, wins_i, games_i))
                    break
        except mp.TimeoutError as te:
            _dbg("pool timeout (dist) while waiting for results; terminating pool …")
            pool.terminate()
            raise RuntimeError(f"Evaluation timeout (dist): no result within SCOPONE_EVAL_POOL_TIMEOUT_S for {tqdm_desc or 'Eval'}") from te
        except KeyboardInterrupt:
            _dbg("KeyboardInterrupt: terminating pool … (dist)")
            pool.terminate()
            raise
        except Exception as e:
            _dbg(f"pool.imap_unordered(dist) raised: {type(e).__name__}: {e}")
            pool.terminate()
            raise
    if pbar is not None:
        pbar.close()
    _dbg(f"aggregating {len(results)} dist results")
    total_games = 0
    total_wins = 0
    total_diff_sum = 0.0
    agg = {0: {}, 1: {}}
    for diff_sum_i, bd_sum_i, wins_i, games_i in results:
        total_games += int(games_i)
        total_wins += int(wins_i)
        total_diff_sum += float(diff_sum_i)
        for t in [0, 1]:
            for k, v in bd_sum_i[t].items():
                agg[t][k] = agg[t].get(k, 0.0) + float(v)
    if total_games <= 0:
        return 0.0, {0: {}, 1: {}}
    bd_avg = {0: {}, 1: {}}
    for t in [0, 1]:
        for k, v in agg[t].items():
            bd_avg[t][k] = (float(v) / float(total_games))
    wr = float(total_wins) / float(total_games)
    bd_avg['meta'] = {'win_rate': float(wr)}
    diff_avg = float(total_diff_sum) / float(total_games)
    return float(diff_avg), bd_avg

def league_eval_and_update(league_dir='checkpoints/league', games=20, target_points=11):
    """Esegue sfide tra ultimi due checkpoint registrati e aggiorna Elo nel league in base alla differenza media di punti (reward)."""
    league = League(base_dir=league_dir)
    if len(league.history) < 2:
        return
    a, b = league.history[-2], league.history[-1]
    # Usa serie di partite per stimare la differenza media di punti di A contro B
    diff, bd = evaluate_pair_actors(a, b, games=games, k_history=12, mcts=None)
    # Aggiorna Elo usando la differenza media di punti -> mapping lineare a score
    league.update_elo_from_diff(a, b, diff)
    return league.elo
