import torch
from tqdm import tqdm
from typing import Any, Dict, List, Callable, Optional, Tuple
import os
import time
import sys
import multiprocessing as mp
import platform
import queue
import random
import numpy as np

# Ensure project root is on sys.path when running as script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import ScoponeEnvMA
from algorithms.ppo_ac import ActionConditionedPPO
from utils.device import get_compute_device
 
from selfplay.league import League
from models.action_conditioned import ActionConditionedActor
from utils.compile import maybe_compile_module
from utils.seed import set_global_seeds, resolve_seed, temporary_seed
from evaluation.eval import evaluate_pair_actors

import torch.optim as optim

device = get_compute_device()
# Global perf flags
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Profiling controls (single boolean -> full detail when enabled)
_PAR_TIMING = (os.environ.get('SCOPONE_PROFILE', '0') != '0')
_PAR_DEBUG = (os.environ.get('SCOPONE_PAR_DEBUG', '0') in ['1','true','yes','on'])
_PPO_DEBUG = (os.environ.get('SCOPONE_PPO_DEBUG', '0').strip().lower() in ['1', 'true', 'yes', 'on'])

# Reuse cached tensors to limit per-step allocations inside workers
_SEAT_VEC_CACHE: Dict[int, torch.Tensor] = {}
_BELIEF_ZERO = torch.zeros(120, dtype=torch.float32)
_OTHERS_HANDS_ZERO = torch.zeros((3, 40), dtype=torch.float32)
_EMPTY_LEGAL = torch.zeros((0, 80), dtype=torch.float32)

_SERIAL_RNG_STATE: Dict[int, Dict[str, Any]] = {}


def _serial_seed_enter(seed: Optional[int]) -> Optional[Dict[str, Any]]:
    if seed is None:
        return None
    try:
        seed_key = int(seed)
    except Exception:
        return None
    if seed_key < 0:
        return None
    py_outer = random.getstate()
    try:
        np_outer = np.random.get_state()
    except AttributeError:
        np_outer = None
    torch_outer = torch.get_rng_state()
    cuda_outer = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    stored = _SERIAL_RNG_STATE.get(seed_key)
    if stored is None:
        set_global_seeds(seed_key)
    else:
        py_state = stored.get('py')
        if py_state is not None:
            random.setstate(py_state)
        np_state = stored.get('np')
        if np_state is not None:
            np.random.set_state(np_state)  # type: ignore[arg-type]
        torch_state = stored.get('torch')
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        cuda_state = stored.get('cuda')
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]

    return {
        'seed': seed_key,
        'py_outer': py_outer,
        'np_outer': np_outer,
        'torch_outer': torch_outer,
        'cuda_outer': cuda_outer,
    }


def _serial_seed_exit(token: Optional[Dict[str, Any]]) -> None:
    if token is None:
        return
    seed_key = int(token['seed'])
    try:
        np_new = np.random.get_state()
    except AttributeError:
        np_new = None
    _SERIAL_RNG_STATE[seed_key] = {
        'py': random.getstate(),
        'np': np_new,
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    py_outer = token.get('py_outer')
    if py_outer is not None:
        random.setstate(py_outer)
    np_outer = token.get('np_outer')
    if np_outer is not None:
        np.random.set_state(np_outer)  # type: ignore[arg-type]
    torch_outer = token.get('torch_outer')
    if torch_outer is not None:
        torch.set_rng_state(torch_outer)
    cuda_outer = token.get('cuda_outer')
    if cuda_outer is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_outer)  # type: ignore[arg-type]

def _dbg(msg: str) -> None:
    if _PAR_DEBUG:
        try:
            tqdm.write(str(msg))
        except Exception:
            print(str(msg), flush=True)


def _flatten_cpu(tensor: Optional[torch.Tensor], dtype: Optional[torch.dtype] = torch.float32) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.numel() == 0:
        return tensor.detach().to('cpu', dtype=dtype) if dtype is not None else tensor.detach().to('cpu')
    out = tensor.detach().to('cpu')
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype=dtype)
    return out.reshape(-1)


def _tensor_basic_stats(tensor: Optional[torch.Tensor]) -> Dict[str, float]:
    if tensor is None:
        return {'count': 0}
    numel = int(tensor.numel())
    if numel == 0:
        return {'count': 0}
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item()) if numel > 1 else 0.0
    min_v = float(tensor.min().item())
    max_v = float(tensor.max().item())
    abs_mean = float(tensor.abs().mean().item())
    return {
        'count': numel,
        'mean': mean,
        'std': std,
        'min': min_v,
        'max': max_v,
        'abs_mean': abs_mean,
    }


def _format_stats(name: str, stats: Dict[str, float]) -> str:
    if stats.get('count', 0) == 0:
        return f"{name}=c0"
    return (
        f"{name}=c{stats['count']} μ{stats['mean']:.4f} σ{stats['std']:.4f} "
        f"|μ|{stats['abs_mean']:.4f} min{stats['min']:.4f} max{stats['max']:.4f}"
    )


def _maybe_log_ppo_batch(
    label: str,
    rew: Optional[torch.Tensor],
    ret: Optional[torch.Tensor],
    adv: Optional[torch.Tensor],
    val: Optional[torch.Tensor],
    next_val: Optional[torch.Tensor],
    done_mask: Optional[torch.Tensor],
    *,
    old_logp: Optional[torch.Tensor] = None,
    seat_tensor: Optional[torch.Tensor] = None,
    episode_lengths: Optional[List[int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not _PPO_DEBUG:
        return
    try:
        rew_cpu = _flatten_cpu(rew)
        ret_cpu = _flatten_cpu(ret)
        adv_cpu = _flatten_cpu(adv)
        val_cpu = _flatten_cpu(val)
        next_cpu = _flatten_cpu(next_val)
        logp_cpu = _flatten_cpu(old_logp) if old_logp is not None else None
        done_cpu = _flatten_cpu(done_mask, dtype=torch.float32)
        seat_cpu = _flatten_cpu(seat_tensor, dtype=torch.float32) if seat_tensor is not None else None

        rew_stats = _tensor_basic_stats(rew_cpu)
        ret_stats = _tensor_basic_stats(ret_cpu)
        adv_stats = _tensor_basic_stats(adv_cpu)
        val_stats = _tensor_basic_stats(val_cpu)
        next_stats = _tensor_basic_stats(next_cpu)
        logp_stats = _tensor_basic_stats(logp_cpu) if logp_cpu is not None else {'count': 0}

        batch_size = ret_stats.get('count', 0)
        msg_parts = [f"[ppo-debug {label}] B={batch_size}"]
        msg_parts.append(_format_stats('rew', rew_stats))
        msg_parts.append(_format_stats('ret', ret_stats))
        msg_parts.append(_format_stats('val', val_stats))
        msg_parts.append(_format_stats('adv', adv_stats))
        msg_parts.append(_format_stats('next', next_stats))
        if logp_stats.get('count', 0) > 0:
            msg_parts.append(_format_stats('old_logp', logp_stats))

        if ret_cpu is not None and val_cpu is not None and ret_stats.get('count', 0) == val_stats.get('count', 0) and ret_stats.get('count', 0) > 0:
            diff = ret_cpu - val_cpu
            mse = float((diff.pow(2).mean()).item())
            l1 = float(diff.abs().mean().item())
            msg_parts.append(f"ret-v mse={mse:.4f} l1={l1:.4f}")

        if adv_cpu is not None and adv_stats.get('count', 0) > 0:
            adv_abs_max = float(adv_cpu.abs().max().item())
            msg_parts.append(f"adv|max|={adv_abs_max:.4f}")

        if done_cpu is not None and done_cpu.numel() > 0:
            done_sum = float(done_cpu.sum().item())
            done_frac = done_sum / float(done_cpu.numel())
            msg_parts.append(f"done_sum={done_sum:.0f} done_frac={done_frac:.4f}")

        if seat_tensor is not None and seat_cpu is not None and seat_cpu.numel() > 0 and ret_cpu is not None and rew_cpu is not None:
            seat_cpu_mat = seat_cpu.view(-1, 6)
            seat_idx = torch.argmax(seat_cpu_mat[:, :4], dim=1)
            seat_counts = tuple(int((seat_idx == i).sum().item()) for i in range(4))
            team0_mask = seat_cpu_mat[:, 4] > 0.5
            team1_mask = seat_cpu_mat[:, 5] > 0.5
            team_counts = (int(team0_mask.sum().item()), int(team1_mask.sum().item()))
            team_reward = (
                float(rew_cpu[team0_mask].sum().item()) if team_counts[0] > 0 else 0.0,
                float(rew_cpu[team1_mask].sum().item()) if team_counts[1] > 0 else 0.0,
            )
            msg_parts.append(f"seat_counts={seat_counts} team_counts={team_counts} team_rew={team_reward}")

        if episode_lengths:
            ep_count = len(episode_lengths)
            ep_min = min(episode_lengths)
            ep_max = max(episode_lengths)
            ep_mean = sum(episode_lengths) / float(ep_count)
            msg_parts.append(f"ep_len(c={ep_count} min={ep_min} max={ep_max} mean={ep_mean:.2f})")

        if extra:
            extra_str = ' '.join([f"{k}={v}" for k, v in extra.items()])
            if extra_str:
                msg_parts.append(extra_str)

        try:
            tqdm.write(' '.join(msg_parts))
        except Exception:
            print(' '.join(msg_parts), flush=True)
    except Exception as exc:
        try:
            tqdm.write(f"[ppo-debug {label}] logging failed: {exc}")
        except Exception:
            print(f"[ppo-debug {label}] logging failed: {exc}", flush=True)


def _normalize_adv_tensor(label: str, adv_tensor: torch.Tensor) -> torch.Tensor:
    if adv_tensor.numel() == 0:
        return adv_tensor
    dbg_before = _tensor_basic_stats(_flatten_cpu(adv_tensor)) if _PPO_DEBUG else None
    mean = adv_tensor.mean()
    std = adv_tensor.std(unbiased=False)
    std = torch.clamp(std, min=1e-8)
    normalized = (adv_tensor - mean) / std
    if _PPO_DEBUG and dbg_before is not None:
        dbg_after = _tensor_basic_stats(_flatten_cpu(normalized))
        mean_val = float(mean.item()) if torch.is_tensor(mean) else float(mean)
        std_val = float(std.item()) if torch.is_tensor(std) else float(std)
        msg = (
            f"[ppo-debug adv-norm {label}] {_format_stats('adv_pre', dbg_before)} "
            f"{_format_stats('adv_post', dbg_after)} mean_before={mean_val:.4f} std_before={std_val:.4f}"
        )
        try:
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)
    return normalized

# Sub-profilers (used whenever _PAR_TIMING is True)
_BATCHSEL_PROF = {
    'count': 0,
    't_stack': 0.0,
    't_leg_stack': 0.0,
    't_state_proj': 0.0,
    't_action_enc': 0.0,
    't_mask': 0.0,
    't_score': 0.0,
    't_softmax_sample': 0.0,
    't_total': 0.0,
    'sum_B': 0,
    'sum_M': 0,
    'cnt_hist': {},
}

# get_reqs sub-profiler
_GETREQS_PROF = {
    'batches': 0,
    'total_reqs': 0,
    't_first_block': 0.0,
    't_nowait_drain': 0.0,
    't_topup_block': 0.0,
    't_post_drain': 0.0,
    'cnt_first_ok': 0,
    'cnt_first_to': 0,
    'cnt_nowait_ok': 0,
    'cnt_nowait_empty': 0,
    'cnt_topup_ok': 0,
    'cnt_topup_to': 0,
    'bs_hist': {},
}

# One-time run flags
_HORIZON_ADJUST_LOGGED = False


def _seat_vec_for(cp: int) -> torch.Tensor:
    v = _SEAT_VEC_CACHE.get(cp)
    if v is None:
        base = torch.zeros(6, dtype=torch.float32)
        base[cp] = 1.0
        base[4] = 1.0 if cp in [0, 2] else 0.0
        base[5] = 1.0 if cp in [1, 3] else 0.0
        _SEAT_VEC_CACHE[cp] = base
        v = base
    return v


def _to_cpu_float32(tensor_like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(tensor_like):
        if tensor_like.device.type == 'cpu' and tensor_like.dtype == torch.float32:
            return tensor_like.detach()
        return tensor_like.detach().to('cpu', dtype=torch.float32)
    return torch.as_tensor(tensor_like, dtype=torch.float32)


def _env_worker(worker_id: int,
                cfg: Dict,
                request_q: mp.Queue,
                action_q: mp.Queue,
                episode_q: mp.Queue):
    _WDBG = (os.environ.get('SCOPONE_PAR_DEBUG', '0') in ['1','true','yes','on'])
    def _wdbg(msg: str) -> None:
        if _WDBG:
            try:
                tqdm.write(f"[worker {int(worker_id)} pid={os.getpid()}] {msg}")
            except Exception:
                print(f"[worker {int(worker_id)} pid={os.getpid()}] {msg}", flush=True)
    # Limit CPU threads per worker to reduce contention on the host
    wt = int(os.environ.get('SCOPONE_WORKER_THREADS', '1'))
    os.environ['OMP_NUM_THREADS'] = str(wt)
    os.environ['MKL_NUM_THREADS'] = str(wt)
    torch.set_num_threads(wt)
    torch.set_num_interop_threads(1)
    # Ensure different RNG streams per worker for robustness
    set_global_seeds(int(cfg.get('seed', 0)) + int(worker_id))
    env = ScoponeEnvMA(rules=cfg.get('rules', {'shape_scopa': False}),
                       k_history=int(cfg.get('k_history', 39)))
    episodes_per_env = int(cfg.get('episodes_per_env', 1))
    send_legals = bool(cfg.get('send_legals', True))
    use_mcts = bool(cfg.get('use_mcts', False))
    train_both_teams = bool(cfg.get('train_both_teams', False))
    main_seats = cfg.get('main_seats', None)
    # MCTS config
    mcts_sims = int(cfg.get('mcts_sims', 128))
    mcts_dets = int(cfg.get('mcts_dets', 4))
    mcts_c_puct = float(cfg.get('mcts_c_puct', 1.0))
    mcts_root_temp = float(cfg.get('mcts_root_temp', 0.0))
    mcts_prior_smooth_eps = float(cfg.get('mcts_prior_smooth_eps', 0.0))
    mcts_dirichlet_alpha = float(cfg.get('mcts_dirichlet_alpha', 0.25))
    mcts_dirichlet_eps = float(cfg.get('mcts_dirichlet_eps', 0.25))
    mcts_progress_start = float(cfg.get('mcts_progress_start', 0.25))
    mcts_progress_full = float(cfg.get('mcts_progress_full', 0.75))
    mcts_min_sims = int(cfg.get('mcts_min_sims', 0))
    mcts_train_factor = float(cfg.get('mcts_train_factor', 1.0))
    rpc_timeout_s = float(os.environ.get('SCOPONE_RPC_TIMEOUT_S', '30'))
    episode_put_timeout_s = float(os.environ.get('SCOPONE_EP_PUT_TIMEOUT_S', '15'))
    belief_aux_coef = float(os.environ.get('BELIEF_AUX_COEF', '0.1'))

    # Torch profiler in worker: enabled by default when torch profiler mode is active in the main process
    _tp_active = (os.environ.get('SCOPONE_TORCH_PROF', '0') == '1')
    _worker_prof = None
    from torch.profiler import record_function as _record_function  # type: ignore
    if _tp_active:
        from torch.profiler import profile as _tp_profile, ProfilerActivity as _PA, tensorboard_trace_handler as _tb_handler_factory  # type: ignore
        _tb_dir_env = os.environ.get('SCOPONE_TORCH_TB_DIR', '').strip()
        _tb_write_fn = (_tb_handler_factory(_tb_dir_env, worker_name=f"env-{int(worker_id)}-pid{os.getpid()}") if _tb_dir_env else None)
        _worker_prof = _tp_profile(
            activities=[_PA.CPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
        )
        _worker_prof.start()

    t_env_reset = 0.0; t_get_obs = 0.0; t_get_legals = 0.0; t_mcts = 0.0; t_rpc = 0.0; t_step = 0.0; t_pack = 0.0
    t0_glob = time.time()
    _wdbg(f"start (episodes_per_env={int(episodes_per_env)})")
    for ep in range(episodes_per_env):
        t0 = time.time(); env.reset(); t_env_reset += (time.time() - t0) if _PAR_TIMING else 0.0
        _wdbg(f"ep {ep} reset")
        obs_list, next_obs_list = [], []
        act_list = []
        rew_list, done_list = [], []
        seat_team_list = []
        belief_sum_list = []
        legals_list, legals_offset, legals_count = [], [], []
        chosen_index_list = []
        mcts_policy_list = []
        mcts_weight_list = []
        others_hands_list = []

        done = False
        info = {}
        step_idx = 0
        while not done:
            t0 = time.time();
            with _record_function('env._get_observation'):
                obs = env._get_observation(env.current_player)
            t_get_obs += (time.time() - t0) if _PAR_TIMING else 0.0
            if step_idx < 3:
                _wdbg(f"ep {ep} step {step_idx}: got obs (cp={env.current_player})")
            t0 = time.time();
            with _record_function('env.get_valid_actions'):
                legal = env.get_valid_actions()
            t_get_legals += (time.time() - t0) if _PAR_TIMING else 0.0
            if step_idx < 3:
                _wdbg(f"ep {ep} step {step_idx}: legals={int(len(legal))}")
            if torch.is_tensor(legal) and (legal.size(0) == 0):
                raise RuntimeError(f"collect_trajectory_parallel: worker {worker_id} got 0 legal actions (player={env.current_player})")
            # Avoid ambiguous truth-value on tensors; treat empty action sets explicitly
            is_empty = (int(legal.numel()) == 0)
            if is_empty:
                raise RuntimeError(f"worker {worker_id}: no legal actions at episode start (player={env.current_player})")
            cp = env.current_player
            seat_vec = _seat_vec_for(cp)
            seat_cpu = seat_vec
            obs_cpu = _to_cpu_float32(obs)
            if torch.is_tensor(legal):
                legal_cpu_tensor = legal.detach().to('cpu', dtype=torch.float32)
                legal_cpu_entries = list(legal_cpu_tensor.unbind(0))
            else:
                legal_cpu_entries = [_to_cpu_float32(x) for x in legal]
                legal_cpu_tensor = torch.stack(legal_cpu_entries, dim=0) if len(legal_cpu_entries) > 0 else _EMPTY_LEGAL
            legal_count = len(legal_cpu_entries)
            is_main = True if train_both_teams else ((main_seats is None and cp in [0, 2]) or (main_seats is not None and cp in main_seats))
            store_sample = bool(train_both_teams or is_main)
            policy_entries: List[float] = []
            policy_weight = 0.0
            bsum_tensor = _BELIEF_ZERO
            # Only attempt MCTS in worker if explicitly enabled and mcts_sims > 0
            _both_sides = str(os.environ.get('SCOPONE_MCTS_BOTH_SIDES', '1')).strip().lower() in ['1','true','yes','on']
            if (is_main or _both_sides) and use_mcts and len(legal) > 0 and int(mcts_sims) > 0:
                # Build helper RPCs to master for batched scoring
                if send_legals:
                    leg_serial = [x.tolist() for x in legal_cpu_entries]
                else:
                    leg_serial = []
                def policy_fn_mcts(_obs, _legals):
                    nonlocal t_rpc
                    # Invia tensori CPU direttamente (no conversione a liste) per ridurre overhead IPC
                    o_cpu = (_obs.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(_obs) else torch.as_tensor(_obs, dtype=torch.float32))
                    leg_cpu = torch.stack([ (y.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)) for y in _legals ], dim=0)
                    t1 = time.time()
                    if step_idx < 3:
                        _wdbg(f"ep {ep} step {step_idx}: MCTS policy -> send, waiting priors")
                    request_q.put({
                        'type': 'score_policy',
                        'wid': worker_id,
                        'obs': o_cpu,
                        'legals': leg_cpu,
                        'seat': seat_cpu,
                    })
                    try:
                        resp = action_q.get(timeout=rpc_timeout_s)
                    except queue.Empty as e:
                        raise TimeoutError(f"worker {worker_id}: Timeout waiting for score_policy priors") from e
                    t_rpc_local = (time.time() - t1)
                    if _PAR_TIMING:
                        t_rpc = t_rpc + t_rpc_local
                    if step_idx < 3:
                        _wdbg(f"ep {ep} step {step_idx}: MCTS policy <- got priors")
                    pri = resp.get('priors', None)
                    import numpy as _np
                    if pri is None or (len(pri) != len(_legals)):
                        raise RuntimeError(f"worker {worker_id}: invalid priors from master (priors={type(pri)}, expected_len={len(_legals)})")
                    return _np.asarray(pri, dtype=_np.float32)
                def value_fn_mcts(_obs, _env):
                    s_vec = _seat_vec_for(_env.current_player)
                    o_cpu = (_obs.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(_obs) else torch.as_tensor(_obs, dtype=torch.float32))
                    if step_idx < 3:
                        _wdbg(f"ep {ep} step {step_idx}: MCTS value -> send, waiting value")
                    request_q.put({
                        'type': 'score_value',
                        'wid': worker_id,
                        'obs': o_cpu,
                        'seat': s_vec,
                    })
                    try:
                        resp = action_q.get(timeout=rpc_timeout_s)
                    except queue.Empty as e:
                        raise TimeoutError(f"worker {worker_id}: Timeout waiting for score_value") from e
                    if step_idx < 3:
                        _wdbg(f"ep {ep} step {step_idx}: MCTS value <- got value")
                    return float(resp.get('value', 0.0))
                def belief_sampler_neural(_env):
                    o_cur = _env._get_observation(_env.current_player)
                    s_vec = _seat_vec_for(_env.current_player)
                    o_cpu = (o_cur.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(o_cur) else torch.as_tensor(o_cur, dtype=torch.float32))
                    if step_idx < 3:
                        _wdbg(f"ep {ep} step {step_idx}: MCTS belief -> send, waiting probs")
                    request_q.put({
                        'type': 'score_belief',
                        'wid': worker_id,
                        'obs': o_cpu,
                        'seat': s_vec,
                    })
                    try:
                        resp = action_q.get(timeout=rpc_timeout_s)
                    except queue.Empty as e:
                        raise TimeoutError(f"worker {worker_id}: Timeout waiting for score_belief") from e
                    if step_idx < 3:
                        _wdbg(f"ep {ep} step {step_idx}: MCTS belief <- got probs")
                    probs_flat = resp.get('belief_probs', None)
                    if probs_flat is None:
                        return None
                    import numpy as _np
                    probs = _np.asarray(probs_flat, dtype=_np.float32).reshape(3, 40)
                    # visible mask from obs on worker side
                    if torch.is_tensor(o_cur):
                        o_t = o_cur.detach().to('cpu', dtype=torch.float32).unsqueeze(0)
                    else:
                        o_t = torch.as_tensor(o_cur, dtype=torch.float32).unsqueeze(0)
                    hand_table = o_t[:, :83]
                    hand_mask = hand_table[:, :40] > 0.5
                    table_mask = hand_table[:, 43:83] > 0.5
                    captured = o_t[:, 83:165]
                    cap0_mask = captured[:, :40] > 0.5
                    cap1_mask = captured[:, 40:80] > 0.5
                    vis = (hand_mask | table_mask | cap0_mask | cap1_mask).squeeze(0).numpy().astype(bool)
                    unknown_ids = [cid for cid in range(40) if not vis[cid]]
                    others = [(_env.current_player + 1) % 4, (_env.current_player + 2) % 4, (_env.current_player + 3) % 4]
                    counts = {pid: len(_env.game_state['hands'][pid]) for pid in others}
                    caps = [int(counts.get(pid, 0)) for pid in others]
                    n = len(unknown_ids)
                    if sum(caps) != n:
                        caps[2] = max(0, n - caps[0] - caps[1])
                        if sum(caps) != n:
                            base = n // 3
                            rem = n - 3 * base
                            caps = [base, base, base]
                            for i in range(rem):
                                caps[i] += 1
                    import numpy as _np
                    noise_scale = float(os.environ.get('DET_NOISE', '0.0'))
                    costs = []
                    for cid in unknown_ids:
                        pc = probs[:, cid]
                        ps = pc / max(1e-12, pc.sum())
                        c = [-_np.log(max(1e-12, ps[i])) for i in range(3)]
                        if noise_scale > 0:
                            u = _np.random.uniform(1e-9, 1.0-1e-9, size=3)
                            g = -_np.log(-_np.log(u)) * noise_scale
                            c = [c[i] + float(g[i]) for i in range(3)]
                        costs.append(c)
                    INF = 1e12
                    cap0, cap1, cap2 = caps
                    dp = [[[INF]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                    bk = [[[-1]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                    dp[0][0][0] = 0.0
                    for t in range(n):
                        c0, c1, c2 = costs[t]
                        for a in range(0, min(t, cap0)+1):
                            for b in range(0, min(t-a, cap1)+1):
                                cur = dp[t][a][b]
                                if cur >= INF:
                                    continue
                                if a+1 <= cap0 and dp[t+1][a+1][b] > cur + c0:
                                    dp[t+1][a+1][b] = cur + c0
                                    bk[t+1][a+1][b] = 0
                                if b+1 <= cap1 and dp[t+1][a][b+1] > cur + c1:
                                    dp[t+1][a][b+1] = cur + c1
                                    bk[t+1][a][b+1] = 1
                                assigned2 = t - a - b
                                if assigned2 + 1 <= cap2 and dp[t+1][a][b] > cur + c2:
                                    dp[t+1][a][b] = cur + c2
                                    bk[t+1][a][b] = 2
                    if dp[n][cap0][cap1] >= INF:
                        return None
                    det = {pid: [] for pid in others}
                    a, b = cap0, cap1
                    for t in range(n, 0, -1):
                        choice = bk[t][a][b]
                        cid = unknown_ids[t-1]
                        if choice == 0:
                            det[others[0]].append(cid)
                            a -= 1
                        elif choice == 1:
                            det[others[1]].append(cid)
                            b -= 1
                        else:
                            det[others[2]].append(cid)
                    return det
                # Progress-based scaling (uniform with single-env) — optionally gated by env
                progress = float(min(1.0, max(0.0, len(env.game_state.get('history', [])) / 40.0)))
                denom = max(1e-6, (mcts_progress_full - mcts_progress_start))
                alpha = min(1.0, max(0.0, (progress - mcts_progress_start) / denom))
                scaling_on = (str(os.environ.get('SCOPONE_MCTS_SCALING', '1')).strip().lower() in ['1','true','yes','on'])
                # Permetti 0 simulazioni se mcts_min_sims==0
                base_min = int(mcts_min_sims) if (mcts_min_sims is not None and int(mcts_min_sims) >= 0) else 0
                import math
                if mcts_train_factor is not None and float(mcts_train_factor) <= 0.0:
                    sims_scaled = 0
                else:
                    if scaling_on:
                        sims_base = math.ceil(mcts_sims * (0.25 + 0.75 * alpha))
                    else:
                        sims_base = math.ceil(mcts_sims * float(mcts_train_factor if mcts_train_factor is not None else 1.0))
                    if mcts_train_factor is not None and scaling_on:
                        sims_base = math.ceil(sims_base * float(mcts_train_factor))
                    sims_scaled = int(max(base_min, sims_base))
                    if int(mcts_sims) > 0 and sims_scaled <= 0:
                        sims_scaled = 1
                # Root temperature dynamic only when scaling is enabled (or explicitly overridden)
                root_temp_dyn = (float(mcts_root_temp) if (not scaling_on or float(mcts_root_temp) > 0)
                                 else float(max(0.0, 1.0 - alpha)))

                if sims_scaled <= 0:
                    # Sims ended up <= 0 despite mcts_sims > 0 — log, optionally raise for debug, then fall back to master step
                    if os.environ.get('SCOPONE_RAISE_ON_INVALID_SIMS', '0') == '1':
                        raise RuntimeError(f"invalid sims_scaled: sims_scaled={sims_scaled} mcts_sims={mcts_sims} progress_start={mcts_progress_start} progress_full={mcts_progress_full} history_len={len(env.game_state.get('history', [])) if isinstance(env.game_state.get('history', []), list) else 'n/a'} alpha={alpha}")
                    leg_serial = legal_cpu_tensor if send_legals else _EMPTY_LEGAL
                    request_q.put({
                        'type': 'step',
                        'wid': worker_id,
                        'obs': obs_cpu,
                        'legals': leg_serial,
                        'seat': seat_cpu,
                    })
                    try:
                        resp = action_q.get(timeout=rpc_timeout_s)
                        idx = int(resp.get('idx', 0))
                    except queue.Empty as e:
                        raise TimeoutError('Timeout waiting for step index for MCTS (invalid_sims_scaled)') from e
                    idx = max(0, min(idx, len(legal) - 1))
                    act_t = legal[idx] if torch.is_tensor(legal[idx]) else torch.as_tensor(legal[idx], dtype=torch.float32)
                    with _record_function('env.step'):
                        next_obs, rew, done, info = env.step(act_t)
                    # No distillation target
                    policy_entries = [0.0] * legal_count
                    policy_weight = 0.0
                else:
                    from algorithms.is_mcts import run_is_mcts
                    # Dynamic defaults for smoothing and root Dirichlet based on context
                    t1 = time.time(); priors_probe = policy_fn_mcts(obs, legal); t_mcts += (time.time() - t1) if _PAR_TIMING else 0.0
                    pri_t = (priors_probe if torch.is_tensor(priors_probe) else torch.as_tensor(priors_probe, dtype=torch.float32))
                    peak = float(pri_t.max().item()) if pri_t.numel() > 0 else (1.0 / max(1, len(legal)))
                    A = int(len(legal))
                    sims_fac = 1.0 if sims_scaled < 128 else (0.5 if sims_scaled < 256 else 0.25)
                    peak_fac = min(1.0, max(0.0, (peak - 0.5) / 0.4))
                    prior_eps_eff = 0.1 * sims_fac * peak_fac * (1.0 - alpha)
                    prior_eps_eff = float(max(0.0, min(0.15, prior_eps_eff)))
                    if A <= 3:
                        dir_eps_eff = 0.0
                    else:
                        a_fac = min(1.0, max(0.0, (A - 3) / 10.0))
                        sim_att = (0.7 if sims_scaled >= 256 else 1.0)
                        prog_att = (0.7 if alpha > 0.7 else 1.0)
                        dir_eps_eff = 0.25 * a_fac * sim_att * prog_att
                        dir_eps_eff = float(max(0.0, min(0.3, dir_eps_eff)))

                    mcts_action, mcts_visits = run_is_mcts(env,
                        policy_fn=policy_fn_mcts,
                        value_fn=value_fn_mcts,
                        num_simulations=int(sims_scaled),
                        c_puct=float(mcts_c_puct),
                        belief=None,
                        num_determinization=int(mcts_dets),
                        root_temperature=root_temp_dyn,
                        prior_smooth_eps=prior_eps_eff,
                        robust_child=True,
                        root_dirichlet_alpha=float(mcts_dirichlet_alpha),
                        root_dirichlet_eps=dir_eps_eff,
                        return_stats=True,
                        belief_sampler=belief_sampler_neural)
                    chosen_act = mcts_action if torch.is_tensor(mcts_action) else torch.as_tensor(mcts_action, dtype=torch.float32)
                    def _act_key(x_t: torch.Tensor):
                        xt = x_t if torch.is_tensor(x_t) else torch.as_tensor(x_t, dtype=torch.float32)
                        return tuple(torch.nonzero(xt > 0.5, as_tuple=False).flatten().tolist())
                    key_target = _act_key(chosen_act)
                    idx = 0
                    for i_a, a in enumerate(legal):
                        if _act_key(a) == key_target:
                            idx = int(i_a)
                            break
                    act_t = legal[idx] if torch.is_tensor(legal[idx]) else torch.as_tensor(legal[idx], dtype=torch.float32)
                    t1 = time.time();
                    with _record_function('env.step'):
                        next_obs, rew, done, info = env.step(act_t)
                    t_step += (time.time() - t1) if _PAR_TIMING else 0.0
                    # Distillation targets
                    mcts_probs = torch.as_tensor(mcts_visits, dtype=torch.float32)
                    ssum = float(mcts_probs.sum().item())
                    if ssum > 0:
                        mcts_probs = mcts_probs / ssum
                    policy_entries = (mcts_probs.tolist() if hasattr(mcts_probs, 'tolist') else list(mcts_probs))
                    policy_weight = 1.0
            else:
                # Request action selection from master (GPU)
                leg_serial = legal_cpu_tensor if send_legals else _EMPTY_LEGAL
                if step_idx < 3:
                    _wdbg(f"ep {ep} step {step_idx}: STEP -> send, waiting idx (A={len(legal)})")
                t1 = time.time(); request_q.put({
                    'type': 'step',
                    'wid': worker_id,
                    'obs': obs_cpu,
                    'legals': leg_serial,
                    'seat': seat_cpu,
                })
                try:
                    resp = action_q.get(timeout=rpc_timeout_s)
                    idx = int(resp.get('idx', 0))
                except queue.Empty as e:
                    raise TimeoutError(f"worker {worker_id}: Timeout waiting for step index (main path)") from e
                t_rpc += (time.time() - t1) if _PAR_TIMING else 0.0
                if step_idx < 3:
                    _wdbg(f"ep {ep} step {step_idx}: STEP <- got idx={int(idx)}")
                idx = max(0, min(idx, len(legal) - 1))
                act_t = legal[idx] if torch.is_tensor(legal[idx]) else torch.as_tensor(legal[idx], dtype=torch.float32)
                t1 = time.time();
                with _record_function('env.step'):
                    next_obs, rew, done, info = env.step(act_t)
                if step_idx < 3:
                    _wdbg(f"ep {ep} step {step_idx}: env.step done={bool(done)} rew={float(rew)}")
                step_idx += 1
                t_step += (time.time() - t1) if _PAR_TIMING else 0.0
                # No distillation target
                policy_entries = [0.0] * legal_count
                policy_weight = 0.0

            next_obs_cpu = _to_cpu_float32(next_obs)
            act_cpu = _to_cpu_float32(act_t)

            if store_sample:
                obs_list.append(obs_cpu)
                next_obs_list.append(next_obs_cpu)
                act_list.append(act_cpu)
                rew_list.append(float(rew))
                done_list.append(bool(done))
                seat_team_list.append(seat_cpu)
                belief_sum_list.append(bsum_tensor)
                legals_offset.append(len(legals_list))
                legals_count.append(legal_count)
                if send_legals:
                    legals_list.extend(legal_cpu_entries)
                else:
                    from utils.fallback import notify_fallback
                    notify_fallback('trainer.collect_trajectory.legals_missing_for_store')
                chosen_index_list.append(int(idx))
                entries = policy_entries if len(policy_entries) > 0 else ([0.0] * legal_count)
                mcts_policy_list.extend(entries)
                mcts_weight_list.append(policy_weight if len(policy_entries) > 0 else 0.0)
                # Others' hands supervision target (3x40) — skip if BELIEF_AUX_COEF <= 0
                if belief_aux_coef <= 0.0:
                    others_hands_list.append(_OTHERS_HANDS_ZERO)
                else:
                    hands = env.game_state.get('hands', None)
                    if hands is not None:
                        others = [ (cp + 1) % 4, (cp + 2) % 4, (cp + 3) % 4 ]
                        target = torch.zeros((3,40), dtype=torch.float32)
                        for i,pid in enumerate(others):
                            for c in hands[pid]:
                                if isinstance(c, int):
                                    cid = c
                                else:
                                    r, s = c
                                    suit_to_int = {'denari': 0, 'coppe': 1, 'spade': 2, 'bastoni': 3}
                                    cid = int((r - 1) * 4 + suit_to_int[s])
                                target[i, int(cid)] = 1.0
                        others_hands_list.append(target)
                    else:
                        others_hands_list.append(_OTHERS_HANDS_ZERO)
            else:
                # Non-main seats skipped when training only main seats; ensure last stored sample closes episode
                if done and len(done_list) > 0:
                    done_list[-1] = True
                    if len(next_obs_list) > 0:
                        next_obs_list[-1] = next_obs_cpu

        # Ensure at least one step per episode to avoid empty payloads
        if len(obs_list) == 0:
            obs = env._get_observation(env.current_player)
            legal = env.get_valid_actions()
            is_empty = (int(legal.numel()) == 0)
            if not is_empty:
                cp = env.current_player
                seat_vec = _seat_vec_for(cp)
                seat_cpu = seat_vec
                obs_cpu = _to_cpu_float32(obs)
                if torch.is_tensor(legal):
                    legals_cpu_tensor = legal.detach().to('cpu', dtype=torch.float32)
                    legal_entries = list(legals_cpu_tensor.unbind(0))
                else:
                    legal_entries = [_to_cpu_float32(x) for x in legal]
                    legals_cpu_tensor = torch.stack(legal_entries, dim=0) if len(legal_entries) > 0 else _EMPTY_LEGAL
                leg_serial = legals_cpu_tensor if send_legals else _EMPTY_LEGAL
                request_q.put({
                    'type': 'step',
                    'wid': worker_id,
                    'obs': obs_cpu,
                    'legals': leg_serial,
                    'seat': seat_cpu,
                })
                try:
                    resp = action_q.get(timeout=rpc_timeout_s)
                    idx = int(resp.get('idx', 0))
                except queue.Empty as e:
                    raise TimeoutError(f"worker {worker_id}: Timeout waiting for step index (forced first step)") from e
                idx = max(0, min(idx, len(legal) - 1))
                act_t = legal[idx] if torch.is_tensor(legal[idx]) else torch.as_tensor(legal[idx], dtype=torch.float32)
                with _record_function('env.step'):
                    next_obs, rew, done, info = env.step(act_t)
                next_obs_cpu = _to_cpu_float32(next_obs)
                act_cpu = _to_cpu_float32(act_t)
                obs_list.append(obs_cpu)
                next_obs_list.append(next_obs_cpu)
                act_list.append(act_cpu)
                rew_list.append(float(rew))
                done_list.append(bool(done))
                seat_team_list.append(seat_cpu)
                belief_sum_list.append(_BELIEF_ZERO)
                legals_offset.append(len(legals_list))
                legals_count.append(len(legal))
                if send_legals:
                    legals_list.extend(legal_entries)
                chosen_index_list.append(int(idx))
                # default others_hands zero target for this forced step
                others_hands_list.append(_OTHERS_HANDS_ZERO)
        
        # Episode payload back to master using NumPy arrays (avoid Torch resource_sharer FDs entirely)
        import numpy as _np
        if len(obs_list) > 0:
            t1 = time.time(); obs_t = torch.stack(obs_list, dim=0).to('cpu', dtype=torch.float32).numpy(); t_pack += (time.time() - t1) if _PAR_TIMING else 0.0
        else:
            obs_t = _np.zeros((0, 1), dtype=_np.float32)
        if len(next_obs_list) > 0:
            t1 = time.time(); next_obs_t = torch.stack(next_obs_list, dim=0).to('cpu', dtype=torch.float32).numpy(); t_pack += (time.time() - t1) if _PAR_TIMING else 0.0
        else:
            next_obs_t = _np.zeros((0, 1), dtype=_np.float32)
        if len(act_list) > 0:
            t1 = time.time(); act_t = torch.stack(act_list, dim=0).to('cpu', dtype=torch.float32).numpy(); t_pack += (time.time() - t1) if _PAR_TIMING else 0.0
        else:
            act_t = _np.zeros((0, 80), dtype=_np.float32)
        if len(seat_team_list) > 0:
            t1 = time.time(); seat_t = torch.stack(seat_team_list, dim=0).to('cpu', dtype=torch.float32).numpy(); t_pack += (time.time() - t1) if _PAR_TIMING else 0.0
        else:
            seat_t = _np.zeros((0, 6), dtype=_np.float32)
        if len(belief_sum_list) > 0:
            t1 = time.time(); belief_t = torch.stack(belief_sum_list, dim=0).to('cpu', dtype=torch.float32).numpy(); t_pack += (time.time() - t1) if _PAR_TIMING else 0.0
        else:
            belief_t = _np.zeros((0, 120), dtype=_np.float32)
        if len(legals_list) > 0:
            t1 = time.time(); legals_t = torch.stack(legals_list, dim=0).to('cpu', dtype=torch.float32).numpy(); t_pack += (time.time() - t1) if _PAR_TIMING else 0.0
        else:
            legals_t = _np.zeros((0, 80), dtype=_np.float32)
        leg_off_t = _np.asarray(legals_offset, dtype=_np.int64)
        leg_cnt_t = _np.asarray(legals_count, dtype=_np.int64)
        chosen_idx_t = _np.asarray(chosen_index_list, dtype=_np.int64)
        mcts_policy_t = _np.asarray(mcts_policy_list, dtype=_np.float32) if len(mcts_policy_list) > 0 else _np.zeros((0,), dtype=_np.float32)
        mcts_weight_t = _np.asarray(mcts_weight_list, dtype=_np.float32) if len(mcts_weight_list) > 0 else _np.zeros((0,), dtype=_np.float32)
        if len(others_hands_list) > 0:
            t1 = time.time(); others_hands_t = torch.stack(others_hands_list, dim=0).to('cpu', dtype=torch.float32).numpy(); t_pack += (time.time() - t1) if _PAR_TIMING else 0.0
        else:
            others_hands_t = _np.zeros((0, 3, 40), dtype=_np.float32)

        # Non-blocking put with timeout to avoid deadlocks if the master is slow
        payload = {
            'wid': worker_id,
            'obs': obs_t,
            'next_obs': next_obs_t,
            'act': act_t,
            'rew': rew_list,
            'done': done_list,
            'seat': seat_t,
            'belief_summary': belief_t,
            'legals': legals_t,
            'leg_off': leg_off_t,
            'leg_cnt': leg_cnt_t,
            'chosen_idx': chosen_idx_t,
            'team_rewards': (info.get('team_rewards', [0.0, 0.0]) if isinstance(info, dict) else [0.0, 0.0]),
            'mcts_policy': mcts_policy_t,
            'mcts_weight': mcts_weight_t,
            'others_hands': others_hands_t,
        }
        try:
            if len(obs_list) == 0:
                raise RuntimeError(f"worker {worker_id}: episode finished with zero steps — invalid episode")
            _wdbg(f"ep {ep} putting episode payload (steps={len(obs_list)})")
            episode_q.put(payload, timeout=episode_put_timeout_s)
            _wdbg(f"ep {ep} payload put ok")
        except Exception as e:
            raise RuntimeError(f"worker {worker_id}: episode_q.put failed") from e
        # Do not advance worker profiler schedule per-episode to avoid Kineto stack issues
    # Stop and export worker torch profiler if active
    if _worker_prof is not None:
        _worker_prof.stop()
        from datetime import datetime as _dt  # local import to avoid changing global imports
        _dir = os.environ.get('SCOPONE_TORCH_PROF_DIR', os.path.abspath(os.path.join(_PROJECT_ROOT, 'profiles')))
        _run = os.environ.get('SCOPONE_TORCH_PROF_RUN', _dt.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(_dir, exist_ok=True)
        _out = os.path.abspath(os.path.join(_dir, f"tp_worker_{int(worker_id)}_{os.getpid()}_{_run}.json"))
        _worker_prof.export_chrome_trace(_out)
        # Also write into the combined TensorBoard trace directory, if configured
        if '_tb_write_fn' in locals() and _tb_write_fn is not None:
            _tb_write_fn(_worker_prof)

    # Signal completion to master
    _wdbg("sending done marker")
    episode_q.put({'wid': worker_id, 'type': 'done'}, timeout=2.0)
    if _PAR_TIMING:
        total = time.time() - t0_glob
        episode_q.put({'wid': worker_id, 'type': 'timing',
                       't_env_reset': t_env_reset,
                       't_get_obs': t_get_obs,
                       't_get_legals': t_get_legals,
                       't_mcts': t_mcts,
                       't_rpc': t_rpc,
                       't_step': t_step,
                       't_pack': t_pack,
                       't_total': total})


def _batched_select_indices(agent: ActionConditionedPPO,
                            reqs: List[Dict]) -> List[Tuple[int, int]]:
    # Returns list of (wid, idx) per req
    if len(reqs) == 0:
        return []
    # Basic validation
    for r in reqs:
        if 'obs' not in r or 'legals' not in r or 'wid' not in r:
            raise KeyError("_batched_select_indices: missing 'obs'/'legals'/'wid' in request")
    # Stack CPU tensors; keep selection on CPU to avoid small GPU kernels
    _t_total0 = time.time() if _PAR_TIMING else 0.0
    _t0 = time.time() if _PAR_TIMING else 0.0
    obs_cpu = torch.stack([torch.as_tensor(r['obs'], dtype=torch.float32) for r in reqs], dim=0)
    seat_cpu = torch.stack([torch.as_tensor(r['seat'], dtype=torch.float32) for r in reqs], dim=0)
    if _PAR_TIMING:
        _BATCHSEL_PROF['t_stack'] += (time.time() - _t0)
    # Flatten legals and build offsets
    _t0 = time.time() if _PAR_TIMING else 0.0
    flat_legals = []
    offs, cnts = [], []
    for r in reqs:
        offs.append(len(flat_legals))
        cnt = len(r['legals'])
        cnts.append(cnt)
        if cnt > 0:
            flat_legals.extend(r['legals'])
    if len(flat_legals) == 0:
        # No legals, return zeros
        return [(int(reqs[i]['wid']), 0) for i in range(len(reqs))]
    if _PAR_TIMING:
        _BATCHSEL_PROF['t_leg_stack'] += (time.time() - _t0)
    _t0 = time.time() if _PAR_TIMING else 0.0
    leg_cpu = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in flat_legals], dim=0)
    if _PAR_TIMING:
        _BATCHSEL_PROF['t_leg_stack'] += (time.time() - _t0)
    with torch.no_grad():
        # Keep selection on CPU device
        cpu_device = torch.device('cpu')
        o_t = obs_cpu
        s_t = seat_cpu
        leg_t = leg_cpu
        # Validate legal encoding in STRICT mode only
        if os.environ.get('SCOPONE_STRICT_CHECKS', '0') == '1':
            ones = leg_t[:, :40].sum(dim=1)
            if not torch.allclose(ones, torch.ones_like(ones)):
                raise RuntimeError("_batched_select_indices: each legal must have one played bit in [:40]")
        _t0 = time.time() if _PAR_TIMING else 0.0
        state_proj = agent.actor.compute_state_proj(o_t, s_t)  # (B,64)
        if _PAR_TIMING:
            _BATCHSEL_PROF['t_state_proj'] += (time.time() - _t0)
        _t0 = time.time() if _PAR_TIMING else 0.0
        a_emb_all = agent.actor.action_enc(leg_t)             # (M,64)
        if _PAR_TIMING:
            _BATCHSEL_PROF['t_action_enc'] += (time.time() - _t0)
        B = o_t.size(0)
        cnts_t = torch.as_tensor(cnts, dtype=torch.long, device=device)
        max_cnt = int(cnts_t.max().item()) if B > 0 else 0
        pos = torch.arange(max_cnt, device=device, dtype=torch.long) if max_cnt > 0 else None
        out_idx = []
        if _PAR_TIMING:
            _BATCHSEL_PROF['sum_B'] += int(B)
            _BATCHSEL_PROF['sum_M'] += int(leg_t.size(0))
            ch = _BATCHSEL_PROF['cnt_hist']
            ch[max_cnt] = int(ch.get(int(max_cnt), 0)) + 1
        if max_cnt > 0:
            _t0 = time.time() if _PAR_TIMING else 0.0
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts_t.unsqueeze(1)
            offs_t = torch.as_tensor(offs, dtype=torch.long, device=device)
            abs_idx = (offs_t.unsqueeze(1) + rel_pos_2d)[mask]
            sample_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            if _PAR_TIMING:
                _BATCHSEL_PROF['t_mask'] += (time.time() - _t0)
            _t0 = time.time() if _PAR_TIMING else 0.0
            a_emb_mb = a_emb_all[abs_idx]
            legal_scores = (a_emb_mb * state_proj[sample_idx]).sum(dim=1)
            if _PAR_TIMING:
                _BATCHSEL_PROF['t_score'] += (time.time() - _t0)
            _t0 = time.time() if _PAR_TIMING else 0.0
            padded = torch.full((B, max_cnt), -1e9, device=device, dtype=legal_scores.dtype)
            padded[mask] = legal_scores
            logits = padded
            probs = torch.softmax(logits, dim=1)
            # Keep only legal positions; set others to zero
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            # Select only rows with at least one legal action for sampling
            valid_rows = (cnts_t > 0)
            sel = torch.zeros((B,), dtype=torch.long, device=device)
            if bool(valid_rows.any()):
                pv = probs[valid_rows]
                mv = mask[valid_rows]
                # Sanitize probabilities: remove NaN/Inf and negatives; strictly validate
                pv = pv.nan_to_num(0.0)
                pv = torch.clamp(pv, min=0.0)
                rs = pv.sum(dim=1, keepdim=True)
                bad_rows = (~torch.isfinite(rs)) | (rs <= 0)
                if bool(bad_rows.any()):
                    raise RuntimeError("_batched_select_indices: invalid probability rows (NaN/Inf or zero-sum)")
                # Sample for exploration
                sub = torch.multinomial(pv, num_samples=1).squeeze(1)
                # Write back selections to full batch
                sel[valid_rows] = sub
            # rows with no legal actions keep sel=0
            sel_cpu = sel.detach().to('cpu')
            for i in range(B):
                out_idx.append((int(reqs[i]['wid']), int(sel_cpu[i].item())))
            if _PAR_TIMING:
                _BATCHSEL_PROF['t_softmax_sample'] += (time.time() - _t0)
        else:
            out_idx = [(int(reqs[i]['wid']), 0) for i in range(B)]
    if _PAR_TIMING:
        _BATCHSEL_PROF['t_total'] += (time.time() - _t_total0)
        _BATCHSEL_PROF['count'] += 1
    return out_idx

def _batched_service(agent: ActionConditionedPPO, reqs: List[Dict]) -> List[Dict]:
    """Service batched policy/value/belief scoring requests.
    Returns a list of dicts with results in the same order as input reqs.
    """
    if len(reqs) == 0:
        return []
    # Stack obs/seat for all reqs once
    obs_cpu = torch.stack([torch.as_tensor(r['obs'], dtype=torch.float32) for r in reqs], dim=0)
    seat_cpu = torch.stack([torch.as_tensor(r.get('seat', [0,0,0,0,0,0]), dtype=torch.float32) for r in reqs], dim=0)
    with torch.no_grad():
        # Use CPU tensors directly when not targeting CUDA to avoid unnecessary pinned memory
        if getattr(device, 'type', str(device)) == 'cuda':
            if device.type == 'cuda':
                o_t = obs_cpu.pin_memory().to(device=device, non_blocking=True)
                s_t = seat_cpu.pin_memory().to(device=device, non_blocking=True)
            else:
                o_t = obs_cpu.to(device=device)
                s_t = seat_cpu.to(device=device)
        else:
            o_t = obs_cpu
            s_t = seat_cpu
        state_proj_all = agent.actor.compute_state_proj(o_t, s_t)  # (N,64)

    results: List[Dict] = [{} for _ in reqs]

    # 1) Batch all score_policy requests together
    policy_positions: List[int] = []
    policy_cnts: List[int] = []
    policy_legals_rows: List[torch.Tensor] = []
    for i, r in enumerate(reqs):
        if r.get('type') == 'score_policy':
            leg = r.get('legals', [])
            if isinstance(leg, torch.Tensor):
                if leg.dim() == 1:
                    leg = leg.unsqueeze(0)
                cnt = int(leg.size(0))
                if cnt > 0:
                    policy_positions.append(i)
                    policy_cnts.append(cnt)
                    policy_legals_rows.append(leg)
                else:
                    results[i] = {'priors': []}
            else:
                # list-like
                cnt = len(leg)
                if cnt > 0:
                    policy_positions.append(i)
                    policy_cnts.append(cnt)
                    policy_legals_rows.append(torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in leg], dim=0))
                else:
                    results[i] = {'priors': []}

    if len(policy_positions) > 0:
        with torch.no_grad():
            # Concatenate legals
            legals_flat = torch.cat(policy_legals_rows, dim=0)
            if device.type == 'cuda':
                legals_flat = legals_flat.pin_memory().to(device=device, non_blocking=True)
            else:
                legals_flat = legals_flat.to(device=device)
            Bp = len(policy_positions)
            cnts_t = torch.as_tensor(policy_cnts, dtype=torch.long, device=device)
            max_cnt = int(cnts_t.max().item()) if Bp > 0 else 0
            # Map from each legal to its policy-sample index
            sample_idx_per_legal = torch.repeat_interleave(torch.arange(Bp, device=device, dtype=torch.long), cnts_t)
            sp = state_proj_all[torch.as_tensor(policy_positions, dtype=torch.long, device=device)]  # (Bp,64)

            # Card logits and mask per sample
            card_logits_all = torch.matmul(sp, agent.actor.card_emb_play.t())  # (Bp,40)
            played_ids_all = torch.argmax(legals_flat[:, :40], dim=1)  # (M_flat)
            logp_cards_all = torch.log_softmax(card_logits_all, dim=1)  # (Bp,40)
            logp_cards_per_legal = logp_cards_all[sample_idx_per_legal, played_ids_all]

            # Capture logits per-legal
            a_emb_flat = agent.actor.action_enc(legals_flat)  # (M_flat,64)
            cap_logits = (a_emb_flat * sp[sample_idx_per_legal]).sum(dim=1)
            # Log-softmax within (sample,card) group
            group_ids = sample_idx_per_legal * 40 + played_ids_all
            num_groups = Bp * 40
            group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
            group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
            gmax_per_legal = group_max[group_ids]
            exp_shifted = torch.exp(cap_logits - gmax_per_legal).to(cap_logits.dtype)
            group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
            group_sum.index_add_(0, group_ids, exp_shifted)
            lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
            logp_cap_per_legal = cap_logits - lse_per_legal

            logp_total_per_legal = logp_cards_per_legal + logp_cap_per_legal
            if not torch.isfinite(logp_total_per_legal).all():
                raise RuntimeError("_batched_service: non-finite log-probs for legals")

            # Softmax over legals per sample (strict validation)
            if max_cnt > 0:
                pos = torch.arange(max_cnt, device=device, dtype=torch.long)
                rel_pos_2d = pos.unsqueeze(0).expand(Bp, max_cnt)
                mask = rel_pos_2d < cnts_t.unsqueeze(1)
                padded = torch.full((Bp, max_cnt), float('-inf'), dtype=logp_total_per_legal.dtype, device=device)
                # Write in flatten order
                # Compute absolute indices in flattened per-sample view (vectorized)
                if Bp > 0:
                    cnts_t = torch.as_tensor(policy_cnts, dtype=torch.long, device=device)
                    starts = torch.cumsum(torch.nn.functional.pad(cnts_t[:-1], (1, 0)), dim=0)
                    offs_t = torch.repeat_interleave(starts, cnts_t)
                else:
                    offs_t = torch.zeros((0,), dtype=torch.long, device=device)
                # Above is heavy; alternatively fill by iterating slices
                # Use a simple loop in CUDA graph-free context for clarity
                start = 0
                for j, c in enumerate(policy_cnts):
                    if c > 0:
                        padded[j, :c] = logp_total_per_legal[start:start+c]
                        start += c
                priors_padded = torch.softmax(padded, dim=1).nan_to_num(0.0)
                priors_padded = torch.clamp(priors_padded, min=0.0)
                row_sums = priors_padded.sum(dim=1, keepdim=True)
                if bool(((row_sums <= 0) | (~torch.isfinite(row_sums))).any()):
                    raise RuntimeError("_batched_service: invalid priors row (NaN/Inf or zero-sum)")
                priors_padded = priors_padded / row_sums
                # Scatter back to per-request lists
                start = 0
                for j, (pos_j, c) in enumerate(zip(policy_positions, policy_cnts)):
                    if c > 0:
                        pri = priors_padded[j, :c].detach().to('cpu').tolist()
                    else:
                        pri = []
                    results[pos_j] = {'priors': pri}
            else:
                for pos_j in policy_positions:
                    results[pos_j] = {'priors': []}

    # 2) Process score_value and score_belief (can remain per-request)
    for i, r in enumerate(reqs):
        rtype = r.get('type')
        if rtype == 'score_value':
            with torch.no_grad():
                o_one = o_t[i:i+1]
                s_one = s_t[i:i+1]
                state_feat = agent.actor.state_enc(o_one, s_one)
                # Align dtype with BeliefNet parameters to avoid Half/Float mismatch
                bn_dtype = agent.actor.belief_net.fc_in.weight.dtype
                if state_feat.dtype != bn_dtype:
                    state_feat = state_feat.to(dtype=bn_dtype)
                logits = agent.actor.belief_net(state_feat)
                hand_table = o_one[:, :83]
                hand_mask = hand_table[:, :40] > 0.5
                table_mask = hand_table[:, 43:83] > 0.5
                captured = o_one[:, 83:165]
                cap0_mask = captured[:, :40] > 0.5
                cap1_mask = captured[:, 40:80] > 0.5
                visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)
                probs_flat = agent.actor.belief_net.probs(logits, visible_mask)
                oh = probs_flat.view(1, 3, 40)
                val = agent.critic(o_one, s_one, oh).squeeze(0)
            results[i] = {'value': float(val.detach().cpu().item())}
        elif rtype == 'score_belief':
            with torch.no_grad():
                _state = agent.actor.state_enc(o_t[i:i+1], s_t[i:i+1])
                bn_dtype = agent.actor.belief_net.fc_in.weight.dtype
                if _state.dtype != bn_dtype:
                    _state = _state.to(dtype=bn_dtype)
                logits = agent.actor.belief_net(_state)
                hand_table = o_t[i:i+1, :83]
                hand_mask = hand_table[:, :40] > 0.5
                table_mask = hand_table[:, 43:83] > 0.5
                captured = o_t[i:i+1, 83:165]
                cap0_mask = captured[:, :40] > 0.5
                cap1_mask = captured[:, 40:80] > 0.5
                visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)
                probs_flat = agent.actor.belief_net.probs(logits, visible_mask).squeeze(0).detach().cpu().tolist()
            results[i] = {'belief_probs': probs_flat}

    return results

def _batched_select_indices_with_actor(actor: ActionConditionedActor, reqs: List[Dict]) -> List[Tuple[int, int]]:
    """Like _batched_select_indices but using a provided frozen actor (no critic)."""
    if len(reqs) == 0:
        return []
    for r in reqs:
        if 'obs' not in r or 'legals' not in r or 'wid' not in r:
            raise KeyError("_batched_select_indices_with_actor: missing 'obs'/'legals'/'wid' in request")
    _t_total0 = time.time() if _PAR_TIMING else 0.0
    obs_cpu = torch.stack([torch.as_tensor(r['obs'], dtype=torch.float32) for r in reqs], dim=0)
    seat_cpu = torch.stack([torch.as_tensor(r['seat'], dtype=torch.float32) for r in reqs], dim=0)
    flat_legals = []
    offs, cnts = [], []
    for r in reqs:
        offs.append(len(flat_legals))
        cnt = len(r['legals'])
        cnts.append(cnt)
        if cnt > 0:
            flat_legals.extend(r['legals'])
    if len(flat_legals) == 0:
        return [(int(reqs[i]['wid']), 0) for i in range(len(reqs))]
    leg_cpu = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in flat_legals], dim=0)
    with torch.no_grad():
        # Use CPU to avoid device churn
        o_t = obs_cpu
        s_t = seat_cpu
        leg_t = leg_cpu
        # State projection via frozen actor's encoder
        state_proj = actor.compute_state_proj(o_t, s_t)
        a_emb_all = actor.action_enc(leg_t)
        B = o_t.size(0)
        cnts_t = torch.as_tensor(cnts, dtype=torch.long)
        max_cnt = int(cnts_t.max().item()) if B > 0 else 0
        pos = torch.arange(max_cnt, dtype=torch.long) if max_cnt > 0 else None
        out_idx = []
        if max_cnt > 0:
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts_t.unsqueeze(1)
            offs_t = torch.as_tensor(offs, dtype=torch.long)
            abs_idx = (offs_t.unsqueeze(1) + rel_pos_2d)[mask]
            sample_idx = torch.arange(B, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            a_emb_mb = a_emb_all[abs_idx]
            legal_scores = (a_emb_mb * state_proj[sample_idx]).sum(dim=1)
            padded = torch.full((B, max_cnt), -1e9, dtype=legal_scores.dtype)
            padded[mask] = legal_scores
            probs = torch.softmax(padded, dim=1)
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            valid_rows = (cnts_t > 0)
            sel = torch.zeros((B,), dtype=torch.long)
            if bool(valid_rows.any()):
                pv = probs[valid_rows]
                mv = mask[valid_rows]
                pv = pv.nan_to_num(0.0)
                pv = torch.clamp(pv, min=0.0)
                rs = pv.sum(dim=1, keepdim=True)
                bad_rows = (~torch.isfinite(rs)) | (rs <= 0)
                if bool(bad_rows.any()):
                    raise RuntimeError("_batched_select_indices_with_actor: invalid probability rows (NaN/Inf or zero-sum)")
                sub = torch.multinomial(pv, num_samples=1).squeeze(1)
                sel[valid_rows] = sub
            sel_cpu = sel.detach()
            for i in range(B):
                out_idx.append((int(reqs[i]['wid']), int(sel_cpu[i].item())))
        else:
            out_idx = [(int(reqs[i]['wid']), 0) for i in range(B)]
    if _PAR_TIMING:
        _BATCHSEL_PROF['t_total'] += (time.time() - _t_total0)
        _BATCHSEL_PROF['count'] += 1
    return out_idx
def _compute_per_seat_diagnostics(agent: ActionConditionedPPO, batch: Dict) -> Dict[str, torch.Tensor]:
    """Calcola approx_kl, entropia e clip_frac per gruppi seat 0/2 vs 1/3.

    Tutte le operazioni restano su GPU; ritorna tensori 0-D su device.
    """
    obs = batch['obs'] if torch.is_tensor(batch['obs']) else torch.as_tensor(batch['obs'], dtype=torch.float32)
    obs = obs.to(device=device)
    seat = batch.get('seat_team', None)
    if seat is None:
        seat = torch.zeros((obs.size(0), 6), dtype=torch.float32)
    elif not torch.is_tensor(seat):
        seat = torch.as_tensor(seat, dtype=torch.float32)
    seat = seat.to(device=device)
    legals = batch['legals'] if torch.is_tensor(batch['legals']) else torch.as_tensor(batch['legals'], dtype=torch.float32)
    offs = batch['legals_offset'] if torch.is_tensor(batch['legals_offset']) else torch.as_tensor(batch['legals_offset'], dtype=torch.long)
    cnts = batch['legals_count'] if torch.is_tensor(batch['legals_count']) else torch.as_tensor(batch['legals_count'], dtype=torch.long)
    chosen_idx = batch['chosen_index'] if torch.is_tensor(batch['chosen_index']) else torch.as_tensor(batch['chosen_index'], dtype=torch.long)
    old_logp = torch.as_tensor(batch['old_logp'], dtype=torch.float32)
    legals = legals.to(device=device)
    offs = offs.to(device=device)
    cnts = cnts.to(device=device)
    chosen_idx = chosen_idx.to(device=device)
    old_logp = old_logp.to(device=device)

    approx_kl = torch.zeros(obs.size(0), dtype=torch.float32, device=device)
    entropy = torch.zeros(obs.size(0), dtype=torch.float32, device=device)
    with torch.no_grad():
        B = obs.size(0)
        # Compute state projection (B,64) con belief neurale interno
        state_proj = agent.actor.compute_state_proj(obs, seat)
        # Action embeddings for all legals in the batch
        a_emb_global = agent.actor.action_enc(legals)  # (M_all,64)
        max_cnt = int(cnts.max().item()) if B > 0 else 0
        if max_cnt > 0:
            pos = torch.arange(max_cnt, device=device, dtype=torch.long)
            rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
            mask = rel_pos_2d < cnts.unsqueeze(1)
            sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
            abs_idx_2d = offs.unsqueeze(1) + rel_pos_2d
            abs_idx = abs_idx_2d[mask]
            a_emb_mb = a_emb_global[abs_idx]
            legal_scores = (a_emb_mb * state_proj[sample_idx_per_legal]).sum(dim=1)
            padded = torch.full((B, max_cnt), -float('inf'), device=device, dtype=legal_scores.dtype)
            padded[mask] = legal_scores
        else:
            padded = torch.full((B, 0), -float('inf'), device=device, dtype=state_proj.dtype)
        logp_group = torch.log_softmax(padded, dim=1)
        probs_group = torch.softmax(padded, dim=1)
        entropy[:] = (-(probs_group * logp_group).sum(dim=1))
        chosen_clamped = torch.minimum(chosen_idx, (cnts - 1).clamp_min(0)) if max_cnt > 0 else chosen_idx
        new_logp_chosen = logp_group[torch.arange(B, device=device), chosen_clamped]
        approx_kl[:] = (old_logp - new_logp_chosen).abs()
    # clip fraction per-sample sul chosen
    # ricalcolo ratio corretto: new_logp_chosen - old_logp
    ratio = torch.exp(new_logp_chosen - old_logp)
    clip_low = 1.0 - agent.clip_ratio
    clip_high = 1.0 + agent.clip_ratio
    clipped_mask = (ratio < clip_low) | (ratio > clip_high)

    seats = torch.argmax(seat[:, :4], dim=1)
    mask_02 = (seats == 0) | (seats == 2)
    mask_13 = (seats == 1) | (seats == 3)
    out: Dict[str, torch.Tensor] = {}
    # Evita branching su CPU: usa conteggi e where per gestire gruppi vuoti
    one = torch.tensor(1.0, device=device, dtype=torch.float32)
    count_02 = mask_02.float().sum()
    count_13 = mask_13.float().sum()
    sum_kl_02 = (approx_kl * mask_02.float()).sum()
    sum_kl_13 = (approx_kl * mask_13.float()).sum()
    sum_en_02 = (entropy * mask_02.float()).sum()
    sum_en_13 = (entropy * mask_13.float()).sum()
    sum_cf_02 = (clipped_mask.float() * mask_02.float()).sum()
    sum_cf_13 = (clipped_mask.float() * mask_13.float()).sum()
    mean_kl_02 = torch.where(count_02 > 0, sum_kl_02 / count_02, torch.zeros((), device=device, dtype=torch.float32))
    mean_kl_13 = torch.where(count_13 > 0, sum_kl_13 / count_13, torch.zeros((), device=device, dtype=torch.float32))
    mean_en_02 = torch.where(count_02 > 0, sum_en_02 / count_02, torch.zeros((), device=device, dtype=torch.float32))
    mean_en_13 = torch.where(count_13 > 0, sum_en_13 / count_13, torch.zeros((), device=device, dtype=torch.float32))
    mean_cf_02 = torch.where(count_02 > 0, sum_cf_02 / count_02, torch.zeros((), device=device, dtype=torch.float32))
    mean_cf_13 = torch.where(count_13 > 0, sum_cf_13 / count_13, torch.zeros((), device=device, dtype=torch.float32))
    out['by_seat/kl_02'] = mean_kl_02
    out['by_seat/kl_13'] = mean_kl_13
    out['by_seat/entropy_02'] = mean_en_02
    out['by_seat/entropy_13'] = mean_en_13
    out['by_seat/clip_frac_02'] = mean_cf_02
    out['by_seat/clip_frac_13'] = mean_cf_13
    return out


def _load_frozen_actor(ckpt_path: str, obs_dim: int) -> ActionConditionedActor:
    actor = ActionConditionedActor(obs_dim=obs_dim)
    actor = maybe_compile_module(actor, name='ActionConditionedActor[trainer_partner]')
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'actor' in ckpt:
            actor.load_state_dict(ckpt['actor'])
        # else: leave randomly init
    except Exception as e:
        # Detailed diagnostics on why loading failed
        import os as _os
        import pickle as _pickle  # noqa: F401
        exists = bool(ckpt_path and _os.path.isfile(ckpt_path))
        size_b = int(_os.path.getsize(ckpt_path)) if exists else -1
        etype = type(e).__name__
        emsg = str(e) if str(e) else repr(e)
        # Heuristics for human-readable reason
        if not exists:
            reason = "not_found"
        elif size_b <= 0:
            reason = "empty_file"
        elif etype == 'EOFError':
            reason = "truncated_or_corrupt"
        elif etype in ('UnpicklingError', 'AttributeError', 'ModuleNotFoundError'):
            reason = "invalid_pickle_or_missing_class"
        elif etype == 'RuntimeError' and ('PytorchStreamReader' in emsg or 'invalid header or archive' in emsg):
            reason = "invalid_archive_or_format"
        else:
            reason = "unknown_error"
        tqdm.write(f"[WARN] Failed to load checkpoint {ckpt_path}: reason={reason} type={etype} size={max(size_b,0)}B msg={emsg}")
        # Optional: escalate failure for CI/debug
        if _os.environ.get('SCOPONE_RAISE_ON_CKPT_FAIL', '0') == '1':
            raise RuntimeError(f"Checkpoint load failed: {ckpt_path} (reason={reason}, type={etype}, size={max(size_b,0)}B)") from e
    actor.eval()
    return actor


def _collect_trajectory_impl(env: ScoponeEnvMA, agent: ActionConditionedPPO, horizon: int = 128,
                       gamma: float = 1.0, lam: float = 0.95,
                       partner_actor: ActionConditionedActor = None,
                       opponent_actor: ActionConditionedActor = None,
                       main_seats: List[int] = None,
                       belief_particles: int = 512, belief_ess_frac: float = 0.5,
                       episodes: int = None, final_reward_only: bool = True,
                       use_mcts: bool = True,
                       mcts_sims: int = 128, mcts_dets: int = 4, mcts_c_puct: float = 1.0,
                       mcts_root_temp: float = 0.0, mcts_prior_smooth_eps: float = 0.0,
                       mcts_dirichlet_alpha: float = 0.25, mcts_dirichlet_eps: float = 0.0,
                       mcts_train_factor: float = 1.0,
                       mcts_progress_start: float = 0.25,
                       mcts_progress_full: float = 0.75,
                       mcts_min_sims: int = 0,
                       train_both_teams: bool = False) -> Dict:
    # Enforce minimum horizon of 40 (full hand length)
    horizon = max(40, int(horizon))
    # Enforce horizon multiple of LCM(minibatch_size, per-episode useful transitions)
    import os as _os
    import math as _math
    minibatch_size = 4096
    env_mb = int(_os.environ.get('SCOPONE_MINIBATCH', str(minibatch_size)))
    if env_mb > 0:
        minibatch_size = env_mb
    per_ep_util = 40 if bool(train_both_teams) else 20
    lcm_mb_ep = (abs(minibatch_size * per_ep_util) // _math.gcd(minibatch_size, per_ep_util)) if (minibatch_size > 0 and per_ep_util > 0) else max(minibatch_size, per_ep_util)
    if lcm_mb_ep > 0 and (horizon % lcm_mb_ep) != 0:
        new_h = ((horizon + lcm_mb_ep - 1) // lcm_mb_ep) * lcm_mb_ep
        global _HORIZON_ADJUST_LOGGED
        if not _HORIZON_ADJUST_LOGGED:
            tqdm.write(f"[horizon] adjusted to LCM(mb={minibatch_size}, per_ep={per_ep_util})={lcm_mb_ep}: {horizon} -> {new_h}")
            _HORIZON_ADJUST_LOGGED = True
        horizon = new_h
    # After alignment, horizon is divisible by per_ep_util by construction
    # Validate gamma/lam
    if float(gamma) < 0 or float(gamma) > 1:
        raise ValueError("collect_trajectory: gamma must be in [0,1]")
    if float(lam) < 0 or float(lam) > 1:
        raise ValueError("collect_trajectory: lam must be in [0,1]")
    obs_list, next_obs_list = [], []
    act_list = []
    rew_list, done_list = [], []
    legals_list, legals_offset, legals_count = [], [], []
    chosen_index_t_list = []
    seat_team_list = []
    belief_sum_list = []
    mcts_policy_flat = []  # concatenazione delle distribuzioni visite MCTS per ciascun sample main
    mcts_weight_list = []  # 1.0 se MCTS usato per il sample, altrimenti 0.0
    # supervision per belief aux: per ogni sample, vettore (3,40) one-hot di mani reali altrui
    others_hands_targets = []


    routing_log = []  # (player_id, source)

    belief_aux_coef = float(os.environ.get('BELIEF_AUX_COEF', '0.1'))
    skip_step_validation = (os.environ.get('SCOPONE_SKIP_STEP_VALIDATION', '1') == '1')

    # Profiling accumulators (shared naming with parallel collector for consistency)
    t_collect_start = time.time() if _PAR_TIMING else 0.0
    t_env_reset = 0.0
    t_get_obs = 0.0
    t_get_legals = 0.0
    t_mcts = 0.0
    t_step = 0.0
    t_select = 0.0
    t_build = 0.0
    t_state_proj = 0.0
    t_action_enc = 0.0
    t_sampling = 0.0
    t_values_gae = 0.0
    t_oldlogp = 0.0
    env_reset_count = 0

    steps = 0
    if final_reward_only:
        # Raccogli per episodi completi: per-episodio util = 40 se alleni entrambe le squadre, altrimenti 20
        _per_ep_util = (40 if bool(train_both_teams) else 20)
        episodes = (max(1, horizon // _per_ep_util) if episodes is None else max(1, int(episodes)))
        episodes_done = 0
    # Tracciamento delle slice episodio e dei team rewards per reward flat
    current_ep_start_idx = 0
    ep_slices: List[Tuple[int, int]] = []
    ep_team_rewards: List[List[float]] = []
    while True:
        if env.done:
            if _PAR_TIMING:
                _t0_env_reset = time.time()
            env.reset()
            if _PAR_TIMING:
                t_env_reset += (time.time() - _t0_env_reset)
                env_reset_count += 1
            current_ep_start_idx = len(obs_list)

        # All env logic on CPU
        if _PAR_TIMING:
            _t0_get_obs = time.time()
        obs = env._get_observation(env.current_player)
        if _PAR_TIMING:
            t_get_obs += (time.time() - _t0_get_obs)
        # Fast-path: get_valid_actions already caches by state; avoid recomputing identical lists
        if _PAR_TIMING:
            _t0_get_legals = time.time()
        legal = env.get_valid_actions()
        if _PAR_TIMING:
            t_get_legals += (time.time() - _t0_get_legals)
        if torch.is_tensor(legal) and (legal.size(0) == 0):
            cp = env.current_player
            hand_ids_dbg = list(env._hands_ids.get(cp, [])) if hasattr(env, '_hands_ids') else []
            table_ids_dbg = list(env._table_ids) if hasattr(env, '_table_ids') else []
            hist_len = len(env.game_state.get('history', [])) if isinstance(env.game_state, dict) else None
            rules_dbg = getattr(env, 'rules', {})
            raise RuntimeError(f"collect_trajectory: get_valid_actions returned 0 legals; player={cp}, hand={hand_ids_dbg}, table={table_ids_dbg}, done={bool(env.done)}, history_len={hist_len}, rules={rules_dbg}")

        cp = env.current_player
        seat_vec = _seat_vec_for(cp)

        # Selezione azione: se train_both_teams è True, tutti i seat sono "main"
        is_main = True if train_both_teams else ((main_seats is None and cp in [0, 2]) or (main_seats is not None and cp in main_seats))
        bsum_tensor = _BELIEF_ZERO
        use_mcts_cur = False
        sims_scaled = 0
        alpha = 0.0
        root_temp_dyn = float(mcts_root_temp)
        _both_sides = str(os.environ.get('SCOPONE_MCTS_BOTH_SIDES', '1')).strip().lower() in ['1','true','yes','on']
        if is_main or _both_sides:
            # MCTS sempre attivo (stile AlphaZero): poche simulazioni sempre, scala con il progresso della mano
            progress = float(min(1.0, max(0.0, len(env.game_state.get('history', [])) / 40.0)))
            if use_mcts and len(legal) > 0 and int(mcts_sims) > 0:
                if mcts_train_factor is None or float(mcts_train_factor) > 0.0:
                    denom = max(1e-6, (mcts_progress_full - mcts_progress_start))
                    alpha = min(1.0, max(0.0, (progress - mcts_progress_start) / denom))
                    import math
                    scaling_on = (str(os.environ.get('SCOPONE_MCTS_SCALING', '1')).strip().lower() in ['1','true','yes','on'])
                    if scaling_on:
                        sims_base = int(math.ceil(float(mcts_sims) * (0.25 + 0.75 * alpha)))
                        if mcts_train_factor is not None:
                            sims_base = int(math.ceil(sims_base * float(mcts_train_factor)))
                    else:
                        sims_base = int(math.ceil(float(mcts_sims) * float(mcts_train_factor if mcts_train_factor is not None else 1.0)))
                    base_min = int(mcts_min_sims) if (mcts_min_sims is not None and int(mcts_min_sims) >= 0) else 0
                    sims_scaled = max(base_min, sims_base)
                    if int(mcts_sims) > 0 and sims_scaled <= 0:
                        sims_scaled = 1
                    use_mcts_cur = sims_scaled > 0
                else:
                    sims_scaled = 0
            if use_mcts_cur:
                scaling_on = (str(os.environ.get('SCOPONE_MCTS_SCALING', '1')).strip().lower() in ['1','true','yes','on'])
                root_temp_dyn = float(mcts_root_temp) if (not scaling_on or float(mcts_root_temp) > 0) else float(max(0.0, 1.0 - alpha))

            # Belief summary per il giocatore corrente: opzionale (disabilitato di default)
            _enable_bsum = (os.environ.get('ENABLE_BELIEF_SUMMARY', '0') == '1')
            if _enable_bsum and use_mcts_cur:
                o_cpu = obs.clone().detach().to('cpu', dtype=torch.float32) if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32, device='cpu')
                s_cpu = seat_vec.detach().to('cpu', dtype=torch.float32)
                if device.type == 'cuda':
                    o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                else:
                    o_t = o_cpu.unsqueeze(0).to(device=device)
                    s_t = s_cpu.unsqueeze(0).to(device=device)
                with torch.no_grad():
                    state_feat = agent.actor.state_enc(o_t, s_t)
                    bn_dtype = agent.actor.belief_net.fc_in.weight.dtype
                    if state_feat.dtype != bn_dtype:
                        state_feat = state_feat.to(dtype=bn_dtype)
                    logits = agent.actor.belief_net(state_feat)
                    hand_table = o_t[:, :83]
                    hand_mask = hand_table[:, :40] > 0.5
                    table_mask = hand_table[:, 43:83] > 0.5
                    captured = o_t[:, 83:165]
                    cap0_mask = captured[:, :40] > 0.5
                    cap1_mask = captured[:, 40:80] > 0.5
                    visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)
                    probs_flat = agent.actor.belief_net.probs(logits, visible_mask)
                bsum_tensor = probs_flat.squeeze(0).detach().to('cpu')
            if use_mcts_cur:
                if _PAR_TIMING:
                    _t0_mcts = time.time()
                # MCTS con determinizzazione dal belief del giocatore corrente
                from algorithms.is_mcts import run_is_mcts
                import numpy as _np
                # Policy: usa l'actor per generare prior sui legali
                def policy_fn_mcts(_obs, _legals):
                    # Prior coerenti con policy fattorizzata (carta ⊕ presa)
                    o_cpu = _obs if torch.is_tensor(_obs) else torch.as_tensor(_obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    leg_cpu = torch.stack([
                        (x.detach().to('cpu', dtype=torch.float32) if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
                    for x in _legals], dim=0)
                    if device.type == 'cuda':
                        o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
                    else:
                        o_t = o_cpu.unsqueeze(0).to(device=device)
                        leg_t = leg_cpu.to(device=device)
                    with torch.no_grad():
                        sp = agent.actor.compute_state_proj(o_t, _seat_vec_for(env.current_player).unsqueeze(0).to(device=device))
                        # Evita cast/copie ripetute del parametro
                        card_logits_all = torch.matmul(sp, agent.actor.card_emb_play.t()).squeeze(0)
                        played_ids = torch.argmax(leg_t[:, :40], dim=1)
                        unique_ids, inv_idx = torch.unique(played_ids, sorted=False, return_inverse=True)
                        allowed_logits = card_logits_all[unique_ids]
                        logp_cards_unique = torch.log_softmax(allowed_logits, dim=0)
                        logp_cards_per_legal = logp_cards_unique[inv_idx]
                        a_emb = agent.actor.action_enc(leg_t)
                        cap_logits = torch.matmul(a_emb, sp.squeeze(0))
                        group_ids = played_ids
                        num_groups = 40
                        group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=leg_t.device)
                        group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
                        gmax_per_legal = group_max[group_ids]
                        exp_shifted = torch.exp(cap_logits - gmax_per_legal).to(cap_logits.dtype)
                        group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=leg_t.device)
                        group_sum.index_add_(0, group_ids, exp_shifted)
                        lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
                        logp_cap_per_legal = cap_logits - lse_per_legal
                        logp_total = logp_cards_per_legal + logp_cap_per_legal
                        priors = torch.softmax(logp_total, dim=0).nan_to_num(0.0)
                        priors = torch.clamp(priors, min=0.0)
                        ssum = priors.sum()
                        if (not torch.isfinite(ssum)) or (ssum <= 0):
                            raise RuntimeError("collect_trajectory.policy_fn_mcts: invalid priors (NaN/Inf or zero-sum)")
                    return priors.detach().cpu().numpy()
                # Value: usa il critic con belief neurale interno e others_hands predetto
                def value_fn_mcts(_obs, _env):
                    # Prepara seat_team
                    o_cpu = _obs if torch.is_tensor(_obs) else torch.as_tensor(_obs, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    # seat vector
                    seat_vec = torch.zeros(6, dtype=torch.float32)
                    cp_loc = _env.current_player
                    seat_vec[cp_loc] = 1.0
                    seat_vec[4] = 1.0 if cp_loc in [0, 2] else 0.0
                    seat_vec[5] = 1.0 if cp_loc in [1, 3] else 0.0
                    # To CUDA
                    if device.type == 'cuda':
                        o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        s_t = seat_vec.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    else:
                        o_t = o_cpu.unsqueeze(0).to(device=device)
                        s_t = seat_vec.unsqueeze(0).to(device=device)
                    # Costruisci others_hands predetto dal BeliefNet con masking delle carte visibili
                    with torch.no_grad():
                        state_feat = agent.actor.state_enc(o_t, s_t)
                        bn_dtype = agent.actor.belief_net.fc_in.weight.dtype
                        if state_feat.dtype != bn_dtype:
                            state_feat = state_feat.to(dtype=bn_dtype)
                        logits = agent.actor.belief_net(state_feat)  # (1,120)
                        hand_table = o_t[:, :83]
                        hand_mask = hand_table[:, :40] > 0.5
                        table_mask = hand_table[:, 43:83] > 0.5
                        captured = o_t[:, 83:165]
                        cap0_mask = captured[:, :40] > 0.5
                        cap1_mask = captured[:, 40:80] > 0.5
                        visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)  # (1,40)
                        probs_flat = agent.actor.belief_net.probs(logits, visible_mask)  # (1,120)
                        oh = probs_flat.view(1, 3, 40)
                        v = agent.critic(o_t, s_t, oh)
                    return float(v.squeeze(0).detach().cpu().item())
                # Belief determinization sampler dal BeliefNet: campiona assignment coerenti
                def belief_sampler_neural(_env):
                    # Costruisci marginali 3x40 dal BeliefNet
                    obs_cur = _env._get_observation(_env.current_player)
                    o_cpu = obs_cur if torch.is_tensor(obs_cur) else torch.as_tensor(obs_cur, dtype=torch.float32)
                    if torch.is_tensor(o_cpu):
                        o_cpu = o_cpu.detach().to('cpu', dtype=torch.float32)
                    s_cpu = _seat_vec_for(_env.current_player).detach().to('cpu', dtype=torch.float32)
                    if device.type == 'cuda':
                        o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    else:
                        o_t = o_cpu.unsqueeze(0).to(device=device)
                        s_t = s_cpu.unsqueeze(0).to(device=device)
                    with torch.no_grad():
                        state_feat = agent.actor.state_enc(o_t, s_t)
                        bn_dtype = agent.actor.belief_net.fc_in.weight.dtype
                        if state_feat.dtype != bn_dtype:
                            state_feat = state_feat.to(dtype=bn_dtype)
                        logits = agent.actor.belief_net(state_feat)  # (1,120)
                        # Visibilità dall'osservazione
                        hand_table = o_t[:, :83]
                        hand_mask = hand_table[:, :40] > 0.5
                        table_mask = hand_table[:, 43:83] > 0.5
                        captured = o_t[:, 83:165]
                        cap0_mask = captured[:, :40] > 0.5
                        cap1_mask = captured[:, 40:80] > 0.5
                        visible_mask = (hand_mask | table_mask | cap0_mask | cap1_mask)  # (1,40)
                        probs_flat = agent.actor.belief_net.probs(logits, visible_mask)  # (1,120)
                    probs = probs_flat.view(3, 40).detach().cpu().numpy()  # (3,40)
                    vis = visible_mask.squeeze(0).detach().cpu().numpy().astype(bool)
                    unknown_ids = [cid for cid in range(40) if not vis[cid]]
                    # Capacità (conteggi mano) correnti degli altri giocatori
                    others = [(_env.current_player + 1) % 4, (_env.current_player + 2) % 4, (_env.current_player + 3) % 4]
                    counts = {pid: len(_env.game_state['hands'][pid]) for pid in others}
                    caps = [int(counts.get(pid, 0)) for pid in others]
                    n = len(unknown_ids)
                    # riallinea capacità se necessario
                    if sum(caps) != n:
                        caps[2] = max(0, n - caps[0] - caps[1])
                        if sum(caps) != n:
                            from utils.fallback import notify_fallback
                            notify_fallback('trainer.worker.belief_sampler.uniform_caps')
                    # Costi = -log p con piccolo rumore per diversità
                    noise_scale = float(os.environ.get('DET_NOISE', '0.0'))
                    costs = []  # shape (n,3)
                    for cid in unknown_ids:
                        pc = probs[:, cid]
                        ps = pc / max(1e-12, pc.sum())
                        c = [-_np.log(max(1e-12, ps[i])) for i in range(3)]
                        if noise_scale > 0:
                            u = _np.random.uniform(1e-9, 1.0-1e-9, size=3)
                            g = -_np.log(-_np.log(u)) * noise_scale
                            c = [c[i] + float(g[i]) for i in range(3)]
                        costs.append(c)
                    # DP ottimo per 3 giocatori con capacità note
                    INF = 1e12
                    cap0, cap1, cap2 = caps
                    dp = [[[INF]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                    bk = [[[-1]*(cap1+1) for _ in range(cap0+1)] for __ in range(n+1)]
                    dp[0][0][0] = 0.0
                    for t in range(n):
                        c0, c1, c2 = costs[t]
                        for a in range(0, min(t, cap0)+1):
                            for b in range(0, min(t-a, cap1)+1):
                                cur = dp[t][a][b]
                                if cur >= INF: 
                                    continue
                                # assign to player 0
                                if a+1 <= cap0:
                                    if dp[t+1][a+1][b] > cur + c0:
                                        dp[t+1][a+1][b] = cur + c0
                                        bk[t+1][a+1][b] = 0
                                # assign to player 1
                                if b+1 <= cap1:
                                    if dp[t+1][a][b+1] > cur + c1:
                                        dp[t+1][a][b+1] = cur + c1
                                        bk[t+1][a][b+1] = 1
                                # assign to player 2 (implicit count)
                                assigned2 = t - a - b
                                if assigned2 + 1 <= cap2:
                                    if dp[t+1][a][b] > cur + c2:
                                        dp[t+1][a][b] = cur + c2
                                        bk[t+1][a][b] = 2
                    if dp[n][cap0][cap1] >= INF:
                        from utils.fallback import notify_fallback
                        notify_fallback('trainer.belief_sampler.dp_infeasible')
                    # Ricostruisci percorso
                    det = {pid: [] for pid in others}
                    a, b = cap0, cap1
                    for t in range(n, 0, -1):
                        choice = bk[t][a][b]
                        cid = unknown_ids[t-1]
                        if choice == 0:
                            det[others[0]].append(cid)
                            a -= 1
                        elif choice == 1:
                            det[others[1]].append(cid)
                            b -= 1
                        else:
                            det[others[2]].append(cid)
                    return det
                # temperatura radice dinamica: alta a inizio mano, bassa verso la fine (solo se scaling attivo)
                scaling_on = (str(os.environ.get('SCOPONE_MCTS_SCALING', '1')).strip().lower() in ['1','true','yes','on'])
                root_temp_dyn = float(mcts_root_temp) if (not scaling_on or float(mcts_root_temp) > 0) else float(max(0.0, 1.0 - alpha))
                # Auto-tune exploration at root: smoothing and Dirichlet
                priors_probe = policy_fn_mcts(obs, legal)
                pri_t = (priors_probe if torch.is_tensor(priors_probe) else torch.as_tensor(priors_probe, dtype=torch.float32))
                peak = float(pri_t.max().item()) if pri_t.numel() > 0 else (1.0 / max(1, len(legal)))
                A = int(len(legal))
                sims_fac = 1.0 if sims_scaled < 128 else (0.5 if sims_scaled < 256 else 0.25)
                peak_fac = min(1.0, max(0.0, (peak - 0.5) / 0.4))
                prior_eps_dyn = 0.1 * sims_fac * peak_fac * (1.0 - alpha)
                prior_eps_dyn = float(max(0.0, min(0.15, prior_eps_dyn)))
                if A <= 3:
                    dir_eps_dyn = 0.0
                else:
                    a_fac = min(1.0, max(0.0, (A - 3) / 10.0))
                    sim_att = (0.7 if sims_scaled >= 256 else 1.0)
                    prog_att = (0.7 if alpha > 0.7 else 1.0)
                    dir_eps_dyn = 0.25 * a_fac * sim_att * prog_att
                    dir_eps_dyn = float(max(0.0, min(0.3, dir_eps_dyn)))
                # Respect explicit overrides if provided (>0)
                prior_eps_eff = float(mcts_prior_smooth_eps) if float(mcts_prior_smooth_eps) > 0 else prior_eps_dyn
                dir_eps_eff = float(mcts_dirichlet_eps) if float(mcts_dirichlet_eps) > 0 else dir_eps_dyn

                mcts_action, mcts_visits = run_is_mcts(env,
                                          policy_fn=policy_fn_mcts,
                                          value_fn=value_fn_mcts,
                                          num_simulations=int(sims_scaled),
                                          c_puct=float(mcts_c_puct),
                                          belief=None,
                                          num_determinization=int(mcts_dets),
                                          root_temperature=root_temp_dyn,
                                          prior_smooth_eps=prior_eps_eff,
                                          robust_child=True,
                                          root_dirichlet_alpha=float(mcts_dirichlet_alpha),
                                          root_dirichlet_eps=dir_eps_eff,
                                          return_stats=True,
                                          belief_sampler=belief_sampler_neural)
                chosen_act = mcts_action if torch.is_tensor(mcts_action) else torch.as_tensor(mcts_action, dtype=torch.float32)
                # trova indice dell'azione scelta tra i legali in O(A) vettoriale
                # Bitset hashing mapping: encode action as (played_id, capture_bits) to find index in O(A)
                def _encode_action_64(vec80: torch.Tensor) -> torch.Tensor:
                    played_id = torch.argmax(vec80[:40]).to(torch.int64)
                    cap_mask = (vec80[40:] > 0.5).to(torch.int64)
                    # pack 40 capture bits into 64-bit integer
                    idxs = torch.arange(40, dtype=torch.int64)
                    bits = (cap_mask << idxs).sum()
                    return (played_id | (bits << 6)).to(torch.int64)
                code_ch = _encode_action_64(chosen_act)
                if torch.is_tensor(legal):
                    legals_t = legal
                else:
                    legals_t = torch.stack([(x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)) for x in legal], dim=0)
                # vectorized encode for legals
                played_ids = torch.argmax(legals_t[:, :40], dim=1).to(torch.int64)
                cap_mask = (legals_t[:, 40:] > 0.5).to(torch.int64)
                idxs = torch.arange(40, dtype=torch.int64).unsqueeze(0)
                bits = (cap_mask << idxs).sum(dim=1)
                codes = (played_ids | (bits << 6)).to(torch.int64)
                matches = (codes == code_ch)
                nz = torch.nonzero(matches, as_tuple=False).flatten()
                idx_t = (nz[0] if nz.numel() > 0 else torch.tensor(0, dtype=torch.long))
                if _PAR_TIMING:
                    _t0_step = time.time()
                next_obs, rew, done, info = env.step(chosen_act)
                if _PAR_TIMING:
                    t_step += (time.time() - _t0_step)
                routing_log.append((cp, 'mcts'))
                # registra target distillazione per questo sample (ordine legali corrente)
                mcts_probs = torch.as_tensor(mcts_visits, dtype=torch.float32)
                # normalizza in caso di degenerazione
                s = float(mcts_probs.sum().item())
                if s > 0:
                    mcts_probs = mcts_probs / s
                mcts_policy_flat.extend((mcts_probs.tolist() if hasattr(mcts_probs, 'tolist') else list(mcts_probs)))
                mcts_weight_list.append(1.0)
                if _PAR_TIMING:
                    t_mcts += (time.time() - _t0_mcts)
            else:
                if _PAR_TIMING:
                    prof_bins = {'state_proj': 0.0, 'action_enc': 0.0, 'sampling': 0.0}
                    _t0_select = time.time()
                else:
                    prof_bins = None
                chosen_act, _logp, idx_t = agent.select_action(obs, legal, seat_vec, profiling_bins=prof_bins)
                if _PAR_TIMING:
                    select_elapsed = time.time() - _t0_select
                    t_select += select_elapsed
                    t_state_proj += prof_bins.get('state_proj', 0.0)
                    t_action_enc += prof_bins.get('action_enc', 0.0)
                    t_sampling += prof_bins.get('sampling', 0.0)
                    _t0_step = time.time()
                next_obs, rew, done, info = env.step(chosen_act)
                if _PAR_TIMING:
                    t_step += (time.time() - _t0_step)
                routing_log.append((cp, 'main'))
                # Per mantenere l'allineamento per-sample della distillazione, aggiungi zeri
                mcts_policy_flat.extend([0.0] * len(legal))
                mcts_weight_list.append(0.0)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            act_list.append(chosen_act)
            rew_list.append(rew)
            done_list.append(done)
            seat_team_list.append(seat_vec)
            belief_sum_list.append(bsum_tensor)
            legals_offset.append(len(legals_list))
            legals_count.append(len(legal))
            chosen_index_t_list.append(idx_t)
            legals_list.extend(legal)
            # costruisci target mani reali altrui (3x40) vettoriale sfruttando bitset CPU
            if belief_aux_coef <= 0.0:
                others_hands_targets.append(_OTHERS_HANDS_ZERO)
            else:
                others = [ (cp + 1) % 4, (cp + 2) % 4, (cp + 3) % 4 ]
                # usa mirror bitset CPU se disponibile per evitare loop Python
                if hasattr(env, '_hands_bits_t') and hasattr(env, '_id_range'):
                    ids = env._id_range.detach().to('cpu', dtype=torch.long) if torch.is_tensor(env._id_range) else torch.arange(40, dtype=torch.long)
                    target = _OTHERS_HANDS_ZERO.clone()
                    for i, pid in enumerate(others):
                        bits_t = env._hands_bits_t[pid]
                        bits = int(bits_t.item()) if torch.is_tensor(bits_t) else int(bits_t)
                        mask = (((torch.tensor(bits, dtype=torch.int64) >> ids) & 1).to(torch.float32))
                        target[i] = mask
                    others_hands_targets.append(target)
                else:
                    from utils.fallback import notify_fallback
                    notify_fallback('trainer.others_hands_targets.slow_game_state_path')
                    others_hands_targets.append(_OTHERS_HANDS_ZERO)
        else:
            # partner congelato sui seat del compagno; opponent sugli avversari
            is_partner_seat = (cp in [0, 2] and (main_seats == [1, 3])) or (cp in [1, 3] and (main_seats == [0, 2]))
            frozen = partner_actor if (is_partner_seat and partner_actor is not None) else opponent_actor
            if frozen is not None:
                # Fast, inference-only path for frozen actor
                _inference = torch.inference_mode if hasattr(torch, 'inference_mode') else torch.no_grad
                with _inference():
                    # Use GPU for frozen actor scoring but keep env data on CPU
                    o_cpu = obs.clone().detach().to('cpu', dtype=torch.float32) if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32, device='cpu')
                    leg_cpu = torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device='cpu') for x in legal], dim=0)
                    if device.type == 'cuda':
                        o_t = o_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                        leg_t = leg_cpu.pin_memory().to(device=device, non_blocking=True)
                    else:
                        o_t = o_cpu.unsqueeze(0).to(device=device)
                        leg_t = leg_cpu.to(device=device)
                    # Build seat_team_vec (B,6) for the frozen actor forward
                    s_cpu = seat_vec.clone().detach().to('cpu', dtype=torch.float32)
                    if device.type == 'cuda':
                        s_t = s_cpu.pin_memory().unsqueeze(0).to(device=device, non_blocking=True)
                    else:
                        s_t = s_cpu.unsqueeze(0).to(device=device)
                    logits = frozen(o_t, leg_t, s_t)
                    idx_t = torch.argmax(logits).to('cpu')
                    act = leg_cpu[idx_t]
            else:
                idx_t = torch.randint(len(legal), (1,), device='cpu').squeeze(0)
                leg_t = torch.stack([
                    x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32, device='cpu')
                for x in legal], dim=0)
                act = leg_t[idx_t]
                # Fast chosen index via 64-bit action code (played_id | capture_bits<<6)
                def _encode_action_64_cpu(vec80: torch.Tensor) -> torch.Tensor:
                    played_id = torch.argmax(vec80[:40]).to(torch.int64)
                    cap_mask = (vec80[40:] > 0.5).to(torch.int64)
                    idxs = torch.arange(40, dtype=torch.int64)
                    bits = (cap_mask << idxs).sum()
                    return (played_id | (bits << 6)).to(torch.int64)
                code_ch = _encode_action_64_cpu(act)
                played_ids = torch.argmax(leg_t[:, :40], dim=1).to(torch.int64)
                cap_mask = (leg_t[:, 40:] > 0.5).to(torch.int64)
                idxs = torch.arange(40, dtype=torch.int64).unsqueeze(0)
                bits = (cap_mask << idxs).sum(dim=1)
                codes = (played_ids | (bits << 6)).to(torch.int64)
                matches = (codes == code_ch)
                nz = torch.nonzero(matches, as_tuple=False).flatten()
                idx_t = (nz[0] if nz.numel() > 0 else torch.tensor(0, dtype=torch.long))
            if _PAR_TIMING:
                _t0_step = time.time()
            next_obs, rew, done, info = env.step(act)
            if _PAR_TIMING:
                t_step += (time.time() - _t0_step)
            routing_log.append((cp, 'partner' if is_partner_seat else 'opponent'))
            # Per i seat non-main mantieni l'allineamento dei target
            mcts_policy_flat.extend([0.0] * len(legal))
            mcts_weight_list.append(0.0)

            # Quando l'episodio termina su un seat non-main, marca l'ultimo
            # sample "main" come terminale e azzera la sua next_obs per evitare
            # che il GAE bootstrappi sul valore dello stato dell'avversario.
            if done and len(done_list) > 0:
                done_list[-1] = True
                try:
                    final_obs = _to_cpu_float32(next_obs)
                except Exception:
                    final_obs = torch.zeros_like(next_obs_list[-1]) if len(next_obs_list) > 0 else _to_cpu_float32(next_obs)
                if len(next_obs_list) > 0:
                    next_obs_list[-1] = final_obs

        steps += 1
        # Condizioni di uscita: per episodi o per passi
        if final_reward_only:
            if done:
                # Registra i confini dell'episodio corrente e i team rewards
                ep_slices.append((current_ep_start_idx, len(obs_list)))
                tr = info.get('team_rewards', [0.0, 0.0]) if isinstance(info, dict) else [0.0, 0.0]
                ep_team_rewards.append(tr)
                if episodes_done >= (episodes - 1):
                    break
                else:
                    episodes_done += 1
                    current_ep_start_idx = len(obs_list)
                    continue
        else:
            if steps >= horizon:
                break

    # Precompute others_hands tensor (CPU) for CTDE targets and critic calls
    oh_cpu_tensor = torch.stack(others_hands_targets, dim=0) if len(others_hands_targets) > 0 else None

    # CTDE: stima V(next) vettorizzata su GPU
    next_val_t = None
    oh_dev_tensor: Optional[torch.Tensor] = None
    if len(next_obs_list) > 0:
        with torch.no_grad():
            next_obs_t = torch.stack([torch.as_tensor(no, dtype=torch.float32, device=device) for no in next_obs_list], dim=0)
            s_all = torch.stack(seat_team_list, dim=0)
            done_mask_bool = torch.as_tensor([bool(d) for d in done_list], dtype=torch.bool, device=device)
            if oh_cpu_tensor is not None:
                if device.type == 'cuda':
                    oh_dev_tensor = oh_cpu_tensor.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
                else:
                    oh_dev_tensor = oh_cpu_tensor.to(device=device, dtype=torch.float32)
                oh_next = torch.zeros_like(oh_dev_tensor)
                if oh_next.size(0) > 1:
                    oh_next[:-1] = oh_dev_tensor[1:]
            else:
                oh_dev_tensor = None
                oh_next = None
            next_val_t = agent.critic(next_obs_t, s_all, oh_next)
            next_val_t = torch.where(done_mask_bool, torch.zeros_like(next_val_t), next_val_t)

    # Compute V(obs) in batch su GPU e GAE
    T = len(rew_list)
    # Consistency check: numero di transizioni utili per episodio (basato su obs_list)
    if final_reward_only:
        per_ep_util = 40 if bool(train_both_teams) else 20
        expected_util = int(episodes) * int(per_ep_util)
        obs_len = int(len(obs_list))
        if obs_len != expected_util:
            import torch as _t
            if len(seat_team_list) > 0:
                st = _t.stack(seat_team_list, dim=0)
                seat_idx = _t.argmax(st[:, :4], dim=1)
                c0 = int((seat_idx == 0).sum().item())
                c1 = int((seat_idx == 1).sum().item())
                c2 = int((seat_idx == 2).sum().item())
                c3 = int((seat_idx == 3).sum().item())
                seat_counts = (c0, c1, c2, c3)
            else:
                seat_counts = (0, 0, 0, 0)
            # distribuzione lunghezze episodiche
            ep_lengths = [int(j - i) for (i, j) in ep_slices]
            uniq = sorted(set(ep_lengths))
            freq = {L: ep_lengths.count(L) for L in uniq}
            raise RuntimeError(
                f"collect_trajectory: util transitions mismatch; got_obs={obs_len}, expected={expected_util}; "
                f"episodes_completed={len(ep_slices)} (requested={int(episodes)}), per_ep_expected={per_ep_util}, "
                f"length_stats={freq}, seat_counts={seat_counts}"
            )
        # I done provenienti dall'env possono rimanere tutti False quando l'ultimo
        # step utile dell'episodio è giocato da un seat non-main. In tal caso il
        # loop salta quel passo e la flag done non viene mai registrata, causando
        # un'unica traccia lunga attraverso più episodi. Forza quindi il flag
        # done sull'ultimo sample di ciascun episodio raccolto, così che il GAE
        # venga azzerato correttamente tra un episodio e il successivo.
        if len(ep_slices) > 0 and len(done_list) > 0:
            for start, end in ep_slices:
                if end > start:
                    done_list[end - 1] = True
            if next_val_t is not None:
                # Assicura che le stime V(s') dei terminali rispettino il flag done forzato
                updated_done_mask = torch.as_tensor(done_list, dtype=torch.bool, device=next_val_t.device)
                next_val_t = torch.where(updated_done_mask, torch.zeros_like(next_val_t), next_val_t)
    # Costruisci reward flat ±1 per tutte le transizioni dell'episodio, per entrambi i team
    if T > 0 and len(ep_slices) > 0:
        flat_rew = [0.0] * T
        for i_ep, (s, e) in enumerate(ep_slices):
            tr = ep_team_rewards[i_ep] if i_ep < len(ep_team_rewards) else [0.0, 0.0]
            # Determina la reward per team 0 e team 1 (proporzionale al risultato finale)
            t0 = float(tr[0])
            t1 = float(tr[1])
            for i in range(s, e):
                st = seat_team_list[i]
                team0_flag = bool(float(st[4]) > 0.5)
                flat_rew[i] = t0 if team0_flag else t1
        rew_t = torch.as_tensor(flat_rew, dtype=torch.float32, device=device)
    else:
        rew_t = torch.as_tensor(rew_list, dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    done_mask = torch.as_tensor([0.0 if not d else 1.0 for d in done_list], dtype=torch.float32, device=device) if T>0 else torch.zeros((0,), dtype=torch.float32, device=device)
    if T > 0:
        with torch.no_grad():
            o_all = torch.stack([torch.as_tensor(o, dtype=torch.float32, device=device) for o in obs_list], dim=0)
            s_all = torch.stack(seat_team_list, dim=0)
            # others_hands per-step (CTDE)
            if oh_dev_tensor is None and oh_cpu_tensor is not None:
                if device.type == 'cuda':
                    oh_dev_tensor = oh_cpu_tensor.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
                else:
                    oh_dev_tensor = oh_cpu_tensor.to(device=device, dtype=torch.float32)
            val_t = agent.critic(o_all, s_all, oh_dev_tensor)
            nval_t = next_val_t if next_val_t is not None else torch.zeros_like(val_t)
    else:
        val_t = torch.zeros((0,), dtype=torch.float32, device=device)
        nval_t = torch.zeros((0,), dtype=torch.float32, device=device)
    adv_vec = torch.zeros_like(rew_t)
    gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    for t in reversed(range(T)):
        delta = rew_t[t] + gamma * nval_t[t] - val_t[t]
        gae = delta + gamma * lam * (1.0 - done_mask[t]) * gae
        adv_vec[t] = gae
    ret_vec = adv_vec + val_t

    # Keep batch entirely as torch tensors on CUDA
    # Build CPU tensors first, then pin and transfer as a batch later in update
    if _PAR_TIMING:
        _t0_build = time.time()
    obs_cpu = torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o, dtype=torch.float32) for o in obs_list], dim=0) if len(obs_list)>0 else torch.zeros((0, env.observation_space.shape[0]), dtype=torch.float32)
    act_cpu = torch.stack([a if torch.is_tensor(a) else torch.as_tensor(a, dtype=torch.float32) for a in act_list], dim=0) if len(act_list)>0 else torch.zeros((0, 80), dtype=torch.float32)
    legals_cpu = torch.stack([l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.float32) for l in legals_list], dim=0) if legals_list else _EMPTY_LEGAL
    seat_team_cpu = torch.stack(seat_team_list, dim=0) if len(seat_team_list)>0 else torch.zeros((0,6), dtype=torch.float32)
    belief_sum_cpu = torch.stack(belief_sum_list, dim=0) if len(belief_sum_list)>0 else torch.zeros((0,120), dtype=torch.float32)
    legals_offset_cpu = torch.as_tensor(legals_offset, dtype=torch.long) if len(legals_offset)>0 else torch.zeros((0,), dtype=torch.long)
    legals_count_cpu = torch.as_tensor(legals_count, dtype=torch.long) if len(legals_count)>0 else torch.zeros((0,), dtype=torch.long)
    chosen_index_cpu = (torch.stack(chosen_index_t_list, dim=0).to(dtype=torch.long) if len(chosen_index_t_list)>0 else torch.zeros((0,), dtype=torch.long))
    # Evita tensori inutili pieni di zeri quando MCTS non è usato
    mcts_policy_cpu = torch.as_tensor(mcts_policy_flat, dtype=torch.float32) if any((x != 0.0) for x in mcts_policy_flat) else torch.zeros((0,), dtype=torch.float32)
    mcts_weight_cpu = torch.as_tensor(mcts_weight_list, dtype=torch.float32) if any((x != 0.0) for x in mcts_weight_list) else torch.zeros((0,), dtype=torch.float32)
    if oh_cpu_tensor is not None:
        others_hands_cpu = oh_cpu_tensor
    else:
        others_hands_cpu = torch.zeros((0,3,40), dtype=torch.float32)
    if _PAR_TIMING:
        t_build += (time.time() - _t0_build)
    # Sanitizza lunghezze: policy_flat deve avere somma(cnts) elementi e weight deve avere len(obs)
    total_legals = int(legals_count_cpu.sum().item()) if len(legals_count_cpu) > 0 else 0
    if mcts_policy_cpu.numel() != total_legals:
        if mcts_policy_cpu.numel() > total_legals:
            mcts_policy_cpu = mcts_policy_cpu[:total_legals]
        else:
            pad = torch.zeros((total_legals - mcts_policy_cpu.numel(),), dtype=torch.float32)
            mcts_policy_cpu = torch.cat([mcts_policy_cpu, pad], dim=0)
    total_samples = len(obs_list)
    if mcts_weight_cpu.numel() != total_samples:
        if mcts_weight_cpu.numel() > total_samples:
            mcts_weight_cpu = mcts_weight_cpu[:total_samples]
        else:
            padw = torch.zeros((total_samples - mcts_weight_cpu.numel(),), dtype=torch.float32)
            mcts_weight_cpu = torch.cat([mcts_weight_cpu, padw], dim=0)
    # Keep rewards/done on GPU for GAE already computed; store CPU too for logging if needed
    ret_t = ret_vec
    adv_t = adv_vec
    rew_t = rew_t
    done_t = torch.as_tensor(done_list, dtype=torch.bool, device=device) if len(done_list)>0 else torch.zeros((0,), dtype=torch.bool, device=device)

    # Calcola old_logp in batch per evitare sincronizzazioni step-by-step (transfer to CUDA once)
    if obs_cpu.size(0) > 0:
        with torch.no_grad():
            # Ensure source are CPU tensors before pinning
            def to_pinned(x):
                x_cpu = (x.detach().to('cpu') if torch.is_tensor(x) else torch.as_tensor(x))
                return x_cpu.pin_memory() if device.type == 'cuda' else x_cpu
            nb = (device.type == 'cuda')
            obs_t = to_pinned(obs_cpu).to(device=device, dtype=torch.float32, non_blocking=nb)
            seat_team_t = to_pinned(seat_team_cpu).to(device=device, non_blocking=nb)
            legals_t = to_pinned(legals_cpu).to(device=device, non_blocking=nb)
            legals_offset_t = to_pinned(legals_offset_cpu).to(device=device, non_blocking=nb)
            legals_count_t = to_pinned(legals_count_cpu).to(device=device, non_blocking=nb)
            chosen_index_t = to_pinned(chosen_index_cpu).to(device=device, non_blocking=nb)
            # belief summary may be absent
            # belief non più richiesto per calcolare old_logp
            # Two-stage old_logp
            B = obs_t.size(0)
            max_cnt = int(legals_count_t.max().item()) if B > 0 else 0
            if max_cnt > 0:
                state_proj = agent.actor.compute_state_proj(obs_t, seat_team_t)  # (B,64)
                card_logits_all = torch.matmul(state_proj, agent.actor.card_emb_play.t())  # (B,40)
                pos = torch.arange(max_cnt, device=device, dtype=torch.long)
                rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
                mask = rel_pos_2d < legals_count_t.unsqueeze(1)
                abs_idx = (legals_offset_t.unsqueeze(1) + rel_pos_2d)[mask]
                sample_idx_per_legal = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
                legals_mb = legals_t[abs_idx].contiguous()
                played_ids_mb = torch.argmax(legals_mb[:, :40], dim=1)
                # Card log-prob restricted to allowed cards only (two-stage policy)
                allowed_mask = torch.zeros((B, 40), dtype=torch.bool, device=device)
                allowed_mask[sample_idx_per_legal, played_ids_mb] = True
                neg_inf = torch.full_like(card_logits_all, float('-inf'))
                masked_logits = torch.where(allowed_mask, card_logits_all, neg_inf)
                max_allowed = torch.amax(masked_logits, dim=1)
                exp_shift_allowed = torch.exp(card_logits_all - max_allowed.unsqueeze(1)) * allowed_mask.to(card_logits_all.dtype)
                sum_allowed = exp_shift_allowed.sum(dim=1)
                lse_allowed = max_allowed + torch.log(torch.clamp_min(sum_allowed, 1e-12))
                chosen_clamped = torch.minimum(chosen_index_t, (legals_count_t - 1).clamp_min(0))
                chosen_abs = (legals_offset_t + chosen_clamped)
                total_legals = legals_t.size(0)
                pos_map = torch.full((total_legals,), -1, dtype=torch.long, device=device)
                pos_map[abs_idx] = torch.arange(abs_idx.numel(), device=device, dtype=torch.long)
                chosen_pos = pos_map[chosen_abs]
                # Validate chosen_pos mapping succeeded for all rows
                if bool((chosen_pos < 0).any().item() if chosen_pos.numel() > 0 else False):
                    bad_rows = torch.nonzero(chosen_pos < 0, as_tuple=False).flatten().tolist()
                    raise RuntimeError(f"collect_trajectory: chosen_pos mapping failed for rows {bad_rows}")
                played_ids_all = torch.argmax(legals_t[:, :40], dim=1)
                chosen_card_ids = played_ids_all[chosen_abs]
                logp_card = card_logits_all[torch.arange(B, device=device), chosen_card_ids] - lse_allowed[torch.arange(B, device=device)]
                # capture
                a_emb_mb = agent.actor.action_enc(legals_mb)
                cap_logits = (a_emb_mb * state_proj[sample_idx_per_legal]).sum(dim=1)
                group_ids = sample_idx_per_legal * 40 + played_ids_mb
                num_groups = B * 40
                group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
                group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)

                gmax_per_legal = group_max[group_ids]
                exp_shifted = torch.exp(cap_logits - gmax_per_legal)
                group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
                group_sum.index_add_(0, group_ids, exp_shifted)
                lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
                logp_cap_per_legal = cap_logits - lse_per_legal
                logp_cap = logp_cap_per_legal[chosen_pos]
                old_logp_t = (logp_card + logp_cap)
                # Early validity for old_logp
                if not torch.isfinite(old_logp_t).all():
                    raise RuntimeError("collect_trajectory: old_logp contains non-finite values")
                if bool((old_logp_t > 1e-6).any().item() if old_logp_t.numel() > 0 else False):
                    mx = float(old_logp_t.max().item())
                    raise RuntimeError(f"collect_trajectory: old_logp contains positive values (max={mx})")
                if bool((old_logp_t < -100.0).any().item() if old_logp_t.numel() > 0 else False):
                    idx_bad = torch.nonzero(old_logp_t < -100.0, as_tuple=False).flatten()
                    r = int(idx_bad[0].item()) if idx_bad.numel() > 0 else 0
                    # diagnostics for the first bad row
                    chosen_card = int(chosen_card_ids[r].item()) if chosen_card_ids.numel() > r else -1
                    cnt_r = int(legals_count_t[r].item()) if legals_count_t.numel() > r else -1
                    # group size for (r, chosen_card)
                    grp_mask = (sample_idx_per_legal == r) & (played_ids_mb == chosen_card)
                    grp_size = int(grp_mask.sum().item())
                    cap_logits_grp = cap_logits[grp_mask]
                    def _stats(t):
                        return {
                            'min': float(t.min().item()) if t.numel() > 0 else None,
                            'max': float(t.max().item()) if t.numel() > 0 else None,
                            'mean': float(t.mean().item()) if t.numel() > 0 else None,
                            'numel': int(t.numel())
                        }
                    raise RuntimeError(
                        f"collect_trajectory: old_logp extremely small (min={float(old_logp_t.min().item())}); "
                        f"row={r}, chosen_card={chosen_card}, legals_count={cnt_r}, group_size={grp_size}, "
                        f"logp_card={float(logp_card[r].item())}, logp_cap={float(logp_cap[r].item())}, "
                        f"card_logits_all_stats={_stats(card_logits_all[r])}, cap_logits_grp_stats={_stats(cap_logits_grp)}"
                    )
            else:
                old_logp_t = torch.zeros((B,), dtype=torch.float32, device=device)
    else:
        old_logp_t = torch.zeros((0,), dtype=torch.float32, device=device)

    ep_debug_lengths = [int(end - start) for (start, end) in ep_slices] if len(ep_slices) > 0 else None
    _maybe_log_ppo_batch(
        'serial',
        rew_t,
        ret_vec,
        adv_vec,
        val_t,
        nval_t,
        done_mask,
        old_logp=old_logp_t,
        seat_tensor=seat_team_cpu,
        episode_lengths=ep_debug_lengths,
        extra={'episodes': len(ep_slices)} if len(ep_slices) > 0 else None,
    )

    # Validate batch structure sizes/coherence before returning
    if not skip_step_validation:
        B = int(obs_cpu.size(0)) if torch.is_tensor(obs_cpu) else len(obs_cpu)
        def _len(x):
            return (int(x.size(0)) if torch.is_tensor(x) else len(x))
        # next_obs length equals obs length by construction
        if not (B == _len(act_cpu) == _len(done_t) == _len(seat_team_cpu)):
            raise RuntimeError("collect_trajectory: per-step arrays length mismatch among obs/act/next_obs/done/seat")
        if int(legals_count_cpu.sum().item()) != int(legals_cpu.size(0)):
            raise RuntimeError("collect_trajectory: sum(legals_count) != len(legals)")
        if (chosen_index_cpu < 0).any() or ((chosen_index_cpu >= legals_count_cpu) & (legals_count_cpu > 0)).any():
            raise RuntimeError("collect_trajectory: chosen_index out of range for some rows")
        if seat_team_cpu.size(1) != 6 or not (seat_team_cpu[:, :4].sum(dim=1) == 1).all():
            raise RuntimeError("collect_trajectory: seat_team must be (B,6) with one-hot seat")

    # Mantieni batch già tensori CPU; l'update ora li mapperà tutti su CUDA in un solo passaggio
    if _PAR_TIMING:
        total_collect = time.time() - t_collect_start
        episodes_total = len(ep_slices) if len(ep_slices) > 0 else (1 if len(obs_list) > 0 else 0)
        tqdm.write(
            f"[serial-collect] steps={len(obs_list)} episodes={episodes_total} resets={env_reset_count} "
            f"t_reset={t_env_reset:.3f}s t_obs={t_get_obs:.3f}s t_legals={t_get_legals:.3f}s "
            f"t_mcts={t_mcts:.3f}s t_select={t_select:.3f}s state={t_state_proj:.3f}s action_enc={t_action_enc:.3f}s sample={t_sampling:.3f}s t_step={t_step:.3f}s build={t_build:.3f}s "
            f"values={t_values_gae:.3f}s old_logp={t_oldlogp:.3f}s total={total_collect:.3f}s"
        )
        profiling_payload = {
            'serial/steps': len(obs_list),
            'serial/episodes': episodes_total,
            'serial/resets': env_reset_count,
            'serial/t_env_reset': t_env_reset,
            'serial/t_get_obs': t_get_obs,
            'serial/t_get_legals': t_get_legals,
            'serial/t_mcts': t_mcts,
            'serial/t_select': t_select,
            'serial/t_state_proj': t_state_proj,
            'serial/t_action_enc': t_action_enc,
            'serial/t_sampling': t_sampling,
            'serial/t_step': t_step,
            'serial/t_build': t_build,
            'serial/t_values_gae': t_values_gae,
            'serial/t_oldlogp': t_oldlogp,
            'serial/t_total': total_collect,
        }
        env_profile = {}
        consume_env = getattr(env, 'consume_profile_stats', None)
        if callable(consume_env):
            try:
                env_profile = consume_env() or {}
            except Exception as exc:
                tqdm.write(f"[serial-collect] env profiling failed: {exc}")
                env_profile = {}
        if env_profile:
            msg_parts = []
            step_count = env_profile.get('step/count', 0)
            if step_count:
                msg_parts.append(
                    f"step_total={env_profile.get('step/total', 0.0):.3f}s decode={env_profile.get('step/decode_total', 0.0):.3f}s "
                    f"validate={env_profile.get('step/validate_total', 0.0):.3f}s apply={env_profile.get('step/apply_total', 0.0):.3f}s avg={env_profile.get('step/total_avg', 0.0)*1000:.2f}ms"
                )
            leg_calls = env_profile.get('legals/calls', 0)
            if leg_calls:
                msg_parts.append(
                    f"legals_total={env_profile.get('legals/total', 0.0):.3f}s avg={env_profile.get('legals/avg', 0.0)*1000:.2f}ms"
                )
            if msg_parts:
                tqdm.write(f"[env-profile] {' '.join(msg_parts)}")
            profiling_payload.update({f'env/{k}': v for k, v in env_profile.items()})
    else:
        profiling_payload = None
        env_profile = {}

    batch = {
        'obs': obs_cpu,
        'act': act_cpu,
        # lascia tensori chiave su CUDA quando presenti ma evita copie inutili se già device-correct
        'old_logp': (old_logp_t if (torch.is_tensor(old_logp_t) and old_logp_t.device.type == device.type) else old_logp_t.detach()),
        'ret': (ret_t if (torch.is_tensor(ret_t) and ret_t.device.type == device.type) else ret_t.detach()),
        'adv': (adv_t if (torch.is_tensor(adv_t) and adv_t.device.type == device.type) else adv_t.detach()),
        'rew': rew_t,
        'done': done_t,
        'seat_team': seat_team_cpu,
        'belief_summary': belief_sum_cpu,
        'legals': legals_cpu,
        'legals_offset': legals_offset_cpu,
        'legals_count': legals_count_cpu,
        'chosen_index': chosen_index_cpu,
        'mcts_policy': mcts_policy_cpu,
        'mcts_weight': mcts_weight_cpu,
        'others_hands': others_hands_cpu,
        'routing_log': routing_log,
    }
    if profiling_payload is not None:
        batch.setdefault('profiling', {}).update(profiling_payload)
    elif env_profile:
        batch.setdefault('profiling', {}).update({f'env/{k}': v for k, v in env_profile.items()})
    return batch


def collect_trajectory(env: ScoponeEnvMA, agent: ActionConditionedPPO, horizon: int = 128,
                       gamma: float = 1.0, lam: float = 0.95,
                       partner_actor: ActionConditionedActor = None,
                       opponent_actor: ActionConditionedActor = None,
                       main_seats: List[int] = None,
                       belief_particles: int = 512, belief_ess_frac: float = 0.5,
                       episodes: int = None, final_reward_only: bool = True,
                       use_mcts: bool = True,
                       mcts_sims: int = 128, mcts_dets: int = 4, mcts_c_puct: float = 1.0,
                       mcts_root_temp: float = 0.0, mcts_prior_smooth_eps: float = 0.0,
                       mcts_dirichlet_alpha: float = 0.25, mcts_dirichlet_eps: float = 0.0,
                       mcts_train_factor: float = 1.0,
                       mcts_progress_start: float = 0.25,
                       mcts_progress_full: float = 0.75,
                       mcts_min_sims: int = 0,
                       train_both_teams: bool = False,
                       seed: Optional[int] = None) -> Dict:
    seed_token = _serial_seed_enter(seed)
    try:
        return _collect_trajectory_impl(env=env,
                                        agent=agent,
                                        horizon=horizon,
                                        gamma=gamma,
                                        lam=lam,
                                        partner_actor=partner_actor,
                                        opponent_actor=opponent_actor,
                                        main_seats=main_seats,
                                        belief_particles=belief_particles,
                                        belief_ess_frac=belief_ess_frac,
                                        episodes=episodes,
                                        final_reward_only=final_reward_only,
                                        use_mcts=use_mcts,
                                        mcts_sims=mcts_sims,
                                        mcts_dets=mcts_dets,
                                        mcts_c_puct=mcts_c_puct,
                                        mcts_root_temp=mcts_root_temp,
                                        mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                        mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                                        mcts_dirichlet_eps=mcts_dirichlet_eps,
                                        mcts_train_factor=mcts_train_factor,
                                        mcts_progress_start=mcts_progress_start,
                                        mcts_progress_full=mcts_progress_full,
                                        mcts_min_sims=mcts_min_sims,
                                        train_both_teams=train_both_teams)
    finally:
        _serial_seed_exit(seed_token)


def collect_trajectory_parallel(agent: ActionConditionedPPO,
                                num_envs: int = 32,
                                episodes_total_hint: int = 8,
                                k_history: int = 39,
                                gamma: float = 1.0,
                                lam: float = 0.95,
                                use_mcts: bool = True,
                                train_both_teams: bool = True,
                                main_seats: List[int] = None,
                                mcts_sims: int = 128,
                                mcts_dets: int = 4,
                                mcts_c_puct: float = 1.0,
                                mcts_root_temp: float = 0.0,
                                mcts_prior_smooth_eps: float = 0.0,
                                mcts_dirichlet_alpha: float = 0.25,
                                mcts_dirichlet_eps: float = 0.0,
                                mcts_progress_start: float = 0.25,
                                mcts_progress_full: float = 0.75,
                                mcts_min_sims: int = 0,
                                mcts_train_factor: float = 1.0,
                                seed: int = 0,
                                show_progress_env: bool = True,
                                tqdm_base_pos: int = 2,
                                frozen_actor: ActionConditionedActor = None,
                                frozen_non_main: bool = False) -> Dict:
    # Validate configuration invariants
    if int(num_envs) <= 0:
        raise ValueError("collect_trajectory_parallel: num_envs must be > 0")
    if int(episodes_total_hint) <= 0:
        raise ValueError("collect_trajectory_parallel: episodes_total_hint must be > 0")
    if int(k_history) <= 0:
        raise ValueError("collect_trajectory_parallel: k_history must be > 0")
    if float(gamma) < 0 or float(gamma) > 1:
        raise ValueError("collect_trajectory_parallel: gamma must be in [0,1]")
    if float(lam) < 0 or float(lam) > 1:
        raise ValueError("collect_trajectory_parallel: lam must be in [0,1]")
    if float(mcts_prior_smooth_eps) < 0 or float(mcts_prior_smooth_eps) > 1:
        raise ValueError("collect_trajectory_parallel: mcts_prior_smooth_eps must be in [0,1]")
    if float(mcts_dirichlet_eps) < 0 or float(mcts_dirichlet_eps) > 1:
        raise ValueError("collect_trajectory_parallel: mcts_dirichlet_eps must be in [0,1]")
    if float(mcts_dirichlet_alpha) < 0:
        raise ValueError("collect_trajectory_parallel: mcts_dirichlet_alpha must be >= 0")
    # Choose start method with env override and platform/device awareness.
    # - Windows: 'spawn'
    # - POSIX with CUDA or background threads: prefer 'forkserver' (avoid forking after CUDA/threads)
    # - Otherwise: 'fork'
    override = str(os.environ.get('SCOPONE_MP_START', '')).strip().lower()
    if override in {'spawn', 'fork', 'forkserver'}:
        start_method = override
    else:
        if platform.system().lower() == 'windows':
            start_method = 'spawn'
        else:
            # If CUDA is in use, prefer 'spawn' to avoid CUDA re-init issues in children
            if getattr(device, 'type', str(device)) == 'cuda' or (hasattr(torch, 'cuda') and torch.cuda.is_available()):
                start_method = 'spawn'
            else:
                start_method = 'fork'
    ctx = mp.get_context(start_method)
    request_q = ctx.Queue(maxsize=num_envs * 4)
    # Make episode queue large enough to avoid backpressure when multiple workers finish close together
    _episodes_hint = max(1, int(episodes_total_hint))
    episode_q = ctx.Queue(maxsize=max(num_envs * 4, _episodes_hint * 2))
    action_queues = [ctx.Queue(maxsize=2) for _ in range(num_envs)]
    # Distribute episodes exactly across workers without overshoot
    base = int(episodes_total_hint) // int(num_envs)
    rem = int(episodes_total_hint) % int(num_envs)
    episodes_per_env_list = [base + (1 if wid < rem else 0) for wid in range(num_envs)]
    total_eps = sum(episodes_per_env_list)
    tqdm.write(f"[collector] num_envs={num_envs} episodes_total_hint={episodes_total_hint} "
                   f"episodes_per_env_list(min..max)={min(episodes_per_env_list)}..{max(episodes_per_env_list)} total_env_episodes={total_eps}")
    workers = []
    cfg_base = {
        'rules': {'shape_scopa': False},
        'k_history': int(k_history),
        'send_legals': True,
        'use_mcts': bool(use_mcts),
        'train_both_teams': bool(train_both_teams),
        'main_seats': main_seats if main_seats is not None else [0,2],
        'frozen_non_main': bool(frozen_non_main),
        'mcts_sims': int(mcts_sims),
        'mcts_dets': int(mcts_dets),
        'mcts_c_puct': float(mcts_c_puct),
        'mcts_root_temp': float(mcts_root_temp),
        'mcts_prior_smooth_eps': float(mcts_prior_smooth_eps),
        'mcts_dirichlet_alpha': float(mcts_dirichlet_alpha),
        'mcts_dirichlet_eps': float(mcts_dirichlet_eps),
        'mcts_progress_start': float(mcts_progress_start),
        'mcts_progress_full': float(mcts_progress_full),
        'mcts_min_sims': int(mcts_min_sims),
        'mcts_train_factor': float(mcts_train_factor),
        'seed': int(seed),
    }
    for wid in range(num_envs):
        cfg = dict(cfg_base)
        cfg['episodes_per_env'] = int(episodes_per_env_list[wid])
        p = ctx.Process(target=_env_worker, args=(wid, cfg, request_q, action_queues[wid], episode_q), daemon=True)
        p.start()
        workers.append(p)

    episodes_received = 0
    episodes_payloads = []
    produced_count = [0 for _ in range(num_envs)]
    # Optional timing buffers
    timing_from_workers = [] if _PAR_TIMING else None
    t_drain = 0.0; t_get_reqs = 0.0; t_batch_select = 0.0; t_batch_service = 0.0; t_dispatch = 0.0
    # Fine-grained get_reqs profiling (new consolidated struct)
    getreqs_batches = 0
    getreqs_total_reqs = 0
    t_get_nowait = 0.0
    t_get_block = 0.0
    cnt_nowait_ok = 0
    cnt_nowait_empty = 0
    cnt_block_ok = 0
    cnt_block_timeout = 0
    # Extra timers for post-collector breakdown
    t_build = 0.0; t_values_gae = 0.0; t_oldlogp = 0.0; t_teardown = 0.0
    t_master_start = time.time() if _PAR_TIMING else 0.0
    # Batch size histogram and request type counters (active when profiling)
    bs_hist = {}
    step_reqs_count = 0
    other_reqs_count = 0
    # Optional per-env progress bars
    env_pbars = []
    _TQDM_DISABLE = (os.environ.get('TQDM_DISABLE','0') in ['1','true','yes','on'])
    # Respect global per-env tqdm policy via env; default ON unless explicitly disabled
    _PER_ENV_TQDM = (str(os.environ.get('SCOPONE_PER_ENV_TQDM', '1')).strip().lower() in ['1','true','yes','on'])
    # Hide per-env progress bars when debug logs are enabled to avoid clutter
    _SHOW_PB = bool(show_progress_env) and _PER_ENV_TQDM and (not _TQDM_DISABLE) and (not _PAR_DEBUG)
    if _SHOW_PB:
        for wid in range(num_envs):
            b = tqdm(total=int(episodes_per_env_list[wid]), desc=f"env {wid}", position=(tqdm_base_pos + wid), leave=False, dynamic_ncols=True, disable=_TQDM_DISABLE)
            env_pbars.append(b)
    else:
        env_pbars = [None] * num_envs
    # Main loop: drain episodes first (to avoid backpressure), then service requests
    _last_activity_ts = time.time()
    _expected_total = sum(episodes_per_env_list)
    _done_flags = [False] * num_envs
    while episodes_received < _expected_total:
        # 1) Drain any completed episodes first to free episode_q
        drained_any = False
        t0_drain = time.time() if _PAR_TIMING else 0.0
        while True:
            try:
                ep = episode_q.get_nowait()
                # Handle completion markers
                if isinstance(ep, dict) and ep.get('type') == 'done':
                    wid = int(ep.get('wid', -1))
                    if 0 <= wid < len(_done_flags):
                        _done_flags[wid] = True
                        expected_eps = episodes_per_env_list[wid] if 0 <= wid < len(episodes_per_env_list) else None
                        if (expected_eps is None) or (expected_eps > 0):
                            if produced_count[wid] == 0:
                                raise RuntimeError(f"collector: worker wid={wid} signaled done without producing any episode")
                        if 0 <= wid < len(env_pbars) and env_pbars[wid] is not None:
                            env_pbars[wid].close()
                            env_pbars[wid] = None
                    drained_any = True
                    continue
                elif _PAR_TIMING and isinstance(ep, dict) and ep.get('type') == 'timing':
                    timing_from_workers.append(ep)
                    drained_any = True
                else:
                    # validate episode payload
                    if (not isinstance(ep, dict)) or ('obs' not in ep) or (len(ep['obs']) == 0):
                        wid = (ep.get('wid') if isinstance(ep, dict) else 'unknown')
                        raise RuntimeError(f"collector: empty or invalid episode payload from wid={wid}")
                    episodes_payloads.append(ep)
                    episodes_received += 1
                    wid = int(ep.get('wid', -1))
                    if 0 <= wid < len(produced_count):
                        produced_count[wid] += 1
                    drained_any = True
                    # Update per-env progress bar
                    wid = int(ep.get('wid', -1))
                    if 0 <= wid < len(env_pbars) and env_pbars[wid] is not None:
                        env_pbars[wid].update(1)
            except queue.Empty:
                break
        if drained_any:
            _last_activity_ts = time.time()
        if _PAR_TIMING:
            t_drain += (time.time() - t0_drain)

        # 2) Gather a micro-batch of requests without blocking
        reqs = []
        min_batch = int(os.environ.get('SCOPONE_COLLECT_MIN_BATCH', '0'))
        if min_batch <= 0:
            min_batch = max(32, 2 * int(num_envs))
        batch_target = max(min_batch, 64)
        max_latency_ms = float(os.environ.get('SCOPONE_COLLECT_MAX_LATENCY_MS', '3.0'))
        max_latency_s = max(0.0, max_latency_ms / 1000.0)
        t0_get = time.time() if _PAR_TIMING else 0.0
        deadline = (t0_get + max_latency_s)
        # Block briefly for the first item (up to max_latency)
        try:
            t0_blk = time.time() if _PAR_TIMING else 0.0
            r0 = request_q.get(timeout=max_latency_s)
            reqs.append(r0)
            if _PAR_TIMING:
                cnt_block_ok += 1
                t_get_block += (time.time() - t0_blk)
                _GETREQS_PROF['cnt_first_ok'] += 1
                _GETREQS_PROF['t_first_block'] += (time.time() - t0_blk)
        except queue.Empty:
            if _PAR_TIMING:
                cnt_block_timeout += 1
                t_get_block += (time.time() - t0_blk)
                _GETREQS_PROF['cnt_first_to'] += 1
                _GETREQS_PROF['t_first_block'] += (time.time() - t0_blk)
            # No requests arrived within latency budget
            if _PAR_TIMING:
                t_get_reqs += (time.time() - t0_get)
            # Continue to next loop iteration
            continue
        # After first item, drain quickly without spinning
        t0_now = time.time() if _PAR_TIMING else 0.0
        while len(reqs) < batch_target:
            try:
                r = request_q.get_nowait()
                # If configured, label non-main requests for shadow (frozen) selection
                if bool(cfg_base.get('frozen_non_main', False)) and isinstance(r, dict) and r.get('type') == 'step':
                    seat = r.get('seat', None)
                    is_main = False
                    if seat is not None:
                        st = torch.as_tensor(seat, dtype=torch.float32)
                        seat_idx = int(torch.argmax(st[:4]).item())
                        main_set = set(cfg_base.get('main_seats', [0, 2]))
                        is_main = (seat_idx in main_set)
                    if not is_main:
                        r = dict(r)
                        r['type'] = 'step_shadow'
                reqs.append(r)
                if _PAR_TIMING:
                    cnt_nowait_ok += 1
                    _GETREQS_PROF['cnt_nowait_ok'] += 1
            except queue.Empty:
                if _PAR_TIMING:
                    cnt_nowait_empty += 1
                    _GETREQS_PROF['cnt_nowait_empty'] += 1
                break
        if _PAR_TIMING:
            dt_now = (time.time() - t0_now)
            t_get_nowait += dt_now
            _GETREQS_PROF['t_nowait_drain'] += dt_now
        # If under min_batch and we still have latency budget, do a short block to top up
        while (len(reqs) < min_batch) and (time.time() < deadline):
            try:
                remaining = max(0.0, deadline - time.time())
                t0_blk = time.time() if _PAR_TIMING else 0.0
                r = request_q.get(timeout=remaining)
                reqs.append(r)
                if _PAR_TIMING:
                    cnt_block_ok += 1
                    t_get_block += (time.time() - t0_blk)
                    _GETREQS_PROF['cnt_topup_ok'] += 1
                    _GETREQS_PROF['t_topup_block'] += (time.time() - t0_blk)
                # Opportunistically drain any burst
                t0_now = time.time() if _PAR_TIMING else 0.0
                while len(reqs) < batch_target:
                    try:
                        r2 = request_q.get_nowait()
                        reqs.append(r2)
                        if _PAR_TIMING:
                            cnt_nowait_ok += 1
                            _GETREQS_PROF['cnt_nowait_ok'] += 1
                    except queue.Empty:
                        if _PAR_TIMING:
                            cnt_nowait_empty += 1
                            _GETREQS_PROF['cnt_nowait_empty'] += 1
                        break
                if _PAR_TIMING:
                    dt_now2 = (time.time() - t0_now)
                    t_get_nowait += dt_now2
                    _GETREQS_PROF['t_post_drain'] += dt_now2
            except queue.Empty:
                if _PAR_TIMING:
                    cnt_block_timeout += 1
                    t_get_block += (time.time() - t0_blk)
                    _GETREQS_PROF['cnt_topup_to'] += 1
                    _GETREQS_PROF['t_topup_block'] += (time.time() - t0_blk)
                break
        if _PAR_TIMING:
            t_get_reqs += (time.time() - t0_get)
        # 3) Process batch on GPU
        if len(reqs) > 0:
            _last_activity_ts = time.time()
            if _PAR_TIMING:
                getreqs_batches += 1
                getreqs_total_reqs += len(reqs)
                _GETREQS_PROF['batches'] += 1
                _GETREQS_PROF['total_reqs'] += len(reqs)
                _GETREQS_PROF['bs_hist'][len(reqs)] = int(_GETREQS_PROF['bs_hist'].get(len(reqs), 0)) + 1
            if _PAR_TIMING:
                bs_hist[len(reqs)] = bs_hist.get(len(reqs), 0) + 1
            step_reqs = [r for r in reqs if r.get('type') == 'step']
            shadow_reqs = [r for r in reqs if r.get('type') == 'step_shadow']
            other_reqs = [r for r in reqs if r.get('type') != 'step']
            other_reqs = [r for r in other_reqs if r.get('type') != 'step_shadow']
            if _PAR_TIMING:
                step_reqs_count += len(step_reqs)
                other_reqs_count += len(other_reqs)
            if len(step_reqs) > 0:
                _t0 = time.time() if _PAR_TIMING else 0.0
                sel = _batched_select_indices(agent, step_reqs)
                if _PAR_TIMING:
                    t_batch_select += (time.time() - _t0)
                _t0d = time.time() if _PAR_TIMING else 0.0
                for (wid, idx) in sel:
                    if _PAR_DEBUG:
                        _dbg(f"[master] dispatch step wid={int(wid)} idx={int(idx)}")
                    action_queues[wid].put({'idx': int(idx)}, block=False)
                if _PAR_TIMING:
                    t_dispatch += (time.time() - _t0d)
            if len(shadow_reqs) > 0:
                # Use frozen actor to score non-main seats if provided
                if frozen_actor is None:
                    # Fallback: select with main actor if frozen not provided
                    _t0 = time.time() if _PAR_TIMING else 0.0
                    sel = _batched_select_indices(agent, shadow_reqs)
                    if _PAR_TIMING:
                        t_batch_select += (time.time() - _t0)
                else:
                    _t0 = time.time() if _PAR_TIMING else 0.0
                    sel = _batched_select_indices_with_actor(frozen_actor, shadow_reqs)
                    if _PAR_TIMING:
                        t_batch_select += (time.time() - _t0)
                _t0d = time.time() if _PAR_TIMING else 0.0
                for (wid, idx) in sel:
                    if _PAR_DEBUG:
                        _dbg(f"[master] dispatch shadow wid={int(wid)} idx={int(idx)}")
                    action_queues[wid].put({'idx': int(idx)}, block=False)
                if _PAR_TIMING:
                    t_dispatch += (time.time() - _t0d)
            if len(other_reqs) > 0:
                _t0 = time.time() if _PAR_TIMING else 0.0
                outs = _batched_service(agent, other_reqs)
                if _PAR_TIMING:
                    t_batch_service += (time.time() - _t0)
                _t0d = time.time() if _PAR_TIMING else 0.0
                for r, out in zip(other_reqs, outs):
                    wid = int(r.get('wid', 0))
                    if _PAR_DEBUG:
                        _dbg(f"[master] dispatch {r.get('type')} wid={int(wid)} keys={list(out.keys())}")
                    action_queues[wid].put(out, block=False)
                if _PAR_TIMING:
                    t_dispatch += (time.time() - _t0d)

        # 4) Detect worker crashes and stale state
        for _wid, _p in enumerate(workers):
            if (not _p.is_alive()) and (_p.exitcode is not None) and (int(_p.exitcode) != 0):
                raise RuntimeError(f"Env worker wid={_wid} pid={_p.pid} crashed with exitcode={_p.exitcode}")
        # 5) Watchdog: if no activity for long, attempt graceful shutdown based on done flags
        if (time.time() - _last_activity_ts) > float(os.environ.get('SCOPONE_COLLECTOR_STALL_S', '30')):
            # If all workers have signaled 'done', break even if some episodes were dropped
            if all(_done_flags):
                break
            alive = [(i, p.pid, p.is_alive(), p.exitcode) for i, p in enumerate(workers)]
            raise RuntimeError(f"Collector stalled: episodes_received={episodes_received}/{_expected_total}; workers={alive}")

    # Close per-env progress bars
    _t_td0 = time.time() if _PAR_TIMING else 0.0
    for b in env_pbars:
        if b is not None:
            b.close()
    # Join workers, then terminate any stragglers and close queues
    for p in workers:
        p.join(timeout=0.1)
    # If any worker exited with non-zero code, surface it now
    for _wid, _p in enumerate(workers):
        if (_p.exitcode is not None) and (int(_p.exitcode) != 0):
            raise RuntimeError(f"Env worker wid={_wid} pid={_p.pid} exited with exitcode={_p.exitcode}")
    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.1)
    # Close queues to free resources
    request_q.close(); request_q.join_thread()
    episode_q.close(); episode_q.join_thread()
    for q in action_queues:
        q.close(); q.join_thread()
    if _PAR_TIMING:
        t_teardown = (time.time() - _t_td0)

    # If no episodes were produced, raise with context (no fallback)
    if len(episodes_payloads) == 0:
        raise RuntimeError(f"collector: No episodes produced (num_envs={num_envs}, episodes_total_hint={episodes_total_hint}, use_mcts={use_mcts}, mcts_sims={mcts_sims}, mcts_dets={mcts_dets})")

    # End-of-collection invariant: each episode must contribute exactly E transitions
    # E = 40 if training both teams, else 20 (main seats only)
    E = 40 if bool(train_both_teams) else 20
    episode_lengths = [len(ep.get('obs', [])) for ep in episodes_payloads]
    episodes = len(episodes_payloads)
    total_steps = sum(episode_lengths)
    # Compute per-seat counts over the whole collection for diagnostics
    import torch as _t
    if episodes > 0 and len(episodes_payloads[0].get('seat', [])) > 0:
        seats_all = []
        for ep in episodes_payloads:
            seats_all.extend(ep.get('seat', []))
        st = _t.stack([_t.as_tensor(x, dtype=_t.float32) for x in seats_all], dim=0)
        seat_idx = _t.argmax(st[:, :4], dim=1)
        c0 = int((seat_idx == 0).sum().item())
        c1 = int((seat_idx == 1).sum().item())
        c2 = int((seat_idx == 2).sum().item())
        c3 = int((seat_idx == 3).sum().item())
        seat_counts = (c0, c1, c2, c3)
    else:
        seat_counts = (0, 0, 0, 0)
    if total_steps != episodes * E:
        # Try alternate E (20<->40) to accommodate upstream per-episode util choice
        alt_E = (40 if E == 20 else 20)
        if total_steps == episodes * alt_E:
            # Adopt observed per-episode util and continue with a warning
            from tqdm import tqdm as _tqdm
            _tqdm.write(
                f"[collector] Adjusting per-episode util from {E} to observed {alt_E} (episodes={episodes}, total_steps={total_steps})"
            )
            E = alt_E
        else:
            # Build compact stats of episode lengths
            lengths_unique = sorted(set(episode_lengths))
            # frequency per length
            freq = {L: episode_lengths.count(L) for L in lengths_unique}
            raise RuntimeError(
                f"collector: util transitions mismatch; got_total={total_steps}, expected_total={episodes * E}; "
                f"episodes_done={episodes}, per_ep_expected={E}, length_stats={freq}, seat_counts={seat_counts}"
            )

    # Additional guard: ensure we produced exactly the requested number of episodes overall
    if episodes != int(episodes_total_hint):
        raise RuntimeError(
            f"collector: episodes produced {episodes} != requested {int(episodes_total_hint)} (num_envs={num_envs}, "
            f"episodes_per_env_list={episodes_per_env_list}, produced_per_worker={produced_count})"
        )
    # Print timing after validation
    if _PAR_TIMING:
        master_total = (time.time() - t_master_start)
        import numpy as _np
        if timing_from_workers is not None and len(timing_from_workers) > 0:
            def _agg(key):
                arr = _np.asarray([x.get(key, 0.0) for x in timing_from_workers], dtype=_np.float64)
                return float(arr.sum()), float(arr.mean()), float(arr.max())
            s_env, m_env, M_env = _agg('t_env_reset')
            s_obs, m_obs, M_obs = _agg('t_get_obs')
            s_leg, m_leg, M_leg = _agg('t_get_legals')
            s_mcts, m_mcts, M_mcts = _agg('t_mcts')
            s_rpc, m_rpc, M_rpc = _agg('t_rpc')
            s_step, m_step, M_step = _agg('t_step')
            s_pack, m_pack, M_pack = _agg('t_pack')
            s_tot, m_tot, M_tot = _agg('t_total')
            tqdm.write(f"[par-timing] workers={num_envs} workers_total={s_tot:.3f}s (avg={m_tot:.3f}, max={M_tot:.3f})\n"
                        f"  env.reset={s_env:.3f} get_obs={s_obs:.3f} get_legals={s_leg:.3f} rpc_wait={s_rpc:.3f} mcts={s_mcts:.3f} env.step={s_step:.3f} pack={s_pack:.3f}")
        tqdm.write(f"[par-timing-master] total={master_total:.3f}s drain={t_drain:.3f} get_reqs={t_get_reqs:.3f} batch_select={t_batch_select:.3f} batch_service={t_batch_service:.3f} dispatch={t_dispatch:.3f}")
        # Detailed get_reqs breakdown
        avg_batch = (getreqs_total_reqs / max(getreqs_batches, 1))
        tqdm.write(f"[par-getreqs] batches={getreqs_batches} total_reqs={getreqs_total_reqs} nowait_ok={cnt_nowait_ok} nowait_empty={cnt_nowait_empty} block_ok={cnt_block_ok} block_to={cnt_block_timeout} t_nowait={t_get_nowait:.3f} t_block={t_get_block:.3f} avg_batch={avg_batch:.2f}")
        # New sub-breakdown for get_reqs gather loop
        if _PAR_TIMING:
            b = max(1, int(_GETREQS_PROF['batches']))
            avg_b = float(_GETREQS_PROF['total_reqs']) / float(b)
            top_bins = sorted(_GETREQS_PROF['bs_hist'].items(), key=lambda kv: kv[1], reverse=True)[:8]
            bins_str = ' '.join([f"{k}x{v}" for k, v in top_bins]) if len(top_bins) > 0 else ''
            tqdm.write(
                f"[par-getreqs-sub] batches={_GETREQS_PROF['batches']} total_reqs={_GETREQS_PROF['total_reqs']} avg_batch={avg_b:.2f}\n"
                f"  first_block={_GETREQS_PROF['t_first_block']:.3f}s(now={_GETREQS_PROF['cnt_first_ok']} to={_GETREQS_PROF['cnt_first_to']}) "
                f"nowait_drain={_GETREQS_PROF['t_nowait_drain']:.3f}s(ok={_GETREQS_PROF['cnt_nowait_ok']} empty={_GETREQS_PROF['cnt_nowait_empty']})\n"
                f"  topup_block={_GETREQS_PROF['t_topup_block']:.3f}s(ok={_GETREQS_PROF['cnt_topup_ok']} to={_GETREQS_PROF['cnt_topup_to']}) post_drain={_GETREQS_PROF['t_post_drain']:.3f}s\n"
                f"  bs_top_bins={bins_str}"
            )
        if _PAR_TIMING:
            # Batch size histogram summary (top bins)
            if len(bs_hist) > 0:
                import math as _math
                tot_batches = int(sum(bs_hist.values()))
                top_bins = sorted(bs_hist.items(), key=lambda kv: kv[1], reverse=True)[:8]
                bins_str = ' '.join([f"{k}x{v}" for k,v in top_bins])
                tqdm.write(f"[par-getreqs-hist] batches={tot_batches} step_reqs={step_reqs_count} other_reqs={other_reqs_count} top_bins={bins_str}")
            # Sub-timers for batched_select
            if _BATCHSEL_PROF['count'] > 0:
                c = max(1, int(_BATCHSEL_PROF['count']))
                avg_B = _BATCHSEL_PROF['sum_B'] / c
                avg_M = _BATCHSEL_PROF['sum_M'] / c
                top_cnt = sorted(_BATCHSEL_PROF['cnt_hist'].items(), key=lambda kv: kv[1], reverse=True)[:8]
                cnt_str = ' '.join([f"{k}x{v}" for k,v in top_cnt]) if len(top_cnt) > 0 else ''
                tqdm.write(
                    f"[par-batch_select-sub] calls={_BATCHSEL_PROF['count']} total={_BATCHSEL_PROF['t_total']:.3f}s avg_B={avg_B:.1f} avg_M={avg_M:.1f}\n"
                    f"  stack={_BATCHSEL_PROF['t_stack']:.3f}s leg_stack={_BATCHSEL_PROF['t_leg_stack']:.3f}s "
                    f"state_proj={_BATCHSEL_PROF['t_state_proj']:.3f}s action_enc={_BATCHSEL_PROF['t_action_enc']:.3f}s "
                    f"mask={_BATCHSEL_PROF['t_mask']:.3f}s score={_BATCHSEL_PROF['t_score']:.3f}s softmax_sample={_BATCHSEL_PROF['t_softmax_sample']:.3f}s\n"
                    f"  cnt_max_hist={cnt_str}")
            # Actor.compute_state_proj internal breakdown (if available)
            from utils.prof import snapshot_actor_stateproj, reset_actor_stateproj
            _sp = snapshot_actor_stateproj()
            if int(_sp.get('count', 0)) > 0:
                tqdm.write(
                    f"[par-state_proj-sub] calls={int(_sp['count'])} state_enc={_sp['t_state_enc']:.3f}s "
                    f"belief_logits={_sp['t_belief_logits']:.3f}s belief_probs={_sp['t_belief_probs']:.3f}s "
                    f"gates={_sp['t_belief_gates']:.3f}s merge={_sp['t_merge']:.3f}s proj={_sp['t_proj']:.3f}s"
                )
            reset_actor_stateproj()
        tqdm.write(f"[par-collect-post] build={t_build:.3f} values_gae={t_values_gae:.3f} old_logp={t_oldlogp:.3f} teardown={t_teardown:.3f}")

    # Build batch CPU tensors from payloads
    _t_build0 = time.time() if _PAR_TIMING else 0.0
    obs_cpu = []
    act_cpu = []
    next_obs_cpu = []
    rew_list = []
    done_list = []
    seat_cpu = []
    belief_cpu = []
    legals_cpu = []
    leg_off = []
    leg_cnt = []
    chosen_idx = []
    for ep in episodes_payloads:
        # Incoming payload is NumPy; convert lazily per-sample
        obs_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['obs']]
        next_obs_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['next_obs']]
        act_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['act']]
        seat_ep = [torch.as_tensor(x, dtype=torch.float32) for x in ep['seat']]
        # Determina reward episodio dal team 0/1 (proporzionale al risultato)
        tr = ep.get('team_rewards', [0.0, 0.0])
        t0 = float(tr[0])
        t1 = float(tr[1])
        rew_ep = []
        for st in seat_ep:
            team0_flag = bool(float(st[4]) > 0.5)
            if not team0_flag:
                seat_idx = int(torch.argmax(st[:4]).item())
                team0_flag = seat_idx in [0, 2]
            rew_ep.append(t0 if team0_flag else t1)

        obs_cpu.extend(obs_ep)
        next_obs_cpu.extend(next_obs_ep)
        act_cpu.extend(act_ep)
        rew_list.extend(rew_ep)
        done_list.extend(ep['done'])
        seat_cpu.extend(seat_ep)
        belief_cpu.extend([torch.as_tensor(x, dtype=torch.float32) for x in ep['belief_summary']])
        # Adjust offsets by current length of legals_cpu
        base = len(legals_cpu)
        leg_off.extend([base + off for off in ep['leg_off']])
        leg_cnt.extend(ep['leg_cnt'])
        legals_cpu.extend([torch.as_tensor(x, dtype=torch.float32) for x in ep['legals']])
        chosen_idx.extend(ep['chosen_idx'])
        # distill/belief aux
        if 'mcts_policy' in ep:
            # adjust offsets later after concat
            pass

    # Validate per-episode structure coherence before stacking
    if not (len(obs_cpu) == len(act_cpu) == len(next_obs_cpu) == len(done_list) == len(seat_cpu)):
        raise RuntimeError("collector: per-step arrays length mismatch among obs/act/next_obs/done/seat")
    # Stack to tensors
    _obs_dim = int(episodes_payloads[0]['obs'][0].__len__()) if len(episodes_payloads) > 0 and len(episodes_payloads[0]['obs']) > 0 else 1
    obs_cpu_t = torch.stack(obs_cpu, dim=0) if len(obs_cpu) > 0 else torch.zeros((0, _obs_dim), dtype=torch.float32)
    next_obs_cpu_t = torch.stack(next_obs_cpu, dim=0) if len(next_obs_cpu) > 0 else torch.zeros_like(obs_cpu_t)
    act_cpu_t = torch.stack(act_cpu, dim=0) if len(act_cpu) > 0 else torch.zeros((0, 80), dtype=torch.float32)
    seat_cpu_t = torch.stack(seat_cpu, dim=0) if len(seat_cpu) > 0 else torch.zeros((0, 6), dtype=torch.float32)
    belief_cpu_t = torch.stack(belief_cpu, dim=0) if len(belief_cpu) > 0 else torch.zeros((0, 120), dtype=torch.float32)
    legals_cpu_t = torch.stack(legals_cpu, dim=0) if len(legals_cpu) > 0 else torch.zeros((0, 80), dtype=torch.float32)
    leg_off_t = torch.as_tensor(leg_off, dtype=torch.long)
    leg_cnt_t = torch.as_tensor(leg_cnt, dtype=torch.long)
    chosen_idx_t = torch.as_tensor(chosen_idx, dtype=torch.long)
    # Distill/belief aux targets from episodes
    # Ricostruisci mcts_policy_flat per-sample mantenendo allineamento con leg_cnt/leg_off
    mcts_policy_flat = []
    mcts_weight = []
    others_hands = []
    for ep in episodes_payloads:
        cnts_ep = ep['leg_cnt']
        # policy vector per episodio: concatena, inserendo zeri per sample senza MCTS
        mcts_policy_ep = ep.get('mcts_policy', [])
        mcts_weight_ep = ep.get('mcts_weight', [0.0]*len(ep.get('obs', [])))
        # Sanity: se mancano, crea zeri del giusto totale
        total_legals_ep = sum(cnts_ep) if isinstance(cnts_ep, list) else int(torch.as_tensor(cnts_ep).sum().item())
        if len(mcts_policy_ep) == 0:
            mcts_policy_ep = [0.0] * total_legals_ep
        if len(mcts_weight_ep) == 0:
            mcts_weight_ep = [0.0] * len(ep.get('obs', []))
        # Append
        mcts_policy_flat.extend(mcts_policy_ep)
        mcts_weight.extend(mcts_weight_ep)
        if 'others_hands' in ep:
            for oh in ep['others_hands']:
                others_hands.append(torch.as_tensor(oh, dtype=torch.float32))
    mcts_policy_t = torch.as_tensor(mcts_policy_flat, dtype=torch.float32) if len(mcts_policy_flat)>0 else torch.zeros((0,), dtype=torch.float32)
    mcts_weight_t = torch.as_tensor(mcts_weight, dtype=torch.float32) if len(mcts_weight)>0 else torch.zeros((0,), dtype=torch.float32)
    others_hands_t = torch.stack(others_hands, dim=0) if len(others_hands)>0 else torch.zeros((0,3,40), dtype=torch.float32)

    # End build
    if _PAR_TIMING:
        t_build = (time.time() - _t_build0)

    # Validate ragged legals structure before computing values
    total_legals = len(legals_cpu)
    if int(torch.as_tensor(leg_cnt, dtype=torch.long).sum().item()) != total_legals:
        raise RuntimeError("collector: sum(legals_count) != len(legals)")
    # chosen indices must be within per-row counts
    for i, (off_i, cnt_i, ch_i) in enumerate(zip(leg_off, leg_cnt, chosen_idx)):
        if cnt_i <= 0:
            continue
        if not (0 <= int(ch_i) < int(cnt_i)):
            raise RuntimeError(f"collector: chosen_index out of range at row {i} (idx={int(ch_i)}, cnt={int(cnt_i)})")
    # seat_team sanity: shape and one-hot seat
    if len(seat_cpu) > 0:
        seat_mat = torch.stack(seat_cpu, dim=0)
        if seat_mat.size(1) != 6:
            raise RuntimeError("collector: seat_team must have shape (B,6)")
        if not (seat_mat[:, :4].sum(dim=1) == 1).all():
            raise RuntimeError("collector: seat one-hot invalid (sum != 1)")
    # Compute values and advantages on GPU similar to collect_trajectory (CTDE coerente)
    rew_t = torch.as_tensor(rew_list, dtype=torch.float32, device=device) if len(rew_list) > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
    done_mask = torch.as_tensor([0.0 if not d else 1.0 for d in done_list], dtype=torch.float32, device=device) if len(done_list) > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
    _t_val0 = time.time() if _PAR_TIMING else 0.0
    if len(rew_list) > 0:
        with torch.no_grad():
            if device.type == 'cuda':
                o_all = obs_cpu_t.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
                s_all = seat_cpu_t.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                o_all = obs_cpu_t.to(device=device, dtype=torch.float32)
                s_all = seat_cpu_t.to(device=device, dtype=torch.float32)
            # others_hands per-step (CTDE): se raccolti dagli env worker, usa quelli
            if others_hands_t.numel() > 0 and others_hands_t.size(0) == o_all.size(0):
                oh_all = (others_hands_t.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True) if device.type == 'cuda' else others_hands_t.to(device=device, dtype=torch.float32))
            else:
                oh_all = None
            val_t = agent.critic(o_all, s_all, oh_all)
            n_all = (next_obs_cpu_t.pin_memory().to(device=device, dtype=torch.float32, non_blocking=True) if device.type == 'cuda' else next_obs_cpu_t.to(device=device, dtype=torch.float32))
            # costruisci others_hands next (shift) se disponibile
            if oh_all is not None:
                oh_next = torch.zeros_like(oh_all)
                if oh_all.size(0) > 1:
                    oh_next[:-1] = oh_all[1:]
            else:
                oh_next = None
            nval_t = agent.critic(n_all, s_all, oh_next)
            nval_t = torch.where(done_mask.bool(), torch.zeros_like(nval_t), nval_t)
    else:
        val_t = torch.zeros((0,), dtype=torch.float32, device=device)
        nval_t = torch.zeros((0,), dtype=torch.float32, device=device)
    adv_vec = torch.zeros_like(rew_t)
    gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    T = int(rew_t.numel())
    for t in reversed(range(T)):
        delta = rew_t[t] + nval_t[t] * gamma - val_t[t]
        gae = delta + gamma * lam * (1.0 - done_mask[t]) * gae
        adv_vec[t] = gae
    ret_vec = adv_vec + val_t
    if _PAR_TIMING:
        t_values_gae = (time.time() - _t_val0)

    # Compute old_logp in batch on GPU (factored card + capture scheme)
    _t_olp0 = time.time() if _PAR_TIMING else 0.0
    if obs_cpu_t.size(0) > 0:
        with torch.no_grad():
            def to_pinned(x):
                x_cpu = (x.detach().to('cpu') if torch.is_tensor(x) else torch.as_tensor(x))
                return x_cpu.pin_memory() if device.type == 'cuda' else x_cpu
            nb = (device.type == 'cuda')
            obs_t = to_pinned(obs_cpu_t).to(device=device, dtype=torch.float32, non_blocking=nb)
            seat_t = to_pinned(seat_cpu_t).to(device=device, non_blocking=nb)
            leg_t = to_pinned(legals_cpu_t).to(device=device, non_blocking=nb)
            offs = (leg_off_t.pin_memory().to(device=device, non_blocking=nb) if device.type == 'cuda' else leg_off_t.to(device=device))
            cnts = (leg_cnt_t.pin_memory().to(device=device, non_blocking=nb) if device.type == 'cuda' else leg_cnt_t.to(device=device))
            B = obs_t.size(0)
            max_cnt = int(cnts.max().item()) if B > 0 else 0
            if max_cnt > 0:
                # State projection and card logits
                state_proj = agent.actor.compute_state_proj(obs_t, seat_t)  # (B,64)
                card_logits_all = torch.matmul(state_proj, agent.actor.card_emb_play.t())  # (B,40)
                pos = torch.arange(max_cnt, device=device, dtype=torch.long)
                rel_pos_2d = pos.unsqueeze(0).expand(B, max_cnt)
                mask = rel_pos_2d < cnts.unsqueeze(1)
                abs_idx = (offs.unsqueeze(1) + rel_pos_2d)[mask]
                sample_idx = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, max_cnt)[mask]
                legals_mb = leg_t[abs_idx].contiguous()                   # (M_mb,80)
                played_ids_mb = torch.argmax(legals_mb[:, :40], dim=1)    # (M_mb)
                # Card log-prob restricted to allowed cards only (two-stage policy)
                allowed_mask = torch.zeros((B, 40), dtype=torch.bool, device=device)
                allowed_mask[sample_idx, played_ids_mb] = True
                neg_inf = torch.full_like(card_logits_all, float('-inf'))
                masked_logits = torch.where(allowed_mask, card_logits_all, neg_inf)
                max_allowed = torch.amax(masked_logits, dim=1)
                exp_shift_allowed = torch.exp(card_logits_all - max_allowed.unsqueeze(1)) * allowed_mask.to(card_logits_all.dtype)
                sum_allowed = exp_shift_allowed.sum(dim=1)
                lse_allowed = max_allowed + torch.log(torch.clamp_min(sum_allowed, 1e-12))
                # Map chosen indices to absolute and card ids
                chosen_clamped = (torch.minimum(chosen_idx_t.pin_memory().to(device=device), (cnts - 1).clamp_min(0)) if device.type == 'cuda' else torch.minimum(chosen_idx_t.to(device=device), (cnts - 1).clamp_min(0)))
                chosen_abs = (offs + chosen_clamped)
                total_legals = leg_t.size(0)
                pos_map = torch.full((total_legals,), -1, dtype=torch.long, device=device)
                pos_map[abs_idx] = torch.arange(abs_idx.numel(), device=device, dtype=torch.long)
                chosen_pos = pos_map[chosen_abs]
                # Validate chosen_pos mapping succeeded for all rows
                if bool((chosen_pos < 0).any().item() if chosen_pos.numel() > 0 else False):
                    bad_rows = torch.nonzero(chosen_pos < 0, as_tuple=False).flatten().tolist()
                    raise RuntimeError(f"collect_trajectory_parallel: chosen_pos mapping failed for rows {bad_rows}")
                played_ids_all = torch.argmax(leg_t[:, :40], dim=1)
                chosen_card_ids = played_ids_all[chosen_abs]
                logp_card = card_logits_all[torch.arange(B, device=device), chosen_card_ids] - lse_allowed[torch.arange(B, device=device)]
                # Capture log-softmax within card groups
                a_emb_mb = agent.actor.action_enc(legals_mb)               # (M_mb,64)
                cap_logits = (a_emb_mb * state_proj[sample_idx]).sum(dim=1)
                group_ids = sample_idx * 40 + played_ids_mb
                num_groups = B * 40
                group_max = torch.full((num_groups,), float('-inf'), dtype=cap_logits.dtype, device=device)
                group_max.scatter_reduce_(0, group_ids, cap_logits, reduce='amax', include_self=True)
                gmax_per_legal = group_max[group_ids]
                exp_shifted = torch.exp(cap_logits - gmax_per_legal)
                group_sum = torch.zeros((num_groups,), dtype=cap_logits.dtype, device=device)
                group_sum.index_add_(0, group_ids, exp_shifted)
                lse_per_legal = gmax_per_legal + torch.log(torch.clamp_min(group_sum[group_ids], 1e-12))
                logp_cap_per_legal = cap_logits - lse_per_legal
                logp_cap = logp_cap_per_legal[chosen_pos]
                old_logp_t = (logp_card + logp_cap)
                # Early validity for old_logp
                if not torch.isfinite(old_logp_t).all():
                    raise RuntimeError("collect_trajectory_parallel: old_logp contains non-finite values")
                if bool((old_logp_t > 1e-6).any().item() if old_logp_t.numel() > 0 else False):
                    mx = float(old_logp_t.max().item())
                    raise RuntimeError(f"collect_trajectory_parallel: old_logp contains positive values (max={mx})")
                if bool((old_logp_t < -100.0).any().item() if old_logp_t.numel() > 0 else False):
                    idx_bad = torch.nonzero(old_logp_t < -100.0, as_tuple=False).flatten()
                    r = int(idx_bad[0].item()) if idx_bad.numel() > 0 else 0
                    chosen_card = int(chosen_card_ids[r].item()) if chosen_card_ids.numel() > r else -1
                    cnt_r = int(cnts[r].item()) if cnts.numel() > r else -1
                    grp_mask = (sample_idx == r) & (played_ids_mb == chosen_card)
                    cap_logits_grp = cap_logits[grp_mask]
                    def _stats(t):
                        return {
                            'min': float(t.min().item()) if t.numel() > 0 else None,
                            'max': float(t.max().item()) if t.numel() > 0 else None,
                            'mean': float(t.mean().item()) if t.numel() > 0 else None,
                            'numel': int(t.numel())
                        }
                    raise RuntimeError(
                        f"collect_trajectory_parallel: old_logp extremely small (min={float(old_logp_t.min().item())}); "
                        f"row={r}, chosen_card={chosen_card}, legals_count={cnt_r}, "
                        f"logp_card={float(logp_card[r].item())}, logp_cap={float(logp_cap[r].item())}, "
                        f"card_logits_all_stats={_stats(card_logits_all[r])}, cap_logits_grp_stats={_stats(cap_logits_grp)}"
                    )
            else:
                old_logp_t = torch.zeros((B,), dtype=torch.float32, device=device)
    else:
        old_logp_t = torch.zeros((0,), dtype=torch.float32, device=device)
    if _PAR_TIMING:
        t_oldlogp = (time.time() - _t_olp0)

    ep_debug_lengths = episode_lengths if len(episode_lengths) > 0 else None
    _maybe_log_ppo_batch(
        'parallel',
        rew_t,
        ret_vec,
        adv_vec,
        val_t,
        nval_t,
        done_mask,
        old_logp=old_logp_t,
        seat_tensor=seat_cpu_t,
        episode_lengths=ep_debug_lengths,
        extra={'episodes': episodes},
    )

    batch = {
        'obs': obs_cpu_t,
        'act': act_cpu_t,
        # Mantieni tensori chiave su CUDA per evitare D2H→H2D
        'old_logp': old_logp_t.detach(),
        'ret': ret_vec.detach(),
        'adv': adv_vec.detach(),
        'rew': rew_t,
        'done': torch.as_tensor(done_list, dtype=torch.bool, device=device) if len(done_list)>0 else torch.zeros((0,), dtype=torch.bool, device=device),
        'seat_team': seat_cpu_t,
        'belief_summary': belief_cpu_t,
        'legals': legals_cpu_t,
        'legals_offset': leg_off_t,
        'legals_count': leg_cnt_t,
        'chosen_index': chosen_idx_t,
        'mcts_policy': mcts_policy_t,
        'mcts_weight': mcts_weight_t,
        'others_hands': others_hands_t,
        'routing_log': [('parallel', 'main')],
    }
    return batch


def train_ppo(num_iterations: int = 1000, horizon: int = 256, save_every: int = 10, ckpt_path: str = 'checkpoints/ppo_ac.pth', k_history: int = 39, seed: int = 0,
              entropy_schedule_type: str = 'linear', eval_every: int = 0, eval_games: int = 10, belief_particles: int = 512, belief_ess_frac: float = 0.5,
              mcts_in_eval: bool = True, mcts_train: bool = True, mcts_sims: int = 128, mcts_sims_eval: Optional[int] = None, mcts_dets: int = 4, mcts_c_puct: float = 1.0, mcts_root_temp: float = 0.0,
              mcts_prior_smooth_eps: float = 0.0, mcts_dirichlet_alpha: float = 0.25, mcts_dirichlet_eps: float = 0.25,
              num_envs: int = 32,
              train_both_teams: bool = True,
              use_selfplay: bool = True,
              mcts_warmup_iters: Optional[int] = 500,
              on_iter_end: Optional[Callable[[int], None]] = None):
    # Enforce minimum horizon of 40 and align horizon to minibatch size
    horizon = max(40, int(horizon))
    # Read minibatch size once and align horizon to LCM(minibatch_size, per-episode useful transitions)
    import os as _os
    import math as _math
    minibatch_size = 4096
    _mb_env = int(_os.environ.get('SCOPONE_MINIBATCH', str(minibatch_size)))
    if _mb_env > 0:
        minibatch_size = _mb_env
    # Determine effective per-episode useful transitions based on desired collection semantics
    _tfb_env = str(_os.environ.get('SCOPONE_TRAIN_FROM_BOTH_TEAMS', '0')).strip().lower() in ['1','true','yes','on']
    _frozen_env = str(_os.environ.get('SCOPONE_OPP_FROZEN', '1')).strip().lower() in ['1','true','yes','on']
    # In selfplay, collecting from both teams only makes sense when opponent is not frozen
    _collect_both = (bool(use_selfplay) and _tfb_env and (not _frozen_env))
    _per_ep_util = (40 if _collect_both else 20)
    lcm_mb_ep = (abs(minibatch_size * _per_ep_util) // _math.gcd(minibatch_size, _per_ep_util)) if (minibatch_size > 0 and _per_ep_util > 0) else max(minibatch_size, _per_ep_util)
    if lcm_mb_ep > 0 and (horizon % lcm_mb_ep) != 0:
        new_h = ((horizon + lcm_mb_ep - 1) // lcm_mb_ep) * lcm_mb_ep
        tqdm.write(f"[horizon] adjusted to LCM(mb={minibatch_size}, per_ep={_per_ep_util})={lcm_mb_ep}: {horizon} -> {new_h}")
        horizon = new_h
    # Resolve and announce final seed (seed<0 => random per run)
    seed = resolve_seed(seed)
    set_global_seeds(seed)
    tqdm.write(f"[seed] Using seed={seed}")
    # Disattiva reward shaping intermedio: solo reward finale
    env = ScoponeEnvMA(rules={'shape_scopa': False}, k_history=k_history)
    obs_dim = env.observation_space.shape[0]
    # Passa k_history al modello per evitare inferenze fragili di k
    agent = ActionConditionedPPO(obs_dim=obs_dim, k_history=k_history)
    # Allow big-compute-on-GPU during update while env/collection stays on CPU.
    # Controlled by env var SCOPONE_TRAIN_DEVICE (cpu|cuda[:id]). Defaults to cpu.
    # Models live on CPU for collection; moved to train device inside agent.update.

    # Discover existing checkpoints to possibly resume later (actual load happens after league init)
    best_ckpt_path = ckpt_path.replace('.pth', '_best.pth') if isinstance(ckpt_path, str) else 'checkpoints/ppo_ac_best.pth'
    best_wr_ckpt_path = ckpt_path.replace('.pth', '_bestwr.pth') if isinstance(ckpt_path, str) else 'checkpoints/ppo_ac_bestwr.pth'
    resume_ckpt = None
    def _is_bootstrap_path(p: str) -> bool:
        try:
            return ('bootstrap' in os.path.basename(p).lower())
        except Exception:
            return False

    for _p in [ckpt_path, best_ckpt_path, best_wr_ckpt_path]:
        if _p and (not _is_bootstrap_path(_p)) and os.path.isfile(_p) and os.path.getsize(_p) > 0:
            resume_ckpt = _p
            break
    if resume_ckpt is None:
        import glob as _glob
        _all = [p for p in _glob.glob(os.path.join('checkpoints', '*.pth')) if (not _is_bootstrap_path(p)) and os.path.isfile(p) and os.path.getsize(p) > 0]
        if _all:
            resume_ckpt = max(_all, key=lambda p: os.path.getmtime(p))

    warmup_iters = 0 if mcts_warmup_iters is None else max(0, int(mcts_warmup_iters))

    # Cosine annealing LR schedulers
    actor_sched = optim.lr_scheduler.CosineAnnealingLR(agent.opt_actor, T_max=max(1, num_iterations))
    critic_sched = optim.lr_scheduler.CosineAnnealingLR(agent.opt_critic, T_max=max(1, num_iterations))
    agent.add_lr_schedulers(actor_sched, critic_sched)

    # entropy schedules
    def entropy_schedule_linear(step: int, start: float = 0.01, end: float = 0.001, decay_steps: int = 100000):
        if step >= decay_steps:
            return end
        frac = (decay_steps - step) / decay_steps
        return end + (start - end) * frac

    def entropy_schedule_cosine(step: int, start: float = 0.01, end: float = 0.001, period: int = 100000):
        import math
        t = min(step, period)
        cos = (1 + math.cos(math.pi * t / period)) / 2.0
        return end + (start - end) * cos

    if entropy_schedule_type == 'cosine':
        agent.set_entropy_schedule(lambda s: entropy_schedule_cosine(s))
    else:
        agent.set_entropy_schedule(lambda s: entropy_schedule_linear(s))

    writer = None
    if os.environ.get('SCOPONE_DISABLE_TB', '0') != '1':
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
        writer = _SummaryWriter(log_dir='runs/ppo_ac')
    if writer is not None:
        # Spiega le metriche chiave in TensorBoard
        writer.add_text('help/metrics',
                        '\n'.join([
                            'train/loss_pi: Surrogate loss PPO (policy). Negativo = miglioramento dell\'obiettivo.',
                            'train/loss_v: Value loss (MSE). Alto = critico lontano dai return.',
                            'train/entropy: Entropia media della policy (azioni legali). Più alta = più esplorazione.',
                            'train/approx_kl: KL media (stima) tra policy nuova e vecchia (per early stop).',
                            'train/avg_kl: KL media aggregata su tutti i minibatch dell\'update.',
                            'train/clip_frac: Frazione di campioni con rapporto clipppato (|r-1| > ε).',
                            'train/avg_clip_frac: Media del clip_frac sui minibatch.',
                            'train/grad_norm_actor / train/grad_norm_critic: Norma L2 dei gradienti (diagnostica stabilità).',
                            'train/lr_actor / train/lr_critic: Learning rate correnti (post scheduler).',
                            'train/episode_time_s: Tempo per iterazione (raccolta + update).',
                            'train/avg_return: Return medio del batch raccolto (proxy qualità corrente).',
                            'by_seat/ret_02, by_seat/ret_13: Return medio per gruppo di posti (0/2 vs 1/3).',
                            'by_seat/kl_02, by_seat/kl_13: KL media per gruppo di posti.',
                            'by_seat/entropy_02, by_seat/entropy_13: Entropia media per gruppo di posti.',
                            'by_seat/clip_frac_02, by_seat/clip_frac_13: Frazione di clipping per gruppo.',
                            'league/mini_eval_wr: Win-rate nella mini-valutazione vs checkpoint precedente.',
                            'league/elo_current / league/elo_previous: Elo nel league per corrente/precedente.',
                        ]), 0)
    league = League(base_dir='checkpoints/league')
    # Helper to build MCTS config for eval consistently from trainer flags
    def _make_mcts_cfg_for_eval():
        if not mcts_in_eval:
            return None
        return {
            'sims': int(mcts_sims_eval if mcts_sims_eval is not None else mcts_sims),
            'dets': mcts_dets,
            'c_puct': mcts_c_puct,
            'root_temp': mcts_root_temp,
            'prior_smooth_eps': mcts_prior_smooth_eps,
            'root_dirichlet_alpha': mcts_dirichlet_alpha,
            'root_dirichlet_eps': mcts_dirichlet_eps,
            'robust_child': True,
        }
    # Optional startup refresh: SCOPONE_LEAGUE_REFRESH in {1,0}
    _refresh_flag = str(os.environ.get('SCOPONE_LEAGUE_REFRESH', '1')).strip().lower() in ['1','true','yes','on']
    if _refresh_flag:
        tqdm.write("[league-refresh] starting league refresh from disk …")
        # Refresh league from disk: scan for existing checkpoints not yet registered.
        # We scan both 'checkpoints/' and 'checkpoints/league/' and ignore bootstrap/empty files.
        def _refresh_league_from_disk(_league):
            old_hist = list(getattr(_league, 'history', []) or [])
            roots = ['checkpoints', os.path.join('checkpoints', 'league')]
            seen = set(getattr(_league, 'history', []) or [])
            added = 0
            scanned = 0
            pth_seen = 0
            skipped = 0
            added_paths = []
            for root in roots:
                if not os.path.isdir(root):
                    tqdm.write(f"[league-refresh] root '{root}' missing; skipping root scan")
                    continue
                tqdm.write(f"[league-refresh] scanning root: {os.path.abspath(root)}")
                for dirpath, _dirs, files in os.walk(root):
                    for fname in files:
                        scanned += 1
                        fpath = os.path.join(dirpath, fname)
                        if not fname.endswith('.pth'):
                            tqdm.write(f"[league-refresh] ignore {fpath}: reason=ext_mismatch")
                            skipped += 1
                            continue
                        pth_seen += 1
                        low = fname.lower()
                        reason = None
                        if 'bootstrap' in low:
                            reason = 'bootstrap_file'
                        elif fpath in seen:
                            reason = 'already_registered'
                        elif not os.path.isfile(fpath):
                            reason = 'not_a_file'
                        else:
                            if os.path.getsize(fpath) <= 0:
                                reason = 'empty_file'
                        if reason is not None:
                            tqdm.write(f"[league-refresh] skip {fpath}: reason={reason}")
                            skipped += 1
                            continue
                        # Deep validation: try loading the checkpoint to detect truncated/invalid formats
                        try:
                            _sz = os.path.getsize(fpath)
                            _ck = torch.load(fpath, map_location='cpu')
                            del _ck
                        except Exception as _load_e:
                            tqdm.write(
                                f"[league-refresh] skip {fpath}: reason=load_error type={type(_load_e).__name__} size={_sz} msg={_load_e!r}"
                            )
                            skipped += 1
                            continue
                        _league.register(fpath)
                        seen.add(fpath)
                        added += 1
                        added_paths.append(fpath)
            tqdm.write(f"[league] Refresh scan: roots={len(roots)} scanned_files={scanned} pth_seen={pth_seen} added={added} skipped={skipped} total_now={len(_league.history)}")
            if added > 0:
                tqdm.write(f"[league] Refreshed from disk: +{added} new checkpoints (total={len(_league.history)})")
            # Evaluate newly added checkpoints to update Elo
            if added > 0:
                # Reference is the current top1 among the pre-existing history, if any
                ref_candidates = [p for p in old_hist if os.path.isfile(p)]
                if len(ref_candidates) >= 1:
                    # pick highest Elo among old history
                    ref = max(ref_candidates, key=lambda p: float(_league.elo.get(p, 1000.0)))
                    for npth in added_paths:
                        if npth == ref:
                            continue
                        diff, bd = evaluate_pair_actors(npth, ref, games=eval_games, k_history=k_history,
                                                     mcts=_make_mcts_cfg_for_eval(),
                                                     belief_particles=(belief_particles if mcts_in_eval else 0),
                                                     belief_ess_frac=belief_ess_frac,
                                                     tqdm_desc=f"League refresh: {os.path.basename(npth)} vs top",
                                                     tqdm_position=2)
                        _league.update_elo_from_diff(npth, ref, diff)
                elif len(added_paths) >= 2:
                    # Evaluate sequentially among new paths to seed relative Elo
                    anchor = added_paths[0]
                    for npth in added_paths[1:]:
                        diff, bd = evaluate_pair_actors(npth, anchor, games=eval_games, k_history=k_history,
                                                     mcts=_make_mcts_cfg_for_eval(),
                                                     belief_particles=(belief_particles if mcts_in_eval else 0),
                                                     belief_ess_frac=belief_ess_frac,
                                                     tqdm_desc=f"League refresh: {os.path.basename(npth)} vs anchor",
                                                     tqdm_position=2)
                        _league.update_elo_from_diff(npth, anchor, diff)

        _LP = globals().get('LINE_PROFILE_DECORATOR', None)
        if _LP is not None:
            _refresh_league_from_disk = _LP(_refresh_league_from_disk)
        tqdm.write("[league-refresh] invoking refresh function")
        _refresh_league_from_disk(league)
    else:
        tqdm.write("[league] Startup refresh disabled (SCOPONE_LEAGUE_REFRESH=0)")
        
    # Purge any bootstrap entries from league history on startup
    if getattr(league, 'history', None):
        boot_hist = [p for p in list(league.history) if 'bootstrap' in os.path.basename(p).lower()]
        if boot_hist:
            league.history = [p for p in league.history if p not in boot_hist]
            for p in boot_hist:
                if p in league.elo:
                    del league.elo[p]
            # Save cleaned league state
            league._save()
            tqdm.write(f"[league] Purged bootstrap entries from league: {len(boot_hist)} removed")
    if len(getattr(league, 'history', [])) == 0:
        # Warm-start the league with the resume checkpoint if present (ignore bootstrap files)
        if resume_ckpt and (('bootstrap' not in os.path.basename(resume_ckpt).lower())) and os.path.isfile(resume_ckpt) and os.path.getsize(resume_ckpt) > 0:
            league.register(resume_ckpt)
            tqdm.write(f"[league] Warm-started with existing checkpoint: {resume_ckpt}")

    # Helper to get top-k checkpoints by Elo (fallback Elo=1000.0 for missing entries)
    def _league_top_k(_league: 'League', k: int) -> List[str]:
        hist = list(getattr(_league, 'history', []) or [])
        if not hist:
            return []
        elo_map = getattr(_league, 'elo', {}) or {}
        ranked = sorted(hist, key=lambda p: float(elo_map.get(p, 1000.0)), reverse=True)
        return ranked[:max(0, int(k))]

    

    # Warm-start policy controlled by SCOPONE_WARM_START: '0' start-from-scratch, '1' force top1 clone, '2' use top2 if available
    warm_start_mode = str(os.environ.get('SCOPONE_WARM_START', '2')).strip()
    league_hist = list(getattr(league, 'history', []) or [])
    topN = _league_top_k(league, 2) if league_hist else []
    # Decide resume checkpoint(s)
    resume_A = None
    resume_B = None
    if warm_start_mode == '0':
        resume_A = None; resume_B = None
    elif warm_start_mode == '1':
        resume_A = (topN[0] if len(topN) >= 1 else resume_ckpt)
        resume_B = resume_A
    else:  # '2' default
        resume_A = (topN[0] if len(topN) >= 1 else resume_ckpt)
        resume_B = (topN[1] if len(topN) >= 2 else resume_A)
    # Apply resume to main agent (always)
    if resume_A:
        agent.load(resume_A, map_location='cpu')
        tqdm.write(f"[resume] Loaded agent(A) from {resume_A}")

    partner_actor = None
    opponent_actor = None
    # Semantics:
    # - SELFPLAY decides number of nets: True -> 1 net; False -> 2 nets (A,B)
    # - SCOPONE_TRAIN_FROM_BOTH_TEAMS decides transitions usage when applicable
    #   NOTE: TRAIN_FROM_BOTH_TEAMS is effective ONLY when SELFPLAY=1 and OPP_FROZEN=0.
    #         Rationale: on-policy PPO must update a policy with its own data. In dual-nets or
    #         when the opponent is frozen, we never mix opponent (non-learning) transitions into
    #         the learner's update.
    train_from_both = str(os.environ.get('SCOPONE_TRAIN_FROM_BOTH_TEAMS', '0')).strip().lower() in ['1','true','yes','on']
    opp_frozen_env = str(os.environ.get('SCOPONE_OPP_FROZEN', '1')).strip().lower() in ['1','true','yes','on']
    # Validate combinations under new semantics
    # With SELFPLAY=False we always have two nets; frozen=0 means co-training live; frozen=1 means alternate.

    if use_selfplay:
        # In selfplay we have a single net. train_from_both means use both teams' transitions.
        # If opp_frozen_env is True, we must not use opponent transitions (they come from a frozen mirror).
        train_both_teams = bool(train_from_both and (not opp_frozen_env))
        dual_team_nets = False
    else:
        # In non-selfplay we always maintain two separate nets (A,B)
        dual_team_nets = True
        # In dual-nets collection, each update call uses only main seats for that net
        # so treat per-episode util as 20 everywhere in collection paths
        train_both_teams = False
        agent_teamB = ActionConditionedPPO(obs_dim=obs_dim, k_history=k_history)
        # Load resume_B according to warm-start policy
        if warm_start_mode == '0':
            resumeB = None
        elif warm_start_mode == '1':
            resumeB = resume_A
        else:
            resumeB = resume_B
        if resumeB:
            agent_teamB.load(resumeB, map_location='cpu')
            tqdm.write(f"[resume] Loaded team-B agent from {resumeB}")
    # alterna il main actor tra seat 0/2 e 1/3 per episodi
    even_main_seats = [0, 2]
    odd_main_seats = [1, 3]

    best_return = -1e9
    best_ckpt_path = ckpt_path.replace('.pth', '_best.pth')
    best_wr = -1e9
    best_wr_ckpt_path = ckpt_path.replace('.pth', '_bestwr.pth')

    # Two-line UI: top line shows metrics, second line is the progress bar
    _TQDM_DISABLE = (os.environ.get('TQDM_DISABLE','0') in ['1','true','yes','on'])
    metrics_bar = tqdm(total=1, desc="", position=0, dynamic_ncols=True, bar_format='{desc}', leave=True, disable=_TQDM_DISABLE)
    pbar = tqdm(range(num_iterations), desc="PPO iterations", dynamic_ncols=True, position=1, leave=True, disable=_TQDM_DISABLE)
    shadow_actor = None
    shadow_every = max(1, int(os.environ.get('SCOPONE_FROZEN_UPDATE_EVERY', '1')))
    for it in pbar:
        t0 = time.time()
        # Iteration-level timing breakdown (enabled by SCOPONE_PROFILE=1)
        _iter_t_collect = 0.0
        _iter_t_preproc = 0.0
        _iter_t_update = 0.0
        # Partner/opponent selection policy
        # Opponent selection
        # SELFPLAY=1: no frozen unless explicitly requested via OPP_FROZEN (still treated as single learner vs frozen)
        # SELFPLAY=0: choose according to SCOPONE_START_OPP='top1'|'top2'|'bootstrap' when OPP_FROZEN=1, else live opponent (requires BOTH=1)
        # Auto-bootstrap only when a frozen opponent is required but the league is empty
        if opp_frozen_env and (len(getattr(league, 'history', [])) == 0):
            _bootstrap_dir = 'checkpoints'
            _bootstrap_path = os.path.join(_bootstrap_dir, 'bootstrap_random.pth')
            os.makedirs(_bootstrap_dir, exist_ok=True)
            if (not os.path.isfile(_bootstrap_path)) or (os.path.getsize(_bootstrap_path) <= 0):
                tqdm.write("[league] Bootstrapping empty league with random frozen actor → checkpoints/bootstrap_random.pth")
                _rand_actor = ActionConditionedActor(obs_dim=obs_dim)
                torch.save({'actor': _rand_actor.state_dict()}, _bootstrap_path)
            # Do NOT register bootstrap into league; use ad-hoc only
        if use_selfplay:
            p_ckpt, o_ckpt = None, None
            if opp_frozen_env:
                # Single learner vs frozen mirror
                top1 = _league_top_k(league, 1)
                p_ckpt = o_ckpt = (top1[0] if top1 else None)
        else:
            if opp_frozen_env:
                start_opp = str(os.environ.get('SCOPONE_START_OPP', 'top1')).strip().lower()
                hist = list(getattr(league, 'history', []) or [])
                if start_opp == 'bootstrap' and not hist:
                    _bootstrap_dir = 'checkpoints'
                    _bootstrap_path = os.path.join(_bootstrap_dir, 'bootstrap_random.pth')
                    os.makedirs(_bootstrap_dir, exist_ok=True)
                    if (not os.path.isfile(_bootstrap_path)) or (os.path.getsize(_bootstrap_path) <= 0):
                        tqdm.write("[league] Bootstrapping empty league with random frozen actor → checkpoints/bootstrap_random.pth")
                        _rand_actor = ActionConditionedActor(obs_dim=obs_dim)
                        torch.save({'actor': _rand_actor.state_dict()}, _bootstrap_path)
                    # Do NOT register bootstrap; use directly for this start if needed
                    hist = []
                if start_opp == 'top2':
                    top2 = _league_top_k(league, 2)
                    if len(top2) >= 2:
                        p_ckpt, o_ckpt = top2[0], top2[1]
                    elif len(top2) == 1:
                        p_ckpt = o_ckpt = top2[0]
                    else:
                        # Fallback to bootstrap pair only if history empty
                        _bootstrap_path = os.path.join('checkpoints', 'bootstrap_random.pth')
                        p_ckpt = o_ckpt = (_bootstrap_path if os.path.isfile(_bootstrap_path) and os.path.getsize(_bootstrap_path) > 0 else None)
                else:
                    # default 'top1': both sides face strongest
                    top1 = _league_top_k(league, 1)
                    if top1:
                        p_ckpt = o_ckpt = top1[0]
                    else:
                        _bootstrap_path = os.path.join('checkpoints', 'bootstrap_random.pth')
                        p_ckpt = o_ckpt = (_bootstrap_path if os.path.isfile(_bootstrap_path) and os.path.getsize(_bootstrap_path) > 0 else None)
            else:
                # live opponent (co-training). Requires BOTH=1; validated above
                p_ckpt, o_ckpt = None, None
        # Simple local cache to avoid reloading frozen actors every iter
        if not use_selfplay:
            if not hasattr(league, '_frozen_cache'):
                league._frozen_cache = {}
            if p_ckpt and os.path.isfile(p_ckpt):
                partner_actor = league._frozen_cache.get(p_ckpt)
                if partner_actor is None:
                    partner_actor = _load_frozen_actor(p_ckpt, obs_dim)
                    league._frozen_cache[p_ckpt] = partner_actor
            if o_ckpt and os.path.isfile(o_ckpt):
                opponent_actor = league._frozen_cache.get(o_ckpt)
                if opponent_actor is None:
                    opponent_actor = _load_frozen_actor(o_ckpt, obs_dim)
                    league._frozen_cache[o_ckpt] = opponent_actor
            # Live co-training override: use teamB live actor as partner/opponent
            if dual_team_nets and (not opp_frozen_env):
                partner_actor = agent_teamB.actor
                opponent_actor = agent_teamB.actor
        # Alternation controls for dual nets with frozen opponent
        # SCOPONE_ALTERNATE_ITERS: in dual-nets+frozen, number of iterations to train one net
        #   before swapping roles (A<->B). Larger values give more stable short-term learning
        #   per net; smaller values promote frequent role switches.
        alt_k = max(1, int(os.environ.get('SCOPONE_ALTERNATE_ITERS', '1')))
        train_A_now = ((it // alt_k) % 2 == 0)
        main_seats = even_main_seats if (it % 2 == 0) else odd_main_seats
        # Complementary seats for the other team (used when swapping A<->B)
        main_seats_B = odd_main_seats if (main_seats == even_main_seats) else even_main_seats
        use_parallel = (num_envs is not None and int(num_envs) > 1)
        if bool(mcts_train):
            mcts_train_factor = 0.0 if it < warmup_iters else 1.0
        else:
            mcts_train_factor = 0.0
        # Eval mode during data collection (dropout/BN off)
        agent.actor.eval()
        agent.critic.eval()
        if use_parallel:
            # Mirror single-env logic: choose episodes so that batch B is a multiple of minibatch_size
            per_ep_util = (40 if train_both_teams else 20)
            # Compute LCM(mb, per_ep_util) and derive episodes from aligned horizon
            import math as _math
            lcm_mb_ep = (abs(minibatch_size * per_ep_util) // _math.gcd(minibatch_size, per_ep_util)) if (minibatch_size > 0 and per_ep_util > 0) else max(minibatch_size, per_ep_util)
            if lcm_mb_ep > 0 and (horizon % lcm_mb_ep) != 0:
                new_h = ((horizon + lcm_mb_ep - 1) // lcm_mb_ep) * lcm_mb_ep
                print(f"[horizon] adjusted to LCM(mb={minibatch_size}, per_ep={per_ep_util})={lcm_mb_ep}: {horizon} -> {new_h}")
                horizon = new_h
            episodes_hint = max(1, horizon // per_ep_util)
            # Debug: mostra hint e distribuzione per-env
            eps_per_env_dbg = max(1, (episodes_hint + int(num_envs) - 1) // int(num_envs))
            total_eps_dbg = eps_per_env_dbg * int(num_envs)
            tqdm.write(f"[episodes] it={it+1} horizon={horizon} num_envs={num_envs} episodes_hint={episodes_hint} "
                        f"episodes_per_env={eps_per_env_dbg} total_env_episodes={total_eps_dbg}")
            # Abilita/disabilita MCTS in parallelo in base al flag di training
            parallel_use_mcts = bool(mcts_train and mcts_train_factor > 0.0)
            if dual_team_nets and (not opp_frozen_env):
                # Collect for team A vs live team B
                _t_c0 = time.time() if _PAR_TIMING else 0.0
                batch_A = collect_trajectory_parallel(agent,
                                                       num_envs=int(num_envs),
                                                       episodes_total_hint=episodes_hint,
                                                       k_history=k_history,
                                                       gamma=1.0,
                                                       lam=0.95,
                                                       use_mcts=parallel_use_mcts,
                                                       train_both_teams=False,
                                                       main_seats=main_seats,
                                                       mcts_sims=mcts_sims,
                                                       mcts_dets=mcts_dets,
                                                       mcts_c_puct=mcts_c_puct,
                                                       mcts_root_temp=mcts_root_temp,
                                                       mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                       mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                                                       mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                       mcts_train_factor=mcts_train_factor,
                                                       seed=int(seed),
                                                       show_progress_env=True,
                                                       tqdm_base_pos=3)
                # Collect for team B vs live team A (swap seats)
                main_seats_B = odd_main_seats if (main_seats == even_main_seats) else even_main_seats
                batch_B = collect_trajectory_parallel(agent_teamB,
                                                       num_envs=int(num_envs),
                                                       episodes_total_hint=episodes_hint,
                                                       k_history=k_history,
                                                       gamma=1.0,
                                                       lam=0.95,
                                                       use_mcts=parallel_use_mcts,
                                                       train_both_teams=False,
                                                       main_seats=main_seats_B,
                                                       mcts_sims=mcts_sims,
                                                       mcts_dets=mcts_dets,
                                                       mcts_c_puct=mcts_c_puct,
                                                       mcts_root_temp=mcts_root_temp,
                                                       mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                       mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                                                       mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                       mcts_train_factor=mcts_train_factor,
                                                       seed=int(seed + 1),
                                                       show_progress_env=True,
                                                       tqdm_base_pos=4)
                if _PAR_TIMING:
                    _iter_t_collect = (time.time() - _t_c0)
            elif dual_team_nets and opp_frozen_env:
                # Alternate training: one team learns while the other team acts frozen (no update)
                _t_c0 = time.time() if _PAR_TIMING else 0.0
                if train_A_now:
                    batch_A = collect_trajectory_parallel(agent,
                                                           num_envs=int(num_envs),
                                                           episodes_total_hint=episodes_hint,
                                                           k_history=k_history,
                                                           gamma=1.0,
                                                           lam=0.95,
                                                           use_mcts=parallel_use_mcts,
                                                           train_both_teams=False,
                                                           main_seats=main_seats,
                                                           mcts_sims=mcts_sims,
                                                           mcts_dets=mcts_dets,
                                                           mcts_c_puct=mcts_c_puct,
                                                           mcts_root_temp=mcts_root_temp,
                                                           mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                           mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                                                           mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                           mcts_train_factor=mcts_train_factor,
                                                           seed=int(seed),
                                                           show_progress_env=True,
                                                           tqdm_base_pos=3,
                                                           frozen_actor=agent_teamB.actor,
                                                           frozen_non_main=True)
                else:
                    batch_B = collect_trajectory_parallel(agent_teamB,
                                                           num_envs=int(num_envs),
                                                           episodes_total_hint=episodes_hint,
                                                           k_history=k_history,
                                                           gamma=1.0,
                                                           lam=0.95,
                                                           use_mcts=parallel_use_mcts,
                                                           train_both_teams=False,
                                                           main_seats=main_seats_B,
                                                           mcts_sims=mcts_sims,
                                                           mcts_dets=mcts_dets,
                                                           mcts_c_puct=mcts_c_puct,
                                                           mcts_root_temp=mcts_root_temp,
                                                           mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                           mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                                                           mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                           mcts_train_factor=mcts_train_factor,
                                                           seed=int(seed + 1),
                                                           show_progress_env=True,
                                                           tqdm_base_pos=4,
                                                           frozen_actor=agent.actor,
                                                           frozen_non_main=True)
                if _PAR_TIMING:
                    _iter_t_collect = (time.time() - _t_c0)
            else:
                _t_c0 = time.time() if _PAR_TIMING else 0.0
                # Refresh shadow actor on schedule in selfplay+frozen
                # SCOPONE_FROZEN_UPDATE_EVERY: in selfplay+frozen, how often to refresh the
                #   "shadow" opponent from the current live net. The shadow remains fixed
                #   between refreshes to create a stationary opponent for the non-main seats.
                if use_selfplay and opp_frozen_env:
                    if (shadow_actor is None) or ((it % shadow_every) == 0):
                        # clone lightweight copy of actor state
                        shadow_actor = ActionConditionedActor(obs_dim=obs_dim)
                        shadow_actor.load_state_dict(agent.actor.state_dict())
                batch = collect_trajectory_parallel(agent,
                                                    num_envs=int(num_envs),
                                                    episodes_total_hint=episodes_hint,
                                                    k_history=k_history,
                                                    gamma=1.0,
                                                    lam=0.95,
                                                    use_mcts=parallel_use_mcts,
                                                    train_both_teams=train_both_teams,
                                                    main_seats=main_seats,
                                                    mcts_sims=mcts_sims,
                                                    mcts_dets=mcts_dets,
                                                    mcts_c_puct=mcts_c_puct,
                                                    mcts_root_temp=mcts_root_temp,
                                                    mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                    mcts_dirichlet_alpha=mcts_dirichlet_alpha,
                                                    mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                    mcts_train_factor=mcts_train_factor,
                                                    seed=int(seed),
                                                    show_progress_env=True,
                                                    tqdm_base_pos=3,
                                                    frozen_actor=(shadow_actor if (use_selfplay and opp_frozen_env) else None),
                                                    frozen_non_main=bool(use_selfplay and opp_frozen_env))
                if _PAR_TIMING:
                    _iter_t_collect = (time.time() - _t_c0)
        else:
            # Strategia MCTS: warmup senza MCTS per le prime iterazioni, poi scala con il progresso mano
            if dual_team_nets and (not opp_frozen_env):
                _t_c0 = time.time() if _PAR_TIMING else 0.0
                # Team A as main vs live team B
                batch_A = collect_trajectory(env, agent, horizon=horizon, partner_actor=agent_teamB.actor, opponent_actor=agent_teamB.actor, main_seats=main_seats,
                                                 belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                                 episodes=None, final_reward_only=True,
                                                 use_mcts=bool(mcts_train),
                                                 mcts_sims=mcts_sims, mcts_dets=mcts_dets, mcts_c_puct=mcts_c_puct,
                                                 mcts_root_temp=mcts_root_temp, mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                 mcts_dirichlet_alpha=mcts_dirichlet_alpha, mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                 mcts_train_factor=mcts_train_factor,
                                                 mcts_progress_start=0.25, mcts_progress_full=0.75,
                                                 mcts_min_sims=0,
                                                 train_both_teams=False,
                                                 gamma=1.0,
                                                 lam=0.95,
                                                 seed=int(seed))
                # Team B as main vs live team A (swap seats)
                main_seats_B = odd_main_seats if (main_seats == even_main_seats) else even_main_seats
                batch_B = collect_trajectory(env, agent_teamB, horizon=horizon, partner_actor=agent.actor, opponent_actor=agent.actor, main_seats=main_seats_B,
                                                 belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                                 episodes=None, final_reward_only=True,
                                                 use_mcts=bool(mcts_train),
                                                 mcts_sims=mcts_sims, mcts_dets=mcts_dets, mcts_c_puct=mcts_c_puct,
                                                 mcts_root_temp=mcts_root_temp, mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                 mcts_dirichlet_alpha=mcts_dirichlet_alpha, mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                 mcts_train_factor=mcts_train_factor,
                                                 mcts_progress_start=0.25, mcts_progress_full=0.75,
                                                 mcts_min_sims=0,
                                                 train_both_teams=False,
                                                 gamma=1.0,
                                                 lam=0.95,
                                                 seed=int(seed + 1))
                if _PAR_TIMING:
                    _iter_t_collect = (time.time() - _t_c0)
            elif dual_team_nets and opp_frozen_env:
                _t_c0 = time.time() if _PAR_TIMING else 0.0
                if train_A_now:
                    # Team A learns vs frozen Team B
                    batch_A = collect_trajectory(env, agent, horizon=horizon, partner_actor=agent_teamB.actor, opponent_actor=agent_teamB.actor, main_seats=main_seats,
                                                 belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                                 episodes=None, final_reward_only=True,
                                                 use_mcts=bool(mcts_train),
                                                 mcts_sims=mcts_sims, mcts_dets=mcts_dets, mcts_c_puct=mcts_c_puct,
                                                 mcts_root_temp=mcts_root_temp, mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                 mcts_dirichlet_alpha=mcts_dirichlet_alpha, mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                 mcts_train_factor=mcts_train_factor,
                                                 mcts_progress_start=0.25, mcts_progress_full=0.75,
                                                 mcts_min_sims=0,
                                                 train_both_teams=False,
                                                 gamma=1.0,
                                                 lam=0.95,
                                                 seed=int(seed))
                else:
                    # Team B learns vs frozen Team A
                    batch_B = collect_trajectory(env, agent_teamB, horizon=horizon, partner_actor=agent.actor, opponent_actor=agent.actor, main_seats=main_seats,
                                                 belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                                 episodes=None, final_reward_only=True,
                                                 use_mcts=bool(mcts_train),
                                                 mcts_sims=mcts_sims, mcts_dets=mcts_dets, mcts_c_puct=mcts_c_puct,
                                                 mcts_root_temp=mcts_root_temp, mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                                 mcts_dirichlet_alpha=mcts_dirichlet_alpha, mcts_dirichlet_eps=mcts_dirichlet_eps,
                                                 mcts_train_factor=mcts_train_factor,
                                                 mcts_progress_start=0.25, mcts_progress_full=0.75,
                                                 mcts_min_sims=0,
                                                 train_both_teams=False,
                                                 gamma=1.0,
                                                 lam=0.95,
                                                 seed=int(seed + 1))
                if _PAR_TIMING:
                    _iter_t_collect = (time.time() - _t_c0)
            else:
                _t_c0 = time.time() if _PAR_TIMING else 0.0
                batch = collect_trajectory(env, agent, horizon=horizon, partner_actor=partner_actor, opponent_actor=opponent_actor, main_seats=main_seats,
                                           belief_particles=belief_particles, belief_ess_frac=belief_ess_frac,
                                           episodes=None, final_reward_only=True,
                                           use_mcts=bool(mcts_train),
                                           mcts_sims=mcts_sims, mcts_dets=mcts_dets, mcts_c_puct=mcts_c_puct,
                                           mcts_root_temp=mcts_root_temp, mcts_prior_smooth_eps=mcts_prior_smooth_eps,
                                           mcts_dirichlet_alpha=mcts_dirichlet_alpha, mcts_dirichlet_eps=mcts_dirichlet_eps,
                                           mcts_train_factor=mcts_train_factor,
                                           mcts_progress_start=0.25, mcts_progress_full=0.75,
                                           mcts_min_sims=0,
                                           train_both_teams=train_both_teams,
                                           gamma=1.0,
                                           lam=0.95,
                                           seed=int(seed))
                if _PAR_TIMING:
                    _iter_t_collect = (time.time() - _t_c0)
        if dual_team_nets and (not opp_frozen_env):
            if len(batch_A['obs']) == 0 or len(batch_B['obs']) == 0:
                continue
        elif dual_team_nets and opp_frozen_env:
            if train_A_now:
                if len(batch_A['obs']) == 0:
                    continue
            else:
                if len(batch_B['obs']) == 0:
                    continue
        else:
            if len(batch['obs']) == 0:
                continue
        # normalizza vantaggi completamente su GPU (no sync)
        _t_p0 = time.time() if _PAR_TIMING else 0.0
        if dual_team_nets and (not opp_frozen_env):
            for label, _b in (('dual-teamA', batch_A), ('dual-teamB', batch_B)):
                adv = _b['adv']
                if adv.numel() > 0:
                    _b['adv'] = _normalize_adv_tensor(label, adv)
        elif dual_team_nets and opp_frozen_env:
            if train_A_now:
                adv = batch_A['adv']
                if adv.numel() > 0:
                    batch_A['adv'] = _normalize_adv_tensor('frozen-teamA', adv)
            else:
                adv = batch_B['adv']
                if adv.numel() > 0:
                    batch_B['adv'] = _normalize_adv_tensor('frozen-teamB', adv)
        else:
            adv = batch['adv']
            if adv.numel() > 0:
                batch['adv'] = _normalize_adv_tensor('single', adv)
        # Ensure training mode on update
        agent.actor.train()
        agent.critic.train()
        if dual_team_nets and (not opp_frozen_env):
            agent_teamB.actor.train(); agent_teamB.critic.train()
        # Pre-filter: require all rows to have at least one legal action
        def _ensure_legals(_b):
            lc = _b['legals_count']
            keep = (lc > 0)
            if bool((~keep).any().item()):
                num_drop = int((~keep).sum().item()); total = int(lc.size(0))
                bad_idx = torch.nonzero(~keep, as_tuple=False).flatten().tolist()[:10]
                raise RuntimeError(f"train_ppo: refusing to drop transitions with zero legals (would drop {num_drop}/{total}); examples idx={bad_idx}")
        if dual_team_nets and (not opp_frozen_env):
            _ensure_legals(batch_A); _ensure_legals(batch_B)
        elif dual_team_nets and opp_frozen_env:
            if train_A_now:
                _ensure_legals(batch_A)
            else:
                _ensure_legals(batch_B)
        else:
            _ensure_legals(batch)
        # Enforce B multiple of minibatch_size by dropping tail
        def _ensure_mb(_b):
            B_now = int(_b['obs'].size(0)) if 'obs' in _b and torch.is_tensor(_b['obs']) else 0
            mb = minibatch_size
            drop = (B_now % mb)
            if B_now > 0 and drop > 0:
                raise RuntimeError(f"train_ppo: batch size {B_now} is not a multiple of minibatch_size={mb}; adjust horizon/collection to avoid dropping {drop} transitions")
        if dual_team_nets and (not opp_frozen_env):
            _ensure_mb(batch_A); _ensure_mb(batch_B)
        elif dual_team_nets and opp_frozen_env:
            if train_A_now:
                _ensure_mb(batch_A)
            else:
                _ensure_mb(batch_B)
        else:
            _ensure_mb(batch)
        if _PAR_TIMING:
            _iter_t_preproc = (time.time() - _t_p0)
        # Aumenta minibatch_size approfittando della VRAM ampia
        if dual_team_nets and (not opp_frozen_env):
            _t_u0 = time.time() if _PAR_TIMING else 0.0
            info_A = agent.update(batch_A, epochs=4, minibatch_size=minibatch_size)
            info_B = agent_teamB.update(batch_B, epochs=4, minibatch_size=minibatch_size)
            if _PAR_TIMING:
                _iter_t_update = (time.time() - _t_u0)
        elif dual_team_nets and opp_frozen_env:
            _t_u0 = time.time() if _PAR_TIMING else 0.0
            if train_A_now:
                info_A = agent.update(batch_A, epochs=4, minibatch_size=minibatch_size)
            else:
                info_B = agent_teamB.update(batch_B, epochs=4, minibatch_size=minibatch_size)
            if _PAR_TIMING:
                _iter_t_update = (time.time() - _t_u0)
        else:
            _t_u0 = time.time() if _PAR_TIMING else 0.0
            info = agent.update(batch, epochs=4, minibatch_size=minibatch_size)
            if _PAR_TIMING:
                _iter_t_update = (time.time() - _t_u0)
        dt = time.time() - t0

        # proxy per best: media return del batch
        # All device tensors; compute small stats without moving large arrays
        if dual_team_nets and (not opp_frozen_env):
            avg_return_A = float(batch_A['ret'].mean().detach().cpu().item()) if len(batch_A['ret']) else 0.0
            avg_return_B = float(batch_B['ret'].mean().detach().cpu().item()) if len(batch_B['ret']) else 0.0
            avg_return = avg_return_A
            if avg_return_A > best_return:
                best_return = avg_return_A
                os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                agent.save(best_ckpt_path.replace('.pth', '_teamA.pth'))
            # Save best for team B separately
            if avg_return_B > (globals().get('_best_return_B', -1e9)):
                _best_return_B = avg_return_B
                os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                agent_teamB.save(best_ckpt_path.replace('.pth', '_teamB.pth'))
        elif dual_team_nets and opp_frozen_env:
            if train_A_now:
                avg_return_A = float(batch_A['ret'].mean().detach().cpu().item()) if len(batch_A['ret']) else 0.0
                avg_return = avg_return_A
                if avg_return_A > best_return:
                    best_return = avg_return_A
                    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                    agent.save(best_ckpt_path.replace('.pth', '_teamA.pth'))
            else:
                avg_return_B = float(batch_B['ret'].mean().detach().cpu().item()) if len(batch_B['ret']) else 0.0
                avg_return = avg_return_B
                if avg_return_B > (globals().get('_best_return_B', -1e9)):
                    _best_return_B = avg_return_B
                    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                    agent_teamB.save(best_ckpt_path.replace('.pth', '_teamB.pth'))
        else:
            if len(batch['ret']):
                avg_return = float(batch['ret'].mean().detach().cpu().item())
            else:
                avg_return = 0.0
            if avg_return > best_return:
                best_return = avg_return
                os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                agent.save(best_ckpt_path)

        # mini-eval periodica e Elo update
        if eval_every and (it + 1) % eval_every == 0 and len(league.history) >= 1:
            cur_tmp = ckpt_path.replace('.pth', f'_tmp_it{it+1}.pth')
            agent.save(cur_tmp)
            league.register(cur_tmp)
            if dual_team_nets and (not opp_frozen_env):
                cur_tmp_B = ckpt_path.replace('.pth', f'_tmpB_it{it+1}.pth')
                agent_teamB.save(cur_tmp_B)
                league.register(cur_tmp_B)
            prev_ckpt = league.history[-2] if len(league.history) >= 2 else None
            if prev_ckpt is not None:
                mcts_cfg = None
                if mcts_in_eval:
                    mcts_cfg = {
                        'sims': int(mcts_sims_eval if mcts_sims_eval is not None else mcts_sims),
                        'dets': mcts_dets,
                        'c_puct': mcts_c_puct,
                        'root_temp': mcts_root_temp,
                        'prior_smooth_eps': mcts_prior_smooth_eps,
                        'root_dirichlet_alpha': mcts_dirichlet_alpha,
                        'root_dirichlet_eps': mcts_dirichlet_eps,
                        'robust_child': True,
                    }
                # Ensure deterministic MCTS during evaluation/post-game analysis
                mcts_eval_seed = int(os.environ.get('MCTS_EVAL_SEED', resolve_seed(0)))
                with temporary_seed(mcts_eval_seed):
                    diff, bd = evaluate_pair_actors(cur_tmp, prev_ckpt, games=eval_games, k_history=k_history,
                                                 mcts=_make_mcts_cfg_for_eval(),
                                                 belief_particles=(belief_particles if mcts_in_eval else 0),
                                                 belief_ess_frac=belief_ess_frac,
                                                 tqdm_desc=f"Mini-eval: it{it+1}",
                                                 tqdm_position=2)
                league.update_elo_from_diff(cur_tmp, prev_ckpt, diff)
                if writer is not None:
                    wr = float((bd.get('meta') or {}).get('win_rate', 0.0))
                    writer.add_scalar('league/mini_eval_diff', diff, it)
                    writer.add_scalar('league/mini_eval_wr', wr, it)
                    writer.add_scalar('league/elo_current', league.elo.get(cur_tmp, 1000.0), it)
                    writer.add_scalar('league/elo_previous', league.elo.get(prev_ckpt, 1000.0), it)
                # salva checkpoint best wr con soglia/CI (Wilson interval)
                import math
                n = max(1, eval_games)
                p = wr
                z = 1.96
                denom = 1 + z*z/n
                center = p + z*z/(2*n)
                margin = z*math.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
                lower = (center - margin) / denom
                improved = wr > best_wr
                meets_threshold = lower >= 0.5  # lower bound sopra il 50%
                if improved or meets_threshold:
                    best_wr = max(best_wr, wr)
                    os.makedirs(os.path.dirname(best_wr_ckpt_path), exist_ok=True)
                    agent.save(best_wr_ckpt_path)
                if wr > best_wr:
                    best_wr = wr
                    os.makedirs(os.path.dirname(best_wr_ckpt_path), exist_ok=True)
                    agent.save(best_wr_ckpt_path)

        # Aggiorna output terminale ad ogni iterazione
        def _to_float(x):
            return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
        preview_keys = ['loss_pi', 'loss_v', 'approx_kl', 'clip_frac', 'avg_clip_frac', 'avg_kl']
        if dual_team_nets and (not opp_frozen_env):
            preview_A = {k: round(_to_float(info_A[k]), 4) for k in preview_keys if k in info_A}
            preview_A.update({'avg_ret': round(avg_return_A, 4)})
            preview_B = {k: round(_to_float(info_B[k]), 4) for k in preview_keys if k in info_B}
            preview_B.update({'avg_ret_B': round(avg_return_B, 4)})
            preview = {**{f"A_{k}": v for k, v in preview_A.items()}, **{f"B_{k}": v for k, v in preview_B.items()}, 't_s': round(dt, 2)}
        elif dual_team_nets and opp_frozen_env:
            if train_A_now:
                preview_A = {k: round(_to_float(info_A[k]), 4) for k in preview_keys if k in info_A}
                preview_A.update({'avg_ret': round(avg_return, 4)})
                preview = {**{f"A_{k}": v for k, v in preview_A.items()}, 't_s': round(dt, 2)}
            else:
                preview_B = {k: round(_to_float(info_B[k]), 4) for k in preview_keys if k in info_B}
                preview_B.update({'avg_ret_B': round(avg_return, 4)})
                preview = {**{f"B_{k}": v for k, v in preview_B.items()}, 't_s': round(dt, 2)}
        else:
            preview = {k: round(_to_float(info[k]), 4) for k in preview_keys if k in info}
            preview.update({'avg_ret': round(avg_return, 4), 't_s': round(dt, 2)})
        metrics_body = " ".join([f"{k}:{v}" for k, v in preview.items()]) if preview else ""
        metrics_prefix = f"it {it + 1}/{num_iterations}"
        metrics_bar.set_description_str(
            f"{metrics_prefix} | {metrics_body}" if metrics_body else metrics_prefix,
            refresh=False
        )
        # Iteration-level timing print (sums to dt)
        if _PAR_TIMING:
            _post = max(0.0, dt - (_iter_t_collect + _iter_t_preproc + _iter_t_update))
            tqdm.write(f"[iter-timing] total={dt:.3f}s collect={_iter_t_collect:.3f} preproc={_iter_t_preproc:.3f} update={_iter_t_update:.3f} post={_post:.3f}")

        if writer is not None:
            def _to_float(x):
                return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
            if dual_team_nets and (not opp_frozen_env):
                for k, v in info_A.items():
                    writer.add_scalar(f'train/A_{k}', _to_float(v), it)
                for k, v in info_B.items():
                    writer.add_scalar(f'train/B_{k}', _to_float(v), it)
                writer.add_scalar('train/A_episode_time_s', dt, it)
                writer.add_scalar('train/A_avg_return', avg_return_A, it)
                writer.add_scalar('train/B_avg_return', avg_return_B, it)
            elif dual_team_nets and opp_frozen_env:
                if train_A_now:
                    for k, v in info_A.items():
                        writer.add_scalar(f'train/A_{k}', _to_float(v), it)
                    writer.add_scalar('train/A_episode_time_s', dt, it)
                    writer.add_scalar('train/A_avg_return', avg_return, it)
                else:
                    for k, v in info_B.items():
                        writer.add_scalar(f'train/B_{k}', _to_float(v), it)
                    writer.add_scalar('train/B_episode_time_s', dt, it)
                    writer.add_scalar('train/B_avg_return', avg_return, it)
            else:
                for k, v in info.items():
                    writer.add_scalar(f'train/{k}', _to_float(v), it)
                writer.add_scalar('train/episode_time_s', dt, it)
                writer.add_scalar('train/avg_return', avg_return, it)
            writer.add_text('train/main_seats', str(main_seats), it)
            # Log by seat completamente su GPU; singola sync per tutto il blocco
            if dual_team_nets and opp_frozen_env:
                _b_used = (batch_A if train_A_now else batch_B)
            elif dual_team_nets and (not opp_frozen_env):
                _b_used = batch_A  # primary reference
            else:
                _b_used = batch
            seats_t = torch.argmax(_b_used['seat_team'][:, :4], dim=1)
            ret_t = _b_used['ret']
            mask_02_t = (seats_t == 0) | (seats_t == 2)
            mask_13_t = (seats_t == 1) | (seats_t == 3)
            cnt_02 = mask_02_t.float().sum()
            cnt_13 = mask_13_t.float().sum()
            sum_ret_02 = (ret_t * mask_02_t.float()).sum()
            sum_ret_13 = (ret_t * mask_13_t.float()).sum()
            mean_ret_02 = torch.where(cnt_02 > 0, sum_ret_02 / cnt_02, torch.zeros((), device=device, dtype=torch.float32))
            mean_ret_13 = torch.where(cnt_13 > 0, sum_ret_13 / cnt_13, torch.zeros((), device=device, dtype=torch.float32))
            diag = _compute_per_seat_diagnostics(agent, _b_used)
            # Prepara chiavi e valori da sincronizzare in una volta
            keys = [
                'by_seat/ret_02', 'by_seat/ret_13',
                'by_seat/kl_02', 'by_seat/kl_13',
                'by_seat/entropy_02', 'by_seat/entropy_13',
                'by_seat/clip_frac_02', 'by_seat/clip_frac_13'
            ]
            vals = [
                mean_ret_02.to(torch.float32),
                mean_ret_13.to(torch.float32),
                diag['by_seat/kl_02'].to(torch.float32),
                diag['by_seat/kl_13'].to(torch.float32),
                diag['by_seat/entropy_02'].to(torch.float32),
                diag['by_seat/entropy_13'].to(torch.float32),
                diag['by_seat/clip_frac_02'].to(torch.float32),
                diag['by_seat/clip_frac_13'].to(torch.float32),
            ]
            stacked = torch.stack(vals)
            numbers = stacked.detach().cpu().tolist()  # unica sincronizzazione CPU
            for key, num in zip(keys, numbers):
                writer.add_scalar(key, float(num), it)
        # Save periodically every 'save_every' iters. Additionally, save on the last
        # iteration to avoid runs with zero checkpoints.
        if ((it + 1) % save_every == 0) or ((it + 1) == num_iterations):
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            # Timestamped filenames, with suffixes by mode
            from datetime import datetime as _dt
            ts = _dt.now().strftime('%Y%m%d_%H%M%S')
            if use_selfplay:
                out_A = ckpt_path.replace('.pth', f'_selfplay_{ts}.pth')
                agent.save(out_A)
                # Register only non-bootstrap
                league.register(out_A)
                # Elo evaluation: current vs best historical reference (top Elo)
                if len(league.history) >= 2:
                    # pick highest-Elo historical different from current
                    hist = [p for p in league.history if p != out_A]
                    if hist:
                        ref = max(hist, key=lambda p: float(league.elo.get(p, 1000.0)))
                        diff, _ = evaluate_pair_actors(out_A, ref, games=eval_games, k_history=k_history,
                                                    mcts=_make_mcts_cfg_for_eval(),
                                                    belief_particles=(belief_particles if mcts_in_eval else 0),
                                                    belief_ess_frac=belief_ess_frac,
                                                    tqdm_desc=f"Eval: A vs ref",
                                                    tqdm_position=2)
                        league.update_elo_from_diff(out_A, ref, diff)
            else:
                out_A = ckpt_path.replace('.pth', f'_teamA_{ts}.pth')
                agent.save(out_A)
                league.register(out_A)
                out_B = ckpt_path.replace('.pth', f'_teamB_{ts}.pth') if dual_team_nets else None
                if out_B:
                    agent_teamB.save(out_B)
                    league.register(out_B)
                # Elo evaluation: A vs B (if both) and vs league top1
                if out_B:
                    diffAB, bdAB = evaluate_pair_actors(out_A, out_B, games=eval_games, k_history=k_history,
                                                   mcts=_make_mcts_cfg_for_eval(),
                                                   belief_particles=(belief_particles if mcts_in_eval else 0),
                                                   belief_ess_frac=belief_ess_frac,
                                                   tqdm_desc=f"Eval: A vs B",
                                                   tqdm_position=2)
                    league.update_elo_from_diff(out_A, out_B, diffAB)
                # pick highest-Elo historical reference different from the new ones
                hist = [p for p in league.history if p not in ([out_A] + ([out_B] if out_B else []))]
                if hist:
                    ref = max(hist, key=lambda p: float(league.elo.get(p, 1000.0)))
                    diffA, bdA = evaluate_pair_actors(out_A, ref, games=eval_games, k_history=k_history,
                                                  mcts=_make_mcts_cfg_for_eval(),
                                                  belief_particles=(belief_particles if mcts_in_eval else 0),
                                                  belief_ess_frac=belief_ess_frac,
                                                  tqdm_desc=f"Eval: A vs ref",
                                                  tqdm_position=2)
                    league.update_elo_from_diff(out_A, ref, diffA)
                    if out_B:
                        diffB, bdB = evaluate_pair_actors(out_B, ref, games=eval_games, k_history=k_history,
                                                      mcts=_make_mcts_cfg_for_eval(),
                                                      belief_particles=(belief_particles if mcts_in_eval else 0),
                                                      belief_ess_frac=belief_ess_frac,
                                                      tqdm_desc=f"Eval: B vs ref",
                                                      tqdm_position=2)
                        league.update_elo_from_diff(out_B, ref, diffB)
            # After saving a real checkpoint, delete any bootstrap files present
            _bootstrap_path = os.path.join('checkpoints', 'bootstrap_random.pth')
            if os.path.isfile(_bootstrap_path):
                os.remove(_bootstrap_path)
                tqdm.write("[bootstrap] Removed bootstrap_random.pth after saving real checkpoint")
        # Optional profiler or external hook per-iteration
        if on_iter_end is not None:
            on_iter_end(it)
    if writer is not None:
        writer.close()
    metrics_bar.close()
    pbar.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train PPO Action-Conditioned for Scopone')
    parser.add_argument('--iters', type=int, default=2000, help='Number of PPO iterations')
    parser.add_argument('--horizon', type=int, default=256, help='Rollout horizon (steps) per iteration (minimo 40); con solo reward finale raccoglie ~horizon//40 episodi')
    parser.add_argument('--save-every', type=int, default=200, help='Save checkpoint every N iterations')
    parser.add_argument('--ckpt', type=str, default='checkpoints/ppo_ac.pth', help='Checkpoint path')
    parser.add_argument('--compact', action='store_true', help='Use compact observation (recommended)')
    parser.add_argument('--k-history', type=int, default=39, help='Number of recent moves in compact history')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--entropy-schedule', type=str, default='linear', choices=['linear','cosine'], help='Entropy schedule type')
    parser.add_argument('--eval-every', type=int, default=0, help='Run mini-eval every N iters (0=off)')
    parser.add_argument('--eval-games', type=int, default=10, help='Games per mini-eval')
    parser.add_argument('--belief-particles', type=int, default=512, help='Belief particles for trainer')
    parser.add_argument('--belief-ess-frac', type=float, default=0.5, help='Belief ESS fraction for trainer')
    parser.add_argument('--mcts-eval', action='store_true', help='Use MCTS in mini-eval')
    parser.add_argument('--mcts-train', action='store_true', default=True, help='Use MCTS during training action selection for main seats')
    parser.add_argument('--train-both-teams', action='store_true', help='Train both teams simultaneously (all seats are main)')
    parser.add_argument('--mcts-sims', type=int, default=128)
    parser.add_argument('--mcts-sims-eval', type=int, default=None, help='Override eval MCTS sims (default: same as --mcts-sims)')
    parser.add_argument('--mcts-dets', type=int, default=4)
    parser.add_argument('--mcts-c-puct', type=float, default=1.0)
    parser.add_argument('--mcts-root-temp', type=float, default=0.0)
    parser.add_argument('--mcts-prior-smooth-eps', type=float, default=0.0)
    parser.add_argument('--mcts-dirichlet-alpha', type=float, default=0.25)
    parser.add_argument('--mcts-dirichlet-eps', type=float, default=0.25)
    parser.add_argument('--num-envs', type=int, default=32, help='Number of parallel env workers (>=1). 1 disables parallel mode')
    args = parser.parse_args()
    train_ppo(num_iterations=args.iters, horizon=max(40, int(args.horizon)), save_every=args.save_every, ckpt_path=args.ckpt,
              k_history=args.k_history, seed=args.seed,
              entropy_schedule_type=args.entropy_schedule, eval_every=args.eval_every, eval_games=args.eval_games,
              belief_particles=args.belief_particles, belief_ess_frac=args.belief_ess_frac,
              mcts_in_eval=args.mcts_eval, mcts_train=args.mcts_train, mcts_sims=args.mcts_sims, mcts_sims_eval=args.mcts_sims_eval, mcts_dets=args.mcts_dets, mcts_c_puct=args.mcts_c_puct,
              mcts_root_temp=args.mcts_root_temp, mcts_prior_smooth_eps=args.mcts_prior_smooth_eps,
              mcts_dirichlet_alpha=args.mcts_dirichlet_alpha, mcts_dirichlet_eps=args.mcts_dirichlet_eps,
              num_envs=args.num_envs)
