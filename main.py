import os
import threading
import torch as _torch
_train_dev = os.environ.get('SCOPONE_TRAIN_DEVICE', 'cpu')
if (_train_dev.startswith('cuda') and _torch.cuda.is_available()):
    _n_threads = int(os.environ.get('SCOPONE_TRAIN_THREADS', '2'))
    _n_interop = int(os.environ.get('SCOPONE_TRAIN_INTEROP_THREADS', '1'))
else:
    _cores = int(max(1, (os.cpu_count() or 1)))
    _target = max(1, int(_cores * 0.60))
    _n_threads = int(os.environ.get('SCOPONE_TRAIN_THREADS', str(_target)))
    _n_interop_default = max(1, _n_threads // 8)
    _n_interop = int(os.environ.get('SCOPONE_TRAIN_INTEROP_THREADS', str(_n_interop_default)))

# Silence TensorFlow/absl noise before any heavy imports
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # hide INFO/WARNING/ERROR from TF C++ logs
os.environ.setdefault('ABSL_LOGGING_MIN_LOG_LEVEL', '3')  # absl logs: only FATAL
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # disable oneDNN custom ops info spam

# Abilita TensorBoard di default (override con SCOPONE_DISABLE_TB=1 per disattivarlo)
os.environ.setdefault('SCOPONE_DISABLE_TB', '0')

## Abilita torch.compile di default per l'intero progetto (override via env)
os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
os.environ.setdefault('SCOPONE_TORCH_COMPILE_MODE', 'reduce-overhead')
os.environ.setdefault('SCOPONE_TORCH_COMPILE_BACKEND', 'inductor')
os.environ.setdefault('SCOPONE_COMPILE_VERBOSE', '1')


## Autotune controllabile: di default ON su CPU beneficia di fusioni; può essere disattivato via env
os.environ.setdefault('SCOPONE_INDUCTOR_AUTOTUNE', '1')
os.environ.setdefault('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM', '0')

## Evita graph break su .item() catturando scalari nei grafi
os.environ.setdefault('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', '1')

## Abilita dynamic shapes per ridurre errori di symbolic shapes FX
os.environ.setdefault('TORCHDYNAMO_DYNAMIC_SHAPES', '0')

## Alza il limite del cache di Dynamo per ridurre recompilazioni
os.environ.setdefault('TORCHDYNAMO_CACHE_SIZE_LIMIT', '32')

## Non impostare TORCH_LOGS ad un valore invalido; lascia al default o definisci mapping esplicito se necessario
# Abilita e blocca i flag dell'osservazione all'avvio (usati da observation/environment al load)
# Se l'utente li ha già impostati nel proprio run, li rispettiamo (setdefault)
os.environ.setdefault('OBS_INCLUDE_DEALER', '1')
os.environ.setdefault('OBS_INCLUDE_INFERRED', '0')
os.environ.setdefault('OBS_INCLUDE_RANK_PROBS', '0')
os.environ.setdefault('OBS_INCLUDE_SCOPA_PROBS', '0')

# Imposta ENV_DEVICE una sola volta coerente con SCOPONE_DEVICE o disponibilità CUDA
os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
os.environ.setdefault('ENV_DEVICE', 'cpu')

# Training compute device (models stay on CPU during env collection; moved only inside update)
os.environ.setdefault('SCOPONE_TRAIN_DEVICE', 'cpu')

# Enable approximate GELU and gate all runtime checks via a single flag
os.environ.setdefault('SCOPONE_APPROX_GELU', '1')
os.environ.setdefault('SCOPONE_STRICT_CHECKS', '0')
os.environ.setdefault('SCOPONE_PROFILE', '0')

# Additional trainer/eval tunables exposed via environment (defaults; override as needed)
os.environ.setdefault('SCOPONE_PAR_DEBUG', '0')
os.environ.setdefault('SCOPONE_WORKER_THREADS', '1')
os.environ.setdefault('SCOPONE_TORCH_PROF', '0')
os.environ.setdefault('SCOPONE_TORCH_TB_DIR', '')
os.environ.setdefault('SCOPONE_RPC_TIMEOUT_S', '30')
os.environ.setdefault('SCOPONE_RAISE_ON_INVALID_SIMS', '0')
os.environ.setdefault('SCOPONE_EP_PUT_TIMEOUT_S', '15')
os.environ.setdefault('SCOPONE_TORCH_PROF_DIR', 'profiles')
os.environ.setdefault('SCOPONE_RAISE_ON_CKPT_FAIL', '0')
os.environ.setdefault('ENABLE_BELIEF_SUMMARY', '0')
os.environ.setdefault('DET_NOISE', '0.0')
os.environ.setdefault('SCOPONE_COLLECT_MIN_BATCH', '0')
os.environ.setdefault('SCOPONE_COLLECT_MAX_LATENCY_MS', '3.0')
os.environ.setdefault('SCOPONE_COLLECTOR_STALL_S', '30')

# Gameplay/training topology flags
os.environ.setdefault('SCOPONE_START_OPP', os.environ.get('SCOPONE_START_OPP', 'top1'))

# Evaluation process knobs
os.environ.setdefault('SCOPONE_EVAL_DEBUG', '0')
os.environ.setdefault('SCOPONE_EVAL_MP_START', os.environ.get('SCOPONE_MP_START', 'forkserver'))
os.environ.setdefault('SCOPONE_EVAL_POOL_TIMEOUT_S', '600')
os.environ.setdefault('SCOPONE_ELO_DIFF_SCALE', '6.0')

# TQDM_DISABLE: 1=disattiva progress bar/logging di tqdm; 0=abilitato
os.environ.setdefault('TQDM_DISABLE', '0')

# SELFPLAY: 1=single net (self-play), 0=dual nets (Team A/B)
_selfplay_env = str(os.environ.get('SCOPONE_SELFPLAY', '1')).strip().lower()
_selfplay = (_selfplay_env in ['1', 'true', 'yes', 'on'])

# SCOPONE_OPP_FROZEN: 1=freeze the opponent, 0=co-train with the opponent
_opp_frozen = os.environ.get('SCOPONE_OPP_FROZEN','0') in ['1','true','yes','on']

# SCOPONE_TRAIN_FROM_BOTH_TEAMS: effective ONLY when SELFPLAY=1 and OPP_FROZEN=0.
# Uses transitions from both teams for the single net; otherwise ignored (on-policy).
_tfb = os.environ.get('SCOPONE_TRAIN_FROM_BOTH_TEAMS','0') in ['1','true','yes','on']

# Warm-start policy controlled by SCOPONE_WARM_START: '0' start-from-scratch, '1' force top1 clone, '2' use top2 if available
os.environ.setdefault('SCOPONE_WARM_START', '2')

# SCOPONE_ALTERNATE_ITERS: in dual-nets+frozen, train A for N iters then swap to B (and vice versa)
os.environ.setdefault('SCOPONE_ALTERNATE_ITERS', '1')

# SCOPONE_FROZEN_UPDATE_EVERY: in selfplay+frozen, refresh the shadow (frozen) opponent every N iters
os.environ.setdefault('SCOPONE_FROZEN_UPDATE_EVERY', '1')

# Refresh League from disk at startup (scan checkpoints/). 1=ON, 0=OFF
os.environ.setdefault('SCOPONE_LEAGUE_REFRESH', '0')

# Parallel eval workers: 1=serial, >1 parallel via multiprocessing
os.environ.setdefault('SCOPONE_EVAL_WORKERS', str(max(1, (os.cpu_count() or 1)//2)))

# Training flags (manual overrides available via env)
_save_every = int(os.environ.get('SCOPONE_SAVE_EVERY','10'))
os.environ.setdefault('SCOPONE_MINIBATCH', '4096')

# Checkpoint path control
os.environ.setdefault('SCOPONE_CKPT', 'checkpoints/ppo_ac.pth')

# Default to random seed for training runs (set -1); stable only if user sets it
seed_env = int(os.environ.get('SCOPONE_SEED', '-1'))

# Allow configuring iterations/horizon/num_envs via env; sensible defaults
iters = int(os.environ.get('SCOPONE_ITERS', '3'))
horizon = int(os.environ.get('SCOPONE_HORIZON', '16384'))
num_envs = int(os.environ.get('SCOPONE_NUM_ENVS', '8'))

# Read checkpoint path from env for training
ckpt_path_env = os.environ.get('SCOPONE_CKPT', 'checkpoints/ppo_ac.pth')

_mcts_warmup_iters = int(os.environ.get('SCOPONE_MCTS_WARMUP_ITERS', '500'))

# MCTS eval flags
_eval_every = int(os.environ.get('SCOPONE_EVAL_EVERY', '10'))
_eval_c_puct = float(os.environ.get('SCOPONE_EVAL_MCTS_C_PUCT','1.0'))
_eval_root_temp = float(os.environ.get('SCOPONE_EVAL_MCTS_ROOT_TEMP','0.0'))
_eval_prior_eps = float(os.environ.get('SCOPONE_EVAL_MCTS_PRIOR_SMOOTH_EPS','0.0'))
_eval_dir_alpha = float(os.environ.get('SCOPONE_EVAL_MCTS_DIRICHLET_ALPHA','0.25'))
_eval_dir_eps = float(os.environ.get('SCOPONE_EVAL_MCTS_DIRICHLET_EPS','0.25'))
_eval_belief_particles = int(os.environ.get('SCOPONE_EVAL_BELIEF_PARTICLES','0'))
_eval_belief_ess = float(os.environ.get('SCOPONE_EVAL_BELIEF_ESS_FRAC','0.5'))
_eval_use_mcts = os.environ.get('SCOPONE_EVAL_USE_MCTS','0').lower() in ['1','true','yes','on']
_eval_mcts_sims = int(os.environ.get('SCOPONE_EVAL_MCTS_SIMS','128'))
_eval_mcts_dets = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS','1'))
_eval_kh = int(os.environ.get('SCOPONE_EVAL_K_HISTORY','39'))
_eval_games = int(os.environ.get('SCOPONE_EVAL_GAMES','1000'))
_eval_max_games_per_chunk = int(os.environ.get('SCOPONE_EVAL_MAX_GAMES_PER_CHUNK', '32'))

# Training config flags (from env)
_entropy_sched = os.environ.get('SCOPONE_ENTROPY_SCHED','linear')
_belief_particles = int(os.environ.get('SCOPONE_BELIEF_PARTICLES','512'))
_belief_ess = float(os.environ.get('SCOPONE_BELIEF_ESS_FRAC','0.5'))
_mcts_c_puct = float(os.environ.get('SCOPONE_MCTS_C_PUCT','1.0'))
_mcts_root_temp = float(os.environ.get('SCOPONE_MCTS_ROOT_TEMP','0.0'))
_mcts_prior_eps = float(os.environ.get('SCOPONE_MCTS_PRIOR_SMOOTH_EPS','0.0'))
_mcts_dir_alpha = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_ALPHA','0.25'))
_mcts_dir_eps = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_EPS','0.25'))
_mcts_train = os.environ.get('SCOPONE_MCTS_TRAIN','0') in ['1','true','yes','on']
_mcts_sims = int(os.environ.get('SCOPONE_MCTS_SIMS','128'))
_mcts_dets = int(os.environ.get('SCOPONE_MCTS_DETS','4'))


# Targeted FD-level stderr filter to drop absl/TF CUDA registration warnings from C++
_SILENCE_ABSL = os.environ.get('SCOPONE_SILENCE_ABSL', '1') == '1'
if _SILENCE_ABSL:
    _SUPPRESS_SUBSTRINGS = (
        "All log messages before absl::InitializeLog() is called are written to STDERR",
        "Unable to register cuDNN factory",
        "Unable to register cuBLAS factory",
        "cuda_dnn.cc",
        "cuda_blas.cc",
    )

    _orig_fd2 = os.dup(2)
    _r_fd, _w_fd = os.pipe()
    os.dup2(_w_fd, 2)

    def _stderr_reader(r_fd, orig_fd, suppressed):
        with os.fdopen(r_fd, 'rb', buffering=0) as r:
            buffer = b""
            while True:
                chunk = r.read(1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    txt = line.decode('utf-8', errors='ignore')
                    if not any(s in txt for s in suppressed):
                        os.write(orig_fd, line + b"\n")
            if buffer:
                txt = buffer.decode('utf-8', errors='ignore')
                if not any(s in txt for s in suppressed):
                    os.write(orig_fd, buffer)

    _t = threading.Thread(target=_stderr_reader, args=(_r_fd, _orig_fd2, _SUPPRESS_SUBSTRINGS), daemon=True)
    _t.start()

os.environ.setdefault('ENV_DEVICE', os.environ.get('SCOPONE_DEVICE', 'cpu'))
## Imposta metodo mp sicuro per CUDA: forkserver (override con SCOPONE_MP_START)
os.environ.setdefault('SCOPONE_MP_START', 'forkserver')

import torch
from utils.device import get_compute_device
from tqdm import tqdm
from trainers.train_ppo import train_ppo
from utils.seed import resolve_seed

def _maybe_launch_tensorboard():
    """Launch TensorBoard in background if enabled.
    Controlled by env:
      - SCOPONE_AUTO_TB (default '1'): enable/disable auto launch
      - TB_LOGDIR (default 'runs'): log directory
      - TB_HOST (default '0.0.0.0'): bind host
      - TB_PORT (default '6006'): port
    """
    if os.environ.get('SCOPONE_DISABLE_TB', '0') == '1':
        return
    if os.environ.get('SCOPONE_AUTO_TB', '1') != '1':
        return
    logdir = os.environ.get('TB_LOGDIR', os.path.abspath('runs'))
    host = os.environ.get('TB_HOST', '0.0.0.0')
    port = os.environ.get('TB_PORT', '6006')
    import subprocess
    # Start TensorBoard as a detached background process
    cmd = ['tensorboard', '--logdir', logdir, '--host', host, '--port', str(port)]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tqdm.write(f"TensorBoard auto-started at http://localhost:{port}/ (logdir={logdir})")

# Minimal entrypoint: launch PPO training only
if __name__ == "__main__":
    device = get_compute_device()
    tqdm.write(f"Using device: {device}")
    tqdm.write(f"Training compute device: {os.environ.get('SCOPONE_TRAIN_DEVICE', 'cpu')}")
    tqdm.write(f"Self-play: {'ON' if _selfplay else 'OFF (League)'}")
    # Configure CPU threads for training in the main process only (workers handled in trainers/train_ppo.py)
    # On GPU training, keep minimal CPU threads to reduce host contention
    torch.set_num_threads(_n_threads)
    torch.set_num_interop_threads(_n_interop)
    tqdm.write(f"Training threads: num_threads={_n_threads} interop={_n_interop}")
    _maybe_launch_tensorboard()

    tqdm.write(f"Parallel envs: {num_envs}  (SCOPONE_PROFILE={os.environ.get('SCOPONE_PROFILE','0')})")
    tqdm.write(f"Train from both team transitions: {'ON' if _tfb else 'OFF'}")
    tqdm.write(f"Opponent frozen: {'ON' if _opp_frozen else 'OFF'}")
    tqdm.write(f"Warm start mode: {os.environ.get('SCOPONE_WARM_START','2')}")
    tqdm.write(f"League startup refresh: {'ON' if os.environ.get('SCOPONE_LEAGUE_REFRESH','1') in ['1','true','yes','on'] else 'OFF'}")
    tqdm.write(f"Eval cfg: games={_eval_games} mcts={'ON' if _eval_use_mcts else 'OFF'} sims={_eval_mcts_sims} dets={_eval_mcts_dets} kh={_eval_kh}")


    train_ppo(num_iterations=iters, horizon=horizon, k_history=_eval_kh,
              num_envs=num_envs,
              mcts_train=_mcts_train,
              mcts_sims=_mcts_sims,
              mcts_sims_eval=_eval_mcts_sims,
              save_every=_save_every,
              ckpt_path=ckpt_path_env,
              entropy_schedule_type=_entropy_sched,
              eval_every=_eval_every,
              mcts_in_eval=_eval_use_mcts,
              mcts_dets=_mcts_dets,
              mcts_c_puct=_mcts_c_puct,
              mcts_root_temp=_mcts_root_temp,
              mcts_prior_smooth_eps=_mcts_prior_eps,
              mcts_dirichlet_alpha=_mcts_dir_alpha,
              mcts_dirichlet_eps=_mcts_dir_eps,
              belief_particles=_belief_particles,
              belief_ess_frac=_belief_ess,
              eval_games=_eval_games,
              seed=seed_env,
              use_selfplay=_selfplay,
              train_both_teams=_tfb,
              mcts_warmup_iters=_mcts_warmup_iters)
