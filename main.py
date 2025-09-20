import os
import threading

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
os.environ.setdefault('SCOPONE_TRAIN_DEVICE', 'cuda')
# Enable approximate GELU and gate all runtime checks via a single flag
os.environ.setdefault('SCOPONE_APPROX_GELU', '1')
os.environ.setdefault('SCOPONE_STRICT_CHECKS', '0')
os.environ.setdefault('SCOPONE_PAR_PROFILE', '1')
# Default to random seed for training runs (set -1); stable only if user sets it
seed_env = int(os.environ.get('SCOPONE_SEED', '-1'))
# Allow configuring iterations/horizon/num_envs via env; sensible defaults
iters = int(os.environ.get('SCOPONE_ITERS', '3'))
horizon = int(os.environ.get('SCOPONE_HORIZON', '16384'))
num_envs = int(os.environ.get('SCOPONE_NUM_ENVS', '8'))

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

try:
    import torch as _t
    _env_def = 'cuda' if _t.cuda.is_available() else 'cpu'
except Exception:
    _env_def = 'cpu'
os.environ.setdefault('ENV_DEVICE', os.environ.get('SCOPONE_DEVICE', _env_def))
## Imposta metodo mp sicuro per CUDA: forkserver (override con SCOPONE_MP_START)
os.environ.setdefault('SCOPONE_MP_START', 'forkserver')

import torch
from utils.device import get_compute_device
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
    try:
        import subprocess
        # Start TensorBoard as a detached background process
        cmd = ['tensorboard', '--logdir', logdir, '--host', host, '--port', str(port)]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"TensorBoard auto-started at http://localhost:{port}/ (logdir={logdir})")
    except Exception:
        # Best-effort: do not block training if TB isn't available
        pass

# Minimal entrypoint: launch PPO training only
if __name__ == "__main__":
    device = get_compute_device()
    print(f"Using device: {device}")
    print(f"Training compute device: {os.environ.get('SCOPONE_TRAIN_DEVICE', 'cpu')}")
    #"""
    # Configure CPU threads for training in the main process only (workers handled in trainers/train_ppo.py)
    # On GPU training, keep minimal CPU threads to reduce host contention
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
    torch.set_num_threads(_n_threads)
    torch.set_num_interop_threads(_n_interop)
    print(f"Training threads: num_threads={_n_threads} interop={_n_interop}")
    #"""
    _maybe_launch_tensorboard()

    print(f"Parallel envs: {num_envs}  (SCOPONE_PAR_PROFILE={os.environ.get('SCOPONE_PAR_PROFILE','0')})")
    # Enable optional GPU for heavy training compute while env remains on CPU.
    # Use env SCOPONE_TRAIN_DEVICE=cuda to turn it on; default is CPU.
    train_ppo(num_iterations=iters, horizon=horizon, k_history=39,
              num_envs=num_envs,
              mcts_sims=0,
              mcts_sims_eval=0,
              eval_every=0,
              mcts_in_eval=False,
              seed=seed_env)



