import os
import sys
import threading
import io

# Targeted FD-level stderr filter to drop absl/TF CUDA registration warnings from C++
_SILENCE_ABSL = os.environ.get('SCOPONE_SILENCE_ABSL', '1') == '1'
if _SILENCE_ABSL:
    try:
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
    except Exception as e:
        raise

# Silence TensorFlow/absl noise before any heavy imports
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # hide INFO/WARNING/ERROR from TF C++ logs
os.environ.setdefault('ABSL_LOGGING_MIN_LOG_LEVEL', '3')  # absl logs: only FATAL
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # disable oneDNN custom ops info spam
# Abilita TensorBoard di default (override con SCOPONE_DISABLE_TB=1 per disattivarlo)
os.environ.setdefault('SCOPONE_DISABLE_TB', '0')
## Abilita torch.compile di default per l'intero progetto (override via env)
os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
os.environ.setdefault('SCOPONE_TORCH_COMPILE_MODE', 'max-autotune')
os.environ.setdefault('SCOPONE_COMPILE_VERBOSE', '1')
## Disabilita max_autotune_gemm di Inductor (alcune GPU loggano warning inutili)
os.environ.setdefault('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM', '0')
## Evita graph break su .item() catturando scalari nei grafi
os.environ.setdefault('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', '1')
## Abilita dynamic shapes per ridurre errori di symbolic shapes FX
os.environ.setdefault('TORCHDYNAMO_DYNAMIC_SHAPES', '1')
## Alza il limite del cache di Dynamo per ridurre recompilazioni
os.environ.setdefault('TORCHDYNAMO_CACHE_SIZE_LIMIT', '32')
## Non impostare TORCH_LOGS ad un valore invalido; lascia al default o definisci mapping esplicito se necessario
# Abilita di default feature dell'osservazione
os.environ.setdefault('OBS_INCLUDE_DEALER', '1')
# Preferire env su GPU per eliminare copie H2D/D2H nell'interazione
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
    _maybe_launch_tensorboard()
    # Default to random seed for training runs (set -1); stable only if user sets it
    seed_env = int(os.environ.get('SCOPONE_SEED', '-1'))
    # Note: train_ppo will resolve and announce the effective seed
    train_ppo(num_iterations=10, horizon=2048, use_compact_obs=True, k_history=39,
              num_envs=1,
              mcts_sims=0,
              mcts_sims_eval=4,
              eval_every=0,
              mcts_in_eval=True,
              seed=seed_env)


