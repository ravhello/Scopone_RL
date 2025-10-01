import os
import sys
import argparse
import threading
import torch as _torch

# Ensure project root on sys.path for module imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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
_opp_frozen = os.environ.get('SCOPONE_OPP_FROZEN', '0') in ['1', 'true', 'yes', 'on']

# SCOPONE_TRAIN_FROM_BOTH_TEAMS: effective ONLY when SELFPLAY=1 and OPP_FROZEN=0.
# Uses transitions from both teams for the single net; otherwise ignored (on-policy).
_tfb = os.environ.get('SCOPONE_TRAIN_FROM_BOTH_TEAMS', '0') in ['1', 'true', 'yes', 'on']

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
_save_every = int(os.environ.get('SCOPONE_SAVE_EVERY', '10'))
os.environ.setdefault('SCOPONE_MINIBATCH', '4096')

# Checkpoint path control
os.environ.setdefault('SCOPONE_CKPT', 'checkpoints/ppo_ac.pth')

# Default to random seed for training runs (set -1); stable only if user sets it
seed_env = int(os.environ.get('SCOPONE_SEED', '-1'))

# Allow configuring iterations/horizon/num_envs via env; sensible defaults
iters = int(os.environ.get('SCOPONE_ITERS', '1'))
horizon = int(os.environ.get('SCOPONE_HORIZON', '16384'))
_DEFAULT_NUM_ENVS = int(os.environ.get('SCOPONE_NUM_ENVS', '1'))

# Read checkpoint path from env for training
ckpt_path_env = os.environ.get('SCOPONE_CKPT', 'checkpoints/ppo_ac.pth')

_mcts_warmup_iters = int(os.environ.get('SCOPONE_MCTS_WARMUP_ITERS', '500'))

# MCTS eval flags
_eval_every = int(os.environ.get('SCOPONE_EVAL_EVERY', '10'))
_eval_c_puct = float(os.environ.get('SCOPONE_EVAL_MCTS_C_PUCT', '1.0'))
_eval_root_temp = float(os.environ.get('SCOPONE_EVAL_MCTS_ROOT_TEMP', '0.0'))
_eval_prior_eps = float(os.environ.get('SCOPONE_EVAL_MCTS_PRIOR_SMOOTH_EPS', '0.0'))
_eval_dir_alpha = float(os.environ.get('SCOPONE_EVAL_MCTS_DIRICHLET_ALPHA', '0.25'))
_eval_dir_eps = float(os.environ.get('SCOPONE_EVAL_MCTS_DIRICHLET_EPS', '0.25'))
_eval_belief_particles = int(os.environ.get('SCOPONE_EVAL_BELIEF_PARTICLES', '0'))
_eval_belief_ess = float(os.environ.get('SCOPONE_EVAL_BELIEF_ESS_FRAC', '0.5'))
_eval_use_mcts = os.environ.get('SCOPONE_EVAL_USE_MCTS', '0').lower() in ['1', 'true', 'yes', 'on']
_eval_mcts_sims = int(os.environ.get('SCOPONE_EVAL_MCTS_SIMS', '128'))
_eval_mcts_dets = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS', '1'))
_eval_kh = int(os.environ.get('SCOPONE_EVAL_K_HISTORY', '39'))
_eval_games = int(os.environ.get('SCOPONE_EVAL_GAMES', '1000'))

# Training config flags (from env)
_entropy_sched = os.environ.get('SCOPONE_ENTROPY_SCHED', 'linear')
_belief_particles = int(os.environ.get('SCOPONE_BELIEF_PARTICLES', '512'))
_belief_ess = float(os.environ.get('SCOPONE_BELIEF_ESS_FRAC', '0.5'))
_mcts_c_puct = float(os.environ.get('SCOPONE_MCTS_C_PUCT', '1.0'))
_mcts_root_temp = float(os.environ.get('SCOPONE_MCTS_ROOT_TEMP', '0.0'))
_mcts_prior_eps = float(os.environ.get('SCOPONE_MCTS_PRIOR_SMOOTH_EPS', '0.0'))
_mcts_dir_alpha = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_ALPHA', '0.25'))
_mcts_dir_eps = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_EPS', '0.25'))
_mcts_train = os.environ.get('SCOPONE_MCTS_TRAIN', '0') in ['1', 'true', 'yes', 'on']
_mcts_sims = int(os.environ.get('SCOPONE_MCTS_SIMS', '128'))
_mcts_dets = int(os.environ.get('SCOPONE_MCTS_DETS', '4'))

# Targeted FD-level stderr filter to drop absl/TF CUDA registration warnings from C++
_SILENCE_ABSL = os.environ.get('SCOPONE_SILENCE_ABSL', '1') == '1'
if os.environ.get('SCALENE_RUNNING', '0') == '1':  # scalene needs raw stderr
    _SILENCE_ABSL = False
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

import io
import torch
import subprocess
import webbrowser
from datetime import datetime
import platform

# Default CPU for profiling unless user sets GPU via env

from trainers.train_ppo import train_ppo
from utils.seed import resolve_seed


def _resolve_num_envs(args, clamp: int | None = None) -> int:
    raw = getattr(args, 'num_envs', None)
    if raw is None:
        val = _DEFAULT_NUM_ENVS
    else:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            val = _DEFAULT_NUM_ENVS
    if clamp is not None:
        val = min(val, clamp)
    return max(1, val)


def main():
    parser = argparse.ArgumentParser(description='Profile short PPO run (torch or line-level).')
    parser.add_argument('--iters', type=int, default=iters, help='Iterations to run (default from SCOPONE_ITERS)')
    parser.add_argument('--horizon', type=int, default=horizon, help='Rollout horizon per iteration (default from SCOPONE_HORIZON)')
    parser.add_argument('--line', dest='line', action='store_true', default=False, help='Enable line-by-line profiler with per-line timings (default: on)')
    parser.add_argument('--no-line', dest='line', action='store_false', help='Disable line-by-line profiler')
    parser.add_argument('--wrap-update', dest='wrap_update', action='store_true', default=True, help='Also profile ActionConditionedPPO.update (default: on; slower)')
    parser.add_argument('--no-wrap-update', dest='wrap_update', action='store_false', help='Disable profiling of ActionConditionedPPO.update')
    parser.add_argument('--report', action='store_true', help='Print extended line-profiler report')
    parser.add_argument('--num-envs', type=int, default=_DEFAULT_NUM_ENVS, help='Number of parallel environments (default from SCOPONE_NUM_ENVS)')
    # Line-profiler scope controls
    parser.add_argument('--add-func', action='append', default=[], help='Qualified function or method to include, e.g. pkg.mod:Class.method or algorithms.ppo_ac:ActionConditionedPPO.update')
    parser.add_argument('--add-module', action='append', default=[], help='Module to include all functions from, e.g. algorithms.ppo_ac or environment')
    parser.add_argument('--profile-all', dest='profile_all', action='store_true', default=False, help='Profile all modules under project root (heavy)')
    parser.add_argument('--no-profile-all', dest='profile_all', action='store_false', help=argparse.SUPPRESS)
    parser.add_argument('--line-full', dest='line_full', action='store_true', default=False, help='Show all per-line rows (not just top 30)')
    parser.add_argument('--line-csv', dest='line_csv', type=str, default=None, help='Path to write full per-line CSV (file, line, hits, cpu_s, gpu_s, transfer_s)')
    parser.add_argument('--line-self-only', dest='line_self_only', action='store_true', default=False, help='Use only self CPU time (drop lines without self time)')
    parser.add_argument('--cprofile', action='store_true', default=False, help='Use Python cProfile instead of torch or line-profiler')
    parser.add_argument('--cprofile-out', type=str, default=None, help='Output path for cProfile stats file (.prof). Default: timestamped file')
    parser.add_argument('--snakeviz', dest='snakeviz', action='store_true', default=True, help='Open SnakeViz on the generated .prof (default: on)')
    parser.add_argument('--no-snakeviz', dest='snakeviz', action='store_false', help='Do not open SnakeViz after profiling')
    parser.add_argument('--scalene', action='store_true', default=False, help='Run training under Scalene (CLI by default)')
    parser.add_argument('--scalene-out', type=str, default=None, help='Output path base for Scalene report. If ends with .html, generate HTML; else CLI only. Default: timestamped base')
    parser.add_argument('--scalene-open', dest='scalene_open', action='store_true', default=True, help='Open Scalene HTML report in a browser (default: on)')
    parser.add_argument('--no-scalene-open', dest='scalene_open', action='store_false', help='Do not open Scalene report after profiling')
    parser.add_argument('--scalene-cli', dest='scalene_cli', action='store_true', default=True, help='Print Scalene text report to terminal (default: on)')
    parser.add_argument('--no-scalene-cli', dest='scalene_cli', action='store_false', help='Do not print Scalene text report to terminal')
    parser.add_argument('--scalene-cpu-only', dest='scalene_cpu_only', action='store_true', default=True, help='Limit Scalene to CPU profiling only (default: on)')
    parser.add_argument('--no-scalene-cpu-only', dest='scalene_cpu_only', action='store_false', help='Include memory/GPU metrics (disables CPU-only)')
    parser.add_argument('--scalene-gpu-modes', dest='scalene_gpu_modes', action='store_true', default=False, help='Attempt to enable per-process GPU accounting via scalene.set_nvidia_gpu_modes (default: off)')
    parser.add_argument('--no-scalene-gpu-modes', dest='scalene_gpu_modes', action='store_false', help='Do not attempt to set NVIDIA GPU modes for Scalene')
    parser.add_argument('--scalene-run', action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('--torch-profiler', dest='torch_profiler', action='store_true', default=False, help='Use PyTorch profiler (no default)')
    args = parser.parse_args()
    # Default to random seed for profiling runs; allow override via env/CLI passthrough
    seed_env = int(os.environ.get('SCOPONE_SEED', '-1'))
    seed = resolve_seed(seed_env)

    # Ensure default profiles directory exists for profiler outputs
    DEFAULT_PROFILES_DIR = os.path.abspath(os.path.join(ROOT, 'profiles'))
    try:
        os.makedirs(DEFAULT_PROFILES_DIR, exist_ok=True)
    except Exception:
        pass

    # If requested, re-exec this script under Scalene to produce an HTML report.
    if getattr(args, 'scalene', False) and not getattr(args, 'scalene_run', False):
        script_path = os.path.abspath(__file__)
        # Determine output mode and HTML path based on --scalene-out
        want_html = False
        html_path = None
        if getattr(args, 'scalene_out', None):
            val = str(args.scalene_out).strip().lower()
            if val == 'html':
                want_html = True
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Default directory: profiles/; default name: ppo_scalene_<timestamp>.html
                html_path = os.path.abspath(os.path.join(DEFAULT_PROFILES_DIR, f'ppo_scalene_{ts}.html'))
            elif val == 'cli':
                want_html = False
            else:
                _raw = os.path.abspath(args.scalene_out)
                _, ext = os.path.splitext(_raw)
                if ext.lower() == '.html':
                    want_html = True
                    html_path = _raw
        else:
            # No explicit output provided: default to HTML in profiles/
            want_html = True
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_path = os.path.abspath(os.path.join(DEFAULT_PROFILES_DIR, f'ppo_scalene_{ts}.html'))
        # Ensure directory exists for HTML output
        if want_html and html_path:
            out_dir = os.path.dirname(html_path) or DEFAULT_PROFILES_DIR
            os.makedirs(out_dir, exist_ok=True)
        ne = _resolve_num_envs(args)
        scalene_cmd = [sys.executable, '-m', 'scalene']
        if getattr(args, 'scalene_cpu_only', True):
            scalene_cmd.append('--cpu-only')
        if getattr(args, 'scalene_gpu_modes', False):
            try:
                proc = subprocess.run([sys.executable, '-m', 'scalene.set_nvidia_gpu_modes'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode != 0:
                    is_wsl = ('WSL_INTEROP' in os.environ) or ('microsoft' in platform.release().lower())
                    if is_wsl:
                        print('Scalene GPU modes not supported on WSL2; skipping.')
                    else:
                        print('Failed to set NVIDIA GPU modes for Scalene (non-zero exit). Try: sudo python -m scalene.set_nvidia_gpu_modes')
            except FileNotFoundError:
                print('Scalene GPU modes helper not found. Update scalene to a recent version.')
            except Exception as e:
                print(f'Failed to set NVIDIA GPU modes for Scalene: {e}')
        # Decide HTML vs CLI
        if want_html and html_path:
            scalene_cmd += ['--html', '--reduced-profile', '--outfile', html_path]
        else:
            # CLI only; request CLI output explicitly
            scalene_cmd += ['--cli', '--cpu-percent-threshold', '0', '--malloc-threshold', '1000000000']
        # Include all modules (not only the executed file's dir) and restrict to project root
        scalene_cmd += ['--profile-all', '--profile-only', ROOT]
        scalene_cmd += [
            script_path,
            '--scalene-run',
            '--no-line',
            '--no-wrap-update',
            '--iters', str(max(0, args.iters)),
            '--horizon', str(max(40, args.horizon)),
            '--num-envs', str(ne),
        ]
        # propagate selfplay via env
        env = os.environ.copy()
        env['SCOPONE_SEED'] = str(seed)
        env['SCALENE_RUNNING'] = '1'
        env['SCOPONE_SILENCE_ABSL'] = '0'
        env['TQDM_DISABLE'] = '0'
        # read desired selfplay from env or default to ON; do not override here
        # keep SCOPONE_SELFPLAY as-is so outer env controls behavior
        print("Running under Scalene... this may add overhead.")
        try:
            # Stream output live while capturing for summary at the end
            proc = subprocess.Popen(scalene_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            captured_lines = []
            assert proc.stdout is not None
            # Suppress frequent Scalene JSON validation/pydantic warnings and absl CUDA factory spam
            def _suppress_scalene_line(txt: str) -> bool:
                txt_low = txt.lower()
                if ('json failed validation' in txt_low) or ('validation error for' in txt_low):
                    return True
                if ('n_core_utilization' in txt_low) or ('scalenejsonschema' in txt_low):
                    return True
                if ('errors.pydantic.dev' in txt_low):
                    return True
                # Newer Scalene prints just the validation rows without headers
                if ('input should be less than or equal to 1' in txt_low) or ('less_than_equal' in txt_low):
                    return True
                # absl/CUDA registration noise
                if ('unable to register cudnn factory' in txt_low) or ('unable to register cublas factory' in txt_low):
                    return True
                if ('all log messages before absl::initializelog() is called are written to stderr'.lower() in txt_low):
                    return True
                return False
            for _ln in proc.stdout:
                if _suppress_scalene_line(_ln):
                    continue
                print(_ln, end='')
                captured_lines.append(_ln)
            proc.wait()
            full_cli_output = ''.join(captured_lines)
        except FileNotFoundError:
            print("Scalene is not installed. Install with: pip install scalene")
        except Exception as e:
            print(f"Failed to run Scalene: {e}")
        else:
            if 'full_cli_output' in locals() and full_cli_output:
                # Save full CLI output to file to avoid flooding terminal
                ts_cli = datetime.now().strftime('%Y%m%d_%H%M%S')
                cli_out_path = os.path.abspath(os.path.join(DEFAULT_PROFILES_DIR, f'ppo_scalene_{ts_cli}_cli.txt'))
                with open(cli_out_path, 'w', encoding='utf-8') as f:
                    f.write(full_cli_output)
                print("\nScalene CLI saved to:", cli_out_path)

                # Print compact, line-profiler-like summary at the bottom
                print("\n===== Scalene — Compact summary (by file) =====")
                clean_output = full_cli_output  # ensure defined for fallback paths
                try:
                    import re as _re

                    def _strip_ansi(txt: str) -> str:
                        return _re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", txt)

                    def _to_rel(pth: str) -> str:
                        ap = os.path.abspath(pth)
                        if ROOT in ap:
                            return os.path.relpath(ap, ROOT)
                        return ap

                    def _parse_secs(s: str) -> float:
                        s = s.strip()
                        total = 0.0
                        for m in _re.findall(r'(\d+(?:\.\d+)?)\s*([hms])', s):
                            val = float(m[0])
                            unit = m[1]
                            if unit == 'h':
                                total += val * 3600.0
                            elif unit == 'm':
                                total += val * 60.0
                            else:
                                total += val
                        if total == 0.0:
                            mmss = _re.match(r'(?:(\d+)m:)?(\d+(?:\.\d+)?)s', s)
                            if mmss:
                                mins = float(mmss.group(1) or '0')
                                secs = float(mmss.group(2))
                                total = mins * 60.0 + secs
                        return total

                    clean_output = _strip_ansi(full_cli_output)
                    per_file = {}
                    for line in clean_output.splitlines():
                        if '.py: % of time' in line:
                            before, after = line.split(': % of time', 1)
                            file_abs = before.strip()
                            # seconds may be absent in some builds
                            m_pct = _re.search(r'=\s*([0-9]+(?:\.[0-9]+)?)%\s*(?:\(([^\)]*)\))?', after)
                            if not m_pct:
                                continue
                            pct = float(m_pct.group(1))
                            secs = _parse_secs(m_pct.group(2)) if (m_pct.lastindex and m_pct.lastindex >= 2) else 0.0
                            per_file[_to_rel(file_abs)] = {'seconds': secs, 'percent': pct}

                    if per_file:
                        ranked = sorted(per_file.items(), key=lambda kv: kv[1]['seconds'], reverse=True)
                        for i, (fn, s) in enumerate(ranked[:30], 1):
                            print(f"{i:>2}. {fn}\n    Python: {s['seconds']:8.2f}s  ({s['percent']:5.1f}%)")
                        # Top lines across files (approximate absolute time via per-file totals * per-line percentage)
                        print("\n===== Scalene — Top lines (by total time) =====")
                        per_site = {}
                        current_file = None
                        current_file_secs = 0.0
                        for _line in clean_output.splitlines():
                            m_file2 = _re.match(r"^\s*(/.*?\.py):\s*% of time\s*=\s*([0-9.]+)%\s*(?:\(([^)]*)\))?", _line)
                            if m_file2:
                                current_file = _to_rel(m_file2.group(1).strip())
                                current_file_secs = _parse_secs(m_file2.group(3)) if (m_file2.lastindex and m_file2.lastindex >= 3) else 0.0
                                continue
                            if not current_file or current_file_secs <= 0.0:
                                continue
                            m_row = _re.match(r"^\s*(\d+)\s*[│|]\s*([^│|]*)[│|]([^│|]*)[│|]([^│|]*)[│|]", _line)
                            if not m_row:
                                continue
                            ln_no = int(m_row.group(1))
                            def _pct_to_float(txt: str) -> float:
                                t = txt.strip()
                                m = _re.search(r"([0-9]+(?:\.[0-9]+)?)%", t)
                                return float(m.group(1)) if m else 0.0
                            py_pct = _pct_to_float(m_row.group(2))
                            na_pct = _pct_to_float(m_row.group(3))
                            sy_pct = _pct_to_float(m_row.group(4))
                            total_pct = py_pct + na_pct + sy_pct
                            if total_pct <= 0.0:
                                continue
                            secs = current_file_secs * (total_pct / 100.0)
                            key = (current_file, ln_no)
                            agg = per_site.get(key)
                            if not agg:
                                per_site[key] = {'seconds': 0.0, 'py_pct': 0.0, 'na_pct': 0.0, 'sy_pct': 0.0}
                                agg = per_site[key]
                            agg['seconds'] += secs
                            agg['py_pct'] += py_pct
                            agg['na_pct'] += na_pct
                            agg['sy_pct'] += sy_pct
                        if per_site:
                            ranked_sites = sorted(per_site.items(), key=lambda kv: kv[1]['seconds'], reverse=True)
                            for i, ((fn, ln_no), svals) in enumerate(ranked_sites[:30], 1):
                                print(f"{i:>2}. {fn}:{ln_no}  total: {svals['seconds']:8.2f}s  (py {svals['py_pct']:5.1f}%, nat {svals['na_pct']:5.1f}%, sys {svals['sy_pct']:5.1f}%)")
                        else:
                            # Fallback: parse 'function summary' blocks and attribute function % to definition line
                            per_site2 = {}
                            last_file = None
                            last_file_secs = 0.0
                            in_func_summary = False
                            for _line in clean_output.splitlines():
                                mf = _re.match(r"^\s*(/.*?\.py):\s*% of time\s*=\s*([0-9.]+)%\s*(?:\(([^)]*)\))?", _line)
                                if mf:
                                    last_file = _to_rel(mf.group(1).strip())
                                    last_file_secs = _parse_secs(mf.group(3)) if (mf.lastindex and mf.lastindex >= 3) else 0.0
                                    in_func_summary = False
                                    continue
                                if 'function summary for' in _line:
                                    in_func_summary = True
                                    continue
                                if in_func_summary:
                                    mfr = _re.match(r"^\s*(\d+)\s*[│|]\s*([^│|]*)[│|]([^│|]*)[│|]([^│|]*)[│|]", _line)
                                    if mfr and last_file and last_file_secs > 0.0:
                                        ln_no = int(mfr.group(1))
                                        def _pctf(txt: str) -> float:
                                            mm = _re.search(r"([0-9]+(?:\.[0-9]+)?)%", txt.strip())
                                            return float(mm.group(1)) if mm else 0.0
                                        py = _pctf(mfr.group(2)); na = _pctf(mfr.group(3)); sy = _pctf(mfr.group(4))
                                        tot = py + na + sy
                                        if tot <= 0.0:
                                            continue
                                        secs = last_file_secs * (tot / 100.0)
                                        key = (last_file, ln_no)
                                        agg = per_site2.get(key)
                                        if not agg:
                                            per_site2[key] = {'seconds': 0.0, 'py_pct': 0.0, 'na_pct': 0.0, 'sy_pct': 0.0}
                                            agg = per_site2[key]
                                        agg['seconds'] += secs
                                        agg['py_pct'] += py
                                        agg['na_pct'] += na
                                        agg['sy_pct'] += sy
                                    else:
                                        # Heuristic end of block
                                        if _line.strip().startswith('=====') or '.py:' in _line:
                                            in_func_summary = False
                            if per_site2:
                                ranked_sites = sorted(per_site2.items(), key=lambda kv: kv[1]['seconds'], reverse=True)
                                for i, ((fn, ln_no), svals) in enumerate(ranked_sites[:30], 1):
                                    print(f"{i:>2}. {fn}:{ln_no}  total: {svals['seconds']:8.2f}s  (py {svals['py_pct']:5.1f}%, nat {svals['na_pct']:5.1f}%, sys {svals['sy_pct']:5.1f}%)")
                            else:
                                print("(no per-line/function rows parsed; try increasing iters/horizon)")
                    else:
                        # Fallback: print last 80 lines of raw output
                        tail = '\n'.join(clean_output.splitlines()[-80:])
                        print(tail)
                except Exception:
                    # On parser error, still show the tail of cleaned output or raw output
                    tail = '\n'.join((clean_output or full_cli_output).splitlines()[-80:])
                    print(tail)

            if want_html and html_path:
                print(f"\nScalene HTML report: {html_path}")
                if getattr(args, 'scalene_open', True):
                    url = 'file://' + html_path if not html_path.startswith('file://') else html_path
                    opened = webbrowser.open(url)
                    if not opened:
                        subprocess.Popen(['xdg-open', html_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("If it didn't open, open the HTML file manually in your browser.")
        return

    # Inner run invoked by Scalene: execute training without additional profilers.
    if getattr(args, 'scalene_run', False):
        num_envs_eff = _resolve_num_envs(args)
        _selfplay_env = str(os.environ.get('SCOPONE_SELFPLAY', '1')).strip().lower()
        _selfplay = (_selfplay_env in ['1', 'true', 'yes', 'on'])
        # Training flags for profiling as well
        _save_every = int(os.environ.get('SCOPONE_SAVE_EVERY','10'))
        _entropy_sched = os.environ.get('SCOPONE_ENTROPY_SCHED','linear')
        _belief_particles = int(os.environ.get('SCOPONE_BELIEF_PARTICLES','512'))
        _belief_ess = float(os.environ.get('SCOPONE_BELIEF_ESS_FRAC','0.5'))
        _mcts_train = os.environ.get('SCOPONE_MCTS_TRAIN','0') in ['1','true','yes','on']
        _mcts_sims = int(os.environ.get('SCOPONE_MCTS_SIMS','128'))
        _mcts_dets = int(os.environ.get('SCOPONE_MCTS_DETS','4'))
        _mcts_c_puct = float(os.environ.get('SCOPONE_MCTS_C_PUCT','1.0'))
        _mcts_root_temp = float(os.environ.get('SCOPONE_MCTS_ROOT_TEMP','0.0'))
        _mcts_prior_eps = float(os.environ.get('SCOPONE_MCTS_PRIOR_SMOOTH_EPS','0.0'))
        _mcts_dir_alpha = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_ALPHA','0.25'))
        _mcts_dir_eps = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_EPS','0.25'))
        _eval_games = int(os.environ.get('SCOPONE_EVAL_GAMES','1000'))
        train_ppo(num_iterations=max(0, args.iters), horizon=max(40, args.horizon), k_history=39, num_envs=num_envs_eff,
                  mcts_sims=_mcts_sims, mcts_sims_eval=0, save_every=_save_every,
                  entropy_schedule_type=_entropy_sched,
                  eval_every=0, eval_games=_eval_games, mcts_in_eval=False, seed=seed, use_selfplay=_selfplay,
                  mcts_warmup_iters=_mcts_warmup_iters)
        return

    # cProfile mode takes precedence over line/torch profiler
    if getattr(args, 'cprofile', False):
        import cProfile
        import pstats

        prof = cProfile.Profile()
        # Honor training/profile env config (consistent with other modes)
        _save_every = int(os.environ.get('SCOPONE_SAVE_EVERY','10'))
        _entropy_sched = os.environ.get('SCOPONE_ENTROPY_SCHED','linear')
        _belief_particles = int(os.environ.get('SCOPONE_BELIEF_PARTICLES','512'))
        _belief_ess = float(os.environ.get('SCOPONE_BELIEF_ESS_FRAC','0.5'))
        _mcts_train = os.environ.get('SCOPONE_MCTS_TRAIN','0') in ['1','true','yes','on']
        _mcts_sims = int(os.environ.get('SCOPONE_MCTS_SIMS','128'))
        _mcts_dets = int(os.environ.get('SCOPONE_MCTS_DETS','4'))
        _mcts_c_puct = float(os.environ.get('SCOPONE_MCTS_C_PUCT','1.0'))
        _mcts_root_temp = float(os.environ.get('SCOPONE_MCTS_ROOT_TEMP','0.0'))
        _mcts_prior_eps = float(os.environ.get('SCOPONE_MCTS_PRIOR_SMOOTH_EPS','0.0'))
        _mcts_dir_alpha = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_ALPHA','0.25'))
        _mcts_dir_eps = float(os.environ.get('SCOPONE_MCTS_DIRICHLET_EPS','0.25'))
        _eval_games = int(os.environ.get('SCOPONE_EVAL_GAMES','1000'))

        num_envs_eff = _resolve_num_envs(args)

        def _run():
            _selfplay_env = str(os.environ.get('SCOPONE_SELFPLAY', '1')).strip().lower()
            _selfplay = (_selfplay_env in ['1', 'true', 'yes', 'on'])
            train_ppo(num_iterations=max(0, args.iters), horizon=max(40, args.horizon), k_history=39, num_envs=num_envs_eff,
                      save_every=_save_every,
                      entropy_schedule_type=_entropy_sched,
                      belief_particles=_belief_particles, belief_ess_frac=_belief_ess,
                      mcts_in_eval=False, mcts_train=_mcts_train,
                      mcts_sims=_mcts_sims, mcts_sims_eval=0, mcts_dets=_mcts_dets, mcts_c_puct=_mcts_c_puct,
                      mcts_root_temp=_mcts_root_temp, mcts_prior_smooth_eps=_mcts_prior_eps,
                      mcts_dirichlet_alpha=_mcts_dir_alpha, mcts_dirichlet_eps=_mcts_dir_eps,
                      eval_every=0, eval_games=_eval_games,
                      seed=seed, use_selfplay=_selfplay,
                      mcts_warmup_iters=_mcts_warmup_iters)

        prof.enable()
        try:
            _run()
        finally:
            prof.disable()

        if getattr(args, 'cprofile_out', None):
            out_path = args.cprofile_out
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = os.path.abspath(os.path.join(DEFAULT_PROFILES_DIR, f'ppo_profile_{ts}.prof'))
        try:
            prof.dump_stats(out_path)
            print(f"\ncProfile stats written to: {out_path}")
        except Exception as e:
            print(f"Failed to write cProfile stats: {e}")

        # Print concise summaries (top by cumulative and self time)
        try:
            import io as _io
            s1 = _io.StringIO()
            ps = pstats.Stats(prof, stream=s1).sort_stats('cumtime')
            ps.print_stats(30)
            print("\nTop functions by cumulative time (cumtime):\n" + s1.getvalue())

            s2 = _io.StringIO()
            pstats.Stats(prof, stream=s2).sort_stats('tottime').print_stats(30)
            print("\nTop functions by self time (tottime):\n" + s2.getvalue())
        except Exception as e:
            print(f"Failed to print cProfile summary: {e}")

        # Optionally launch SnakeViz
        if getattr(args, 'snakeviz', False):
            try:
                # Prefer python -m snakeviz to avoid PATH issues
                subprocess.Popen([sys.executable, '-m', 'snakeviz', out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("Launched SnakeViz in background. If not opening, run: snakeviz " + out_path)
            except Exception as e:
                print(f"Could not launch SnakeViz automatically: {e}\nInstall with: pip install snakeviz\nThen run: snakeviz {out_path}")
        return

    if args.line:
        # Lightweight line-by-line profiling for Python code with file:line output
        from profilers.line_profiler import profile as line_profile, global_profiler
        # Helper to preserve original callable signatures for the type checker
        from typing import Any, Callable, TypeVar, cast
        _F = TypeVar('_F', bound=Callable[..., object])
        def _lp(fn: _F) -> _F:
            return cast(_F, line_profile(fn))

        # Optional: register additional targets for profiling before running training
        def _resolve_attr(path: str):
            import importlib as _importlib
            if ':' in path:
                mod_path, attr_path = path.split(':', 1)
            else:
                parts = path.split('.')
                for i in range(len(parts), 0, -1):
                    mod_path = '.'.join(parts[:i])
                    _importlib.import_module(mod_path)
                    attr_path = '.'.join(parts[i:])
                    break
                else:
                    mod_path, attr_path = path, ''
            mod = _importlib.import_module(mod_path)
            if not attr_path:
                return mod
            obj = mod
            for name in attr_path.split('.'):
                obj = getattr(obj, name)
            return obj

        def _iter_module_functions(module):
            import inspect as _inspect
            for name, obj in _inspect.getmembers(module):
                if _inspect.isfunction(obj) and getattr(obj, '__module__', None) == module.__name__:
                    yield obj
                if _inspect.isclass(obj) and getattr(obj, '__module__', None) == module.__name__:
                    for _, member in _inspect.getmembers(obj):
                        if _inspect.isfunction(member) and getattr(member, '__qualname__', '').startswith(obj.__name__ + '.'):
                            yield member

        def _iter_package_functions(pkg_module):
            import pkgutil as _pkgutil
            import importlib as _importlib
            for _finder, _name, _is_pkg in _pkgutil.walk_packages(pkg_module.__path__, pkg_module.__name__ + "."):
                submod = _importlib.import_module(_name)
                for fn in _iter_module_functions(submod):
                    yield fn

        # Register user-specified functions and modules
        any_registered = False
        if getattr(args, 'add_func', None):
            for spec in (args.add_func or []):
                obj = _resolve_attr(spec)
                if obj is None:
                    continue
                line_profile(getattr(obj, '__func__', obj))
                any_registered = True
        if getattr(args, 'add_module', None):
            for mod_spec in (args.add_module or []):
                mod = _resolve_attr(mod_spec)
                if mod is None:
                    continue
                for fn in _iter_module_functions(mod):
                    line_profile(fn)
                    any_registered = True

        # If requested, register most project modules/functions automatically (heavy)
        if getattr(args, 'profile_all', False):
            import importlib as _importlib
            default_targets = [
                'environment',
                'observation',
                'benchmark',
                'main',
                'algorithms',
                'models',
                'trainers',
                'evaluation',
            ]
            for tgt in default_targets:
                mod = _importlib.import_module(tgt)
                # Package: recurse; Module: shallow
                if hasattr(mod, '__path__'):
                    for fn in _iter_package_functions(mod):
                        line_profile(fn)
                        any_registered = True
        # Wrap hotspots and also enable global tracing fallback
        import trainers.train_ppo as train_mod
        train_mod.collect_trajectory = _lp(train_mod.collect_trajectory)
        train_fn = _lp(train_mod.train_ppo)
        # Trainer data path internals
        train_mod.collect_trajectory_parallel = _lp(train_mod.collect_trajectory_parallel)
        train_mod._batched_select_indices = _lp(train_mod._batched_select_indices)
        train_mod._batched_select_indices_with_actor = _lp(train_mod._batched_select_indices_with_actor)
        train_mod._batched_service = _lp(train_mod._batched_service)
        # Worker loop
        if hasattr(train_mod, '_env_worker'):
            train_mod._env_worker = _lp(train_mod._env_worker)
        # Trainer parallel/batching internals
        if hasattr(train_mod, 'collect_trajectory_parallel'):
            train_mod.collect_trajectory_parallel = _lp(train_mod.collect_trajectory_parallel)
        for _name in ['_batched_select_indices', '_batched_select_indices_with_actor', '_batched_service']:
            if hasattr(train_mod, _name):
                setattr(train_mod, _name, _lp(getattr(train_mod, _name)))

        # Always profile eval, environment and MCTS stack by default under --line
        import evaluation.eval as eval_mod
        eval_mod.evaluate_pair_actors = _lp(eval_mod.evaluate_pair_actors)
        eval_mod.play_match = _lp(eval_mod.play_match)
        # Environment hotspots
        import environment as env_mod
        if hasattr(env_mod, 'ScoponeEnvMA'):
            _Env = env_mod.ScoponeEnvMA
            if hasattr(_Env, 'step'):
                setattr(cast(Any, _Env), 'step', _lp(_Env.step))
            if hasattr(_Env, 'get_valid_actions'):
                setattr(cast(Any, _Env), 'get_valid_actions', _lp(_Env.get_valid_actions))
            if hasattr(_Env, '_get_observation'):
                setattr(cast(Any, _Env), '_get_observation', _lp(_Env._get_observation))
            if hasattr(_Env, 'reset'):
                setattr(cast(Any, _Env), 'reset', _lp(_Env.reset))
        import algorithms.is_mcts as mcts_mod
        if hasattr(mcts_mod, 'run_is_mcts'):
            mcts_mod.run_is_mcts = _lp(mcts_mod.run_is_mcts)
        # If IS-MCTS exposes subroutines, profile them too (best-effort)
        for sub_name in ['tree_policy', 'expand', 'simulate', 'backpropagate', 'select_child']:
            if hasattr(mcts_mod, sub_name):
                setattr(mcts_mod, sub_name, _lp(getattr(mcts_mod, sub_name)))
        # Model forward paths (actor/belief)
        import models.action_conditioned as ac_mod
        if hasattr(ac_mod, 'ActionConditionedActor'):
            _Act = ac_mod.ActionConditionedActor
            if hasattr(_Act, 'forward'):
                setattr(cast(Any, _Act), 'forward', _lp(_Act.forward))
        if hasattr(ac_mod, 'CentralValueNet'):
            _Crit = ac_mod.CentralValueNet
            if hasattr(_Crit, 'forward'):
                setattr(cast(Any, _Crit), 'forward', _lp(_Crit.forward))
        # PPO core
        import algorithms.ppo_ac as ppo_mod
        if hasattr(ppo_mod, 'ActionConditionedPPO'):
            _PPO = ppo_mod.ActionConditionedPPO
            if hasattr(_PPO, 'update'):
                setattr(cast(Any, _PPO), 'update', _lp(_PPO.update))
            if hasattr(_PPO, 'compute_loss'):
                setattr(cast(Any, _PPO), 'compute_loss', _lp(_PPO.compute_loss))
            if hasattr(_PPO, 'select_action'):
                setattr(cast(Any, _PPO), 'select_action', _lp(_PPO.select_action))
            if hasattr(_PPO, '_select_action_core'):
                setattr(cast(Any, _PPO), '_select_action_core', _lp(_PPO._select_action_core))
            # Try common names for submodules
            for attr in ['state_enc', 'belief_net']:
                if hasattr(_Act, attr):
                    sub = getattr(_Act, attr)
                    # If it's a Module class attribute, decorate .forward
                    if hasattr(sub, 'forward'):
                        setattr(cast(Any, sub), 'forward', _lp(sub.forward))
                if hasattr(_Act, attr):
                    sub = getattr(_Act, attr)
                    # If it's a Module class attribute, decorate .forward
                    if hasattr(sub, 'forward'):
                        sub.forward = line_profile(sub.forward)

        # Allow nested local functions in trainer to be decorated if they opt-in
        setattr(train_mod, 'LINE_PROFILE_DECORATOR', line_profile)

        # Observation encoding hot path
        import observation as obs_mod
        # The environment binds maybe_compile_function(_encode_state_compact_for_player_fast)
        # We decorate both the symbol and the compiled wrapper if present
        if hasattr(obs_mod, 'encode_state_compact_for_player_fast'):
            obs_mod.encode_state_compact_for_player_fast = line_profile(obs_mod.encode_state_compact_for_player_fast)
        for _obs_fn in ['bitset_rank_counts', 'bitset_table_sum']:
            if hasattr(obs_mod, _obs_fn):
                setattr(obs_mod, _obs_fn, line_profile(getattr(obs_mod, _obs_fn)))

        # Algorithms (PPO) heavy paths
        import algorithms.ppo_ac as ppo_ac_mod
        if hasattr(ppo_ac_mod, 'ActionConditionedPPO'):
            _PPO = ppo_ac_mod.ActionConditionedPPO
            if hasattr(_PPO, 'update'):
                _PPO.update = line_profile(_PPO.update)
            if hasattr(_PPO, 'compute_loss'):
                _PPO.compute_loss = line_profile(_PPO.compute_loss)

        if args.wrap_update:
            import algorithms.ppo_ac as ppo_mod
            setattr(cast(Any, ppo_mod.ActionConditionedPPO), 'update', _lp(ppo_mod.ActionConditionedPPO.update))
            actions_mod.decode_action_ids = line_profile(actions_mod.decode_action_ids)

        if args.wrap_update:
            import algorithms.ppo_ac as ppo_mod
            ppo_mod.ActionConditionedPPO.update = line_profile(ppo_mod.ActionConditionedPPO.update)

        # Shorter run for line profiler to keep overhead manageable
        # If profiler supports global tracing, register key functions
        if hasattr(global_profiler, 'allowed_codes'):
            global_profiler.allowed_codes.add(train_mod.collect_trajectory.__code__)
            global_profiler.allowed_codes.add(train_mod.train_ppo.__code__)
        # Use the same default as other modes for apples-to-apples comparisons
        num_envs_eff = _resolve_num_envs(args)
        _selfplay_env = str(os.environ.get('SCOPONE_SELFPLAY', '1')).strip().lower()
        _selfplay = (_selfplay_env in ['1', 'true', 'yes', 'on'])
        # Eval parity with main: read eval flags from env
        _eval_games = int(os.environ.get('SCOPONE_EVAL_GAMES','1000'))
        _eval_use_mcts = os.environ.get('SCOPONE_EVAL_USE_MCTS','0').lower() in ['1','true','yes','on']
        _eval_mcts_sims = int(os.environ.get('SCOPONE_EVAL_MCTS_SIMS','128'))
        _eval_mcts_dets = int(os.environ.get('SCOPONE_EVAL_MCTS_DETS','1'))
        _eval_c_puct = float(os.environ.get('SCOPONE_EVAL_MCTS_C_PUCT','1.0'))
        _eval_root_temp = float(os.environ.get('SCOPONE_EVAL_MCTS_ROOT_TEMP','0.0'))
        _eval_prior_eps = float(os.environ.get('SCOPONE_EVAL_MCTS_PRIOR_SMOOTH_EPS','0.0'))
        _eval_dir_alpha = float(os.environ.get('SCOPONE_EVAL_MCTS_DIRICHLET_ALPHA','0.25'))
        _eval_dir_eps = float(os.environ.get('SCOPONE_EVAL_MCTS_DIRICHLET_EPS','0.25'))
        train_fn(num_iterations=max(0, args.iters), horizon=max(40, args.horizon), k_history=39, num_envs=num_envs_eff,
                 mcts_sims=0, mcts_sims_eval=_eval_mcts_sims, eval_every=0, eval_games=_eval_games,
                 mcts_in_eval=_eval_use_mcts, mcts_dets=_eval_mcts_dets, mcts_c_puct=_eval_c_puct,
                 mcts_root_temp=_eval_root_temp, mcts_prior_smooth_eps=_eval_prior_eps,
                 mcts_dirichlet_alpha=_eval_dir_alpha, mcts_dirichlet_eps=_eval_dir_eps,
                 seed=seed, use_selfplay=_selfplay)

        # Print per-function and per-line stats sorted by self time (TOT time per line)
        # Options: sort_by in {'cpu','gpu','transfer','self_cpu','self_gpu'} depending on implementation.
        # Prefer self CPU time to focus on exclusive line cost.
        global_profiler.print_stats(sort_by='self_cpu')
        if args.report:
            print(global_profiler.generate_report(include_line_details=True))

        # Aggregate by file and by file:line from line-profiler results (use self time where available)
        import inspect
        from collections import defaultdict

        project_root = ROOT
        def relpath(p):
            ap = os.path.abspath(p)
            if project_root in ap:
                return os.path.relpath(ap, project_root)
            return ap

        per_file = defaultdict(lambda: {'cpu_s': 0.0, 'gpu_s': 0.0, 'transfer_s': 0.0, 'hits': 0})
        per_site = defaultdict(lambda: {'hits': 0, 'cpu_s': 0.0, 'gpu_s': 0.0, 'transfer_s': 0.0})

        # Exclude profiler internals and this script from aggregation
        def _is_excluded(rel_path: str) -> bool:
            rp = (rel_path or '').replace('\\', '/')
            return (
                rp.startswith('profilers/') or
                rp.endswith('/profilers/line_profiler.py') or
                rp.endswith('profilers/line_profiler.py') or
                rp.endswith('tools/profile_ppo.py')
            )

        use_self_only = bool(getattr(args, 'line_self_only', False))

        for func_name, lines in global_profiler.results.items():
            func_obj = global_profiler.functions.get(func_name)
            if func_obj is None:
                continue
            filename = inspect.getsourcefile(func_obj) or func_obj.__code__.co_filename
            rfile = relpath(filename)
            if _is_excluded(rfile):
                continue
            for line_no, payload in lines.items():
                # Support both 4-tuple and 5-tuple (with self_cpu)
                if isinstance(payload, (list, tuple)) and len(payload) >= 4:
                    hits, cpu_time, gpu_time, transfer_time = payload[:4]
                    self_cpu = payload[4] if len(payload) >= 5 else None
                else:
                    continue
                # Decide which metric to use
                metric_cpu = (self_cpu if (self_cpu is not None) else (None if use_self_only else cpu_time))
                if metric_cpu is None:
                    continue
                # Some implementations provide self_cpu in rest[0]; prefer it when present
                pf = per_file[rfile]
                pf['cpu_s'] += metric_cpu
                pf['gpu_s'] += gpu_time
                pf['transfer_s'] += transfer_time
                pf['hits'] += hits
                key = (rfile, int(line_no))
                site = per_site[key]
                site['hits'] += hits
                site['cpu_s'] += metric_cpu
                site['gpu_s'] += gpu_time
                site['transfer_s'] += transfer_time

        # Print file summary
        print("\n===== Line-profiler — Time by file (self CPU) =====" if use_self_only else "\n===== Line-profiler — Time by file (preferring self CPU) =====")
        ranked_files = sorted(per_file.items(), key=lambda kv: kv[1]['cpu_s'], reverse=True)
        for i, (f, s) in enumerate(ranked_files[:20], 1):
            total = s['cpu_s']
            print(f"{i:>2}. {f}\n    CPU: {total:8.4f}s  GPU: {s['gpu_s']:8.4f}s  Transfer: {s['transfer_s']:8.4f}s  Hits: {s['hits']}")

        # Print top lines across files (optionally full)
        print("\n===== Line-profiler — Top lines (by Self CPU time) =====" if use_self_only else "\n===== Line-profiler — Top lines (preferring Self CPU) =====")
        ranked_sites = sorted(per_site.items(), key=lambda kv: kv[1]['cpu_s'], reverse=True)
        limit = None
        limit = None if getattr(args, 'line_full', False) else 30
        for i, ((f, ln), agg) in enumerate(ranked_sites[: (None if limit is None else limit) ], 1):
            print(f"{i:>2}. {f}:{ln}  Self CPU: {agg['cpu_s']:.6f}s  GPU: {agg['gpu_s']:.6f}s  Transfer: {agg['transfer_s']:.6f}s  Hits: {agg['hits']}")

        # Optional CSV dump for full per-line times across all files
        line_csv = getattr(args, 'line_csv', None)
        if line_csv:
            import csv as _csv
            abs_csv = os.path.abspath(line_csv)
            out_dir = os.path.dirname(abs_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(abs_csv, 'w', newline='', encoding='utf-8') as f:
                w = _csv.writer(f)
                w.writerow(['file', 'line', 'hits', 'cpu_s', 'gpu_s', 'transfer_s'])
                for (frel, lno), agg in ranked_sites:
                    w.writerow([frel, int(lno), int(agg['hits']), f"{agg['cpu_s']:.9f}", f"{agg['gpu_s']:.9f}", f"{agg['transfer_s']:.9f}"])
            print("Saved per-line CSV to:", abs_csv)
        return

    #trace_path = os.path.abspath('profile_trace.json')
    #print(f"Profiling short PPO run... trace -> {trace_path}")

    # Require explicit selection of torch profiler; otherwise error out
    if not getattr(args, 'torch_profiler', False):
        print("Error: no profiler selected. Use one of: --torch-profiler, --line, --cprofile, --scalene")
        return

    # Keep run short to avoid OOM without scheduler
    # Wrap selected hotspots with record_function via monkeypatch, without editing sources
    from torch.profiler import record_function as _record_function
    # observation
    import observation as _obs_mod
    if hasattr(_obs_mod, 'encode_state_compact_for_player_fast'):
        _orig_encode = _obs_mod.encode_state_compact_for_player_fast
        def _wrap_encode(*a, **kw):
            with _record_function('obs.encode_state_compact_for_player_fast'):
                return _orig_encode(*a, **kw)
        _obs_mod.encode_state_compact_for_player_fast = _wrap_encode  # type: ignore
    # environment
    import environment as _env_mod
    if hasattr(_env_mod.ScoponeEnvMA, 'get_valid_actions'):
        _orig_gva = _env_mod.ScoponeEnvMA.get_valid_actions
        def _wrap_gva(self, *a, **kw):
            with _record_function('env.get_valid_actions'):
                return _orig_gva(self, *a, **kw)
        _env_mod.ScoponeEnvMA.get_valid_actions = _wrap_gva  # type: ignore
    if hasattr(_env_mod.ScoponeEnvMA, '_get_observation'):
        _orig_go = _env_mod.ScoponeEnvMA._get_observation
        def _wrap_go(self, *a, **kw):
            with _record_function('env._get_observation'):
                return _orig_go(self, *a, **kw)
        _env_mod.ScoponeEnvMA._get_observation = _wrap_go  # type: ignore
    if hasattr(_env_mod.ScoponeEnvMA, 'step'):
        _orig_step = _env_mod.ScoponeEnvMA.step
        def _wrap_step(self, *a, **kw):
            with _record_function('env.step'):
                return _orig_step(self, *a, **kw)
        _env_mod.ScoponeEnvMA.step = _wrap_step  # type: ignore
    # algorithms / actor
    import algorithms.ppo_ac as _ppo_mod
    if hasattr(_ppo_mod.ActionConditionedPPO, '_select_action_core'):
        _orig_core = _ppo_mod.ActionConditionedPPO._select_action_core
        def _wrap_core(self, *a, **kw):
            with _record_function('algo._select_action_core'):
                return _orig_core(self, *a, **kw)
        _ppo_mod.ActionConditionedPPO._select_action_core = _wrap_core  # type: ignore
    if hasattr(_ppo_mod.ActionConditionedPPO, 'select_action'):
        _orig_sel = _ppo_mod.ActionConditionedPPO.select_action
        def _wrap_sel(self, *a, **kw):
            with _record_function('algo.select_action'):
                return _orig_sel(self, *a, **kw)
        _ppo_mod.ActionConditionedPPO.select_action = _wrap_sel  # type: ignore
    # model internals
    import models.action_conditioned as _ac_mod
    if hasattr(_ac_mod.ActionConditionedActor, 'compute_state_proj'):
        _orig_csp = _ac_mod.ActionConditionedActor.compute_state_proj
        def _wrap_csp(self, *a, **kw):
            with _record_function('model.compute_state_proj'):
                return _orig_csp(self, *a, **kw)
        _ac_mod.ActionConditionedActor.compute_state_proj = _wrap_csp  # type: ignore
    if hasattr(_ac_mod.ActionConditionedActor, '_mha_masked_mean'):
        _orig_mmm = _ac_mod.ActionConditionedActor._mha_masked_mean
        def _wrap_mmm(self, *a, **kw):
            with _record_function('model._mha_masked_mean'):
                return _orig_mmm(self, *a, **kw)
        _ac_mod.ActionConditionedActor._mha_masked_mean = _wrap_mmm  # type: ignore

    # Configure child workers to run a torch profiler by default when torch-profiler is selected
    _profiles_dir = os.path.abspath(os.path.join(ROOT, 'profiles'))
    os.makedirs(_profiles_dir, exist_ok=True)
    os.environ['SCOPONE_TORCH_PROF'] = '1'
    os.environ['SCOPONE_TORCH_PROF_DIR'] = _profiles_dir
    from datetime import datetime as _dt
    _run_tag = _dt.now().strftime('%Y%m%d_%H%M%S')
    os.environ['SCOPONE_TORCH_PROF_RUN'] = _run_tag
    # TensorBoard combined timeline directory (shared across main + workers)
    _tb_dir = os.path.abspath(os.path.join(ROOT, 'runs', 'profiler', _run_tag))
    os.makedirs(_tb_dir, exist_ok=True)
    os.environ['SCOPONE_TORCH_TB_DIR'] = _tb_dir

    from torch.profiler import schedule as _tp_schedule, tensorboard_trace_handler as _tb_handler
    _tb_handler_main = _tb_handler(_tb_dir, worker_name=f"main-pid{os.getpid()}")

    # Relax timeouts and workload for profiling stability
    os.environ['SCOPONE_RPC_TIMEOUT_S'] = '300'
    os.environ['SCOPONE_COLLECTOR_STALL_S'] = '300'
    os.environ['SCOPONE_EP_PUT_TIMEOUT_S'] = '60'

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_modules=True,
        schedule=_tp_schedule(wait=0, warmup=0, active=1, repeat=1000000),
        on_trace_ready=_tb_handler_main,
    ) as prof:
        # Constrain workload under profiler
        _prof_max_envs = int(os.environ.get('SCOPONE_PROF_NUM_ENVS', '8'))
        _prof_max_horizon = int(os.environ.get('SCOPONE_PROF_HORIZON', '2048'))
        num_envs_eff = _resolve_num_envs(args, clamp=_prof_max_envs)
        _selfplay_env = str(os.environ.get('SCOPONE_SELFPLAY', '1')).strip().lower()
        _selfplay = (_selfplay_env in ['1', 'true', 'yes', 'on'])
        def _on_iter_end_cb(it_idx: int):
            prof.step()
        _h_eff = int(min(max(40, args.horizon), _prof_max_horizon))
        print(f"[torch-profiler] Effective profiling config: num_envs={num_envs_eff}, horizon={_h_eff}, RPC_TIMEOUT_S={os.environ.get('SCOPONE_RPC_TIMEOUT_S')}")
        _eval_games = int(os.environ.get('SCOPONE_EVAL_GAMES','1000'))
        train_ppo(num_iterations=max(0, args.iters), horizon=_h_eff, k_history=39, num_envs=num_envs_eff,
                  mcts_sims=0, mcts_sims_eval=0, eval_every=0, eval_games=_eval_games, mcts_in_eval=False, seed=seed, use_selfplay=_selfplay,
                  on_iter_end=_on_iter_end_cb,
                  mcts_warmup_iters=_mcts_warmup_iters)

    # Export chrome trace for main process
    out_main = os.path.abspath(os.path.join(_profiles_dir, f"tp_main_{os.getpid()}_{os.environ.get('SCOPONE_TORCH_PROF_RUN','run')}.json"))
    prof.export_chrome_trace(out_main)
    print(f"\nSaved main torch profiler trace: {out_main}")
    print(f"TensorBoard combined timeline saved under: {_tb_dir}")
    print("Per-worker traces are saved in profiles/ as tp_worker_<wid>_<pid>_<run>.json")

    # Print top operators by CUDA time and CPU time
    print("\nTop ops by CUDA time:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=25))
    print("\nTop ops by CPU time:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=25))

    # Stack-grouped tables (gives file:line attribution)
    print("\nTop by CUDA time (grouped by stack):")
    print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cuda_time_total", row_limit=30))
    print("\nTop by CPU time (grouped by stack):")
    print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=30))

    # Summarize record_function tags to make hotspots immediately visible
    from collections import defaultdict as _dd
    tag_totals = _dd(lambda: { 'cpu_us': 0.0, 'cuda_us': 0.0, 'count': 0 })
    for avg in prof.key_averages():
        name = getattr(avg, 'key', '') or getattr(avg, 'name', '') or ''
        if not isinstance(name, str):
            continue
        if not (name.startswith('env.') or name.startswith('algo.') or name.startswith('model.') or name.startswith('obs.')):
            continue
        cpu_us = getattr(avg, 'self_cpu_time_total', 0.0) or 0.0
        cuda_us = getattr(avg, 'self_device_time_total', None)
        if cuda_us is None:
            cuda_us = getattr(avg, 'self_cuda_time_total', 0.0) or 0.0
        tag_totals[name]['cpu_us'] += float(cpu_us)
        tag_totals[name]['cuda_us'] += float(cuda_us)
        tag_totals[name]['count'] += getattr(avg, 'count', 1) or 1

    if tag_totals:
        def _to_ms(us: float) -> float:
            return float(us) / 1000.0
        print("\nTop by tag (record_function):")
        ranked_tags = sorted(tag_totals.items(), key=lambda kv: (kv[1]['cuda_us'] + kv[1]['cpu_us']), reverse=True)
        for i, (tag, s) in enumerate(ranked_tags[:20], 1):
            total_ms = _to_ms(s['cpu_us'] + s['cuda_us'])
            print(f"{i:>2}. {tag}\n    Total: {total_ms:8.2f} ms | CPU: {_to_ms(s['cpu_us']):8.2f} ms | CUDA: {_to_ms(s['cuda_us']):8.2f} ms | count: {int(s['count'])}")

    # Aggregate by user source file (from Python stacks) and highlight H2D/D2H memcpys
    try:
        from collections import defaultdict
        import re

        project_root = ROOT
        events = prof.events()

        def to_ms(us):
            return float(us) / 1000.0

        def frame_filename_and_line(frame_like):
            """Return (filename, line) from a frame-like object or string."""
            fn = getattr(frame_like, 'filename', None)
            ln = getattr(frame_like, 'line', None)
            if fn:
                return fn, ln if isinstance(ln, int) else None
            s = str(frame_like)
            m = re.search(r"(.*?\.py):(\d+)", s) or re.search(r"(.*?\.py)\((\d+)\)", s)
            if m:
                fn = m.group(1)
                ln = int(m.group(2))
                return fn, ln
            if '.py' in s:
                pre = s.split('.py', 1)[0] + '.py'
                ln_m = re.search(r":(\d+)", s)
                ln = int(ln_m.group(1)) if ln_m else None
                return pre, ln
            return None, None

        def iter_stack_frames(stack_obj):
            if not stack_obj:
                return
            frames_attr = getattr(stack_obj, 'frames', None)
            if isinstance(frames_attr, (list, tuple)):
                for fr in frames_attr:
                    yield fr
                return
            if isinstance(stack_obj, (list, tuple)):
                for fr in stack_obj:
                    yield fr
                return
            if isinstance(stack_obj, str):
                for line in stack_obj.splitlines():
                    yield line
                return
            yield stack_obj

        def is_in_project(abs_path):
            if project_root in abs_path:
                return True
            return os.sep + os.path.basename(project_root) + os.sep in abs_path

        def find_user_file(stack_obj):
            for fr in iter_stack_frames(stack_obj):
                fn, ln = frame_filename_and_line(fr)
                if not fn:
                    continue
                abs_fn = os.path.abspath(fn)
                if is_in_project(abs_fn):
                    rel = os.path.relpath(abs_fn, project_root)
                    return rel, (ln if isinstance(ln, int) else None)
            return None, None

        per_file = defaultdict(lambda: {
            'cpu_ms': 0.0,
            'cuda_ms': 0.0,
            'count': 0,
            'memcpy_h2d_count': 0,
            'memcpy_d2h_count': 0,
            'memcpy_ms': 0.0,
        })
        memcpy_sites = defaultdict(lambda: {'count': 0, 'ms': 0.0})

        for evt in events:
            name = str(getattr(evt, 'name', ''))
            cpu_us = getattr(evt, 'self_cpu_time_total', 0.0) or 0.0
            cuda_us = getattr(evt, 'self_device_time_total', None)
            if cuda_us is None:
                cuda_us = getattr(evt, 'self_cuda_time_total', 0.0)
            cuda_us = cuda_us or 0.0
            stack = getattr(evt, 'stack', None)
            file_rel, line_no = find_user_file(stack)
            if not file_rel:
                file_rel = '<external/CUDA or Library>'
                line_no = -1
            stats = per_file[file_rel]
            stats['cpu_ms'] += to_ms(cpu_us)
            stats['cuda_ms'] += to_ms(cuda_us)
            stats['count'] += 1

            lname = name.lower().replace(' ', '')
            is_memcpy = ('memcpy' in lname) or ('memcpyasync' in lname) or ('dtoh' in lname) or ('htod' in lname)
            if is_memcpy:
                memcpy_ms = to_ms(cuda_us if cuda_us else cpu_us)
                stats['memcpy_ms'] += memcpy_ms
                kind = 'H2D' if ('h2d' in lname or 'htod' in lname) else ('D2H' if ('d2h' in lname or 'dtoh' in lname) else 'UNK')
                if kind == 'H2D':
                    stats['memcpy_h2d_count'] += 1
                elif kind == 'D2H':
                    stats['memcpy_d2h_count'] += 1
                site_key = (file_rel, int(line_no) if isinstance(line_no, int) else -1, kind)
                memcpy_sites[site_key]['count'] += 1
                memcpy_sites[site_key]['ms'] += memcpy_ms

        def fmt_row(idx, file_rel, s):
            total_ms = s['cpu_ms'] + s['cuda_ms']
            memcpy_pct = (100.0 * s['memcpy_ms'] / total_ms) if total_ms > 0 else 0.0
            return (f"{idx:>2}. {file_rel}\n"
                    f"    CUDA: {s['cuda_ms']:8.2f} ms | CPU: {s['cpu_ms']:8.2f} ms | Total: {total_ms:8.2f} ms\n"
                    f"    memcpy H2D: {s['memcpy_h2d_count']:4d}  D2H: {s['memcpy_d2h_count']:4d} | memcpy time: {s['memcpy_ms']:7.2f} ms ({memcpy_pct:4.1f}%)")

        print("\n===== Time by source file (self times) =====")
        ranked = sorted(per_file.items(), key=lambda kv: (kv[1]['cuda_ms'] + kv[1]['cpu_ms']), reverse=True)
        for i, (file_rel, s) in enumerate(ranked[:30], 1):
            print(fmt_row(i, file_rel, s))

        only_external = all(k == '<external/CUDA or Library>' for k, _ in per_file.items()) or len(per_file) == 0
        if only_external:
            per_file_agg = defaultdict(lambda: {
                'cpu_ms': 0.0,
                'cuda_ms': 0.0,
                'count': 0,
                'memcpy_h2d_count': 0,
                'memcpy_d2h_count': 0,
                'memcpy_ms': 0.0,
            })
            for avg in prof.key_averages(group_by_stack_n=25):
                stack_obj = getattr(avg, 'stack', None)
                file_rel, _ = find_user_file(stack_obj)
                if not file_rel:
                    file_rel = '<external/CUDA or Library>'
                cuda_us = getattr(avg, 'self_device_time_total', None)
                if cuda_us is None:
                    cuda_us = getattr(avg, 'self_cuda_time_total', 0.0)
                cpu_us = getattr(avg, 'self_cpu_time_total', 0.0) or 0.0
                s = per_file_agg[file_rel]
                s['cpu_ms'] += to_ms(cpu_us)
                s['cuda_ms'] += to_ms(cuda_us or 0.0)
                s['count'] += getattr(avg, 'count', 1) or 1
            print("\n===== Fallback (aggregated stacks) — Time by source file =====")
            ranked2 = sorted(per_file_agg.items(), key=lambda kv: (kv[1]['cuda_ms'] + kv[1]['cpu_ms']), reverse=True)
            for i, (file_rel, s) in enumerate(ranked2[:30], 1):
                print(fmt_row(i, file_rel, s))

        if memcpy_sites:
            print("\n===== Top memcpy sites (by time) =====")
            ranked_sites = sorted(memcpy_sites.items(), key=lambda kv: kv[1]['ms'], reverse=True)
            for i, ((file_rel, ln, kind), agg) in enumerate(ranked_sites[:30], 1):
                loc = f"{file_rel}:{ln if ln and ln>0 else '?'}"
                print(f"{i:>2}. {kind:>3}  {agg['ms']:8.2f} ms  | count: {agg['count']:4d}  | {loc}")
        else:
            print("\n(no memcpy events detected; if you expect transfers, ensure CUDA profiling is enabled)")
    except Exception as e:
        print(f"Per-file aggregation failed: {e}")


if __name__ == "__main__":
    main()
