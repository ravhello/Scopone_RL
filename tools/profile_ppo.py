import os
import sys
import argparse

# Ensure project root on sys.path for module imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Align environment/device behavior with main.py BEFORE importing training code
# Silence TF/absl noise and enable TB by default
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('ABSL_LOGGING_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('SCOPONE_DISABLE_TB', '0')
## Abilita torch.compile di default anche nel profiler (override via env)
os.environ.setdefault('SCOPONE_TORCH_COMPILE', '0')
os.environ.setdefault('SCOPONE_TORCH_COMPILE_MODE', 'max-autotune')
os.environ.setdefault('SCOPONE_COMPILE_VERBOSE', '1')
## Disabilita max_autotune_gemm di Inductor per evitare warning su GPU con poche SM
os.environ.setdefault('TORCHINDUCTOR_MAX_AUTOTUNE_GEMM', '0')
## Evita graph break su .item() catturando scalari nei grafi
os.environ.setdefault('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', '1')
os.environ.setdefault('TORCHDYNAMO_CACHE_SIZE_LIMIT', '32')
## Non forzare TORCH_LOGS ad un valore non valido; lascia default
## Abilita dynamic shapes per ridurre errori di symbolic shapes FX
os.environ.setdefault('TORCHDYNAMO_DYNAMIC_SHAPES', '1')
## Abilita di default feature dell'osservazione (dealer one-hot)
os.environ.setdefault('OBS_INCLUDE_DEALER', '1')
## Default to CPU unless overridden by user env
os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
os.environ.setdefault('ENV_DEVICE', 'cpu')
## Mantieni ENV_DEVICE allineato a main.py per profili consistenti
try:
    import torch as _t
    _env_def = 'cuda' if _t.cuda.is_available() else 'cpu'
except Exception:
    _env_def = 'cpu'
os.environ.setdefault('ENV_DEVICE', os.environ.get('SCOPONE_DEVICE', _env_def))
## Forza metodo di start del multiprocessing a 'fork' di default su Linux/WSL per evitare hang
os.environ.setdefault('SCOPONE_MP_START', 'fork')

_SILENCE_ABSL = (os.environ.get('SCOPONE_SILENCE_ABSL', '1') == '1') and (os.environ.get('SCALENE_RUNNING', '0') != '1')
if _SILENCE_ABSL:
    _SUPPRESS_SUBSTRINGS = (
        "All log messages before absl::InitializeLog() is called are written to STDERR",
        "Unable to register cuDNN factory",
        "Unable to register cuBLAS factory",
        "cuda_dnn.cc",
        "cuda_blas.cc",
    )
    # Install OS-level fd2 filter like main.py so native C++ logs are filtered
    import threading  # local import to avoid changing global import order
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

import threading
import io
import torch
import subprocess
import webbrowser
from datetime import datetime
import platform

# Default CPU for profiling unless user sets GPU via env

from trainers.train_ppo import train_ppo
from utils.seed import resolve_seed


def main():
    parser = argparse.ArgumentParser(description='Profile short PPO run (torch or line-level).')
    parser.add_argument('--iters', type=int, default=5, help='Iterations to run')
    parser.add_argument('--horizon', type=int, default=2048, help='Rollout horizon per iteration')
    parser.add_argument('--line', dest='line', action='store_true', default=False, help='Enable line-by-line profiler with per-line timings (default: on)')
    parser.add_argument('--no-line', dest='line', action='store_false', help='Disable line-by-line profiler')
    parser.add_argument('--wrap-update', dest='wrap_update', action='store_true', default=True, help='Also profile ActionConditionedPPO.update (default: on; slower)')
    parser.add_argument('--no-wrap-update', dest='wrap_update', action='store_false', help='Disable profiling of ActionConditionedPPO.update')
    parser.add_argument('--report', action='store_true', help='Print extended line-profiler report')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments (default: 17 with --line, 1 without)')
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

    # Perf flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

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
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
        try:
            ne = max(1, int(args.num_envs)) if getattr(args, 'num_envs', None) is not None else 1
        except Exception:
            ne = 1
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
        try:
            scalene_cmd += ['--profile-all', '--profile-only', ROOT]
        except Exception:
            pass
        scalene_cmd += [
            script_path,
            '--scalene-run',
            '--no-line',
            '--no-wrap-update',
            '--iters', str(max(1, args.iters)),
            '--horizon', str(max(40, args.horizon)),
            '--num-envs', str(ne),
        ]
        print("Running under Scalene... this may add overhead.")
        try:
            env = os.environ.copy()
            env['SCOPONE_SEED'] = str(seed)
            # Mark inner run so we can disable stderr filtering; allow live progress bars
            env['SCALENE_RUNNING'] = '1'
            env['SCOPONE_SILENCE_ABSL'] = '0'
            env['TQDM_DISABLE'] = '0'
            # Stream output live while capturing for summary at the end
            proc = subprocess.Popen(scalene_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            captured_lines = []
            assert proc.stdout is not None
            for _ln in proc.stdout:
                try:
                    print(_ln, end='', flush=True)
                except Exception:
                    sys.stdout.write(_ln)
                    try:
                        sys.stdout.flush()
                    except Exception:
                        pass
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
                try:
                    ts_cli = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cli_out_path = os.path.abspath(os.path.join(DEFAULT_PROFILES_DIR, f'ppo_scalene_{ts_cli}_cli.txt'))
                    with open(cli_out_path, 'w', encoding='utf-8') as f:
                        f.write(full_cli_output)
                    print("\nScalene CLI saved to:", cli_out_path)
                except Exception:
                    cli_out_path = None

                # Print compact, line-profiler-like summary at the bottom
                print("\n===== Scalene — Compact summary (by file) =====")
                try:
                    import re as _re

                    def _strip_ansi(txt: str) -> str:
                        try:
                            return _re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", txt)
                        except Exception:
                            return txt

                    def _to_rel(pth: str) -> str:
                        ap = os.path.abspath(pth)
                        if ROOT in ap:
                            return os.path.relpath(ap, ROOT)
                        return ap

                    def _parse_secs(s: str) -> float:
                        try:
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
                        except Exception:
                            return 0.0

                    clean_output = _strip_ansi(full_cli_output)
                    per_file = {}
                    for line in clean_output.splitlines():
                        if '.py: % of time' in line:
                            try:
                                before, after = line.split(': % of time', 1)
                                file_abs = before.strip()
                                # seconds may be absent in some builds
                                m_pct = _re.search(r'=\s*([0-9]+(?:\.[0-9]+)?)%\s*(?:\(([^\)]*)\))?', after)
                                if not m_pct:
                                    continue
                                pct = float(m_pct.group(1))
                                secs = _parse_secs(m_pct.group(2)) if (m_pct.lastindex and m_pct.lastindex >= 2) else 0.0
                                per_file[_to_rel(file_abs)] = {'seconds': secs, 'percent': pct}
                            except Exception:
                                continue

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
                            try:
                                ln_no = int(m_row.group(1))
                            except Exception:
                                continue
                            def _pct_to_float(txt: str) -> float:
                                try:
                                    t = txt.strip()
                                    m = _re.search(r"([0-9]+(?:\.[0-9]+)?)%", t)
                                    return float(m.group(1)) if m else 0.0
                                except Exception:
                                    return 0.0
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
                                        try:
                                            ln_no = int(mfr.group(1))
                                        except Exception:
                                            continue
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
                    # On parser error, still show the tail of cleaned output
                    try:
                        tail = '\n'.join(clean_output.splitlines()[-80:])
                        print(tail)
                    except Exception:
                        pass

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
        num_envs = max(1, int(args.num_envs)) if getattr(args, 'num_envs', None) is not None else 1
        train_ppo(num_iterations=max(1, args.iters), horizon=max(40, args.horizon), use_compact_obs=True, k_history=39, num_envs=num_envs, mcts_sims=0, mcts_sims_eval=0, eval_every=0, mcts_in_eval=False, seed=seed)
        return

    # cProfile mode takes precedence over line/torch profiler
    if getattr(args, 'cprofile', False):
        try:
            import cProfile
            import pstats
        except Exception as e:
            print(f"cProfile unavailable: {e}")
            return

        prof = cProfile.Profile()
        # Keep run short to avoid OOM without scheduler
        num_envs = max(1, int(args.num_envs)) if getattr(args, 'num_envs', None) is not None else 1

        def _run():
            train_ppo(num_iterations=max(1, args.iters), horizon=max(40, args.horizon), use_compact_obs=True, k_history=39, num_envs=num_envs, mcts_sims=0, seed=seed)

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

        # Wrap hotspots and also enable global tracing fallback
        import trainers.train_ppo as train_mod
        train_mod.collect_trajectory = line_profile(train_mod.collect_trajectory)
        train_fn = line_profile(train_mod.train_ppo)

        if args.wrap_update:
            import algorithms.ppo_ac as ppo_mod
            ppo_mod.ActionConditionedPPO.update = line_profile(ppo_mod.ActionConditionedPPO.update)

        # Shorter run for line profiler to keep overhead manageable
        # If profiler supports global tracing, register key functions
        if hasattr(global_profiler, 'allowed_codes'):
            global_profiler.allowed_codes.add(train_mod.collect_trajectory.__code__)
            global_profiler.allowed_codes.add(train_mod.train_ppo.__code__)
        # Use the same default as other modes for apples-to-apples comparisons
        num_envs = max(1, int(args.num_envs)) if getattr(args, 'num_envs', None) is not None else 1
        train_fn(num_iterations=max(1, args.iters), horizon=max(40, args.horizon), use_compact_obs=True, k_history=39, num_envs=num_envs, mcts_sims=0, mcts_sims_eval=0, eval_every=0, mcts_in_eval=False, seed=seed)

        # Print per-function and per-line stats (includes line numbers and source)
        global_profiler.print_stats(sort_by='cpu')
        if args.report:
            print(global_profiler.generate_report(include_line_details=True))

        # Aggregate by file and by file:line from line-profiler results
        try:
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

            for func_name, lines in global_profiler.results.items():
                func_obj = global_profiler.functions.get(func_name)
                if func_obj is None:
                    continue
                filename = inspect.getsourcefile(func_obj) or func_obj.__code__.co_filename
                rfile = relpath(filename)
                for line_no, (hits, cpu_time, gpu_time, transfer_time) in lines.items():
                    pf = per_file[rfile]
                    pf['cpu_s'] += cpu_time
                    pf['gpu_s'] += gpu_time
                    pf['transfer_s'] += transfer_time
                    pf['hits'] += hits
                    key = (rfile, int(line_no))
                    site = per_site[key]
                    site['hits'] += hits
                    site['cpu_s'] += cpu_time
                    site['gpu_s'] += gpu_time
                    site['transfer_s'] += transfer_time

            # Print file summary
            print("\n===== Line-profiler — Time by file =====")
            ranked_files = sorted(per_file.items(), key=lambda kv: kv[1]['cpu_s'], reverse=True)
            for i, (f, s) in enumerate(ranked_files[:20], 1):
                total = s['cpu_s']
                print(f"{i:>2}. {f}\n    CPU: {total:8.4f}s  GPU: {s['gpu_s']:8.4f}s  Transfer: {s['transfer_s']:8.4f}s  Hits: {s['hits']}")

            # Print top lines across files
            print("\n===== Line-profiler — Top lines (by CPU time) =====")
            ranked_sites = sorted(per_site.items(), key=lambda kv: kv[1]['cpu_s'], reverse=True)
            for i, ((f, ln), agg) in enumerate(ranked_sites[:30], 1):
                print(f"{i:>2}. {f}:{ln}  CPU: {agg['cpu_s']:.6f}s  GPU: {agg['gpu_s']:.6f}s  Transfer: {agg['transfer_s']:.6f}s  Hits: {agg['hits']}")
        except Exception as e:
            print(f"Line-profiler per-file aggregation failed: {e}")
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
    try:
        # observation
        import observation as _obs_mod
        if hasattr(_obs_mod, 'encode_state_compact_for_player_fast'):
            _orig_encode = _obs_mod.encode_state_compact_for_player_fast
            def _wrap_encode(*a, **kw):
                with _record_function('obs.encode_state_compact_for_player_fast'):
                    return _orig_encode(*a, **kw)
            _obs_mod.encode_state_compact_for_player_fast = _wrap_encode  # type: ignore
    except Exception:
        pass
    try:
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
    except Exception:
        pass
    try:
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
    except Exception:
        pass
    try:
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
    except Exception:
        pass

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_modules=True,
        #experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        # Short run for signal; adjust if needed
        num_envs = max(1, int(args.num_envs)) if getattr(args, 'num_envs', None) is not None else 1
        train_ppo(num_iterations=max(1, args.iters), horizon=max(40, args.horizon), use_compact_obs=True, k_history=39, num_envs=num_envs, mcts_sims=0, seed=seed)

    # Export chrome trace
    #prof.export_chrome_trace(trace_path)

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
    try:
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
                try:
                    return float(us) / 1000.0
                except Exception:
                    return 0.0
            print("\nTop by tag (record_function):")
            ranked_tags = sorted(tag_totals.items(), key=lambda kv: (kv[1]['cuda_us'] + kv[1]['cpu_us']), reverse=True)
            for i, (tag, s) in enumerate(ranked_tags[:20], 1):
                total_ms = _to_ms(s['cpu_us'] + s['cuda_us'])
                print(f"{i:>2}. {tag}\n    Total: {total_ms:8.2f} ms | CPU: {_to_ms(s['cpu_us']):8.2f} ms | CUDA: {_to_ms(s['cuda_us']):8.2f} ms | count: {int(s['count'])}")
    except Exception:
        pass

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
            try:
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
            except Exception:
                # If stack grouping fails, continue without aggregated section
                pass

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


