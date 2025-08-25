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

import torch

# Prefer GPU for models if available; never force CPU unless explicitly requested
if os.environ.get('TESTS_FORCE_CPU') == '1':
    try:
        del os.environ['TESTS_FORCE_CPU']
    except Exception:
        pass

# Ensure device selection consistent with main
try:
    from utils.device import get_compute_device
    _dev = get_compute_device()
    # If CUDA is available but not selected, allow override
    if torch.cuda.is_available() and str(_dev) != 'cuda':
        os.environ.setdefault('SCOPONE_DEVICE', 'cuda')
except Exception:
    pass

from trainers.train_ppo import train_ppo


def main():
    parser = argparse.ArgumentParser(description='Profile short PPO run (torch or line-level).')
    parser.add_argument('--iters', type=int, default=30, help='Iterations to run')
    parser.add_argument('--horizon', type=int, default=2048, help='Rollout horizon per iteration')
    parser.add_argument('--line', dest='line', action='store_true', default=True, help='Enable line-by-line profiler with per-line timings (default: on)')
    parser.add_argument('--no-line', dest='line', action='store_false', help='Disable line-by-line profiler')
    parser.add_argument('--wrap-update', dest='wrap_update', action='store_true', default=True, help='Also profile ActionConditionedPPO.update (default: on; slower)')
    parser.add_argument('--no-wrap-update', dest='wrap_update', action='store_false', help='Disable profiling of ActionConditionedPPO.update')
    parser.add_argument('--report', action='store_true', help='Print extended line-profiler report')
    args = parser.parse_args()

    # Perf flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    if args.line:
        # Lightweight line-by-line profiling for Python code with file:line output
        try:
            from profilers.line_profiler import profile as line_profile, global_profiler
        except Exception as e:
            print(f"Line profiler unavailable: {e}")
            return

        # Wrap hotspots and also enable global tracing fallback
        import trainers.train_ppo as train_mod
        train_mod.collect_trajectory = line_profile(train_mod.collect_trajectory)
        train_fn = line_profile(train_mod.train_ppo)

        if args.wrap_update:
            try:
                import algorithms.ppo_ac as ppo_mod
                ppo_mod.ActionConditionedPPO.update = line_profile(ppo_mod.ActionConditionedPPO.update)
            except Exception:
                pass

        # Shorter run for line profiler to keep overhead manageable
        try:
            # If profiler supports global tracing, register key functions
            if hasattr(global_profiler, 'allowed_codes'):
                # Register also methods we can't easily wrap (e.g., bound methods created at runtime)
                global_profiler.allowed_codes.add(train_mod.collect_trajectory.__code__)
                global_profiler.allowed_codes.add(train_mod.train_ppo.__code__)
        except Exception:
            pass
        train_fn(num_iterations=max(1, args.iters), horizon=max(40, args.horizon), use_compact_obs=True, k_history=39, num_envs=1, mcts_sims=0, mcts_sims_eval=0, eval_every=0, mcts_in_eval=False)

        # Print per-function and per-line stats (includes line numbers and source)
        try:
            global_profiler.print_stats(sort_by='cpu')
            if args.report:
                print(global_profiler.generate_report(include_line_details=True))
        except Exception:
            pass

        # Aggregate by file and by file:line from line-profiler results
        try:
            import inspect
            from collections import defaultdict

            project_root = ROOT
            def relpath(p):
                try:
                    ap = os.path.abspath(p)
                    if project_root in ap:
                        return os.path.relpath(ap, project_root)
                    return ap
                except Exception:
                    return str(p)

            per_file = defaultdict(lambda: {'cpu_s': 0.0, 'gpu_s': 0.0, 'transfer_s': 0.0, 'hits': 0})
            per_site = defaultdict(lambda: {'hits': 0, 'cpu_s': 0.0, 'gpu_s': 0.0, 'transfer_s': 0.0})

            for func_name, lines in global_profiler.results.items():
                func_obj = global_profiler.functions.get(func_name)
                if func_obj is None:
                    continue
                try:
                    filename = inspect.getsourcefile(func_obj) or func_obj.__code__.co_filename
                except Exception:
                    filename = getattr(func_obj.__code__, 'co_filename', '<unknown>')
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

    # Keep run short to avoid OOM without scheduler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        #experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        # Short run for signal; adjust if needed
        train_ppo(num_iterations=max(1, args.iters), horizon=max(40, args.horizon), use_compact_obs=True, k_history=39, num_envs=1, mcts_sims=0)

    # Export chrome trace
    #prof.export_chrome_trace(trace_path)

    # Print top operators by CUDA time and CPU time
    print("\nTop ops by CUDA time:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=25))
    print("\nTop ops by CPU time:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=25))

    # Stack-grouped tables (gives file:line attribution)
    try:
        print("\nTop by CUDA time (grouped by stack):")
        print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cuda_time_total", row_limit=30))
        print("\nTop by CPU time (grouped by stack):")
        print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=30))
    except Exception:
        pass

    # Aggregate by user source file (from Python stacks) and highlight H2D/D2H memcpys
    try:
        from collections import defaultdict
        import re

        project_root = ROOT
        events = prof.events()

        def to_ms(us):
            try:
                return float(us) / 1000.0
            except Exception:
                return 0.0

        def frame_filename_and_line(frame_like):
            """Return (filename, line) from a frame-like object or string."""
            try:
                fn = getattr(frame_like, 'filename', None)
                ln = getattr(frame_like, 'line', None)
                if fn:
                    return fn, ln if isinstance(ln, int) else None
                s = str(frame_like)
                m = re.search(r"(.*?\.py):(\d+)", s)
                if not m:
                    m = re.search(r"(.*?\.py)\((\d+)\)", s)
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
            except Exception:
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
            try:
                if project_root in abs_path:
                    return True
                # Fallback: contains repo folder name
                return os.sep + os.path.basename(project_root) + os.sep in abs_path
            except Exception:
                return False

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


