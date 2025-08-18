import os
import sys
import torch

# Ensure project root on sys.path for module imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainers.train_ppo import train_ppo


def main():
    # Perf flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

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
        #experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        # Short run for signal; adjust if needed
        train_ppo(num_iterations=20, horizon=256, use_compact_obs=True, k_history=12)

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
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=30))
        print("\nTop by CPU time (grouped by stack):")
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=30))
    except Exception:
        pass

    # Aggregate by user source file (from Python stacks) and highlight H2D/D2H memcpys
    try:
        from collections import defaultdict

        project_root = ROOT
        events = prof.events()

        def to_ms(us):
            try:
                return float(us) / 1000.0
            except Exception:
                return 0.0

        def frame_filename(frame):
            try:
                fn = getattr(frame, 'filename', None)
                if fn is None:
                    s = str(frame)
                    # Heuristic: extract path before ':' if present
                    return s.split(':', 1)[0]
                return fn
            except Exception:
                return None

        def find_user_file(stack_frames):
            if not stack_frames:
                return None, None
            for fr in stack_frames:
                fn = frame_filename(fr)
                if not fn:
                    continue
                # Normalize and keep only frames from project
                abs_fn = os.path.abspath(fn)
                if abs_fn.startswith(project_root):
                    rel = os.path.relpath(abs_fn, project_root)
                    # Line number (best-effort)
                    ln = getattr(fr, 'line', None)
                    if ln is None:
                        try:
                            s = str(fr)
                            parts = s.split(':')
                            if len(parts) >= 3 and parts[1].isdigit():
                                ln = int(parts[1])
                        except Exception:
                            ln = None
                    return rel, ln
            return None, None

        per_file = defaultdict(lambda: {
            'cpu_ms': 0.0,
            'cuda_ms': 0.0,
            'count': 0,
            'memcpy_h2d_count': 0,
            'memcpy_d2h_count': 0,
            'memcpy_ms': 0.0,
        })
        memcpy_sites = defaultdict(lambda: {'count': 0, 'ms': 0.0})  # keyed by (file, line, kind)

        for evt in events:
            name = str(getattr(evt, 'name', ''))
            cpu_us = getattr(evt, 'self_cpu_time_total', 0.0) or 0.0
            # Prefer device time if present, fallback to legacy cuda time
            cuda_us = getattr(evt, 'self_device_time_total', None)
            if cuda_us is None:
                cuda_us = getattr(evt, 'self_cuda_time_total', 0.0)
            cuda_us = cuda_us or 0.0
            stack = getattr(evt, 'stack', None)
            file_rel, line_no = find_user_file(stack)
            # Attribute external ops (no project stack) to a synthetic bucket
            if not file_rel:
                file_rel = '<external/CUDA or Library>'
                line_no = -1
            stats = per_file[file_rel]
            stats['cpu_ms'] += to_ms(cpu_us)
            stats['cuda_ms'] += to_ms(cuda_us)
            stats['count'] += 1

            lname = name.lower()
            lname = lname.replace(' ', '')
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


