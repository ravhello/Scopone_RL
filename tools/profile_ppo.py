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

    trace_path = os.path.abspath('profile_trace.json')
    print(f"Profiling short PPO run... trace -> {trace_path}")

    # Keep run short to profile quickly
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Short run for signal; adjust if needed
        train_ppo(num_iterations=12, horizon=256, use_compact_obs=True, k_history=12)

    # Export chrome trace
    prof.export_chrome_trace(trace_path)

    # Print top operators by CUDA time and CPU time
    print("\nTop ops by CUDA time:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=25))
    print("\nTop ops by CPU time:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=25))


if __name__ == "__main__":
    main()


