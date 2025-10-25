#!/usr/bin/env python3
"""
Automatic tuner for PPO rollout horizon and minibatch size.

The script executes several short training runs (on CPU) with different
configurations, aggregates their reward/time statistics across multiple
seeds, and reports the configuration that delivers the best reward gain
per second with low variance. All runs are launched via the project entry
point (default: main.py), so no internal trainer modifications are needed.

Example:
    python tools/auto_tune.py --iters 6 --seed-count 3 --out-json tuning.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Regex used to parse per-iteration timing emitted when SCOPONE_PROFILE=1
TIMING_RE = re.compile(
    r"\[iter-timing\]\s+total=(?P<total>[0-9.]+)s\s+collect=(?P<collect>[0-9.]+)\s+"
    r"preproc=(?P<preproc>[0-9.]+)\s+update=(?P<update>[0-9.]+)"
)


@dataclasses.dataclass(frozen=True)
class TrialConfig:
    """Configuration for a single tuning trial."""

    horizon: int
    minibatch: int

    def key(self) -> Tuple[int, int]:
        return (self.horizon, self.minibatch)


@dataclasses.dataclass
class TrialStats:
    """Outcome of one training run."""

    config: TrialConfig
    seed: int
    exit_code: int
    reward_series: List[float]
    iter_time_series: List[float]
    iterations: int
    wall_time_s: float
    stdout: str
    stderr: str
    error: Optional[str] = None

    @property
    def total_time(self) -> float:
        if self.iter_time_series:
            return sum(self.iter_time_series)
        return max(self.wall_time_s, 0.0)


@dataclasses.dataclass
class ConfigSummary:
    """Aggregated statistics for one (horizon, minibatch) pair."""

    config: TrialConfig
    seeds: List[int]
    reward_start_mean: float
    reward_final_mean: float
    reward_gain_mean: float
    reward_final_std: float
    total_time_mean: float
    iter_time_mean: float
    reward_gain_per_sec: float
    score: float
    trials: List[TrialStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically search horizon/minibatch configurations on CPU."
    )
    parser.add_argument(
        "--entry",
        default="main.py",
        help="Training entrypoint to execute (default: main.py).",
    )
    parser.add_argument(
        "--entry-args",
        default="",
        help="Extra arguments passed to the entrypoint (parsed with shlex).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of training iterations for each trial (default: 5).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(os.environ.get("SCOPONE_NUM_ENVS", "32")),
        help="Number of parallel environments to use during tuning.",
    )
    parser.add_argument(
        "--train-both-teams",
        action="store_true",
        help="Force training from both teams during the sweep (affects horizon alignment).",
    )
    parser.add_argument(
        "--horizons",
        default="auto",
        help="Comma-separated horizon candidates or 'auto' to derive sensible values.",
    )
    parser.add_argument(
        "--minibatches",
        default="auto",
        help="Comma-separated minibatch candidates or 'auto' for defaults.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=2,
        help="Number of seeds per configuration when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1234,
        help="Base seed used to derive multiple seeds when --seeds is omitted.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Explicit comma-separated list of seeds (overrides --seed-count/base).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional timeout (seconds) per trial. 0 disables timeout.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional path to dump the aggregated results as JSON.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment overrides for the training subprocess (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned trials and exit without running training.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout/stderr of each trial for debugging.",
    )
    return parser.parse_args()


def safe_positive_int(value: int, minimum: int = 1) -> int:
    return max(minimum, int(value))


def lcm(a: int, b: int) -> int:
    a = abs(int(a))
    b = abs(int(b))
    if a == 0:
        return b
    if b == 0:
        return a
    return abs(a * b) // math.gcd(a, b)


def align_horizon(horizon: int, minibatch: int, per_episode_transitions: int) -> int:
    horizon = max(40, int(horizon))
    minibatch = max(1, int(minibatch))
    per_episode_transitions = max(1, int(per_episode_transitions))
    block = lcm(minibatch, per_episode_transitions)
    if block > 0 and horizon % block != 0:
        horizon = ((horizon + block - 1) // block) * block
    return horizon


def auto_horizon_candidates(num_envs: int, per_episode_transitions: int) -> List[int]:
    num_envs = safe_positive_int(num_envs)
    steps_per_env = [256, 384, 512, 640]
    horizons = [num_envs * s for s in steps_per_env]
    aligned = {
        align_horizon(h, 4096, per_episode_transitions) for h in horizons
    }
    return sorted(aligned)


def auto_minibatch_candidates() -> List[int]:
    return [2048, 4096, 8192]


def parse_int_list(arg: str) -> List[int]:
    if not arg:
        return []
    out: List[int] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError as exc:
            raise ValueError(f"Cannot parse integer from '{chunk}'") from exc
    return out


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        seeds = parse_int_list(args.seeds)
        if not seeds:
            raise ValueError("--seeds provided but no valid integers were found.")
        return seeds
    base = int(args.seed_base)
    count = safe_positive_int(args.seed_count)
    return [base + i * 9973 for i in range(count)]


def parse_env_overrides(pairs: Sequence[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for entry in pairs:
        if "=" not in entry:
            raise ValueError(f"Invalid --env override '{entry}'. Expected KEY=VALUE.")
        key, value = entry.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def prepare_configs(
    horizons: Iterable[int],
    minibatches: Iterable[int],
    per_episode: int,
) -> List[TrialConfig]:
    unique: Dict[Tuple[int, int], TrialConfig] = {}
    for h in horizons:
        for mb in minibatches:
            aligned_h = align_horizon(h, mb, per_episode)
            cfg = TrialConfig(horizon=aligned_h, minibatch=mb)
            unique[cfg.key()] = cfg
    configs = sorted(unique.values(), key=lambda c: (c.horizon, c.minibatch))
    if not configs:
        raise ValueError("No configurations to evaluate.")
    return configs


def parse_progress(raw: str) -> List[Tuple[int, int, Dict[str, float]]]:
    updates: Dict[int, Tuple[int, int, Dict[str, float]]] = {}
    for segment in raw.split("\r"):
        segment = segment.strip()
        if not segment or not segment.startswith("it"):
            continue
        if "|" not in segment:
            continue
        prefix, body = segment.split("|", 1)
        match = re.search(r"it\s+(\d+)\s*/\s*(\d+)", prefix)
        if not match:
            continue
        iter_idx = int(match.group(1))
        total = int(match.group(2))
        metrics: Dict[str, float] = {}
        for token in body.strip().split():
            if ":" not in token:
                continue
            k, v = token.split(":", 1)
            try:
                metrics[k] = float(v)
            except ValueError:
                continue
        updates[iter_idx] = (iter_idx, total, metrics)
    return [updates[k] for k in sorted(updates.keys())]


def parse_timings(raw: str) -> List[Dict[str, float]]:
    timings: List[Dict[str, float]] = []
    for line in raw.splitlines():
        match = TIMING_RE.search(line)
        if not match:
            continue
        timings.append(
            {
                "total": float(match.group("total")),
                "collect": float(match.group("collect")),
                "preproc": float(match.group("preproc")),
                "update": float(match.group("update")),
            }
        )
    return timings


def extract_reward(metrics: Dict[str, float]) -> Optional[float]:
    for key in (
        "avg_ret",
        "avg_return",
        "avg_ret_A",
        "avg_ret_B",
        "A_avg_return",
        "B_avg_return",
    ):
        if key in metrics:
            return metrics[key]
    return None


def run_trial(
    entry: str,
    entry_args: Sequence[str],
    config: TrialConfig,
    seed: int,
    args: argparse.Namespace,
    env_overrides: Dict[str, str],
) -> TrialStats:
    env = os.environ.copy()
    env.update(
        {
            "SCOPONE_ITERS": str(args.iters),
            "SCOPONE_HORIZON": str(config.horizon),
            "SCOPONE_MINIBATCH": str(config.minibatch),
            "SCOPONE_NUM_ENVS": str(args.num_envs),
            "SCOPONE_SEED": str(seed),
            "SCOPONE_PROFILE": "1",
            "SCOPONE_AUTO_TB": "0",
            "SCOPONE_DISABLE_TB": "1",
            "SCOPONE_DEVICE": "cpu",
            "SCOPONE_TRAIN_DEVICE": "cpu",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if args.train_both_teams:
        env["SCOPONE_TRAIN_FROM_BOTH_TEAMS"] = "1"
    for key, value in env_overrides.items():
        env[key] = value

    cmd = [sys.executable, entry, *entry_args]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout if args.timeout > 0 else None,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        wall = time.monotonic() - start
        return TrialStats(
            config=config,
            seed=seed,
            exit_code=124,
            reward_series=[],
            iter_time_series=[],
            iterations=0,
            wall_time_s=wall,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=f"Timeout after {args.timeout}s",
        )
    wall = time.monotonic() - start

    progress_updates = parse_progress(stderr)
    rewards: List[float] = []
    iter_times: List[float] = []
    iterations = 0
    for iter_idx, _total, metrics in progress_updates:
        reward = extract_reward(metrics)
        if reward is not None:
            rewards.append(reward)
        if "t_s" in metrics:
            iter_times.append(metrics["t_s"])
        iterations = max(iterations, iter_idx)

    # If timing lines exist, prefer them over t_s from the preview string.
    timing_metrics = parse_timings(stdout + "\n" + stderr)
    if timing_metrics:
        iter_times = [itm["total"] for itm in timing_metrics]

    error_msg: Optional[str] = None
    if exit_code != 0:
        error_msg = f"Training exited with code {exit_code}"

    return TrialStats(
        config=config,
        seed=seed,
        exit_code=exit_code,
        reward_series=rewards,
        iter_time_series=iter_times,
        iterations=iterations,
        wall_time_s=wall,
        stdout=stdout,
        stderr=stderr,
        error=error_msg,
    )


def summarize_config(trials: List[TrialStats]) -> Optional[ConfigSummary]:
    valid = [t for t in trials if not t.error and t.reward_series]
    if not valid:
        return None

    starts = [t.reward_series[0] for t in valid if t.reward_series]
    finals = [t.reward_series[-1] for t in valid if t.reward_series]
    gains = [
        (t.reward_series[-1] - t.reward_series[0]) for t in valid if len(t.reward_series) >= 2
    ]
    total_times = [t.total_time for t in valid if t.total_time > 0]
    iter_means = [
        statistics.mean(t.iter_time_series) for t in valid if t.iter_time_series
    ]

    reward_start_mean = statistics.mean(starts) if starts else 0.0
    reward_final_mean = statistics.mean(finals) if finals else 0.0
    reward_gain_mean = statistics.mean(gains) if gains else 0.0
    reward_final_std = statistics.pstdev(finals) if len(finals) > 1 else 0.0
    total_time_mean = statistics.mean(total_times) if total_times else statistics.mean(
        [t.wall_time_s for t in valid]
    )
    iter_time_mean = statistics.mean(iter_means) if iter_means else (
        total_time_mean / valid[0].iterations if valid[0].iterations else total_time_mean
    )
    gain_per_sec = (
        reward_gain_mean / total_time_mean if total_time_mean > 0 else 0.0
    )
    noise_penalty = 1.0 + max(reward_final_std, 1e-6)
    score = gain_per_sec / noise_penalty

    return ConfigSummary(
        config=valid[0].config,
        seeds=[t.seed for t in valid],
        reward_start_mean=reward_start_mean,
        reward_final_mean=reward_final_mean,
        reward_gain_mean=reward_gain_mean,
        reward_final_std=reward_final_std,
        total_time_mean=total_time_mean,
        iter_time_mean=iter_time_mean,
        reward_gain_per_sec=gain_per_sec,
        score=score,
        trials=valid,
    )


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def print_summary(summaries: List[ConfigSummary]) -> None:
    if not summaries:
        print("No successful trials to summarize.", file=sys.stderr)
        return

    summaries = sorted(summaries, key=lambda s: s.score, reverse=True)

    header = (
        "rank horizon minibatch reward_start reward_final "
        "gain gain/sec reward_std iter_time score seeds"
    )
    print(header)
    for idx, summary in enumerate(summaries, start=1):
        gain = summary.reward_gain_mean
        gain_sec = summary.reward_gain_per_sec
        total_time = summary.total_time_mean
        iter_time = summary.iter_time_mean
        row = (
            f"{idx:>4} "
            f"{summary.config.horizon:>7d} "
            f"{summary.config.minibatch:>9d} "
            f"{summary.reward_start_mean:>12.3f} "
            f"{summary.reward_final_mean:>12.3f} "
            f"{gain:>6.3f} "
            f"{gain_sec:>8.4f} "
            f"{summary.reward_final_std:>8.4f} "
            f"{iter_time:>8.3f}s "
            f"{summary.score:>7.4f} "
            f"{','.join(str(s) for s in summary.seeds)}"
        )
        print(row)

    best = summaries[0]
    print("\nRecommended configuration:")
    print(
        f"  horizon={best.config.horizon} minibatch={best.config.minibatch} "
        f"(score={best.score:.4f}, gain/sec={best.reward_gain_per_sec:.4f})"
    )


def dump_json(path: Path, summaries: List[ConfigSummary]) -> None:
    payload = []
    for summary in summaries:
        payload.append(
            {
                "horizon": summary.config.horizon,
                "minibatch": summary.config.minibatch,
                "seeds": summary.seeds,
                "reward_start_mean": summary.reward_start_mean,
                "reward_final_mean": summary.reward_final_mean,
                "reward_gain_mean": summary.reward_gain_mean,
                "reward_final_std": summary.reward_final_std,
                "total_time_mean": summary.total_time_mean,
                "iter_time_mean": summary.iter_time_mean,
                "reward_gain_per_sec": summary.reward_gain_per_sec,
                "score": summary.score,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote JSON results to {path}")


def main() -> None:
    args = parse_args()
    entry = args.entry
    if entry.endswith(".py"):
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry = str(entry_path)
    entry_args = shlex.split(args.entry_args) if args.entry_args else []

    seeds = resolve_seeds(args)
    per_episode_transitions = 40 if args.train_both_teams else 20

    if args.horizons.lower() == "auto":
        horizon_candidates = auto_horizon_candidates(args.num_envs, per_episode_transitions)
    else:
        horizon_candidates = [
            align_horizon(h, 4096, per_episode_transitions)
            for h in parse_int_list(args.horizons)
        ]

    if args.minibatches.lower() == "auto":
        minibatch_candidates = auto_minibatch_candidates()
    else:
        minibatch_candidates = [
            safe_positive_int(mb) for mb in parse_int_list(args.minibatches)
        ]

    configs = prepare_configs(
        horizons=horizon_candidates,
        minibatches=minibatch_candidates,
        per_episode=per_episode_transitions,
    )

    env_overrides = parse_env_overrides(args.env)

    print("Planned configurations:")
    for cfg in configs:
        print(f"  horizon={cfg.horizon} minibatch={cfg.minibatch}")
    print(f"Seeds: {', '.join(str(s) for s in seeds)}")
    if args.dry_run:
        return

    all_trials: Dict[Tuple[int, int], List[TrialStats]] = {cfg.key(): [] for cfg in configs}

    for cfg in configs:
        for seed in seeds:
            print(
                f"\n=== Running horizon={cfg.horizon} minibatch={cfg.minibatch} seed={seed} ==="
            )
            trial = run_trial(entry, entry_args, cfg, seed, args, env_overrides)
            all_trials[cfg.key()].append(trial)
            if args.verbose:
                print("--- stdout ---")
                print(trial.stdout)
                print("--- stderr ---")
                print(trial.stderr)
            if trial.error:
                print(f"[WARN] {trial.error}")
            else:
                reward_info = (
                    f"reward_start={trial.reward_series[0]:.3f} "
                    f"reward_final={trial.reward_series[-1]:.3f}"
                    if trial.reward_series
                    else "reward_series=<missing>"
                )
                mean_iter = (
                    statistics.mean(trial.iter_time_series)
                    if trial.iter_time_series
                    else trial.total_time / max(trial.iterations, 1)
                )
                print(
                    f"Completed in {trial.wall_time_s:.2f}s | "
                    f"{reward_info} | iter_time≈{mean_iter:.3f}s"
                )

    summaries: List[ConfigSummary] = []
    for key, trials in all_trials.items():
        summary = summarize_config(trials)
        if summary:
            summaries.append(summary)
        else:
            horizon, minibatch = key
            print(
                f"[WARN] No valid data for horizon={horizon} minibatch={minibatch}; check logs.",
                file=sys.stderr,
            )

    print_summary(summaries)
    if args.out_json:
        dump_json(Path(args.out_json), summaries)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
