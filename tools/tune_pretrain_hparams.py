#!/usr/bin/env python3
"""
Hyperparameter tuner for the PPO agent prior to horizon/batch/network sweeps.

It keeps the project defaults from main.py as baseline, samples candidate
settings for the early-stage PPO knobs, launches short training runs, and
reports the combination that maximises the proxy reward signal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# Ensure project root is on sys.path when running from tools/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing main triggers the project's default env wiring without starting training.
import main  # noqa: F401
from algorithms import ppo_ac
from trainers import train_ppo as trainer


@dataclass
class TrialConfig:
    belief_aux_coef: float
    entropy_schedule: str
    lr: float
    clip_ratio: float
    value_coef: float
    entropy_coef: float
    value_clip: float
    target_kl: float
    scheduler_mode: str
    scheduler_scale: float


@dataclass
class TrialResult:
    index: int
    config: TrialConfig
    best_avg_return: float
    final_avg_return: float
    mean_last_k: float
    history: List[Dict[str, Any]]
    wall_time_s: float
    status: str
    error: Optional[str] = None


class MetricsRecorder:
    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._records.clear()

    def add(self, record: Dict[str, Any]) -> None:
        self._records.append(record)

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self._records

    def best(self, key: str) -> float:
        if not self._records:
            return float("-inf")
        return max(float(rec.get(key, float("-inf"))) for rec in self._records)

    def last(self, key: str) -> float:
        if not self._records:
            return float("-inf")
        return float(self._records[-1].get(key, float("-inf")))

    def mean_last(self, key: str, count: int) -> float:
        if not self._records:
            return float("-inf")
        tail = self._records[-count:]
        vals = [float(rec.get(key, float("-inf"))) for rec in tail]
        vals = [v for v in vals if math.isfinite(v)]
        return sum(vals) / len(vals) if vals else float("-inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random search tuner for PPO pre-hyperparameters."
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of sampled trials.")
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=int(os.environ.get("SCOPONE_TUNE_ITERS", "10")),
        help="Training iterations per trial.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=int(os.environ.get("SCOPONE_HORIZON", "32768")),
        help="Rollout horizon per iteration.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=int(os.environ.get("SCOPONE_NUM_ENVS", "32")),
        help="Parallel environments to collect trajectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tuning_results.json"),
        help="Where to dump the JSON summary.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("checkpoints") / "tuning_runs",
        help="Directory for temporary checkpoints (one per trial).",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep generated checkpoints instead of deleting them.",
    )
    parser.add_argument(
        "--enable-tb",
        action="store_true",
        help="Enable TensorBoard logging during tuning (default: disabled).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="Mini-eval games per trial (0 skips trainer eval).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Checkpoint (.pth) to copy for each trial so runs start from an existing policy.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic checkpoint warm-start (default: use latest checkpoint).",
    )
    return parser.parse_args()


def find_latest_checkpoint(base_dir: Path = Path("checkpoints")) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates: List[Path] = []
    for path in base_dir.rglob("*.pth"):
        try:
            if not path.is_file():
                continue
            if "bootstrap" in path.name.lower():
                continue
            if path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_config(rng: random.Random) -> TrialConfig:
    return TrialConfig(
        belief_aux_coef=rng.choice([0.0, 0.05, 0.1, 0.2]),
        entropy_schedule=rng.choice(["linear", "cosine"]),
        lr=log_uniform(rng, 1.2e-4, 3.2e-4),
        clip_ratio=rng.choice([0.1, 0.15, 0.2]),
        value_coef=rng.choice([0.5, 0.75, 1.0]),
        entropy_coef=log_uniform(rng, 2e-4, 1e-2),
        value_clip=rng.choice([0.05, 0.1, 0.2]),
        target_kl=rng.choice([0.01, 0.015, 0.02]),
        scheduler_mode=rng.choice(["cosine", "cosine_half", "constant"]),
        scheduler_scale=rng.choice([1.0, 0.75, 0.5]),
    )


@contextmanager
def temporary_environ(updates: Dict[str, Any]) -> Iterable[None]:
    original: Dict[str, Optional[str]] = {}
    for key, value in updates.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in updates.items():
            prev = original.get(key)
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


class NullScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return None


def run_single_trial(
    index: int,
    config: TrialConfig,
    args: argparse.Namespace,
    recorder: MetricsRecorder,
) -> TrialResult:
    start = time.time()

    recorder.reset()

    env_updates = {
        "BELIEF_AUX_COEF": config.belief_aux_coef,
        "SCOPONE_ENTROPY_SCHED": config.entropy_schedule,
        "SCOPONE_DISABLE_TB": "0" if args.enable_tb else "1",
        # Disable side effects during tuning (no saves/evals/refresh)
        "SCOPONE_DISABLE_SAVE": "1",
        "SCOPONE_DISABLE_EVAL": "1",
        "SCOPONE_LEAGUE_REFRESH": 0,
        # Enforce strict/runtime settings used in main
        "SCOPONE_STRICT_CHECKS": "0",
        "SCOPONE_APPROX_GELU": "1",
        "SCOPONE_PROFILE": "0",
        # Suppress TF/absl noise as in main
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "ABSL_LOGGING_MIN_LOG_LEVEL": "3",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        # Torch compile/Dynamo/Inductor defaults from main
        "SCOPONE_TORCH_COMPILE": os.environ.get("SCOPONE_TORCH_COMPILE", "0"),
        "SCOPONE_TORCH_COMPILE_MODE": os.environ.get("SCOPONE_TORCH_COMPILE_MODE", "reduce-overhead"),
        "SCOPONE_TORCH_COMPILE_BACKEND": os.environ.get("SCOPONE_TORCH_COMPILE_BACKEND", "inductor"),
        "SCOPONE_COMPILE_VERBOSE": os.environ.get("SCOPONE_COMPILE_VERBOSE", "1"),
        "SCOPONE_INDUCTOR_AUTOTUNE": os.environ.get("SCOPONE_INDUCTOR_AUTOTUNE", "1"),
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0"),
        "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1"),
        "TORCHDYNAMO_DYNAMIC_SHAPES": os.environ.get("TORCHDYNAMO_DYNAMIC_SHAPES", "0"),
        "TORCHDYNAMO_CACHE_SIZE_LIMIT": os.environ.get("TORCHDYNAMO_CACHE_SIZE_LIMIT", "32"),
        # Observation encoding flags from main
        "OBS_INCLUDE_DEALER": os.environ.get("OBS_INCLUDE_DEALER", "1"),
        "OBS_INCLUDE_INFERRED": os.environ.get("OBS_INCLUDE_INFERRED", "0"),
        "OBS_INCLUDE_RANK_PROBS": os.environ.get("OBS_INCLUDE_RANK_PROBS", "0"),
        "OBS_INCLUDE_SCOPA_PROBS": os.environ.get("OBS_INCLUDE_SCOPA_PROBS", "0"),
        # Console/Progress defaults
        "TQDM_DISABLE": os.environ.get("TQDM_DISABLE", "0"),
        "SCOPONE_PER_ENV_TQDM": os.environ.get("SCOPONE_PER_ENV_TQDM", "0"),
    }

    current_scheduler_cfg = {"mode": config.scheduler_mode, "scale": config.scheduler_scale}

    orig_init = ppo_ac.ActionConditionedPPO.__init__
    orig_update = ppo_ac.ActionConditionedPPO.update
    orig_collect = trainer.collect_trajectory
    orig_collect_parallel = trainer.collect_trajectory_parallel
    orig_cosine = trainer.optim.lr_scheduler.CosineAnnealingLR

    def tuned_init(self, *args, **kwargs):
        kwargs.setdefault("lr", config.lr)
        kwargs.setdefault("clip_ratio", config.clip_ratio)
        kwargs.setdefault("value_coef", config.value_coef)
        kwargs.setdefault("entropy_coef", config.entropy_coef)
        kwargs.setdefault("value_clip", config.value_clip)
        kwargs.setdefault("target_kl", config.target_kl)
        return orig_init(self, *args, **kwargs)

    def tuned_update(self, batch, *args, **kwargs):
        info = orig_update(self, batch, *args, **kwargs)
        info_flat: Dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                info_flat[key] = float(value.detach().cpu().item())
            else:
                info_flat[key] = float(value)
        ret_tensor = batch.get("ret")
        if isinstance(ret_tensor, torch.Tensor) and ret_tensor.numel() > 0:
            info_flat["avg_return"] = float(ret_tensor.detach().mean().cpu().item())
        else:
            info_flat["avg_return"] = float("-inf")
        info_flat["update_step"] = int(getattr(self, "update_steps", 0))
        info_flat["lr_actor"] = float(self.opt_actor.param_groups[0]["lr"])
        info_flat["lr_critic"] = float(self.opt_critic.param_groups[0]["lr"])
        info_flat["entropy_coef_active"] = float(self.entropy_coef)
        recorder.add(info_flat)
        return info

    def tuned_collect(*args, **kwargs):
        kwargs.setdefault("gamma", 1.0)
        kwargs.setdefault("lam", 1.0)
        return orig_collect(*args, **kwargs)

    def tuned_collect_parallel(*args, **kwargs):
        kwargs.setdefault("gamma", 1.0)
        kwargs.setdefault("lam", 1.0)
        return orig_collect_parallel(*args, **kwargs)

    class PatchedCosine:
        def __init__(self, optimizer, T_max, **kwargs):
            mode = current_scheduler_cfg["mode"]
            scale = current_scheduler_cfg["scale"]
            if mode == "constant":
                self._inner = NullScheduler(optimizer)
            else:
                scaled_T = max(1, int(max(1, T_max) * scale))
                if mode == "cosine_half":
                    scaled_T = max(1, scaled_T // 2)
                self._inner = orig_cosine(optimizer, T_max=scaled_T, **kwargs)

        def step(self):
            return getattr(self._inner, "step", lambda: None)()

        def state_dict(self):
            if hasattr(self._inner, "state_dict"):
                return self._inner.state_dict()
            return {}

        def load_state_dict(self, state_dict):
            if hasattr(self._inner, "load_state_dict"):
                self._inner.load_state_dict(state_dict)

    try:
        with temporary_environ(env_updates):
            ppo_ac.ActionConditionedPPO.__init__ = tuned_init  # type: ignore[assignment]
            ppo_ac.ActionConditionedPPO.update = tuned_update  # type: ignore[assignment]
            trainer.collect_trajectory = tuned_collect  # type: ignore[assignment]
            trainer.collect_trajectory_parallel = tuned_collect_parallel  # type: ignore[assignment]
            trainer.optim.lr_scheduler.CosineAnnealingLR = PatchedCosine  # type: ignore[assignment]

            # Mirror main.py runtime/game configuration
            _selfplay = str(os.environ.get("SCOPONE_SELFPLAY", "0")).strip().lower() in [
                "1",
                "true",
                "yes",
                "on",
            ]
            _tfb = str(os.environ.get("SCOPONE_TRAIN_FROM_BOTH_TEAMS", "1")).strip().lower() in [
                "1",
                "true",
                "yes",
                "on",
            ]
            _mcts_train = str(os.environ.get("SCOPONE_MCTS_TRAIN", "0")).strip().lower() in [
                "1",
                "true",
                "yes",
                "on",
            ]
            _mcts_warmup_iters = int(os.environ.get("SCOPONE_MCTS_WARMUP_ITERS", "0"))
            _eval_use_mcts = str(os.environ.get("SCOPONE_EVAL_USE_MCTS", "0")).strip().lower() in [
                "1",
                "true",
                "yes",
                "on",
            ]
            _mcts_sims = int(os.environ.get("SCOPONE_MCTS_SIMS", "4"))
            _eval_mcts_sims = int(os.environ.get("SCOPONE_EVAL_MCTS_SIMS", "4"))
            _mcts_dets = int(os.environ.get("SCOPONE_TRAIN_MCTS_DETS_PRIOR", "2"))
            _mcts_c_puct = float(os.environ.get("SCOPONE_MCTS_C_PUCT", "1.0"))
            _mcts_root_temp = float(os.environ.get("SCOPONE_MCTS_ROOT_TEMP", "0.0"))
            _mcts_prior_eps = float(os.environ.get("SCOPONE_MCTS_PRIOR_SMOOTH_EPS", "0.0"))
            _mcts_dir_alpha = float(os.environ.get("SCOPONE_MCTS_DIRICHLET_ALPHA", "0.25"))
            _mcts_dir_eps = float(os.environ.get("SCOPONE_MCTS_DIRICHLET_EPS", "0.25"))
            _belief_particles = int(os.environ.get("SCOPONE_BELIEF_PARTICLES", "512"))
            _belief_ess = float(os.environ.get("SCOPONE_BELIEF_ESS_FRAC", "0.5"))
            _eval_every = int(os.environ.get("SCOPONE_EVAL_EVERY", "35"))
            _eval_games = (
                args.eval_games if args.eval_games > 0 else int(os.environ.get("SCOPONE_EVAL_GAMES", "10000"))
            )
            _k_history = int(os.environ.get("SCOPONE_EVAL_K_HISTORY", "39"))
            _ckpt_path_env = os.environ.get("SCOPONE_CKPT", "checkpoints/ppo_ac.pth")

            trainer.train_ppo(
                num_iterations=args.iters,
                horizon=args.horizon,
                save_every=max(args.iters + 1, 10_000),
                ckpt_path=_ckpt_path_env,
                k_history=_k_history,
                seed=args.seed,
                entropy_schedule_type=config.entropy_schedule,
                eval_every=_eval_every,
                eval_games=_eval_games,
                belief_particles=_belief_particles,
                belief_ess_frac=_belief_ess,
                mcts_in_eval=_eval_use_mcts,
                mcts_train=_mcts_train,
                mcts_sims=_mcts_sims,
                mcts_sims_eval=_eval_mcts_sims,
                mcts_dets=_mcts_dets,
                mcts_c_puct=_mcts_c_puct,
                mcts_root_temp=_mcts_root_temp,
                mcts_prior_smooth_eps=_mcts_prior_eps,
                mcts_dirichlet_alpha=_mcts_dir_alpha,
                mcts_dirichlet_eps=_mcts_dir_eps,
                num_envs=args.num_envs,
                train_both_teams=_tfb,
                use_selfplay=_selfplay,
                mcts_warmup_iters=_mcts_warmup_iters,
            )
    except Exception as exc:  # noqa: BLE001
        return TrialResult(
            index=index,
            config=config,
            best_avg_return=float("-inf"),
            final_avg_return=float("-inf"),
            mean_last_k=float("-inf"),
            history=recorder.records.copy(),
            wall_time_s=time.time() - start,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        ppo_ac.ActionConditionedPPO.__init__ = orig_init  # type: ignore[assignment]
        ppo_ac.ActionConditionedPPO.update = orig_update  # type: ignore[assignment]
        trainer.collect_trajectory = orig_collect  # type: ignore[assignment]
        trainer.collect_trajectory_parallel = orig_collect_parallel  # type: ignore[assignment]
        trainer.optim.lr_scheduler.CosineAnnealingLR = orig_cosine  # type: ignore[assignment]

        # No per-trial checkpoint directory management: delegate warm-start and
        # checkpoint handling entirely to the main/trainer defaults.

    best_avg = recorder.best("avg_return")
    final_avg = recorder.last("avg_return")
    mean_tail = recorder.mean_last("avg_return", min(3, len(recorder.records)))

    return TrialResult(
        index=index,
        config=config,
        best_avg_return=best_avg,
        final_avg_return=final_avg,
        mean_last_k=mean_tail,
        history=recorder.records.copy(),
        wall_time_s=time.time() - start,
        status="ok",
    )


def summarise(results: List[TrialResult]) -> Dict[str, Any]:
    completed = [r for r in results if r.status == "ok" and math.isfinite(r.best_avg_return)]
    if not completed:
        return {"status": "no-success", "results": [asdict(r) for r in results]}
    best = max(completed, key=lambda r: r.best_avg_return)
    leaderboard = sorted(completed, key=lambda r: r.best_avg_return, reverse=True)
    return {
        "status": "ok",
        "best": asdict(best),
        "top_5": [asdict(r) for r in leaderboard[:5]],
        "failed": [asdict(r) for r in results if r.status != "ok"],
    }


def main_cli() -> None:
    args = parse_args()
    if not args.no_resume and args.resume_from is None:
        latest = find_latest_checkpoint()
        if latest is not None:
            args.resume_from = latest
            print(f"[tuner] Using latest checkpoint for warm-start: {args.resume_from}", flush=True)
        else:
            print("[tuner] No checkpoint found; running trials from scratch.", flush=True)
    if args.resume_from is not None:
        args.resume_from = args.resume_from.resolve()
        if not args.resume_from.exists():
            raise FileNotFoundError(f"Checkpoint specified for resume not found: {args.resume_from}")

    rng = random.Random(args.seed)
    recorder = MetricsRecorder()
    results: List[TrialResult] = []

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.trials):
        config = sample_config(rng)
        result = run_single_trial(idx, config, args, recorder)
        results.append(result)
        status = "OK" if result.status == "ok" else f"FAIL ({result.error})"
        print(
            f"[trial {idx+1}/{args.trials}] status={status} "
            f"best_avg_return={result.best_avg_return:.4f} "
            f"final_avg_return={result.final_avg_return:.4f} "
            f"config={config}",
            flush=True,
        )

    summary = summarise(results)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if summary["status"] == "ok":
        best_cfg = summary["best"]["config"]
        print("\nBest configuration (apply before horizon/batch/architecture sweeps):", flush=True)
        for key, value in best_cfg.items():
            print(f"  {key}: {value}", flush=True)
        print("\nFull summary saved to:", args.output, flush=True)
    else:
        print("Tuning failed to produce a valid configuration.", flush=True)


if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        sys.exit(1)
