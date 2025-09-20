# Scopone RL - CTDE + PPO Action-Conditioned

## Overview
This project trains an agent for Scopone (10-card base) using CTDE + PPO with action-conditioned policy, compact observation, belief + IS-MCTS (optional), self-play league, and a GUI.

## Key Features
- Compact observation (hist-k + sets), seat/team embedding.
- Action-conditioned actor/critic; PPO with GAE, KL early-stop, cosine LR, entropy schedule.
- Belief module (particle filter) and IS-MCTS booster with determinisations.
- Self-play league with Elo and softmax sampling.
- Benchmark tools and evaluation utilities.

## Quick Start
### Install
```
pip install -r requirements.txt
```

### Train (PPO)
```
python trainers/train_ppo.py --iters 2000 --horizon 256 --compact --k-history 12 --seed 0 --ckpt checkpoints/ppo_ac.pth
```
Logs (if TensorBoard available):
```
tensorboard --logdir runs
```

### Benchmark (AC policy / policy+MCTS)
```
python tools/benchmark_ac.py --games 100 --compact --k-history 12 --ckpt checkpoints/ppo_ac.pth \
  --out-csv results.csv --out-json summary.json

python tools/benchmark_ac.py --mcts --sims 256 --dets 16 --games 50 --compact --k-history 12 \
  --ckpt checkpoints/ppo_ac.pth --out-json summary_mcts.json
```

### Evaluation utilities
- Versus baseline heuristic:
```
python -c "from evaluation.eval import eval_vs_baseline; print(eval_vs_baseline(games=50))"
```
- League Elo update between last two checkpoints:
```
python -c "from evaluation.eval import league_eval_and_update; print(league_eval_and_update())"
```

### GUI
Integrate the actor/critic into `scopone_gui.py` selecting checkpoint and optional IS-MCTS (work-in-progress).

Note: la codifica a storia completa legacy è stata rimossa. Usare l'osservazione compatta con `--k-history`.

## Parameters
- Observation: `--compact`, `--k-history`
- PPO: cosine LR, entropy schedule (linear), KL target, minibatch, multi-epoch
- IS-MCTS: `--mcts`, `--sims`, `--dets`
- Seeds: `--seed`

## Dependencies
See `requirements.txt` (torch, numpy, tqdm, gymnasium, pandas, openpyxl, scipy, tensorboard, numba optional).

## Structure
- `environment.py` — Gym env with compact obs and caches
- `observation.py` — encoders and features (compact + compatibilità 10823 fissa per legacy)
- `models/` — action-conditioned actor/critic encoders
- `algorithms/ppo_ac.py` — PPO (CTDE-ready) with schedules and KL control
- `belief/` — particle filter for hidden hands
- `algorithms/is_mcts.py` — IS-MCTS with determinisations
- `selfplay/league.py` — checkpoint league and Elo
- `trainers/train_ppo.py` — trainer with self-play multi-seat
- `tools/benchmark_ac.py` — benchmark CLI for AC
- `evaluation/eval.py` — evaluation helpers
- `tests/` — unit tests

## Reproducibility
Use `--seed` in trainer/benchmark. If `--seed < 0`, a random non-negative seed is generated and printed. Checkpoints include run config.

## Notes on MCTS defaults
- Training uses IS-MCTS optionally. Defaults are now neutral: `prior_smooth_eps=0.0`, `root_dirichlet_eps=0.0`. Set them explicitly if you want smoothing or root noise.
- Root temperature can be scheduled during rollout; pass `--mcts-root-temp` to override.

## Devices and AMP
- Environment runs on CPU by default; models use `SCOPONE_DEVICE` (auto-selects CUDA if available unless overridden).
- GradScaler uses the unified AMP API when on CUDA; falls back gracefully otherwise.
