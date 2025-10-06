#!/usr/bin/env python3
"""
Benchmark for Action-Conditioned Actor/Critic with optional IS-MCTS.

CLI:
  python tools/benchmark_ac.py \
    --games 100 \
    --compact --k-history 12 \
    --ckpt checkpoints/ppo_ac.pth \
    --mcts --sims 128 --dets 16 \
    --seed 0 \
    --out-csv results.csv --out-json summary.json
"""
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch

from environment import ScoponeEnvMA
from algorithms.is_mcts import run_is_mcts
from belief import sample_determinization
from models.action_conditioned import ActionConditionedActor, CentralValueNet


def load_actor_critic(ckpt_path: str, obs_dim: int, map_dev: torch.device):
    actor = ActionConditionedActor(obs_dim=obs_dim)
    critic = CentralValueNet(obs_dim=obs_dim)
    try:
        ckpt = torch.load(ckpt_path, map_location=map_dev)
        if 'actor' in ckpt:
            actor.load_state_dict(ckpt['actor'])
        if 'critic' in ckpt:
            critic.load_state_dict(ckpt['critic'])
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint {ckpt_path}: {e}")
    return actor, critic


def run_benchmark(games=50, use_mcts=False, sims=128, dets=16, compact=True, k_history=12, ckpt_path='', seed=0,
                  c_puct=1.0, root_temp=0.0, prior_smooth_eps=0.0, belief_particles=256, belief_ess_frac=0.5, robust_child=True,
                  root_dirichlet_alpha=0.0, root_dirichlet_eps=0.0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build nets with correct obs_dim from env
    # Create a temp env to read obs_dim
    tmp_env = ScoponeEnvMA(k_history=k_history)
    obs_dim = tmp_env.observation_space.shape[0]
    del tmp_env
    actor = ActionConditionedActor(obs_dim=obs_dim)
    critic = CentralValueNet(obs_dim=obs_dim)
    if ckpt_path:
        try:
            import os
            map_dev = torch.device(os.environ.get(
                'SCOPONE_DEVICE',
                ('cuda' if torch.cuda.is_available() and os.environ.get('TESTS_FORCE_CPU') != '1' else 'cpu')
            ))
            actor, critic = load_actor_critic(ckpt_path, obs_dim, map_dev)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint {ckpt_path}: {e}")
    actor.eval(); critic.eval()

    results = []
    import os
    device = torch.device(os.environ.get(
        'SCOPONE_DEVICE',
        ('cuda' if torch.cuda.is_available() and os.environ.get('TESTS_FORCE_CPU') != '1' else 'cpu')
    ))
    for g in range(games):
        env = ScoponeEnvMA(k_history=k_history)
        done = False
        info = {}
        while not done:
            obs = env._get_observation(env.current_player)
            legals = env.get_valid_actions()
            # legals è sempre un Tensor (A,80)
            if legals.numel() == 0:
                break
            if use_mcts:
                def policy_fn(o, leg):
                    o_t = o.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(o) else torch.tensor(o, dtype=torch.float32, device=device)
                    # leg è sempre Tensor (A,80)
                    leg_t = leg if leg.dim() == 2 else leg.unsqueeze(0)
                    leg_t = leg_t.to(device=device, dtype=torch.float32)
                    cp = getattr(env, 'current_player', 0)
                    seat = torch.zeros((1,6), dtype=torch.float32, device=device)
                    seat[0, cp] = 1.0
                    seat[0, 4] = 1.0 if cp in [0,2] else 0.0
                    seat[0, 5] = 1.0 if cp in [1,3] else 0.0
                    with torch.no_grad():
                        logits = actor(o_t.unsqueeze(0), leg_t, seat)
                    return torch.softmax(logits, dim=0)
                def value_fn(o, _env=None):
                    o_t = o.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(o) else torch.tensor(o, dtype=torch.float32, device=device)
                    cp = _env.current_player if _env is not None else 0
                    s = torch.zeros(6, dtype=torch.float32, device=device)
                    s[cp] = 1.0
                    s[4] = 1.0 if cp in [0, 2] else 0.0
                    s[5] = 1.0 if cp in [1, 3] else 0.0
                    with torch.no_grad():
                        return critic(o_t.unsqueeze(0), s.unsqueeze(0)).item()
                # belief sampler neurale per determinizzazione
                def belief_sampler_neural(_env):
                    alpha = float(os.environ.get('SCOPONE_BELIEF_BLEND_ALPHA', '0.65'))
                    noise_scale = float(os.environ.get('DET_NOISE', '0.0'))
                    return sample_determinization(_env, alpha=alpha, noise_scale=noise_scale)
                action = run_is_mcts(env, policy_fn, value_fn, num_simulations=sims, c_puct=c_puct,
                                     belief=None, num_determinization=dets, root_temperature=root_temp,
                                     prior_smooth_eps=prior_smooth_eps, robust_child=robust_child,
                                     root_dirichlet_alpha=root_dirichlet_alpha, root_dirichlet_eps=root_dirichlet_eps,
                                     belief_sampler=belief_sampler_neural)
            else:
                with torch.no_grad():
                    o_t = obs.clone().detach().to(device=device, dtype=torch.float32) if torch.is_tensor(obs) else torch.tensor(obs, dtype=torch.float32, device=device)
                    # legals è sempre Tensor (A,80)
                    leg_t = legals if legals.dim() == 2 else legals.unsqueeze(0)
                    leg_t = leg_t.to(device=device, dtype=torch.float32)
                    cp = getattr(env, 'current_player', 0)
                    seat = torch.zeros((1,6), dtype=torch.float32, device=device)
                    seat[0, cp] = 1.0
                    seat[0, 4] = 1.0 if cp in [0,2] else 0.0
                    seat[0, 5] = 1.0 if cp in [1,3] else 0.0
                    logits = actor(o_t.unsqueeze(0), leg_t, seat)
                    idx = torch.argmax(logits).item()
                action = legals[idx]
            _, _, done, info = env.step(action)
        # per-game record
        entry = {'game': g}
        if 'score_breakdown' in info:
            bd = info['score_breakdown']
            for t in [0, 1]:
                for k in ['carte', 'denari', 'settebello', 'primiera', 'scope', 'total']:
                    entry[f'team{t}_{k}'] = bd[t].get(k, 0)
            entry['win_team0'] = 1 if bd[0]['total'] > bd[1]['total'] else 0
        elif 'team_rewards' in info:
            tr = info['team_rewards']
            entry['team0_total'] = tr[0]
            entry['team1_total'] = tr[1]
            entry['win_team0'] = 1 if tr[0] > tr[1] else 0
        results.append(entry)

    df = pd.DataFrame(results)
    win_rate = float(df['win_team0'].mean()) if 'win_team0' in df else 0.0
    means = {k: float(v) for k, v in df.mean(numeric_only=True).to_dict().items()}
    vars_ = {k: float(v) for k, v in df.var(numeric_only=True).to_dict().items()}
    summary = {'games': games, 'win_rate_team0': win_rate, 'means': means, 'vars': vars_}
    return df, summary


def main():
    ap = argparse.ArgumentParser(description='Benchmark AC Actor/Critic with optional IS-MCTS')
    ap.add_argument('--mcts', action='store_true', help='Use IS-MCTS booster')
    ap.add_argument('--sims', type=int, default=128, help='MCTS simulations per move')
    ap.add_argument('--dets', type=int, default=16, help='Belief determinisations per search')
    ap.add_argument('--compact', action='store_true', help='Use compact observation')
    ap.add_argument('--k-history', type=int, default=12, help='Recent moves for compact observation')
    ap.add_argument('--ckpt', type=str, default='', help='Checkpoint path for actor/critic (optional)')
    ap.add_argument('--games', type=int, default=50, help='Number of games to play')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--c-puct', type=float, default=1.0, help='PUCT exploration constant')
    ap.add_argument('--root-temp', type=float, default=0.0, help='Root temperature for visit-based sampling')
    ap.add_argument('--prior-smooth-eps', type=float, default=0.0, help='Prior smoothing epsilon')
    ap.add_argument('--belief-particles', type=int, default=256, help='Belief particles')
    ap.add_argument('--belief-ess-frac', type=float, default=0.5, help='Belief ESS fraction')
    ap.add_argument('--robust-child', action='store_true', help='Use robust child (max visits) else max-Q')
    ap.add_argument('--root-dirichlet-alpha', type=float, default=0.0, help='Root Dirichlet alpha')
    ap.add_argument('--root-dirichlet-eps', type=float, default=0.0, help='Root Dirichlet epsilon')
    ap.add_argument('--out-csv', type=str, default='', help='Path to save per-game CSV report')
    ap.add_argument('--out-json', type=str, default='', help='Path to save summary JSON report')
    args = ap.parse_args()

    df, summary = run_benchmark(games=args.games, use_mcts=args.mcts, sims=args.sims, dets=args.dets,
                                compact=args.compact, k_history=args.k_history, ckpt_path=args.ckpt, seed=args.seed,
                                c_puct=args.c_puct, root_temp=args.root_temp, prior_smooth_eps=args.prior_smooth_eps,
                                belief_particles=args.belief_particles, belief_ess_frac=args.belief_ess_frac,
                                robust_child=args.robust_child,
                                root_dirichlet_alpha=args.root_dirichlet_alpha, root_dirichlet_eps=args.root_dirichlet_eps)
    print('Summary:', summary)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print('Saved CSV to', args.out_csv)
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print('Saved JSON to', args.out_json)


if __name__ == '__main__':
    main()

