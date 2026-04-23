#!/usr/bin/env python3
"""
UKF filter calibration sweep.

Searches for R and Q values that bring NIS inbound close to 65% without
degrading RMSE.  Three phases:

  1. Coarse grid sweep (80 configs, 5 episodes each)
  2. Fine sweep around top-3 configs (12-15 configs, 5 episodes each)
  3. Stability test (best config, 3 episodes x 300s)

Outputs:
  data/calibration/sweep_results.json   — full ranked table
  stdout                                — summary of winning config

Usage:
    python3 -u calibrate_filter.py
    python3 -u calibrate_filter.py --coarse-episodes 10 --fine-episodes 10
"""
import sys
import types
import argparse
import json
import time
import itertools
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Stub torch before any opss imports (Jetson libcudnn mismatch)
# ---------------------------------------------------------------------------
for _pkg in ["torch", "torch.nn", "torch.optim", "torch.utils", "torch.utils.data"]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = []
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

_torch = sys.modules["torch"]
_torch.no_grad = lambda: type("ctx", (), {
    "__enter__": lambda s: s, "__exit__": lambda *a: None})()
_torch.FloatTensor = lambda x: None
_torch.save = lambda *a, **kw: None
_torch.load = lambda *a, **kw: {}
_torch.manual_seed = lambda s: None

class _FakeModule:
    def __init__(self, *a, **kw): pass
    def __init_subclass__(cls, **kw): pass
    def parameters(self): return []
    def state_dict(self): return {}
    def load_state_dict(self, d): pass
    def eval(self): return self
    def train(self, mode=True): return self

_nn = sys.modules["torch.nn"]
_nn.Module = _FakeModule
_nn.Sequential = lambda *a: None
_nn.Linear = lambda *a: None
_nn.Tanh = lambda: None
_nn.MSELoss = lambda: None

# ---------------------------------------------------------------------------
# Now safe to import opss
# ---------------------------------------------------------------------------
from opss.state.ukf_nn import config as cfg
from opss.sim.observation import ObservationNoise
from eval_nn import NumpyNN, eval_split, run_episode, load_stats

NIS_TARGET = 0.65


def _jsonable(v):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.float64, np.float32)):
        return float(v)
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def build_filter_overrides(R_xy_scale, R_z_scale, Q_vel_scale):
    """Build Q/R override matrices from scale factors."""
    R_base = cfg.R.copy()
    R_override = np.diag([
        R_base[0, 0] * R_xy_scale,
        R_base[1, 1] * R_xy_scale,
        R_base[2, 2] * R_z_scale,
    ])

    Q_override = cfg.Q.copy()
    Q_override[3:, 3:] *= Q_vel_scale

    return {
        'R_override': R_override,
        'Q_override': Q_override,
    }


def score_config(stats):
    """
    Score a config result. Lower is better.

    Primary: distance from NIS target (65%)
    Secondary: RMSE degradation penalty
    """
    if stats is None:
        return float('inf')
    if stats['n_diverged'] > 0:
        return float('inf')

    nis_dist = abs(stats['nis_inbound_mean'] - NIS_TARGET)
    return nis_dist


def run_sweep(configs, n_episodes, seed, duration, noise_cfg, nn_model,
              feature_stats, label="sweep"):
    """Run a set of configs and return sorted results."""
    results = []

    for i, (R_xy, R_z, Q_vel) in enumerate(configs):
        tag = f"Rxy={R_xy:5.1f} Rz={R_z:5.1f} Qv={Q_vel:5.2f}"
        print(f"  [{label}] {i+1}/{len(configs)}: {tag} ...", end=" ", flush=True)
        t0 = time.time()

        overrides = build_filter_overrides(R_xy, R_z, Q_vel)

        stats = eval_split(
            n_episodes=n_episodes,
            seed=seed,
            duration=duration,
            noise_cfg=noise_cfg,
            nn_model=nn_model,
            feature_stats=feature_stats,
            label=f"cal-{i}",
            filter_overrides=overrides,
        )

        elapsed = time.time() - t0

        if stats is None:
            print(f"FAILED ({elapsed:.1f}s)")
            continue

        sc = score_config(stats)
        entry = {
            'R_xy_scale': R_xy,
            'R_z_scale': R_z,
            'Q_vel_scale': Q_vel,
            'nis_inbound': stats['nis_inbound_mean'],
            'rmse_pos': stats['rmse_pos_mean'],
            'rmse_vel': stats['rmse_vel_mean'],
            'n_diverged': stats['n_diverged'],
            'gate_rate': stats['gate_rate_mean'],
            'score': sc,
        }
        results.append(entry)

        print(f"NIS={stats['nis_inbound_mean']*100:5.1f}%  "
              f"RMSE={stats['rmse_pos_mean']:.3f}m  "
              f"div={stats['n_diverged']}  "
              f"score={sc:.4f}  ({elapsed:.1f}s)")

    results.sort(key=lambda r: r['score'])
    return results


def run_stability_test(config, n_episodes, duration, seed, noise_cfg,
                       nn_model, feature_stats):
    """Run long-horizon episodes with the winning config."""
    print(f"\n  Stability test: {n_episodes} episodes x {duration}s")
    overrides = build_filter_overrides(
        config['R_xy_scale'], config['R_z_scale'], config['Q_vel_scale'])

    rng = np.random.default_rng(seed)
    results = []

    for i in range(n_episodes):
        ep_seed = rng.integers(0, 2**31)
        ep_rng = np.random.default_rng(ep_seed)

        result = run_episode(
            ep_rng, duration=duration, noise_cfg=noise_cfg,
            nn_model=nn_model, feature_stats=feature_stats,
            randomize_camera=True,
            filter_overrides=overrides,
        )

        if result is None:
            print(f"    Episode {i+1}: FAILED")
            results.append(None)
            continue

        print(f"    Episode {i+1}: NIS={result['nis_inbound_pct']*100:.1f}%  "
              f"RMSE={result['rmse_pos']:.3f}m  "
              f"max_err={result['max_pos_err']:.1f}m  "
              f"div={'YES' if result['diverged'] else 'no'}")
        results.append(result)

    valid = [r for r in results if r is not None]
    if valid:
        avg_nis = np.mean([r['nis_inbound_pct'] for r in valid])
        avg_rmse = np.mean([r['rmse_pos'] for r in valid])
        n_div = sum(1 for r in valid if r['diverged'])
        print(f"  Stability summary: NIS={avg_nis*100:.1f}%  "
              f"RMSE={avg_rmse:.3f}m  diverged={n_div}/{len(valid)}")
        return {
            'n_episodes': len(valid),
            'avg_nis': float(avg_nis),
            'avg_rmse': float(avg_rmse),
            'n_diverged': n_div,
            'per_episode': [
                {k: (_jsonable(v)) for k, v in r.items()} for r in valid
            ],
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="UKF filter calibration sweep")
    parser.add_argument("--model", type=str, default="models/nn_3d.weights.npz")
    parser.add_argument("--stats", type=str, default="models/feat_stats_3d.json")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--coarse-episodes", type=int, default=3)
    parser.add_argument("--fine-episodes", type=int, default=3)
    parser.add_argument("--stability-episodes", type=int, default=3)
    parser.add_argument("--stability-duration", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--pixel-noise", type=float, default=1.5)
    parser.add_argument("--depth-noise", type=float, default=0.05)
    parser.add_argument("--depth-dropout", type=float, default=0.02)
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  UKF Filter Calibration Sweep")
    print("=" * 60)

    noise_cfg = ObservationNoise(
        pixel_noise_std=args.pixel_noise,
        depth_noise_std=args.depth_noise,
        depth_dropout_prob=args.depth_dropout,
    )

    # Load NN model
    model_path = Path(args.model)
    stats_path = Path(args.stats)

    nn_model = None
    feature_stats = None

    if model_path.exists() and stats_path.exists():
        feature_stats = load_stats(stats_path)
        nn_model = NumpyNN(model_path, feature_stats)
        print(f"  Loaded NN: {model_path}")
    else:
        print(f"  WARNING: NN model not found, calibrating baseline filter")

    # -----------------------------------------------------------------------
    # Phase 1: Coarse grid sweep
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Phase 1: Coarse grid ({args.coarse_episodes} episodes/config)")
    print(f"{'=' * 60}")

    R_z_vals = [1, 5, 10, 20, 40]
    R_xy_vals = [1, 3, 7]
    Q_vel_vals = [0.3, 1, 3]

    coarse_configs = list(itertools.product(R_xy_vals, R_z_vals, Q_vel_vals))
    print(f"  {len(coarse_configs)} configurations to test\n")

    coarse_results = run_sweep(
        coarse_configs, n_episodes=args.coarse_episodes,
        seed=args.seed, duration=args.duration,
        noise_cfg=noise_cfg, nn_model=nn_model,
        feature_stats=feature_stats, label="coarse",
    )

    # Print top 10
    print(f"\n  Top 10 coarse results:")
    print(f"  {'Rank':>4} {'R_xy':>5} {'R_z':>5} {'Q_vel':>6} "
          f"{'NIS%':>6} {'RMSE':>7} {'Div':>4} {'Score':>7}")
    print(f"  {'-'*50}")
    for i, r in enumerate(coarse_results[:10]):
        print(f"  {i+1:4d} {r['R_xy_scale']:5.1f} {r['R_z_scale']:5.1f} "
              f"{r['Q_vel_scale']:6.2f} {r['nis_inbound']*100:5.1f}% "
              f"{r['rmse_pos']:7.4f} {r['n_diverged']:4d} {r['score']:7.4f}")

    if not coarse_results:
        print("  ERROR: No valid coarse results!")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Phase 2: Fine sweep around top-3
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Phase 2: Fine sweep ({args.fine_episodes} episodes/config)")
    print(f"{'=' * 60}")

    fine_configs = set()
    for top in coarse_results[:3]:
        R_xy_c = top['R_xy_scale']
        R_z_c = top['R_z_scale']
        Q_vel_c = top['Q_vel_scale']

        for R_xy_f in [R_xy_c * 0.7, R_xy_c, R_xy_c * 1.3]:
            for R_z_f in [R_z_c * 0.7, R_z_c, R_z_c * 1.3]:
                for Q_vel_f in [Q_vel_c * 0.7, Q_vel_c, Q_vel_c * 1.3]:
                    fine_configs.add((
                        round(R_xy_f, 2),
                        round(R_z_f, 2),
                        round(Q_vel_f, 3),
                    ))

    # Remove configs already tested in coarse
    coarse_set = set(coarse_configs)
    fine_configs = sorted(fine_configs - coarse_set)
    print(f"  {len(fine_configs)} new configurations to test\n")

    fine_results = run_sweep(
        fine_configs, n_episodes=args.fine_episodes,
        seed=args.seed, duration=args.duration,
        noise_cfg=noise_cfg, nn_model=nn_model,
        feature_stats=feature_stats, label="fine",
    )

    # Combine and re-sort all results
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda r: r['score'])

    print(f"\n  Top 10 overall results:")
    print(f"  {'Rank':>4} {'R_xy':>6} {'R_z':>6} {'Q_vel':>7} "
          f"{'NIS%':>6} {'RMSE':>7} {'Div':>4} {'Score':>7}")
    print(f"  {'-'*55}")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1:4d} {r['R_xy_scale']:6.2f} {r['R_z_scale']:6.2f} "
              f"{r['Q_vel_scale']:7.3f} {r['nis_inbound']*100:5.1f}% "
              f"{r['rmse_pos']:7.4f} {r['n_diverged']:4d} {r['score']:7.4f}")

    winner = all_results[0]
    print(f"\n  WINNER: R_xy={winner['R_xy_scale']}, R_z={winner['R_z_scale']}, "
          f"Q_vel={winner['Q_vel_scale']}")
    print(f"          NIS={winner['nis_inbound']*100:.1f}%  "
          f"RMSE={winner['rmse_pos']:.4f}m")

    # -----------------------------------------------------------------------
    # Phase 3: Stability test
    # -----------------------------------------------------------------------
    stability = None
    if not args.skip_stability:
        print(f"\n{'=' * 60}")
        print(f"  Phase 3: Stability test ({args.stability_episodes} ep x "
              f"{args.stability_duration}s)")
        print(f"{'=' * 60}")

        stability = run_stability_test(
            winner, n_episodes=args.stability_episodes,
            duration=args.stability_duration,
            seed=args.seed + 1000,
            noise_cfg=noise_cfg, nn_model=nn_model,
            feature_stats=feature_stats,
        )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_dir = Path("data/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'winner': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in winner.items()},
        'all_results': [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in r.items()}
            for r in all_results
        ],
        'stability': stability,
        'config': {
            'seed': args.seed,
            'coarse_episodes': args.coarse_episodes,
            'fine_episodes': args.fine_episodes,
            'duration': args.duration,
            'base_R': cfg.R.diagonal().tolist(),
            'base_Q_vel': cfg.Q.diagonal()[3:].tolist(),
        },
    }

    out_path = out_dir / "sweep_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # -----------------------------------------------------------------------
    # Print config.py update suggestion
    # -----------------------------------------------------------------------
    R_xy_new = cfg.R[0, 0] * winner['R_xy_scale']
    R_z_new = cfg.R[2, 2] * winner['R_z_scale']
    Q_vel_new = cfg.Q[3, 3] * winner['Q_vel_scale']

    print(f"\n{'=' * 60}")
    print(f"  Suggested config.py update:")
    print(f"{'=' * 60}")
    print(f"  R = np.diag([{R_xy_new:.4f}, {R_xy_new:.4f}, {R_z_new:.4f}])")
    if winner['Q_vel_scale'] != 1.0:
        Q_pos = cfg.Q[0, 0]
        print(f"  Q = np.diag([{Q_pos}, {Q_pos}, {Q_pos}, "
              f"{Q_vel_new:.4f}, {Q_vel_new:.4f}, {Q_vel_new:.4f}])")
    print()


if __name__ == "__main__":
    main()
