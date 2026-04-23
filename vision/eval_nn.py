#!/usr/bin/env python3
"""
Evaluate NN-on vs NN-off tracker performance on test episodes.

Runs each test episode through the full sim→detection→tracker pipeline twice:
  1. NN disabled (baseline): delta_a = 0
  2. NN enabled: delta_a = model(features)

Reports per-episode and aggregate RMSE (position + velocity), plus
the improvement ratio.

Usage:
    python eval_nn.py
    python eval_nn.py --model models/nn_3d.pt --stats models/feat_stats_3d.json
    python eval_nn.py --test-dir data/episodes/test --episodes 30
"""
import sys
import types
import argparse
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
from opss.sim.camera import look_at_camera
from opss.sim.projection import world_to_detection
from opss.sim.observation import ObservationNoise
from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics
from opss.state.ukf_nn import config as cfg
from opss.state.ukf_nn.features import load_stats, normalize_features
from generate_training_data import simulate_drone, resample


class NumpyNN:
    """
    Lightweight NN wrapper that runs inference with numpy (no torch needed).

    Forward: y = tanh(tanh(x @ W1.T + b1) @ W2.T + b2) * a_max

    Compatible with UKF3D.predict() which calls predict_numpy(features_normalized).
    Features are already normalized by FeatureExtractor3D.normalize() before
    this method is called.
    """

    def __init__(self, weights_path, stats):
        self.stats = stats
        self.a_max = cfg.A_MAX
        self._load_weights(weights_path)

    def _load_weights(self, path):
        """Load weights from .npz format."""
        path = Path(path)

        if path.suffix == '.npz':
            data = np.load(path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            return

        # Check for companion .weights.npz
        npz_path = path.with_suffix('.weights.npz')
        if npz_path.exists():
            data = np.load(npz_path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            return

        raise FileNotFoundError(
            f"Cannot load weights from {path}. "
            f"Provide a .weights.npz file (train with train_numpy.py)."
        )

    def predict_numpy(self, features_normalized):
        """
        Forward pass with pre-normalized features.

        Called by UKF3D.predict() — features are already normalized by
        FeatureExtractor3D.normalize() using the loaded stats.

        Args:
            features_normalized: (15,) numpy array

        Returns:
            delta_a: (3,) numpy array
        """
        x = features_normalized
        h = np.tanh(x @ self.W1.T + self.b1)
        out = np.tanh(h @ self.W2.T + self.b2) * self.a_max
        return out


def run_episode(rng, duration, noise_cfg, nn_model=None, feature_stats=None,
                randomize_camera=True, cam_override=None, sim_overrides=None,
                filter_overrides=None, save_trace=False, dropout_intervals=None):
    """
    Run one episode through sim→detection→tracker and return metrics.

    Args:
        sim_overrides: optional dict with keys like C_d_override, tau_override,
                       wind_tau_override, wind_strength_override for OOD testing.
        save_trace: if True, include per-step trace arrays in return dict.
        dropout_intervals: list of (start_frame, end_frame) tuples where
                          measurements are suppressed.

    Returns:
        dict with rmse_pos, rmse_vel, rmse_per_axis, n_steps, n_updates
        or None if episode failed.
    """
    dt = cfg.DT
    sim_kw = sim_overrides or {}

    # 1. Generate truth trajectory
    history, tau = simulate_drone(rng, duration=duration, dt=0.001, **sim_kw)
    resampled = resample(history, dt)
    n_steps = len(resampled)

    truth_t = np.array([s[0] for s in resampled])
    truth_x = np.array([s[1] for s in resampled])
    truth_v = np.array([s[2] for s in resampled])
    truth_v_des = np.array([s[5] for s in resampled])

    # 2. Set up camera
    if cam_override is not None:
        sim_cam = cam_override
    elif randomize_camera:
        from generate_training_episodes import random_camera
        sim_cam = random_camera(rng)
    else:
        sim_cam = look_at_camera(
            position=np.array([10.0, -5.0, 10.0]),
            target=np.array([10.0, 45.0, 10.0]),
        )

    # 3. Project to detections
    det_rng = np.random.default_rng(rng.integers(0, 2**31))
    detections = []
    for k in range(n_steps):
        det = world_to_detection(truth_x[k], sim_cam, noise=noise_cfg, rng=det_rng)
        detections.append(det)

    # 4. Run tracker
    cam_intrinsics = CameraIntrinsics(
        fx=sim_cam.fx, fy=sim_cam.fy,
        cx=sim_cam.cx, cy=sim_cam.cy,
    )

    x_est = np.full((n_steps, 3), np.nan)
    v_est = np.full((n_steps, 3), np.nan)
    nis_values = []  # normalized innovation squared
    nis_per_step = np.full(n_steps, np.nan)
    was_updated = np.zeros(n_steps, dtype=bool)
    filter_kw = filter_overrides or {}
    tracker = None
    n_updates = 0
    n_gated = 0  # outlier-gated updates
    dropout_set = set()
    for intv in (dropout_intervals or []):
        for f in range(intv[0], intv[1]):
            dropout_set.add(f)

    for k in range(n_steps):
        t = truth_t[k]
        det = detections[k]
        has_det = det is not None and det["depth"] > 0
        if k in dropout_set:
            has_det = False

        if tracker is None:
            if not has_det:
                continue
            tracker = UKFNNTracker(
                track_id=0, initial_detection=det, timestamp=t,
                nn_model=nn_model, feature_stats=feature_stats,
                camera=cam_intrinsics,
                R_world_from_cam=sim_cam.R_world_from_cam,
                t_world_from_cam=sim_cam.t_world_from_cam,
                **filter_kw,
            )
            tracker.feature_extractor.set_v_des(truth_v_des[k])
            x_est[k] = tracker.x[:3]
            v_est[k] = tracker.x[3:]
            n_updates += 1
            continue

        # Set desired velocity for v_error feature computation
        tracker.feature_extractor.set_v_des(truth_v_des[k])

        # Compute analytical control acceleration
        a_control = (truth_v_des[k] - tracker.x[3:]) / tau

        # Save pre-update state to detect gating
        P_before = tracker.P.copy()

        # Predict with fixed dt (constant frame rate).
        tracker.predict(dt, a_control=a_control)

        if has_det:
            # Get innovation and S for NIS computation
            x_pred = tracker.x.copy()
            P_pred = tracker.P.copy()

            tracker.update(det, t)
            n_updates += 1
            was_updated[k] = True

            # Compute NIS from the innovation stored in feature_extractor
            innov = tracker.feature_extractor.prev_innovation
            if not np.any(np.isnan(innov)):
                # NIS = innovation^T @ S^-1 @ innovation
                # Approximate S from P_pred (measurement noise + predicted state uncertainty)
                S_approx = P_pred[:3, :3] + tracker.ukf.R
                try:
                    S_inv = np.linalg.inv(S_approx)
                    nis = innov @ S_inv @ innov
                    nis_values.append(nis)
                    nis_per_step[k] = nis
                    if nis > cfg.INNOVATION_THRESHOLD_SIGMA ** 2:
                        n_gated += 1
                except np.linalg.LinAlgError:
                    pass
        else:
            tracker.mark_missed()
            tracker.last_update = t

        x_est[k] = tracker.x[:3]
        v_est[k] = tracker.x[3:]

    # 5. Compute metrics (skip warmup)
    valid = ~np.isnan(x_est[:, 0])
    if valid.sum() < 30:
        return None

    # Use indices where we have estimates
    idx = np.where(valid)[0]
    # Skip first 30 steps for convergence
    warmup = min(30, len(idx) // 2)
    idx = idx[warmup:]

    if len(idx) < 10:
        return None

    pos_err = x_est[idx] - truth_x[idx]
    vel_err = v_est[idx] - truth_v[idx]

    rmse_pos = np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1)))
    rmse_vel = np.sqrt(np.mean(np.sum(vel_err ** 2, axis=1)))
    rmse_per_axis = np.sqrt(np.mean(pos_err ** 2, axis=0))

    # Per-axis velocity RMSE
    vel_rmse_per_axis = np.sqrt(np.mean(vel_err ** 2, axis=0))

    # Bias decomposition (mean signed error)
    pos_bias = np.mean(pos_err, axis=0)
    vel_bias = np.mean(vel_err, axis=0)

    # Position error percentiles (scalar norm)
    pos_err_norms = np.sqrt(np.sum(pos_err ** 2, axis=1))
    pos_percentiles = {
        'p50': float(np.percentile(pos_err_norms, 50)),
        'p75': float(np.percentile(pos_err_norms, 75)),
        'p90': float(np.percentile(pos_err_norms, 90)),
        'p95': float(np.percentile(pos_err_norms, 95)),
        'p99': float(np.percentile(pos_err_norms, 99)),
    }

    # NIS metrics
    nis_arr = np.array(nis_values) if nis_values else np.array([])
    chi2_95 = cfg.CHI2_95  # 7.815 for 3 DOF
    nis_inbound = (np.sum(nis_arr < chi2_95) / len(nis_arr)) if len(nis_arr) > 0 else 0.0

    # Divergence detection: position error > 10m at any point after warmup
    max_pos_err = np.sqrt(np.max(np.sum(pos_err ** 2, axis=1)))
    diverged = max_pos_err > 10.0

    result = {
        'rmse_pos': rmse_pos,
        'rmse_vel': rmse_vel,
        'rmse_per_axis': rmse_per_axis,
        'vel_rmse_per_axis': vel_rmse_per_axis,
        'pos_bias': pos_bias,
        'vel_bias': vel_bias,
        'pos_percentiles': pos_percentiles,
        'n_steps': n_steps,
        'n_updates': n_updates,
        'n_eval_steps': len(idx),
        'nis_inbound_pct': nis_inbound,
        'nis_mean': float(nis_arr.mean()) if len(nis_arr) > 0 else float('nan'),
        'n_gated': n_gated,
        'gate_rate': n_gated / max(n_updates, 1),
        'diverged': diverged,
        'max_pos_err': max_pos_err,
    }

    if save_trace:
        result['trace'] = {
            't': truth_t,
            'err_x': x_est[:, 0] - truth_x[:, 0],
            'err_y': x_est[:, 1] - truth_x[:, 1],
            'err_z': x_est[:, 2] - truth_x[:, 2],
            'err_pos': np.sqrt(np.sum((x_est - truth_x) ** 2, axis=1)),
            'nis': nis_per_step,
            'was_updated': was_updated,
        }

    return result


def eval_split(n_episodes, seed, duration, noise_cfg, nn_model=None,
               feature_stats=None, label="", sim_overrides=None,
               filter_overrides=None, save_trace=False):
    """Run n_episodes and aggregate metrics."""
    rng = np.random.default_rng(seed)
    results = []

    for i in range(n_episodes):
        # Use consistent random state per episode
        ep_seed = rng.integers(0, 2**31)
        ep_rng = np.random.default_rng(ep_seed)

        result = run_episode(
            ep_rng, duration=duration, noise_cfg=noise_cfg,
            nn_model=nn_model, feature_stats=feature_stats,
            randomize_camera=True, sim_overrides=sim_overrides,
            filter_overrides=filter_overrides, save_trace=save_trace,
        )

        if result is not None:
            results.append(result)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{label}] {i+1}/{n_episodes} episodes done")

    if not results:
        return None

    rmse_pos = np.array([r['rmse_pos'] for r in results])
    rmse_vel = np.array([r['rmse_vel'] for r in results])
    rmse_axes = np.array([r['rmse_per_axis'] for r in results])
    vel_rmse_axes = np.array([r['vel_rmse_per_axis'] for r in results])
    pos_biases = np.array([r['pos_bias'] for r in results])
    vel_biases = np.array([r['vel_bias'] for r in results])
    nis_inbound = np.array([r['nis_inbound_pct'] for r in results])
    gate_rates = np.array([r['gate_rate'] for r in results])
    diverged = np.array([r['diverged'] for r in results])

    return {
        'n_episodes': len(results),
        'rmse_pos_mean': rmse_pos.mean(),
        'rmse_pos_std': rmse_pos.std(),
        'rmse_pos_median': np.median(rmse_pos),
        'rmse_pos_p95': np.percentile(rmse_pos, 95),
        'rmse_vel_mean': rmse_vel.mean(),
        'rmse_vel_std': rmse_vel.std(),
        'rmse_per_axis_mean': rmse_axes.mean(axis=0),
        'vel_rmse_per_axis_mean': vel_rmse_axes.mean(axis=0),
        'pos_bias_mean': pos_biases.mean(axis=0),
        'vel_bias_mean': vel_biases.mean(axis=0),
        'nis_inbound_mean': nis_inbound.mean(),
        'gate_rate_mean': gate_rates.mean(),
        'n_diverged': int(diverged.sum()),
        'per_episode': results,
    }


def print_results(label, stats):
    """Pretty-print evaluation results."""
    print(f"\n  {label}")
    print(f"  {'-' * 50}")
    print(f"  Episodes:     {stats['n_episodes']}")
    print(f"  Position RMSE:  {stats['rmse_pos_mean']:.4f} +/- {stats['rmse_pos_std']:.4f} m"
          f"  (median={stats['rmse_pos_median']:.4f}, p95={stats['rmse_pos_p95']:.4f})")
    print(f"  Velocity RMSE:  {stats['rmse_vel_mean']:.4f} +/- {stats['rmse_vel_std']:.4f} m/s")
    ax = stats['rmse_per_axis_mean']
    print(f"  Per-axis pos RMSE:  x={ax[0]:.4f}  y={ax[1]:.4f}  z={ax[2]:.4f} m")
    vax = stats['vel_rmse_per_axis_mean']
    print(f"  Per-axis vel RMSE:  x={vax[0]:.4f}  y={vax[1]:.4f}  z={vax[2]:.4f} m/s")
    pb = stats['pos_bias_mean']
    vb = stats['vel_bias_mean']
    print(f"  Position bias:  x={pb[0]:+.4f}  y={pb[1]:+.4f}  z={pb[2]:+.4f} m")
    print(f"  Velocity bias:  x={vb[0]:+.4f}  y={vb[1]:+.4f}  z={vb[2]:+.4f} m/s")
    print(f"  NIS inbound:    {stats['nis_inbound_mean']*100:.1f}% (target: 65%)")
    print(f"  Gate rate:      {stats['gate_rate_mean']*100:.2f}%")
    print(f"  Diverged:       {stats['n_diverged']}/{stats['n_episodes']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NN-on vs NN-off")
    parser.add_argument("--model", type=str, default="models/nn_3d.weights.npz",
                        help="Path to NN weights (.npz or .pt)")
    parser.add_argument("--stats", type=str, default="models/feat_stats_3d.json",
                        help="Path to feature normalization stats")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of test episodes")
    parser.add_argument("--seed", type=int, default=9999,
                        help="Random seed (different from training!)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Seconds per episode")
    parser.add_argument("--pixel-noise", type=float, default=1.5)
    parser.add_argument("--depth-noise", type=float, default=0.05)
    parser.add_argument("--depth-dropout", type=float, default=0.02)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run NN-off baseline (skip NN-on)")
    parser.add_argument("--ood", action="store_true",
                        help="Run unseen-dynamics (out-of-distribution) eval")
    parser.add_argument("--save-traces", type=str, default=None, metavar="DIR",
                        help="Save per-step trace arrays as NPZ to DIR")
    parser.add_argument("--dropout-test", action="store_true",
                        help="Run dropout resilience test")
    args = parser.parse_args()

    print("OPSS NN Evaluation: NN-on vs NN-off")
    print("=" * 55)

    noise_cfg = ObservationNoise(
        pixel_noise_std=args.pixel_noise,
        depth_noise_std=args.depth_noise,
        depth_dropout_prob=args.depth_dropout,
    )

    save_trace = args.save_traces is not None

    # --- Baseline: NN off ---
    print("\n  Running baseline (NN off)...")
    baseline = eval_split(
        n_episodes=args.episodes,
        seed=args.seed,
        duration=args.duration,
        noise_cfg=noise_cfg,
        nn_model=None,
        feature_stats=None,
        label="baseline",
        save_trace=save_trace,
    )

    if baseline is None:
        print("  ERROR: No valid baseline episodes!")
        sys.exit(1)

    print_results("BASELINE (NN off)", baseline)

    if args.baseline_only:
        print("\n  (Skipping NN-on — baseline only mode)")
        if args.save_traces is not None:
            _save_traces(args.save_traces, [("baseline", baseline)])
        if args.dropout_test:
            run_dropout_test(args, noise_cfg, None, None)
        return

    # --- NN on ---
    nn_model = None
    stats = None
    nn_on = None
    model_path = Path(args.model)
    stats_path = Path(args.stats)

    if not model_path.exists():
        print(f"\n  WARNING: Model not found at {model_path}")
        print("  Skipping NN-on evaluation.")
    elif not stats_path.exists():
        print(f"\n  WARNING: Stats not found at {stats_path}")
        print("  Skipping NN-on evaluation.")
    else:
        print(f"\n  Loading model from {model_path}")
        stats = load_stats(stats_path)
        nn_model = NumpyNN(model_path, stats)
        print(f"  Loaded: W1={nn_model.W1.shape}, W2={nn_model.W2.shape}")

        print("\n  Running NN-on evaluation...")
        nn_on = eval_split(
            n_episodes=args.episodes,
            seed=args.seed,  # Same seed = same trajectories & noise
            duration=args.duration,
            noise_cfg=noise_cfg,
            nn_model=nn_model,
            feature_stats=stats,
            label="nn-on",
            save_trace=save_trace,
        )

        if nn_on is None:
            print("  ERROR: No valid NN-on episodes!")

    if nn_on is not None:
        print_results("NN ON", nn_on)

    # --- Comparison ---
    if nn_on is not None:
        print(f"\n  {'=' * 55}")
        print(f"  COMPARISON")
        print(f"  {'=' * 55}")

        pos_improvement = (1 - nn_on['rmse_pos_mean'] / baseline['rmse_pos_mean']) * 100
        vel_improvement = (1 - nn_on['rmse_vel_mean'] / baseline['rmse_vel_mean']) * 100

        print(f"  Position RMSE: {baseline['rmse_pos_mean']:.4f} -> {nn_on['rmse_pos_mean']:.4f} m"
              f"  ({pos_improvement:+.1f}%)")
        print(f"  Velocity RMSE: {baseline['rmse_vel_mean']:.4f} -> {nn_on['rmse_vel_mean']:.4f} m/s"
              f"  ({vel_improvement:+.1f}%)")

        if pos_improvement > 0:
            print(f"\n  NN IMPROVES position RMSE by {pos_improvement:.1f}%")
        else:
            print(f"\n  WARNING: NN WORSENS position RMSE by {-pos_improvement:.1f}%")

        # --- Statistical significance ---
        baseline_rmses = np.array([r['rmse_pos'] for r in baseline['per_episode']])
        nn_rmses = np.array([r['rmse_pos'] for r in nn_on['per_episode']])

        n_paired = min(len(baseline_rmses), len(nn_rmses))
        if n_paired >= 6:
            from scipy.stats import wilcoxon
            # Test H_a: baseline > nn (i.e., NN improves)
            stat, p_value = wilcoxon(
                baseline_rmses[:n_paired], nn_rmses[:n_paired],
                alternative='greater',
            )
            print(f"\n  Wilcoxon signed-rank test (n={n_paired}):")
            print(f"    statistic = {stat:.1f},  p-value = {p_value:.4f}")
            if p_value < 0.05:
                print(f"    NN improvement is statistically significant (p < 0.05)")
            else:
                print(f"    NN improvement is NOT statistically significant (p >= 0.05)")

            # Bootstrap 95% CI on mean improvement
            diffs = baseline_rmses[:n_paired] - nn_rmses[:n_paired]
            boot_rng = np.random.default_rng(12345)
            n_boot = 10000
            boot_means = np.array([
                diffs[boot_rng.integers(0, n_paired, size=n_paired)].mean()
                for _ in range(n_boot)
            ])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            print(f"    Bootstrap 95% CI on mean improvement: [{ci_lo:.4f}, {ci_hi:.4f}] m")
        else:
            print(f"\n  (Skipping significance test: need >= 6 episodes, have {n_paired})")

    # --- Unseen dynamics (out-of-distribution) ---
    if args.ood and nn_on is not None:
        print(f"\n\n{'=' * 55}")
        print("  UNSEEN DYNAMICS (out-of-distribution)")
        print(f"{'=' * 55}")
        # Training ranges: C_d=[0.02,0.08], tau=[0.8,1.5], wind_tau=[2,8], wind_strength=[0,2.5]
        ood_scenarios = [
            ("High drag (C_d=0.15, 2x training max)",
             {"C_d_override": 0.15}),
            ("Sluggish response (tau=3.0, 2x training max)",
             {"tau_override": 3.0}),
            ("Fast wind shifts (wind_tau=0.5, 4x faster than training min)",
             {"wind_tau_override": 0.5}),
            ("Strong wind (strength=5.0, 2x training max)",
             {"wind_strength_override": 5.0}),
            ("Combined OOD (high drag + sluggish + strong wind)",
             {"C_d_override": 0.12, "tau_override": 2.5, "wind_strength_override": 4.0}),
        ]

        ood_episodes = max(10, args.episodes // 3)
        ood_seed = args.seed + 5000

        for scenario_name, overrides in ood_scenarios:
            print(f"\n  --- {scenario_name} ---")

            ood_baseline = eval_split(
                n_episodes=ood_episodes, seed=ood_seed, duration=args.duration,
                noise_cfg=noise_cfg, nn_model=None, feature_stats=None,
                label=f"ood-base", sim_overrides=overrides,
            )
            ood_nn = eval_split(
                n_episodes=ood_episodes, seed=ood_seed, duration=args.duration,
                noise_cfg=noise_cfg, nn_model=nn_model, feature_stats=stats,
                label=f"ood-nn", sim_overrides=overrides,
            )

            if ood_baseline and ood_nn:
                ood_pos_imp = (1 - ood_nn['rmse_pos_mean'] / ood_baseline['rmse_pos_mean']) * 100
                print(f"  Baseline: {ood_baseline['rmse_pos_mean']:.4f}m"
                      f"  NN-on: {ood_nn['rmse_pos_mean']:.4f}m"
                      f"  ({ood_pos_imp:+.1f}%)")
                print(f"  NIS inbound: {ood_nn['nis_inbound_mean']*100:.1f}%"
                      f"  Gate rate: {ood_nn['gate_rate_mean']*100:.2f}%"
                      f"  Diverged: {ood_nn['n_diverged']}/{ood_nn['n_episodes']}")
            else:
                print("  WARNING: OOD scenario failed to produce valid episodes")

    # --- Save traces ---
    if args.save_traces is not None:
        _save_traces(args.save_traces, [("baseline", baseline), ("nn_on", nn_on)])

    # --- Dropout resilience test ---
    if args.dropout_test:
        _nn = nn_model if nn_on is not None else None
        _st = stats if nn_on is not None else None
        run_dropout_test(args, noise_cfg, _nn, _st)


def _save_traces(trace_dir, splits):
    """Save per-step trace arrays as compressed NPZ files."""
    import os
    os.makedirs(trace_dir, exist_ok=True)
    count = 0
    for label_name, split_data in splits:
        if split_data is None:
            continue
        for i, ep in enumerate(split_data['per_episode']):
            trace = ep.get('trace')
            if trace is None:
                continue
            fname = os.path.join(trace_dir, f"{label_name}_ep{i:03d}.npz")
            np.savez_compressed(fname, **trace)
            count += 1
    print(f"\n  Traces saved to {trace_dir}/ ({count} files)")


def run_dropout_test(args, noise_cfg, nn_model, feature_stats):
    """Test tracker resilience to random measurement dropout intervals."""
    print(f"\n\n{'=' * 55}")
    print("  DROPOUT RESILIENCE TEST")
    print(f"{'=' * 55}")

    n_episodes = args.episodes
    duration = args.duration
    n_frames = int(duration / cfg.DT)
    dropout_len = 30  # 1 second at 30fps
    n_dropouts = 3

    for label, nn, fs in [("Baseline", None, None), ("NN-on", nn_model, feature_stats)]:
        if label == "NN-on" and nn is None:
            continue

        rng = np.random.default_rng(args.seed + 7000)
        growth_rates = []
        recovery_times = []
        rmses = []

        for i in range(n_episodes):
            ep_rng = np.random.default_rng(rng.integers(0, 2**31))

            # Generate random dropout intervals
            max_start = n_frames - dropout_len - 60
            if max_start <= 60:
                continue  # Episode too short for dropout test
            dropout_starts = sorted(ep_rng.integers(60, max_start, size=n_dropouts))
            intervals = [(s, s + dropout_len) for s in dropout_starts]

            result = run_episode(
                ep_rng, duration=duration, noise_cfg=noise_cfg,
                nn_model=nn, feature_stats=fs,
                randomize_camera=True, save_trace=True,
                dropout_intervals=intervals,
            )
            if result is None:
                continue

            rmses.append(result['rmse_pos'])
            trace = result.get('trace')
            if trace is None:
                continue

            err_pos = trace['err_pos']
            for start, end in intervals:
                if start >= len(err_pos) or end >= len(err_pos):
                    continue
                # Error growth rate during dropout
                pre_err = err_pos[start] if np.isfinite(err_pos[start]) else 0
                post_err = err_pos[min(end, len(err_pos)-1)]
                if np.isfinite(post_err) and np.isfinite(pre_err):
                    growth = (post_err - pre_err) / dropout_len
                    growth_rates.append(growth)

                # Recovery time: frames after dropout to return to pre-dropout error
                recovery = 0
                for f in range(end, min(end + 90, len(err_pos))):
                    if np.isfinite(err_pos[f]) and err_pos[f] <= pre_err * 1.1:
                        recovery = f - end
                        break
                else:
                    recovery = 90  # did not recover
                recovery_times.append(recovery)

        if rmses:
            print(f"\n  {label} ({len(rmses)} episodes):")
            print(f"    RMSE (with dropouts): {np.mean(rmses):.4f} m")
            if growth_rates:
                print(f"    Error growth rate:    {np.mean(growth_rates):.4f} m/frame"
                      f"  (p95={np.percentile(growth_rates, 95):.4f})")
            if recovery_times:
                print(f"    Recovery time:        {np.mean(recovery_times):.1f} frames"
                      f"  (p95={np.percentile(recovery_times, 95):.1f})")


if __name__ == "__main__":
    main()
