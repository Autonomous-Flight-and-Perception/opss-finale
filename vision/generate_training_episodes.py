#!/usr/bin/env python3
"""
Generate training episodes by running the tracker on simulated detections.

Unlike generate_training_data.py (which logs world-frame truth only), this
script runs the full sim→detection→tracker pipeline and logs the EXACT
features that FeatureExtractor3D builds at runtime.  This makes the training
distribution identical to inference.

Supports two modes:
  - NN-off (default): tracker runs without NN, with periodic state resets.
  - NN-on (--model): DAgger mode — tracker runs with trained NN, longer reset
    interval, conservative A_MAX clamp.

Each episode NPZ contains:
    t          (N,)     timestamps
    x_true     (N, 3)   ground truth position (world frame)
    v_true     (N, 3)   ground truth velocity (world frame)
    v_des      (N, 3)   desired velocity from sim
    tau        scalar    velocity response time constant
    x_est      (N, 3)   tracker state position after update
    v_est      (N, 3)   tracker state velocity after update
    feat       (N, 15)  feature vector from build_features() BEFORE predict
    innovation (N, 3)   innovation from UKF update (NaN if skipped)
    target     (N, 3)   NN training target: a_total - a_control (drag+wind)
    valid      (N,)     bool: True if detection was in view + had valid depth

Usage:
    python generate_training_episodes.py
    python generate_training_episodes.py --n_train 200 --n_val 30 --n_test 30
    python generate_training_episodes.py --model models/nn_3d.weights.npz \\
        --stats models/feat_stats_3d.json --n_train 80 --output data/dagger
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
import json
from opss.sim.camera import look_at_camera
from opss.sim.projection import world_to_detection
from opss.sim.observation import ObservationNoise
from opss.state.ukf_nn_tracker import UKFNNTracker, CameraIntrinsics
from opss.state.ukf_nn import config as cfg
from generate_training_data import simulate_drone, resample


# ---------------------------------------------------------------------------
# Lightweight numpy NN for DAgger rollouts (no torch needed)
# ---------------------------------------------------------------------------

class NumpyNN:
    """NN inference with numpy. Compatible with UKF3D.predict_numpy() interface."""

    def __init__(self, weights_path, a_max=None):
        data = np.load(weights_path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.a_max = a_max if a_max is not None else cfg.A_MAX

    def predict_numpy(self, features_normalized):
        x = features_normalized
        h = np.tanh(x @ self.W1.T + self.b1)
        return np.tanh(h @ self.W2.T + self.b2) * self.a_max


# ---------------------------------------------------------------------------
# Camera pose randomization
# ---------------------------------------------------------------------------

def random_camera(rng, flight_bounds=None):
    """
    Generate a random camera pose that views the flight volume.

    The camera is placed along the y-axis boundary (looking into the volume)
    with randomized position and look-at point.
    """
    if flight_bounds is None:
        flight_bounds = {"x": (0, 20), "y": (0, 90), "z": (0, 20)}

    bx = flight_bounds["x"]
    by = flight_bounds["y"]
    bz = flight_bounds["z"]

    # Camera position: near the y-min edge, randomized x and z
    cam_x = rng.uniform(bx[0] + 2, bx[1] - 2)
    cam_y = rng.uniform(-10.0, -2.0)   # outside the volume
    cam_z = rng.uniform(bz[0] + 2, bz[1] - 2)

    # Look-at: somewhere in the middle of the volume
    tgt_x = rng.uniform(bx[0] + 3, bx[1] - 3)
    tgt_y = rng.uniform(by[0] + 10, by[1] - 10)
    tgt_z = rng.uniform(bz[0] + 3, bz[1] - 3)

    cam = look_at_camera(
        position=np.array([cam_x, cam_y, cam_z]),
        target=np.array([tgt_x, tgt_y, tgt_z]),
    )
    return cam


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episode(rng, duration=30.0, noise_cfg=None, randomize_camera=True,
                     nn_model=None, feature_stats=None, reset_interval=30):
    """
    Generate one training episode: sim → detections → tracker → logs.

    Args:
        nn_model: optional NumpyNN for DAgger (NN-on) rollouts.
        feature_stats: normalization stats dict (required if nn_model given).
        reset_interval: steps between state resets (30 for NN-off, 90+ for NN-on).

    Returns:
        dict with all logged arrays, or None if too few valid steps.
    """
    dt = cfg.DT  # 1/30

    # 1. Generate truth trajectory
    history, tau = simulate_drone(rng, duration=duration, dt=0.001)
    resampled = resample(history, dt)
    n_steps = len(resampled)

    truth_t = np.array([s[0] for s in resampled])
    truth_x = np.array([s[1] for s in resampled])
    truth_v = np.array([s[2] for s in resampled])
    truth_v_des = np.array([s[5] for s in resampled])

    # 2. Set up camera
    if randomize_camera:
        sim_cam = random_camera(rng)
    else:
        sim_cam = look_at_camera(
            position=np.array([10.0, -5.0, 10.0]),
            target=np.array([10.0, 45.0, 10.0]),
        )

    if noise_cfg is None:
        noise_cfg = ObservationNoise()

    # 3. Project to detections
    det_rng = np.random.default_rng(rng.integers(0, 2**31))
    detections = []
    for k in range(n_steps):
        det = world_to_detection(truth_x[k], sim_cam, noise=noise_cfg, rng=det_rng)
        detections.append(det)

    # 4. Run tracker (NN disabled)
    cam_intrinsics = CameraIntrinsics(
        fx=sim_cam.fx, fy=sim_cam.fy,
        cx=sim_cam.cx, cy=sim_cam.cy,
    )

    # Allocate log arrays
    x_est = np.full((n_steps, 3), np.nan)
    v_est = np.full((n_steps, 3), np.nan)
    feat = np.full((n_steps, cfg.FEAT_DIM), np.nan)
    innovations = np.full((n_steps, 3), np.nan)
    valid = np.zeros(n_steps, dtype=bool)

    tracker = None
    init_k = None  # step at which tracker was initialized

    # How often to reset tracker state to truth.
    # NN-off: every 30 steps (1s) — prevents divergence from unmodeled thrust.
    # NN-on (DAgger): every 90+ steps (3s+) — NN keeps tracker stable, but
    # resets still bound long-term drift.
    RESET_INTERVAL = reset_interval

    for k in range(n_steps):
        t = truth_t[k]
        det = detections[k]

        has_det = det is not None and det["depth"] > 0

        if tracker is None:
            if not has_det:
                continue
            # Initialize tracker on first valid detection
            tracker = UKFNNTracker(
                track_id=0, initial_detection=det, timestamp=t,
                nn_model=nn_model, feature_stats=feature_stats,
                camera=cam_intrinsics,
                R_world_from_cam=sim_cam.R_world_from_cam,
                t_world_from_cam=sim_cam.t_world_from_cam,
            )
            init_k = k
            # Set v_des on feature extractor for initial features
            tracker.feature_extractor.set_v_des(truth_v_des[k])
            x_est[k] = tracker.x[:3]
            v_est[k] = tracker.x[3:]
            feat[k] = tracker.feature_extractor.build_features(tracker.x)
            valid[k] = True
            continue

        # Periodic state reset: snap tracker state to truth to prevent
        # NN-off divergence.  Also reset prev_innovation — extreme innovations
        # from divergent steps would otherwise pollute features after the reset.
        # At inference with NN on, innovations should be small, so resetting
        # to zero is representative.
        if (k - init_k) % RESET_INTERVAL == 0:
            tracker.x[:3] = truth_x[k].copy()
            tracker.x[3:] = truth_v[k].copy()
            tracker.feature_extractor.prev_innovation = np.zeros(3)

        # Set desired velocity for v_error feature computation
        tracker.feature_extractor.set_v_des(truth_v_des[k])

        # --- Tap features BEFORE predict (this is what NN sees at inference) ---
        feat[k] = tracker.feature_extractor.build_features(tracker.x)

        # Compute analytical control acceleration: a_control = (v_des - v_est) / tau
        a_control = (truth_v_des[k] - tracker.x[3:]) / tau

        # Predict with fixed dt (episodes are generated at constant frame rate).
        # Do NOT use (t - tracker.last_update): last_update only advances on
        # measurement updates, so missed detections would cause dt to grow and
        # gravity to be double-counted across consecutive predict steps.
        tracker.predict(dt, a_control=a_control)

        if has_det:
            # Update with detection (also sets tracker.last_update = t)
            tracker.update(det, t)
            # Innovation was committed inside update; read it back
            innovations[k] = tracker.feature_extractor.prev_innovation
        else:
            tracker.mark_missed()
            # Keep last_update current so downstream code (e.g. eval) that
            # computes dt from last_update doesn't accumulate stale gaps.
            tracker.last_update = t

        # valid = tracker is initialized (feat and state are usable for training).
        # Includes predict-only steps (missed detections), because the NN also
        # runs during predict when measurements are missing at inference time.
        valid[k] = True

        x_est[k] = tracker.x[:3]
        v_est[k] = tracker.x[3:]

    # 5. Compute targets: residual after removing known physics + control
    # Process model: v_next = v + dt*(gravity + a_hover + a_control + delta_a_nn)
    # With gravity + a_hover = [0,0,0]:
    #   target = a_total - a_control = a_drag + a_wind
    target = np.full((n_steps, 3), np.nan)
    for k in range(n_steps - 1):
        a_total = (truth_v[k + 1] - truth_v[k]) / dt
        a_control_k = (truth_v_des[k] - truth_v[k]) / tau
        # Target is everything EXCEPT known physics and control
        target[k] = a_total - a_control_k
    # Clip to A_MAX
    target = np.clip(target, -cfg.A_MAX, cfg.A_MAX)

    # 6. Count usable steps (valid AND has target)
    usable = valid & ~np.isnan(target[:, 0])
    n_usable = usable.sum()

    if n_usable < 10:
        return None

    return {
        "t": truth_t,
        "x_true": truth_x,
        "v_true": truth_v,
        "v_des": truth_v_des,
        "tau": float(tau),
        "x_est": x_est,
        "v_est": v_est,
        "feat": feat,
        "innovation": innovations,
        "target": target,
        "valid": valid,
        "n_usable": int(n_usable),
        "n_steps": n_steps,
        "cam_pos": sim_cam.t_world_from_cam,
        "R_world_from_cam": sim_cam.R_world_from_cam,
    }


def write_episode_npz(path, episode):
    """Write episode data to NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        t=episode["t"],
        x_true=episode["x_true"],
        v_true=episode["v_true"],
        v_des=episode["v_des"],
        tau=episode["tau"],
        x_est=episode["x_est"],
        v_est=episode["v_est"],
        feat=episode["feat"],
        innovation=episode["innovation"],
        target=episode["target"],
        valid=episode["valid"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate logged-feature training episodes")
    parser.add_argument("--n_train", type=int, default=200, help="Training episodes")
    parser.add_argument("--n_val", type=int, default=30, help="Validation episodes")
    parser.add_argument("--n_test", type=int, default=30, help="Test episodes")
    parser.add_argument("--output", type=str, default="data/episodes", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds per episode")
    parser.add_argument("--pixel-noise", type=float, default=1.5)
    parser.add_argument("--depth-noise", type=float, default=0.05)
    parser.add_argument("--depth-dropout", type=float, default=0.02)
    # DAgger mode: provide trained model for NN-on rollouts
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained NN weights (.npz) for DAgger mode")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to feature stats (.json) for DAgger mode")
    parser.add_argument("--dagger-a-max", type=float, default=10.0,
                        help="Conservative A_MAX clamp for DAgger NN output")
    parser.add_argument("--reset-interval", type=int, default=None,
                        help="State reset interval (default: 30 NN-off, 90 NN-on)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    rng = np.random.default_rng(args.seed)

    noise_cfg = ObservationNoise(
        pixel_noise_std=args.pixel_noise,
        depth_noise_std=args.depth_noise,
        depth_dropout_prob=args.depth_dropout,
    )

    # Load NN model for DAgger mode
    nn_model = None
    feature_stats = None
    if args.model is not None:
        nn_model = NumpyNN(args.model, a_max=args.dagger_a_max)
        print(f"  DAgger mode: NN-on rollouts (A_MAX clamp={args.dagger_a_max})")
        print(f"    Model: {args.model} (W1={nn_model.W1.shape})")

        if args.stats is None:
            raise ValueError("--stats required when --model is given")
        with open(args.stats) as f:
            feature_stats = json.load(f)
        print(f"    Stats: {args.stats}")

    # Default reset interval: 30 for NN-off, 90 for DAgger
    reset_interval = args.reset_interval
    if reset_interval is None:
        reset_interval = 90 if nn_model is not None else 30
    print(f"  Reset interval: {reset_interval} steps ({reset_interval/30:.1f}s)")

    splits = [
        ("train", args.n_train),
        ("val", args.n_val),
        ("test", args.n_test),
    ]

    total = sum(n for _, n in splits)
    run_idx = 0

    for split_name, n_episodes in splits:
        split_dir = out_dir / split_name
        n_usable_total = 0
        n_failed = 0

        for i in range(n_episodes):
            episode = generate_episode(
                rng, duration=args.duration,
                noise_cfg=noise_cfg, randomize_camera=True,
                nn_model=nn_model, feature_stats=feature_stats,
                reset_interval=reset_interval,
            )

            if episode is None:
                n_failed += 1
                run_idx += 1
                continue

            npz_path = split_dir / f"ep_{i:04d}.npz"
            write_episode_npz(npz_path, episode)
            n_usable_total += episode["n_usable"]

            run_idx += 1
            if run_idx % 20 == 0 or run_idx == total:
                print(f"  [{run_idx}/{total}] generated")

        n_written = n_episodes - n_failed
        print(f"\n  {split_name}: {n_written}/{n_episodes} episodes"
              f" ({n_usable_total:,} usable steps)")
        if n_failed > 0:
            print(f"    ({n_failed} skipped — too few valid detections)")

    # Summary
    train_dir = out_dir / "train"
    total_samples = 0
    if train_dir.exists():
        for p in sorted(train_dir.glob("*.npz")):
            d = np.load(p)
            total_samples += d["valid"].sum()

    print(f"\n  Total training samples: {total_samples:,}")
    print(f"  NN parameters: ~611")
    if total_samples > 0:
        print(f"  Samples/parameter ratio: {total_samples / 611:.0f}:1")
    print(f"  Output: {out_dir.resolve()}")


if __name__ == "__main__":
    print("OPSS Training Episode Generator (logged features)")
    print("=" * 50)
    main()
    print("\nDone.")
