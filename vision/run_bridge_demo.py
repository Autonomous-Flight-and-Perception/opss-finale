#!/usr/bin/env python3
"""
Sim-to-tracker bridge demo.

End-to-end acceptance test:
  1. Generate one drone trajectory (world-frame truth)
  2. Project to synthetic detections at 30 Hz via SimCamera
  3. Run MultiObjectUKFNN with NN disabled
  4. Report: visibility, noise stats, position RMSE vs truth

This confirms the full pipeline (sim → projection → pixel → extrinsics →
world-frame UKF) is coherent before training data is generated.

Usage:
    python run_bridge_demo.py
    python run_bridge_demo.py --seed 123 --duration 10
"""
import argparse
import sys
import types
import numpy as np

# Stub torch before any opss imports — opss.__init__ eagerly imports
# pipeline → vision → torch, which fails on Jetson due to libcudnn mismatch.
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

from opss.sim.camera import look_at_camera
from opss.sim.projection import world_to_detection, world_to_camera
from opss.sim.observation import ObservationNoise
from opss.state.ukf_nn_tracker import MultiObjectUKFNN, CameraIntrinsics

# generate_training_data is a standalone script, import its functions
from generate_training_data import simulate_drone, resample


def run_bridge(
    seed=42,
    duration=30.0,
    pixel_noise_std=1.5,
    depth_noise_std=0.05,
    depth_dropout_prob=0.02,
):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 30.0

    # ---------------------------------------------------------------
    # 1. Generate truth trajectory
    # ---------------------------------------------------------------
    print("Generating trajectory...")
    history, tau = simulate_drone(rng, duration=duration, dt=0.001)
    resampled = resample(history, dt)
    n_steps = len(resampled)

    truth_t = np.array([s[0] for s in resampled])
    truth_x = np.array([s[1] for s in resampled])
    truth_v = np.array([s[2] for s in resampled])
    truth_v_des = np.array([s[5] for s in resampled])

    print(f"  Trajectory: {n_steps} steps, {duration:.0f}s at {1/dt:.0f} Hz")
    print(f"  Position range: x=[{truth_x[:,0].min():.1f}, {truth_x[:,0].max():.1f}]"
          f"  y=[{truth_x[:,1].min():.1f}, {truth_x[:,1].max():.1f}]"
          f"  z=[{truth_x[:,2].min():.1f}, {truth_x[:,2].max():.1f}]")

    # ---------------------------------------------------------------
    # 2. Set up camera
    # ---------------------------------------------------------------
    # Place camera looking into the flight volume.
    # Volume is x∈[0,20], y∈[0,90], z∈[0,20].
    # Camera at the y=-5 edge, centered on x/z, looking along +Y.
    cam_pos = np.array([10.0, -5.0, 10.0])
    cam_target = np.array([10.0, 45.0, 10.0])
    sim_cam = look_at_camera(position=cam_pos, target=cam_target)

    print(f"\n  Camera at ({cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f})"
          f" looking at ({cam_target[0]:.0f}, {cam_target[1]:.0f}, {cam_target[2]:.0f})")

    # Verify camera axes
    R = sim_cam.R_world_from_cam
    print(f"  cam +Z (forward) in world: [{R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f}]")
    print(f"  cam +X (right) in world:   [{R[0,0]:.3f}, {R[1,0]:.3f}, {R[2,0]:.3f}]")
    print(f"  cam +Y (down) in world:    [{R[0,1]:.3f}, {R[1,1]:.3f}, {R[2,1]:.3f}]")

    # Frame-convention sanity check: pick a mid-trajectory point
    mid = n_steps // 2
    p_world_mid = truth_x[mid]
    p_cam_mid = world_to_camera(p_world_mid, sim_cam)
    from opss.state.ukf_nn import config as cfg
    print(f"\n  --- Frame convention check (step {mid}) ---")
    print(f"  truth (world): [{p_world_mid[0]:.2f}, {p_world_mid[1]:.2f}, {p_world_mid[2]:.2f}]")
    print(f"  p_cam (before extrinsics): [{p_cam_mid[0]:.2f}, {p_cam_mid[1]:.2f}, {p_cam_mid[2]:.2f}]")
    p_world_check = sim_cam.R_world_from_cam @ p_cam_mid + sim_cam.t_world_from_cam
    print(f"  p_world (after extrinsics): [{p_world_check[0]:.2f}, {p_world_check[1]:.2f}, {p_world_check[2]:.2f}]")
    print(f"  gravity vector: {cfg.GRAVITY}")
    print(f"  world-z is UP: truth z range = [{truth_x[:,2].min():.1f}, {truth_x[:,2].max():.1f}] "
          f"(drone altitude, should be positive)")
    print(f"  round-trip error: {np.linalg.norm(p_world_check - p_world_mid):.2e}")

    noise = ObservationNoise(
        pixel_noise_std=pixel_noise_std,
        depth_noise_std=depth_noise_std,
        depth_dropout_prob=depth_dropout_prob,
    )

    # ---------------------------------------------------------------
    # 3. Generate detections
    # ---------------------------------------------------------------
    print(f"\nProjecting {n_steps} truth positions to detections...")
    detections = []
    det_noise_rng = np.random.default_rng(seed + 1)
    in_view_count = 0
    depth_dropout_count = 0

    for k in range(n_steps):
        det = world_to_detection(
            truth_x[k], sim_cam, noise=noise, rng=det_noise_rng,
        )
        if det is not None:
            in_view_count += 1
            if det["depth"] <= 0:
                depth_dropout_count += 1
        detections.append(det)

    out_of_view = n_steps - in_view_count
    print(f"  In view: {in_view_count}/{n_steps} ({100*in_view_count/n_steps:.1f}%)")
    print(f"  Out of view: {out_of_view}")
    print(f"  Depth dropout: {depth_dropout_count}")

    # Compute detection depth/pixel stats for visible frames
    visible_depths = []
    pixel_errors_u = []
    pixel_errors_v = []
    for k in range(n_steps):
        if detections[k] is None:
            continue
        det = detections[k]
        # Compare to noiseless projection
        p_cam = world_to_camera(truth_x[k], sim_cam)
        u_true = sim_cam.fx * p_cam[0] / p_cam[2] + sim_cam.cx
        v_true = sim_cam.fy * p_cam[1] / p_cam[2] + sim_cam.cy
        pixel_errors_u.append(det["center"]["x"] - u_true)
        pixel_errors_v.append(det["center"]["y"] - v_true)
        if det["depth"] > 0:
            visible_depths.append(det["depth"])

    if pixel_errors_u:
        pe_u = np.array(pixel_errors_u)
        pe_v = np.array(pixel_errors_v)
        print(f"\n  Pixel noise (u): mean={pe_u.mean():.2f}, std={pe_u.std():.2f} px")
        print(f"  Pixel noise (v): mean={pe_v.mean():.2f}, std={pe_v.std():.2f} px")
    if visible_depths:
        vd = np.array(visible_depths)
        print(f"  Depth range: [{vd.min():.1f}, {vd.max():.1f}] m")

    # ---------------------------------------------------------------
    # 4. Run tracker (NN disabled)
    # ---------------------------------------------------------------
    print("\nRunning tracker (NN disabled)...")

    cam_intrinsics = CameraIntrinsics(
        fx=sim_cam.fx, fy=sim_cam.fy,
        cx=sim_cam.cx, cy=sim_cam.cy,
    )
    tracker = MultiObjectUKFNN(
        max_distance=150.0,
        nn_model=None,         # NN disabled
        feature_stats=None,
        camera=cam_intrinsics,
        R_world_from_cam=sim_cam.R_world_from_cam,
        t_world_from_cam=sim_cam.t_world_from_cam,
    )

    # Feed detections one at a time
    track_estimates = []  # (t, x_est, truth_x) for RMSE
    n_updates = 0
    n_skipped = 0

    for k in range(n_steps):
        t = truth_t[k]
        det = detections[k]

        if det is None:
            # Out of view: predict only with fixed dt (constant frame rate).
            # Using (t - last_update) would double-count gravity across
            # consecutive predict-only steps.
            if tracker.trackers:
                for tr in tracker.trackers.values():
                    tr.predict(dt)
                    tr.mark_missed()
                    tr.last_update = t
            n_skipped += 1
            continue

        states = tracker.update([det], t)
        n_updates += 1

        # If we have a confirmed tracker, record its estimate
        if tracker.trackers:
            # Use the first (only) tracker
            tr = list(tracker.trackers.values())[0]
            est = tr.x[:3].copy()
            track_estimates.append((t, est, truth_x[k].copy()))

    print(f"  Updates: {n_updates}, skipped: {n_skipped}")
    print(f"  Track estimates logged: {len(track_estimates)}")

    # ---------------------------------------------------------------
    # 5. Compute RMSE
    # ---------------------------------------------------------------
    if track_estimates:
        est_arr = np.array([e[1] for e in track_estimates])
        tru_arr = np.array([e[2] for e in track_estimates])
        errors = est_arr - tru_arr
        rmse_3d = np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))
        rmse_per_axis = np.sqrt(np.mean(errors ** 2, axis=0))

        print(f"\n  3D Position RMSE: {rmse_3d:.4f} m")
        print(f"  Per-axis RMSE:    x={rmse_per_axis[0]:.4f}  "
              f"y={rmse_per_axis[1]:.4f}  z={rmse_per_axis[2]:.4f} m")

        # Exclude first N steps (tracker converging)
        n_warmup = min(30, len(track_estimates) // 2)
        if len(track_estimates) > n_warmup:
            errors_ss = errors[n_warmup:]
            rmse_ss = np.sqrt(np.mean(np.sum(errors_ss ** 2, axis=1)))
            rmse_ss_axis = np.sqrt(np.mean(errors_ss ** 2, axis=0))
            print(f"\n  Steady-state RMSE (after {n_warmup} steps): {rmse_ss:.4f} m")
            print(f"  Per-axis:         x={rmse_ss_axis[0]:.4f}  "
                  f"y={rmse_ss_axis[1]:.4f}  z={rmse_ss_axis[2]:.4f} m")
    else:
        print("\n  WARNING: No track estimates recorded!")

    print("\nDone.")
    return rmse_3d if track_estimates else float('inf')


def main():
    parser = argparse.ArgumentParser(description="Sim-to-tracker bridge demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--pixel-noise", type=float, default=1.5)
    parser.add_argument("--depth-noise", type=float, default=0.05)
    parser.add_argument("--depth-dropout", type=float, default=0.02)
    args = parser.parse_args()

    print("OPSS Sim-to-Tracker Bridge Demo")
    print("=" * 50)

    run_bridge(
        seed=args.seed,
        duration=args.duration,
        pixel_noise_std=args.pixel_noise,
        depth_noise_std=args.depth_noise,
        depth_dropout_prob=args.depth_dropout,
    )


if __name__ == "__main__":
    main()
