#!/usr/bin/env python3
"""
Profile MultiObjectUKFNN.update() at various track counts.

Generates synthetic multi-drone detections and benchmarks the tracker
to identify scaling bottlenecks (predict, association, update).

Usage:
    python3 profile_pipeline.py
    python3 profile_pipeline.py --tracks 1,5,10,20,50
    python3 profile_pipeline.py --frames 500
"""
import sys
import types
import argparse
import time

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
from opss.state.ukf_nn_tracker import MultiObjectUKFNN, CameraIntrinsics
from opss.state.ukf_nn import config as cfg
from generate_training_data import simulate_drone, resample


def generate_drone_detections(n_drones, n_frames, rng):
    """Generate n_drones worth of detections for n_frames."""
    dt = cfg.DT
    sim_cam = look_at_camera(
        position=np.array([10.0, -5.0, 10.0]),
        target=np.array([10.0, 45.0, 10.0]),
    )
    noise_cfg = ObservationNoise(
        pixel_noise_std=1.5,
        depth_noise_std=0.05,
        depth_dropout_prob=0.02,
    )

    # Generate trajectories for each drone
    all_detections = []  # list of lists: [frame][drone]
    for _ in range(n_frames):
        all_detections.append([])

    for d in range(n_drones):
        drone_rng = np.random.default_rng(rng.integers(0, 2**31))
        history, _ = simulate_drone(drone_rng, duration=n_frames * dt + 1.0, dt=0.001)
        resampled = resample(history, dt)

        for k in range(min(n_frames, len(resampled))):
            pos = resampled[k][1]
            # Offset each drone to prevent overlap
            offset = np.array([d * 3.0, 0.0, 0.0])
            det = world_to_detection(
                pos + offset, sim_cam, noise=noise_cfg,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
            )
            if det is not None:
                all_detections[k].append(det)

    return all_detections, sim_cam


def benchmark_config(n_tracks, n_frames, warmup_frames, rng):
    """Benchmark MultiObjectUKFNN for a given track count."""
    detections, sim_cam = generate_drone_detections(n_tracks, n_frames, rng)

    cam_intrinsics = CameraIntrinsics(
        fx=sim_cam.fx, fy=sim_cam.fy,
        cx=sim_cam.cx, cy=sim_cam.cy,
    )

    tracker = MultiObjectUKFNN(
        max_distance=200.0,
        camera=cam_intrinsics,
        R_world_from_cam=sim_cam.R_world_from_cam,
        t_world_from_cam=sim_cam.t_world_from_cam,
    )

    update_times = []
    predict_times = []
    assoc_times = []
    update_loop_times = []

    for k in range(n_frames):
        t = k * cfg.DT
        dets = detections[k]

        t0 = time.perf_counter()
        tracker.update(dets, t)
        t1 = time.perf_counter()

        if k >= warmup_frames:
            update_times.append((t1 - t0) * 1000)
            timing = tracker.last_timing
            if timing:
                predict_times.append(timing.get('predict_ms', 0))
                assoc_times.append(timing.get('assoc_ms', 0))
                update_loop_times.append(timing.get('update_ms', 0))

    return {
        'n_tracks': n_tracks,
        'update_mean': np.mean(update_times) if update_times else 0,
        'update_p95': np.percentile(update_times, 95) if update_times else 0,
        'predict_ms': np.mean(predict_times) if predict_times else 0,
        'assoc_ms': np.mean(assoc_times) if assoc_times else 0,
        'update_loop_ms': np.mean(update_loop_times) if update_loop_times else 0,
        'actual_tracks': len(tracker.trackers),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile MultiObjectUKFNN")
    parser.add_argument("--tracks", type=str, default="1,5,10,20,50",
                        help="Comma-separated track counts to benchmark")
    parser.add_argument("--frames", type=int, default=300,
                        help="Frames per config (default: 300 = 10s at 30fps)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup frames to skip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    track_counts = [int(x) for x in args.tracks.split(",")]
    rng = np.random.default_rng(args.seed)

    print("OPSS Pipeline Profiling")
    print("=" * 75)
    print(f"Frames per config: {args.frames} ({args.frames / 30:.1f}s at 30fps)")
    print(f"Warmup: {args.warmup} frames\n")

    header = (f"{'n_tracks':>8} | {'update_mean':>11} | {'update_p95':>10} | "
              f"{'predict_ms':>10} | {'assoc_ms':>8} | {'update_ms':>9} | {'actual':>6}")
    print(header)
    print("-" * len(header))

    results = []
    for n in track_counts:
        print(f"  Benchmarking {n} tracks...", end="", flush=True)
        result = benchmark_config(
            n_tracks=n,
            n_frames=args.frames,
            warmup_frames=args.warmup,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )
        results.append(result)
        print(f"\r{result['n_tracks']:>8} | "
              f"{result['update_mean']:>9.3f}ms | "
              f"{result['update_p95']:>8.3f}ms | "
              f"{result['predict_ms']:>8.3f}ms | "
              f"{result['assoc_ms']:>6.3f}ms | "
              f"{result['update_loop_ms']:>7.3f}ms | "
              f"{result['actual_tracks']:>6}")

    # Scaling summary
    if len(results) >= 2:
        base = results[0]['update_mean']
        print(f"\nScaling (relative to {results[0]['n_tracks']} tracks):")
        for r in results:
            ratio = r['update_mean'] / base if base > 0 else 0
            print(f"  {r['n_tracks']:>3} tracks: {ratio:.2f}x")


if __name__ == "__main__":
    main()
