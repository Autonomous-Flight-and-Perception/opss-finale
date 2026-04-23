#!/usr/bin/env python3
"""
Generate 3D drone flight training data for UKF-NN.

Simulates realistic drone trajectories with:
  - Waypoint-based navigation with smooth transitions
  - Realistic speeds (0-8 m/s cruise, up to 12 m/s max)
  - Hover segments, turns, altitude changes
  - Wind perturbation (Ornstein-Uhlenbeck process)
  - Bounded flight volume

Outputs:
  - Single Blender-compatible CSV (120 Hz) for visual validation
  - Training NPZ files (30 Hz) for UKF-NN training

Usage:
  python generate_training_data.py
  python generate_training_data.py --n_train 256 --n_val 32 --n_test 32
"""
import sys
import argparse
import csv
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Drone flight model
# ---------------------------------------------------------------------------

def generate_waypoints(rng, n_waypoints=8, bounds=None):
    """Generate random waypoints in a bounded flight volume."""
    if bounds is None:
        bounds = {
            "x": (0, 20),
            "y": (0, 90),
            "z": (0, 20),
        }
    waypoints = np.zeros((n_waypoints, 3))
    for i in range(n_waypoints):
        waypoints[i, 0] = rng.uniform(*bounds["x"])
        waypoints[i, 1] = rng.uniform(*bounds["y"])
        waypoints[i, 2] = rng.uniform(*bounds["z"])
    return waypoints


def simulate_drone(rng, duration=30.0, dt=0.001, C_d_override=None,
                   tau_override=None, wind_tau_override=None,
                   wind_strength_override=None):
    """
    Simulate a drone flying between waypoints with realistic dynamics.

    Model:
      - Velocity tracks desired velocity with first-order lag (tau ~ 0.8-1.5s)
      - Desired velocity points toward next waypoint, magnitude = cruise speed
      - Drone slows near waypoints, hovers briefly, then heads to next
      - Aerodynamic drag: a_drag = -C_d * ||v|| * v (quadratic, opposing motion)
      - Wind: Ornstein-Uhlenbeck process affecting both velocity and position

    Optional overrides allow testing with out-of-distribution parameters.

    Returns: (history, tau) where history is a list of
             (t, pos, vel, accel, quat, desired_vel) tuples at dt intervals.
    """
    # Flight parameters
    cruise_speed = rng.uniform(2, 8)       # m/s
    max_speed = rng.uniform(8, 12)         # m/s
    tau = tau_override if tau_override is not None else rng.uniform(0.8, 1.5)
    hover_time = rng.uniform(0.5, 2.0)     # pause at each waypoint
    waypoint_radius = rng.uniform(1.0, 2.5)  # "arrived" threshold

    # Aerodynamic drag coefficient (quadratic drag: a = -C_d * ||v|| * v)
    # C_d ~ 0.02-0.08 for a small drone; gives drag ~0.5-4 m/s² at 8 m/s
    C_d = C_d_override if C_d_override is not None else rng.uniform(0.02, 0.08)

    # Wind (Ornstein-Uhlenbeck process)
    wind_strength = wind_strength_override if wind_strength_override is not None else rng.uniform(0, 2.5)
    wind_tau = wind_tau_override if wind_tau_override is not None else rng.uniform(2, 8)
    wind_mean = rng.uniform(-1, 1, size=3) * 0.5  # slight prevailing wind
    wind_vel_coupling = rng.uniform(0.05, 0.20)  # how much wind accelerates the drone

    # Generate waypoints
    n_wp = rng.integers(5, 12)
    waypoints = generate_waypoints(rng, n_wp)

    # Start near first waypoint
    pos = waypoints[0].copy() + rng.normal(0, 0.5, size=3)
    pos[0] = np.clip(pos[0], 0, 20)
    pos[1] = np.clip(pos[1], 0, 90)
    pos[2] = np.clip(pos[2], 1.5, 20)
    vel = np.zeros(3)
    wind = wind_mean.copy()

    # Heading state (smooth yaw tracking)
    to_first = waypoints[1] - pos
    heading = np.arctan2(to_first[0], to_first[1])  # initial yaw toward first target
    heading_tau = rng.uniform(0.3, 0.8)  # yaw response time (seconds)

    wp_idx = 1
    hover_remaining = 0.0

    history = []
    t = 0.0
    prev_vel = vel.copy()

    while t < duration:
        # Wind update (OU process)
        dw = -(wind - wind_mean) / wind_tau * dt + wind_strength * np.sqrt(2 * dt / wind_tau) * rng.normal(size=3)
        wind += dw

        # Target waypoint
        target = waypoints[wp_idx % len(waypoints)]
        to_target = target - pos
        dist = np.linalg.norm(to_target)

        if hover_remaining > 0:
            # Hovering at waypoint
            desired_vel = np.zeros(3)
            hover_remaining -= dt
            if hover_remaining <= 0:
                wp_idx += 1
        elif dist < waypoint_radius:
            # Arrived at waypoint, start hover
            hover_remaining = hover_time
            desired_vel = np.zeros(3)
        else:
            # Fly toward waypoint
            direction = to_target / dist
            # Slow down when approaching
            approach_speed = min(cruise_speed, dist / 2.0)
            approach_speed = min(approach_speed, max_speed)
            desired_vel = direction * approach_speed

        # First-order velocity dynamics (drone response lag)
        vel_error = desired_vel - vel
        vel += vel_error * (dt / tau)

        # Aerodynamic drag (quadratic, opposing motion)
        speed = np.linalg.norm(vel)
        if speed > 1e-6:
            a_drag = -C_d * speed * vel
            vel += a_drag * dt

        # Wind effect on velocity (wind pushes the drone)
        vel += wind * wind_vel_coupling * dt

        # Clamp speed
        speed = np.linalg.norm(vel)
        if speed > max_speed:
            vel *= max_speed / speed

        # Add wind effect on position (drone drifts with wind)
        effective_vel = vel + wind * 0.3  # partial wind coupling on position

        # Integrate position
        pos = pos + effective_vel * dt

        # Keep within flight volume
        pos[0] = np.clip(pos[0], 0, 20)
        pos[1] = np.clip(pos[1], 0, 90)
        pos[2] = np.clip(pos[2], 0.5, 20)

        # Smooth heading: yaw follows velocity direction with lag
        h_speed = np.linalg.norm(vel[:2])
        if h_speed > 0.3:  # only update heading when actually moving
            desired_heading = np.arctan2(vel[0], vel[1])
            # Shortest-path angle difference
            d_heading = desired_heading - heading
            d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
            heading += d_heading * (dt / heading_tau)
            heading = (heading + np.pi) % (2 * np.pi) - np.pi  # normalize

        # Damped pitch from vertical velocity
        pitch = np.arctan2(-vel[2], max(h_speed, 0.5)) * 0.3

        # Quaternion from smoothed heading + pitch
        cy, sy = np.cos(heading / 2), np.sin(heading / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        quat = (cy * sp, sy * sp, sy * cp, cy * cp)  # (qx, qy, qz, qw)

        # Compute acceleration from velocity change
        accel = (vel - prev_vel) / dt if t > 0 else np.zeros(3)
        prev_vel = vel.copy()

        history.append((round(t, 6), pos.copy(), vel.copy(), accel.copy(), quat, desired_vel.copy()))
        t += dt

    return history, tau


def resample(history, dt_out):
    """Resample high-res trajectory to uniform dt_out intervals.

    Handles both 5-element (legacy) and 6-element (with desired_vel) tuples.
    Output tuples always have 6 elements: (t, pos, vel, accel, quat, desired_vel).
    """
    if not history:
        return []

    has_v_des = len(history[0]) >= 6

    times = np.array([s[0] for s in history])
    positions = np.array([s[1] for s in history])
    velocities = np.array([s[2] for s in history])
    accels = np.array([s[3] for s in history])
    quats = np.array([s[4] for s in history])
    if has_v_des:
        desired_vels = np.array([s[5] for s in history])
    else:
        desired_vels = np.zeros_like(velocities)

    t_end = times[-1]
    out = []
    t = 0.0
    while t <= t_end + 1e-9:
        idx = np.searchsorted(times, t)
        idx = min(idx, len(times) - 1)
        if idx > 0 and abs(times[idx - 1] - t) < abs(times[idx] - t):
            idx = idx - 1

        out.append((
            round(t, 6),
            positions[idx].copy(),
            velocities[idx].copy(),
            accels[idx].copy(),
            quats[idx].copy(),
            desired_vels[idx].copy(),
        ))
        t += dt_out

    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_combined_csv(path, all_resampled, dt_csv):
    """Write single Blender-compatible CSV with all trajectories concatenated smoothly."""
    path.parent.mkdir(parents=True, exist_ok=True)

    hz = round(1.0 / dt_csv)
    frame = 0

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Time (Seconds)", "QX", "QY", "QZ", "QW", "X", "Y", "Z"])

        for traj_idx, resampled in enumerate(all_resampled):
            # Trajectory frames (quaternion comes from simulation, already smooth)
            for t, pos, vel, _acc, q, _vdes in resampled:
                writer.writerow([
                    frame, f"{frame / hz:.6f}",
                    f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}",
                    f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}",
                ])
                frame += 1

    return frame


def write_training_npz(path, resampled, params):
    """Write training NPZ: t(N,), x(N,3), v(N,3), a(N,3), v_des(N,3), params."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([s[0] for s in resampled])
    x = np.array([s[1] for s in resampled])
    v = np.array([s[2] for s in resampled])
    a = np.array([s[3] for s in resampled])
    v_des = np.array([s[5] for s in resampled])
    np.savez(path, t=t, x=x, v=v, a=a, v_des=v_des, params=params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 3D drone training data")
    parser.add_argument("--n_train", type=int, default=256, help="Training trajectories")
    parser.add_argument("--n_val", type=int, default=32, help="Validation trajectories")
    parser.add_argument("--n_test", type=int, default=32, help="Test trajectories")
    parser.add_argument("--output", type=str, default="data/generated", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dt_csv", type=float, default=0.008333, help="Blender CSV timestep (120 Hz)")
    parser.add_argument("--dt_train", type=float, default=0.0333, help="Training timestep (~30 Hz)")
    parser.add_argument("--duration", type=float, default=30.0, help="Trajectory duration (seconds)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    rng = np.random.default_rng(args.seed)

    splits = [
        ("train", args.n_train),
        ("val", args.n_val),
        ("test", args.n_test),
    ]

    total = sum(n for _, n in splits)
    run_idx = 0

    all_csv_data = []  # for combined CSV

    for split_name, n_runs in splits:
        npz_dir = out_dir / "training" / split_name
        split_csv_data = []

        for i in range(n_runs):
            # Simulate drone flight
            history, tau = simulate_drone(rng, duration=args.duration, dt=0.001)

            # Resample for Blender CSV (120 Hz)
            csv_data = resample(history, args.dt_csv)
            split_csv_data.append(csv_data)

            # Resample for training (30 Hz)
            train_data = resample(history, args.dt_train)
            npz_path = npz_dir / f"run_{i:03d}.npz"

            params = {"duration": args.duration, "split": split_name, "index": i, "tau": float(tau)}
            write_training_npz(npz_path, train_data, params)

            run_idx += 1
            if run_idx % 20 == 0 or run_idx == total:
                print(f"  [{run_idx}/{total}] generated")

        all_csv_data.extend(split_csv_data)

        # Per-split stats
        samples = [len(resample(simulate_drone(np.random.default_rng(), duration=1), args.dt_train))
                    for _ in []]  # skip recount
        print(f"\n  {split_name}: {n_runs} trajectories ({args.duration}s each)")
        print(f"    NPZ dir: {npz_dir}")

    # Write Blender CSV (first trajectory only, keeps file size manageable)
    csv_path = out_dir / "training_trajectories.csv"
    total_frames = write_combined_csv(csv_path, [all_csv_data[0]], args.dt_csv)
    print(f"\n  Blender CSV: {csv_path}")
    print(f"  Total frames: {total_frames:,} ({total_frames / 120:.1f}s at 120 Hz)")

    # Training stats
    total_train_samples = 0
    train_dir = out_dir / "training" / "train"
    if train_dir.exists():
        for npz_path in sorted(train_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            total_train_samples += len(d["t"])

    print(f"\n  Total training samples: {total_train_samples:,}")
    print(f"  NN parameters: ~611")
    if total_train_samples > 0:
        print(f"  Samples/parameter ratio: {total_train_samples / 611:.0f}:1")
    print(f"\n  Output: {out_dir.resolve()}")


if __name__ == "__main__":
    print("OPSS Drone Training Data Generator")
    print("Waypoint navigation + wind perturbation\n")
    main()
    print("\nDone.")
