"""
Feature extraction and normalization utilities for 3D UKF-NN.

Feature vector (15D) -- 5 features per axis i in {x, y, z}:
  1. v_i          -- velocity estimate
  2. r_prev_i     -- previous innovation (signed, from last update)
  3. |r_prev_i|   -- previous innovation magnitude
  4. ||v||         -- speed magnitude (repeated per axis for NN locality)
  5. v_error_i    -- control error: v_des_i - v_est_i (control intent)

All features are available at predict time (no current-innovation terms).
The NN is a process-residual model evaluated inside predict(), so it must
not depend on the current measurement innovation (which only exists after
update).

Normalization is per-column (each of the 15 features gets its own
mean/std).  Because the feature layout is [x-block, y-block, z-block],
this is inherently per-axis aware.
"""
import numpy as np
import json
from pathlib import Path
from . import config as cfg


class FeatureExtractor3D:
    """
    Stateful feature extractor for 3D UKF-NN.

    Tracks previous innovation and desired velocity across calls.
    Call reset() when starting a new track.

    API contract:
        build_features(state)         -- pure, no state mutation
        commit_innovation(innovation) -- updates prev_innovation for next call
        set_v_des(v_des)              -- updates desired velocity for v_error
        normalize(features, stats)    -- pure
    """

    def __init__(self):
        self.prev_innovation = np.zeros(3)
        self.v_des = np.zeros(3)  # desired velocity (set externally)

    def set_v_des(self, v_des):
        """Set desired velocity for v_error feature computation."""
        self.v_des = v_des.copy()

    def build_features(self, state):
        """
        Build the 15D feature vector from current state.

        Uses stored prev_innovation and v_des.
        Does NOT mutate internal state.

        Args:
            state: (6,) array [x, y, z, vx, vy, vz]

        Returns:
            features: (15,) array
        """
        vel = state[3:6]
        speed = np.linalg.norm(vel)
        v_error = self.v_des - vel

        features = np.zeros(cfg.FEAT_DIM)
        for i in range(3):
            base = i * 5
            features[base + 0] = vel[i]
            features[base + 1] = self.prev_innovation[i]
            features[base + 2] = abs(self.prev_innovation[i])
            features[base + 3] = speed
            features[base + 4] = v_error[i]

        return features

    def commit_innovation(self, innovation):
        """
        Store the innovation from the latest update for use in the next
        predict step.  NaN values (from partial observations) are replaced
        with 0.0 to prevent propagation.

        Args:
            innovation: (3,) array (may contain NaN for unobserved dims)
        """
        self.prev_innovation = np.where(
            np.isnan(innovation), 0.0, innovation
        ).copy()

    def normalize(self, features, stats):
        """Apply normalization using precomputed stats."""
        mean = np.array(stats['mean'])
        std = np.array(stats['std'])
        return (features - mean) / std

    def reset(self):
        """Reset state for a new track."""
        self.prev_innovation = np.zeros(3)
        self.v_des = np.zeros(3)


def extract_training_features_3d(positions, velocities, dt=None, a_phys=None,
                                 desired_velocities=None):
    """
    Extract features and targets from a single trajectory for training.

    The training target is the residual acceleration that the NN must learn:

        delta_a = (v[k+1] - v[k]) / dt  -  a_phys

    where a_phys includes known physics (gravity, drag model, etc.).
    If a_phys is None, defaults to cfg.GRAVITY + cfg.A_HOVER so the NN
    only learns unmodeled dynamics.

    Features use only predict-time information (no current innovation):
      [v_i, prev_innov_i, |prev_innov_i|, ||v||, v_error_i] x 3 axes = 15D

    Args:
        positions:  (N, 3) array of [x, y, z]
        velocities: (N, 3) array of [vx, vy, vz]
        dt:         timestep (scalar or None for config default)
        a_phys:     (3,) known physics acceleration to subtract from target.
                    None => cfg.GRAVITY + cfg.A_HOVER.
        desired_velocities: (N, 3) or None. If provided, used for v_error
                    features.  If None, v_error = 0.

    Returns:
        features: (N-2, 15) array  (need k and k-1 for prev_innovation)
        targets:  (N-2, 3)  array of delta_a (NN residual)
    """
    if dt is None:
        dt = cfg.DT
    if a_phys is None:
        a_phys = cfg.GRAVITY + cfg.A_HOVER  # [0,0,0] for hover-compensated

    n = len(positions)
    if n < 3:
        return np.zeros((0, cfg.FEAT_DIM)), np.zeros((0, cfg.ACCEL_DIM))

    # Total observed acceleration
    a_total = (velocities[1:] - velocities[:-1]) / dt  # (N-1, 3)

    # Residual = total - known physics
    delta_a = a_total - a_phys[np.newaxis, :]
    delta_a = np.clip(delta_a, -cfg.A_MAX, cfg.A_MAX)

    # Simulate innovations: position prediction error under CV + physics
    predicted_pos = positions[:-1] + dt * velocities[:-1] + 0.5 * a_phys * dt ** 2
    innovations = positions[1:] - predicted_pos  # (N-1, 3)

    # Build features for k = 1..N-2 (need prev_innovation from k-1)
    n_samples = n - 2
    features = np.zeros((n_samples, cfg.FEAT_DIM))

    for k in range(n_samples):
        vel = velocities[k + 1]
        prev_innov = innovations[k]
        speed = np.linalg.norm(vel)

        # v_error = v_des - v_est (control intent)
        # When desired_velocities is None, default v_des=0 => v_error = -vel
        # (matches runtime FeatureExtractor3D default)
        if desired_velocities is not None:
            v_error = desired_velocities[k + 1] - vel
        else:
            v_error = -vel

        for i in range(3):
            base = i * 5
            features[k, base + 0] = vel[i]
            features[k, base + 1] = prev_innov[i]
            features[k, base + 2] = abs(prev_innov[i])
            features[k, base + 3] = speed
            features[k, base + 4] = v_error[i]

    # Targets aligned: delta_a[k+1] for k = 0..N-3
    targets = delta_a[1:]  # (N-2, 3)

    return features, targets


def compute_normalization_stats(features_list):
    """
    Compute per-column mean and std for feature normalization.

    Each of the 15 feature columns is normalized independently.
    Because the layout is [x_block(5), y_block(5), z_block(5)],
    axes with different scales get independent statistics.

    Args:
        features_list: List of feature arrays from multiple trajectories

    Returns:
        dict with 'mean' and 'std' arrays (as lists for JSON serialization)
    """
    all_features = np.vstack(features_list)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    std = np.maximum(std, 1e-8)

    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'feat_dim': cfg.FEAT_DIM,
        'nn_output_dim': cfg.NN_OUTPUT_DIM,
        'version': 'feat15_3d_v2',
    }


def normalize_features(features, stats):
    """Apply normalization using precomputed stats."""
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    return (features - mean) / std


def save_stats(stats, path):
    """Save normalization stats to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)


def load_stats(path):
    """Load normalization stats from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
