"""
Metrics computation for 3D filter evaluation.

Provides both 3D and legacy 1D metric functions.
"""
import numpy as np
from . import config as cfg


# ---------------------------------------------------------------------------
# 3D metrics
# ---------------------------------------------------------------------------

def compute_rmse_3d(estimates, truth):
    """
    3D position or velocity RMSE.

    Args:
        estimates: (N, 3) array
        truth:     (N, 3) array

    Returns:
        scalar RMSE (Euclidean distance averaged over timesteps)
    """
    errors = np.linalg.norm(estimates - truth, axis=1)
    return np.sqrt(np.mean(errors ** 2))


def compute_rmse_per_axis(estimates, truth):
    """
    Per-axis RMSE breakdown.

    Returns:
        (3,) array of RMSE values [rmse_x, rmse_y, rmse_z]
    """
    return np.sqrt(np.mean((estimates - truth) ** 2, axis=0))


def compute_nis_3d(innovations, S_values):
    """
    Normalized Innovation Squared for 3D measurements (Mahalanobis).

    Args:
        innovations: (N, 3) array
        S_values:    (N, 3, 3) array of innovation covariances

    Returns:
        nis_values:   (N,) array
        inbound_rate: fraction within chi-squared(3, 0.95) = 7.815
    """
    n = len(innovations)
    nis_values = np.zeros(n)
    for k in range(n):
        try:
            S_inv = np.linalg.inv(S_values[k])
            nis_values[k] = innovations[k] @ S_inv @ innovations[k]
        except np.linalg.LinAlgError:
            nis_values[k] = np.nan

    valid = ~np.isnan(nis_values)
    if np.sum(valid) > 0:
        inbound_rate = np.mean(nis_values[valid] < cfg.CHI2_95)
    else:
        inbound_rate = np.nan

    return nis_values, inbound_rate


def evaluate_run_3d(pos_est, vel_est, pos_true, vel_true, innovations, S_values):
    """
    Evaluate a single 3D run.

    Args:
        pos_est:     (N, 3) estimated positions
        vel_est:     (N, 3) estimated velocities
        pos_true:    (N, 3) ground-truth positions
        vel_true:    (N, 3) ground-truth velocities
        innovations: (M, 3) measurement innovations (may have NaN rows)
        S_values:    (M, 3, 3) innovation covariances

    Returns:
        dict with rmse_pos, rmse_vel, rmse_per_axis_pos, nis_inbound_rate, mean_nis
    """
    rmse_pos = compute_rmse_3d(pos_est, pos_true)
    rmse_vel = compute_rmse_3d(vel_est, vel_true)
    rmse_axis = compute_rmse_per_axis(pos_est, pos_true)

    # Filter out NaN innovations
    valid = ~np.any(np.isnan(innovations), axis=1)
    if np.sum(valid) > 0:
        nis_values, inbound_rate = compute_nis_3d(
            innovations[valid], S_values[valid]
        )
        mean_nis = np.nanmean(nis_values)
    else:
        inbound_rate = np.nan
        mean_nis = np.nan

    return {
        'rmse_pos': rmse_pos,
        'rmse_vel': rmse_vel,
        'rmse_per_axis_pos': rmse_axis.tolist(),
        'nis_inbound_rate': inbound_rate,
        'mean_nis': mean_nis,
    }


def check_acceptance_criteria_3d(metrics):
    """
    Check if 3D metrics meet acceptance criteria.

    Returns:
        bool — True if all criteria pass
    """
    passed = True

    if metrics['rmse_pos'] > cfg.RMSE_POS_THRESHOLD:
        passed = False
    if metrics['rmse_vel'] > cfg.RMSE_VEL_THRESHOLD:
        passed = False
    if not np.isnan(metrics['nis_inbound_rate']) and metrics['nis_inbound_rate'] < cfg.NIS_INBOUND_THRESHOLD:
        passed = False

    return passed


# ---------------------------------------------------------------------------
# Legacy 1D metrics (kept for backward compat)
# ---------------------------------------------------------------------------

def compute_rmse(estimates, truth):
    """Compute 1D RMSE."""
    return np.sqrt(np.mean((estimates - truth) ** 2))


def compute_nis(innovations, S_values):
    """1D NIS."""
    S_safe = np.maximum(S_values, 1e-10)
    nis_values = innovations ** 2 / S_safe
    inbound_rate = np.mean(nis_values < cfg.CHI2_95)
    return nis_values, inbound_rate


def evaluate_run(p_est, v_est, p_true, v_true, innovations, S_values):
    """Evaluate a single 1D run."""
    rmse_p = compute_rmse(p_est, p_true)
    rmse_v = compute_rmse(v_est, v_true)

    valid_mask = ~np.isnan(innovations) & ~np.isnan(S_values)
    if np.sum(valid_mask) > 0:
        nis_values, inbound_rate = compute_nis(
            innovations[valid_mask], S_values[valid_mask]
        )
        mean_nis = np.mean(nis_values)
    else:
        inbound_rate = np.nan
        mean_nis = np.nan

    return {
        'rmse_p': rmse_p,
        'rmse_v': rmse_v,
        'nis_inbound_rate': inbound_rate,
        'mean_nis': mean_nis,
    }


def aggregate_metrics(metrics_list):
    """Aggregate metrics across multiple runs."""
    rmse_p_vals = np.array([m.get('rmse_p', m.get('rmse_pos', 0)) for m in metrics_list])
    rmse_v_vals = np.array([m.get('rmse_v', m.get('rmse_vel', 0)) for m in metrics_list])
    nis_rates = np.array([m['nis_inbound_rate'] for m in metrics_list])
    mean_nis_vals = np.array([m['mean_nis'] for m in metrics_list])

    return {
        'rmse_p_mean': np.nanmean(rmse_p_vals),
        'rmse_p_std': np.nanstd(rmse_p_vals),
        'rmse_v_mean': np.nanmean(rmse_v_vals),
        'rmse_v_std': np.nanstd(rmse_v_vals),
        'nis_inbound_rate_mean': np.nanmean(nis_rates),
        'nis_inbound_rate_std': np.nanstd(nis_rates),
        'mean_nis_mean': np.nanmean(mean_nis_vals),
        'mean_nis_std': np.nanstd(mean_nis_vals),
    }


def print_metrics(metrics, label="Metrics"):
    """Pretty print metrics."""
    print(f"\n{label}:")
    print(f"  RMSE(p): {metrics['rmse_p_mean']:.4f} +/- {metrics['rmse_p_std']:.4f} m")
    print(f"  RMSE(v): {metrics['rmse_v_mean']:.4f} +/- {metrics['rmse_v_std']:.4f} m/s")
    print(f"  NIS inbound: {metrics['nis_inbound_rate_mean']:.2%} +/- {metrics['nis_inbound_rate_std']:.2%}")
    print(f"  Mean NIS: {metrics['mean_nis_mean']:.3f} +/- {metrics['mean_nis_std']:.3f}")


def check_acceptance_criteria(metrics):
    """Check 1D acceptance criteria."""
    passed = True
    if metrics['rmse_p_mean'] > cfg.RMSE_P_THRESHOLD:
        passed = False
    if metrics['rmse_v_mean'] > cfg.RMSE_V_THRESHOLD:
        passed = False
    if metrics['nis_inbound_rate_mean'] < cfg.NIS_INBOUND_THRESHOLD:
        passed = False
    return passed
