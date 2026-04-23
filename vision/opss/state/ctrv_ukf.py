"""
Single-object Unscented Kalman Filter with the Constant Turn Rate / Velocity
(CTRV) nonlinear motion model.

Ported from MonteCarlo/filter-test/src/filters/ukf_ctrv_xy.py
(https://github.com/Autonomous-Flight-and-Perception/MonteCarlo). Original
file is a one-shot batch runner; this module exposes ``predict`` + ``update``
as separate calls so the multi-object tracker wrapper can drive it tick by tick.

State (5-D):
    x = [px, py, v, psi, omega]
        px, py  : pixel-space position (capture resolution)
        v       : speed magnitude (px / s)
        psi     : heading angle (rad)
        omega   : yaw rate (rad / s)

Measurement (2-D):
    z = [px, py]    (detection center, pixel space)

Sigma-point convention is Julier with kappa = 3 - N => (N+kappa) = 3, matching
the upstream MonteCarlo implementation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------- helpers ----------

def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def ctrv_fx(x: np.ndarray, dt: float, eps_omega: float = 1e-6) -> np.ndarray:
    """Nonlinear coordinated-turn state transition."""
    px, py, v, psi, omega = np.asarray(x, dtype=float)
    if abs(omega) > eps_omega:
        px_new = px + (v / omega) * (np.sin(psi + omega * dt) - np.sin(psi))
        py_new = py + (v / omega) * (-np.cos(psi + omega * dt) + np.cos(psi))
    else:
        px_new = px + v * np.cos(psi) * dt
        py_new = py + v * np.sin(psi) * dt
    psi_new = wrap_angle(psi + omega * dt)
    return np.array([px_new, py_new, v, psi_new, omega], dtype=float)


def hx_pos(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[:2], dtype=float)


def make_R(sigma_z: float) -> np.ndarray:
    return (sigma_z ** 2) * np.eye(2, dtype=float)


def make_Q_ctrv(dt: float, sigma_a: float, sigma_omega: float) -> np.ndarray:
    """Diagonal process noise (heuristic; matches upstream)."""
    q_pos   = 0.25 * (dt ** 4) * (sigma_a ** 2) + 1e-6
    q_v     = (dt ** 2) * (sigma_a ** 2)
    q_psi   = 0.25 * (dt ** 4) * (sigma_omega ** 2) + 1e-6
    q_omega = (dt ** 2) * (sigma_omega ** 2) + 1e-6
    return np.diag([q_pos, q_pos, q_v, q_psi, q_omega]).astype(float)


def _weights_julier(N: int) -> Tuple[np.ndarray, np.ndarray, float]:
    Kappa = 3 - N
    NplusKappa = N + Kappa
    if NplusKappa <= 0:
        raise ValueError("N + Kappa must be > 0 for sigma points.")
    w0 = Kappa / NplusKappa
    wi = 1.0 / (2.0 * NplusKappa)
    weights = np.full(2 * N + 1, wi, dtype=float)
    weights[0] = w0
    return weights, np.diag(weights), float(NplusKappa)


def _make_sigma_points(state: np.ndarray, P: np.ndarray, NplusKappa: float) -> np.ndarray:
    N = state.shape[0]
    # Cholesky requires PSD; protect against numerical drift.
    P_safe = (P + P.T) * 0.5 + np.eye(N) * 1e-9
    L = np.linalg.cholesky(NplusKappa * P_safe)
    pts = np.zeros((2 * N + 1, N), dtype=float)
    pts[0] = state
    for i in range(N):
        pts[i + 1]     = state + L[:, i]
        pts[N + i + 1] = state - L[:, i]
    return pts


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return np.arctan2(s, c)


def _state_mean(sigma_pts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = sigma_pts.T @ weights
    mean[3] = wrap_angle(_circular_mean(sigma_pts[:, 3], weights))
    return mean


def _subtract_state_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = A - b
    out[:, 3] = (out[:, 3] + np.pi) % (2.0 * np.pi) - np.pi
    return out


# ---------- single-object online UKF ----------

class CTRVUKF:
    """
    Online CTRV UKF for one tracked object. Stateless between predict + update
    only insofar as you must call them in order each tick.

    Default sigmas are tuned for **pixel-space** capture-resolution detections
    (640×480) at 30 Hz: a drone moving 200 px/s, turning ~1 rad/s. Tweak via
    the constructor.
    """

    N = 5

    def __init__(
        self,
        sigma_a: float = 250.0,        # px / s^2 process accel std
        sigma_omega: float = 1.5,      # rad / s^2 yaw-accel std
        sigma_z: float = 8.0,          # px measurement std
        P0: Optional[np.ndarray] = None,
    ):
        self.sigma_a = float(sigma_a)
        self.sigma_omega = float(sigma_omega)
        self.sigma_z = float(sigma_z)
        self.R = make_R(sigma_z)

        self.weights, self.weights_diag, self.NplusKappa = _weights_julier(self.N)

        if P0 is None:
            P0 = np.diag([
                sigma_z ** 2,            # px var
                sigma_z ** 2,            # py var
                (50.0) ** 2,             # v var (px/s)
                (np.deg2rad(45.0)) ** 2, # psi var
                (1.0) ** 2,              # omega var
            ]).astype(float)
        self.P = P0.copy()
        self.x = np.zeros(self.N, dtype=float)
        self._initialized = False

    def init_from_measurement(self, z_xy: np.ndarray) -> None:
        z = np.asarray(z_xy, dtype=float)
        self.x = np.array([z[0], z[1], 0.0, 0.0, 0.0], dtype=float)
        self._initialized = True

    def predict(self, dt: float) -> None:
        Q = make_Q_ctrv(dt, self.sigma_a, self.sigma_omega)
        sigma = _make_sigma_points(self.x, self.P, self.NplusKappa)
        sigma_pred = np.array([ctrv_fx(sp, dt) for sp in sigma], dtype=float)
        self.x = _state_mean(sigma_pred, self.weights)
        Xminus = _subtract_state_rows(sigma_pred, self.x)
        self.P = (Xminus.T @ self.weights_diag @ Xminus) + Q
        self._sigma_pred = sigma_pred
        self._Xminus = Xminus

    def update(self, z_xy: np.ndarray) -> float:
        """Returns innovation magnitude (px) — useful for the adaptive selector."""
        Zmatrix = np.array([hx_pos(sp) for sp in self._sigma_pred], dtype=float)
        zvector = Zmatrix.T @ self.weights
        Zminus = Zmatrix - zvector
        PZ = (Zminus.T @ self.weights_diag @ Zminus) + self.R
        PXZ = self._Xminus.T @ self.weights_diag @ Zminus
        K = PXZ @ np.linalg.inv(PZ)
        innovation = np.asarray(z_xy, dtype=float) - zvector
        self.x = self.x + K @ innovation
        self.x[3] = wrap_angle(self.x[3])
        self.P = self.P - K @ PZ @ K.T
        return float(np.linalg.norm(innovation))

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity_xy(self) -> Tuple[float, float]:
        v, psi = float(self.x[2]), float(self.x[3])
        return v * np.cos(psi), v * np.sin(psi)

    @property
    def speed(self) -> float:
        return float(self.x[2])

    @property
    def heading(self) -> float:
        return float(self.x[3])

    @property
    def yaw_rate(self) -> float:
        return float(self.x[4])
