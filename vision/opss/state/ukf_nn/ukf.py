"""
Unscented Kalman Filter — 3D with neural network acceleration correction.

UKF3D: 6D state [x,y,z,vx,vy,vz], 3D measurement [x,y,z].
       Includes gravity in the process model; NN learns only the residual.
       Supports partial observations (e.g. missing depth).
UKF:   Legacy 1D (kept for backward compat).
"""
import numpy as np
from scipy.linalg import cholesky
from . import config as cfg


class UKF3D:
    """
    3D Unscented Kalman Filter with NN acceleration correction.

    State:       x = [x, y, z, vx, vy, vz]^T  (6D)
    Measurement: z = [x, y, z]^T                (3D, may be partial)

    Process model:
        v_next = v + dt * (gravity + delta_a_nn)
        p_next = p + dt * v_next

    The NN learns only the residual after physics (gravity) is subtracted.
    """

    def __init__(self, Q=None, R=None, dt=None, gravity=None, a_hover=None,
                 alpha=None, beta=None, kappa=None):
        self.dt = dt if dt is not None else cfg.DT
        self.Q = Q if Q is not None else cfg.Q.copy()
        self.R = R if R is not None else cfg.R.copy()
        self.gravity = gravity if gravity is not None else cfg.GRAVITY.copy()
        self.a_hover = a_hover if a_hover is not None else cfg.A_HOVER.copy()

        self.alpha = alpha if alpha is not None else cfg.ALPHA
        self.beta = beta if beta is not None else cfg.BETA
        self.kappa = kappa if kappa is not None else cfg.KAPPA

        self.n = cfg.STATE_DIM   # 6
        self.m = cfg.MEAS_DIM    # 3
        self._compute_weights()

    def _compute_weights(self):
        """Compute Merwe scaled sigma point weights."""
        n = self.n
        lam = self.alpha ** 2 * (n + self.kappa) - n
        self.lam = lam
        self.gamma = np.sqrt(n + lam)

        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lam / (n + lam)
        self.Wm[1:] = 1.0 / (2 * (n + lam))

        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)
        self.Wc[1:] = 1.0 / (2 * (n + lam))

    def _generate_sigma_points(self, x, P):
        """Generate 2n+1 sigma points with Cholesky + fallback."""
        n = self.n
        sigma_points = np.zeros((2 * n + 1, n))

        P_sym = (P + P.T) / 2
        P_jittered = P_sym + np.eye(n) * cfg.CHOLESKY_JITTER

        try:
            L = cholesky(P_jittered, lower=True)
        except np.linalg.LinAlgError:
            try:
                P_jittered = P_sym + np.eye(n) * cfg.CHOLESKY_JITTER_RETRY
                L = cholesky(P_jittered, lower=True)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(P_sym)
                eigvals = np.maximum(eigvals, cfg.CHOLESKY_JITTER_RETRY)
                P_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = cholesky(P_fixed, lower=True)

        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + self.gamma * L[:, i]
            sigma_points[n + i + 1] = x - self.gamma * L[:, i]

        return sigma_points

    def _process_model(self, x_sigma, delta_a, a_control=None):
        """
        Propagate a single sigma point.

        Dynamics: constant-velocity + gravity + hover + control + NN residual.
            v_next = v + dt * (gravity + a_hover + a_control + delta_a)
            p_next = p + dt * v_next          (semi-implicit Euler)

        Args:
            x_sigma: (6,) [x, y, z, vx, vy, vz]
            delta_a: (3,) NN acceleration correction (residual after physics)
            a_control: (3,) or None — analytical control acceleration

        Returns:
            x_next: (6,) propagated state
        """
        pos = x_sigma[:3]
        vel = x_sigma[3:]

        accel = self.gravity + self.a_hover + delta_a
        if a_control is not None:
            accel = accel + a_control
        vel_next = vel + self.dt * accel
        pos_next = pos + self.dt * vel_next

        return np.concatenate([pos_next, vel_next])

    def _measurement_model(self, x_sigma):
        """Measurement model: observe position only."""
        return x_sigma[:3].copy()

    def predict(self, x, P, nn_model=None, feature_extractor=None, feature_stats=None,
                a_control=None):
        """
        Prediction step with optional NN correction and analytical control.

        The NN is evaluated once on the mean state (not per sigma point)
        for efficiency and stability.

        Args:
            x: (6,) current state estimate
            P: (6,6) current covariance
            nn_model: optional DeltaAccelNN3D
            feature_extractor: optional FeatureExtractor3D (provides features)
            feature_stats: optional normalization stats dict
            a_control: (3,) or None — analytical control acceleration

        Returns:
            x_pred: (6,) predicted state
            P_pred: (6,6) predicted covariance
            delta_a: (3,) NN acceleration correction used (excludes gravity)
        """
        delta_a = np.zeros(3)
        if nn_model is not None and feature_extractor is not None and feature_stats is not None:
            features = feature_extractor.build_features(x)
            features_norm = feature_extractor.normalize(features, feature_stats)
            delta_a = nn_model.predict_numpy(features_norm)
            delta_a = np.clip(delta_a, -cfg.A_MAX, cfg.A_MAX)

        sigma_points = self._generate_sigma_points(x, P)

        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            sigma_points_pred[i] = self._process_model(sigma_points[i], delta_a, a_control)

        x_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)

        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        P_pred += self.Q
        P_pred = (P_pred + P_pred.T) / 2

        return x_pred, P_pred, delta_a

    def update(self, x_pred, P_pred, z):
        """
        Update step with 3D measurement, supporting partial observations.

        Per-dimension NaN in z indicates a missing measurement for that axis.
        Only observed dimensions participate in the update.  The chi-squared
        threshold is set by DOF = number of observed dims.

        Args:
            x_pred: (6,) predicted state
            P_pred: (6,6) predicted covariance
            z: (3,) measurement [x, y, z] — NaN per element for missing

        Returns:
            x_upd:      (6,) updated state
            P_upd:      (6,6) updated covariance
            innovation: (3,) full innovation (NaN for unobserved dims)
            S:          (3,3) full innovation covariance (NaN rows/cols for unobserved)
        """
        observed = ~np.isnan(z)
        dof = int(np.sum(observed))

        # Nothing observed — skip update entirely
        if dof == 0:
            return x_pred, P_pred, np.full(3, np.nan), np.full((3, 3), np.nan)

        obs_idx = np.where(observed)[0]          # indices of observed dims
        z_obs = z[observed]                       # (dof,)
        R_obs = self.R[np.ix_(obs_idx, obs_idx)]  # (dof, dof)

        sigma_points = self._generate_sigma_points(x_pred, P_pred)
        n_sigma = 2 * self.n + 1

        # Project sigma points through measurement model, select observed dims
        z_sigmas = np.zeros((n_sigma, dof))
        for i in range(n_sigma):
            z_full = self._measurement_model(sigma_points[i])
            z_sigmas[i] = z_full[obs_idx]

        # Predicted measurement mean (dof,)
        z_pred = np.sum(self.Wm[:, np.newaxis] * z_sigmas, axis=0)

        # Innovation (dof,)
        innov_obs = z_obs - z_pred

        # Innovation covariance S_obs (dof x dof)
        S_obs = np.zeros((dof, dof))
        for i in range(n_sigma):
            dz = z_sigmas[i] - z_pred
            S_obs += self.Wc[i] * np.outer(dz, dz)
        S_obs += R_obs

        # Mahalanobis outlier check — consistent with 1D "k-sigma" gate:
        # 1D uses |innov| > k*sqrt(S), i.e. NIS > k².
        # 3D analog: Mahalanobis NIS > k² (DOF-independent threshold).
        try:
            S_obs_inv = np.linalg.inv(S_obs)
            mahal = innov_obs @ S_obs_inv @ innov_obs
            if mahal > cfg.INNOVATION_THRESHOLD_SIGMA ** 2:
                S_obs *= cfg.ADAPTIVE_R_MULTIPLIER
                S_obs_inv = np.linalg.inv(S_obs)
        except np.linalg.LinAlgError:
            # S not invertible — skip update
            innovation_full = np.full(3, np.nan)
            innovation_full[obs_idx] = innov_obs
            return x_pred, P_pred, innovation_full, np.full((3, 3), np.nan)

        # Cross-covariance Pxz (n x dof)
        Pxz = np.zeros((self.n, dof))
        for i in range(n_sigma):
            dx = sigma_points[i] - x_pred
            dz = z_sigmas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain K (n x dof)
        K = Pxz @ S_obs_inv

        # State and covariance update
        x_upd = x_pred + K @ innov_obs
        P_upd = P_pred - K @ S_obs @ K.T
        P_upd = (P_upd + P_upd.T) / 2

        # Pack full-size outputs (NaN for unobserved dims)
        innovation_full = np.full(3, np.nan)
        innovation_full[obs_idx] = innov_obs

        S_full = np.full((3, 3), np.nan)
        S_full[np.ix_(obs_idx, obs_idx)] = S_obs

        return x_upd, P_upd, innovation_full, S_full


class UKF:
    """
    Legacy 1D Unscented Kalman Filter (deprecated).

    State: x = [p, v]^T. Use UKF3D for new code.
    """

    def __init__(self, Q=None, R=None, dt=None, alpha=None, beta=None, kappa=None):
        self.dt = dt if dt is not None else cfg.DT
        q_default = np.array([[1e-6, 0.0], [0.0, 1e-4]])
        r_default = np.array([[0.30]])
        self.Q = Q if Q is not None else q_default
        self.R = R if R is not None else r_default

        self.alpha = alpha if alpha is not None else cfg.ALPHA
        self.beta = beta if beta is not None else cfg.BETA
        self.kappa = kappa if kappa is not None else cfg.KAPPA

        self.n = 2
        self._compute_weights()

    def _compute_weights(self):
        n = self.n
        lam = self.alpha ** 2 * (n + self.kappa) - n
        self.lam = lam
        self.gamma = np.sqrt(n + lam)

        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lam / (n + lam)
        self.Wm[1:] = 1.0 / (2 * (n + lam))

        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)
        self.Wc[1:] = 1.0 / (2 * (n + lam))

    def _generate_sigma_points(self, x, P):
        n = self.n
        sigma_points = np.zeros((2 * n + 1, n))
        P_sym = (P + P.T) / 2
        P_jittered = P_sym + np.eye(n) * cfg.CHOLESKY_JITTER

        try:
            L = cholesky(P_jittered, lower=True)
        except np.linalg.LinAlgError:
            try:
                P_jittered = P_sym + np.eye(n) * cfg.CHOLESKY_JITTER_RETRY
                L = cholesky(P_jittered, lower=True)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(P_sym)
                eigvals = np.maximum(eigvals, cfg.CHOLESKY_JITTER_RETRY)
                P_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = cholesky(P_fixed, lower=True)

        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + self.gamma * L[:, i]
            sigma_points[n + i + 1] = x - self.gamma * L[:, i]

        return sigma_points

    def _process_model(self, x_sigma, u, delta_a):
        p, v = x_sigma
        a = u / cfg.MASS + delta_a
        v_next = v + self.dt * a
        p_next = p + self.dt * v_next
        return np.array([p_next, v_next])

    def _measurement_model(self, x_sigma):
        return x_sigma[0]

    def predict(self, x, P, u, nn_model=None, feature_stats=None):
        delta_a = 0.0
        if nn_model is not None and feature_stats is not None:
            features = np.array([u, x[1], 0.0, 0.0, u])
            mean = np.array(feature_stats['mean'])
            std = np.array(feature_stats['std'])
            features_norm = (features - mean) / std
            delta_a = nn_model.predict_numpy(features_norm)
            delta_a = np.clip(delta_a, -cfg.A_MAX, cfg.A_MAX)

        sigma_points = self._generate_sigma_points(x, P)
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            sigma_points_pred[i] = self._process_model(sigma_points[i], u, delta_a)

        x_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        P_pred += self.Q
        P_pred = (P_pred + P_pred.T) / 2

        return x_pred, P_pred, delta_a

    def update(self, x_pred, P_pred, z):
        if np.isnan(z):
            return x_pred, P_pred, np.nan, np.nan

        sigma_points = self._generate_sigma_points(x_pred, P_pred)
        z_sigma = np.zeros(2 * self.n + 1)
        for i in range(2 * self.n + 1):
            z_sigma[i] = self._measurement_model(sigma_points[i])

        z_pred = np.sum(self.Wm * z_sigma)
        innovation = z - z_pred

        S = 0.0
        for i in range(2 * self.n + 1):
            diff = z_sigma[i] - z_pred
            S += self.Wc[i] * diff * diff
        S += self.R[0, 0]

        if np.abs(innovation) > cfg.INNOVATION_THRESHOLD_SIGMA * np.sqrt(S):
            S *= cfg.ADAPTIVE_R_MULTIPLIER

        Pxz = np.zeros(self.n)
        for i in range(2 * self.n + 1):
            x_diff = sigma_points[i] - x_pred
            z_diff = z_sigma[i] - z_pred
            Pxz += self.Wc[i] * x_diff * z_diff

        K = Pxz / S
        x_upd = x_pred + K * innovation
        P_upd = P_pred - np.outer(K, K) * S
        P_upd = (P_upd + P_upd.T) / 2

        return x_upd, P_upd, innovation, S
