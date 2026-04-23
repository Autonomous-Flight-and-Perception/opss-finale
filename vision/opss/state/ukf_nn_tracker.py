"""
UKF-NN Multi-Object Tracker

Wraps UKF3D + DeltaAccelNN3D for multi-object tracking in OPSS.
Provides the same interface as the basic Kalman tracker but with
improved accuracy from neural network correction.

MEASUREMENT CONVENTION:
    Detections arrive as pixel (x,y) + depth (meters).
    This tracker converts to meters using a pinhole camera model,
    then applies camera-to-world extrinsics (rotation + translation)
    before feeding to the UKF.  All UKF state is in world-frame meters
    (z-up, gravity along -z).

FRAME CONVENTIONS:
    Camera frame: +X right, +Y down, +Z forward (standard pinhole).
    World frame:  z-up, gravity = [0, 0, -9.81] m/s².

    The extrinsic transform maps camera→world:
        p_world = R_world_from_cam @ p_cam + t_world_from_cam

    Default extrinsics are identity (R=I, t=0), which is equivalent to
    the previous behavior where camera frame == world frame.
"""
import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from scipy.optimize import linear_sum_assignment

from .kalman import ObjectState
from .ukf_nn import UKF3D, DeltaAccelNN3D, FeatureExtractor3D
from .ukf_nn import Q, R, A_MAX
from .ukf_nn import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Pinhole camera parameters for pixel-to-meter conversion."""
    fx: float = cfg.CAMERA_FX
    fy: float = cfg.CAMERA_FY
    cx: float = cfg.CAMERA_CX
    cy: float = cfg.CAMERA_CY

    def pixel_to_meters(self, u_px, v_px, depth_m):
        """
        Convert pixel coordinates + depth to 3D meters (camera frame).

        Args:
            u_px: pixel x coordinate
            v_px: pixel y coordinate
            depth_m: depth in meters (from stereo/lidar/depth camera)

        Returns:
            (x_m, y_m, z_m) in meters
        """
        x_m = (u_px - self.cx) * depth_m / self.fx
        y_m = (v_px - self.cy) * depth_m / self.fy
        z_m = depth_m
        return x_m, y_m, z_m

    def meters_to_pixel(self, x_m, y_m, z_m):
        """Inverse: 3D meters to pixel coordinates."""
        if abs(z_m) < 1e-6:
            return self.cx, self.cy
        u_px = x_m * self.fx / z_m + self.cx
        v_px = y_m * self.fy / z_m + self.cy
        return u_px, v_px


class UKFNNTracker:
    """
    Single object UKF-NN tracker.

    Delegates all UKF math to UKF3D. Uses FeatureExtractor3D for
    12D feature building.

    State vector: [x, y, z, vx, vy, vz] (6D, all meters)

    Callers MUST ensure the initial detection has valid depth (> 0)
    before constructing.  Use _has_valid_depth() to check.
    """

    def __init__(
        self,
        track_id: int,
        initial_detection: Dict,
        timestamp: float,
        nn_model: Optional[DeltaAccelNN3D] = None,
        feature_stats: Optional[Dict] = None,
        camera: Optional[CameraIntrinsics] = None,
        R_world_from_cam: Optional[np.ndarray] = None,
        t_world_from_cam: Optional[np.ndarray] = None,
        Q_override: Optional[np.ndarray] = None,
        R_override: Optional[np.ndarray] = None,
        P0_override: Optional[np.ndarray] = None,
    ):
        self.track_id = track_id
        self.last_update = timestamp
        self.age = 0
        self.hits = 1
        self.misses = 0
        self.camera = camera or CameraIntrinsics()
        self.R_world_from_cam = R_world_from_cam if R_world_from_cam is not None else np.eye(3)
        self.t_world_from_cam = t_world_from_cam if t_world_from_cam is not None else np.zeros(3)

        # Cache the frame label emitted by get_state() so downstream code
        # (validator / fusion / /states API / annotation drawer) can branch
        # on whether world-frame gravity physics applies here.
        #
        # Identity extrinsics  -> world == camera; the world-frame gravity
        #                         vector (-Z) is not physically aligned;
        #                         predict model degenerates to CV + NN.
        # Non-identity         -> world-frame is distinct; gravity applies.
        _is_identity_extrinsics = (
            np.allclose(self.R_world_from_cam, np.eye(3), atol=1e-12)
            and np.allclose(self.t_world_from_cam, 0.0, atol=1e-12)
        )
        self._frame_tag = "camera_metric" if _is_identity_extrinsics else "world_metric"

        # Convert initial detection from pixels to meters
        x_m, y_m, z_m = self._convert_detection(initial_detection)
        self.x = np.array([x_m, y_m, z_m, 0.0, 0.0, 0.0], dtype=np.float64)

        # Initial covariance (meters)
        if P0_override is not None:
            self.P = P0_override.copy()
        else:
            self.P = np.diag([
                1.0, 1.0, 1.0,       # Position uncertainty (m)
                0.5, 0.5, 0.5,       # Velocity uncertainty (m/s)
            ])

        # UKF3D instance
        self.ukf = UKF3D(Q=Q_override, R=R_override)

        # NN model + feature extractor
        self.nn_model = nn_model
        self.feature_stats = feature_stats
        self.feature_extractor = FeatureExtractor3D()

        # Snapshot of last known-good state for NaN/inf recovery
        self._last_good_x = self.x.copy()
        self._last_good_P = self.P.copy()

        # Store detection info (keep pixel-space originals for display)
        self.confidence = initial_detection.get("confidence", 0.0)
        self.bbox = initial_detection.get("bbox", {})
        self._last_pixel_center = initial_detection.get("center", {})

    @staticmethod
    def _has_valid_depth(detection):
        """Check if a detection has valid depth for meter-space tracking."""
        return detection.get("depth", 0.0) > 0

    def _convert_detection(self, detection):
        """
        Convert a detection dict (pixel + depth) to world-frame meters.

        Pipeline: pixel → camera-frame meters → world-frame meters.

        Assumes depth > 0 — callers must verify with _has_valid_depth().
        Raises AssertionError if the converted position is outside the
        scene radius bound (catches pixel-leak regressions).
        """
        center = detection.get("center", {})
        u_px = center.get("x", 0.0)
        v_px = center.get("y", 0.0)
        depth = detection.get("depth", 0.0)

        # Step 1: pixel → camera-frame meters (standard pinhole inversion)
        p_cam = np.array(self.camera.pixel_to_meters(u_px, v_px, depth))

        # Step 2: camera-frame → world-frame
        p_world = self.R_world_from_cam @ p_cam + self.t_world_from_cam
        x_m, y_m, z_m = p_world

        # Hard invariant: converted position must be in plausible scene range.
        # This catches any future "pixels leaked into meters" regressions.
        assert abs(x_m) < cfg.MAX_SCENE_RADIUS and abs(y_m) < cfg.MAX_SCENE_RADIUS, (
            f"Converted position ({x_m:.1f}, {y_m:.1f}, {z_m:.1f}) m "
            f"exceeds scene bound {cfg.MAX_SCENE_RADIUS} m — likely "
            f"pixel-to-meter conversion error (u={u_px}, v={v_px}, d={depth})"
        )

        return x_m, y_m, z_m

    def _state_is_valid(self):
        """Return True if state and covariance contain only finite values."""
        return np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P))

    def _revert_to_last_good(self, reason: str):
        """Revert state to last known-good snapshot."""
        logger.warning("Track %d: %s — reverting to last good state", self.track_id, reason)
        self.x = self._last_good_x.copy()
        self.P = self._last_good_P.copy()

    def predict(self, dt: float, a_control: Optional[np.ndarray] = None):
        """Predict state forward using UKF3D with optional analytical control."""
        self.ukf.dt = dt

        try:
            self.x, self.P, delta_a = self.ukf.predict(
                self.x, self.P,
                nn_model=self.nn_model,
                feature_extractor=self.feature_extractor,
                feature_stats=self.feature_stats,
                a_control=a_control,
            )
        except (ValueError, np.linalg.LinAlgError):
            self._revert_to_last_good("numerical error in predict")
            return

        if not self._state_is_valid():
            self._revert_to_last_good("NaN/inf after predict")
            return

        # Covariance bounds check
        if np.max(np.diag(self.P)) > cfg.P_MAX_DIAG:
            logger.warning("Track %d: P diagonal exceeds %.0f — resetting covariance",
                           self.track_id, cfg.P_MAX_DIAG)
            self.P = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

        self._last_good_x = self.x.copy()
        self._last_good_P = self.P.copy()
        self.age += 1

    def update(self, detection: Dict, timestamp: float):
        """
        Update state with measurement, converting pixels to meters.

        If depth is missing the update is skipped entirely (predict-only)
        to avoid injecting pixel-scale values into the meter-space UKF.
        """
        if not self._has_valid_depth(detection):
            # No valid depth — skip measurement update, keep prediction
            self.misses += 1
            return

        x_m, y_m, z_m = self._convert_detection(detection)
        z = np.array([x_m, y_m, z_m])

        try:
            self.x, self.P, innovation, S = self.ukf.update(self.x, self.P, z)
        except (ValueError, np.linalg.LinAlgError):
            self._revert_to_last_good("numerical error in update")
            self.misses += 1
            return

        if not self._state_is_valid():
            self._revert_to_last_good("NaN/inf after update")
            self.misses += 1
            return

        # Commit innovation to feature extractor (handles NaN internally)
        self.feature_extractor.commit_innovation(innovation)

        self._last_good_x = self.x.copy()
        self._last_good_P = self.P.copy()

        # Update metadata
        self.last_update = timestamp
        self.hits += 1
        self.misses = 0
        self.confidence = detection.get("confidence", self.confidence)
        self.bbox = detection.get("bbox", self.bbox)
        self._last_pixel_center = detection.get("center", self._last_pixel_center)

    def mark_missed(self):
        """Called when no detection matched this tracker."""
        self.misses += 1

    def get_state(self) -> ObjectState:
        """
        Get current state as ObjectState (meters, camera- or world-frame).

        The frame label is determined by the extrinsics passed at
        construction:
          - Identity extrinsics -> ``frame="camera_metric"`` (world ==
            camera). Gravity is NOT aligned with -Z here; the UKF's
            world-frame gravity + hover vectors cancel each other, so
            the predict model is effectively CV + NN residual. Downstream
            physics validation must treat gravity as unavailable.
          - Non-identity extrinsics -> ``frame="world_metric"``. The UKF's
            gravity vector is physically meaningful.
        """
        return ObjectState(
            track_id=self.track_id,
            timestamp=self.last_update,
            x=float(self.x[0]),
            y=float(self.x[1]),
            z=float(self.x[2]),
            vx=float(self.x[3]),
            vy=float(self.x[4]),
            vz=float(self.x[5]),
            pos_uncertainty=float(np.sqrt(np.trace(self.P[:3, :3]))),
            vel_uncertainty=float(np.sqrt(np.trace(self.P[3:, 3:]))),
            confidence=self.confidence,
            bbox=self.bbox,
            units="meters",
            frame=self._frame_tag,
        )

    def get_pixel_position(self):
        """Project current world-frame state back to pixel coordinates."""
        # World → camera frame (inverse of R @ p_cam + t)
        p_cam = self.R_world_from_cam.T @ (self.x[:3] - self.t_world_from_cam)
        u, v = self.camera.meters_to_pixel(p_cam[0], p_cam[1], p_cam[2])
        return u, v

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    @property
    def is_dead(self) -> bool:
        return self.misses > 5


class MultiObjectUKFNN:
    """
    Multi-object tracker using UKF-NN.
    Drop-in replacement for MultiObjectKalmanFilter with improved accuracy.

    Association is done in pixel space (for compatibility with detection
    coordinates).  UKF state is in meters.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        nn_model: Optional[DeltaAccelNN3D] = None,
        feature_stats: Optional[Dict] = None,
        camera: Optional[CameraIntrinsics] = None,
        R_world_from_cam: Optional[np.ndarray] = None,
        t_world_from_cam: Optional[np.ndarray] = None,
        max_tracks: int = cfg.MAX_TRACKS,
    ):
        self.trackers: Dict[int, UKFNNTracker] = {}
        self.next_id = 0
        self.max_distance = max_distance  # pixels (association space)
        self.max_tracks = max_tracks
        self.nn_model = nn_model
        self.feature_stats = feature_stats
        self.camera = camera or CameraIntrinsics()
        self.R_world_from_cam = R_world_from_cam
        self.t_world_from_cam = t_world_from_cam
        self.last_timing = {}  # sub-stage timing from latest update()

    def update(self, detections: List[Dict], timestamp: float) -> List[ObjectState]:
        """Update all trackers with new detections."""
        if not self.trackers:
            for det in detections:
                self._create_tracker(det, timestamp)
            self.last_timing = {}
            return self._get_confirmed_states()

        # Compute dt
        last_time = max(t.last_update for t in self.trackers.values())
        dt = max(timestamp - last_time, 0.001)

        # Predict all trackers
        t_predict_start = time.perf_counter()
        for tracker in self.trackers.values():
            tracker.predict(dt)
        t_predict_end = time.perf_counter()

        # Associate detections to trackers in pixel space
        t_assoc_start = time.perf_counter()
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(self.trackers.keys())

        if detections and self.trackers:
            cost_matrix = np.zeros((len(detections), len(self.trackers)))
            tracker_ids = list(self.trackers.keys())

            for i, det in enumerate(detections):
                det_center = det.get("center", {})
                det_pos = np.array([
                    det_center.get("x", 0),
                    det_center.get("y", 0),
                ])
                for j, tid in enumerate(tracker_ids):
                    tracker = self.trackers[tid]
                    # Project tracker state to pixel space for association
                    pred_u, pred_v = tracker.get_pixel_position()
                    pred_pos = np.array([pred_u, pred_v])
                    cost_matrix[i, j] = np.linalg.norm(det_pos - pred_pos)

            # Hungarian (optimal) assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_dets = set()
            matched_tracks = set()
        t_assoc_end = time.perf_counter()

        # Update matched trackers
        t_update_start = time.perf_counter()
        if detections and self.trackers:
            for det_idx, track_idx in zip(row_ind, col_ind):
                if cost_matrix[det_idx, track_idx] > self.max_distance:
                    continue
                track_id = tracker_ids[track_idx]
                self.trackers[track_id].update(detections[det_idx], timestamp)
                matched_dets.add(det_idx)
                matched_tracks.add(track_id)

            unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
            unmatched_trackers = [tid for tid in tracker_ids if tid not in matched_tracks]

        # Mark unmatched trackers
        for tid in unmatched_trackers:
            self.trackers[tid].mark_missed()

        # Create new trackers
        for det_idx in unmatched_detections:
            self._create_tracker(detections[det_idx], timestamp)

        # Remove dead trackers
        dead_ids = [tid for tid, t in self.trackers.items() if t.is_dead]
        for tid in dead_ids:
            del self.trackers[tid]
        t_update_end = time.perf_counter()

        self.last_timing = {
            'predict_ms': (t_predict_end - t_predict_start) * 1000,
            'assoc_ms': (t_assoc_end - t_assoc_start) * 1000,
            'update_ms': (t_update_end - t_update_start) * 1000,
            'n_tracks': len(self.trackers),
            'n_detections': len(detections),
        }

        return self._get_confirmed_states()

    def _create_tracker(self, detection: Dict, timestamp: float):
        if not UKFNNTracker._has_valid_depth(detection):
            logger.debug("Skipping detection without valid depth")
            return
        if len(self.trackers) >= self.max_tracks:
            logger.warning("Max tracks (%d) reached, skipping new detection", self.max_tracks)
            return
        tracker = UKFNNTracker(
            self.next_id, detection, timestamp,
            self.nn_model, self.feature_stats, self.camera,
            self.R_world_from_cam, self.t_world_from_cam,
        )
        self.trackers[self.next_id] = tracker
        self.next_id += 1

    def _get_confirmed_states(self) -> List[ObjectState]:
        return [t.get_state() for t in self.trackers.values() if t.is_confirmed]

    def clear(self):
        self.trackers.clear()


class NumpyNN:
    """
    Lightweight NN inference using pure numpy (no torch dependency).

    Forward: y = tanh(tanh(x @ W1.T + b1) @ W2.T + b2) * a_max

    Implements the same predict_numpy() interface as DeltaAccelNN3D,
    so it's a drop-in replacement for UKF3D.predict().
    """

    def __init__(self, weights_path):
        from pathlib import Path
        path = Path(weights_path)

        if path.suffix == '.npz' or str(path).endswith('.weights.npz'):
            data = np.load(path)
        else:
            # Check for companion .weights.npz next to .pt
            npz_path = path.with_suffix('.weights.npz')
            if npz_path.exists():
                data = np.load(npz_path)
            else:
                raise FileNotFoundError(
                    f"Cannot load weights from {path}. "
                    f"Provide a .weights.npz file (train with train_numpy.py)."
                )

        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.a_max = cfg.A_MAX

    def predict_numpy(self, features_normalized):
        """
        Forward pass with pre-normalized features.

        Args:
            features_normalized: (15,) numpy array
        Returns:
            delta_a: (3,) numpy array
        """
        h = np.tanh(features_normalized @ self.W1.T + self.b1)
        out = np.tanh(h @ self.W2.T + self.b2) * self.a_max
        return out


def create_ukf_nn_tracker(
    max_distance: float = 100.0,
    model_path: Optional[str] = None,
    stats_path: Optional[str] = None,
    camera: Optional[CameraIntrinsics] = None,
    R_world_from_cam: Optional[np.ndarray] = None,
    t_world_from_cam: Optional[np.ndarray] = None,
    max_tracks: int = cfg.MAX_TRACKS,
) -> MultiObjectUKFNN:
    """
    Create a UKF-NN multi-object tracker.

    Args:
        max_distance: Max pixel distance for detection-track association
        model_path: Path to trained NN model (.weights.npz or .pt file)
        stats_path: Path to feature statistics (.json file)
        camera: Camera intrinsics for pixel-to-meter conversion
        R_world_from_cam: (3,3) rotation matrix, camera→world. Default: identity.
        t_world_from_cam: (3,) translation vector (camera origin in world). Default: zeros.
    """
    nn_model = None
    feature_stats = None

    if model_path:
        from pathlib import Path
        p = Path(model_path)

        # Prefer numpy weights (works without torch on Jetson)
        if p.suffix == '.npz' or str(p).endswith('.weights.npz'):
            try:
                nn_model = NumpyNN(model_path)
                logger.info("Loaded UKF-NN model (numpy) from %s "
                            "[W1=%s, W2=%s]", model_path,
                            nn_model.W1.shape, nn_model.W2.shape)
            except Exception as e:
                logger.warning("Failed to load UKF-NN numpy model: %s", e)
        else:
            # Try torch .pt loading
            try:
                import torch
                nn_model = DeltaAccelNN3D()
                nn_model.load_state_dict(
                    torch.load(model_path, map_location='cpu', weights_only=True))
                nn_model.eval()
                logger.info("Loaded UKF-NN model (torch) from %s", model_path)
            except Exception as e:
                # Fall back to companion .weights.npz
                npz_path = p.with_suffix('.weights.npz')
                if npz_path.exists():
                    try:
                        nn_model = NumpyNN(str(npz_path))
                        logger.info("Loaded UKF-NN model (numpy fallback) from %s", npz_path)
                    except Exception as e2:
                        logger.warning("Failed to load UKF-NN model: torch=%s, numpy=%s", e, e2)
                else:
                    logger.warning("Failed to load UKF-NN model: %s", e)

    if stats_path:
        try:
            import json
            with open(stats_path, 'r') as f:
                feature_stats = json.load(f)
            logger.info("Loaded feature stats from %s", stats_path)
        except Exception as e:
            logger.warning("Failed to load feature stats: %s", e)

    # Metadata guard: verify loaded stats match current config
    if feature_stats is not None:
        _verify_stats_metadata(feature_stats, stats_path)

    return MultiObjectUKFNN(
        max_distance, nn_model, feature_stats, camera,
        R_world_from_cam, t_world_from_cam,
        max_tracks=max_tracks,
    )


def _verify_stats_metadata(stats, source="(unknown)"):
    """
    Verify that loaded normalization stats are compatible with the
    current config.  Checks feat_dim and stat vector lengths.

    Raises RuntimeError on mismatch to prevent silent wrong-model loads
    after breaking changes (e.g. FEAT_DIM 15 -> 12).
    """
    # Check embedded metadata if present
    meta_feat_dim = stats.get("feat_dim")
    if meta_feat_dim is not None and meta_feat_dim != cfg.FEAT_DIM:
        raise RuntimeError(
            f"Stats file {source} has feat_dim={meta_feat_dim}, "
            f"but config expects FEAT_DIM={cfg.FEAT_DIM}. "
            f"Retrain with current feature definition."
        )

    # Always check stat vector length (works even without embedded metadata)
    mean_len = len(stats.get("mean", []))
    std_len = len(stats.get("std", []))
    if mean_len != cfg.FEAT_DIM or std_len != cfg.FEAT_DIM:
        raise RuntimeError(
            f"Stats file {source} has mean/std length {mean_len}/{std_len}, "
            f"but config expects FEAT_DIM={cfg.FEAT_DIM}. "
            f"Retrain with current feature definition."
        )
