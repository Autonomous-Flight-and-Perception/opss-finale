"""
Kalman Filter for Object State Estimation
Tracks detected objects and estimates position, velocity, and predicts future states.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
from scipy.optimize import linear_sum_assignment


@dataclass
class ObjectState:
    """
    Complete state of a tracked object.

    CANONICAL STATE CONTRACT (load-bearing across tracker -> validator ->
    fusion -> cobot/control output boundaries):

      - ``timestamp``   wall-clock seconds since Unix epoch (``time.time()``)
                        at which this estimate is valid; same clock that
                        the pipeline loop uses to compute dt.
      - ``x, y, z``     position, unit/frame-tagged by ``units`` + ``frame``.
      - ``vx, vy, vz``  first derivative of position per second, same unit
                        and frame as the position triple.
      - ``bbox``        pixel-space, capture resolution, display-only.
                        It is NOT part of the kinematic state vector.
      - ``confidence``  detection confidence in [0, 1].
      - ``units``       legacy tag: one of {"pixels", "meters"}. Describes
                        the PRIMARY unit regime for the tracker. Kalman
                        emits "pixels" and UKF-NN emits "meters"; this is
                        kept for backward compatibility and for the pixel/
                        meter velocity-arrow scaling in the annotation
                        renderer. Prefer ``frame`` for semantic reasoning.
      - ``frame``       LOAD-BEARING coordinate/unit frame tag used by
                        validator and fusion:
                            "pixel"             all six kinematic fields
                                                are pixel-space.
                            "pixel_xy_metric_z" x,y,vx,vy in pixels / px·s⁻¹;
                                                z,vz in meters / m·s⁻¹
                                                (Kalman reality: depth
                                                from RealSense, lateral
                                                from detection centroid).
                            "camera_metric"     all six in meters / m·s⁻¹,
                                                camera-frame (+X right,
                                                +Y down, +Z forward).
                                                World == camera because
                                                extrinsics were not given.
                                                Gravity is NOT aligned to
                                                -Z; do not apply world
                                                gravity physics.
                            "world_metric"      all six in meters / m·s⁻¹,
                                                world-frame (z-up, gravity
                                                along -z). Physics model
                                                applies.
                        Default: "pixel".

    Downstream code that computes scalar speed, mixes axes, or applies
    physics thresholds MUST branch on ``frame``. ``units`` is not
    sufficient because "pixels" is overloaded (Kalman's state is not
    unit-homogeneous under the "pixels" tag).
    """
    track_id: int
    timestamp: float

    # Position (unit/frame-tagged by units + frame fields below)
    x: float
    y: float
    z: float = 0.0  # Depth if available

    # Velocity estimates (same unit/frame as position)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Uncertainty (covariance diagonal — sqrt-trace of P sub-blocks,
    # same unit as position/velocity respectively)
    pos_uncertainty: float = 0.0
    vel_uncertainty: float = 0.0

    # Detection info
    confidence: float = 0.0
    bbox: Dict = field(default_factory=dict)

    # Coordinate units ("pixels" for Kalman tracker, "meters" for UKF-NN).
    # Legacy label — see class docstring. Prefer ``frame`` for semantics.
    units: str = "pixels"

    # Coordinate/unit frame — load-bearing across subsystem boundaries.
    # See class docstring for legal values.
    frame: str = "pixel"

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "velocity": {"vx": self.vx, "vy": self.vy, "vz": self.vz},
            "uncertainty": {
                "position": self.pos_uncertainty,
                "velocity": self.vel_uncertainty
            },
            "confidence": self.confidence,
            "bbox": self.bbox,
            "units": self.units,
            "frame": self.frame,
        }

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector"""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def predict_position(self, dt: float) -> Tuple[float, float, float]:
        """Predict position after dt seconds using constant velocity model"""
        return (
            self.x + self.vx * dt,
            self.y + self.vy * dt,
            self.z + self.vz * dt
        )


class KalmanTracker:
    """
    Single object Kalman filter tracker.
    State vector: [x, y, z, vx, vy, vz]
    """

    def __init__(self, track_id: int, initial_detection: Dict, timestamp: float):
        self.track_id = track_id
        self.last_update = timestamp
        self.age = 0  # Number of frames tracked
        self.hits = 1  # Number of successful updates
        self.misses = 0  # Consecutive frames without detection

        # Initialize state from detection center
        center = initial_detection.get("center", {})
        self.x = np.array([
            center.get("x", 0.0),
            center.get("y", 0.0),
            initial_detection.get("depth", 0.0),
            0.0, 0.0, 0.0  # Initial velocity = 0
        ], dtype=np.float64)

        # State covariance (uncertainty)
        self.P = np.diag([
            100.0, 100.0, 500.0,  # Position uncertainty (pixels)
            50.0, 50.0, 50.0      # Velocity uncertainty (pixels/sec)
        ])

        # Process noise (how much we expect state to change)
        self.Q = np.diag([
            1.0, 1.0, 5.0,       # Position noise
            10.0, 10.0, 10.0     # Velocity noise
        ])

        # Measurement noise (detection uncertainty)
        self.R = np.diag([25.0, 25.0, 100.0])  # x, y, z measurement noise

        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float64)

        # Store detection info
        self.confidence = initial_detection.get("confidence", 0.0)
        self.bbox = initial_detection.get("bbox", {})

    def predict(self, dt: float):
        """Predict state forward by dt seconds"""
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float64)

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt

        self.age += 1

    def update(self, detection: Dict, timestamp: float):
        """Update state with new detection"""
        center = detection.get("center", {})
        z = np.array([
            center.get("x", self.x[0]),
            center.get("y", self.x[1]),
            detection.get("depth", self.x[2])
        ], dtype=np.float64)

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.misses += 1
            return

        # Update state
        y = z - self.H @ self.x  # Innovation
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        # Update metadata
        self.last_update = timestamp
        self.hits += 1
        self.misses = 0
        self.confidence = detection.get("confidence", self.confidence)
        self.bbox = detection.get("bbox", self.bbox)

    def mark_missed(self):
        """Called when no detection matched this tracker"""
        self.misses += 1

    def get_state(self) -> ObjectState:
        """
        Get current state as ObjectState.

        The Kalman filter state is NOT unit-homogeneous:
          - x[0], x[1] are pixel coordinates of the detection centroid
          - x[2]       is depth in meters (sampled from the depth frame)
          - x[3], x[4] are pixels / second
          - x[5]       is meters / second
        So the emitted ObjectState is tagged ``frame="pixel_xy_metric_z"``
        to signal this to validator / fusion. ``units`` stays "pixels" for
        back-compat with the legacy annotation renderer.
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
            units="pixels",
            frame="pixel_xy_metric_z",
        )

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed if it has enough hits"""
        return self.hits >= 3

    @property
    def is_dead(self) -> bool:
        """Track is dead if too many consecutive misses"""
        return self.misses > 5


class MultiObjectKalmanFilter:
    """
    Multi-object tracker using Kalman filters.
    Handles detection-to-track association and track lifecycle.
    """

    def __init__(self, max_distance: float = 100.0, max_tracks: int = 50):
        self.trackers: Dict[int, KalmanTracker] = {}
        self.next_id = 0
        self.max_distance = max_distance  # Max pixels for association
        self.max_tracks = max_tracks

    def update(self, detections: List[Dict], timestamp: float) -> List[ObjectState]:
        """
        Update all trackers with new detections.
        Returns list of current object states.
        """
        if not self.trackers:
            # No existing tracks - create new ones for all detections
            for det in detections:
                self._create_tracker(det, timestamp)
            return self._get_confirmed_states()

        # Compute time delta
        if self.trackers:
            last_time = max(t.last_update for t in self.trackers.values())
            dt = max(timestamp - last_time, 0.001)
        else:
            dt = 0.033  # Assume 30fps

        # Predict all trackers forward
        for tracker in self.trackers.values():
            tracker.predict(dt)

        # Associate detections to trackers (simple nearest neighbor)
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(self.trackers.keys())

        if detections and self.trackers:
            # Build cost matrix (distance between predicted positions and detections)
            cost_matrix = np.zeros((len(detections), len(self.trackers)))
            tracker_ids = list(self.trackers.keys())

            for i, det in enumerate(detections):
                det_center = det.get("center", {})
                det_x = det_center.get("x", 0)
                det_y = det_center.get("y", 0)

                for j, tid in enumerate(tracker_ids):
                    tracker = self.trackers[tid]
                    pred_x, pred_y = tracker.x[0], tracker.x[1]
                    dist = np.sqrt((det_x - pred_x)**2 + (det_y - pred_y)**2)
                    cost_matrix[i, j] = dist

            # Hungarian (optimal) assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_dets = set()
            matched_tracks = set()

            for det_idx, track_idx in zip(row_ind, col_ind):
                if cost_matrix[det_idx, track_idx] > self.max_distance:
                    continue
                track_id = tracker_ids[track_idx]
                self.trackers[track_id].update(detections[det_idx], timestamp)
                matched_dets.add(det_idx)
                matched_tracks.add(track_id)

            unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
            unmatched_trackers = [tid for tid in tracker_ids if tid not in matched_tracks]

        # Mark unmatched trackers as missed
        for tid in unmatched_trackers:
            self.trackers[tid].mark_missed()

        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            self._create_tracker(detections[det_idx], timestamp)

        # Remove dead trackers
        dead_ids = [tid for tid, t in self.trackers.items() if t.is_dead]
        for tid in dead_ids:
            del self.trackers[tid]

        return self._get_confirmed_states()

    def _create_tracker(self, detection: Dict, timestamp: float):
        """Create a new tracker for a detection"""
        if len(self.trackers) >= self.max_tracks:
            return
        tracker = KalmanTracker(self.next_id, detection, timestamp)
        self.trackers[self.next_id] = tracker
        self.next_id += 1

    def _get_confirmed_states(self) -> List[ObjectState]:
        """Get states of all confirmed tracks"""
        return [
            t.get_state()
            for t in self.trackers.values()
            if t.is_confirmed
        ]

    def predict_states(self, dt: float) -> List[ObjectState]:
        """Predict all confirmed states forward by dt seconds"""
        states = []
        for tracker in self.trackers.values():
            if tracker.is_confirmed:
                state = tracker.get_state()
                pred_x, pred_y, pred_z = state.predict_position(dt)
                states.append(ObjectState(
                    track_id=state.track_id,
                    timestamp=state.timestamp + dt,
                    x=pred_x,
                    y=pred_y,
                    z=pred_z,
                    vx=state.vx,
                    vy=state.vy,
                    vz=state.vz,
                    pos_uncertainty=state.pos_uncertainty * (1 + dt),  # Grows with time
                    vel_uncertainty=state.vel_uncertainty,
                    confidence=state.confidence * 0.9,  # Decay confidence for predictions
                    bbox=state.bbox,
                    units=state.units,
                    frame=state.frame,
                ))
        return states

    def clear(self):
        """Clear all trackers"""
        self.trackers.clear()


# Convenience function
def create_tracker(max_distance: float = 100.0, max_tracks: int = 50) -> MultiObjectKalmanFilter:
    """Create a new multi-object Kalman filter tracker"""
    return MultiObjectKalmanFilter(max_distance=max_distance, max_tracks=max_tracks)
