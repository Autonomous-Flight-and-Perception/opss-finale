"""
B2₃ Fusion Module
Combines validated state estimates from multiple sources (tracker state,
physics validation). Outputs final state estimation for robot control.

FUSED-STATE CONTRACT:
    - ``physics_plausible``  True iff the physics validator considers the
                             state physically plausible (velocity /
                             acceleration bounds respected). Previously
                             misnamed ``kalman_valid``.
    - ``physics_valid``      True iff the physics-model prediction agreed
                             with the observation within tolerance AND
                             the state is physics-plausible.
    - ``tracker_valid``      True by default; reserved for a tracker-side
                             health signal (e.g. Kalman innovation gate
                             or UKF NIS check). Not currently populated
                             by either tracker; placeholder kept so the
                             control-output aggregate has a named slot
                             for future use.
    - ``units`` / ``frame``  Propagated verbatim from the input
                             ObjectState. See ObjectState docstring for
                             the meaning of frame values. The control
                             side MUST read ``frame`` to interpret the
                             fused position/velocity.
"""
import numpy as np
from collections import deque
from typing import Dict, Deque, List, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class FusedState:
    """Final fused state output for robot control"""
    track_id: int
    timestamp: float

    # Fused position (weighted combination of sources)
    x: float
    y: float
    z: float

    # Fused velocity
    vx: float
    vy: float
    vz: float

    # Confidence in the fused estimate (0-1)
    confidence: float

    # Source weights used in fusion
    kalman_weight: float = 0.0
    physics_weight: float = 0.0

    # Validation status (see module docstring for semantics)
    physics_plausible: bool = True
    physics_valid: bool = True
    tracker_valid: bool = True

    # Prediction for control
    predicted_position: Optional[Tuple[float, float, float]] = None
    time_to_intercept: Optional[float] = None

    # Bounding box (for visualization)
    bbox: Dict = field(default_factory=dict)

    # Coordinate units / frame (propagated from input ObjectState)
    units: str = "pixels"
    frame: str = "pixel"

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "velocity": {"vx": self.vx, "vy": self.vy, "vz": self.vz},
            "confidence": self.confidence,
            "weights": {
                "kalman": self.kalman_weight,
                "physics": self.physics_weight
            },
            "validation": {
                "physics_plausible": self.physics_plausible,
                "physics_valid": self.physics_valid,
                "tracker_valid": self.tracker_valid,
            },
            "prediction": {
                "position": self.predicted_position,
                "time_to_intercept": self.time_to_intercept
            },
            "bbox": self.bbox,
            "units": self.units,
            "frame": self.frame,
        }

    def to_control_output(self) -> Dict:
        """
        Canonical per-target shape for the cobot control wire protocol
        (schema ``opss.cobot.v1``).

        This is the payload the robot control system receives for each
        target inside the ``targets`` list emitted by
        :meth:`UnixSocketBroadcaster.send_control_output`.

        Shape::

            {
              "track_id":            <int>,
              "timestamp":           <float, Unix-epoch seconds>,
              "position":            {"x", "y", "z"},        # per units+frame
              "velocity":            {"vx", "vy", "vz"},     # per units+frame/s
              "units":               "meters" | "pixels",
              "frame":               "pixel"|"pixel_xy_metric_z"|
                                     "camera_metric"|"world_metric",
              "confidence":          <float in [0,1]>,
              "valid":               <bool>,                 # aggregate
              "validation": {
                "physics_plausible": <bool>,
                "physics_valid":     <bool>,
                "tracker_valid":     <bool>,
              },
              "predicted_position":  {"x", "y", "z"} | null, # horizon s ahead
              "time_to_intercept":   <float seconds> | null, # == prediction_horizon,
                                                             # NOT literal intercept
              "bbox":                {"x1","y1","x2","y2"}   # pixel-space,
                                     | null,                  # display-only, NOT
                                                             # kinematic state
            }

        Units / frame interpretation: the consumer MUST read ``frame`` (or
        equivalently the envelope's ``pipeline.frame``) to interpret
        position / velocity. No field is assumed to be in a fixed frame.
        ``bbox`` is ALWAYS pixel-space at capture resolution, intended for
        visualization only — it is NOT part of the kinematic control state.
        """
        predicted = None
        if self.predicted_position is not None:
            px, py, pz = self.predicted_position
            predicted = {"x": float(px), "y": float(py), "z": float(pz)}

        bbox: Optional[Dict] = None
        if self.bbox:
            # Only include a bbox if we have a usable one; keep it as int
            # pixel coords to match the draw-annotations contract.
            bbox = {
                "x1": int(self.bbox.get("x1", 0)),
                "y1": int(self.bbox.get("y1", 0)),
                "x2": int(self.bbox.get("x2", 0)),
                "y2": int(self.bbox.get("y2", 0)),
            }

        return {
            "track_id": int(self.track_id),
            "timestamp": float(self.timestamp),
            "position": {
                "x": float(self.x),
                "y": float(self.y),
                "z": float(self.z),
            },
            "velocity": {
                "vx": float(self.vx),
                "vy": float(self.vy),
                "vz": float(self.vz),
            },
            "units": self.units,
            "frame": self.frame,
            "confidence": float(self.confidence),
            "valid": bool(
                self.physics_plausible
                and self.physics_valid
                and self.tracker_valid
            ),
            "validation": {
                "physics_plausible": bool(self.physics_plausible),
                "physics_valid": bool(self.physics_valid),
                "tracker_valid": bool(self.tracker_valid),
            },
            "predicted_position": predicted,
            "time_to_intercept": (
                float(self.time_to_intercept)
                if self.time_to_intercept is not None
                else None
            ),
            "bbox": bbox,
        }


@dataclass
class DiagnosticData:
    """Diagnostic metadata for validation loop"""
    timestamp: float
    track_id: int

    # Stored prediction (for future comparison)
    stored_prediction: Optional[Tuple[float, float, float]] = None
    prediction_time: Optional[float] = None

    # Actual observation (when available)
    actual_observation: Optional[Tuple[float, float, float]] = None
    observation_time: Optional[float] = None

    # Computed error
    prediction_error: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "stored_prediction": self.stored_prediction,
            "prediction_time": self.prediction_time,
            "actual_observation": self.actual_observation,
            "observation_time": self.observation_time,
            "prediction_error": self.prediction_error
        }


class B23Fusion:
    """
    B2₃ Fusion: Combines state estimates from multiple sources.

    Flow:
    1. Receive Kalman filter state estimate
    2. Receive physics validation result
    3. Fuse estimates using weighted average based on validation
    4. Output final state for control
    5. Store predictions for diagnostic feedback loop
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize B2₃ fusion module.

        Args:
            config: Configuration dict with keys:
                - kalman_base_weight: Base weight for Kalman estimates (default 0.6)
                - physics_base_weight: Base weight for physics estimates (default 0.4)
                - invalid_penalty: Weight reduction for invalid estimates (default 0.8)
                - prediction_horizon: How far ahead to predict for control (seconds)
        """
        self.config = config or {}

        # Fusion weights
        self.kalman_base_weight = self.config.get("kalman_base_weight", 0.6)
        self.physics_base_weight = self.config.get("physics_base_weight", 0.4)
        self.invalid_penalty = self.config.get("invalid_penalty", 0.8)

        # Control prediction
        self.prediction_horizon = self.config.get("prediction_horizon", 0.5)  # seconds

        # Diagnostic storage.
        # ``_predictions`` now holds a deque of pending predictions PER
        # track so ``compare_prediction`` can match an observation to the
        # prediction that targeted its observation time, instead of the
        # single most-recently-stored one (which by construction never
        # matched the ±0.1 s time window).
        self._predictions: Dict[int, Deque[DiagnosticData]] = {}
        self._prediction_match_tolerance = 0.1  # seconds
        self._prediction_max_age = max(2.0 * self.prediction_horizon, 1.0)
        self._error_history: List[DiagnosticData] = []

    def fuse(
        self,
        kalman_state: 'ObjectState',
        physics_result: 'ValidationResult'
    ) -> FusedState:
        """
        Fuse Kalman filter state with physics validation.

        Args:
            kalman_state: State estimate from Kalman filter
            physics_result: Validation result from physics engine

        Returns:
            FusedState with combined estimate
        """
        # Compute adaptive weights based on validation
        kalman_weight = self.kalman_base_weight
        physics_weight = self.physics_base_weight

        # Adjust weights based on validation
        if not physics_result.physics_plausible:
            # Physics says this is implausible - reduce Kalman weight
            kalman_weight *= self.invalid_penalty

        if not physics_result.is_valid:
            # Large prediction error - trust physics prediction more
            physics_weight *= 1.2
            kalman_weight *= 0.8

        # Normalize weights
        total_weight = kalman_weight + physics_weight
        if total_weight > 0:
            kalman_weight /= total_weight
            physics_weight /= total_weight

        # Fuse position
        if physics_result.predicted_position and physics_result.is_valid:
            # Use weighted average of Kalman state and physics prediction
            fused_x = kalman_weight * kalman_state.x + physics_weight * physics_result.predicted_position[0]
            fused_y = kalman_weight * kalman_state.y + physics_weight * physics_result.predicted_position[1]
            fused_z = kalman_weight * kalman_state.z + physics_weight * physics_result.predicted_position[2]
        else:
            # No physics prediction available - use Kalman only
            fused_x = kalman_state.x
            fused_y = kalman_state.y
            fused_z = kalman_state.z
            kalman_weight = 1.0
            physics_weight = 0.0

        # Fuse velocity (similar logic)
        if physics_result.predicted_velocity:
            fused_vx = kalman_weight * kalman_state.vx + physics_weight * physics_result.predicted_velocity[0]
            fused_vy = kalman_weight * kalman_state.vy + physics_weight * physics_result.predicted_velocity[1]
            fused_vz = kalman_weight * kalman_state.vz + physics_weight * physics_result.predicted_velocity[2]
        else:
            fused_vx = kalman_state.vx
            fused_vy = kalman_state.vy
            fused_vz = kalman_state.vz

        # Compute confidence
        base_confidence = kalman_state.confidence
        if not physics_result.is_valid:
            base_confidence *= 0.7
        if not physics_result.physics_plausible:
            base_confidence *= 0.5

        # Predict future position for control
        predicted_pos = (
            fused_x + fused_vx * self.prediction_horizon,
            fused_y + fused_vy * self.prediction_horizon,
            fused_z + fused_vz * self.prediction_horizon
        )

        # Create fused state. Both ``units`` and ``frame`` are propagated
        # verbatim from the tracker-emitted ObjectState so the control
        # side can correctly interpret the fused kinematics.
        fused = FusedState(
            track_id=kalman_state.track_id,
            timestamp=kalman_state.timestamp,
            x=fused_x,
            y=fused_y,
            z=fused_z,
            vx=fused_vx,
            vy=fused_vy,
            vz=fused_vz,
            confidence=base_confidence,
            kalman_weight=kalman_weight,
            physics_weight=physics_weight,
            physics_plausible=physics_result.physics_plausible,
            physics_valid=physics_result.is_valid,
            # ``tracker_valid`` is a placeholder until a tracker-side
            # health signal is wired in (see module docstring).
            tracker_valid=True,
            predicted_position=predicted_pos,
            time_to_intercept=self.prediction_horizon,
            bbox=kalman_state.bbox,
            units=getattr(kalman_state, "units", "pixels"),
            frame=getattr(kalman_state, "frame", "pixel"),
        )

        # Store prediction for diagnostic loop
        self._store_prediction(fused)

        return fused

    def fuse_batch(
        self,
        kalman_states: List['ObjectState'],
        physics_results: List['ValidationResult']
    ) -> List[FusedState]:
        """Fuse multiple states at once"""
        # Match by track_id
        results_by_id = {r.state_id: r for r in physics_results}

        fused_states = []
        for state in kalman_states:
            if state.track_id in results_by_id:
                fused = self.fuse(state, results_by_id[state.track_id])
                fused_states.append(fused)

        return fused_states

    def _store_prediction(self, fused: FusedState):
        """
        Queue a prediction for future comparison.

        A new DiagnosticData is appended to the per-track deque each
        frame. ``prediction_time`` is ``timestamp + prediction_horizon``
        (the wall-clock at which this prediction targets the observation).
        Stale entries (older than ``_prediction_max_age``) are dropped.
        """
        diagnostic = DiagnosticData(
            timestamp=fused.timestamp,
            track_id=fused.track_id,
            stored_prediction=fused.predicted_position,
            prediction_time=fused.timestamp + self.prediction_horizon,
        )
        queue = self._predictions.setdefault(fused.track_id, deque())
        queue.append(diagnostic)
        cutoff = fused.timestamp - self._prediction_max_age
        while queue and queue[0].timestamp < cutoff:
            queue.popleft()

    def compare_prediction(
        self,
        track_id: int,
        actual_position: Tuple[float, float, float],
        observation_time: float,
        tolerance: Optional[float] = None,
    ) -> Optional[DiagnosticData]:
        """
        Compare a stored prediction with an actual observation.

        Walks the per-track queue for the earliest prediction whose
        ``prediction_time`` falls within ±``tolerance`` of
        ``observation_time``. On match: computes error, records it in
        ``_error_history``, removes the matched entry, and returns it.

        This implements the diagnostic feedback loop from the OPSS
        architecture. It previously returned None on every call because
        ``_predictions[track_id]`` was a scalar overwritten every frame
        before it could age to ``prediction_horizon``.
        """
        if track_id not in self._predictions:
            return None
        tol = tolerance if tolerance is not None else self._prediction_match_tolerance

        queue = self._predictions[track_id]
        match_idx: Optional[int] = None
        for i, diagnostic in enumerate(queue):
            if diagnostic.stored_prediction is None:
                continue
            if diagnostic.prediction_time is None:
                continue
            if abs(observation_time - diagnostic.prediction_time) <= tol:
                match_idx = i
                break

        if match_idx is None:
            return None

        # Walk the deque by index — deques don't allow del by index, so
        # rotate the match to the front, popleft, then rotate back.
        queue.rotate(-match_idx)
        diagnostic = queue.popleft()
        queue.rotate(match_idx)

        pred = np.array(diagnostic.stored_prediction)
        actual = np.array(actual_position)
        error = float(np.linalg.norm(pred - actual))

        diagnostic.actual_observation = actual_position
        diagnostic.observation_time = observation_time
        diagnostic.prediction_error = error

        self._error_history.append(diagnostic)
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-500:]

        return diagnostic

    def get_error_statistics(self) -> Dict:
        """Get statistics on prediction errors (for model improvement)"""
        if not self._error_history:
            return {"count": 0, "mean_error": 0, "max_error": 0, "std_error": 0}

        errors = [d.prediction_error for d in self._error_history if d.prediction_error is not None]
        if not errors:
            return {"count": 0, "mean_error": 0, "max_error": 0, "std_error": 0}

        return {
            "count": len(errors),
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors)),
            "min_error": float(np.min(errors))
        }

    def clear(self):
        """Clear all stored data"""
        self._predictions.clear()
        self._error_history.clear()


# Convenience function
def create_fusion(config: Optional[Dict] = None) -> B23Fusion:
    """Create a new B2₃ fusion module"""
    return B23Fusion(config)
