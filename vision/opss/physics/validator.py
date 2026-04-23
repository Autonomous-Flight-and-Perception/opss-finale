"""
Physics Validator
Validates object states against physics constraints using the b2 physics engine.

SEMANTIC CONTRACT:
    The validator branches on ``state.frame`` (see ObjectState docstring).
    - "world_metric":      meter-space, world-frame. Gravity applies.
                           max_velocity / position_tolerance are SI (m, m/s).
                           _predict_state uses semi-implicit Euler + gravity.
    - "camera_metric":     meter-space, camera-frame (world == camera).
                           Gravity is NOT aligned with -Z, so gravity-aware
                           prediction is NOT applied — CV only.
                           max_velocity / position_tolerance are SI (m, m/s).
    - "pixel":             pure pixel-space. Pixel thresholds apply
                           (max_velocity_pixels, position_tolerance_pixels).
                           CV prediction.
    - "pixel_xy_metric_z": Kalman's mixed frame. Lateral (xy) is bounded
                           with pixel thresholds; depth (z) with meter
                           thresholds. CV prediction in both sub-frames.
                           position_error is not meaningfully comparable
                           across sub-frames; no aggregate tolerance check.
    - unknown:             physics_plausible=False with reason.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Add sim directory to path
sys.path.insert(0, os.path.dirname(__file__))

from .sim.physics_core import ProjectileSystem, ProjectileState
from .sim.engine import Simulation, EngineConfig
from .sim.forces import from_config_dict


# World-frame gravity used for "world_metric" predict (m/s^2, z-up).
# Matches opss/state/ukf_nn/config.py GRAVITY.
_GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])


@dataclass
class ValidationResult:
    """Result of physics validation for a single state"""
    is_valid: bool
    state_id: int
    timestamp: float

    # Input state
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]

    # Physics prediction
    predicted_position: Optional[Tuple[float, float, float]] = None
    predicted_velocity: Optional[Tuple[float, float, float]] = None

    # Validation metrics
    position_error: float = 0.0  # Distance between predicted and observed
    velocity_error: float = 0.0
    physics_plausible: bool = True  # Does it obey basic physics?

    # Frame of the validated state (propagated from ObjectState.frame)
    frame: str = "pixel"

    # Reasons for failure
    failure_reasons: List[str] = None

    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = []

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "state_id": self.state_id,
            "timestamp": self.timestamp,
            "position": self.position,
            "velocity": self.velocity,
            "predicted_position": self.predicted_position,
            "predicted_velocity": self.predicted_velocity,
            "errors": {
                "position": self.position_error,
                "velocity": self.velocity_error
            },
            "physics_plausible": self.physics_plausible,
            "frame": self.frame,
            "failure_reasons": self.failure_reasons
        }


class PhysicsValidator:
    """
    Validates object states against physics models.
    Uses the b2 physics engine for trajectory prediction and validation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the physics validator.

        Args:
            config: Physics configuration dict with keys:
                - gravity: float (m/s^2, default 9.81)
                - max_velocity: float (max plausible speed in meter-frame
                                states, m/s; default 50.0)
                - max_acceleration: float (max plausible acceleration
                                in meter-frame states, m/s^2; default 100.0)
                - position_tolerance: float (max allowed position error
                                for meter-frame states, meters)
                - velocity_tolerance: float (max allowed velocity error
                                for meter-frame states, m/s)
                - max_velocity_pixels: float (max plausible speed in
                                pixel-frame states, px/s; default 2000.0)
                - position_tolerance_pixels: float (max allowed position
                                error for pixel-frame states, px;
                                default 100.0)
                - velocity_tolerance_pixels: float (max allowed velocity
                                error for pixel-frame states, px/s;
                                default 400.0)
        """
        self.config = config or {}

        # Physics parameters (meter-frame)
        self.gravity = self.config.get("gravity", 9.81)
        self.max_velocity = self.config.get("max_velocity", 50.0)  # m/s
        self.max_acceleration = self.config.get("max_acceleration", 100.0)  # m/s^2
        self.position_tolerance = self.config.get("position_tolerance", 50.0)  # m
        self.velocity_tolerance = self.config.get("velocity_tolerance", 100.0)  # m/s

        # Pixel-frame thresholds. A person walking across a 1280x720 frame
        # at ~2 m from a 60° FoV camera moves on the order of 200-400 px/s;
        # a fast projectile can briefly exceed 1000 px/s. 2000 px/s is a
        # generous "physically impossible under 30 fps" ceiling.
        self.max_velocity_pixels = self.config.get("max_velocity_pixels", 2000.0)
        self.position_tolerance_pixels = self.config.get("position_tolerance_pixels", 100.0)
        self.velocity_tolerance_pixels = self.config.get("velocity_tolerance_pixels", 400.0)

        # History for trajectory validation
        self._state_history: Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]] = {}

        # Physics simulation config
        self._sim_config = {
            "gravity": {"g": self.gravity, "scale": 1.0}
        }

    def validate_state(self, state: 'ObjectState') -> ValidationResult:
        """
        Validate a single object state, branching on ``state.frame``.

        Checks performed:
          1. Velocity bounds (frame-dependent units).
          2. Consistency with previous observations (predicted vs observed),
             if history is available.
          3. Acceleration bounds (frame-dependent units).

        Unknown frames are marked physics_plausible=False and skipped.
        """
        position = (state.x, state.y, state.z)
        velocity = (state.vx, state.vy, state.vz)

        failure_reasons = []
        is_valid = True
        physics_plausible = True

        # Prefer the load-bearing ``frame`` field; fall back to legacy
        # ``units`` for states that predate the frame contract.
        frame = getattr(state, "frame", None)
        if frame is None:
            units = getattr(state, "units", "pixels")
            frame = "pixel" if units == "pixels" else "camera_metric"

        _KNOWN_FRAMES = ("pixel", "pixel_xy_metric_z", "camera_metric", "world_metric")
        if frame not in _KNOWN_FRAMES:
            physics_plausible = False
            is_valid = False
            failure_reasons.append(
                f"Unknown frame '{frame}' for track {state.track_id} — "
                f"cannot validate physically"
            )
            return ValidationResult(
                is_valid=False,
                state_id=state.track_id,
                timestamp=state.timestamp,
                position=position,
                velocity=velocity,
                physics_plausible=False,
                frame=frame,
                failure_reasons=failure_reasons,
            )

        # --- Check 1: Velocity bounds (frame-dependent) -------------------
        if frame == "pixel_xy_metric_z":
            # Kalman's mixed frame: bound lateral pixel speed and depth
            # meter speed SEPARATELY; do NOT mix them in a single scalar.
            lateral_pxs = float(np.hypot(state.vx, state.vy))  # px/s
            depth_mps = float(abs(state.vz))                    # m/s
            if lateral_pxs > self.max_velocity_pixels:
                failure_reasons.append(
                    f"Lateral speed {lateral_pxs:.1f} px/s exceeds "
                    f"max {self.max_velocity_pixels:.1f} px/s"
                )
                physics_plausible = False
                is_valid = False
            if depth_mps > self.max_velocity:
                failure_reasons.append(
                    f"Depth speed {depth_mps:.1f} m/s exceeds "
                    f"max {self.max_velocity:.1f} m/s"
                )
                physics_plausible = False
                is_valid = False
        elif frame == "pixel":
            # Pure pixel-space: pixel thresholds, scalar speed is coherent.
            speed_pxs = float(np.sqrt(state.vx**2 + state.vy**2 + state.vz**2))
            if speed_pxs > self.max_velocity_pixels:
                failure_reasons.append(
                    f"Speed {speed_pxs:.1f} px/s exceeds "
                    f"max {self.max_velocity_pixels:.1f} px/s"
                )
                physics_plausible = False
                is_valid = False
        else:
            # camera_metric or world_metric: SI thresholds, scalar speed coherent.
            speed_mps = float(np.sqrt(state.vx**2 + state.vy**2 + state.vz**2))
            if speed_mps > self.max_velocity:
                failure_reasons.append(
                    f"Speed {speed_mps:.1f} m/s exceeds "
                    f"max {self.max_velocity:.1f} m/s"
                )
                physics_plausible = False
                is_valid = False

        # --- Check 2: Compare with history (if available) -----------------
        predicted_position = None
        predicted_velocity = None
        position_error = 0.0
        velocity_error = 0.0

        if state.track_id in self._state_history:
            history = self._state_history[state.track_id]
            if len(history) >= 2:
                prev_t, prev_pos, prev_vel = history[-1]
                dt = state.timestamp - prev_t

                if dt > 0:
                    pred_result = self._predict_state(prev_pos, prev_vel, dt, frame)
                    if pred_result:
                        predicted_position = tuple(pred_result["position"])
                        predicted_velocity = tuple(pred_result["velocity"])

                        pos_arr = np.array(position)
                        pred_pos_arr = np.array(predicted_position)
                        vel_arr = np.array(velocity)
                        pred_vel_arr = np.array(predicted_velocity)

                        if frame == "pixel_xy_metric_z":
                            # Separate lateral (pixels) and depth (meters)
                            # error budgets — do not collapse to a single
                            # norm (dimensionally incoherent).
                            lat_err_px = float(np.hypot(
                                pos_arr[0] - pred_pos_arr[0],
                                pos_arr[1] - pred_pos_arr[1],
                            ))
                            depth_err_m = float(abs(pos_arr[2] - pred_pos_arr[2]))
                            position_error = lat_err_px  # primary signal
                            if lat_err_px > self.position_tolerance_pixels:
                                failure_reasons.append(
                                    f"Lateral position error {lat_err_px:.1f} px "
                                    f"exceeds tolerance {self.position_tolerance_pixels:.1f} px"
                                )
                                is_valid = False
                            if depth_err_m > self.position_tolerance:
                                failure_reasons.append(
                                    f"Depth position error {depth_err_m:.2f} m "
                                    f"exceeds tolerance {self.position_tolerance:.2f} m"
                                )
                                is_valid = False
                            velocity_error = float(np.hypot(
                                vel_arr[0] - pred_vel_arr[0],
                                vel_arr[1] - pred_vel_arr[1],
                            ))
                        elif frame == "pixel":
                            position_error = float(np.linalg.norm(pos_arr - pred_pos_arr))
                            velocity_error = float(np.linalg.norm(vel_arr - pred_vel_arr))
                            if position_error > self.position_tolerance_pixels:
                                failure_reasons.append(
                                    f"Position error {position_error:.1f} px "
                                    f"exceeds tolerance {self.position_tolerance_pixels:.1f} px"
                                )
                                is_valid = False
                            if velocity_error > self.velocity_tolerance_pixels:
                                failure_reasons.append(
                                    f"Velocity error {velocity_error:.1f} px/s "
                                    f"exceeds tolerance {self.velocity_tolerance_pixels:.1f} px/s"
                                )
                                is_valid = False
                        else:
                            # camera_metric / world_metric
                            position_error = float(np.linalg.norm(pos_arr - pred_pos_arr))
                            velocity_error = float(np.linalg.norm(vel_arr - pred_vel_arr))
                            if position_error > self.position_tolerance:
                                failure_reasons.append(
                                    f"Position error {position_error:.2f} m "
                                    f"exceeds tolerance {self.position_tolerance:.2f} m"
                                )
                                is_valid = False
                            if velocity_error > self.velocity_tolerance:
                                failure_reasons.append(
                                    f"Velocity error {velocity_error:.2f} m/s "
                                    f"exceeds tolerance {self.velocity_tolerance:.2f} m/s"
                                )
                                is_valid = False

                    # Acceleration bound — dimensionally coherent for pure
                    # frames; for pixel_xy_metric_z we bound lateral-only.
                    if dt > 0.001:
                        if frame == "pixel_xy_metric_z":
                            lat_accel_pxs2 = float(np.hypot(
                                (velocity[0] - prev_vel[0]) / dt,
                                (velocity[1] - prev_vel[1]) / dt,
                            ))
                            # Max lateral pixel-accel: choose "1 frame to
                            # saturate max velocity", i.e. max_v_px * fps.
                            max_accel_pxs2 = self.max_velocity_pixels * 30.0
                            if lat_accel_pxs2 > max_accel_pxs2:
                                failure_reasons.append(
                                    f"Lateral accel {lat_accel_pxs2:.0f} px/s^2 "
                                    f"exceeds max {max_accel_pxs2:.0f} px/s^2"
                                )
                                physics_plausible = False
                                is_valid = False
                        elif frame == "pixel":
                            accel_pxs2 = float(np.linalg.norm(
                                (np.array(velocity) - prev_vel) / dt
                            ))
                            max_accel_pxs2 = self.max_velocity_pixels * 30.0
                            if accel_pxs2 > max_accel_pxs2:
                                failure_reasons.append(
                                    f"Accel {accel_pxs2:.0f} px/s^2 "
                                    f"exceeds max {max_accel_pxs2:.0f} px/s^2"
                                )
                                physics_plausible = False
                                is_valid = False
                        else:
                            accel_mps2 = float(np.linalg.norm(
                                (np.array(velocity) - prev_vel) / dt
                            ))
                            if accel_mps2 > self.max_acceleration:
                                failure_reasons.append(
                                    f"Accel {accel_mps2:.1f} m/s^2 "
                                    f"exceeds max {self.max_acceleration:.1f} m/s^2"
                                )
                                physics_plausible = False
                                is_valid = False

        # --- Store in history ---------------------------------------------
        if state.track_id not in self._state_history:
            self._state_history[state.track_id] = []

        self._state_history[state.track_id].append((
            state.timestamp,
            np.array(position),
            np.array(velocity)
        ))

        # Keep history bounded
        if len(self._state_history[state.track_id]) > 100:
            self._state_history[state.track_id] = self._state_history[state.track_id][-50:]

        return ValidationResult(
            is_valid=is_valid,
            state_id=state.track_id,
            timestamp=state.timestamp,
            position=position,
            velocity=velocity,
            predicted_position=predicted_position,
            predicted_velocity=predicted_velocity,
            position_error=position_error,
            velocity_error=velocity_error,
            physics_plausible=physics_plausible,
            frame=frame,
            failure_reasons=failure_reasons
        )

    def validate_states(self, states: List['ObjectState']) -> List[ValidationResult]:
        """Validate multiple states"""
        return [self.validate_state(s) for s in states]

    def _predict_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        dt: float,
        frame: str = "pixel",
    ) -> Optional[Dict]:
        """
        Predict the next state under a frame-appropriate physics model.

        - "world_metric":  semi-implicit Euler with gravity ([0,0,-9.81] m/s^2).
                           v' = v + dt*g, p' = p + dt*v'. This is the only
                           frame where the prediction is genuinely "physics";
                           downstream fusion can trust it as an independent
                           physics channel.
        - "camera_metric": CV only (world != camera, gravity direction not
                           known in the camera frame).
        - "pixel" / "pixel_xy_metric_z": CV only (image-plane motion is not
                           governed by world gravity; depth axis in Kalman
                           is meter-space but without a full camera/world
                           pose we cannot apply gravity to it either).
        """
        try:
            if frame == "world_metric":
                new_vel = velocity + _GRAVITY_WORLD * dt
                new_pos = position + new_vel * dt
            else:
                new_pos = position + velocity * dt
                new_vel = velocity.copy()

            return {
                "position": new_pos.tolist(),
                "velocity": new_vel.tolist()
            }

        except Exception as e:
            print(f"[PHYSICS] Prediction error: {e}")
            return None

    def predict_trajectory(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        duration: float,
        dt: float = 0.01
    ) -> List[Dict]:
        """
        Predict full trajectory using physics engine.

        Args:
            position: Initial position (x, y, z) in meters
            velocity: Initial velocity (vx, vy, vz) in m/s
            duration: How far to predict (seconds)
            dt: Time step (seconds)

        Returns:
            List of state dicts with position, velocity at each timestep
        """
        try:
            # Use analytical physics for vacuum trajectory
            system = ProjectileSystem(
                initial_position=np.array(position),
                initial_velocity=np.array(velocity),
                gravity=self.gravity
            )

            result = system.trajectory(t_end=duration, dt=dt, include_metadata=True)

            trajectory = []
            for i, t in enumerate(result["times"]):
                trajectory.append({
                    "t": float(t),
                    "position": result["positions"][i].tolist(),
                    "velocity": result["velocities"][i].tolist()
                })

            return trajectory

        except Exception as e:
            print(f"[PHYSICS] Trajectory prediction error: {e}")
            return []

    def clear_history(self, track_id: Optional[int] = None):
        """Clear state history for a track or all tracks"""
        if track_id is not None:
            self._state_history.pop(track_id, None)
        else:
            self._state_history.clear()


# Convenience function
def create_validator(config: Optional[Dict] = None) -> PhysicsValidator:
    """Create a new physics validator"""
    return PhysicsValidator(config)
