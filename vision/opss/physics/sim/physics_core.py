#!/usr/bin/env python3
"""
OPSS Physics Core - Stage 1: Analytical Baseline (Vacuum) - VERSION 5

Final refinements:
- Enhanced docstrings with explicit clamping effects
- np.hypot for numerical stability in range calculations
- Slots on dataclasses for memory efficiency
- __all__ for stable public API
- time_at_height() helper method
- Improved type hints
"""

from typing import Dict, Tuple, Optional, Any, Literal, TypedDict, Union
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import warnings


__all__ = [
    'PhysicsConstants',
    'ToleranceHelper', 
    'ReferenceFrame',
    'ProjectileState',
    'ProjectileSystem',
    'create_default_system',
    'TrajectoryResult'
]


class TrajectoryResult(TypedDict, total=False):
    """Type definition for trajectory() return value."""
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    accelerations: NDArray[np.float64]
    # Optional metadata fields
    speeds: NDArray[np.float64]
    kinetic_energies: NDArray[np.float64]
    potential_energies: NDArray[np.float64]
    total_energies: NDArray[np.float64]
    specific_energies: NDArray[np.float64]
    trajectory_params: Dict[str, Any]


class PhysicsConstants:
    """Immutable physics constants for the simulation."""
    
    EARTH_GRAVITY: float = 9.80665  # Standard gravity (m/s²) - ISO 80000-3:2006
    NUMERICAL_TOLERANCE: float = 1e-12  # Absolute numerical tolerance
    RELATIVE_TOLERANCE: float = 1e-10  # Relative numerical tolerance
    GROUND_TOLERANCE: float = 1e-6  # Tolerance for ground contact detection
    
    # Flat-Earth validity limits
    MAX_RANGE_FLAT_EARTH: float = 100_000  # 100 km horizontal range limit
    MAX_TIME_FLAT_EARTH: float = 300  # 5 minute flight time limit
    
    # Default simulation parameters
    DEFAULT_MAX_TIME: float = 1000.0  # Default max simulation time (configurable)


class ToleranceHelper:
    """Centralized tolerance management for consistent numerical checks."""
    
    @staticmethod
    def abs_tol() -> float:
        """Get absolute tolerance for general use."""
        return PhysicsConstants.NUMERICAL_TOLERANCE
    
    @staticmethod
    def rel_tol(scale: float = 1.0) -> float:
        """Get relative tolerance scaled appropriately."""
        return PhysicsConstants.RELATIVE_TOLERANCE * abs(scale)
    
    @staticmethod
    def ground_tol() -> float:
        """Get tolerance for ground contact detection."""
        return PhysicsConstants.GROUND_TOLERANCE
    
    @staticmethod
    def adaptive_tol(value: float, scale: Optional[float] = None) -> float:
        """
        Get adaptive tolerance combining absolute and relative.
        
        Args:
            value: Primary value to consider
            scale: Optional secondary scale (defaults to value)
        """
        if scale is None:
            scale = value
        reference = max(abs(value), abs(scale))
        return max(
            PhysicsConstants.NUMERICAL_TOLERANCE,
            PhysicsConstants.RELATIVE_TOLERANCE * reference
        )


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    """Define coordinate system and reference conventions."""
    
    name: str = "ENU"  # East-North-Up
    description: str = "Right-handed: X=East, Y=North, Z=Up"
    gravity_direction: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )
    ground_plane_z: float = 0.0  # Ground reference height
    energy_reference_z: float = 0.0  # Potential energy zero reference
    
    def __post_init__(self) -> None:
        """Normalize gravity direction to unit vector and freeze arrays."""
        if not isinstance(self.gravity_direction, np.ndarray):
            object.__setattr__(self, 'gravity_direction', 
                             np.array(self.gravity_direction, dtype=np.float64))
        
        # Make a copy to prevent external mutation
        object.__setattr__(self, 'gravity_direction', 
                         self.gravity_direction.copy())
        
        # Normalize to unit vector
        norm = np.linalg.norm(self.gravity_direction)
        if norm == 0:
            raise ValueError("Gravity direction cannot be zero vector")
        if not np.isclose(norm, 1.0):
            object.__setattr__(self, 'gravity_direction', 
                             self.gravity_direction / norm)
        
        # Freeze the array
        self.gravity_direction.flags.writeable = False


@dataclass(slots=True)
class ProjectileState:
    """Complete state of a projectile at a given time."""
    
    time: float
    position: NDArray[np.float64]      # no defaults above here
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    gravity_magnitude: float = PhysicsConstants.EARTH_GRAVITY
    mass: float = 1.0
    energy_reference_z: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate state vectors."""
        for attr_name in ['position', 'velocity', 'acceleration']:
            attr = getattr(self, attr_name)
            if not isinstance(attr, np.ndarray):
                object.__setattr__(self, attr_name, np.array(attr, dtype=np.float64))
            if attr.shape != (3,):
                raise ValueError(f"{attr_name} must be a 3D vector, got shape {attr.shape}")
    
    @property
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy: KE = ½mv²."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)
    
    @property
    def potential_energy(self) -> float:
        """Calculate gravitational potential energy: U = mgh (h relative to reference)."""
        height_above_reference = self.position[2] - self.energy_reference_z
        return self.mass * self.gravity_magnitude * height_above_reference
    
    @property
    def total_energy(self) -> float:
        """Total mechanical energy: E = KE + U."""
        return self.kinetic_energy + self.potential_energy
    
    @property
    def specific_energy(self) -> float:
        """Energy per unit mass: e = E/m."""
        return self.total_energy / self.mass
    
    @property
    def speed(self) -> float:
        """Calculate scalar speed: |v|."""
        return np.linalg.norm(self.velocity)


class ProjectileSystem:
    """
    High-precision analytical projectile motion system.
    
    Implements exact closed-form solutions for projectile motion
    in a uniform gravitational field with no air resistance.
    
    Important Notes:
    - clamp_to_ground is for rendering only, not physics validation
    - Model permits sub-ground positions unless explicitly clamped
    - impact_position key absent from trajectory metadata when t_impact is infinite
    - When stop_at_ground=True, trajectory is truncated at impact time
    """
    
    def __init__(
        self,
        initial_position: NDArray[np.float64],
        initial_velocity: NDArray[np.float64],
        gravity: float = PhysicsConstants.EARTH_GRAVITY,
        mass: float = 1.0,
        max_simulation_time: float = PhysicsConstants.DEFAULT_MAX_TIME,
        reference_frame: Optional[ReferenceFrame] = None,
    ) -> None:
        """
        Initialize the projectile system.
        
        Args:
            initial_position: Initial position [x, y, z] in meters
            initial_velocity: Initial velocity [vx, vy, vz] in m/s
            gravity: Gravitational acceleration magnitude (m/s²)
            mass: Projectile mass in kg (default 1.0)
            max_simulation_time: Maximum allowed simulation time
            reference_frame: Coordinate system definition
        
        Raises:
            ValueError: If initial conditions are invalid
        """
        # Make defensive copies of input arrays
        self.r0 = np.array(initial_position, dtype=np.float64).copy()
        self.v0 = np.array(initial_velocity, dtype=np.float64).copy()
        self.g = float(gravity)
        self.mass = float(mass)
        self.max_time = float(max_simulation_time)
        self.frame = reference_frame or ReferenceFrame()
        
        # Validate inputs
        self._validate_initial_conditions()
        
        # Precompute frequently used values
        self.gravity_vector = self.g * self.frame.gravity_direction
        self.half_g = 0.5 * self.gravity_vector
        
        # Calculate trajectory parameters
        self._compute_trajectory_parameters()
        
        # Check validity limits
        self._check_flat_earth_validity()
    
    def _validate_initial_conditions(self) -> None:
        """Validate initial conditions for physical consistency."""
        # Shape validation
        if self.r0.shape != (3,):
            raise ValueError(f"Initial position must be 3D, got shape {self.r0.shape}")
        if self.v0.shape != (3,):
            raise ValueError(f"Initial velocity must be 3D, got shape {self.v0.shape}")
        
        # Physical bounds validation
        if self.g <= 0:
            raise ValueError(f"Gravity must be positive, got {self.g}")
        if self.g > 100:
            raise ValueError(f"Gravity {self.g} m/s² exceeds reasonable bounds")
        if self.mass <= 0:
            raise ValueError(f"Mass must be positive, got {self.mass}")
        if self.max_time <= 0:
            raise ValueError(f"Max simulation time must be positive, got {self.max_time}")
        
        # Numerical validation
        if np.any(np.isnan(self.r0)) or np.any(np.isinf(self.r0)):
            raise ValueError("Initial position contains NaN or Inf")
        if np.any(np.isnan(self.v0)) or np.any(np.isinf(self.v0)):
            raise ValueError("Initial velocity contains NaN or Inf")
        
        # Ground penetration check
        if self.r0[2] < self.frame.ground_plane_z:
            raise ValueError(
                f"Initial position z={self.r0[2]} is below ground plane "
                f"z={self.frame.ground_plane_z}. This model does not support "
                "subsurface trajectories."
            )
        
        # Terminal state check
        if np.isclose(self.r0[2], self.frame.ground_plane_z, 
                     atol=ToleranceHelper.ground_tol()) and self.v0[2] < 0:
            warnings.warn(
                f"Projectile starting at ground level (z={self.r0[2]}) with "
                f"downward velocity (vz={self.v0[2]}). This is a terminal state - "
                "projectile cannot go below ground. Model permits sub-ground "
                "positions unless clamp_to_ground=True is used.",
                UserWarning
            )
    
    def _stable_quadratic_root(self, a: float, b: float, c: float, 
                               positive_only: bool = True) -> Optional[float]:
        """
        Compute quadratic roots with numerical stability.
        
        For ax² + bx + c = 0, uses stable formulation to avoid cancellation.
        
        Args:
            a, b, c: Quadratic coefficients
            positive_only: Return only positive root if True
            
        Returns:
            Stable root or None if no valid solution (when positive_only=True)
        """
        if abs(a) < ToleranceHelper.abs_tol():
            # Linear case: bx + c = 0
            if abs(b) < ToleranceHelper.abs_tol():
                return None  # No solution
            root = -c / b
            # For positive_only, accept roots that are non-negative within tolerance
            # This allows legitimate tiny positive roots near zero
            return root if not positive_only or root > -ToleranceHelper.abs_tol() else None
        
        discriminant = b * b - 4 * a * c
        
        # Handle near-zero discriminant (grazing trajectories)
        if discriminant < -ToleranceHelper.abs_tol():
            return None  # Genuinely no real roots
        
        # Clamp tiny negative discriminants to zero (FP error)
        sqrt_disc = np.sqrt(max(discriminant, 0.0))
        
        # Use stable formulation to avoid cancellation
        if b >= 0:
            # Avoid b + sqrt_disc cancellation
            q = -0.5 * (b + sqrt_disc)
            root1 = q / a
            root2 = c / q if abs(q) > ToleranceHelper.abs_tol() else float('inf')
        else:
            # Avoid -b + sqrt_disc cancellation  
            q = -0.5 * (b - sqrt_disc)
            root1 = c / q if abs(q) > ToleranceHelper.abs_tol() else float('inf')
            root2 = q / a
        
        if positive_only:
            # Return smallest positive root
            # Use same tolerance as linear case for consistency
            roots = [r for r in [root1, root2] if r > -ToleranceHelper.abs_tol()]
            return min(roots) if roots else None
        else:
            # This path is not used in current code
            raise NotImplementedError("Non-positive-only case not needed")
    
    def _compute_trajectory_parameters(self) -> None:
        """Precompute useful trajectory parameters with enhanced precision."""
        # Time to reach maximum height (when vz = 0)
        if self.v0[2] > ToleranceHelper.abs_tol():
            # Upward trajectory - has an apex
            self.t_apex = self.v0[2] / self.g
            self.z_apex = self.r0[2] + self.v0[2]**2 / (2 * self.g)
        else:
            # Downward or horizontal - no upward phase
            self.t_apex = 0.0
            self.z_apex = self.r0[2]
        
        # Time to impact ground using stable quadratic formula
        ground_height = self.frame.ground_plane_z
        relative_height = self.r0[2] - ground_height
        
        # Use strict inequality for ground-start case to avoid flipping
        if relative_height > ToleranceHelper.ground_tol():
            # Above ground: solve z(t) = ground_height
            # 0.5*g*t² - vz*t - h = 0 where h = z0 - ground
            a = 0.5 * self.g
            b = -self.v0[2]
            c = -relative_height
            
            # Use numerically stable root finding
            t_root = self._stable_quadratic_root(a, b, c, positive_only=True)
            self.t_impact = t_root if t_root is not None else float('inf')
            
        elif abs(relative_height) <= ToleranceHelper.ground_tol():
            # At ground level (within tolerance)
            if self.v0[2] > ToleranceHelper.abs_tol():
                # Going up - will return to ground
                self.t_impact = 2 * self.v0[2] / self.g
            else:
                # Already at ground, not going up significantly
                self.t_impact = 0.0
        else:
            # Below ground - invalid but caught in validation
            self.t_impact = 0.0
        
        # Impact position and range
        if self.t_impact < float('inf'):
            # Calculate impact position
            impact_pos = self.r0 + self.v0 * self.t_impact + self.half_g * self.t_impact**2
            
            # Verify ground contact accuracy before clamping
            z_error = abs(impact_pos[2] - ground_height)
            if z_error > ToleranceHelper.ground_tol():
                warnings.warn(
                    f"Impact position calculation error: z_error={z_error:.2e} "
                    f"exceeds tolerance {ToleranceHelper.ground_tol():.2e}",
                    UserWarning
                )
            
            # Clamp to exact ground height
            impact_pos[2] = ground_height
            self.impact_position = impact_pos
            
            # Range is horizontal distance from launch to impact
            dx = self.impact_position[0] - self.r0[0]
            dy = self.impact_position[1] - self.r0[1]
            # Use np.hypot for numerical stability with large values
            self.horizontal_range = np.hypot(dx, dy)
        else:
            # Don't include impact_position key if t_impact is infinite
            self.impact_position = None
            self.horizontal_range = float('inf')
    
    def _check_flat_earth_validity(self) -> None:
        """Check if trajectory exceeds flat-Earth approximation limits."""
        # Only warn for finite values
        if self.horizontal_range < float('inf') and \
           self.horizontal_range > PhysicsConstants.MAX_RANGE_FLAT_EARTH:
            warnings.warn(
                f"Horizontal range {self.horizontal_range:.0f}m exceeds "
                f"flat-Earth validity limit of {PhysicsConstants.MAX_RANGE_FLAT_EARTH:.0f}m. "
                "Earth curvature effects may be significant.",
                UserWarning
            )
        
        if self.t_impact < float('inf') and \
           self.t_impact > PhysicsConstants.MAX_TIME_FLAT_EARTH:
            warnings.warn(
                f"Flight time {self.t_impact:.1f}s exceeds "
                f"flat-Earth validity limit of {PhysicsConstants.MAX_TIME_FLAT_EARTH:.0f}s. "
                "Coriolis effects may be significant.",
                UserWarning
            )
    
    def state(
        self, 
        t: float,
        clamp_to_ground: bool = False
    ) -> ProjectileState:
        """
        Calculate exact state at time t using analytical solution.
        
        Args:
            t: Time in seconds (must be non-negative)
            clamp_to_ground: If True, clamp position to ground for rendering only.
                           Does NOT affect velocity/acceleration for physics consistency.
                           Note: Clamping z-position will change potential_energy property.
            
        Returns:
            ProjectileState object with complete state information
            
        Note:
            When clamp_to_ground=True, only position is clamped. Velocity and
            acceleration remain unchanged to preserve physics calculations.
            However, the potential_energy property will reflect the clamped z-position,
            affecting total_energy calculations. Use clamping for visualization only,
            never for physics validation.
        """
        if t < 0:
            raise ValueError(f"Time must be non-negative, got {t}")
        if t > self.max_time:
            raise ValueError(
                f"Time {t} exceeds maximum simulation time {self.max_time}. "
                "Increase max_simulation_time if longer simulations needed."
            )
        
        # Analytical solution: r(t) = r0 + v0*t + 0.5*g*t²
        position = self.r0 + self.v0 * t + self.half_g * t * t
        
        # Clamp to ground if requested (rendering only)
        if clamp_to_ground and position[2] < self.frame.ground_plane_z:
            position = position.copy()  # Don't modify in-place
            position[2] = self.frame.ground_plane_z
        
        # Velocity: v(t) = v0 + g*t (NOT clamped for physics consistency)
        velocity = self.v0 + self.gravity_vector * t
        
        # Acceleration is constant (NOT clamped)
        acceleration = self.gravity_vector.copy()
        
        return ProjectileState(
            time=t,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            mass=self.mass,
            gravity_magnitude=self.g,
            energy_reference_z=self.frame.energy_reference_z
        )
    
    def trajectory(
        self, 
        t_end: Optional[float] = None,
        dt: float = 0.01,
        stop_at_ground: bool = True,
        include_metadata: bool = False
    ) -> TrajectoryResult:
        """
        Generate complete trajectory from t=0 to t=t_end.
        
        Args:
            t_end: End time in seconds (defaults to impact time if stop_at_ground)
            dt: Time step in seconds
            stop_at_ground: If True, truncate trajectory at ground impact time
            include_metadata: Include additional physics metadata
            
        Returns:
            Dictionary with time array and trajectory data.
            Notes:
            - impact_position key only present in metadata when t_impact is finite
            - When stop_at_ground=True, trajectory ends at impact time for physics consistency
            - Energies in metadata are computed from unclamped physical positions
        """
        # Determine end time
        if t_end is None:
            if stop_at_ground and self.t_impact < float('inf'):
                t_end = self.t_impact
            else:
                t_end = min(self.t_impact, 10.0) if self.t_impact < float('inf') else 10.0
        
        # Validate against max_time
        if t_end > self.max_time:
            raise ValueError(
                f"Requested end time {t_end} exceeds maximum simulation time {self.max_time}"
            )
        
        if t_end <= 0:
            raise ValueError(f"End time must be positive, got {t_end}")
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        if dt > t_end:
            raise ValueError(f"Time step {dt} exceeds end time {t_end}")
        
        # Force truncation at impact when stop_at_ground=True
        if stop_at_ground and self.t_impact < float('inf') and t_end > self.t_impact:
            t_end = self.t_impact
        
        # Generate time points
        n_points = int(np.ceil(t_end / dt)) + 1
        times = np.linspace(0, t_end, n_points)
        
        # Vectorized computation for all time points
        t_col = times.reshape(-1, 1)
        
        # r(t) = r0 + v0*t + 0.5*g*t²
        positions = self.r0 + self.v0 * t_col + self.half_g * (t_col * t_col)
        
        # v(t) = v0 + g*t
        velocities = self.v0 + self.gravity_vector * t_col
        
        # Constant acceleration
        accelerations = np.broadcast_to(self.gravity_vector, (n_points, 3))
        
        result: TrajectoryResult = {
            "times": times,
            "positions": positions,
            "velocities": velocities,
            "accelerations": accelerations,
        }
        
        if include_metadata:
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Compute energies from physical (unclamped) positions
            kinetic_energies = 0.5 * self.mass * np.sum(velocities**2, axis=1)
            heights_above_ref = positions[:, 2] - self.frame.energy_reference_z
            potential_energies = self.mass * self.g * heights_above_ref
            total_energies = kinetic_energies + potential_energies
            
            # Build trajectory params with full provenance
            traj_params = {
                "t_apex": self.t_apex,
                "z_apex": self.z_apex,
                "t_impact": self.t_impact,
                "horizontal_range": self.horizontal_range,
                "mass": self.mass,
                "gravity": self.g,
                "frame": self.frame.name,
                "gravity_direction": self.frame.gravity_direction.tolist(),
                "ground_plane_z": self.frame.ground_plane_z,
                "energy_reference_z": self.frame.energy_reference_z,
            }
            
            # Only include impact_position if it exists (JSON-serializable)
            if self.impact_position is not None:
                traj_params["impact_position"] = self.impact_position.tolist()
            
            result.update({
                "speeds": speeds,
                "kinetic_energies": kinetic_energies,
                "potential_energies": potential_energies,
                "total_energies": total_energies,
                "specific_energies": total_energies / self.mass,
                "trajectory_params": traj_params
            })
        
        return result
    
    def time_at_height(self, z: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Find time(s) when projectile reaches a specific height.
        
        Uses the stable quadratic solver to find when z(t) = z_target.
        Special cases for apex (vz=0) and impact (z=ground) are handled.
        
        Args:
            z: Target height in meters
            
        Returns:
            Tuple of (t_ascending, t_descending) where either may be None if
            the height is never reached. For apex height, both times converge.
            For heights never reached, returns (None, None).
        """
        # Special case: apex height
        if abs(z - self.z_apex) < ToleranceHelper.ground_tol():
            if self.t_apex > 0:
                return (self.t_apex, self.t_apex)
            else:
                # No apex (downward trajectory), check if starting height matches
                if abs(z - self.r0[2]) < ToleranceHelper.ground_tol():
                    return (0.0, 0.0)
                return (None, None)
        
        # Special case: ground height
        if abs(z - self.frame.ground_plane_z) < ToleranceHelper.ground_tol():
            if self.t_impact < float('inf'):
                # If starting at ground, ascending time is 0
                if abs(self.r0[2] - self.frame.ground_plane_z) < ToleranceHelper.ground_tol():
                    return (0.0, self.t_impact)
                else:
                    return (None, self.t_impact)
            return (None, None)
        
        # General case: solve z(t) = z_target
        # z0 + vz*t - 0.5*g*t² = z_target
        # 0.5*g*t² - vz*t - (z0 - z_target) = 0
        a = 0.5 * self.g
        b = -self.v0[2]
        c = -(self.r0[2] - z)
        
        # Get discriminant to check if height is reachable
        discriminant = b * b - 4 * a * c
        if discriminant < -ToleranceHelper.abs_tol():
            return (None, None)  # Height never reached
        
        # Calculate both roots
        sqrt_disc = np.sqrt(max(discriminant, 0.0))
        
        if b >= 0:
            q = -0.5 * (b + sqrt_disc)
            t1 = q / a
            t2 = c / q if abs(q) > ToleranceHelper.abs_tol() else float('inf')
        else:
            q = -0.5 * (b - sqrt_disc)
            t1 = c / q if abs(q) > ToleranceHelper.abs_tol() else float('inf')
            t2 = q / a
        
        # Filter for positive times and order them
        times = []
        for t in [t1, t2]:
            if t > -ToleranceHelper.abs_tol() and t < float('inf'):
                times.append(t)
        
        if not times:
            return (None, None)
        
        times.sort()
        
        # Determine which is ascending vs descending based on velocity
        if len(times) == 1:
            t = times[0]
            vz_at_t = self.v0[2] - self.g * t
            if vz_at_t > 0:
                return (t, None)  # Still ascending
            else:
                return (None, t)  # Descending
        else:
            # Two times - earlier is ascending, later is descending
            return (times[0], times[1])
    
    def time_of_flight(self) -> float:
        """
        Get total time of flight until ground impact.
        
        Returns:
            Time in seconds. Returns infinity if projectile never impacts ground.
        """
        return self.t_impact
    
    def get_range(self) -> float:
        """
        Get horizontal range of trajectory.
        
        Returns:
            Range in meters. Returns infinity if projectile never impacts ground.
        """
        return self.horizontal_range
    
    def get_horizontal_range(self) -> float:
        """
        Get horizontal range of trajectory (backward compatibility alias).
        
        Returns:
            Range in meters. Returns infinity if projectile never impacts ground.
        """
        return self.horizontal_range
    
    def verify_energy_conservation(
        self, 
        times: Optional[NDArray[np.float64]] = None,
        tolerance_mode: Literal["absolute", "relative", "adaptive"] = "adaptive"
    ) -> Tuple[bool, float, str]:
        """
        Verify energy conservation throughout trajectory.
        
        Args:
            times: Time points to check (defaults to trajectory)
            tolerance_mode: How to evaluate conservation tolerance
            
        Returns:
            Tuple of (is_conserved, max_deviation, report)
        """
        if times is None:
            t_max = min(self.t_impact, self.max_time) if self.t_impact < float('inf') else 10.0
            times = np.linspace(0, t_max, 100)
        
        initial_state = self.state(0.0)
        initial_energy = initial_state.total_energy
        
        max_abs_deviation = 0.0
        max_rel_deviation = 0.0
        worst_time = 0.0
        
        for t in times:
            current_state = self.state(t)
            current_energy = current_state.total_energy
            
            abs_deviation = abs(current_energy - initial_energy)
            rel_deviation = abs_deviation / abs(initial_energy) if initial_energy != 0 else abs_deviation
            
            if abs_deviation > max_abs_deviation:
                max_abs_deviation = abs_deviation
                max_rel_deviation = rel_deviation
                worst_time = t
        
        # Determine if conserved based on mode
        if tolerance_mode == "absolute":
            is_conserved = max_abs_deviation < ToleranceHelper.abs_tol()
            tolerance_used = ToleranceHelper.abs_tol()
        elif tolerance_mode == "relative":
            is_conserved = max_rel_deviation < PhysicsConstants.RELATIVE_TOLERANCE
            tolerance_used = PhysicsConstants.RELATIVE_TOLERANCE
        else:  # adaptive
            adaptive_tol = ToleranceHelper.adaptive_tol(initial_energy, abs(initial_energy))
            is_conserved = max_abs_deviation < adaptive_tol
            tolerance_used = adaptive_tol
        
        report = (
            f"Energy Conservation Report:\n"
            f"  Initial energy: {initial_energy:.6e} J\n"
            f"  Max absolute deviation: {max_abs_deviation:.6e} J\n"
            f"  Max relative deviation: {max_rel_deviation:.6e}\n"
            f"  Worst time: t={worst_time:.3f} s\n"
            f"  Tolerance mode: {tolerance_mode}\n"
            f"  Tolerance used: {tolerance_used:.6e}\n"
            f"  Conservation status: {'PASS' if is_conserved else 'FAIL'}"
        )
        
        return is_conserved, max_abs_deviation, report
    
    def verify_invariants(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify all physical invariants hold.
        
        Returns:
            Tuple of (all_valid, detailed_results)
        """
        results = {}
        all_valid = True
        
        # Add frame provenance to results
        results["ground_plane_z"] = self.frame.ground_plane_z
        results["gravity_direction"] = self.frame.gravity_direction.tolist()
        
        # Test times
        t_test = np.linspace(0, min(self.t_impact, 5.0), 20)
        
        # Use separate scales for position and velocity tolerances
        vel_scale = np.linalg.norm(self.v0)
        pos_scale = np.linalg.norm(self.r0)
        
        vel_tol = ToleranceHelper.adaptive_tol(vel_scale, vel_scale)
        pos_tol = ToleranceHelper.adaptive_tol(pos_scale, pos_scale)
        
        # Invariant 1: Constant horizontal velocity
        horiz_vel_const = True
        for t in t_test:
            state = self.state(t)
            vx_err = abs(state.velocity[0] - self.v0[0])
            vy_err = abs(state.velocity[1] - self.v0[1])
            if vx_err > vel_tol or vy_err > vel_tol:
                horiz_vel_const = False
                break
        results["horizontal_velocity_constant"] = horiz_vel_const
        all_valid &= horiz_vel_const
        
        # Invariant 2: Zero horizontal acceleration
        horiz_accel_zero = True
        for t in t_test:
            state = self.state(t)
            if abs(state.acceleration[0]) > ToleranceHelper.abs_tol() or \
               abs(state.acceleration[1]) > ToleranceHelper.abs_tol():
                horiz_accel_zero = False
                break
        results["horizontal_acceleration_zero"] = horiz_accel_zero
        all_valid &= horiz_accel_zero
        
        # Invariant 3: Apex relations (if applicable)
        if self.t_apex > 0:
            state_apex = self.state(self.t_apex)
            vz_at_apex = abs(state_apex.velocity[2])
            z_at_apex = state_apex.position[2]
            
            apex_velocity_zero = vz_at_apex < vel_tol
            apex_height_correct = abs(z_at_apex - self.z_apex) < pos_tol
            
            results["apex_velocity_zero"] = apex_velocity_zero
            results["apex_height_correct"] = apex_height_correct
            all_valid &= apex_velocity_zero and apex_height_correct
        
        # Invariant 4: Impact conditions (if applicable)
        if self.t_impact < float('inf'):
            state_impact = self.state(self.t_impact)
            z_at_impact = state_impact.position[2]
            
            ground_contact = abs(z_at_impact - self.frame.ground_plane_z) < ToleranceHelper.ground_tol()
            results["ground_contact_at_impact"] = ground_contact
            all_valid &= ground_contact
            
            # Range identity: horizontal_range ≈ ||v0_xy|| * t_impact
            v0_horizontal = np.sqrt(self.v0[0]**2 + self.v0[1]**2)
            expected_range = v0_horizontal * self.t_impact
            range_error = abs(self.horizontal_range - expected_range)
            range_tol = ToleranceHelper.adaptive_tol(expected_range, expected_range)
            range_identity_holds = range_error < range_tol
            
            results["range_identity"] = range_identity_holds
            results["range_error"] = range_error
            results["expected_range"] = expected_range
            all_valid &= range_identity_holds
        
        # Invariant 5: Polynomial form of z(t)
        polynomial_valid = True
        for t in t_test:
            state = self.state(t)
            z_expected = self.r0[2] + self.v0[2] * t - 0.5 * self.g * t**2
            z_error = abs(state.position[2] - z_expected)
            if z_error > pos_tol:
                polynomial_valid = False
                break
        results["z_polynomial_form"] = polynomial_valid
        all_valid &= polynomial_valid
        
        return all_valid, results
    
    def verify_trajectory_symmetry(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify trajectory symmetry properties (for parabolic motion).
        
        Returns:
            Tuple of (is_symmetric, metrics)
        """
        # Use per-quantity tolerances
        vel_scale = np.linalg.norm(self.v0)
        # Always compute pos_scale, not conditional on z_apex
        pos_scale = max(np.linalg.norm(self.r0), abs(self.z_apex))
        
        vel_tol = ToleranceHelper.adaptive_tol(vel_scale, vel_scale)
        pos_tol = ToleranceHelper.adaptive_tol(pos_scale, pos_scale)
        
        # Can only verify symmetry for parabolic trajectories with apex
        if self.t_apex <= 0:
            return False, {"reason": "No parabolic trajectory (no upward phase)"}
        
        if self.t_impact <= 0:
            return False, {"reason": "No flight phase (immediate ground contact)"}
        
        # For true symmetry, need launch and impact at same height
        launch_height = self.r0[2]
        impact_height = self.frame.ground_plane_z
        
        if not np.isclose(launch_height, impact_height, atol=pos_tol):
            return False, {
                "reason": "Launch and impact at different heights",
                "launch_height": launch_height,
                "impact_height": impact_height
            }
        
        # Check velocity symmetry at equal times before/after apex
        t_before = self.t_apex * 0.5
        t_after = self.t_apex * 1.5
        
        # Must be within trajectory bounds
        if t_after > self.t_impact:
            return False, {
                "reason": "Cannot verify - trajectory ends before symmetry point",
                "t_after_needed": t_after,
                "t_impact": self.t_impact
            }
        
        state_before = self.state(t_before)
        state_after = self.state(t_after)
        
        # Horizontal velocities should match
        vx_deviation = abs(state_before.velocity[0] - state_after.velocity[0])
        vy_deviation = abs(state_before.velocity[1] - state_after.velocity[1])
        
        # Vertical velocities should be opposite
        vz_sum = abs(state_before.velocity[2] + state_after.velocity[2])
        
        # Heights should match
        z_deviation = abs(state_before.position[2] - state_after.position[2])
        
        metrics = {
            "vx_deviation": vx_deviation,
            "vy_deviation": vy_deviation,
            "vz_symmetry_error": vz_sum,
            "height_deviation": z_deviation,
            "vel_tolerance": vel_tol,
            "pos_tolerance": pos_tol,
        }
        
        # Check against proper tolerances
        vel_checks = [vx_deviation < vel_tol, vy_deviation < vel_tol, vz_sum < vel_tol]
        pos_checks = [z_deviation < pos_tol]
        
        is_symmetric = all(vel_checks) and all(pos_checks)
        metrics["symmetric"] = is_symmetric
        
        return is_symmetric, metrics


def create_default_system() -> ProjectileSystem:
    """Create a default projectile system for testing."""
    return ProjectileSystem(
        initial_position=np.array([0.0, 0.0, 1.0]),
        initial_velocity=np.array([35.0, 0.0, 35.0]),
        gravity=PhysicsConstants.EARTH_GRAVITY
    )

