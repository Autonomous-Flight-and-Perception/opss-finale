"""
Camera model for simulation: intrinsics + extrinsics.

Bundles a pinhole camera (focal length, principal point, image size) with
a rigid pose in the world frame (R_world_from_cam, t_world_from_cam).

FRAME CONVENTIONS:
    Camera frame: +X right, +Y down, +Z forward (standard pinhole).
    World frame:  z-up, gravity = [0, 0, -9.81] m/s².

    p_world = R_world_from_cam @ p_cam + t_world_from_cam
    p_cam   = R_world_from_cam.T @ (p_world - t_world_from_cam)
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class SimCamera:
    """Pinhole camera with known pose in world frame."""
    # Intrinsics
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    # Extrinsics (camera→world)
    R_world_from_cam: np.ndarray = None  # (3,3) rotation
    t_world_from_cam: np.ndarray = None  # (3,) camera origin in world

    def __post_init__(self):
        if self.R_world_from_cam is None:
            self.R_world_from_cam = np.eye(3)
        if self.t_world_from_cam is None:
            self.t_world_from_cam = np.zeros(3)
        self.R_world_from_cam = np.asarray(self.R_world_from_cam, dtype=np.float64)
        self.t_world_from_cam = np.asarray(self.t_world_from_cam, dtype=np.float64)

    @property
    def R_cam_from_world(self):
        """Inverse rotation: world→camera."""
        return self.R_world_from_cam.T

    def is_in_image(self, u, v):
        """Check if pixel coordinates are within image bounds."""
        return 0 <= u < self.width and 0 <= v < self.height


def look_at_camera(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = None,
    **intrinsic_kwargs,
) -> SimCamera:
    """
    Construct a SimCamera at `position` looking toward `target`.

    Uses the standard look-at construction:
        cam_z (forward) = normalize(target - position)
        cam_x (right)   = normalize(cam_z × up_world)
        cam_y (down)    = cam_x × cam_z   (note: NOT cam_z × cam_x)

    The resulting R_world_from_cam has columns = camera axes in world coords.

    Args:
        position: (3,) camera position in world frame
        target:   (3,) point the camera looks at in world frame
        up:       (3,) world up direction (default: [0, 0, 1] for z-up)
        **intrinsic_kwargs: passed to SimCamera (fx, fy, cx, cy, width, height)
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])

    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Camera optical axis (+Z_cam) in world coords
    forward = target - position
    forward = forward / np.linalg.norm(forward)

    # Camera right axis (+X_cam) in world coords
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError("Camera forward direction is parallel to up vector")
    right = right / right_norm

    # Camera down axis (+Y_cam) in world coords
    # Right-handed: X × Y = Z, so right × down = forward ⟹ down = forward × right
    down = np.cross(forward, right)

    # R_world_from_cam: columns are camera axes in world coords
    R_world_from_cam = np.column_stack([right, down, forward])

    return SimCamera(
        R_world_from_cam=R_world_from_cam,
        t_world_from_cam=position.copy(),
        **intrinsic_kwargs,
    )
