"""
World → camera → pixel projection for synthetic observation generation.

Implements the forward projection pipeline:
    1. world_to_camera: p_cam = R_cam_from_world @ (p_world - t_world)
    2. camera_to_pixel: (u, v) = (fx * x/z + cx, fy * y/z + cy), depth = z
    3. world_to_detection: full pipeline producing a detection dict

The inverse (pixel → camera → world) is handled by the tracker's
CameraIntrinsics.pixel_to_meters() + R_world_from_cam transform.
"""
import numpy as np
from .camera import SimCamera
from .observation import ObservationNoise


def world_to_camera(pos_world, camera):
    """
    Transform world-frame position to camera-frame.

    Args:
        pos_world: (3,) or (N, 3) positions in world frame
        camera: SimCamera with extrinsics

    Returns:
        (3,) or (N, 3) positions in camera frame
    """
    pos_world = np.asarray(pos_world)
    return (camera.R_cam_from_world @ (pos_world - camera.t_world_from_cam).T).T


def camera_to_pixel(pos_camera, camera):
    """
    Project camera-frame position to pixel coordinates.

    Args:
        pos_camera: (3,) position in camera frame [x_cam, y_cam, z_cam]
        camera: SimCamera with intrinsics

    Returns:
        (u, v, depth) where depth = z_cam (along optical axis, NOT range),
        or None if z_cam <= 0 (behind camera).
    """
    z = pos_camera[2]
    if z <= 0:
        return None

    u = camera.fx * pos_camera[0] / z + camera.cx
    v = camera.fy * pos_camera[1] / z + camera.cy
    return u, v, z


def world_to_detection(
    pos_world,
    camera,
    noise=None,
    rng=None,
    bbox_half_size=20,
    confidence=0.9,
):
    """
    Project a world-frame 3D position to a detection dict.

    Full pipeline: world → camera → pixel → noise → detection dict.

    Args:
        pos_world: (3,) world-frame position [x, y, z]
        camera: SimCamera with intrinsics and extrinsics
        noise: ObservationNoise config (None = no noise)
        rng: numpy random Generator (required if noise is not None)
        bbox_half_size: half-width of synthetic bounding box in pixels
        confidence: detection confidence value

    Returns:
        Detection dict with keys {center, depth, bbox, confidence},
        or None if the point is behind the camera or outside the image.
    """
    pos_world = np.asarray(pos_world, dtype=np.float64)

    # World → camera
    p_cam = world_to_camera(pos_world, camera)

    # Camera → pixel
    result = camera_to_pixel(p_cam, camera)
    if result is None:
        return None
    u, v, depth = result

    # Check if clean projection is in image (reject before adding noise)
    if not camera.is_in_image(u, v):
        return None

    # Apply noise
    if noise is not None:
        if rng is None:
            raise ValueError("rng required when noise is not None")
        u, v, depth = noise.apply(u, v, depth, rng)

    # Clamp pixels to image bounds (noisy pixels can drift outside)
    u = max(0, min(u, camera.width - 1))
    v = max(0, min(v, camera.height - 1))

    # Build detection dict
    hs = bbox_half_size
    x1 = int(max(0, round(u) - hs))
    y1 = int(max(0, round(v) - hs))
    x2 = int(min(camera.width - 1, round(u) + hs))
    y2 = int(min(camera.height - 1, round(v) + hs))

    return {
        "center": {"x": float(u), "y": float(v)},
        "depth": float(depth),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": confidence,
        "class": "drone",
    }
