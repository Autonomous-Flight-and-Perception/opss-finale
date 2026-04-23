"""
Simulation utilities for OPSS.

Provides synthetic observation generation: world-frame truth → detection dicts
that can be fed to the UKF-NN tracker for testing and training data generation.
"""
from .camera import SimCamera
from .projection import world_to_detection, world_to_camera, camera_to_pixel
from .observation import ObservationNoise

__all__ = [
    "SimCamera",
    "ObservationNoise",
    "world_to_detection",
    "world_to_camera",
    "camera_to_pixel",
]
