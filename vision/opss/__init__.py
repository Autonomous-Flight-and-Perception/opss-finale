"""
OPSS - Optical Projectile Sensing System

A unified computer vision and physics validation system for real-time
object detection, tracking, and state estimation.

Components:
- vision: Camera capture and YOLO-V8 detection
- state: Kalman filter for object tracking
- physics: Physics engine for trajectory validation
- fusion: B2₃ fusion module for combining estimates
- pipeline: Core orchestration
- cobot: Robot communication
- web: FastAPI web interface
"""

__version__ = "2.0.0"
__author__ = "Autonomous Flight and Perception"

from .pipeline.core import OPSSPipeline, PipelineConfig, get_pipeline

__all__ = [
    "OPSSPipeline",
    "PipelineConfig",
    "get_pipeline",
    "__version__"
]
