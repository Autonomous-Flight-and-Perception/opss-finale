"""
RealSense Camera Module
Handles camera initialization, frame capture, and dual-resolution processing.
"""
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Generator
from dataclasses import dataclass

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARN] pyrealsense2 not available - camera functions disabled")


@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    capture_width: int = 1280
    capture_height: int = 720
    infer_width: int = 1280
    infer_height: int = 720
    fps: int = 30

    @property
    def scale_x(self) -> float:
        return self.capture_width / self.infer_width

    @property
    def scale_y(self) -> float:
        return self.capture_height / self.infer_height


class RealSenseCamera:
    """
    RealSense camera wrapper with dual-resolution support.
    Captures at high resolution, provides downscaled frames for inference.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed")

        self.config = config or CameraConfig()
        self._pipeline: Optional[rs.pipeline] = None
        self._lock = threading.Lock()
        self._running = False

        # Frame queues
        self._frame_queue = queue.Queue(maxsize=2)
        self._capture_thread: Optional[threading.Thread] = None

        # Hardware intrinsics (populated after start())
        self.intrinsics = None

        # Performance stats
        self.stats = {
            "capture_fps": 0.0,
            "frame_count": 0,
            "frames_dropped": 0,
            "last_time": time.time()
        }

    def start(self) -> bool:
        """Initialize and start the RealSense pipeline"""
        with self._lock:
            if self._pipeline is not None:
                return True

            try:
                pipeline = rs.pipeline()
                cfg = rs.config()

                # Enable streams at capture resolution
                cfg.enable_stream(
                    rs.stream.color,
                    self.config.capture_width,
                    self.config.capture_height,
                    rs.format.bgr8,
                    self.config.fps
                )
                cfg.enable_stream(
                    rs.stream.depth,
                    self.config.capture_width,
                    self.config.capture_height,
                    rs.format.z16,
                    self.config.fps
                )

                pipeline.start(cfg)
                self._pipeline = pipeline
                self._running = True

                # Read hardware intrinsics
                try:
                    profile = pipeline.get_active_profile()
                    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                    intr = color_stream.get_intrinsics()
                    self.intrinsics = {
                        "fx": intr.fx, "fy": intr.fy,
                        "cx": intr.ppx, "cy": intr.ppy,
                        "width": intr.width, "height": intr.height,
                    }
                    print(f"[CAMERA] Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
                          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
                except Exception as e:
                    print(f"[CAMERA] Could not read intrinsics: {e}")

                print(f"[CAMERA] Started at {self.config.capture_width}x{self.config.capture_height}")
                print(f"[CAMERA] Inference resolution: {self.config.infer_width}x{self.config.infer_height}")

                # Start capture thread
                self._capture_thread = threading.Thread(
                    target=self._capture_loop,
                    daemon=True,
                    name="camera-capture"
                )
                self._capture_thread.start()

                return True

            except Exception as e:
                print(f"[CAMERA] Failed to start: {e}")
                return False

    def stop(self):
        """Stop the camera pipeline"""
        self._running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None

        with self._lock:
            if self._pipeline:
                self._pipeline.stop()
                self._pipeline = None
                print("[CAMERA] Stopped")

    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        last_fps_time = time.time()

        while self._running and self._pipeline:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame:
                    continue

                # Convert to numpy
                color_bgr = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data()) if depth_frame else None

                # Update stats
                self.stats["frame_count"] += 1
                now = time.time()
                if now - last_fps_time >= 2.0:
                    elapsed = now - self.stats["last_time"]
                    self.stats["capture_fps"] = self.stats["frame_count"] / elapsed
                    self.stats["frame_count"] = 0
                    self.stats["last_time"] = now
                    last_fps_time = now

                # Put in queue (non-blocking, drop if full)
                try:
                    self._frame_queue.put_nowait((color_bgr, depth_raw))
                except queue.Full:
                    self.stats["frames_dropped"] += 1

            except Exception as e:
                if self._running:
                    print(f"[CAMERA] Capture error: {e}")
                time.sleep(0.01)

    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Get the latest frame (color, depth).
        Returns None if no frame available.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_frame_dual_res(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        Get frame at both resolutions: (full_res, inference_res, depth).
        Useful for dual-resolution detection pipeline.
        """
        import cv2

        result = self.get_frame(timeout)
        if result is None:
            return None

        color_full, depth = result

        # Downscale for inference
        color_small = cv2.resize(
            color_full,
            (self.config.infer_width, self.config.infer_height),
            interpolation=cv2.INTER_LINEAR
        )

        return color_full, color_small, depth

    def get_intrinsics(self) -> Optional[dict]:
        """Return hardware intrinsics read after start(), or None."""
        return self.intrinsics

    @property
    def is_running(self) -> bool:
        return self._running and self._pipeline is not None

    @property
    def resolution(self) -> dict:
        return {
            "capture": {"width": self.config.capture_width, "height": self.config.capture_height},
            "inference": {"width": self.config.infer_width, "height": self.config.infer_height}
        }


# Singleton instance
_camera: Optional[RealSenseCamera] = None
_camera_lock = threading.Lock()


def get_camera(config: Optional[CameraConfig] = None) -> RealSenseCamera:
    """Get or create the singleton camera instance"""
    global _camera

    with _camera_lock:
        if _camera is None:
            _camera = RealSenseCamera(config)
        return _camera
