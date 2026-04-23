"""
OPSS Core Pipeline
Orchestrates the complete detection -> state estimation -> physics validation -> fusion -> control pipeline.

Architecture (from system engineering diagram):
    Camera Image
         |
      YOLO-V8
         |
    +----+----+
    |         |
Kalman    Embedded
Filter    State CNN*
    |         |
Physics   Physics
Engine 1  Engine 2
    |         |
 Valid?    Valid?
    |         |
    +----+----+
         |
       B2₃
         |
    +----+----+
    |         |
Output 1  Output 2
(Control) (Diagnostic)
    |         |
MyCobot   Compute
  280     Error

* Embedded State CNN slot is filled by the UKF (Unscented Kalman Filter)
"""
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Internal imports
from ..vision.camera import RealSenseCamera, CameraConfig, get_camera
from ..state.kalman import MultiObjectKalmanFilter, ObjectState, create_tracker
from ..physics.validator import PhysicsValidator, ValidationResult, create_validator
from ..fusion.b23 import B23Fusion, FusedState, create_fusion
from ..cobot.broadcaster import UnixSocketBroadcaster, get_broadcaster

# Optional YOLO import (torch may be unavailable on Jetson)
try:
    from ..vision.yolov8_inference import process_frame_bgr
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    process_frame_bgr = None

# Optional UKF import (the underlying tracker class is named MultiObjectUKFNN
# for historical reasons — it supports nn_model=None which gives a pure UKF).
try:
    from ..state.ukf_nn_tracker import MultiObjectUKFNN, create_ukf_nn_tracker
    UKF_AVAILABLE = True
except ImportError:
    UKF_AVAILABLE = False


@dataclass
class PipelineConfig:
    """Configuration for the OPSS pipeline"""
    # Camera
    capture_width: int = 1280
    capture_height: int = 720
    infer_width: int = 1280
    infer_height: int = 720
    fps: int = 30

    # Detection
    detection_threshold: float = 0.35
    max_detections: int = 20

    # Tracking
    max_track_distance: float = 100.0
    tracker_type: str = "kalman"  # "kalman" or "ukf"

    # Physics validation
    max_velocity: float = 50.0
    position_tolerance: float = 50.0

    # Fusion
    prediction_horizon: float = 0.5

    # Pipeline
    detection_workers: int = 2
    enable_cobot: bool = True
    enable_diagnostics: bool = True
    max_tracks: int = 50


class OPSSPipeline:
    """
    Main OPSS pipeline orchestrator.
    Connects all components and manages data flow.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize components
        self._camera: Optional[RealSenseCamera] = None
        self._tracker: Optional[MultiObjectKalmanFilter] = None
        self._validator: Optional[PhysicsValidator] = None
        self._fusion: Optional[B23Fusion] = None
        self._broadcaster: Optional[UnixSocketBroadcaster] = None

        # Threading
        self._running = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pipeline_thread: Optional[threading.Thread] = None

        # Queues for inter-stage communication
        self._detection_queue = queue.Queue(maxsize=5)
        self._state_queue = queue.Queue(maxsize=5)
        self._output_queue = queue.Queue(maxsize=10)

        # Callbacks
        self._on_detection: Optional[Callable] = None
        self._on_state: Optional[Callable] = None
        self._on_output: Optional[Callable] = None

        # Camera reconnection
        self._consecutive_none_frames = 0
        self._max_none_frames = 30  # ~1s at 30fps

        # Stats
        self.stats = {
            "frames_processed": 0,
            "detections_total": 0,
            "tracks_active": 0,
            "valid_states": 0,
            "invalid_states": 0,
            "pipeline_fps": 0.0,
            "last_update": time.time()
        }

        # Latest outputs (for API access)
        self._lock = threading.Lock()
        self._latest_detections: List[Dict] = []
        self._latest_states: List[ObjectState] = []
        self._latest_fused: List[FusedState] = []
        self._latest_frame: Optional[np.ndarray] = None

    def initialize(self) -> bool:
        """Initialize all pipeline components"""
        print("[PIPELINE] Initializing OPSS pipeline...")

        try:
            # Camera — create and start early so hardware intrinsics are available
            camera_config = CameraConfig(
                capture_width=self.config.capture_width,
                capture_height=self.config.capture_height,
                infer_width=self.config.infer_width,
                infer_height=self.config.infer_height,
                fps=self.config.fps
            )
            self._camera = get_camera(camera_config)

            # Start camera now so get_intrinsics() returns real values
            if not self._camera.start():
                print("[PIPELINE] Failed to start camera")
                return False
            print("[PIPELINE] Camera started")

            # Detection availability check
            if not YOLO_AVAILABLE:
                print("[PIPELINE] WARNING: YOLO detection unavailable (torch/ultralytics not loaded)")
                print("[PIPELINE]          Detection will return empty results")

            # State tracker (Kalman or UKF)
            if self.config.tracker_type == "ukf" and UKF_AVAILABLE:
                # Read hardware intrinsics (camera is already started)
                camera_intr = None
                hw_intr = self._camera.get_intrinsics()
                if hw_intr is not None:
                    from ..state.ukf_nn_tracker import CameraIntrinsics
                    from ..state.ukf_nn import config as ukf_cfg
                    camera_intr = CameraIntrinsics(
                        fx=hw_intr["fx"], fy=hw_intr["fy"],
                        cx=hw_intr["cx"], cy=hw_intr["cy"],
                    )
                    # Warn if hardware intrinsics differ from config defaults
                    tol = ukf_cfg.INTRINSIC_WARN_TOLERANCE
                    for name, hw_val, cfg_val in [
                        ("fx", hw_intr["fx"], ukf_cfg.CAMERA_FX),
                        ("fy", hw_intr["fy"], ukf_cfg.CAMERA_FY),
                        ("cx", hw_intr["cx"], ukf_cfg.CAMERA_CX),
                        ("cy", hw_intr["cy"], ukf_cfg.CAMERA_CY),
                    ]:
                        if cfg_val > 0 and abs(hw_val - cfg_val) / cfg_val > tol:
                            print(f"[PIPELINE] WARNING: Hardware {name}={hw_val:.1f} "
                                  f"differs from config {cfg_val:.1f} by "
                                  f"{abs(hw_val - cfg_val) / cfg_val * 100:.1f}%")

                # Pure UKF: model_path=None disables the NN residual term so
                # UKF3D runs with a gravity-only process model. The underlying
                # tracker class is MultiObjectUKFNN by historical naming, but
                # nothing about the algorithm is NN-specific in this mode.
                self._tracker = create_ukf_nn_tracker(
                    max_distance=self.config.max_track_distance,
                    model_path=None,
                    stats_path=None,
                    camera=camera_intr,
                    max_tracks=self.config.max_tracks,
                )
                print("[PIPELINE] UKF tracker initialized (pure UKF, no NN)")
            else:
                self._tracker = create_tracker(
                    max_distance=self.config.max_track_distance,
                    max_tracks=self.config.max_tracks,
                )
                print("[PIPELINE] Kalman tracker initialized")
                if self.config.tracker_type == "ukf" and not UKF_AVAILABLE:
                    print("[PIPELINE] WARNING: UKF requested but unavailable, using Kalman")

            # Physics validator
            self._validator = create_validator({
                "max_velocity": self.config.max_velocity,
                "position_tolerance": self.config.position_tolerance
            })
            print("[PIPELINE] Physics validator initialized")

            # Fusion
            self._fusion = create_fusion({
                "prediction_horizon": self.config.prediction_horizon
            })
            print("[PIPELINE] B2₃ fusion initialized")

            # Cobot broadcaster
            if self.config.enable_cobot:
                self._broadcaster = get_broadcaster()
                print("[PIPELINE] Cobot broadcaster initialized")

            # Thread pool for detection
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.detection_workers,
                thread_name_prefix="opss-detect"
            )

            print("[PIPELINE] Initialization complete")
            return True

        except Exception as e:
            print(f"[PIPELINE] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self) -> bool:
        """Start the pipeline"""
        if self._running:
            return True

        if not self.initialize():
            return False

        # Camera already started in initialize() (needed for intrinsics)

        self._running = True

        # Start pipeline thread
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop,
            daemon=True,
            name="opss-pipeline"
        )
        self._pipeline_thread.start()

        print("[PIPELINE] Started")
        return True

    def stop(self):
        """Stop the pipeline"""
        print("[PIPELINE] Stopping...")
        self._running = False

        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=2.0)

        if self._camera:
            self._camera.stop()

        if self._executor:
            self._executor.shutdown(wait=False)

        print("[PIPELINE] Stopped")

    def _pipeline_loop(self):
        """Main pipeline processing loop"""
        import cv2

        last_stats_time = time.time()
        frame_count = 0
        self._last_frame_time = time.time()

        while self._running:
            try:
                # Stall detection
                now_check = time.time()
                if now_check - self._last_frame_time > 2.0:
                    print(f"[PIPELINE] WARNING: No frame processed for "
                          f"{now_check - self._last_frame_time:.1f}s — possible stall")

                # Get frame at dual resolution
                result = self._camera.get_frame_dual_res(timeout=0.1)
                if result is None:
                    self._consecutive_none_frames += 1
                    if self._consecutive_none_frames >= self._max_none_frames:
                        print(f"[PIPELINE] {self._consecutive_none_frames} consecutive None frames "
                              f"— attempting camera reconnect")
                        try:
                            self._camera.stop()
                            time.sleep(0.5)
                            self._camera.start()
                        except Exception as e:
                            print(f"[PIPELINE] Camera reconnect failed: {e}")
                        self._consecutive_none_frames = 0
                    continue

                self._consecutive_none_frames = 0
                frame_full, frame_small, depth = result
                timestamp = time.time()
                self._last_frame_time = timestamp

                # --- Per-stage timing ---
                t0 = time.perf_counter()

                # Run detection on small frame
                if process_frame_bgr is not None:
                    annotated_small, detections = process_frame_bgr(
                        frame_small,
                        self.config.detection_threshold
                    )
                else:
                    annotated_small, detections = frame_small, []

                # Scale detections to full resolution
                scale_x = self.config.capture_width / self.config.infer_width
                scale_y = self.config.capture_height / self.config.infer_height
                detections_scaled = self._scale_detections(detections, scale_x, scale_y)

                # Add depth information if available
                if depth is not None:
                    detections_scaled = self._add_depth_info(detections_scaled, depth)

                t1 = time.perf_counter()

                # Update tracker
                try:
                    states = self._tracker.update(detections_scaled, timestamp)
                    if hasattr(self._tracker, 'last_timing'):
                        self.stats["tracker_substage"] = self._tracker.last_timing
                except Exception as e:
                    print(f"[PIPELINE] Tracker error: {e}")
                    states = []

                t2 = time.perf_counter()

                # Validate states with physics engine
                try:
                    validation_results = self._validator.validate_states(states)
                except Exception as e:
                    print(f"[PIPELINE] Validator error: {e}")
                    validation_results = []

                t3 = time.perf_counter()

                # Fuse estimates
                try:
                    fused_states = self._fusion.fuse_batch(states, validation_results)
                except Exception as e:
                    print(f"[PIPELINE] Fusion error: {e}")
                    fused_states = []

                t4 = time.perf_counter()

                # Record per-stage latencies
                self.stats["latency_detect_ms"] = (t1 - t0) * 1000
                self.stats["latency_track_ms"] = (t2 - t1) * 1000
                self.stats["latency_validate_ms"] = (t3 - t2) * 1000
                self.stats["latency_fuse_ms"] = (t4 - t3) * 1000
                self.stats["latency_total_ms"] = (t4 - t0) * 1000

                # Count valid/invalid
                valid_count = sum(1 for v in validation_results if v.is_valid)
                invalid_count = len(validation_results) - valid_count

                # Draw on full frame
                annotated_full = self._draw_annotations(frame_full, fused_states)

                # Update latest outputs
                with self._lock:
                    self._latest_detections = detections_scaled
                    self._latest_states = states
                    self._latest_fused = fused_states
                    self._latest_frame = annotated_full

                # Send canonical control output to the cobot (schema
                # ``opss.cobot.v1``; see opss/cobot/broadcaster.py module
                # docstring). This broadcast is unconditional per tick —
                # including when ``fused_states`` is empty — so the
                # datagram cadence acts as the pipeline heartbeat. The
                # cobot-side consumer distinguishes three states:
                #
                #   - healthy + targets:   targets non-empty, fresh ts
                #   - healthy + no targets: targets == [],    fresh ts
                #   - pipeline dead:       no datagram within staleness
                #                           window (~200 ms at 30 Hz)
                #
                # Raw YOLO detections are no longer broadcast on the
                # canonical channel; ``send_raw_detections`` is retained
                # on the broadcaster as a debug-only method.
                if self._broadcaster:
                    pipeline_info = {
                        "healthy": True,
                        "fps": float(self.stats.get("pipeline_fps", 0.0)),
                        "tracker": self.config.tracker_type,
                        "frame": self._control_frame(fused_states),
                    }
                    self._broadcaster.send_control_output(
                        fused_states, pipeline_info
                    )

                # Diagnostic feedback loop
                if self.config.enable_diagnostics:
                    self._run_diagnostic_loop(states, timestamp)

                # Callbacks
                if self._on_detection:
                    self._on_detection(detections_scaled)
                if self._on_state:
                    self._on_state(states)
                if self._on_output:
                    self._on_output(fused_states)

                # Update stats
                frame_count += 1
                self.stats["frames_processed"] += 1
                self.stats["detections_total"] += len(detections_scaled)
                self.stats["tracks_active"] = len(self._tracker.trackers)
                self.stats["valid_states"] += valid_count
                self.stats["invalid_states"] += invalid_count

                # Log stats periodically
                now = time.time()
                if now - last_stats_time >= 2.0:
                    elapsed = now - last_stats_time
                    self.stats["pipeline_fps"] = frame_count / elapsed
                    self.stats["last_update"] = now
                    frame_count = 0
                    last_stats_time = now

                    lat = self.stats.get("latency_total_ms", 0)
                    print(f"[PIPELINE] {self.stats['pipeline_fps']:.1f} FPS | "
                          f"Tracks: {self.stats['tracks_active']} | "
                          f"Valid: {valid_count}/{len(validation_results)} | "
                          f"Latency: {lat:.1f}ms "
                          f"(det={self.stats.get('latency_detect_ms', 0):.1f} "
                          f"trk={self.stats.get('latency_track_ms', 0):.1f} "
                          f"val={self.stats.get('latency_validate_ms', 0):.1f} "
                          f"fuse={self.stats.get('latency_fuse_ms', 0):.1f})")

            except Exception as e:
                print(f"[PIPELINE] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _scale_detections(self, detections: List[Dict], scale_x: float, scale_y: float) -> List[Dict]:
        """Scale detection bounding boxes to full resolution"""
        scaled = []
        for det in detections:
            bbox = det.get("bbox", {})
            x1 = int(bbox.get("x1", 0) * scale_x)
            y1 = int(bbox.get("y1", 0) * scale_y)
            x2 = int(bbox.get("x2", 0) * scale_x)
            y2 = int(bbox.get("y2", 0) * scale_y)

            scaled.append({
                "class": det.get("class", "person"),
                "confidence": det.get("confidence", 0.0),
                "corners": {
                    "top_left": {"x": x1, "y": y1},
                    "top_right": {"x": x2, "y": y1},
                    "bottom_left": {"x": x1, "y": y2},
                    "bottom_right": {"x": x2, "y": y2},
                },
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
            })
        return scaled

    def _add_depth_info(self, detections: List[Dict], depth: np.ndarray) -> List[Dict]:
        """Add depth information to detections from depth frame"""
        for det in detections:
            center = det.get("center", {})
            cx, cy = center.get("x", 0), center.get("y", 0)

            # Sample depth at center (with bounds checking)
            if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                # Get depth value (in mm typically, convert to meters)
                depth_val = float(depth[cy, cx]) / 1000.0
                det["depth"] = depth_val
            else:
                det["depth"] = 0.0

        return detections

    def _draw_annotations(self, frame: np.ndarray, fused_states: List[FusedState]) -> np.ndarray:
        """Draw fused state annotations on frame"""
        import cv2

        annotated = frame.copy()

        for state in fused_states:
            bbox = state.bbox
            if not bbox:
                continue

            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)

            # Color based on validation status.
            # After the FusedState contract fix, physics_plausible replaces the
            # misleading kalman_valid field. Semantics:
            #   physics_plausible -> validator said the state is physically plausible
            #                        (i.e. within velocity/continuity bounds for its frame)
            #   physics_valid     -> downstream physics_valid aggregate (unit-aware by frame)
            # Both true => green; partial => yellow; neither => red.
            if state.physics_plausible and state.physics_valid:
                color = (0, 255, 0)  # Green - fully valid
            elif state.physics_plausible or state.physics_valid:
                color = (0, 255, 255)  # Yellow - partially valid
            else:
                color = (0, 0, 255)  # Red - invalid

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw track ID and confidence
            label = f"ID:{state.track_id} {state.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw velocity vector (use bbox center for arrow origin,
            # since state.x/y may be in meters for the UKF tracker)
            arrow_cx = (x1 + x2) // 2
            arrow_cy = (y1 + y2) // 2
            if getattr(state, "units", "pixels") == "meters":
                # Meter-space velocity: scale for pixel visualization
                vx_scaled = int(state.vx * 10)
                vy_scaled = int(state.vy * 10)
            else:
                vx_scaled = int(state.vx * 0.1)
                vy_scaled = int(state.vy * 0.1)
            if abs(vx_scaled) > 1 or abs(vy_scaled) > 1:
                cv2.arrowedLine(annotated, (arrow_cx, arrow_cy),
                               (arrow_cx + vx_scaled, arrow_cy + vy_scaled),
                               (255, 0, 255), 2)

        return annotated

    def _run_diagnostic_loop(self, states: List[ObjectState], timestamp: float):
        """Run diagnostic feedback loop - compare predictions with observations"""
        for state in states:
            diagnostic = self._fusion.compare_prediction(
                state.track_id,
                (state.x, state.y, state.z),
                timestamp
            )
            if diagnostic and diagnostic.prediction_error is not None:
                if diagnostic.prediction_error > self.config.position_tolerance:
                    print(f"[DIAGNOSTIC] Track {state.track_id}: "
                          f"Prediction error {diagnostic.prediction_error:.1f}px")

    def _control_frame(self, fused_states: List[FusedState]) -> str:
        """
        Authoritative frame tag for the current control-output datagram.

        Every FusedState in a given tick shares a single frame (they all
        come from the same tracker), so when the list is non-empty we
        take the tag from the first state. When the list is empty we
        still need a non-None frame label (the cobot consumer uses it
        to interpret future datagrams), so we derive a default from the
        tracker type — including, for UKF, whether extrinsics were
        set (identity => camera_metric; non-identity => world_metric).

        Returns one of:
          "pixel" | "pixel_xy_metric_z" | "camera_metric" | "world_metric"
        """
        # Preferred source: a live fused state.
        if fused_states:
            tag = getattr(fused_states[0], "frame", None)
            if tag:
                return tag

        # Fallback by tracker type (empty-target heartbeat path).
        tracker_type = self.config.tracker_type
        if tracker_type == "ukf":
            if UKF_AVAILABLE and self._tracker is not None:
                # Identity extrinsics => world == camera; non-identity => world.
                R = getattr(self._tracker, "R_world_from_cam", None)
                t = getattr(self._tracker, "t_world_from_cam", None)
                if R is not None and t is not None:
                    is_identity = (
                        np.allclose(R, np.eye(3), atol=1e-12)
                        and np.allclose(t, 0.0, atol=1e-12)
                    )
                    return "camera_metric" if is_identity else "world_metric"
            # Before the tracker is constructed, UKF defaults to identity
            # extrinsics (= camera_metric). See opss.state.ukf_nn_tracker.
            return "camera_metric"
        # Kalman default.
        return "pixel_xy_metric_z"

    # Public accessors
    def get_latest_detections(self) -> List[Dict]:
        with self._lock:
            return self._latest_detections.copy()

    def get_latest_states(self) -> List[Dict]:
        with self._lock:
            return [s.to_dict() for s in self._latest_states]

    def get_latest_fused(self) -> List[Dict]:
        with self._lock:
            return [s.to_dict() for s in self._latest_fused]

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_stats(self) -> Dict:
        return self.stats.copy()

    def get_error_statistics(self) -> Dict:
        if self._fusion:
            return self._fusion.get_error_statistics()
        return {}

    # Callback setters
    def on_detection(self, callback: Callable):
        self._on_detection = callback

    def on_state(self, callback: Callable):
        self._on_state = callback

    def on_output(self, callback: Callable):
        self._on_output = callback


# Singleton pipeline
_pipeline: Optional[OPSSPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> OPSSPipeline:
    """Get or create the singleton pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = OPSSPipeline(config)
    return _pipeline
