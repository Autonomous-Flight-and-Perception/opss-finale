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

# UKF (CTRV — Constant Turn Rate Velocity, ported from MonteCarlo).
try:
    from ..state.ctrv_ukf_tracker import MultiObjectCTRVUKF, create_ctrv_ukf_tracker
    UKF_AVAILABLE = True
except Exception:
    UKF_AVAILABLE = False

# Particle filter (3D, ported from PF repo).
try:
    from ..state.pf_tracker import MultiObjectParticleFilter, create_pf_tracker
    PF_AVAILABLE = True
except Exception:
    PF_AVAILABLE = False

# Adaptive selector (KF + UKF + PF, switch by motion regime).
try:
    from ..state.adaptive import MultiObjectAdaptive, create_adaptive_tracker
    ADAPTIVE_AVAILABLE = True
except Exception:
    ADAPTIVE_AVAILABLE = False


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

    # Tracking. Multiple trackers can run in parallel against the same
    # detection stream (for filter-comparison demos); ``primary_tracker``
    # selects which one drives the cobot wire feed.
    max_track_distance: float = 100.0
    tracker_types: List[str] = field(
        default_factory=lambda: ["kalman"]
    )  # subset of {"kalman", "ukf"}
    primary_tracker: Optional[str] = None  # defaults to tracker_types[0]

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
        # Trackers run in parallel against the same detection stream.
        # Keyed by tracker name; values are tracker instances with an
        # .update(detections, timestamp) -> List[ObjectState] interface.
        self._trackers: Dict[str, object] = {}
        self._primary_tracker_name: Optional[str] = None
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

        # Latest outputs (for API access). Per-tracker dicts plus a
        # rolled-up "primary" view kept on the legacy single-tracker keys
        # for back-compat with the existing /states/latest, /fused/latest,
        # and /ws/states endpoints.
        self._lock = threading.Lock()
        self._latest_detections: List[Dict] = []
        self._latest_states_by_tracker: Dict[str, List[ObjectState]] = {}
        self._latest_fused_by_tracker: Dict[str, List[FusedState]] = {}
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

            # State trackers — one or more run in parallel against the
            # same detection stream. Each emits independent states/fused
            # output; ``primary`` is the one whose fused list is sent to
            # the cobot wire feed.
            requested = list(self.config.tracker_types) or ["kalman"]
            self._primary_tracker_name = self.config.primary_tracker or requested[0]

            for name in requested:
                if name == "kalman":
                    self._trackers["kalman"] = create_tracker(
                        max_distance=self.config.max_track_distance,
                        max_tracks=self.config.max_tracks,
                    )
                    print("[PIPELINE] Kalman tracker initialized (linear CV)")
                elif name == "ukf":
                    if not UKF_AVAILABLE:
                        print("[PIPELINE] WARNING: UKF requested but unavailable, skipping")
                        continue
                    self._trackers["ukf"] = create_ctrv_ukf_tracker(
                        max_distance=self.config.max_track_distance,
                        max_tracks=self.config.max_tracks,
                    )
                    print("[PIPELINE] UKF tracker initialized (CTRV nonlinear, ported from MonteCarlo)")
                elif name == "pf":
                    if not PF_AVAILABLE:
                        print("[PIPELINE] WARNING: PF requested but unavailable, skipping")
                        continue
                    self._trackers["pf"] = create_pf_tracker(
                        max_distance=self.config.max_track_distance,
                        max_tracks=self.config.max_tracks,
                    )
                    print("[PIPELINE] Particle filter initialized (300 particles, ported from PF repo)")
                elif name == "adaptive":
                    if not ADAPTIVE_AVAILABLE:
                        print("[PIPELINE] WARNING: adaptive requested but unavailable, skipping")
                        continue
                    self._trackers["adaptive"] = create_adaptive_tracker(
                        max_distance=self.config.max_track_distance,
                        max_tracks=self.config.max_tracks,
                    )
                    print("[PIPELINE] Adaptive tracker initialized (auto-switches KF / UKF / PF)")
                else:
                    print(f"[PIPELINE] WARNING: unknown tracker '{name}', skipping")

            if not self._trackers:
                print("[PIPELINE] No trackers initialized — falling back to Kalman")
                self._trackers["kalman"] = create_tracker(
                    max_distance=self.config.max_track_distance,
                    max_tracks=self.config.max_tracks,
                )
                self._primary_tracker_name = "kalman"

            if self._primary_tracker_name not in self._trackers:
                # Fallback: primary not available — use the first one we built.
                self._primary_tracker_name = next(iter(self._trackers))

            print(f"[PIPELINE] Active trackers: {list(self._trackers)}; "
                  f"primary (drives cobot): {self._primary_tracker_name}")

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

                # Run every active tracker against the same detection batch.
                # Each tracker maintains its own internal track state.
                states_by_tracker: Dict[str, List[ObjectState]] = {}
                for name, tracker in self._trackers.items():
                    try:
                        states_by_tracker[name] = tracker.update(detections_scaled, timestamp)
                    except Exception as e:
                        print(f"[PIPELINE] Tracker '{name}' error: {e}")
                        states_by_tracker[name] = []

                t2 = time.perf_counter()

                # Validate + fuse per-tracker so each filter has its own
                # complete output story (states, validations, fused).
                fused_by_tracker: Dict[str, List[FusedState]] = {}
                validation_by_tracker: Dict[str, List[ValidationResult]] = {}
                for name, states in states_by_tracker.items():
                    try:
                        vresults = self._validator.validate_states(states)
                    except Exception as e:
                        print(f"[PIPELINE] Validator error for '{name}': {e}")
                        vresults = []
                    try:
                        fused = self._fusion.fuse_batch(states, vresults)
                    except Exception as e:
                        print(f"[PIPELINE] Fusion error for '{name}': {e}")
                        fused = []
                    validation_by_tracker[name] = vresults
                    fused_by_tracker[name] = fused

                t4 = time.perf_counter()

                # Record per-stage latencies
                self.stats["latency_detect_ms"] = (t1 - t0) * 1000
                self.stats["latency_track_ms"] = (t2 - t1) * 1000
                self.stats["latency_validate_fuse_ms"] = (t4 - t2) * 1000
                self.stats["latency_total_ms"] = (t4 - t0) * 1000

                # Aggregate valid/invalid across all trackers
                valid_count = sum(
                    sum(1 for v in vs if v.is_valid)
                    for vs in validation_by_tracker.values()
                )
                invalid_count = sum(
                    sum(1 for v in vs if not v.is_valid)
                    for vs in validation_by_tracker.values()
                )

                # Draw all trackers' fused boxes on the full frame, each in
                # its own color so the comparison is visible on the stream.
                annotated_full = self._draw_annotations_multi(
                    frame_full, fused_by_tracker
                )

                # Update latest outputs
                primary = self._primary_tracker_name
                primary_fused = fused_by_tracker.get(primary, [])
                with self._lock:
                    self._latest_detections = detections_scaled
                    self._latest_states_by_tracker = states_by_tracker
                    self._latest_fused_by_tracker = fused_by_tracker
                    self._latest_frame = annotated_full

                # Cobot wire feed = primary tracker only. Schema unchanged.
                if self._broadcaster:
                    pipeline_info = {
                        "healthy": True,
                        "fps": float(self.stats.get("pipeline_fps", 0.0)),
                        "tracker": primary,
                        "frame": self._control_frame(primary_fused),
                    }
                    self._broadcaster.send_control_output(
                        primary_fused, pipeline_info
                    )

                # Diagnostic feedback loop — run on primary's states
                if self.config.enable_diagnostics:
                    self._run_diagnostic_loop(
                        states_by_tracker.get(primary, []), timestamp
                    )

                # Callbacks (back-compat: pass primary's data)
                if self._on_detection:
                    self._on_detection(detections_scaled)
                if self._on_state:
                    self._on_state(states_by_tracker.get(primary, []))
                if self._on_output:
                    self._on_output(primary_fused)

                # Update stats
                frame_count += 1
                self.stats["frames_processed"] += 1
                self.stats["detections_total"] += len(detections_scaled)
                self.stats["tracks_active"] = sum(
                    len(getattr(t, "trackers", {})) for t in self._trackers.values()
                )
                self.stats["tracks_per_filter"] = {
                    n: len(getattr(t, "trackers", {})) for n, t in self._trackers.items()
                }
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
                    per_filter = self.stats.get("tracks_per_filter", {})
                    per_filter_str = ", ".join(f"{n}={c}" for n, c in per_filter.items()) or "-"
                    total_validations = sum(len(vs) for vs in validation_by_tracker.values())
                    print(f"[PIPELINE] {self.stats['pipeline_fps']:.1f} FPS | "
                          f"Tracks: {self.stats['tracks_active']} ({per_filter_str}) | "
                          f"Valid: {valid_count}/{total_validations} | "
                          f"Latency: {lat:.1f}ms "
                          f"(det={self.stats.get('latency_detect_ms', 0):.1f} "
                          f"trk={self.stats.get('latency_track_ms', 0):.1f} "
                          f"val+fuse={self.stats.get('latency_validate_fuse_ms', 0):.1f})")

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

    # BGR colors per tracker. Chosen to be visually distinct on top of a
    # typical drone-vs-sky frame; one filter per color so a parallel demo
    # can be read at a glance.
    _TRACKER_COLORS = {
        "kalman":   (0, 165, 255),    # orange
        "ukf":      (255, 200, 0),    # cyan-ish blue
        "pf":       (180, 50, 255),   # magenta-ish (PF = particle cloud)
        "adaptive": (0, 255, 0),      # green (whichever filter is "active")
    }
    _TRACKER_LABEL = {"kalman": "K", "ukf": "U", "pf": "P", "adaptive": "A"}

    def _draw_annotations_multi(
        self,
        frame: np.ndarray,
        fused_by_tracker: Dict[str, List[FusedState]],
    ) -> np.ndarray:
        """
        Draw every active tracker's fused boxes on the frame, each in its
        own color so a comparison demo can read both at once.

        Per-box visuals:
          - Solid bbox in the tracker's color
          - Top-left tag ``[K]`` / ``[U]`` so the source filter is obvious
          - ID + confidence below the tag
          - Magenta arrow for velocity (scale depends on units)
          - Dimmed corners when validator rejected the state
        """
        import cv2
        annotated = frame.copy()

        for name, fused_states in fused_by_tracker.items():
            base_color = self._TRACKER_COLORS.get(name, (255, 255, 255))
            tag = self._TRACKER_LABEL.get(name, name[:1].upper() or "?")

            for state in fused_states:
                bbox = state.bbox
                if not bbox:
                    continue
                x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
                x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)

                # Validation modulates intensity, not hue, so each filter
                # stays colour-coded but you can still see when it's
                # producing an invalid state.
                if state.physics_plausible and state.physics_valid:
                    color = base_color
                    thick = 2
                else:
                    # 50% dim
                    color = tuple(int(c * 0.5) for c in base_color)
                    thick = 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                # For adaptive, surface which underlying filter was picked
                # this tick (set by adaptive.py via bbox["filter_used"]).
                used = state.bbox.get("filter_used")
                tag_full = f"{tag}->{used[0].upper()}" if (name == "adaptive" and used) else tag
                label = f"[{tag_full}] id={state.track_id} {state.confidence:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Velocity arrow from bbox center; meter-space vs pixel-space
                # scaled so both filters' arrows are visually legible.
                acx, acy = (x1 + x2) // 2, (y1 + y2) // 2
                if getattr(state, "units", "pixels") == "meters":
                    vx, vy = int(state.vx * 10), int(state.vy * 10)
                else:
                    vx, vy = int(state.vx * 0.1), int(state.vy * 0.1)
                if abs(vx) > 1 or abs(vy) > 1:
                    cv2.arrowedLine(annotated, (acx, acy),
                                    (acx + vx, acy + vy), color, 2)

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

        # All current trackers (Kalman, CTRV-UKF, PF, adaptive) emit
        # ``pixel_xy_metric_z`` — pixel xy + meter z (RealSense depth).
        return "pixel_xy_metric_z"

    # Public accessors
    def get_latest_detections(self) -> List[Dict]:
        with self._lock:
            return self._latest_detections.copy()

    def get_latest_states(self, tracker: Optional[str] = None) -> List[Dict]:
        """Tracker-keyed states. If ``tracker`` is None, returns the
        primary tracker's states (back-compat with single-tracker callers)."""
        with self._lock:
            name = tracker or self._primary_tracker_name
            states = self._latest_states_by_tracker.get(name, [])
            return [s.to_dict() for s in states]

    def get_latest_fused(self, tracker: Optional[str] = None) -> List[Dict]:
        """Tracker-keyed fused states. ``tracker=None`` returns primary."""
        with self._lock:
            name = tracker or self._primary_tracker_name
            fused = self._latest_fused_by_tracker.get(name, [])
            return [s.to_dict() for s in fused]

    def get_latest_states_all(self) -> Dict[str, List[Dict]]:
        """Per-tracker dict of states for the parallel-comparison view."""
        with self._lock:
            return {
                name: [s.to_dict() for s in states]
                for name, states in self._latest_states_by_tracker.items()
            }

    def get_latest_fused_all(self) -> Dict[str, List[Dict]]:
        """Per-tracker dict of fused states."""
        with self._lock:
            return {
                name: [s.to_dict() for s in fused]
                for name, fused in self._latest_fused_by_tracker.items()
            }

    def get_active_trackers(self) -> List[str]:
        return list(self._trackers.keys())

    def get_primary_tracker(self) -> Optional[str]:
        return self._primary_tracker_name

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
