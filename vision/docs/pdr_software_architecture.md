# Systems Architecture — Preliminary Design Review

## 1. Software System Overview

### 1.1 System Purpose and Operational Concept

The Optical Projectile Sensing System (OPSS) is a real-time, vision-based 3D object detection and tracking system designed to sense airborne targets—primarily small unmanned aerial systems (sUAS)—and provide continuous state estimates to a robotic interceptor. The system acquires RGB-D imagery from a depth camera, detects targets using a convolutional neural network, estimates full 6-DOF kinematic state (position and velocity in three dimensions) through sensor fusion, and broadcasts tracking data to a co-located robotic manipulator over a local inter-process communication channel.

The operational scenario is as follows. A single Intel RealSense D435 stereo depth camera is mounted in a fixed position overlooking an engagement volume. OPSS processes the camera's synchronized color and depth streams at 30 Hz. Each frame passes through a detection stage (YOLOv8 Nano), a state estimation stage (choice of linear Kalman filter or Unscented Kalman Filter with learned neural network corrections), a physics-based validation stage, and a weighted fusion stage before the resulting state estimate is transmitted to a MyCobot 280 robotic arm via Unix domain socket at 30 Hz. A web-based dashboard provides real-time MJPEG video, state telemetry, and system diagnostics over HTTP and WebSocket.

The system is deployed on an NVIDIA Jetson Orin Nano, an edge AI platform with a 6-core ARM Cortex-A78AE CPU and an Ampere-architecture GPU with 1024 CUDA cores. All processing—image capture, neural network inference, state estimation, and communication—executes on this single board, with no cloud or offboard compute dependency.

### 1.2 High-Level Software Architecture

OPSS follows a modular, pipeline-oriented architecture. Seven loosely coupled subsystems communicate through bounded in-process queues and shared data structures, orchestrated by a central pipeline controller. The architecture is organized as a directed dataflow graph:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OPSSPipeline (Orchestrator)                     │
│                                                                         │
│  ┌───────────┐    ┌────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  Vision    │───▶│   State    │───▶│   Physics    │───▶│  Fusion   │  │
│  │  (Capture  │    │ Estimation │    │  Validation  │    │   (B2₃)   │  │
│  │  + Detect) │    │(Kalman/UKF)│    │              │    │           │  │
│  └───────────┘    └────────────┘    └──────────────┘    └─────┬─────┘  │
│        │                                                       │        │
│        │                                                       ▼        │
│        │                                                ┌───────────┐   │
│        │                                                │   Cobot   │   │
│        │                                                │Broadcaster│   │
│        │                                                └───────────┘   │
│        │                                                                │
│        ▼                                                                │
│  ┌───────────┐                                                          │
│  │    Web     │  (MJPEG stream, REST API, WebSocket)                   │
│  │ Dashboard  │                                                         │
│  └───────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────┘

External Interfaces:
  ← RealSense D435 (USB 3.0, 30 Hz RGB-D)
  → MyCobot 280 (Unix domain socket, 30 Hz JSON)
  → Browser client (HTTP/WS, on-demand)
```

Each subsystem is implemented as a Python package under `opss/`, with a factory function that constructs the subsystem from a shared `PipelineConfig` dataclass. This factory pattern allows subsystems to be instantiated, tested, and swapped independently.

### 1.3 Subsystem Summary

The system comprises the following seven subsystems, each mapping to a distinct Python package:

| Subsystem | Package | Primary Responsibility |
|-----------|---------|----------------------|
| **Vision** | `opss.vision` | Camera capture (dual-resolution), YOLOv8 object detection, optional Edge TPU inference |
| **State Estimation** | `opss.state` | Multi-object tracking via linear Kalman filter or UKF with neural network acceleration correction |
| **Physics Validation** | `opss.physics` | Kinematic plausibility checks (velocity, acceleration bounds) and physics-based trajectory prediction |
| **Fusion** | `opss.fusion` | Weighted combination of tracker and physics-validated states into a single estimate per target |
| **Cobot Communication** | `opss.cobot` | Non-blocking Unix socket broadcast of fused states to the robotic manipulator |
| **Web Interface** | `opss.web` | FastAPI REST/WebSocket server with MJPEG video stream for monitoring and control |
| **Simulation** | `opss.sim` | Synthetic camera model, projection pipeline, and observation noise model for offline training and validation |

An eighth package, `opss.pipeline`, serves as the orchestration layer. It owns the main processing loop, threading model, and inter-subsystem data routing.

### 1.4 Design Drivers and Constraints

The following constraints shaped the architectural decisions throughout the system:

**Real-time throughput.** The system must sustain 30 Hz end-to-end processing (capture to cobot output) on a single Jetson Orin Nano. This constraint drives the choice of YOLOv8 Nano (the smallest variant, 6.5 MB) over larger detection models, the use of dual-resolution capture (full 1280×720 for depth, downscaled 640×360 for inference), and the use of a 611-parameter neural network for state estimation correction rather than a larger learned dynamics model.

**Edge deployment with no offboard dependency.** All computation executes on the Jetson. There is no network round-trip to a cloud service. This eliminates latency variability but imposes strict memory and compute budgets. The system operates within approximately 8 GB of unified CPU/GPU memory.

**Sensor-driven state estimation.** The RealSense D435 provides monocular RGB at 1280×720 and active infrared stereo depth at the same resolution. Depth accuracy degrades with range (the D435 specification quotes ±2% error at 2 m, growing quadratically). The tracking pipeline must be robust to depth dropout (stereo matching failure), depth noise that scales with range, and detection jitter from YOLO bounding box regression.

**Platform library incompatibility.** The Jetson's system-installed `libcudnn.so.8` has a version mismatch with the installed PyTorch build, causing a segmentation fault on `import torch`. This constraint is fundamental: any component that executes at runtime or during training must function without importing PyTorch. The neural network (611 parameters) is trained and evaluated in pure NumPy with a hand-written forward and backward pass. Inference uses `.weights.npz` files loaded by NumPy, not `.pt` checkpoints.

**Tracker-agnostic interface.** The pipeline supports two interchangeable tracking backends—a baseline linear Kalman filter and the UKF-NN—behind a common interface (`update(detections, timestamp) → List[ObjectState]`). This allows performance comparison and graceful fallback.

### 1.5 System Boundary and External Interfaces

The OPSS software boundary encompasses all processing from raw frame acquisition to fused state output. The system interfaces with three external entities:

**Intel RealSense D435 (input).** Connected via USB 3.0. The `RealSenseCamera` class wraps `pyrealsense2` and runs a background capture thread that enqueues synchronized color and depth frames. The system consumes aligned 1280×720 RGB and 1280×720 depth (16-bit, millimeter-scale) at 30 Hz.

**MyCobot 280 (output).** Connected via a Unix domain socket at `/tmp/opss_cobot.sock`. The `UnixSocketBroadcaster` sends JSON-encoded `FusedState` objects at up to 30 Hz. The protocol is unidirectional (OPSS → robot), with an optional `CobotStateReceiver` for feedback. Message format:

```json
{
  "x": 1.23, "y": 4.56, "z": 2.78,
  "vx": 0.5, "vy": -0.3, "vz": 0.1,
  "confidence": 0.92,
  "predicted_position": [1.25, 4.54, 2.77],
  "time_to_intercept": 0.83
}
```

**Web client (monitoring).** A FastAPI server exposes REST endpoints (`/pipeline/start`, `/stop`, `/status`, `/detections/latest`, `/states/latest`, `/fused/latest`), an MJPEG video feed (`/video_feed`), and a WebSocket channel (`/ws`) for real-time state streaming. The web interface is for operator monitoring and is not in the control-critical path.

### 1.6 Coordinate Frames and Conventions

The system uses two principal coordinate frames, and correct handling of their relationship is critical to tracking accuracy:

**World frame.** Right-handed, z-up. Gravity acts along −z: $\mathbf{g} = [0,\; 0,\; -9.81]^\top$ m/s². All state estimates, physics validation, and cobot output are expressed in this frame. The flight volume occupies approximately $x \in [0, 20]$ m, $y \in [0, 90]$ m, $z \in [0.5, 20]$ m.

**Camera frame.** Right-handed, following the pinhole convention: +X right, +Y down, +Z forward (into the scene). The transformation from camera to world frame is:

$$\mathbf{p}_\text{world} = \mathbf{R}_{\text{world} \leftarrow \text{cam}} \, \mathbf{p}_\text{cam} + \mathbf{t}_{\text{world} \leftarrow \text{cam}}$$

where $\mathbf{R}_{\text{world} \leftarrow \text{cam}} \in SO(3)$ and $\mathbf{t}_{\text{world} \leftarrow \text{cam}} \in \mathbb{R}^3$ are stored as extrinsic parameters in the tracker. Pixel-to-meter conversion uses the standard pinhole inverse projection:

$$x_\text{cam} = \frac{(u - c_x) \cdot d}{f_x}, \quad y_\text{cam} = \frac{(v - c_y) \cdot d}{f_y}, \quad z_\text{cam} = d$$

where $(u, v)$ are pixel coordinates, $d$ is depth in meters, and $(f_x, f_y, c_x, c_y)$ are the camera intrinsics.

### 1.7 Concurrency Model

The system employs a multi-threaded architecture with bounded queues to decouple producers from consumers:

| Thread | Role | Queue |
|--------|------|-------|
| **Camera capture** | Background daemon; reads USB frames into a frame queue | `frame_queue` (maxsize=2) |
| **Detection workers** | `ThreadPoolExecutor` (configurable count); run YOLO inference | `detection_queue` (maxsize=5) |
| **Pipeline main loop** | Daemon thread; drives the estimation→validation→fusion→output chain | `state_queue`, `output_queue` (maxsize=5–10) |
| **Cobot sender** | Rate-limited (30 Hz) non-blocking socket write | Inline with pipeline loop |
| **Web server** | Uvicorn async event loop; serves REST, MJPEG, WebSocket | Independent |

Queue bounds are deliberately small (2–10 items) to shed stale data under load rather than accumulate latency. If a consumer falls behind, the oldest frames are dropped.
