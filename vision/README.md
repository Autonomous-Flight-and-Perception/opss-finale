# OPSS - Optical Projectile Sensing System

Unified computer vision and physics validation pipeline for real-time object detection, tracking, and state estimation.

## Architecture

```
                    ┌─────────────────┐
                    │  Camera Image   │  RealSense (720p)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    YOLO-V8      │  Detection (640x360)
                    └────────┬────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
   ┌───────▼───────┐                   ┌───────▼───────┐
   │ Kalman Filter │                   │ Embedded CNN  │  (future)
   │               │                   │               │
   └───────┬───────┘                   └───────┬───────┘
           │                                   │
   ┌───────▼───────┐                   ┌───────▼───────┐
   │Static Physics │                   │Static Physics │
   │   Engine 1    │                   │   Engine 2    │
   └───────┬───────┘                   └───────┬───────┘
           │                                   │
     ┌─────▼─────┐                       ┌─────▼─────┐
     │ Physics   │                       │ Physics   │
     │  Valid?   │                       │  Valid?   │
     └─────┬─────┘                       └─────┬─────┘
           │                                   │
           └─────────────┬─────────────────────┘
                         │
                  ┌──────▼──────┐
                  │    B2₃      │  Fusion
                  └──────┬──────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼────────┐                ┌───────▼────────┐
│   Output 1:    │                │   Output 2:    │
│State Estimation│                │  Diagnostic    │
│  for Controls  │                │   Feedback     │
└───────┬────────┘                └────────────────┘
        │
┌───────▼────────┐
│  MyCobot280    │  Unix Socket
└────────────────┘
```

## Hardware Requirements

- **Camera**: Intel RealSense D435 (or D415/D455)
- **Compute**: NVIDIA Jetson Orin Nano (recommended) or x86 Linux with GPU
- **Robot**: MyCobot 280 (optional, for control output)
- **USB**: USB 3.0 port (blue connector) for RealSense

## Quick Start

### One-Line Deploy

```bash
git clone https://github.com/Autonomous-Flight-and-Perception/OPSS.git
cd OPSS
./deploy.sh
```

This automatically:
1. Detects your platform (Jetson/Raspberry Pi/Linux)
2. Installs system dependencies
3. Sets up RealSense SDK
4. Creates Python virtual environment
5. Installs all Python packages
6. Downloads YOLOv8 model
7. Starts the web server

### Deploy Script Options

| Command | Description |
|---------|-------------|
| `./deploy.sh` | Full install + run |
| `./deploy.sh --install` | Install dependencies only |
| `./deploy.sh --run` | Run OPSS (assumes installed) |
| `./deploy.sh --docker` | Run via Docker |
| `./deploy.sh --service` | Install as systemd service (auto-start on boot) |
| `./deploy.sh --verify` | Verify installation |

### Manual Installation

```bash
cd OPSS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Note: pyrealsense2 must be installed system-wide on Jetson
# Link it into venv:
ln -s /usr/lib/python3/dist-packages/pyrealsense2* venv/lib/python3.*/site-packages/
```

### Run

```bash
# Start with web interface (default)
python main.py

# Custom port
python main.py --port 8080

# Pipeline only (no web server)
python main.py --no-web
```

### Access

- **Dashboard**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs
- **Video Stream**: http://localhost:8000/stream/color

## Docker Deployment

```bash
# Build and run
./deploy.sh --docker

# Or manually:
docker compose up --build
```

For Jetson with GPU support, edit `docker-compose.yml`:
```yaml
services:
  opss:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

## Systemd Service (Auto-Start)

```bash
# Install service
./deploy.sh --service

# Control
sudo systemctl start opss
sudo systemctl stop opss
sudo systemctl status opss

# View logs
journalctl -u opss -f
```

## Project Structure

```
OPSS/
├── opss/                       # Main package
│   ├── vision/                 # Camera + Detection
│   │   ├── camera.py           # RealSense dual-resolution capture
│   │   ├── yolov8_inference.py # YOLOv8 person detection
│   │   └── models/             # ML model files
│   │
│   ├── state/                  # State Estimation
│   │   └── kalman.py           # Multi-object Kalman tracker
│   │
│   ├── physics/                # Physics Validation
│   │   ├── validator.py        # State validation against physics
│   │   └── sim/                # Physics engine (from b2)
│   │       ├── physics_core.py # Analytical trajectories
│   │       ├── integrators.py  # Numerical integration (Euler, RK4)
│   │       ├── forces.py       # Drag, wind models
│   │       ├── forces_spin.py  # Magnus effect
│   │       └── engine.py       # Simulation orchestrator
│   │
│   ├── fusion/                 # B2₃ Fusion
│   │   └── b23.py              # State fusion + diagnostic loop
│   │
│   ├── pipeline/               # Core Pipeline
│   │   └── core.py             # Main orchestrator
│   │
│   ├── cobot/                  # Robot Communication
│   │   └── broadcaster.py      # Unix socket to MyCobot280
│   │
│   └── web/                    # Web Interface
│       ├── app.py              # FastAPI application
│       └── static/             # Dashboard HTML/JS/CSS
│
├── tests/                      # Test suite
├── configs/                    # Configuration files
├── deploy.sh                   # Automated deployment script
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Docker orchestration
└── README.md
```

## API Endpoints

### Pipeline Control
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pipeline/start` | POST | Start the full pipeline |
| `/pipeline/stop` | POST | Stop the pipeline |
| `/pipeline/status` | GET | Get status, stats, and config |

### Data Access
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detections/latest` | GET | Raw YOLO detections |
| `/states/latest` | GET | Kalman filter tracked states |
| `/fused/latest` | GET | B2₃ fused estimates |
| `/diagnostics` | GET | Prediction error statistics |

### Streaming
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/color` | GET | MJPEG video stream with annotations |
| `/ws/states` | WebSocket | Real-time state updates (30 Hz) |
| `/ws/detections` | WebSocket | Real-time detections (legacy) |

### Legacy (Vh Compatibility)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/camera/start` | POST | Alias for `/pipeline/start` |
| `/camera/stop` | POST | Alias for `/pipeline/stop` |
| `/camera/status` | GET | Camera/pipeline status |

## Components

### Vision (opss.vision)
- **Camera**: RealSense capture at 720p, downscaled to 640x360 for inference
- **Detector**: YOLOv8 nano for person detection
- **Dual-Resolution**: High-quality streaming + fast inference

### State Estimation (opss.state)

**Basic Kalman Filter** (`kalman.py`):
- Multi-object tracking with velocity estimation
- Simple constant-velocity model
- Fast, suitable for real-time

**UKF** (`ukf_nn/ukf.py::UKF3D`) - Unscented Kalman Filter:
- 3D meter-space state vector with depth from RealSense
- Sigma-point propagation, gravity in process model
- Camera-frame or world-frame, depending on extrinsics
- Selectable via `--tracker ukf`. (The `ukf_nn` package name is
  historical — the same `UKF3D` class supports an optional NN residual
  correction term, but the user-facing CLI exposes only the pure UKF.)

**State Vector**: [x, y, z, vx, vy, vz]
**Track Management**: Automatic track creation, confirmation, and deletion

### Physics (opss.physics)
- **Validator**: Checks states against physics constraints
- **Engine**: Analytical (closed-form) and numerical (RK4) trajectory prediction
- **Forces**: Gravity, quadratic drag, wind, Magnus effect (spin)
- **Validation**: Position/velocity tolerance checks, acceleration bounds

### Fusion (opss.fusion)
- **B2₃**: Weighted combination of Kalman + physics estimates
- **Adaptive Weights**: Adjusts based on validation results
- **Diagnostics**: Stores predictions, compares to future observations, computes error

### Cobot (opss.cobot)
- **Broadcaster**: Unix socket at `/tmp/opss_cobot.sock`
- **Protocol**: JSON datagrams with position, velocity, confidence
- **Non-Blocking**: Fire-and-forget, doesn't stall pipeline

## Configuration

Edit `PipelineConfig` in `opss/pipeline/core.py`:

```python
@dataclass
class PipelineConfig:
    # Camera
    capture_width: int = 1280      # Capture resolution (720p)
    capture_height: int = 720
    infer_width: int = 640         # Inference resolution
    infer_height: int = 360
    fps: int = 30

    # Detection
    detection_threshold: float = 0.55
    max_detections: int = 20

    # Tracking
    max_track_distance: float = 100.0  # Max pixels for association
    tracker_type: str = "kalman"       # "kalman" or "ukf"

    # Physics validation
    max_velocity: float = 50.0         # m/s
    position_tolerance: float = 50.0   # pixels

    # Fusion
    prediction_horizon: float = 0.5    # seconds

    # Pipeline
    detection_workers: int = 2
    enable_cobot: bool = True
    enable_diagnostics: bool = True
```

## Cobot Integration

The pipeline broadcasts a canonical control payload (schema
`opss.cobot.v1`) to the MyCobot280 over a Unix SOCK_DGRAM socket at
`/tmp/opss_cobot.sock`. The canonical producer is
`UnixSocketBroadcaster.send_control_output`; it is called unconditionally
every pipeline tick (≤30 Hz) so the datagram cadence doubles as a
heartbeat.

### Message format (`opss.cobot.v1`)

```json
{
  "schema": "opss.cobot.v1",
  "timestamp_ns": 1234567890123456789,
  "pipeline": {
    "healthy": true,
    "fps": 29.7,
    "tracker": "ukf",
    "frame": "camera_metric"
  },
  "targets": [
    {
      "track_id": 1,
      "timestamp": 1234567890.123,
      "position": {"x": 0.42, "y": -0.15, "z": 2.10},
      "velocity": {"vx": 1.20, "vy": 0.00, "vz": -0.30},
      "units": "meters",
      "frame": "camera_metric",
      "confidence": 0.85,
      "valid": true,
      "validation": {
        "physics_plausible": true,
        "physics_valid": true,
        "tracker_valid": true
      },
      "predicted_position": {"x": 1.02, "y": -0.15, "z": 1.95},
      "time_to_intercept": 0.5,
      "bbox": {"x1": 640, "y1": 360, "x2": 780, "y2": 520}
    }
  ],
  "count": 1
}
```

Targets are sorted confidence-descending; `targets[0]` is the canonical
primary target for single-target actuators. An empty `targets` list is a
valid, meaningful payload meaning **"pipeline healthy, no targets right
now"**. The three pipeline states a consumer can distinguish are:

- healthy + targets:    `targets` non-empty, fresh `timestamp_ns`
- healthy + no targets: `targets: []`, fresh `timestamp_ns`
- pipeline dead:        no datagram within the staleness window
  (recommended: ~200 ms at 30 Hz nominal cadence)

### Units and frames

`pipeline.frame` (and each target's `frame`) is the authoritative
frame tag and MUST be read before interpreting any position/velocity
field. Legal values:

- `pixel`              — image-plane pixels, no depth
- `pixel_xy_metric_z`  — x,y in pixels; z in meters (Kalman hybrid)
- `camera_metric`      — meters, camera frame (identity extrinsics)
- `world_metric`       — meters, world frame (gravity along −z)

`bbox` is always pixel-space at capture resolution; it is for
visualization only and is NOT part of the kinematic control state.

### Reference listener

```python
import socket, json, time

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind("/tmp/opss_cobot.sock")
sock.settimeout(0.2)  # staleness window

last_seen = None
while True:
    try:
        data, _ = sock.recvfrom(8192)
        msg = json.loads(data)
        assert msg["schema"] == "opss.cobot.v1"
        last_seen = time.time_ns()

        frame = msg["pipeline"]["frame"]
        if msg["count"] == 0:
            # Pipeline healthy; no targets to act on this tick.
            continue
        primary = msg["targets"][0]  # highest-confidence target
        if not primary["valid"]:
            continue
        # primary["position"] is in `frame` with units primary["units"]
        #   kalman              => pixel_xy_metric_z (x,y px; z m)
        #   ukf (no extrinsics) => camera_metric    (meters)
        #   ukf (extrinsics)    => world_metric     (meters, gravity -z)
        # ... drive actuator ...
    except socket.timeout:
        # No datagram within staleness window => pipeline dead.
        pass
```

> The pre-canonicalization debug broadcast (raw YOLO detections via
> `send_raw_detections`) is retained as a debug-only method on the
> broadcaster but is NOT called by the live pipeline.

## Switching trackers

Two trackers are exposed via CLI:

```bash
python3 main.py --tracker kalman   # default — linear KF, pixel-space + RealSense depth
python3 main.py --tracker ukf      # Unscented KF, meter-space, gravity in process model
```

Or programmatically:

```python
from opss import OPSSPipeline, PipelineConfig

pipeline = OPSSPipeline(PipelineConfig(tracker_type="ukf"))
pipeline.start()
```

Both feed the same downstream physics validator and B2₃ fusion stages,
so swapping `--tracker` changes only the state-estimation step. Useful
for filter-comparison demos: same scene, different estimator.

> The underlying `UKF3D` class also supports an optional NN residual
> correction term (used historically as "UKF-NN"), but that path is
> not exposed through the CLI in this finale build. The pure UKF is
> what `--tracker ukf` selects.

## Troubleshooting

### Camera not detected
```bash
# Check USB connection
lsusb | grep Intel

# Check RealSense
rs-enumerate-devices

# Ensure user is in plugdev group
sudo usermod -aG plugdev $USER
# Then re-login
```

### PyTorch not using GPU (Jetson)
```bash
# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# If false, reinstall PyTorch for Jetson
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import errors
```bash
# Verify installation
./deploy.sh --verify

# Re-link system packages
ln -sf /usr/lib/python3/dist-packages/pyrealsense2* venv/lib/python3.*/site-packages/
```

### Low FPS
- Ensure running on GPU (not CPU)
- Check `nvidia-smi` or `jtop` for GPU usage
- Reduce `capture_width`/`capture_height` if needed
- Increase `detection_threshold` to reduce detections

## Performance

Expected performance on Jetson Orin Nano:

| Metric | Value |
|--------|-------|
| Capture FPS | ~30 |
| Detection FPS | ~15-20 |
| Pipeline Latency | ~50ms |
| Memory Usage | ~800MB |

## License

MIT License - See LICENSE file

## Authors

Autonomous Flight and Perception (AFP)

## Repository

https://github.com/Autonomous-Flight-and-Perception/OPSS

---

*This project combines the former Vh (Vision Hub) and b2 (Physics Engine) repositories into a unified system.*
