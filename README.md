# opss-finale

End-to-end, self-contained snapshot of the OPSS drone-tracking cobot-arm
demo as of 2026-04-22. Everything needed to reproduce the live pipeline
is in this repo — the source repos that fed into it are preserved under
`legacy/` for reference.

## Topology

```
  ┌──────────────────────── Jetson host ────────────────────────┐
  │                                                              │
  │   RealSense D415/D435 ──► vision/ (host-native Python)       │
  │                           ├─ YOLOv8 (user_best.pt, CUDA)     │
  │                           ├─ Kalman tracker / UKF-NN         │
  │                           ├─ physics validator + B2₃ fusion  │
  │                           └─ FastAPI @ :8000                 │
  │                              ├─ /                 (dashboard)│
  │                              ├─ /stream/color     (MJPEG)    │
  │                              ├─ /ws/states        (30 Hz WS) │
  │                              └─ /pipeline/{start,stop,status}│
  │                                                              │
  │   /dev/ttyACM0 ◄── MyCobot 280                               │
  │         ▲                                                    │
  │         │  pymycobot                                         │
  │         │                                                    │
  │   ┌─────┴─────── opss-cobot (docker) ────────────────────┐   │
  │   │   cobot/run2.py                                      │   │
  │   │     └─ websocket → ws://vh-vision:8000/ws/states     │   │
  │   │        (vh-vision → host-gateway via extra_hosts)    │   │
  │   └──────────────────────────────────────────────────────┘   │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

Vision runs on the host (native CUDA/RealSense); the cobot controller is
a thin Python container that reaches the host-side vision service over
WebSocket. One USB cable to the arm, one USB cable to the camera.

## Repo layout

```
opss-finale/
├── vision/     Host-side vision service (OPSS). RealSense + YOLOv8 + tracking
│               + physics validation + fusion + FastAPI dashboard on :8000.
│               Runs on the Jetson host (not containerized; see below).
│   └── opss/   Package: vision, state, physics, fusion, pipeline, web, cobot.
├── cobot/      MyCobot 280 controller. Containerized. WebSocket client that
│               drives arm joints from /ws/states. Entrypoint: run2.py.
├── deploy/     docker-compose for the cobot container + host setup script.
└── legacy/     Superseded code preserved for reference:
    ├── vision_hub/   Oct 2025 vision service (replaced by vision/).
    └── cobot/        Older cobot controllers (Unix socket, sine demo, etc.).
```

## Hardware assumed

- NVIDIA Jetson Orin, JetPack 5.x / L4T r35.x (aarch64)
- Intel RealSense D415 or D435 on USB
- Elephant Robotics MyCobot 280 on `/dev/ttyACM0` @ 1 000 000 baud
- Host-native `pyrealsense2` (built from librealsense source, not pip)
- Host-native PyTorch with CUDA (NVIDIA Jetson wheel)
- Docker + Docker Compose v2 (for the cobot container)

## Run it end-to-end on a fresh Jetson

```bash
# 0. Clone + fetch LFS weights
git clone https://github.com/Autonomous-Flight-and-Perception/opss-finale.git
cd opss-finale
git lfs pull

# 1. Vision service on the Jetson host (NOT a container)
cd vision
pip install -r requirements.txt   # skip torch/pyrealsense2, install those natively
python3 main.py --host 0.0.0.0 --port 8000 \
                --capture-width 640 --capture-height 480 &
sleep 10 && curl -X POST http://localhost:8000/pipeline/start

# 2. Cobot controller container
cd ../deploy
./setup.sh                                     # first run only (docker, dialout)
docker compose -f docker-compose.unified.yml up -d cobotpy
docker logs -f opss-cobot

# 3. Dashboard
xdg-open http://localhost:8000/                # or browse from another LAN host
```

The cobot container resolves `vh-vision` to the docker-host gateway via
`extra_hosts`, so its `ws://vh-vision:8000/ws/states` URL lands on the
host-side vision service without code changes.

## Why vision isn't containerized

The Jetson toolchain (aarch64 + CUDA + librealsense + NVIDIA's PyTorch
wheel) is easier to get right on the host than in a container. The
host-native path runs YOLOv8-nano at ~23 FPS end-to-end with the
RealSense in BGR/depth dual-stream. A `Dockerfile.jetson` draft is
deliberately not included — treat containerizing vision as future work.

## Model weights (`vision/*.pt`, Git LFS)

| File                          | Arch         | Purpose                      | mAP@50 | FPS (Jetson) |
|-------------------------------|--------------|------------------------------|--------|--------------|
| `user_best.pt` **(active)**   | YOLOv8-nano  | In-use drone detector        | 0.933  | ~23          |
| `yolo_drone_detect.pt`        | YOLOv26-large| Earlier in-house drone model | 0.908  | ~15          |
| `doguilmak_drone_v8x.pt`      | YOLOv8x      | doguilmak drone detector (HF)| unpub. | ~10          |
| `yolov8n.pt`                  | YOLOv8-nano  | Ultralytics COCO default     | 37.3*  | ~30          |

*COCO val mAP@50.

The active model is selected in
`vision/opss/vision/yolov8_inference.py::YOLOv8Detector.__init__`
(`model_path="user_best.pt"`, conf 0.30, class filter `[0]` → "drone").
`user_best.pt` is YOLOv8n fine-tuned on the `drone-1` Colab dataset for
20 epochs — best accuracy *and* fastest inference among the drone models
tested.

## Wire contract (cobot data in)

Clients connect to **`ws://<host>:8000/ws/states`** and receive 30 Hz
JSON frames:

```json
{
  "timestamp": 1745383492.18,
  "detections": [ { "center": {"x": 317, "y": 224}, "bbox": {…}, "confidence": 0.74, … } ],
  "states":     [ { "track_id": 3, "position": {"x":317, "y":224, "z":1.8}, "velocity": {…}, "frame": "pixel_xy_metric_z", … } ],
  "fused":      [ { "track_id": 3, "position": {…}, "velocity": {…}, "physics_plausible": true, "frame": "pixel_xy_metric_z", … } ],
  "stats":      { "pipeline_fps": 22.4, "tracks_active": 1, "latency_total_ms": 31.2, … }
}
```

`run2.py` reads `fused` → `states` → `detections` in that preference
order — Kalman-smoothed fused/states persist through brief YOLO
dropouts, which is why the cobot targets `/ws/states` rather than the
raw `/ws/detections` feed. Pipeline frame semantics (`pixel`,
`pixel_xy_metric_z`, `camera_metric`, `world_metric`) are documented on
`ObjectState` in `vision/opss/state/kalman.py`.

## Key edits vs. upstream sources

- `cobot/run2.py` — velocity-based controller with coast-on-loss
  (replaces the Oct position-integrator that wound up to joint rails).
  J2 sign flipped for the upside-down arm mount; home pose enforced on
  startup; joint clamps prevent pymycobot validation exceptions.
- `cobot/run2.py:14` — `INTERFACE_URL` env-overridable, defaults to
  `ws://vh-vision:8000/ws/states` (no `/api` prefix).
- `deploy/docker-compose.unified.yml` — standalone cobotpy service;
  routes `vh-vision:host-gateway` so the hardcoded WS URL hits the
  host-side vision service.
- `vision/opss/vision/yolov8_inference.py` — YOLO class filter
  `classes=[0]`, label text "drone", conf 0.30. Model default points at
  `user_best.pt`.

## Licensing note

`doguilmak_drone_v8x.pt` and any Ultralytics-derived weight inherit
AGPL-3.0 from the upstream Ultralytics project. Treat use of this repo
accordingly if you're distributing derived work.
