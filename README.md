# opss-finale

Full working snapshot of the OPSS drone-tracking cobot-arm demo as of 2026-04-22.
Everything needed to reproduce the live end-to-end pipeline is here; the earlier
source repos are preserved in `legacy/` for reference.

## What's in the box

```
opss-finale/
├── vision/       Host-side vision service (OPSS). RealSense + YOLO, serves
│                 dashboard and /ws/detections on port 8000. Runs on the Jetson
│                 host (not containerized — see "Why not a container" below).
├── cobot/        MyCobot controller (cobotpy). Containerized. Subscribes to
│                 vision/ via WebSocket, drives the arm joints from pixel
│                 detections. Entrypoint: run2.py.
├── deploy/       Unified docker-compose for the cobot container. Points the
│                 container at the host-side vision service via
│                 `extra_hosts: vh-vision -> host-gateway`.
├── legacy/       Superseded code retained for reference.
│   └── vision_hub/  Oct 2025 version of the vision service. Replaced by
│                    vision/ (OPSS) on Jetson because the Oct code didn't run
│                    end-to-end on aarch64 (CPU inference + thread races).
└── models/       Symlinks into vision/*.pt — the raw weights live in vision/
                  (LFS). See "Model weights" below.
```

## Hardware assumed

- Jetson Orin, JetPack 5.x / L4T r35.x (aarch64)
- Intel RealSense D415 or D435 on USB
- MyCobot 280 on `/dev/ttyACM0` (1000000 baud)
- Host-native `pyrealsense2` (built from librealsense source, not pip)
- Host-native PyTorch with CUDA (NVIDIA Jetson wheel)
- Docker + Docker Compose for the cobot container

## Run it end-to-end on a fresh Jetson

```bash
# 1. Put this repo somewhere on the Jetson
git clone https://github.com/Autonomous-Flight-and-Perception/opss-finale.git
cd opss-finale
git lfs pull    # fetch model weights

# 2. Start the vision service on the Jetson host (NOT a container)
cd vision
python3 main.py --host 0.0.0.0 --port 8000 \
                --capture-width 640 --capture-height 480 &
# Kick the pipeline (doesn't auto-start):
sleep 10 && curl -X POST http://localhost:8000/pipeline/start

# 3. In another shell, bring up the cobot controller container
cd ../deploy
docker compose -f docker-compose.unified.yml up -d cobotpy

# 4. Open the dashboard
xdg-open http://localhost:8000/   # or visit from another machine on the LAN
```

The cobot container resolves `vh-vision` to the docker-host gateway via
`extra_hosts`, so its hardcoded `ws://vh-vision:8000/ws/detections` URL hits the
host-side OPSS service without cobotpy code changes.

## Why vision/ isn't containerized

The Jetson toolchain (aarch64 + CUDA + librealsense + NVIDIA's PyTorch wheel)
is easier to get right on the host than inside a container. The draft
`vision/Dockerfile.jetson` is included but untested — stable path is host
Python.

## Model weights (in `vision/`, tracked via Git LFS)

| File                          | Arch           | Purpose                        | mAP@50 (val) |
|-------------------------------|----------------|--------------------------------|--------------|
| `yolov8n.pt`                  | YOLOv8-nano    | Ultralytics COCO default       | 37.3 (COCO)  |
| `yolo_drone_detect.pt`        | YOLOv26-large  | In-house drone detector        | **0.908**    |
| `doguilmak_drone_v8x.pt`      | YOLOv8x        | doguilmak drone detector (HF)  | unpublished  |

The active model is set in
`vision/opss/vision/yolov8_inference.py::YOLOv8Detector.__init__` as the default
`model_path`.

## Key edits vs. upstream sources

- `cobot/run2.py`: velocity-based controller with coast-on-loss (replaces
  the Oct position-integrator that wound up to joint rails). J2 sign flipped
  for the upside-down arm mount. Home pose enforced on startup. Joint
  clamps prevent pymycobot validation exceptions.
- `cobot/run2.py:14`: `INTERFACE_URL` env-overridable (default points at
  docker service name `vh-vision`, path `/ws/detections`, no `/api` prefix).
- `deploy/docker-compose.unified.yml`: `vh-vision` service removed
  (host-side OPSS replaces it); `extra_hosts: ["vh-vision:host-gateway"]`
  added to `cobotpy` service.
- `vision/opss/vision/yolov8_inference.py`: YOLO class filter is
  `classes=[0]` (drone), label text "drone". Model default points at
  `doguilmak_drone_v8x.pt`.

## Licensing note

The `doguilmak_drone_v8x.pt` checkpoint and any Ultralytics-derived weight
inherits AGPL-3.0 from the upstream Ultralytics project. Treat use of this
repo accordingly if you're distributing derived work.
