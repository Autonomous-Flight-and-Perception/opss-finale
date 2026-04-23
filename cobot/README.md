# cobot — OPSS myCobot 280 controller

Containerized controller that drives a MyCobot 280 off the vision
service's detection stream. The live entrypoint is `run2.py`.

## Data flow

```
  vision (host, port 8000)            cobot container
  ────────────────────────            ───────────────────────
  /ws/states  ───── WebSocket ──────► run2.py (CobotDetectionClient)
                                              │
                                              ▼
                                      velocity controller
                                              │
                                              ▼
                                      MyCobot 280 via /dev/ttyACM0
```

`run2.py` reads `fused` → `states` → `detections` from each `/ws/states`
message (in that preference order) so the arm stays smooth through brief
YOLO dropouts — Kalman-filtered fused states persist across frames where
raw detection drops.

## Control law

Velocity-based tracker with coast-on-loss, 30 Hz loop:

- **Detection present**: commanded joint velocity is proportional to the
  normalized pixel error from frame center. `GAIN=3.0`, `DEADBAND=0.05`.
- **Detection lost**: velocity persists in the last-seen direction and
  decays exponentially (`COAST_DECAY=0.92`, ~1 s to near-zero). This
  avoids the snap-to-zero that made the Oct position-integrator jitter.
- **Re-acquired**: velocity is immediately overwritten by the new
  error-driven target.
- J2 sign is inverted for the upside-down arm mount. Joint clamps
  (±165° on J1, ±130° on J2) zero velocity into the rail to prevent
  wind-up.
- Frame normalization assumes **640×480** capture — match the vision
  service flags.

## Run it

From the repo root:

```bash
# 1. Vision on host (required first — cobot blocks on WS connect)
cd vision
python3 main.py --host 0.0.0.0 --port 8000 \
                --capture-width 640 --capture-height 480 &
sleep 10 && curl -X POST http://localhost:8000/pipeline/start

# 2. Cobot container
cd ../deploy
docker compose -f docker-compose.unified.yml up -d cobotpy
docker logs -f opss-cobot
```

Override the USB device with `DEVICE=/dev/ttyACM1 docker compose …`.
Override the WS target with `INTERFACE_URL=ws://<host>:<port>/ws/states`.

## Standalone (no docker, direct serial on the host)

```bash
pip install -r requirements.txt
INTERFACE_URL=ws://localhost:8000/ws/states python3 run2.py
```

## Files

- `run2.py` — canonical controller (WebSocket client + velocity loop).
- `Dockerfile` — builds `python:3.9-slim + pymycobot + websockets`.
- `requirements.txt` — pinned deps.

Older cobot variants (Unix-socket receiver, sine demo, debug clients)
live in `../legacy/cobot/` for reference.
