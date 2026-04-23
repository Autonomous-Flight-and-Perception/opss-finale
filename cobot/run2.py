#!/usr/bin/env python3
import os
import time
import sys
import asyncio
import json
import websockets
from pymycobot.mycobot import MyCobot

# Host device (can be overridden with PORT env var)
PORT = os.environ.get("PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("BAUD", "1000000"))

INTERFACE_URL = os.environ.get("INTERFACE_URL", "ws://vh-vision:8000/ws/detections")

class CobotDetectionClient:
    def __init__(self, interface_url=INTERFACE_URL):
        self.url = interface_url
        self.latest_detections = []
        self.running = False

    async def connect(self):
        """Connect to detection stream and process detections"""
        print(f"[INFO] Connecting to {self.url}")
        try:
            async with websockets.connect(self.url) as websocket:
                print("[INFO] Connected to detection stream (30Hz)")
                self.running = True
                while self.running:
                    message = await websocket.recv()
                    data = json.loads(message)
                    self.latest_detections = data.get("detections", [])
        except Exception as e:
            print(f"[WARN] Detection client error: {e}")
            await asyncio.sleep(1)
            await self.connect()  # Auto-reconnect

    def get_latest_target(self):
        """Return the first detection if available, else None"""
        return self.latest_detections[0] if self.latest_detections else None

    def stop(self):
        self.running = False
        print("[INFO] Detection client stopped")


async def cobot_motion(mc, detection_client):
    """Velocity-controlled tracker with coast on detection loss.

    - Detection present: commanded joint velocity is proportional to the
      pixel error from frame center. Farther off-center = faster.
    - Detection lost: velocity persists in the last-seen direction and
      decays exponentially, so the arm coasts to a stop instead of
      freezing or snapping.
    - Detection re-acquired: velocity is immediately overwritten by the
      new error-driven target.
    """
    j1, j2 = 0.0, -90.0     # commanded joint angles (j2 inverted for upside-down mount)
    v_j1, v_j2 = 0.0, 0.0   # current commanded velocity, deg per tick

    GAIN = 3.0              # err ±0.5 (frame edge) -> v ±1.5 deg/tick at 30 Hz
    DEADBAND = 0.05         # normalized pixel error inside which target v=0
    COAST_DECAY = 0.92      # per-tick multiplier on v when no detection (~1 s to near-zero)
    V_EPSILON = 0.01        # snap tiny velocities to zero to avoid endless micro-drift
    SPEED = 50              # mc.send_angles speed param (robot slew cap)
    DT = 0.03               # loop period, ~30 Hz

    print("[INFO] Motion loop active. Waiting for detections...")

    while True:
        detection = detection_client.get_latest_target()

        if detection:
            # Normalized error from frame center, range ~[-0.5, 0.5]
            err_x = (detection["center"]["x"] / 640) - 0.5
            err_y = (detection["center"]["y"] / 480) - 0.5

            # Sign convention preserved from prior working code:
            #   person LEFT  (err_x<0) -> j1 increases
            #   person LOW   (err_y>0) -> j2 increases
            v_j1 = 0.0 if abs(err_x) < DEADBAND else -err_x * GAIN
            v_j2 = 0.0 if abs(err_y) < DEADBAND else -err_y * GAIN
            coasting = False
        else:
            # No detection this tick — keep direction, shrink magnitude
            v_j1 *= COAST_DECAY
            v_j2 *= COAST_DECAY
            if abs(v_j1) < V_EPSILON: v_j1 = 0.0
            if abs(v_j2) < V_EPSILON: v_j2 = 0.0
            coasting = True

        # Integrate velocity into commanded position
        j1 += v_j1
        j2 += v_j2

        # Clamp to MyCobot 280 joint limits, and zero the velocity into
        # the rail so we don't accumulate phantom wind-up that would
        # kick the arm off the clamp the instant error flips.
        if j1 >= 165.0 and v_j1 > 0: v_j1 = 0.0
        if j1 <= -165.0 and v_j1 < 0: v_j1 = 0.0
        if j2 >= 130.0 and v_j2 > 0: v_j2 = 0.0
        if j2 <= -130.0 and v_j2 < 0: v_j2 = 0.0
        j1 = max(-165.0, min(165.0, j1))
        j2 = max(-130.0, min(130.0, j2))

        # Skip send_angles when fully parked and no detection — no
        # commanded motion to apply. MyCobot firmware restarts its
        # motion plan on every send_angles, so redundant sends are
        # actively harmful at rest.
        if detection or v_j1 != 0.0 or v_j2 != 0.0:
            mc.send_angles([j1, j2, 0, 90, 0, 0], SPEED)
            tag = "COAST" if coasting else "MOVE"
            cx = detection["center"]["x"] / 640 if detection else float("nan")
            cy = detection["center"]["y"] / 480 if detection else float("nan")
            print(f"[{tag}] j1={j1:+.2f} j2={j2:+.2f} "
                  f"v=({v_j1:+.3f},{v_j2:+.3f}) "
                  f"(cx={cx:.2f}, cy={cy:.2f})")

        await asyncio.sleep(DT)


async def main():
    # Connect to robot
    print(f"[INFO] Connecting to myCobot on {PORT} @ {BAUD}...")
    try:
        mc = MyCobot(PORT, BAUD)
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        sys.exit(1)

    mc.power_on()
    time.sleep(1.0)

    # Drive to home pose before entering tracking loop so the arm
    # starts from a known orientation each session.
    HOME_POSE = [0.0, -90.0, 0.0, 90.0, 0.0, 0.0]
    HOME_SPEED = 30
    print(f"[INFO] Moving to home pose {HOME_POSE} at speed {HOME_SPEED}...")
    mc.send_angles(HOME_POSE, HOME_SPEED)
    time.sleep(3.0)
    print("[INFO] At home pose.")

    # Start detection client
    detection_client = CobotDetectionClient()
    detection_task = asyncio.create_task(detection_client.connect())

    try:
        await cobot_motion(mc, detection_client)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    finally:
        detection_client.stop()
        detection_task.cancel()
        try:
            mc.release_all_servos()
            print("[INFO] Servos released.")
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
