#!/usr/bin/env python3
import os
import time
import sys
import math
import asyncio
import json
import websockets
from pymycobot.mycobot import MyCobot

# Host device (can be overridden with PORT env var)
PORT = os.environ.get("PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("BAUD", "1000000"))

INTERFACE_URL = "ws://localhost:8000/api/ws/detections"

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
    """Drive the robot only when objects are detected"""
    steps = 600
    amplitude = 60
    speed = 8
    dt = 0.025

    print("[INFO] Starting motion loop. Waiting for detections...")

    for i in range(steps):
        # Wait until detection is available
        while detection_client.get_latest_target() is None:
            await asyncio.sleep(0.05)  # check every 50ms

        theta = 2 * math.pi * i / steps
        j1 = amplitude * math.sin(theta)
        j2 = amplitude * math.cos(theta)
        mc.send_angles([j1, j2, 0, 90, 0, 0], speed)
        time.sleep(dt)  # keep servo timing consistent


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
