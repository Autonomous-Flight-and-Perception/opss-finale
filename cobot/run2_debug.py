#!/usr/bin/env python3
import os
import time
import sys
import asyncio
import json
import websockets
from pymycobot.mycobot import MyCobot

PORT = os.environ.get("PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("BAUD", "1000000"))
INTERFACE_URL = "ws://172.31.46.201:8000/api/api/ws/detections"

class CobotDetectionClient:
    def __init__(self, interface_url=INTERFACE_URL):
        self.url = interface_url
        self.latest_detections = []
        self.running = False
    
    async def connect(self):
        print(f"[INFO] Connecting to {self.url}")
        try:
            async with websockets.connect(self.url) as websocket:
                print("[INFO] Connected to detection stream (30Hz)")
                self.running = True
                while self.running:
                    message = await websocket.recv()
                    data = json.loads(message)
                    self.latest_detections = data.get("detections", [])
                    if self.latest_detections:
                        print(f"[DEBUG] Received {len(self.latest_detections)} detections")
        except Exception as e:
            print(f"[WARN] Detection client error: {e}")
            await asyncio.sleep(1)
            await self.connect()
    
    def get_latest_target(self):
        return self.latest_detections[0] if self.latest_detections else None
    
    def stop(self):
        self.running = False

async def cobot_motion(mc, detection_client):
    j1, j2 = 0.0, 90.0
    step_size = 1
    speed = 50
    dt = 0.03
    
    print("[INFO] Motion loop active. Waiting for detections...")
    while True:
        detection = detection_client.get_latest_target()
        if detection:
            print(f"[DEBUG] Processing detection: {detection}")
            center_x = detection["center"]["x"] / 640
            center_y = detection["center"]["y"] / 480
            
            if center_x < 0.45:
                j1 += step_size * abs(1 - center_x)
            elif center_x > 0.55:
                j1 -= step_size * abs(center_x)
            
            if center_y > 0.55:
                j2 += step_size * abs(1 - center_y)
            elif center_y < 0.45:
                j2 -= step_size * abs(center_y)
            
            mc.send_angles([j1, j2, 0, 90, 0, 0], speed)
            print(f"[MOVE] j1={j1:.2f}, j2={j2:.2f} (cx={center_x:.2f}, cy={center_y:.2f})")
        
        await asyncio.sleep(dt)

async def main():
    print(f"[INFO] Connecting to myCobot on {PORT} @ {BAUD}...")
    mc = MyCobot(PORT, BAUD)
    mc.power_on()
    time.sleep(1.0)
    
    detection_client = CobotDetectionClient()
    detection_task = asyncio.create_task(detection_client.connect())
    
    try:
        await cobot_motion(mc, detection_client)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping.")
    finally:
        detection_client.stop()
        detection_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
