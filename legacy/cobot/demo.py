#!/usr/bin/env python3
# afp.py
import os
import time
import sys
import math
from pymycobot.mycobot import MyCobot

# Host device (can be overridden with PORT env var)
PORT = os.environ.get("PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("BAUD", "1000000"))

def main():
    print(f"[INFO] Connecting to myCobot on {PORT} @ {BAUD}...")
    try:
        mc = MyCobot(PORT, BAUD)
    except Exception as e:
        print(f"[ERROR] Failed to open serial port {PORT}: {e}")
        sys.exit(1)

    try:
        print("[INFO] Powering on servos...")
        mc.power_on()
        time.sleep(1.0)

        print("[INFO] Smooth sine/cosine joint motion (circle-like)...")

        steps = 600            # number of points in the motion
        amplitude = 60         # max excursion in degrees
        speed = 8          # myCobot "speed" (0–100)
        dt = 0.025            # pause between updates (s)

        for i in range(steps):
            theta = 2 * math.pi * i / steps   # sweep 0 → 2π
            j1 = amplitude * math.sin(theta)
            j2 = amplitude * math.cos(theta)
            mc.send_angles([j1, j2, 0, 90, 0, 0], speed)
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    except Exception as e:
        print(f"[ERROR] Exception during motion: {e}")
    finally:
        try:
            print("[INFO] Powering off servos (motors limp)...")
        except Exception:
            pass
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
