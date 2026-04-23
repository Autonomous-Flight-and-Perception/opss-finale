#!/usr/bin/env python3
# afp_coords.py
import os
import sys
import time
from pymycobot.mycobot import MyCobot

# Host device (can be overridden with PORT/BAUD env vars)
PORT = os.environ.get("PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("BAUD", "1000000"))

def wait_until_reached(mc, target, timeout=10):
    """
    Wait until myCobot reaches the Cartesian target.
    Target: [x, y, z, rx, ry, rz]
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            if mc.is_in_position(target, 1):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

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

        # --- Cartesian move ---
        print("[INFO] Sending safe Cartesian move...")
        # [X, Y, Z, RX, RY, RZ]
        # Units: mm for XYZ, degrees for RX/RY/RZ
        cart_target = [150, 0, 200, 0, 0, 0]  # forward and slightly up
        mc.send_coords(cart_target, 20, 1)

        if wait_until_reached(mc, cart_target):
            print("[INFO] Reached Cartesian target.")
        else:
            print("[WARN] Cartesian target not reached within timeout.")

        # You can add a return-to-home step if desired:
        # home = [200, 0, 150, 0, 0, 0]
        # mc.send_coords(home, 20, 1)
        # wait_until_reached(mc, home)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    except Exception as e:
        print(f"[ERROR] Exception during motion: {e}")
    finally:
        try:
            print("[INFO] Powering off servos (motors limp)...")
            mc.power_off()
        except Exception:
            pass
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
