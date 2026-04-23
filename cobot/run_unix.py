#!/usr/bin/env python3
"""
Cobot Controller - Unix Socket Version
High-performance, low-latency detection receiver for myCobot 280

Receives detections from Vision Hub via Unix domain socket
Latency: ~0.1-0.5ms 
"""

import os
import sys
import time
import socket
import json
import asyncio
from pymycobot.mycobot import MyCobot



# Robot serial port configuration
PORT = os.environ.get("PORT", "/dev/ttyACM0")
BAUD = int(os.environ.get("BAUD", "1000000"))

# Unix socket path (must match Vision Hub)
SOCKET_PATH = "/tmp/opss_cobot.sock"

# Control parameters
CONTROL_RATE_HZ = 30  # Control loop frequency
STEP_SIZE = 1         # Joint angle step size (degrees)
SPEED = 50            # Robot speed (0-100)


# ============================================================================
# Unix Socket Client
# ============================================================================

class CobotUnixClient:
    """
    Unix domain socket client for receiving detections from Vision Hub
    Non-blocking, high-performance communication
    """
    
    def __init__(self, socket_path=SOCKET_PATH):
        self.socket_path = socket_path
        self.latest_detections = []
        self.running = False
        self.frame_count = 0
        self.last_receive_time = time.time()
        
        # Create Unix domain socket (DGRAM = datagram, like UDP but local)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        
        # Remove old socket file if it exists
        try:
            os.unlink(self.socket_path)
            print(f"[INFO] Removed old socket: {self.socket_path}")
        except OSError:
            pass
        
        # Bind to socket path
        self.sock.bind(self.socket_path)
        
        # Set non-blocking mode (don't wait if no data available)
        self.sock.setblocking(False)
        
        print(f"[INFO] ✓ Listening on Unix socket: {self.socket_path}")
        print(f"[INFO] Waiting for detections from Vision Hub...")
    
    async def receive_loop(self):
        """
        Continuously receive detections from Vision Hub
        Runs in background as async task
        """
        self.running = True
        
        while self.running:
            try:
                # Receive data (non-blocking, up to 64KB)
                data, addr = self.sock.recvfrom(65536)
                
                # Parse JSON
                parsed = json.loads(data.decode('utf-8'))
                self.latest_detections = parsed.get("detections", [])
                self.frame_count += 1
                self.last_receive_time = time.time()
                
                # Log periodically (every 30 frames = ~1 second at 30fps)
                if self.frame_count % 30 == 0:
                    num_dets = len(self.latest_detections)
                    print(f"[RECV] Frame {self.frame_count}: {num_dets} detection(s)")
                
            except BlockingIOError:
                # No data available right now, that's OK
                pass
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error: {e}")
            except Exception as e:
                print(f"[ERROR] Socket receive error: {e}")
            
            # Check for data very frequently (1ms sleep)
            await asyncio.sleep(0.001)
    
    def get_latest_target(self):
        """
        Get the first detection (primary target)
        Returns: detection dict or None
        """
        return self.latest_detections[0] if self.latest_detections else None
    
    def get_all_detections(self):
        """Get all current detections"""
        return self.latest_detections
    
    def is_receiving(self, timeout_sec=1.0):
        """Check if we're still receiving data from Vision Hub"""
        return (time.time() - self.last_receive_time) < timeout_sec
    
    def stop(self):
        """Clean shutdown"""
        self.running = False
        self.sock.close()
        
        # Clean up socket file
        try:
            os.unlink(self.socket_path)
        except:
            pass
        
        print("[INFO] Unix socket closed")


# ============================================================================
# Robot Control Logic
# ============================================================================

async def cobot_motion(mc, client):
    """
    Main control loop: track detected targets with robot
    
    Simple proportional controller:
    - If target is left of center → move joint 1 left
    - If target is right of center → move joint 1 right
    - If target is above center → move joint 2 up
    - If target is below center → move joint 2 down
    """
    
    # Initial joint angles
    j1, j2 = 0.0, 90.0
    
    # Control loop rate
    dt = 1.0 / CONTROL_RATE_HZ
    
    print("[INFO] ✓ Motion control loop active")
    print(f"[INFO] Control rate: {CONTROL_RATE_HZ} Hz")
    print(f"[INFO] Waiting for detections...")
    
    loop_count = 0
    
    while True:
        detection = client.get_latest_target()
        
        if detection:
            # Extract center coordinates
            center_x = detection["center"]["x"]
            center_y = detection["center"]["y"]
            confidence = detection["confidence"]
            
            # Normalize to [0.0, 1.0]
            norm_x = center_x / 640.0
            norm_y = center_y / 480.0
            
            # Dead zone in center (0.45 to 0.55 = 10% dead zone)
            # This prevents jittering around center
            if norm_x < 0.45:
                # Target is LEFT of center
                j1 += STEP_SIZE * abs(1.0 - norm_x)
            elif norm_x > 0.55:
                # Target is RIGHT of center
                j1 -= STEP_SIZE * abs(norm_x)
            
            if norm_y > 0.55:
                # Target is BELOW center (Y axis is inverted in images)
                j2 += STEP_SIZE * abs(1.0 - norm_y)
            elif norm_y < 0.45:
                # Target is ABOVE center
                j2 -= STEP_SIZE * abs(norm_y)
            
            # Clamp joint angles to safe ranges
            j1 = max(-170, min(170, j1))
            j2 = max(0, min(180, j2))
            
            # Send command to robot
            try:
                mc.send_angles([j1, j2, 0, 90, 0, 0], SPEED)
            except Exception as e:
                print(f"[ERROR] Failed to send robot command: {e}")
            
            # Log periodically (every 30 iterations = ~1 second)
            loop_count += 1
            if loop_count % 30 == 0:
                print(f"[TRACK] Target: ({center_x}, {center_y}) | "
                      f"Conf: {confidence:.2f} | "
                      f"Joints: j1={j1:.1f}° j2={j2:.1f}°")
        
        else:
            # No detection - robot holds position
            pass
        
        # Check connection health
        if not client.is_receiving(timeout_sec=2.0):
            print("[WARN] No data from Vision Hub for 2 seconds!")
        
        # Control loop delay
        await asyncio.sleep(dt)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """
    Initialize robot and start control loop
    """
    
    print("=" * 60)
    print("  OPSS Cobot Controller - Unix Socket Version")
    print("=" * 60)
    
    # Connect to robot
    print(f"[INFO] Connecting to myCobot 280 on {PORT} @ {BAUD} baud...")
    try:
        mc = MyCobot(PORT, BAUD)
        print("[INFO] ✓ Robot connected")
    except Exception as e:
        print(f"[ERROR] ✗ Failed to connect to robot: {e}")
        print(f"[ERROR] Check that {PORT} exists and robot is powered on")
        sys.exit(1)
    
    # Power on servos
    print("[INFO] Powering on servos...")
    try:
        mc.power_on()
        time.sleep(1.0)
        print("[INFO] ✓ Servos powered on")
    except Exception as e:
        print(f"[ERROR] Failed to power on servos: {e}")
        sys.exit(1)
    
    # Start Unix socket client
    print("[INFO] Starting Unix socket client...")
    client = CobotUnixClient()
    receive_task = asyncio.create_task(client.receive_loop())
    
    print("[INFO] ✓ System ready")
    print("")
    print("Listening for detections from Vision Hub...")
    print("Press Ctrl+C to stop")
    print("")
    
    try:
        # Run main control loop
        await cobot_motion(mc, client)
        
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received")
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        
    finally:
        # Cleanup
        print("[INFO] Shutting down...")
        
        # Stop client
        client.stop()
        receive_task.cancel()
        
        # Release robot servos
        try:
            mc.release_all_servos()
            print("[INFO] ✓ Servos released (motors off)")
        except Exception as e:
            print(f"[WARN] Could not release servos: {e}")
        
        print("[INFO] ✓ Shutdown complete")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
