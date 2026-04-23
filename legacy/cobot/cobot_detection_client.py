"""
Cobot Detection Client - WebSocket Stream Consumer
Receives real-time bounding box detections from CV interface
Low latency: ~10-30ms
"""

import asyncio
import websockets
import json
from datetime import datetime

# Configure your CV interface URL
# If same machine: "ws://172.31.46.201:8000/api/ws/detections"
# If remote: "ws://YOUR_SERVER_IP:8000/api/ws/detections"
INTERFACE_URL = "ws://172.31.46.201:8000/api/ws/detections"

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
                print("[INFO] Connected! Receiving detection stream at 30Hz...")
                self.running = True
                
                while self.running:
                    # Receive detection data
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Update latest detections
                    self.latest_detections = data['detections']
                    timestamp = data['timestamp']
                    
                    # Process detections for cobot control
                    await self.process_detections(data['detections'], timestamp)
                    
        except websockets.exceptions.ConnectionClosed:
            print("[WARN] Connection closed, attempting to reconnect...")
            await asyncio.sleep(1)
            await self.connect()  # Auto-reconnect
            
        except Exception as e:
            print(f"[ERROR] Connection error: {e}")
            await asyncio.sleep(1)
            await self.connect()  # Auto-reconnect
    
    async def process_detections(self, detections, timestamp):
        """
        Process detections and send commands to cobot
        This is where you integrate with your cobot control logic
        """
        if not detections:
            # No targets detected
            return
        
        for detection in detections:
            # Extract target coordinates
            center_x = detection['center']['x']
            center_y = detection['center']['y']
            confidence = detection['confidence']
            
            # Get all 4 corners for trajectory planning
            corners = detection['corners']
            top_left = (corners['top_left']['x'], corners['top_left']['y'])
            top_right = (corners['top_right']['x'], corners['top_right']['y'])
            bottom_left = (corners['bottom_left']['x'], corners['bottom_left']['y'])
            bottom_right = (corners['bottom_right']['x'], corners['bottom_right']['y'])
            
            # Get bounding box
            bbox = detection['bbox']
            width = bbox['x2'] - bbox['x1']
            height = bbox['y2'] - bbox['y1']
            
            # Example: Track the target
            print(f"[TRACKING] Target at pixel ({center_x}, {center_y}) | "
                  f"Size: {width}x{height} | Confidence: {confidence:.2%}")
            
            # TODO: Insert your cobot control logic here
            # Examples:
            # await self.move_cobot_to_target(center_x, center_y)
            # await self.aim_turret_at(center_x, center_y)
            # self.trajectory_planner.update_target(corners)
    
    async def move_cobot_to_target(self, pixel_x, pixel_y):
        """
        Example method to send movement commands to cobot
        Replace with your actual cobot API/controller
        """
        # Convert pixel coordinates to robot coordinates
        robot_x, robot_y = self.pixel_to_robot_coords(pixel_x, pixel_y)
        
        print(f"[COMMAND] Move to robot coords: ({robot_x:.2f}, {robot_y:.2f})")
        
        # Send command to your cobot controller
        # Example (replace with your actual implementation):
        # await your_cobot_controller.move_to(robot_x, robot_y)
        # OR
        # self.cobot_socket.send(f"MOVE {robot_x} {robot_y}")
        pass
    
    def pixel_to_robot_coords(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates (640x480) to robot coordinate system
        
        YOU MUST CALIBRATE THIS based on your camera-robot setup!
        This is a placeholder transformation.
        """
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 480
        
        # Normalize to [0, 1]
        norm_x = pixel_x / FRAME_WIDTH
        norm_y = pixel_y / FRAME_HEIGHT
        
        # Example: Map to robot workspace
        # Adjust these ranges based on your robot's coordinate system
        ROBOT_X_MIN, ROBOT_X_MAX = -500, 500  # mm or your units
        ROBOT_Y_MIN, ROBOT_Y_MAX = 0, 1000
        
        robot_x = ROBOT_X_MIN + (norm_x * (ROBOT_X_MAX - ROBOT_X_MIN))
        robot_y = ROBOT_Y_MIN + (norm_y * (ROBOT_Y_MAX - ROBOT_Y_MIN))
        
        return robot_x, robot_y
    
    def get_latest_target(self):
        """
        Get the most recent detection (useful for sync code)
        Returns the first detection or None
        """
        if self.latest_detections:
            return self.latest_detections[0]
        return None
    
    def stop(self):
        """Stop the client gracefully"""
        self.running = False
        print("[INFO] Client stopped")

async def main():
    """Main entry point"""
    client = CobotDetectionClient()
    
    try:
        await client.connect()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        client.stop()

if __name__ == "__main__":
    # Run the client
    asyncio.run(main())
