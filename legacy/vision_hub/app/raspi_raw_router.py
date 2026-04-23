from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import socket
import struct
import cv2
import numpy as np
from app.yolov8_inference import process_frame_bgr
import threading
import time

router = APIRouter()

RASPI_IP = "192.168.100.2"
RASPI_PORT = 9000
WIDTH, HEIGHT = 640, 480

# Optimization settings
FRAME_SKIP = 3  # Run detection every 3rd frame
JPEG_QUALITY = 65  # Lower = faster encoding
DETECTION_THRESHOLD = 0.6  # Higher = fewer false positives, faster inference

_latest_frame = None
_lock = threading.Lock()
_running = False
_frame_received = threading.Event()

def receive_frames():
    """Background thread to receive raw frames from RasPi with optimizations"""
    global _latest_frame, _running
    
    while _running:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1MB receive buffer
            sock.settimeout(5.0)
            sock.connect((RASPI_IP, RASPI_PORT))
            print(f"[INFO] Connected to RasPi raw stream")
            
            while _running:
                # Receive frame size
                size_data = sock.recv(4)
                if not size_data or len(size_data) < 4:
                    break
                size = struct.unpack('>I', size_data)[0]
                
                # Receive frame data in larger chunks
                data = bytearray(size)
                view = memoryview(data)
                bytes_received = 0
                
                while bytes_received < size:
                    chunk_size = min(65536, size - bytes_received)  # 64KB chunks
                    chunk = sock.recv(chunk_size)
                    if not chunk:
                        break
                    view[bytes_received:bytes_received + len(chunk)] = chunk
                    bytes_received += len(chunk)
                
                if bytes_received < size:
                    break
                
                # Convert to numpy array (zero-copy where possible)
                frame = np.frombuffer(data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                with _lock:
                    _latest_frame = frame
                    _frame_received.set()
            
            sock.close()
            
        except socket.timeout:
            print(f"[WARN] Connection timeout")
        except Exception as e:
            print(f"[ERROR] Raw stream error: {e}")
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass
        
        if _running:
            print("[INFO] Reconnecting in 1s...")
            time.sleep(1)


@router.get("/raspi/raw/stream")
def stream_raw():
    """Ultra-optimized stream with frame skipping and adaptive quality"""
    global _running
    
    # Start receiver thread if not running
    if not _running:
        _running = True
        thread = threading.Thread(target=receive_frames, daemon=True)
        thread.start()
    
    def generate():
        frame_count = 0
        last_annotated = None
        last_detection_time = time.time()
        
        # Wait for first frame
        _frame_received.wait(timeout=5.0)
        
        while True:
            # Get latest frame
            with _lock:
                if _latest_frame is None:
                    time.sleep(0.001)  # Tiny sleep to prevent CPU spinning
                    continue
                frame = _latest_frame.copy()
            
            frame_count += 1
            current_time = time.time()
            
            # Adaptive detection: skip frames intelligently
            if frame_count % FRAME_SKIP == 0 or last_annotated is None:
                # Run detection
                annotated, detections = process_frame_bgr(frame, thresh=DETECTION_THRESHOLD)
                last_annotated = annotated
                last_detection_time = current_time
            else:
                # Reuse last detection overlay on current frame (smooth playback)
                annotated = frame  # Show raw frame between detections for max speed
            
            # Fast JPEG encoding with lower quality
            encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,  # Enable optimization
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0  # Disable progressive (faster)
            ]
            
            _, buf = cv2.imencode('.jpg', annotated, encode_params)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   buf.tobytes() + 
                   b'\r\n')
    
    return StreamingResponse(
        generate(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )
