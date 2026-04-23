from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import requests
from app.yolov8_inference import process_frame_bgr
import threading

router = APIRouter()

# RasPi configuration
RASPI_IP = "192.168.100.2"
RASPI_PORT = 8001
RASPI_STREAM_URL = f"http://{RASPI_IP}:{RASPI_PORT}/stream"

_lock = threading.Lock()
_latest_detections = []

def parse_mjpeg_stream(response):
    """Parse MJPEG stream into frames"""
    bytes_data = b''
    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # JPEG start
        b = bytes_data.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame

def mjpeg_with_detection():
    """Pull from RasPi, run detection, re-stream"""
    global _latest_detections
    
    response = requests.get(RASPI_STREAM_URL, stream=True, timeout=5)
    
    for frame in parse_mjpeg_stream(response):
        # Run detection
        annotated, detections = process_frame_bgr(frame, thresh=0.55)
        
        # Update detections
        with _lock:
            _latest_detections = detections
        
        # Send to cobot (reuse your existing broadcaster if you want)
        # _cobot_broadcaster.send(detections)
        
        # Encode and stream
        _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

@router.get("/raspi/stream")
def stream_from_raspi():
    return StreamingResponse(
        mjpeg_with_detection(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/raspi/detections")
def get_raspi_detections():
    with _lock:
        return JSONResponse({"detections": _latest_detections.copy()})
