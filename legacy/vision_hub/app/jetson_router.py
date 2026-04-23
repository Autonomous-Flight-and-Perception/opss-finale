from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import asyncio
import socket
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import queue

from app.yolov8_inference import process_frame_bgr

router = APIRouter()
WIDTH, HEIGHT, FPS = 640, 480, 30

# Global state
_lock = threading.Lock()
_pipe = None
_latest_detections = []
_latest_annotated_frame = None
_streaming = False
_detection_executor = None

# High-performance queues
_frame_queue = queue.Queue(maxsize=2)
_result_queue = queue.Queue(maxsize=2)

# Performance tracking
_perf_stats = {
    "capture_fps": 0,
    "detection_fps": 0,
    "stream_fps": 0,
    "last_capture_time": time.time(),
    "last_detection_time": time.time(),
    "last_stream_time": time.time(),
    "capture_count": 0,
    "detection_count": 0,
    "stream_count": 0
}

# Unix socket configuration
COBOT_SOCKET_PATH = "/tmp/opss_cobot.sock"


class UnixSocketBroadcaster:
    """Non-blocking Unix socket broadcaster for cobot communication"""
    
    def __init__(self, socket_path=COBOT_SOCKET_PATH):
        self.socket_path = socket_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.connected = False
        print(f"[INFO] Unix socket broadcaster initialized: {self.socket_path}")
        
    def send(self, detections: list):
        """Send detections to cobot (non-blocking, fire-and-forget)"""
        if not detections:
            return
        
        try:
            data = json.dumps({
                "timestamp": time.time_ns(),
                "detections": detections,
                "frame_size": {"width": WIDTH, "height": HEIGHT}
            }).encode('utf-8')
            
            self.sock.sendto(data, self.socket_path)
            
            if not self.connected:
                self.connected = True
                print("[INFO] ✓ Cobot connected via Unix socket")
                
        except FileNotFoundError:
            if self.connected:
                print("[WARN] ✗ Cobot disconnected")
                self.connected = False
        except Exception as e:
            if self.connected:
                print(f"[ERROR] Unix socket send failed: {e}")
                self.connected = False


# Create global Unix socket broadcaster
_cobot_broadcaster = UnixSocketBroadcaster()


def start():
    """Initialize RealSense camera with optimized settings"""
    global _pipe
    with _lock:
        if _pipe: 
            return
        p = rs.pipeline()
        cfg = rs.config()
        
        # Enable streams
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        
        # Start pipeline
        p.start(cfg)
        _pipe = p
        print("[INFO] RealSense pipeline started")


def stop():
    """Stop RealSense camera and cleanup"""
    global _pipe, _streaming, _detection_executor
    _streaming = False
    time.sleep(0.3)  # Let threads finish
    
    # Shutdown executor
    if _detection_executor:
        _detection_executor.shutdown(wait=False)
        _detection_executor = None
    
    with _lock:
        if _pipe:
            _pipe.stop()
            _pipe = None
            print("[INFO] RealSense pipeline stopped")


def frame_capture_worker():
    """
    HIGH PRIORITY: Capture frames as fast as possible from camera
    Drops old frames if detection can't keep up
    """
    global _pipe, _streaming, _perf_stats
    
    print("[INFO] Frame capture worker started")
    last_fps_log = time.time()
    
    while _streaming and _pipe:
        try:
            # Capture frame (this is FAST, ~1ms)
            frames = _pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # Convert to numpy (also fast, ~1ms)
            bgr = np.asanyarray(color_frame.get_data())
            
            # Performance tracking
            _perf_stats["capture_count"] += 1
            
            # Try to put in queue (non-blocking)
            try:
                _frame_queue.put_nowait(bgr)
            except queue.Full:
                # Queue full means detection is slow - drop this frame
                pass
            
            # Log FPS every 2 seconds
            now = time.time()
            if now - last_fps_log > 2.0:
                elapsed = now - _perf_stats["last_capture_time"]
                fps = _perf_stats["capture_count"] / elapsed
                _perf_stats["capture_fps"] = fps
                _perf_stats["capture_count"] = 0
                _perf_stats["last_capture_time"] = now
                last_fps_log = now
                print(f"[CAPTURE] {fps:.1f} FPS")
        
        except Exception as e:
            print(f"[ERROR] Frame capture error: {e}")
            time.sleep(0.1)
    
    print("[INFO] Frame capture worker stopped")


def detection_worker():
    """
    PARALLEL DETECTION: Processes frames from queue using thread pool
    Multiple detections can run simultaneously
    """
    global _streaming, _latest_detections, _latest_annotated_frame, _detection_executor, _perf_stats
    
    print("[INFO] Detection worker started")
    
    # Create thread pool for parallel YOLOv8 inference
    # Adjust based on your GPU/CPU - 2 workers is usually optimal
    _detection_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
    
    futures = []
    last_fps_log = time.time()
    
    while _streaming:
        try:
            # Get frame from queue (blocking with timeout)
            try:
                bgr = _frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # CRITICAL FIX: Add error handling for process_frame_bgr
            def safe_process_frame(frame):
                """Wrapper with error handling"""
                try:
                    return process_frame_bgr(frame, 0.55)
                except Exception as e:
                    print(f"[ERROR] Detection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return empty result
                    return frame, []
            
            # Submit detection to thread pool (non-blocking)
            future = _detection_executor.submit(safe_process_frame, bgr.copy())
            futures.append(future)
            
            # Clean up completed futures
            completed = [f for f in futures if f.done()]
            for future in completed:
                try:
                    annotated, detections = future.result(timeout=0.1)
                    
                    # Update shared state (fast lock)
                    with _lock:
                        _latest_detections = detections
                        _latest_annotated_frame = annotated
                    
                    # Send to cobot
                    _cobot_broadcaster.send(detections)
                    
                    # Performance tracking
                    _perf_stats["detection_count"] += 1
                    
                    # Try to put in result queue for streaming
                    try:
                        _result_queue.put_nowait(annotated)
                    except queue.Full:
                        pass
                    
                except Exception as e:
                    print(f"[ERROR] Detection result error: {e}")
            
            # Remove completed futures
            futures = [f for f in futures if not f.done()]
            
            # Limit queue depth to prevent memory issues
            if len(futures) > 4:
                print(f"[WARN] Detection queue backing up ({len(futures)} pending), clearing...")
                futures = futures[:2]  # Keep oldest 2 (closest to completion) — dropping newest on slow-CPU paths
            
            # Log FPS every 2 seconds
            now = time.time()
            if now - last_fps_log > 2.0:
                elapsed = now - _perf_stats["last_detection_time"]
                fps = _perf_stats["detection_count"] / elapsed if elapsed > 0 else 0
                _perf_stats["detection_fps"] = fps
                _perf_stats["detection_count"] = 0
                _perf_stats["last_detection_time"] = now
                last_fps_log = now
                print(f"[DETECTION] {fps:.1f} FPS | Queue: {len(futures)} | Detections: {len(_latest_detections)}")
        
        except Exception as e:
            print(f"[ERROR] Detection worker error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    # Cleanup
    if _detection_executor:
        _detection_executor.shutdown(wait=True)
        _detection_executor = None
    
    print("[INFO] Detection worker stopped")


def mjpeg_color():
    """
    ULTRA-FAST streaming: Just grabs latest frame and encodes
    Never blocks on detection
    """
    global _streaming, _latest_annotated_frame, _perf_stats
    
    start()
    _first_stream_client = not _streaming
    _streaming = True

    # Idempotency guard: only spawn worker threads for the FIRST caller.
    # Without this, every /stream/color request (browser refresh, curl
    # retry, second tab) spawns another capture+detection thread pair,
    # and they all race on the same queues.
    if _first_stream_client:
        capture_thread = threading.Thread(target=frame_capture_worker, daemon=True, name="capture")
        capture_thread.start()

        detection_thread = threading.Thread(target=detection_worker, daemon=True, name="detection")
        detection_thread.start()

        print("[INFO] Streaming started")
    else:
        print("[INFO] Additional stream client attached (workers already running)")
    last_fps_log = time.time()
    
    # Better fallback frame
    fallback_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(fallback_frame, "Initializing detector...", (50, HEIGHT//2 - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(fallback_frame, "Please wait...", (50, HEIGHT//2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Track if we ever got a real frame
    got_first_frame = False
    frames_without_detection = 0
    
    while _streaming and _pipe:
        try:
            # Try to get latest result from detection (non-blocking)
            frame = None
            try:
                frame = _result_queue.get_nowait()
                got_first_frame = True
                frames_without_detection = 0
            except queue.Empty:
                # No new detection result, use last known frame or fallback
                with _lock:
                    if _latest_annotated_frame is not None:
                        frame = _latest_annotated_frame
                        got_first_frame = True
                    else:
                        frame = fallback_frame
                        frames_without_detection += 1
            
            # If we haven't gotten a frame in a while, show warning
            if got_first_frame and frames_without_detection > 100:
                frame = fallback_frame.copy()
                cv2.putText(frame, "Detection stalled!", (50, HEIGHT//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Fast JPEG encode (~5-10ms at quality 80)
            encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY), 80,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ]
            _, buf = cv2.imencode(".jpg", frame, encode_params)
            
            # Stream immediately
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

            # Cap stream loop at ~30 FPS. Without this, when detection
            # is slow the queue-empty path uses the cached last frame
            # and the loop busy-spins, holding the GIL and starving
            # the detection worker of CPU.
            time.sleep(1.0 / 30.0)

            # Performance tracking
            _perf_stats["stream_count"] += 1
            
            # Log FPS every 2 seconds
            now = time.time()
            if now - last_fps_log > 2.0:
                elapsed = now - _perf_stats["last_stream_time"]
                fps = _perf_stats["stream_count"] / elapsed if elapsed > 0 else 0
                _perf_stats["stream_fps"] = fps
                _perf_stats["stream_count"] = 0
                _perf_stats["last_stream_time"] = now
                last_fps_log = now
                print(f"[STREAM] {fps:.1f} FPS")
        
        except Exception as e:
            print(f"[ERROR] Streaming error: {e}")
            time.sleep(0.01)
    
    print("[INFO] Streaming stopped")


def mjpeg_depth():
    """Depth stream (unchanged, already fast)"""
    start()
    while _pipe:
        frames = _pipe.wait_for_frames()
        if d := frames.get_depth_frame():
            arr = np.asanyarray(d.get_data())
            vis = cv2.applyColorMap(cv2.convertScaleAbs(arr, alpha=0.03), cv2.COLORMAP_JET)
            _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@router.post("/camera/start")
def camera_start():
    start()
    return JSONResponse({"ok": True})


@router.post("/camera/stop")
def camera_stop():
    stop()
    return JSONResponse({"ok": True})


@router.get("/camera/status")
def status():
    return JSONResponse({
        "started": _pipe is not None, 
        "streaming": _streaming,
        "performance": {
            "capture_fps": round(_perf_stats["capture_fps"], 1),
            "detection_fps": round(_perf_stats["detection_fps"], 1),
            "stream_fps": round(_perf_stats["stream_fps"], 1)
        }
    })


@router.get("/detections/latest")
def latest():
    with _lock:
        detections_copy = _latest_detections.copy()
    return JSONResponse({
        "detections": detections_copy, 
        "frame_size": {"width": WIDTH, "height": HEIGHT}
    })


@router.get("/detections/export")
def export(format: str = "json"):
    with _lock:
        detections_copy = _latest_detections.copy()
    
    if format == "python":
        return JSONResponse(
            content=f"detections = {detections_copy}", 
            media_type="text/plain"
        )
    return JSONResponse({"detections": detections_copy})


@router.get("/stream/color")
def stream_color():
    return StreamingResponse(
        mjpeg_color(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/stream/depth")
def stream_depth():
    return StreamingResponse(
        mjpeg_depth(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """WebSocket endpoint for real-time detection data"""
    await websocket.accept()
    print("[DEBUG] WebSocket client connected")
    
    try:
        while True:
            with _lock:
                detections_copy = _latest_detections.copy()
            
            await websocket.send_json({
                "timestamp": datetime.now().isoformat(),
                "detections": detections_copy,
                "frame_size": {"width": WIDTH, "height": HEIGHT}
            })
            
            await asyncio.sleep(0.033)  # ~30 Hz
            
    except WebSocketDisconnect:
        print("[DEBUG] WebSocket client disconnected")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
