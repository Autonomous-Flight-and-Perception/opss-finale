"""
OPSS Web Application
FastAPI-based web interface for the OPSS pipeline.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional

from ..pipeline.core import OPSSPipeline, PipelineConfig, get_pipeline


# Create FastAPI app
app = FastAPI(
    title="OPSS - Optical Projectile Sensing System",
    description="Real-time object detection, tracking, and physics validation",
    version="2.0.0"
)

# Pipeline instance
_pipeline: Optional[OPSSPipeline] = None


def get_app_pipeline() -> OPSSPipeline:
    """Get or create the pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = get_pipeline()
    return _pipeline


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    index_path = static_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="""
    <html>
        <head><title>OPSS</title></head>
        <body>
            <h1>OPSS - Optical Projectile Sensing System</h1>
            <p>Static files not found. API endpoints available at /docs</p>
        </body>
    </html>
    """)


@app.post("/pipeline/start")
async def start_pipeline():
    """Start the OPSS pipeline"""
    pipeline = get_app_pipeline()
    success = pipeline.start()
    return JSONResponse({
        "ok": success,
        "message": "Pipeline started" if success else "Failed to start pipeline"
    })


@app.post("/pipeline/stop")
async def stop_pipeline():
    """Stop the OPSS pipeline"""
    pipeline = get_app_pipeline()
    pipeline.stop()
    return JSONResponse({"ok": True, "message": "Pipeline stopped"})


@app.get("/pipeline/status")
async def pipeline_status():
    """Get pipeline status and statistics"""
    pipeline = get_app_pipeline()
    return JSONResponse({
        "running": pipeline._running,
        "stats": pipeline.get_stats(),
        "error_statistics": pipeline.get_error_statistics(),
        "config": {
            "capture_resolution": f"{pipeline.config.capture_width}x{pipeline.config.capture_height}",
            "inference_resolution": f"{pipeline.config.infer_width}x{pipeline.config.infer_height}",
            "mode": "dual-resolution"
        }
    })


@app.get("/detections/latest")
async def get_detections():
    """Get latest raw detections"""
    pipeline = get_app_pipeline()
    return JSONResponse({
        "detections": pipeline.get_latest_detections(),
        "frame_size": {
            "width": pipeline.config.capture_width,
            "height": pipeline.config.capture_height
        }
    })


@app.get("/states/latest")
async def get_states():
    """Latest states for the *primary* tracker (back-compat)."""
    pipeline = get_app_pipeline()
    states = pipeline.get_latest_states()
    return JSONResponse({
        "states": states,
        "count": len(states),
        "tracker": pipeline.get_primary_tracker(),
    })


@app.get("/states/{tracker}/latest")
async def get_states_for_tracker(tracker: str):
    """Latest states for a specific tracker (e.g. /states/kalman/latest)."""
    pipeline = get_app_pipeline()
    if tracker not in pipeline.get_active_trackers():
        return JSONResponse(
            {"error": f"tracker '{tracker}' not active",
             "active": pipeline.get_active_trackers()},
            status_code=404,
        )
    states = pipeline.get_latest_states(tracker)
    return JSONResponse({
        "states": states,
        "count": len(states),
        "tracker": tracker,
    })


@app.get("/fused/latest")
async def get_fused():
    """Latest fused states for the *primary* tracker (back-compat)."""
    pipeline = get_app_pipeline()
    fused = pipeline.get_latest_fused()
    return JSONResponse({
        "fused_states": fused,
        "count": len(fused),
        "tracker": pipeline.get_primary_tracker(),
    })


@app.get("/fused/{tracker}/latest")
async def get_fused_for_tracker(tracker: str):
    """Latest fused states for a specific tracker (e.g. /fused/ukf/latest)."""
    pipeline = get_app_pipeline()
    if tracker not in pipeline.get_active_trackers():
        return JSONResponse(
            {"error": f"tracker '{tracker}' not active",
             "active": pipeline.get_active_trackers()},
            status_code=404,
        )
    fused = pipeline.get_latest_fused(tracker)
    return JSONResponse({
        "fused_states": fused,
        "count": len(fused),
        "tracker": tracker,
    })


@app.get("/trackers")
async def list_trackers():
    """List active trackers and which one is primary (drives the cobot)."""
    pipeline = get_app_pipeline()
    return JSONResponse({
        "active": pipeline.get_active_trackers(),
        "primary": pipeline.get_primary_tracker(),
    })


@app.get("/diagnostics")
async def get_diagnostics():
    """Get diagnostic statistics (prediction errors)"""
    pipeline = get_app_pipeline()
    return JSONResponse({
        "error_statistics": pipeline.get_error_statistics(),
        "stats": pipeline.get_stats()
    })


def generate_mjpeg_stream():
    """Generate MJPEG stream from pipeline"""
    pipeline = get_app_pipeline()

    # Ensure pipeline is running
    if not pipeline._running:
        pipeline.start()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

    while pipeline._running:
        frame = pipeline.get_latest_frame()

        if frame is None:
            # Return placeholder
            placeholder = np.zeros(
                (pipeline.config.capture_height, pipeline.config.capture_width, 3),
                dtype=np.uint8
            )
            cv2.putText(placeholder, "Waiting for frames...",
                       (100, placeholder.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            frame = placeholder

        _, buf = cv2.imencode(".jpg", frame, encode_params)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

        time.sleep(0.033)  # ~30 FPS


@app.get("/stream/color")
async def stream_color():
    """Stream annotated color frames with detections"""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/states")
async def websocket_states(websocket: WebSocket):
    """
    Real-time state stream.

    Top-level ``states`` and ``fused`` are the PRIMARY tracker's outputs
    (back-compat: existing cobot consumer reads these). ``trackers`` is
    a per-filter dict keyed by tracker name with each filter's
    independent states + fused outputs — use this for parallel
    comparison / side-by-side visualization.
    """
    await websocket.accept()
    print("[WS] Client connected to /ws/states")
    pipeline = get_app_pipeline()

    try:
        while True:
            states_all = pipeline.get_latest_states_all()
            fused_all = pipeline.get_latest_fused_all()
            primary = pipeline.get_primary_tracker()

            data = {
                "timestamp": time.time(),
                "detections": pipeline.get_latest_detections(),
                "states": states_all.get(primary, []),       # primary, back-compat
                "fused": fused_all.get(primary, []),         # primary, back-compat
                "primary": primary,
                "trackers": {                                # per-filter view
                    name: {
                        "states": states_all.get(name, []),
                        "fused":  fused_all.get(name, []),
                    }
                    for name in pipeline.get_active_trackers()
                },
                "stats": pipeline.get_stats(),
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.033)  # ~30 Hz

    except WebSocketDisconnect:
        print("[WS] Client disconnected from /ws/states")
    except Exception as e:
        print(f"[WS] Error: {e}")


@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """WebSocket endpoint for raw detections (legacy compatibility)"""
    await websocket.accept()
    print("[WS] Client connected to /ws/detections")

    pipeline = get_app_pipeline()

    try:
        while True:
            data = {
                "timestamp": time.time(),
                "detections": pipeline.get_latest_detections(),
                "frame_size": {
                    "width": pipeline.config.capture_width,
                    "height": pipeline.config.capture_height
                }
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        print("[WS] Client disconnected from /ws/detections")
    except Exception as e:
        print(f"[WS] Error: {e}")


# Legacy endpoints for backward compatibility with Vh
@app.post("/camera/start")
async def camera_start():
    """Legacy: Start camera (redirects to pipeline start)"""
    return await start_pipeline()


@app.post("/camera/stop")
async def camera_stop():
    """Legacy: Stop camera (redirects to pipeline stop)"""
    return await stop_pipeline()


@app.get("/camera/status")
async def camera_status():
    """Legacy: Get camera status"""
    pipeline = get_app_pipeline()
    stats = pipeline.get_stats()
    return JSONResponse({
        "started": pipeline._running,
        "streaming": pipeline._running,
        "resolution": {
            "capture": {
                "width": pipeline.config.capture_width,
                "height": pipeline.config.capture_height
            },
            "inference": {
                "width": pipeline.config.infer_width,
                "height": pipeline.config.infer_height
            },
            "mode": "dual-resolution"
        },
        "performance": {
            "capture_fps": stats.get("pipeline_fps", 0),
            "detection_fps": stats.get("pipeline_fps", 0),
            "stream_fps": stats.get("pipeline_fps", 0)
        }
    })


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the OPSS web server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
