from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.tpu_router import _lock, _latest_detections, WIDTH, HEIGHT
from datetime import datetime
import asyncio

router = APIRouter()

@router.websocket("/ws/detections")
async def websocket_detections_compat(websocket: WebSocket):
    """Backward compatible WebSocket endpoint for cobot Docker"""
    await websocket.accept()
    print("[DEBUG] WebSocket client connected (compat route)")
    try:
        while True:
            with _lock:
                detections_copy = _latest_detections.copy()
            await websocket.send_json({
                "timestamp": datetime.now().isoformat(),
                "detections": detections_copy,
                "frame_size": {"width": WIDTH, "height": HEIGHT}
            })
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        print("[DEBUG] WebSocket client disconnected (compat)")
    except Exception as e:
        print(f"[ERROR] WebSocket error (compat): {e}")
