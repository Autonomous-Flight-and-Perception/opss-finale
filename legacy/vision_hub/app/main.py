from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

# Import the router
from app.jetson_router import router

# Create FastAPI app
app = FastAPI(title="OPSS Vision Hub", version="1.0.0")

# Include the router
app.include_router(router)

# Serve static files if they exist
if os.path.exists("/app/app/static"):
    app.mount("/static", StaticFiles(directory="/app/app/static"), name="static")

@app.get("/")
def root():
    """Root endpoint - serve dashboard or status"""
    if os.path.exists("/app/app/static/index.html"):
        return FileResponse("/app/app/static/index.html")
    return {
        "service": "OPSS Vision Hub",
        "status": "running",
        "endpoints": {
            "camera_control": ["/camera/start", "/camera/stop", "/camera/status"],
            "detections": ["/detections/latest", "/detections/export"],
            "streams": ["/stream/color", "/stream/depth"],
            "websocket": "/ws/detections"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
