# OPSS - Optical Projectile Sensing System

Real-time person detection and tracking system using Intel RealSense cameras with Edge TPU acceleration.

## 🎯 Project Overview

OPSS is a computer vision system that:
- Captures RGB and depth streams from Intel RealSense D435 cameras
- Performs real-time person detection using MobileNet SSD
- Provides bounding box coordinates with corner points
- Exports detection data in JSON and Python formats
- Offers a web-based interface for monitoring and control

## 🔧 Hardware Requirements

### Current Setup (Coral Edge TPU)
- Intel RealSense D435 or D435i camera
- Google Coral Edge TPU USB Accelerator
- Host machine with USB 3.0 ports
- Recommended: 4GB+ RAM

### Future Setup (Jetson Orin Nano)
See [Migration Guide](#-migrating-to-jetson-orin-nano) below

## 📋 Software Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04 / 22.04 (recommended) or Debian-based Linux
- **Python**: 3.8 - 3.10 (3.9 recommended)
- **USB Access**: User must be in `plugdev` group

### Critical Dependencies
This project has **strict dependency requirements**. Follow the order below carefully.

## 🚀 Installation Guide

### Step 1: System Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    libusb-1.0-0 \
    udev \
    curl \
    git

# Add user to plugdev group for USB access
sudo usermod -a -G plugdev $USER
# Log out and back in for group changes to take effect
```

### Step 2: Intel RealSense SDK

**⚠️ INSTALL THIS BEFORE CREATING PYTHON ENVIRONMENT**

```bash
# Add Intel RealSense repository
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
    sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
https://librealsense.intel.com/Debian/apt-repo \
$(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt update

# Install RealSense SDK and Python bindings
sudo apt install -y \
    librealsense2-dkms \
    librealsense2-utils \
    librealsense2-dev \
    python3-pyrealsense2

# Verify installation
realsense-viewer
# Camera should appear in viewer if connected
```

### Step 3: Coral Edge TPU Setup

```bash
# Add Coral repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo apt-key add -

sudo apt update

# Install Edge TPU runtime (standard performance)
sudo apt install -y libedgetpu1-std

# For maximum performance (runs hotter):
# sudo apt install -y libedgetpu1-max

# Install PyCoral
sudo apt install -y python3-pycoral
```

### Step 4: Python Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd opss

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Check RealSense camera
python3 -c "import pyrealsense2 as rs; print('RealSense OK')"

# Check Edge TPU
python3 -c "from pycoral.utils import edgetpu; print('Edge TPU OK')"

# Check OpenCV
python3 -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# Test Edge TPU device
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```

## 📦 Project Structure

```
opss/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry
│   ├── tpu_router.py              # Main API routes + streaming
│   ├── websocket_compat.py        # Backward compatible WebSocket
│   ├── edgetpu_inference.py       # Edge TPU model wrapper
│   ├── infer_tpu.py               # Frame processing + detection
│   ├── models/
│   │   ├── mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
│   │   └── coco_labels.txt
│   └── static/
│       ├── index.html
│       ├── style.css
│       ├── app.js
│       └── logo.png
├── requirements.txt
├── README.md
└── .env (optional)
```

## 🎮 Running the Application

### Development Mode

```bash
# Activate environment
source venv/bin/activate

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Run with Gunicorn
gunicorn app.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Access the Interface

Open browser to: `http://localhost:8000`

## 📡 API Endpoints

### Camera Control
- `POST /api/api/camera/start` - Start camera
- `POST /api/api/camera/stop` - Stop camera
- `GET /api/api/camera/status` - Get camera status

### Streaming
- `GET /api/api/stream/color` - MJPEG color stream with detections
- `GET /api/api/stream/depth` - MJPEG depth stream (colorized)

### Detection Data
- `GET /api/detections/latest` - Get latest detection frame (JSON)
- `GET /api/detections/export?format=json` - Export detections as JSON
- `GET /api/detections/export?format=python` - Export as Python script
- `WebSocket /api/ws/detections` - Real-time detection stream (30 FPS)

### Detection Data Format

```json
{
  "timestamp": "2025-10-10T14:30:00.123456",
  "frame_size": {"width": 640, "height": 480},
  "detections": [
    {
      "class": "person",
      "confidence": 0.87,
      "corners": {
        "top_left": {"x": 120, "y": 80},
        "top_right": {"x": 320, "y": 80},
        "bottom_left": {"x": 120, "y": 400},
        "bottom_right": {"x": 320, "y": 400}
      },
      "bbox": {"x1": 120, "y1": 80, "x2": 320, "y2": 400},
      "center": {"x": 220, "y": 240}
    }
  ]
}
```

## 🔍 Troubleshooting

### Camera Not Detected
```bash
# Check USB connection
lsusb | grep Intel

# Test with RealSense viewer
realsense-viewer

# Check permissions
groups $USER  # Should include 'plugdev'
```

### Edge TPU Not Found
```bash
# Check USB connection
lsusb | grep Google

# Verify driver installation
ls /dev/apex_0  # Should exist

# Reinstall runtime if needed
sudo apt reinstall libedgetpu1-std
```

### Import Errors
```bash
# Ensure you're in virtual environment
which python  # Should point to venv/bin/python

# Reinstall pyrealsense2 system-wide
sudo apt install --reinstall python3-pyrealsense2

# Link to venv (if needed)
ln -s /usr/lib/python3/dist-packages/pyrealsense2* venv/lib/python3.*/site-packages/
```

### Low FPS / Performance Issues
- Switch to `libedgetpu1-max` for maximum performance
- Reduce frame resolution in `tpu_router.py` (WIDTH, HEIGHT variables)
- Lower detection threshold in `infer_tpu.py`
- Check CPU usage: `htop`

## 🚢 Migrating to Jetson Orin Nano

### Why Migrate?
The Jetson Orin Nano provides:
- More powerful GPU (1024 CUDA cores)
- Native TensorRT support
- Better thermal management
- Integrated solution (no USB accelerator needed)

### Migration Steps

1. **Convert Model to TensorRT**
```bash
# Install TensorRT (comes with JetPack)
# Convert TFLite to ONNX, then to TensorRT engine
```

2. **Update Inference Code**
```python
# Replace edgetpu_inference.py with TensorRT inference
# Use tensorrt, pycuda libraries instead of pycoral
```

3. **Optimize for Jetson**
- Use CUDA for preprocessing
- Enable DLA (Deep Learning Accelerator) if available
- Batch inference for multiple detections

4. **Installation on Jetson**
```bash
# Flash
