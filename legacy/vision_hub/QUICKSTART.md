# OPSS Quick Start Guide

Get OPSS running in under 10 minutes! ⚡

## ✅ Prerequisites Checklist

Before starting, ensure you have:

- [ ] Ubuntu 20.04 or 22.04 (or compatible Linux)
- [ ] Intel RealSense D435/D435i camera
- [ ] Google Coral Edge TPU USB Accelerator
- [ ] USB 3.0 ports available
- [ ] Sudo/admin access to your machine

## 🚀 Installation (Automated)

### Option 1: One-Line Install (Recommended)

```bash
# Clone and run setup script
git clone <your-repo-url> && cd opss && chmod +x setup.sh && ./setup.sh
```

The script will:
1. Install system dependencies
2. Set up RealSense SDK
3. Install Coral Edge TPU runtime
4. Create Python virtual environment
5. Install all Python packages
6. Verify installation

**Time: ~8-15 minutes** (depending on internet speed)

### Option 2: Manual Install

If you prefer manual control, follow these steps:

```bash
# 1. Clone repository
git clone <your-repo-url>
cd opss

# 2. Run system setup
sudo apt update && sudo apt install -y python3-pip python3-venv libusb-1.0-0

# 3. Install RealSense
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
    sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt update && sudo apt install -y librealsense2-dkms librealsense2-utils python3-pyrealsense2

# 4. Install Edge TPU
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update && sudo apt install -y libedgetpu1-std python3-pycoral

# 5. Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Link system packages
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ln -sf /usr/lib/python3/dist-packages/pyrealsense2* venv/lib/python${PYTHON_VERSION}/site-packages/
ln -sf /usr/lib/python3/dist-packages/pycoral* venv/lib/python${PYTHON_VERSION}/site-packages/
ln -sf /usr/lib/python3/dist-packages/tflite_runtime* venv/lib/python${PYTHON_VERSION}/site-packages/

# 7. Add user to plugdev group (logout/login required)
sudo usermod -a -G plugdev $USER
```

## 🔌 Hardware Setup

### 1. Connect RealSense Camera

```bash
# Connect to USB 3.0 port (blue port)
# Verify detection:
lsusb | grep Intel
# Should show: "Intel Corp. RealSense Camera"

# Test camera:
realsense-viewer
```

### 2. Connect Edge TPU

```bash
# Connect to USB 3.0 port
# Verify detection:
lsusb | grep Google
# Should show: "Google Inc."

# Test Edge TPU:
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
# Should show: [<EdgeTpuDevice ...>]
```

**⚠️ Important:** If you just added yourself to the `plugdev` group, **log out and back in** before the devices will work!

## ▶️ Running OPSS

### Start the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Access the Interface

Open your browser to: **http://localhost:8000**

## 🎮 Using OPSS

### Quick Test Sequence

1. **Click "RealSense Interface" tab**

2. **Start Camera**
   - Click **"Start"** button
   - Status should change to "Running"
   - Device info should appear

3. **Start Streaming**
   - Click **"Start Camera"** (color stream)
   - You should see live video with person detection boxes
   - Green boxes appear around detected people

4. **View Detection Data**
   - Watch the detection counter update
   - Click **"Export JSON"** to download detection data
   - Click **"Export Python"** to get Python script with detections

5. **Check Logs**
   - Click **"Logs"** tab
   - View real-time system logs

### Keyboard Shortcuts

- `Space` - Toggle streaming
- `Ctrl+K` - Connect/disconnect camera
- `C` - Capture frame

## 📊 Verifying Performance

### Expected Performance

| Metric | Expected Value |
|--------|----------------|
| FPS (Color Stream) | ~25-30 FPS |
| Detection Latency | ~30-40ms per frame |
| CPU Usage | 30-50% on quad-core |
| Memory Usage | ~500-800 MB |

### Check Performance

```bash
# In separate terminal, monitor system resources
htop

# Or check specific process
top -p $(pgrep -f "uvicorn app.main:app")
```

### If Performance is Low

1. **Switch to Max Performance Edge TPU**
   ```bash
   sudo apt remove libedgetpu1-std
   sudo apt install libedgetpu1-max
   # Note: Runs hotter!
   ```

2. **Reduce Frame Resolution**
   Edit `app/tpu_router.py`:
   ```python
   WIDTH, HEIGHT = 320, 240  # Lower resolution
   ```

3. **Lower Detection Threshold**
   Edit `app/tpu_router.py`:
   ```python
   result = process_frame_bgr(bgr, 0.65)  # Higher threshold = fewer false positives
   ```

## 🐛 Troubleshooting

### Camera Not Found

```bash
# Check if camera is detected
lsusb | grep Intel

# If not showing, try different USB port
# Must be USB 3.0 (blue port)

# Check permissions
ls -l /dev/video*
# Should be readable by your user or plugdev group
```

### Edge TPU Not Working

```bash
# Check device file
ls -l /dev/apex_0

# If missing, reinstall runtime
sudo apt reinstall libedgetpu1-std

# Verify driver loaded
lsmod | grep apex
```

### Import Errors (pyrealsense2, pycoral)

These packages MUST be installed system-wide, not via pip:

```bash
# Check system installation
dpkg -l | grep realsense
dpkg -l | grep pycoral

# If missing, reinstall
sudo apt install --reinstall python3-pyrealsense2 python3-pycoral

# Re-link to venv
source venv/bin/activate
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ln -sf /usr/lib/python3/dist-packages/pyrealsense2* venv/lib/python${PYTHON_VERSION}/site-packages/
ln -sf /usr/lib/python3/dist-packages/pycoral* venv/lib/python${PYTHON_VERSION}/site-packages/
```

### Port Already in Use

```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

## 🎯 Next Steps

Now that OPSS is running:

1. **Read the full [README.md](README.md)** for detailed documentation
2. **Check [CONTRIBUTING.md](CONTRIBUTING.md)** if you want to develop
3. **See [JETSON_MIGRATION.md](JETSON_MIGRATION.md)** for Jetson Orin Nano setup
4. **Explore the API** at http://localhost:8000/docs (FastAPI auto-docs)

## 🆘 Getting Help

If you're stuck:

1. **Check logs** in the Logs tab
2. **Review [README.md](README.md)** Troubleshooting section
3. **Run diagnostics**:
   ```bash
   python3 -c "import pyrealsense2; import cv2; from pycoral.utils import edgetpu; print('All imports OK')"
   ```
4. **Open an issue** on GitHub with:
   - Your OS version
   - Error messages
   - Output of `lsusb`
   - Steps to reproduce

## 📝 Common Commands Reference

```bash
# Start server
source venv/bin/activate && uvicorn app.main:app --reload

# Check camera
realsense-viewer

# Test Edge TPU
python3 -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"

# View logs
tail -f <log_file>  # If logging configured

# Stop server
# Press Ctrl+C in terminal
```

## ✨ You're All Set!

OPSS is now running and detecting people in real-time. Enjoy! 🎉

For production deployment, see the Production section in [README.md](README.md).
