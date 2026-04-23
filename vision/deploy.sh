#!/bin/bash
#
# OPSS Deployment Script
# Automated setup for Optical Projectile Sensing System
#
# Usage:
#   ./deploy.sh              # Full install + run
#   ./deploy.sh --install    # Install only
#   ./deploy.sh --run        # Run only (assumes installed)
#   ./deploy.sh --docker     # Run via Docker
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     OPSS - Optical Projectile Sensing System              ║"
echo "║     Deployment Script v2.0                                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect platform
detect_platform() {
    if [ -f /etc/nv_tegra_release ]; then
        PLATFORM="jetson"
        echo -e "${GREEN}[✓] Platform: NVIDIA Jetson${NC}"
        cat /etc/nv_tegra_release | head -1
    elif [ -f /proc/device-tree/model ] && grep -q "Raspberry" /proc/device-tree/model 2>/dev/null; then
        PLATFORM="raspi"
        echo -e "${GREEN}[✓] Platform: Raspberry Pi${NC}"
    else
        PLATFORM="linux"
        echo -e "${GREEN}[✓] Platform: Linux x86_64${NC}"
    fi
}

# Check for root (some operations need sudo)
check_sudo() {
    if [ "$EUID" -eq 0 ]; then
        SUDO=""
    else
        SUDO="sudo"
        echo -e "${YELLOW}[!] Some operations require sudo${NC}"
    fi
}

# Install system dependencies
install_system_deps() {
    echo -e "\n${BLUE}[1/6] Installing system dependencies...${NC}"

    $SUDO apt-get update
    $SUDO apt-get install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        cmake \
        pkg-config \
        libusb-1.0-0-dev \
        libgtk-3-dev \
        libglfw3-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        curl \
        git

    # Add user to plugdev for USB device access
    if ! groups | grep -q plugdev; then
        $SUDO usermod -aG plugdev $USER
        echo -e "${YELLOW}[!] Added $USER to plugdev group (re-login required for USB access)${NC}"
    fi

    echo -e "${GREEN}[✓] System dependencies installed${NC}"
}

# Install RealSense SDK
install_realsense() {
    echo -e "\n${BLUE}[2/6] Installing Intel RealSense SDK...${NC}"

    if python3 -c "import pyrealsense2" 2>/dev/null; then
        echo -e "${GREEN}[✓] pyrealsense2 already installed${NC}"
        return
    fi

    if [ "$PLATFORM" = "jetson" ]; then
        echo -e "${YELLOW}[!] Jetson detected - RealSense must be built from source${NC}"
        echo -e "${YELLOW}[!] See: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md${NC}"

        # Check if already installed
        if [ -f /usr/local/lib/python3.*/dist-packages/pyrealsense2*.so ]; then
            echo -e "${GREEN}[✓] RealSense SDK appears to be installed${NC}"
        else
            echo -e "${RED}[!] RealSense SDK not found - manual installation required${NC}"
            echo -e "${YELLOW}    Run these commands:${NC}"
            echo "    git clone https://github.com/IntelRealSense/librealsense.git"
            echo "    cd librealsense && mkdir build && cd build"
            echo "    cmake .. -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=/usr/bin/python3"
            echo "    make -j\$(nproc) && sudo make install"
        fi
    else
        # x86 Linux - use apt
        echo "Adding Intel RealSense repository..."

        $SUDO mkdir -p /etc/apt/keyrings
        curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | $SUDO tee /etc/apt/keyrings/librealsense.pgp > /dev/null

        echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
            $SUDO tee /etc/apt/sources.list.d/librealsense.list

        $SUDO apt-get update
        $SUDO apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev python3-pyrealsense2

        echo -e "${GREEN}[✓] RealSense SDK installed${NC}"
    fi
}

# Create Python virtual environment
setup_venv() {
    echo -e "\n${BLUE}[3/6] Setting up Python virtual environment...${NC}"

    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}[✓] Virtual environment created${NC}"
    else
        echo -e "${GREEN}[✓] Virtual environment exists${NC}"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip wheel setuptools

    # Link system pyrealsense2 into venv (required because it can't be pip installed)
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

    for pyrs in /usr/lib/python3/dist-packages/pyrealsense2* /usr/local/lib/python3.*/dist-packages/pyrealsense2*; do
        if [ -e "$pyrs" ]; then
            ln -sf "$pyrs" "$SITE_PACKAGES/" 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}[✓] Virtual environment configured${NC}"
}

# Install Python dependencies
install_python_deps() {
    echo -e "\n${BLUE}[4/6] Installing Python dependencies...${NC}"

    source "$VENV_DIR/bin/activate"

    # Install from requirements
    pip install -r "$SCRIPT_DIR/requirements.txt"

    # Jetson-specific: Install torch for ARM
    if [ "$PLATFORM" = "jetson" ]; then
        echo "Installing PyTorch for Jetson..."
        # Check if torch is already installed with CUDA
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo -e "${GREEN}[✓] PyTorch with CUDA already installed${NC}"
        else
            echo -e "${YELLOW}[!] Installing PyTorch for Jetson (this may take a while)...${NC}"
            pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
        fi
    fi

    echo -e "${GREEN}[✓] Python dependencies installed${NC}"
}

# Download model weights
download_models() {
    echo -e "\n${BLUE}[5/6] Checking model weights...${NC}"

    if [ -f "$SCRIPT_DIR/yolov8n.pt" ]; then
        echo -e "${GREEN}[✓] YOLOv8 model exists${NC}"
    else
        echo "Downloading YOLOv8n model..."
        source "$VENV_DIR/bin/activate"
        python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
        echo -e "${GREEN}[✓] YOLOv8 model downloaded${NC}"
    fi
}

# Verify installation
verify_install() {
    echo -e "\n${BLUE}[6/6] Verifying installation...${NC}"

    source "$VENV_DIR/bin/activate"

    echo -n "  Checking FastAPI... "
    python3 -c "import fastapi" && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}"

    echo -n "  Checking OpenCV... "
    python3 -c "import cv2" && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}"

    echo -n "  Checking PyTorch... "
    python3 -c "import torch; print(f'OK (CUDA: {torch.cuda.is_available()})')" || echo -e "${RED}FAIL${NC}"

    echo -n "  Checking Ultralytics... "
    python3 -c "from ultralytics import YOLO" && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}"

    echo -n "  Checking pyrealsense2... "
    python3 -c "import pyrealsense2" && echo -e "${GREEN}OK${NC}" || echo -e "${YELLOW}NOT FOUND (camera won't work)${NC}"

    echo -n "  Checking OPSS package... "
    python3 -c "from opss import OPSSPipeline" && echo -e "${GREEN}OK${NC}" || echo -e "${RED}FAIL${NC}"
}

# Run the application
run_app() {
    echo -e "\n${BLUE}Starting OPSS...${NC}"

    source "$VENV_DIR/bin/activate"
    cd "$SCRIPT_DIR"

    # Get local IP for access info
    LOCAL_IP=$(hostname -I | awk '{print $1}')

    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  OPSS is starting...                                      ║"
    echo "║                                                           ║"
    echo "║  Dashboard: http://$LOCAL_IP:8000/                   ║"
    echo "║  API Docs:  http://$LOCAL_IP:8000/docs               ║"
    echo "║                                                           ║"
    echo "║  Press Ctrl+C to stop                                     ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    python3 main.py
}

# Docker deployment
run_docker() {
    echo -e "\n${BLUE}Starting OPSS via Docker...${NC}"

    if [ ! -f "$SCRIPT_DIR/Dockerfile" ]; then
        echo -e "${YELLOW}[!] Creating Dockerfile...${NC}"
        cat > "$SCRIPT_DIR/Dockerfile" << 'DOCKERFILE'
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
DOCKERFILE
    fi

    if [ ! -f "$SCRIPT_DIR/docker-compose.yml" ]; then
        echo -e "${YELLOW}[!] Creating docker-compose.yml...${NC}"
        cat > "$SCRIPT_DIR/docker-compose.yml" << 'COMPOSE'
version: '3.8'

services:
  opss:
    build: .
    ports:
      - "8000:8000"
    devices:
      - /dev/bus/usb:/dev/bus/usb
    volumes:
      - /tmp:/tmp
    privileged: true
    restart: unless-stopped

    # Uncomment for Jetson with GPU
    # runtime: nvidia
    # environment:
    #   - NVIDIA_VISIBLE_DEVICES=all
COMPOSE
    fi

    docker compose up --build
}

# Create systemd service
install_service() {
    echo -e "\n${BLUE}Installing OPSS as systemd service...${NC}"

    SERVICE_FILE="/etc/systemd/system/opss.service"

    $SUDO tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=OPSS - Optical Projectile Sensing System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV_DIR/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    $SUDO systemctl daemon-reload
    $SUDO systemctl enable opss

    echo -e "${GREEN}[✓] Service installed${NC}"
    echo -e "    Start:   sudo systemctl start opss"
    echo -e "    Stop:    sudo systemctl stop opss"
    echo -e "    Status:  sudo systemctl status opss"
    echo -e "    Logs:    journalctl -u opss -f"
}

# Main
main() {
    cd "$SCRIPT_DIR"
    detect_platform
    check_sudo

    case "${1:-}" in
        --install)
            install_system_deps
            install_realsense
            setup_venv
            install_python_deps
            download_models
            verify_install
            echo -e "\n${GREEN}[✓] Installation complete!${NC}"
            echo -e "    Run: ./deploy.sh --run"
            ;;
        --run)
            run_app
            ;;
        --docker)
            run_docker
            ;;
        --service)
            install_service
            ;;
        --verify)
            setup_venv
            verify_install
            ;;
        --help|-h)
            echo "Usage: ./deploy.sh [OPTION]"
            echo ""
            echo "Options:"
            echo "  (none)      Full install and run"
            echo "  --install   Install dependencies only"
            echo "  --run       Run OPSS (assumes installed)"
            echo "  --docker    Run via Docker"
            echo "  --service   Install as systemd service"
            echo "  --verify    Verify installation"
            echo "  --help      Show this help"
            ;;
        *)
            # Full install + run
            install_system_deps
            install_realsense
            setup_venv
            install_python_deps
            download_models
            verify_install
            run_app
            ;;
    esac
}

main "$@"
