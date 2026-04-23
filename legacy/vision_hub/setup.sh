#!/bin/bash
# OPSS Automated Setup Script
# For Ubuntu 20.04/22.04 with Coral Edge TPU

set -e  # Exit on error

echo "=========================================="
echo "OPSS Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect platform
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    echo -e "${GREEN}Detected: Jetson platform${NC}"
else
    PLATFORM="coral"
    echo -e "${GREEN}Detected: Standard Linux (Coral setup)${NC}"
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}Please do not run as root/sudo${NC}"
    echo "The script will ask for sudo password when needed"
    exit 1
fi

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

echo ""
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
check_success "System update"

echo ""
echo "Step 2: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    libusb-1.0-0 \
    udev \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config
check_success "System dependencies"

echo ""
echo "Step 3: Adding user to plugdev group..."
sudo usermod -a -G plugdev $USER
check_success "User group setup"
echo -e "${YELLOW}Note: You'll need to log out and back in for group changes to take effect${NC}"

if [ "$PLATFORM" == "coral" ]; then
    echo ""
    echo "Step 4: Installing Intel RealSense SDK..."
    
    # Check if already installed
    if dpkg -l | grep -q librealsense2; then
        echo -e "${YELLOW}RealSense already installed, skipping...${NC}"
    else
        sudo mkdir -p /etc/apt/keyrings
        curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
            sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
        
        echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
            sudo tee /etc/apt/sources.list.d/librealsense.list
        
        sudo apt update
        sudo apt install -y \
            librealsense2-dkms \
            librealsense2-utils \
            librealsense2-dev \
            python3-pyrealsense2
        check_success "RealSense SDK"
    fi

    echo ""
    echo "Step 5: Installing Coral Edge TPU runtime..."
    
    if dpkg -l | grep -q libedgetpu; then
        echo -e "${YELLOW}Edge TPU runtime already installed, skipping...${NC}"
    else
        echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
            sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
        
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
        
        sudo apt update
        
        echo -e "${YELLOW}Choose Edge TPU performance mode:${NC}"
        echo "1) Standard (recommended, cooler)"
        echo "2) Maximum (faster, runs hotter)"
        read -p "Enter choice [1-2]: " choice
        
        if [ "$choice" == "2" ]; then
            sudo apt install -y libedgetpu1-max
        else
            sudo apt install -y libedgetpu1-std
        fi
        
        sudo apt install -y python3-pycoral
        check_success "Edge TPU runtime"
    fi

elif [ "$PLATFORM" == "jetson" ]; then
    echo ""
    echo "Step 4: Installing RealSense SDK (from source for Jetson)..."
    
    if [ -d ~/librealsense ]; then
        echo -e "${YELLOW}RealSense source already exists, skipping clone...${NC}"
    else
        cd ~
        git clone https://github.com/IntelRealSense/librealsense.git
        cd librealsense
        
        mkdir -p build && cd build
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_PYTHON_BINDINGS=ON \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DFORCE_RSUSB_BACKEND=ON \
            -DBUILD_WITH_CUDA=ON
        
        make -j$(nproc)
        sudo make install
        sudo ldconfig
        check_success "RealSense SDK (Jetson)"
    fi
fi

echo ""
echo "Step 6: Setting up Python virtual environment..."
cd "$(dirname "$0")"

if [ -d venv ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Recreate? (y/N): " recreate
    if [ "$recreate" == "y" ] || [ "$recreate" == "Y" ]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi
check_success "Virtual environment"

source venv/bin/activate

echo ""
echo "Step 7: Upgrading pip..."
pip install --upgrade pip
check_success "Pip upgrade"

echo ""
echo "Step 8: Installing Python dependencies..."
if [ "$PLATFORM" == "jetson" ] && [ -f requirements.jetson.txt ]; then
    pip install -r requirements.jetson.txt
else
    pip install -r requirements.txt
fi
check_success "Python dependencies"

# Link system packages to venv if needed
echo ""
echo "Step 9: Linking system Python packages..."
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITE_PACKAGES="venv/lib/python${PYTHON_VERSION}/site-packages"

if [ "$PLATFORM" == "coral" ]; then
    # Link pyrealsense2
    if [ -d "/usr/lib/python3/dist-packages/pyrealsense2" ]; then
        ln -sf /usr/lib/python3/dist-packages/pyrealsense2* $SITE_PACKAGES/
        echo -e "${GREEN}✓ Linked pyrealsense2${NC}"
    fi
    
    # Link pycoral
    if [ -d "/usr/lib/python3/dist-packages/pycoral" ]; then
        ln -sf /usr/lib/python3/dist-packages/pycoral* $SITE_PACKAGES/
        ln -sf /usr/lib/python3/dist-packages/tflite_runtime* $SITE_PACKAGES/
        echo -e "${GREEN}✓ Linked pycoral${NC}"
    fi
fi

echo ""
echo "Step 10: Verifying installation..."

echo -n "  Checking pyrealsense2... "
if python -c "import pyrealsense2 as rs" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo -e "${YELLOW}  You may need to manually link pyrealsense2${NC}"
fi

if [ "$PLATFORM" == "coral" ]; then
    echo -n "  Checking pycoral... "
    if python -c "from pycoral.utils import edgetpu" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo -e "${YELLOW}  You may need to manually link pycoral${NC}"
    fi
    
    echo -n "  Checking Edge TPU device... "
    if python -c "from pycoral.utils.edgetpu import list_edge_tpus; devs=list_edge_tpus(); exit(0 if devs else 1)" 2>/dev/null; then
        echo -e "${GREEN}FOUND${NC}"
    else
        echo -e "${YELLOW}NOT FOUND - Make sure Edge TPU is connected${NC}"
    fi
fi

echo -n "  Checking OpenCV... "
if python -c "import cv2" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

echo -n "  Checking FastAPI... "
if python -c "import fastapi" 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Log out and back in (for group changes)"
echo "2. Connect your RealSense camera"
if [ "$PLATFORM" == "coral" ]; then
    echo "3. Connect your Coral Edge TPU"
fi
echo "4. Activate environment: source venv/bin/activate"
echo "5. Run application: uvicorn app.main:app --reload"
echo "6. Open browser to: http://localhost:8000"
echo ""
echo "For troubleshooting, see README.md"
echo ""
