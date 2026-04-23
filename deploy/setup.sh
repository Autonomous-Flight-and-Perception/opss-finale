#!/bin/bash
set -e

echo "============================================"
echo "  OPSS Unified System Setup"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on correct hardware
echo -e "${YELLOW}Checking system...${NC}"
if [ -f /etc/nv_tegra_release ]; then
    echo -e "${GREEN}✓ Detected NVIDIA Jetson platform${NC}"
    PLATFORM="jetson"
elif lsusb | grep -q "Arduino"; then
    echo -e "${GREEN}✓ Detected cobot hardware${NC}"
    PLATFORM="dev"
else
    echo -e "${YELLOW}⚠ Running in development mode (no hardware detected)${NC}"
    PLATFORM="dev"
fi

# Install Docker if needed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}✓ Docker installed${NC}"
    echo -e "${RED}⚠ Please log out and log back in for Docker permissions${NC}"
    exit 1
fi

# Install Docker Compose V2 if needed
if ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}Installing Docker Compose V2...${NC}"
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
    echo -e "${GREEN}✓ Docker Compose V2 installed${NC}"
fi

# Add user to dialout group for USB devices
if ! groups $USER | grep -q dialout; then
    echo -e "${YELLOW}Adding user to dialout group for USB access...${NC}"
    sudo usermod -aG dialout $USER
    echo -e "${YELLOW}⚠ You need to log out and log back in for this to take effect${NC}"
fi

# Check for required repositories
echo -e "${YELLOW}Checking repositories...${NC}"

REPOS_MISSING=0
if [ ! -d "../b2" ]; then
    echo -e "${RED}✗ b2 repository not found${NC}"
    REPOS_MISSING=1
fi
if [ ! -d "../vision_hub" ]; then
    echo -e "${RED}✗ vision_hub repository not found${NC}"
    REPOS_MISSING=1
fi
if [ ! -d "../cobotpy" ]; then
    echo -e "${RED}✗ cobotpy repository not found${NC}"
    REPOS_MISSING=1
fi

if [ $REPOS_MISSING -eq 1 ]; then
    echo -e "${RED}Missing repositories. Please clone them first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All repositories found${NC}"

# Create Unix socket directory
sudo mkdir -p /tmp
sudo chmod 777 /tmp

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "To start the unified system:"
echo "  cd ~/opss-union"
echo "  ./start.sh"
echo ""
echo "To stop the system:"
echo "  ./stop.sh"
echo ""
