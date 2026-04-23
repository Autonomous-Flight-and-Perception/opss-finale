#!/bin/bash
# One-shot host setup for the OPSS cobot-container deploy path.
#   - Installs Docker + Docker Compose V2 if missing
#   - Adds the current user to the `dialout` group for /dev/ttyACM* access
# Vision itself runs on the Jetson host (not inside this compose);
# see the repo README for vision bring-up steps.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}[opss-finale] Host setup${NC}"

if [ -f /etc/nv_tegra_release ]; then
    echo -e "${GREEN}✓ NVIDIA Jetson platform detected${NC}"
else
    echo -e "${YELLOW}⚠ Non-Jetson host — vision service must be run with compatible PyTorch + RealSense${NC}"
fi

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Installing Docker…${NC}"
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker "$USER"
    echo -e "${RED}⚠ Log out/in to pick up docker group membership, then re-run.${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}Installing Docker Compose V2…${NC}"
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
fi

if ! groups "$USER" | grep -q dialout; then
    echo -e "${YELLOW}Adding $USER to dialout group for USB serial access…${NC}"
    sudo usermod -aG dialout "$USER"
    echo -e "${YELLOW}⚠ Log out/in to pick up dialout membership.${NC}"
fi

echo ""
echo -e "${GREEN}Setup complete.${NC}"
echo ""
echo "Next:"
echo "  1. Start vision on the host (from repo root):"
echo "       cd ../vision && python3 main.py --host 0.0.0.0 --port 8000 \\"
echo "           --capture-width 640 --capture-height 480 &"
echo "       sleep 10 && curl -X POST http://localhost:8000/pipeline/start"
echo ""
echo "  2. Start cobot:"
echo "       docker compose -f docker-compose.unified.yml up -d cobotpy"
echo ""
