# CobotPy — myCobot 280 Docker Demo

This repository provides a minimal example of controlling the **Elephant Robotics myCobot 280** using Python inside Docker, with the `pymycobot` library.

The demo script moves the robot slightly, so you can verify connection, serial port setup, and basic motion.

---

## 🐳 Repository Contents

- `mycobot_test.py` — example script that powers on the robot, moves a joint slightly, then powers off  
- `Dockerfile` — defines the Docker image  
- `requirements.txt` — pinned Python dependencies  
- `README.md` — this file  

---

## 🔧 Prerequisites

- Docker installed on your system  
- myCobot 280 connected via USB and powered on (blue LED)  
- Know your USB/serial port (e.g. `/dev/ttyUSB0` on Linux, or via WSL2 + usbipd on Windows)

---

## ✅ Clone & Build

```bash
git clone https://github.com/Autonomous-Flight-and-Perception/cobotpy.git
cd cobotpy
docker build -t cobotpy-demo .
```
# (optional) Install or ensure Docker is running
# Arch
```bash
sudo pacman -Syu docker
sudo systemctl enable --now docker
```
# Ubuntu / Debian

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl enable --now docker
```
# Identify device
```bash
ls /dev/ttyUSB*
```
# Run container with serial passthrough
```bash
docker run --rm -it \
  --device=/dev/ttyUSB0:/dev/ttyUSB0 \
  cobotpy-demo
```
```bash
DEVICE=/dev/ttyACM0 docker compose up --build
```

