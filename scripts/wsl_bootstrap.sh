#!/usr/bin/env bash
set -euo pipefail

echo "== YOLO-SLAM WSL Bootstrap =="

sudo apt update
# Gazebo is needed for simulation and may not be present.
sudo apt install -y gazebo || true

sudo apt install -y \
  git curl wget build-essential cmake pkg-config \
  python3-pip python3-venv python3-dev \
  libopencv-dev libeigen3-dev \
  ninja-build exiftool genromfs python3-empy python3-jinja2 python3-numpy python3-toml

# Pangolin is optional in some apt channels; continue if unavailable.
if ! sudo apt install -y libpangolin-dev; then
  echo "Warning: libpangolin-dev not available in current apt repos."
  echo "ORB-SLAM3 may need Pangolin built from source later."
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Running project install scripts..."
bash scripts/install_px4.sh || true
bash scripts/install_orbslam3.sh || true
bash scripts/download_yolo_weights.sh || true

echo
echo "If ORB vocabulary is missing, place file at:"
echo "  ORB_SLAM3/Vocabulary/ORBvoc.txt"
echo
echo "WSL bootstrap complete."
