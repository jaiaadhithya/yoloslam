#!/usr/bin/env bash
# Install RTAB-Map CLI tools in WSL/Ubuntu for rgbd odometry benchmark.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y rtabmap rtabmap-utils

echo "RTAB-Map CLI installed: $(command -v rtabmap-rgbd_odometry)"
