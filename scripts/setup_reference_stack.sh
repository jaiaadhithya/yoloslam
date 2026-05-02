#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-$HOME/third_party}"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

if [[ ! -d PX4-ROS2-Gazebo-YOLOv8 ]]; then
  git clone https://github.com/monemati/PX4-ROS2-Gazebo-YOLOv8.git
fi

echo "Reference stack cloned at: $BASE_DIR/PX4-ROS2-Gazebo-YOLOv8"
echo
echo "Next:"
echo "  1) Follow that repo setup to start PX4 SITL + Gazebo"
echo "  2) Ensure camera bridge publishes /camera"
echo "  3) Launch this repo runtime using:"
echo "     ros2 launch yolo_slam_landing full_runtime_with_bridge.launch.py"
