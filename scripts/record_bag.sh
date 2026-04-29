#!/usr/bin/env bash
set -euo pipefail

ros2 bag record \
  /drone/camera/image_raw \
  /yolo/detections \
  /slam/pose \
  /fusion/landing_target \
  /controller/state
