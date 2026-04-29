#!/usr/bin/env bash
set -euo pipefail

mkdir -p models
python3 - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")
print("Downloaded yolov8n.pt via ultralytics cache.")
PY

cp -f "${HOME}/.cache/ultralytics/yolov8n.pt" models/yolov8_terrain.pt 2>/dev/null || true
echo "Copy/fine-tune weights to models/yolov8_terrain.pt"
