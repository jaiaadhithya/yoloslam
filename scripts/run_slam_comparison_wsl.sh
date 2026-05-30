#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz"
DST="$HOME/tum_rgbd_freiburg1_xyz"
if [[ ! -d "$DST/rgb" ]]; then
  echo "Copying TUM dataset to $DST ..."
  cp -r "$SRC" "$DST"
fi
echo "RGB frames: $(find "$DST/rgb" -maxdepth 1 -type f | wc -l)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
python3 scripts/run_slam_comparison.py \
  --dataset-dir "$DST" \
  --max-frames 792 \
  --frame-stride 1 \
  "$@"
