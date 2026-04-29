#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-$HOME/.gazebo/models}"
mkdir -p "$TARGET_DIR"

echo "Downloading model library from OSRF gazebo_models..."
TMP_DIR="$(mktemp -d)"
git clone --depth=1 https://github.com/osrf/gazebo_models.git "$TMP_DIR/gazebo_models"

# Copy a practical subset for safe landing scenes.
for MODEL in \
  ground_plane \
  sun \
  pine_tree \
  oak_tree \
  house_1 \
  house_2 \
  construction_barrel \
  stop_sign \
  pioneer2dx \
  ambulance \
  person_standing
do
  if [[ -d "$TMP_DIR/gazebo_models/$MODEL" ]]; then
    rm -rf "$TARGET_DIR/$MODEL"
    cp -R "$TMP_DIR/gazebo_models/$MODEL" "$TARGET_DIR/$MODEL"
    echo "Installed model: $MODEL"
  fi
done

rm -rf "$TMP_DIR"
echo "Environment models ready in: $TARGET_DIR"
