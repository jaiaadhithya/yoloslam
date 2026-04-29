#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d PX4-Autopilot ]]; then
  git clone https://github.com/PX4/PX4-Autopilot.git
fi

cd PX4-Autopilot
git fetch --all
git checkout v1.15.0
bash ./Tools/setup/ubuntu.sh --no-sim-tools || true
make px4_sitl_default

echo "PX4 setup complete."
