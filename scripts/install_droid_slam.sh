#!/usr/bin/env bash
# Install DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM) in WSL/Linux.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/third_party/DROID-SLAM"

mkdir -p "${ROOT}/third_party"
if [[ ! -d "${DEST}/.git" ]]; then
  git clone --depth 1 https://github.com/princeton-vl/DROID-SLAM.git "${DEST}"
fi

cd "${DEST}"
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install scipy opencv-python tqdm matplotlib

python3 - <<'PY'
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/princeton-vl/lietorch.git"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
PY

echo "DROID-SLAM ready at ${DEST}"
