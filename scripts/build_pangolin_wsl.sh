#!/usr/bin/env bash
# Build Pangolin from source (avoids broken /usr/local sigslot headers).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/third_party/Pangolin"
INSTALL="${DEST}/install"

if [[ ! -d "${DEST}/.git" ]]; then
  git clone --depth 1 --branch v0.8 https://github.com/stevenlovegrove/Pangolin.git "${DEST}"
fi

mkdir -p "${DEST}/build"
cd "${DEST}/build"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF \
  -DBUILD_PANGOLIN_PYTHON=OFF
make -j"$(nproc)"
make install

echo "Pangolin installed to ${INSTALL}"
echo "Use: export Pangolin_DIR=${INSTALL}/lib/cmake/Pangolin"
