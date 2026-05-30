#!/usr/bin/env bash
# Build ORB-SLAM3 against bundled Pangolin (no system /usr/local sigslot).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORB="${ROOT}/third_party/ORB_SLAM3"
PANGOLIN_INSTALL="${ROOT}/third_party/Pangolin/install"

bash "${ROOT}/scripts/build_pangolin_wsl.sh"

if [[ ! -d "${ORB}/.git" ]]; then
  git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git "${ORB}"
fi

cd "${ORB}"

# ORB-SLAM3 upstream targets C++11/14; keep C++14 (C++17 breaks bool++ in LoopClosing.cc).
sed -i 's/-std=c++11/-std=c++14/g' CMakeLists.txt || true
sed -i 's/-std=c++17/-std=c++14/g' CMakeLists.txt || true

export Pangolin_DIR="${PANGOLIN_INSTALL}/lib/cmake/Pangolin"
export CMAKE_PREFIX_PATH="${PANGOLIN_INSTALL}:${CMAKE_PREFIX_PATH:-}"

rm -rf build Thirdparty/DBoW2/build Thirdparty/g2o/build Thirdparty/Sophus/build
chmod +x build.sh
./build.sh

mkdir -p Vocabulary
if [[ ! -f Vocabulary/ORBvoc.txt ]]; then
  if [[ -f Vocabulary/ORBvoc.txt.tar.gz ]]; then
    tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
  else
    wget -q https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz -O /tmp/ORBvoc.txt.tar.gz
    tar -xzf /tmp/ORBvoc.txt.tar.gz -C Vocabulary
  fi
fi

BIN="${ORB}/Examples/RGB-D/rgbd_tum"
if [[ -x "${BIN}" ]]; then
  echo "ORB-SLAM3 ready: ${BIN}"
else
  echo "ORB-SLAM3 build finished but rgbd_tum missing — check build log."
  exit 1
fi
