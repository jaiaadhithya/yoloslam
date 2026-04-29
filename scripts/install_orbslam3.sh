#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ORB_SLAM3 ]]; then
  git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
fi

cd ORB_SLAM3
chmod +x build.sh
./build.sh || true

mkdir -p Vocabulary
if [[ ! -f Vocabulary/ORBvoc.txt ]]; then
  echo "Downloading ORB vocabulary..."
  wget -q https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz -O /tmp/ORBvoc.txt.tar.gz || true
  if [[ -f /tmp/ORBvoc.txt.tar.gz ]]; then
    tar -xzf /tmp/ORBvoc.txt.tar.gz -C Vocabulary || true
  fi
fi

if [[ ! -f Vocabulary/ORBvoc.txt ]]; then
  echo "Failed automatic ORBvoc download."
  echo "Download manually and place at ORB_SLAM3/Vocabulary/ORBvoc.txt"
fi

echo "ORB-SLAM3 setup scaffold complete."
