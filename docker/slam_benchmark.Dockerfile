# SLAM benchmark container (Ubuntu 22.04 + bundled Pangolin + ORB-SLAM3)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip \
    libopencv-dev libeigen3-dev libboost-all-dev \
    libssl-dev libglew-dev libgl1-mesa-dev \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN bash scripts/build_pangolin_wsl.sh && bash scripts/install_orbslam3_wsl.sh

CMD ["bash", "-lc", "PYTHONPATH=src python3 scripts/run_slam_comparison.py --max-frames 792 --frame-stride 1"]
