# YOLO-SLAM Safe-Zone Landing (Unknown Terrain)

This repository provides a full scaffold for infrastructure-free UAV landing by fusing multi-class YOLO scene understanding and visual SLAM to select safe landing zones in unknown terrain.

## Current status

- Complete project structure and baseline code are generated.
- Multi-class terrain and obstacle detection with safe/unsafe semantics is wired.
- Safety-grid fusion and contiguous safe-zone selection are implemented.
- Controller logic includes `SURVEY`, `EVALUATE`, and mid-descent abort behavior.
- Some system-heavy components (PX4, ORB-SLAM3 build, Gazebo assets, model weights) require local setup steps.

## Prerequisites

- Ubuntu 22.04 (recommended for ROS2 + PX4 workflow)
- ROS 2 Humble
- Gazebo Harmonic (or Gazebo Classic 11)
- PX4 Autopilot (SITL)
- Python 3.10+
- CUDA 11.8+ (optional but recommended for YOLO acceleration)

## Quick start (Docker path)

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# PX4 + ORB-SLAM3 setup scripts
bash scripts/install_px4.sh
bash scripts/install_orbslam3.sh
bash scripts/download_yolo_weights.sh
bash scripts/download_environment_models.sh
```

## Run experiments

```bash
# Full pipeline demo (single landing)
ros2 launch yolo_slam_landing full_pipeline.launch.py

# Open mixed-terrain world
ros2 launch yolo_slam_landing safe_world.launch.py

# Realistic world + PX4 x500 depth camera + YOLO/SLAM/fusion/controller
# (requires PX4-Autopilot checkout and Gazebo Sim setup)
export PX4_AUTOPILOT_PATH=~/PX4-Autopilot
ros2 launch yolo_slam_landing px4_baylands_autonomy.launch.py

# ROS topic runtime (camera->YOLO->SLAM->fusion->controller)
ros2 launch yolo_slam_landing full_px4_runtime.launch.py

# Runtime with explicit Gazebo camera bridge (x500_depth)
ros2 launch yolo_slam_landing full_runtime_with_bridge.launch.py

# YOLO-only ablation
ros2 launch yolo_slam_landing yolo_only.launch.py

# SLAM-only ablation
ros2 launch yolo_slam_landing slam_only.launch.py

# Run 30-trial evaluation for all conditions
ros2 launch yolo_slam_landing evaluation.launch.py num_trials:=30

# Generate plots and tables
python3 src/evaluation/plot_results.py --results-dir results/
```

## Scenario design

- The Gazebo world is a mixed 100m x 100m terrain with clearings, water, trees, buildings, vehicles, and pedestrians.
- YOLO classes are mapped to safety labels (`safe` or `unsafe`) per class.
- Fusion projects detections to world-grid cells through SLAM pose and accumulates safety evidence over time.
- Zone selection picks the best contiguous safe cluster instead of a single marker target.
- During descent, newly observed unsafe intrusions (e.g., person/vehicle) trigger abort and re-survey.

## Paper build

```bash
cd paper
make pdf
```

## Notes

- This scaffold includes lightweight fallback logic where ROS/PX4/ORB-SLAM3 are not available so local unit-level testing is still possible.
- Running `gazebo worlds/safe_landing_world.sdf` only opens a static demo world; it does not spawn PX4 SITL or drone autonomy.
- For a realistic upstream world + drone simulation, use `px4_baylands_autonomy.launch.py` (Baylands + `gz_x500_depth`).
- Replace placeholder SDF assets and ORB-SLAM3 wrapper bindings with your final simulation-specific implementation.
- Detailed environment and execution steps: `docs/SETUP_RUNBOOK.md`.
- Framework alignment checklist: `docs/FRAMEWORK_UPDATE_STATUS.md`.
- Realistic third-party stack integration: `docs/THIRD_PARTY_RUNBOOK.md`.
- Reference baseline integration: `docs/REFERENCE_STACK_INTEGRATION.md`.
