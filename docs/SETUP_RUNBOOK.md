# Setup Runbook (Do This In Order)

## A. Windows-side actions (you can do now)

1. Open PowerShell in project root.
2. Run:
   - `powershell -ExecutionPolicy Bypass -File scripts/windows_bootstrap.ps1`
3. If WSL is not installed:
   - `powershell -ExecutionPolicy Bypass -File scripts/windows_bootstrap.ps1 -InstallWSL`
   - Reboot when prompted.

## B. First-time WSL Ubuntu setup

1. Open Ubuntu (WSL).
2. Navigate to your project folder (recommended: inside Linux home):
   - `cd ~/projects/yoloslam`
3. Run:
   - `bash scripts/wsl_bootstrap.sh`

## C. Install ROS2 + Gazebo manually

Follow official ROS2 Humble + Gazebo docs for Ubuntu 22.04, then verify:

- `source /opt/ros/humble/setup.bash`
- `ros2 --version`
- `gz sim --version` or `gazebo --version`

Persist ROS source:

- `echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc`
- `source ~/.bashrc`

## D. Train / provide multiclass YOLO weights

### If you already have weights

Copy file:

- `models/yolov8_terrain_multiclass.pt`

### Train detection model

- `source .venv/bin/activate`
- `python scripts/train_multiclass_yolo.py --data datasets/data.yaml --epochs 100 --imgsz 640 --batch 16 --name terrain_multiclass`

### Train segmentation model (optional)

- `source .venv/bin/activate`
- `python scripts/train_multiclass_yolo.py --seg --data datasets/data.yaml --epochs 100 --imgsz 640 --batch 8 --name terrain_multiclass_seg`

## E. ORB vocabulary

Place:

- `ORB_SLAM3/Vocabulary/ORBvoc.txt`

## F. Simulation run

Use separate terminals:

1. PX4 SITL terminal
2. Gazebo terminal
3. ROS bridge terminal
4. Pipeline terminal (`ros2 launch ... full_pipeline.launch.py`)

## G. Evaluation

- `ros2 launch yolo_slam_landing evaluation.launch.py num_trials:=30`
- `python src/evaluation/plot_results.py --results-dir results`

## H. If things fail quickly

- Ensure `.venv` is active for Python commands.
- Ensure ROS is sourced in every terminal.
- Confirm model exists at `models/yolov8_terrain_multiclass.pt`.
- Confirm SLAM vocab exists at `ORB_SLAM3/Vocabulary/ORBvoc.txt`.
