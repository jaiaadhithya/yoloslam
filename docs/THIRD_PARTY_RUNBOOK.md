# Third-Party Integration Runbook (Realistic UAV Videos)

Use this when you want realistic drone + world visuals quickly, then plug in this repo's safe-zone logic.

## Option A (recommended): PX4 + ROS2 + Gazebo + YOLO baseline

Reference repo:
- https://github.com/monemati/PX4-ROS2-Gazebo-YOLOv8

### Steps

1. Clone:
   - `cd ~/third_party`
   - `git clone https://github.com/monemati/PX4-ROS2-Gazebo-YOLOv8.git`

2. Follow that repo's setup and launch to verify:
   - PX4 SITL
   - Gazebo world with moving targets
   - camera bridge to ROS2
   - YOLO live detection

3. Once baseline works, run this repo runtime nodes in parallel:
   - `ros2 launch yolo_slam_landing full_px4_runtime.launch.py`

4. Ensure camera topic remap exists as `/camera` from the bridge.

## Option B: Add stronger SLAM wrapper

Reference:
- https://github.com/suchetanrs/ORB-SLAM3-ROS2-Docker

Use this if you want a production-grade ORB-SLAM3 ROS2 wrapper fast.

## Recommended full stack terminal layout

1. Terminal A: PX4 SITL + Gazebo world
2. Terminal B: ROS2 camera bridge (`/camera`)
3. Terminal C: ORB-SLAM3 wrapper (or this repo's SLAM node)
4. Terminal D: `ros2 launch yolo_slam_landing full_px4_runtime.launch.py`
5. Terminal E: monitor topics + record bag

## Topic checks

- `ros2 topic list | rg "camera|slam|yolo|fusion|controller"`
- `ros2 topic echo /fusion/zone_json`
- `ros2 topic echo /controller/state`

## Video recording

- `sudo apt install -y ffmpeg`
- `ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i $DISPLAY -c:v libx264 -preset veryfast -crf 23 safe_landing_demo.mp4`
- press `q` to stop

## Notes

- This repo now has a ROS topic runtime launch (`full_px4_runtime.launch.py`) but still requires camera/pose topics from simulation bridges.
- Controller currently publishes `/controller/cmd_vel`; wiring that to PX4 offboard setpoints is the next extension for true closed-loop autonomous landing.
