# Reference Stack Integration (Realistic World + Drone)

This project now supports attaching to a mature baseline simulation stack:

- [monemati/PX4-ROS2-Gazebo-YOLOv8](https://github.com/monemati/PX4-ROS2-Gazebo-YOLOv8)

The purpose is to avoid toy-looking worlds and use a proven drone+Gazebo setup while running this repo's safe-zone fusion/controller logic.

## 1) Clone the reference stack

```bash
cd ~/Desktop/yoloslam/yoloslam
bash scripts/setup_reference_stack.sh
```

## 2) Launch reference stack services (in that repo)

Start:
- PX4 SITL
- Gazebo world
- Micro XRCE / required middleware

Use that repository README commands exactly.

## 3) Camera bridge

If /camera is not already bridged in reference stack, run:

```bash
source /opt/ros/humble/setup.bash
ros2 launch yolo_slam_landing monemati_camera_bridge.launch.py
```

## 4) Launch this repo's runtime

In this repo:

```bash
source /opt/ros/humble/setup.bash
cd ~/Desktop/yoloslam/yoloslam
source install/setup.bash
ros2 launch yolo_slam_landing full_runtime_with_bridge.launch.py
```

## 5) Validate

```bash
ros2 topic list | rg "camera|slam|yolo|fusion|controller"
ros2 topic echo /fusion/zone_json
```

## 6) Record video

```bash
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i $DISPLAY -c:v libx264 -preset veryfast -crf 23 safe_landing_demo.mp4
```

Press `q` to stop.
