"""
ROS 2 launch: Gazebo camera bridge + YOLO + SLAM + Fusion + Controller (demo_fast).

Prerequisites:
  - PX4 SITL + Gazebo Baylands already running (or start in another terminal).
  - ROS 2 Humble + ros_gz_bridge + workspace sourced.

Terminal A (example):
  PX4_GZ_WORLD=baylands make px4_sitl gz_x500_depth

Terminal B:
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  ros2 launch yolo_slam_landing final_integrated_demo.launch.py

Terminal C (record 10–20 s demo video):
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  cd /path/to/yoloslam
  PYTHONPATH=src python3 -m evaluation.final_demo_recorder --duration 18 --fps 15

Output video: results/final_demo/final_landing_demo.mp4
Report: results/final_demo/demo_report.json
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    env = os.environ.copy()
    sep = os.pathsep
    env["PYTHONPATH"] = f"{src}{sep}{env.get('PYTHONPATH', '')}"
    env["MPLBACKEND"] = "Agg"

    bridge_baylands = [
        "ros2",
        "run",
        "ros_gz_bridge",
        "parameter_bridge",
        "/world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "--ros-args",
        "-r",
        "/world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera",
    ]
    bridge_default = [
        "ros2",
        "run",
        "ros_gz_bridge",
        "parameter_bridge",
        "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "--ros-args",
        "-r",
        "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera",
    ]

    def py_mod(mod: str, *ros_args: str) -> list:
        return ["python3", "-m", mod, *ros_args]

    return LaunchDescription(
        [
            ExecuteProcess(cmd=bridge_baylands, output="screen", env=env),
            ExecuteProcess(cmd=bridge_default, output="screen", env=env),
            ExecuteProcess(
                cmd=py_mod("yolo_detector.yolo_ros_node", "--ros-args", "-p", "inference_stride:=2"),
                cwd=str(repo),
                output="screen",
                env=env,
            ),
            ExecuteProcess(
                cmd=py_mod("slam_module.slam_ros_node"),
                cwd=str(repo),
                output="screen",
                env=env,
            ),
            ExecuteProcess(
                cmd=py_mod("fusion.fusion_ros_node"),
                cwd=str(repo),
                output="screen",
                env=env,
            ),
            ExecuteProcess(
                cmd=py_mod("landing_controller.controller_ros_node", "--ros-args", "-p", "demo_fast:=true"),
                cwd=str(repo),
                output="screen",
                env=env,
            ),
        ]
    )
