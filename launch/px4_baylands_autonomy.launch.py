import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    # Set PX4_AUTOPILOT_PATH to your PX4 checkout. Falls back to ~/PX4-Autopilot.
    px4_path = os.environ.get("PX4_AUTOPILOT_PATH", "~/PX4-Autopilot")
    px4_cmd = (
        f"cd {px4_path} && "
        "PX4_GZ_WORLD=baylands make px4_sitl gz_x500_depth"
    )

    # Bridge x500_depth camera stream from Gazebo -> ROS /camera.
    # Spawn both mappings because world name may resolve to either default or baylands.
    bridge_cmd_baylands = [
        "ros2",
        "run",
        "ros_gz_bridge",
        "parameter_bridge",
        "/world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "--ros-args",
        "-r",
        "/world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera",
    ]
    bridge_cmd_default = [
        "ros2",
        "run",
        "ros_gz_bridge",
        "parameter_bridge",
        "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "--ros-args",
        "-r",
        "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera",
    ]

    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=["bash", "-lc", px4_cmd],
                output="screen",
            ),
            ExecuteProcess(cmd=bridge_cmd_baylands, output="screen"),
            ExecuteProcess(cmd=bridge_cmd_default, output="screen"),
            ExecuteProcess(
                cmd=["python3", "-m", "yolo_detector.yolo_ros_node"],
                output="screen",
                additional_env={"MPLBACKEND": "Agg"},
            ),
            ExecuteProcess(cmd=["python3", "-m", "slam_module.slam_ros_node"], output="screen"),
            ExecuteProcess(cmd=["python3", "-m", "fusion.fusion_ros_node"], output="screen"),
            ExecuteProcess(cmd=["python3", "-m", "landing_controller.controller_ros_node"], output="screen"),
        ]
    )
