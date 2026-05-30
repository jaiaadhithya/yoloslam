from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image",
                    "--ros-args",
                    "-r",
                    "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera",
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "/depth_camera@sensor_msgs/msg/Image[gz.msgs.Image",
                    "--ros-args",
                    "-r",
                    "/depth_camera:=/camera/depth",
                ],
                output="screen",
            ),
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
