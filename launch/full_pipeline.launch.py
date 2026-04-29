from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=["python3", "-m", "yolo_detector.yolo_node"],
                output="screen",
                additional_env={"MPLBACKEND": "Agg"},
            ),
            ExecuteProcess(cmd=["python3", "-m", "slam_module.slam_node"], output="screen"),
            ExecuteProcess(cmd=["python3", "-m", "fusion.fusion_node"], output="screen"),
            ExecuteProcess(cmd=["python3", "-m", "landing_controller.controller_node"], output="screen"),
        ]
    )
