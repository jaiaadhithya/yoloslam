from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [ExecuteProcess(cmd=["python3", "-m", "slam_module.slam_node"], output="screen")]
    )
