from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=["gazebo", "--verbose", "worlds/safe_landing_world.sdf"],
                output="screen",
            )
        ]
    )
