from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    num_trials = LaunchConfiguration("num_trials")

    return LaunchDescription(
        [
            DeclareLaunchArgument("num_trials", default_value="30"),
            ExecuteProcess(
                cmd=[
                    "python3",
                    "-m",
                    "evaluation.run_trials",
                    "--num-trials",
                    num_trials,
                ],
                output="screen",
            ),
        ]
    )
