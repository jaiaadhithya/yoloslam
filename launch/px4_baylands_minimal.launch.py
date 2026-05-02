"""
Minimal PX4 SITL + Gazebo Baylands + x500 spawn.

Starts only the upstream PX4 make target (no YOLO, fusion, controller, or other ROS nodes).

Prerequisites: PX4-Autopilot checkout, Gazebo (Harmonic per PX4 docs), build finished.

Usage (from workspace, after `colcon build` + `source install/setup.bash`):

  export PX4_AUTOPILOT_PATH=/path/to/PX4-Autopilot
  ros2 launch yolo_slam_landing px4_baylands_minimal.launch.py

Or run the shell entrypoint directly on Linux/WSL:

  ./scripts/px4_baylands_minimal.sh
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "px4_baylands_minimal.sh"
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=["/usr/bin/env", "bash", str(script)],
                cwd=str(repo_root),
                output="screen",
            ),
        ]
    )
