from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description() -> LaunchDescription:
    # Bridge camera stream from PX4 Gazebo x500_depth model to /camera.
    bridge_cmd = [
        "ros2",
        "run",
        "ros_gz_bridge",
        "parameter_bridge",
        "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image",
        "--ros-args",
        "-r",
        "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/camera",
    ]
    return LaunchDescription([ExecuteProcess(cmd=bridge_cmd, output="screen")])
