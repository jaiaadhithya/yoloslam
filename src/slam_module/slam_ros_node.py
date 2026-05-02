import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from slam_module.slam_wrapper import OrbSlamWrapper


def _image_msg_to_rgb(msg: Image) -> np.ndarray:
    if msg.height == 0 or msg.width == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    step = int(msg.step) if msg.step else 0
    if step > 0 and len(arr) >= step * msg.height:
        row = arr[: step * msg.height].reshape((msg.height, step))
        channels = max(1, step // msg.width)
        img = row[:, : msg.width * channels].reshape((msg.height, msg.width, channels))
    else:
        channels = max(1, int(len(arr) / (msg.height * msg.width)))
        img = arr.reshape((msg.height, msg.width, channels))

    enc = (msg.encoding or "").lower()
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    if "bgr" in enc:
        return img[:, :, :3][:, :, ::-1].copy()
    return img[:, :, :3].copy()


class SlamRosNode(Node):
    def __init__(self) -> None:
        super().__init__("slam_ros_node")
        self.wrapper = OrbSlamWrapper()
        self.image_sub = self.create_subscription(Image, "/camera", self.on_image, 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/slam/pose", 10)
        self.pose_json_pub = self.create_publisher(String, "/slam/pose_json", 10)

    def on_image(self, msg: Image) -> None:
        frame = _image_msg_to_rgb(msg)
        pose = self.wrapper.track(frame, time.time())

        ros_pose = PoseStamped()
        ros_pose.header.stamp = self.get_clock().now().to_msg()
        ros_pose.header.frame_id = "map"
        ros_pose.pose.position.x = float(pose.position[0])
        ros_pose.pose.position.y = float(pose.position[1])
        ros_pose.pose.position.z = float(pose.position[2])
        ros_pose.pose.orientation.x = float(pose.quaternion_xyzw[0])
        ros_pose.pose.orientation.y = float(pose.quaternion_xyzw[1])
        ros_pose.pose.orientation.z = float(pose.quaternion_xyzw[2])
        ros_pose.pose.orientation.w = float(pose.quaternion_xyzw[3])
        self.pose_pub.publish(ros_pose)

        payload = String()
        payload.data = json.dumps(
            {
                "x": ros_pose.pose.position.x,
                "y": ros_pose.pose.position.y,
                "z": ros_pose.pose.position.z,
                "qx": ros_pose.pose.orientation.x,
                "qy": ros_pose.pose.orientation.y,
                "qz": ros_pose.pose.orientation.z,
                "qw": ros_pose.pose.orientation.w,
                "state": pose.tracking_state,
                "timestamp": time.time(),
            }
        )
        self.pose_json_pub.publish(payload)
        self.get_logger().info(
            f"SLAM state={pose.tracking_state} x={pose.position[0]:.2f} y={pose.position[1]:.2f}",
            throttle_duration_sec=5.0,
        )


def main() -> None:
    rclpy.init()
    node = SlamRosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
