import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from slam_module.ros_image_utils import image_msg_to_bgr, image_msg_to_depth_m
from slam_module.slam_wrapper import SlamWrapper


class SlamRosNode(Node):
    def __init__(self) -> None:
        super().__init__("slam_ros_node")
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("rgb_topic", "/camera")
        depth_topic = str(self.get_parameter("depth_topic").value)
        rgb_topic = str(self.get_parameter("rgb_topic").value)

        self.wrapper = SlamWrapper()
        self._latest_depth_m = None
        self._depth_received = False

        self.image_sub = self.create_subscription(Image, rgb_topic, self.on_image, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.on_depth, 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/slam/pose", 10)
        self.pose_json_pub = self.create_publisher(String, "/slam/pose_json", 10)
        self.get_logger().info(
            f"SLAM backend={self.wrapper.backend_name} rgb={rgb_topic} depth={depth_topic}",
            throttle_duration_sec=0.0,
        )

    def on_depth(self, msg: Image) -> None:
        depth = image_msg_to_depth_m(msg)
        if depth is not None:
            self._latest_depth_m = depth
            self._depth_received = True

    def on_image(self, msg: Image) -> None:
        frame = image_msg_to_bgr(msg)
        depth = self._latest_depth_m if self._depth_received else None
        pose = self.wrapper.track(frame, time.time(), depth_m=depth)

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
                "has_depth": bool(depth is not None),
                "timestamp": time.time(),
            }
        )
        self.pose_json_pub.publish(payload)
        self.get_logger().info(
            f"SLAM state={pose.tracking_state} depth={depth is not None} "
            f"x={pose.position[0]:.2f} y={pose.position[1]:.2f}",
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
