import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from fusion.fusion_node import FusionNode


class FusionRosNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_ros_node")
        self.fusion = FusionNode()
        self.last_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "state": "INITIALIZING"}
        self.last_detections = []
        self.pose_sub = self.create_subscription(String, "/slam/pose_json", self.on_pose, 10)
        self.det_sub = self.create_subscription(String, "/yolo/detections_json", self.on_detections, 10)
        self.zone_pub = self.create_publisher(String, "/fusion/zone_json", 10)
        self.create_timer(0.2, self.publish_zone)

    def on_pose(self, msg: String) -> None:
        try:
            self.last_pose = json.loads(msg.data)
        except Exception:
            pass

    def on_detections(self, msg: String) -> None:
        try:
            self.last_detections = json.loads(msg.data).get("detections", [])
        except Exception:
            self.last_detections = []

    def publish_zone(self) -> None:
        zone = self.fusion.fuse(self.last_detections, self.last_pose)
        zone["intrusion_detected"] = not self.fusion.zone_is_still_safe() if zone.get("has_valid_zone") else False
        out = String()
        out.data = json.dumps(zone)
        self.zone_pub.publish(out)
        self.get_logger().info(
            f"Fusion valid={zone.get('has_valid_zone')} score={zone.get('zone_score', 0.0):.2f}",
            throttle_duration_sec=5.0,
        )


def main() -> None:
    rclpy.init()
    node = FusionRosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
