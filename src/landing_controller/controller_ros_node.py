import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from landing_controller.controller_node import ControllerNode
from landing_controller.state_machine import LandingSMConfig


class ControllerRosNode(Node):
    def __init__(self) -> None:
        super().__init__("controller_ros_node")
        self.declare_parameter("demo_fast", False)
        demo_fast = bool(self.get_parameter("demo_fast").get_parameter_value().bool_value)
        sm_cfg = None
        if demo_fast:
            sm_cfg = LandingSMConfig(
                survey_duration_s=4.0,
                zone_score_evaluate_commit=3.5,
                zone_score_search_commit=1.5,
                approach_xy_tolerance_m=2.5,
                align_xy_tolerance_m=0.25,
                align_stable_s=1.2,
            )
            self.get_logger().info("demo_fast=true: relaxed landing SM thresholds for demo video")
        self.controller = ControllerNode(sm_cfg)
        self.last_pose = {"x": 0.0, "y": 0.0, "z": 10.0}
        self.last_zone = {
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "zone_score": 0.0,
            "has_valid_zone": False,
            "intrusion_detected": False,
        }
        self.pose_sub = self.create_subscription(String, "/slam/pose_json", self.on_pose, 10)
        self.zone_sub = self.create_subscription(String, "/fusion/zone_json", self.on_zone, 10)
        self.cmd_pub = self.create_publisher(Twist, "/controller/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/controller/state", 10)
        self.state_json_pub = self.create_publisher(String, "/controller/state_json", 10)
        self.create_timer(0.2, self.step)

    def on_pose(self, msg: String) -> None:
        try:
            self.last_pose = json.loads(msg.data)
        except Exception:
            pass

    def on_zone(self, msg: String) -> None:
        try:
            self.last_zone = json.loads(msg.data)
        except Exception:
            pass

    def step(self) -> None:
        cmd = self.controller.update(self.last_zone, self.last_pose, dt=0.2)
        twist = Twist()
        twist.linear.x = float(cmd["vx"])
        twist.linear.y = float(cmd["vy"])
        twist.linear.z = float(cmd["vz"])
        twist.angular.z = float(cmd["yaw_rate"])
        self.cmd_pub.publish(twist)

        state = String()
        state.data = cmd["state"]
        self.state_pub.publish(state)
        sj = String()
        sj.data = json.dumps(
            {
                "state": cmd["state"],
                "transition_reason": cmd.get("transition_reason", ""),
                "vx": cmd["vx"],
                "vy": cmd["vy"],
                "vz": cmd["vz"],
            }
        )
        self.state_json_pub.publish(sj)
        self.get_logger().info(
            f"Controller state={cmd['state']} vx={cmd['vx']:.2f} vy={cmd['vy']:.2f} vz={cmd['vz']:.2f}",
            throttle_duration_sec=5.0,
        )


def main() -> None:
    rclpy.init()
    node = ControllerRosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
