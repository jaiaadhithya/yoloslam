import json
import time
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from yolo_detector.detector import DetectorConfig, LandingPadDetector


def _image_msg_to_bgr(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.height == 0 or msg.width == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    channels = max(1, int(len(arr) / (msg.height * msg.width)))
    img = arr.reshape((msg.height, msg.width, channels))
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    return img[:, :, :3].copy()


class YoloRosNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_ros_node")
        self.detector = LandingPadDetector(DetectorConfig())
        self.last_latency_ms = 0.0
        self.image_sub = self.create_subscription(Image, "/camera", self.on_image, 10)
        self.det_pub = self.create_publisher(String, "/yolo/detections_json", 10)
        self.safe_pub = self.create_publisher(String, "/yolo/safe_zones_json", 10)

    def on_image(self, msg: Image) -> None:
        t0 = time.perf_counter()
        image = _image_msg_to_bgr(msg)
        raw = self.detector.infer(image)
        detections: List[dict] = self.detector.enrich(raw)
        self.last_latency_ms = (time.perf_counter() - t0) * 1e3

        out = String()
        out.data = json.dumps({"timestamp": time.time(), "detections": detections})
        self.det_pub.publish(out)

        safe = [d for d in detections if d.get("is_safe", False)]
        out_safe = String()
        out_safe.data = json.dumps({"timestamp": time.time(), "detections": safe})
        self.safe_pub.publish(out_safe)

        self.get_logger().info(
            f"YOLO detections={len(detections)} safe={len(safe)} latency_ms={self.last_latency_ms:.2f}",
            throttle_duration_sec=5.0,
        )


def main() -> None:
    rclpy.init()
    node = YoloRosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
