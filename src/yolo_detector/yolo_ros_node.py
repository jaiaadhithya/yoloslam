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
    if "rgb" in enc:
        return img[:, :, :3][:, :, ::-1].copy()
    return img[:, :, :3].copy()


class YoloRosNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_ros_node")
        self.declare_parameter("inference_stride", 1)
        self.detector = LandingPadDetector(DetectorConfig())
        self.last_latency_ms = 0.0
        self._frame_id = 0
        self._last_enriched: list = []
        self.image_sub = self.create_subscription(Image, "/camera", self.on_image, 10)
        self.det_pub = self.create_publisher(String, "/yolo/detections_json", 10)
        self.safe_pub = self.create_publisher(String, "/yolo/safe_zones_json", 10)

    def on_image(self, msg: Image) -> None:
        t0 = time.perf_counter()
        image = _image_msg_to_bgr(msg)
        self._frame_id += 1
        stride = max(1, int(self.get_parameter("inference_stride").get_parameter_value().integer_value or 1))
        if self._frame_id % stride == 0 or not self._last_enriched:
            raw = self.detector.infer(image)
            self._last_enriched = self.detector.enrich(raw)
        detections: List[dict] = self._last_enriched
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
