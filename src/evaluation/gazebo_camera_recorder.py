"""
Subscribe to /camera (sensor_msgs/Image) and write a fixed-FPS MP4 (performance-friendly size).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image

# ros_gz_bridge QoS varies by version/RMW; subscribe with both profiles and dedupe by stamp.
_CAM_SUB_QOS_RELIABLE = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)


def _bgr_from_ros(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    if h <= 0 or w <= 0:
        raise ValueError("empty image")
    nchan = len(msg.data) // (h * w)
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, nchan))
    enc = (msg.encoding or "").lower()
    if nchan == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if nchan == 4:
        arr = arr[:, :, :3]
        nchan = 3
    if nchan == 3:
        if "rgb" in enc:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    raise ValueError(f"unsupported channels {nchan} encoding={msg.encoding}")


class CameraRecorder(Node):
    def __init__(
        self,
        *,
        output: Path,
        duration: float,
        fps: float,
        overlay: bool,
        metrics_path: Path,
        width: int,
        height: int,
    ) -> None:
        super().__init__("gazebo_camera_recorder")
        self._output = output
        self._duration = duration
        self._fps = fps
        self._overlay = overlay
        self._metrics_path = metrics_path
        self._tw, self._th = width, height

        self._writer: cv2.VideoWriter | None = None
        self._t0 = time.time()
        self._next_sample_t = self._t0
        self._interval = 1.0 / max(1.0, fps)
        self._count = 0
        self._finalized = False
        self._logged_first = False
        self._last_stamp: tuple[int, int] | None = None

        self.create_subscription(Image, "/camera", self._on_image, _CAM_SUB_QOS_RELIABLE)
        self.create_subscription(Image, "/camera", self._on_image, qos_profile_sensor_data)

    def _on_image(self, msg: Image) -> None:
        if self._finalized:
            return
        st = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if st != (0, 0) and st == self._last_stamp:
            return
        self._last_stamp = st
        if not self._logged_first:
            self._logged_first = True
            self.get_logger().info(
                f"first /camera frame {msg.width}x{msg.height} {msg.encoding}",
            )
        now = time.time()
        if now > self._t0 + self._duration:
            self._finalize()
            return
        if now < self._next_sample_t:
            return

        try:
            frame = _bgr_from_ros(msg)
        except Exception as e:
            self.get_logger().warn(f"frame decode: {e}")
            return

        if frame.shape[1] != self._tw or frame.shape[0] != self._th:
            frame = cv2.resize(frame, (self._tw, self._th), interpolation=cv2.INTER_AREA)

        if self._overlay:
            cv2.putText(
                frame,
                "Baylands demo",
                (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cy, cx = self._th // 2, self._tw // 2
            cv2.drawMarker(
                frame,
                (cx, cy),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )

        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self._output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self._fps),
                (w, h),
            )
        self._writer.write(frame)
        self._count += 1
        self._next_sample_t += self._interval

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        wall = time.time() - self._t0
        expected = max(1e-6, wall * self._fps)
        payload = {
            "wall_time_record_s": wall,
            "frames_written": self._count,
            "target_fps": self._fps,
            "camera_write_rtf": round(self._count / expected, 3),
            "output": str(self._output.resolve()),
            "resolution": [self._tw, self._th],
        }
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.get_logger().info(f"wrote {self._count} frames -> {self._output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record /camera to MP4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--metrics", type=Path, default=Path("results/gazebo_demo/demo_metrics.json"))
    args = parser.parse_args()

    rclpy.init()
    node = CameraRecorder(
        output=args.output,
        duration=args.duration,
        fps=args.fps,
        overlay=args.overlay,
        metrics_path=args.metrics,
        width=args.width,
        height=args.height,
    )
    # Allow slow first frame after bridge / lockstep startup (especially WSL2).
    deadline = time.time() + float(args.duration) + 120.0
    try:
        while rclpy.ok() and time.time() < deadline and not node._finalized:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._finalized:
            node.get_logger().warn("Recorder stopping without full duration (no frames or shutdown).")
            node._finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
