"""
Record an integrated landing demo: camera + YOLO boxes + fusion grid inset + controller state overlay.

  PYTHONPATH=src python3 -m evaluation.final_demo_recorder --duration 18
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

_CAM_QOS = QoSProfile(
    depth=10,
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
    if nchan >= 3:
        if "rgb" in enc:
            return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        return arr[:, :, :3].copy()
    raise ValueError(f"bad channels {nchan}")


def _world_to_uv(wx: float, wy: float, pose: dict, fx: float, fy: float, cx: float, cy: float, depth_m: float) -> tuple[int, int]:
    dx = float(wx) - float(pose.get("x", 0.0))
    dy = float(wy) - float(pose.get("y", 0.0))
    u = int(dx * fx / depth_m + cx)
    v = int(dy * fy / depth_m + cy)
    return u, v


class FinalDemoRecorder(Node):
    def __init__(
        self,
        *,
        output: Path,
        duration_s: float,
        out_fps: float,
        max_w: int,
    ) -> None:
        super().__init__("final_demo_recorder")
        self._output = output
        self._duration_s = duration_s
        self._out_fps = out_fps
        self._max_w = max_w

        self._pose = {"x": 0.0, "y": 0.0, "z": 12.0}
        self._writer: cv2.VideoWriter | None = None
        self._t0 = time.time()
        self._next_t = self._t0
        self._interval = 1.0 / max(1.0, out_fps)
        self._n_written = 0
        self._done = False

        self._dets: list = []
        self._zone: dict = {}
        self._viz: dict = {}
        self._state: dict = {"state": "—", "transition_reason": ""}
        self._last_reason_shown = ""
        self._reason_until = 0.0

        self.events = {"descend": False, "abort": False, "landed": False}

        self.create_subscription(Image, "/camera", self._on_cam, _CAM_QOS)
        self.create_subscription(String, "/slam/pose_json", self._on_pose, 10)
        self.create_subscription(String, "/yolo/detections_json", self._on_dets, 10)
        self.create_subscription(String, "/fusion/zone_json", self._on_zone, 10)
        self.create_subscription(String, "/fusion/viz_json", self._on_viz, 10)
        self.create_subscription(String, "/controller/state_json", self._on_state, 10)

    def _on_pose(self, msg: String) -> None:
        try:
            p = json.loads(msg.data)
            self._pose["x"] = float(p.get("x", 0.0))
            self._pose["y"] = float(p.get("y", 0.0))
            self._pose["z"] = float(p.get("z", 0.0))
        except json.JSONDecodeError:
            pass

    def _on_dets(self, msg: String) -> None:
        try:
            self._dets = json.loads(msg.data).get("detections", [])
        except json.JSONDecodeError:
            pass

    def _on_zone(self, msg: String) -> None:
        try:
            self._zone = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_viz(self, msg: String) -> None:
        try:
            self._viz = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_state(self, msg: String) -> None:
        try:
            p = json.loads(msg.data)
            r = str(p.get("transition_reason", "") or "")
            if r:
                self._last_reason_shown = r
                self._reason_until = time.time() + 4.0
            self._state = p
            s = str(p.get("state", ""))
            if s == "DESCEND":
                self.events["descend"] = True
            if s == "ABORT":
                self.events["abort"] = True
            if s == "LANDED":
                self.events["landed"] = True
        except json.JSONDecodeError:
            pass

    def _on_cam(self, msg: Image) -> None:
        if self._done:
            return
        now = time.time()
        if now > self._t0 + self._duration_s:
            self._finalize()
            return
        if now < self._next_t:
            return
        self._next_t += self._interval
        if now > self._next_t + self._interval:
            self._next_t = now

        try:
            frame = _bgr_from_ros(msg)
        except Exception as e:
            self.get_logger().warn(f"frame: {e}")
            return

        h0, w0 = frame.shape[:2]
        scale = min(1.0, self._max_w / float(w0))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        fx, fy = 320.0 * scale, 320.0 * scale
        cx, cy = 320.0 * scale, 240.0 * scale
        depth_m = 10.0

        for d in self._dets:
            x, y, bw, bh = float(d["x_center"]), float(d["y_center"]), float(d["width"]), float(d["height"])
            x *= scale
            y *= scale
            bw *= scale
            bh *= scale
            x1, y1 = int(x - bw / 2), int(y - bh / 2)
            x2, y2 = int(x + bw / 2), int(y + bh / 2)
            unsafe = d.get("safety_label") in ("unsafe", "unknown") or not d.get("is_safe", False)
            col = (0, 0, 220) if unsafe else (0, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            lab = str(d.get("class_name", "?"))[:18]
            cv2.putText(frame, lab, (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        if self._zone.get("has_valid_zone"):
            u, v = _world_to_uv(
                float(self._zone["center_x"]),
                float(self._zone["center_y"]),
                self._pose,
                fx,
                fy,
                cx,
                cy,
                depth_m,
            )
            cv2.circle(frame, (u, v), 14, (0, 220, 0), 2)
            cv2.circle(frame, (u, v), 3, (0, 255, 100), -1)

        inset = 160
        if self._viz.get("safety_ds"):
            sm = np.array(self._viz["safety_ds"], dtype=np.float32)
            un = np.array(self._viz.get("unsafe_ds", np.zeros_like(sm)), dtype=np.float32)
            lo, hi = float(np.percentile(sm, 5)), float(np.percentile(sm, 95))
            if hi <= lo:
                lo, hi = -1.0, 1.0
            norm = np.clip((sm - lo) / (hi - lo + 1e-6), 0, 1)
            heat = (norm * 255).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
            uh = np.clip(un / (un.max() + 1e-6), 0, 1)
            heat_bgr[:, :, 2] = np.clip(heat_bgr[:, :, 2] + (uh * 120).astype(np.uint8), 0, 255)
            heat_bgr = cv2.resize(heat_bgr, (inset, inset), interpolation=cv2.INTER_NEAREST)
            ov = heat_bgr.copy()
            gh, gw = sm.shape
            st = max(1, int(self._viz.get("stride_cells", 1)))
            res = float(self._viz.get("resolution_m", 1.0))
            wsz = float(self._viz.get("world_size_m", 100.0))
            half = wsz / 2.0
            if self._zone.get("has_valid_zone") and gw > 0 and gh > 0:
                wx = float(self._zone["center_x"])
                wy = float(self._zone["center_y"])
                ix_f = int(np.clip((wx + half) / res, 0, wsz / res - 1e-6))
                iy_f = int(np.clip((wy + half) / res, 0, wsz / res - 1e-6))
                ix_ds = min(ix_f // st, gw - 1)
                iy_ds = min(iy_f // st, gh - 1)
                px = int(ix_ds / max(1, gw - 1) * (inset - 1))
                py = int((1.0 - iy_ds / max(1, gh - 1)) * (inset - 1))
                cv2.circle(ov, (px, py), 6, (0, 255, 0), 2)
            x0, y0 = w - inset - 12, h - inset - 12
            roi = frame[y0 : y0 + inset, x0 : x0 + inset]
            if roi.shape[:2] == (inset, inset):
                blended = cv2.addWeighted(roi, 0.35, ov, 0.65, 0)
                frame[y0 : y0 + inset, x0 : x0 + inset] = blended
            cv2.rectangle(frame, (x0, y0), (x0 + inset, y0 + inset), (255, 255, 255), 1)

        st = str(self._state.get("state", "?"))
        reason = self._last_reason_shown if time.time() < self._reason_until else ""
        panel_h = 78
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        cv2.putText(frame, f"STATE: {st}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        if reason:
            cv2.putText(
                frame,
                f"event: {reason[:56]}",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )
        zs = float(self._zone.get("zone_score", 0.0))
        hv = bool(self._zone.get("has_valid_zone", False))
        cv2.putText(
            frame,
            f"zone: score={zs:.1f} valid={int(hv)}",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        if self._writer is None:
            self._writer = cv2.VideoWriter(
                str(self._output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self._out_fps),
                (w, h),
            )
        self._writer.write(frame)
        self._n_written += 1

    def _finalize(self) -> None:
        if self._done:
            return
        self._done = True
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.get_logger().info(f"wrote {self._n_written} frames -> {self._output}")


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/final_demo/final_landing_demo.mp4"))
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-width", type=int, default=1280, help="cap width (<=720p width typical)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = FinalDemoRecorder(
        output=args.output.resolve(),
        duration_s=float(args.duration),
        out_fps=float(args.fps),
        max_w=int(args.max_width),
    )

    deadline = time.time() + float(args.duration) + 120.0
    try:
        while rclpy.ok() and time.time() < deadline and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._done:
            node._finalize()
        report = {
            "video": str(args.output.resolve()),
            "frames": node._n_written,
            "saw_descend": node.events["descend"],
            "saw_abort": node.events["abort"],
            "saw_landed": node.events["landed"],
            "duration_requested_s": args.duration,
        }
        rep_path = Path("results/final_demo/demo_report.json")
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
