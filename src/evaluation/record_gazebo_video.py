"""
Record a short MP4 from the Gazebo x500 RGB topic via ROS 2 + ros_gz_bridge.

Default output: results/gazebo_demo/demo.mp4

Typical workflow (Linux/WSL, ROS 2 Humble + ros_gz_bridge):
  1) Terminal A: ./scripts/px4_baylands_minimal.sh
  2) Terminal B: PYTHONPATH=src python3 -m evaluation.record_gazebo_video --with-bridge
  Optional motion: PYTHONPATH=src python3 -m evaluation.px4_demo_motion --gentle

This module does not start PX4; it only bridges (optional) and records /camera.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy

from evaluation.gazebo_camera_recorder import CameraRecorder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _gpu_status_line() -> str:
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1":
        return "software_gl (LIBGL_ALWAYS_SOFTWARE=1)"
    v = shutil.which("nvidia-smi")
    if v and subprocess.call([v, "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        return "gpu_hint (nvidia-smi present; LIBGL not forced to software)"
    return "gpu_preferred (LIBGL_ALWAYS_SOFTWARE unset; vendor depends on host/WSLg)"


def _wait_for_camera(timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if out.returncode == 0 and "/camera" in out.stdout.splitlines():
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        time.sleep(1.0)
    return False


def _start_bridge_alt(log_fp) -> subprocess.Popen:
    gz_topic = "/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
    cmd = [
        "ros2",
        "run",
        "ros_gz_bridge",
        "parameter_bridge",
        f"{gz_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
        "--ros-args",
        "-r",
        f"{gz_topic}:=/camera",
    ]
    return subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT)


def _estimate_rtf(world: str, samples: int = 10) -> float | None:
    topic = f"/world/{world}/stats"
    try:
        proc = subprocess.run(
            ["gz", "topic", "-e", "-t", topic, "-n", str(max(2, samples))],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    # Parse nested "sec:" lines from gz protobuf text format.
    secs = [int(m.group(1)) for m in re.finditer(r"^\s*sec:\s*(\d+)\s*$", proc.stdout, re.MULTILINE)]
    nsecs = [int(m.group(1)) for m in re.finditer(r"^\s*nsec:\s*(\d+)\s*$", proc.stdout, re.MULTILINE)]
    if len(secs) < 4 or len(nsecs) < 4:
        return None
    # Heuristic: first pair ~ real time, second pair ~ sim time (message layout varies by version).
    r0 = float(secs[0]) + float(nsecs[0]) * 1e-9
    r1 = float(secs[1]) + float(nsecs[1]) * 1e-9
    s0 = float(secs[2]) + float(nsecs[2]) * 1e-9
    s1 = float(secs[3]) + float(nsecs[3]) * 1e-9
    dr = r1 - r0
    ds = s1 - s0
    if dr <= 1e-6 or ds <= 1e-9:
        return None
    return float(ds / dr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Gazebo /camera to MP4 (10–15s demo)")
    parser.add_argument("--output", type=Path, default=Path("results/gazebo_demo/demo.mp4"))
    parser.add_argument("--duration", type=float, default=12.0, help="Wall seconds to record (10–15 typical)")
    parser.add_argument("--fps", type=float, default=12.0, help="Writer FPS (<=15; downsample heavy streams)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--world", type=str, default="baylands", help="Gazebo world name for gz bridge + RTF topic")
    parser.add_argument(
        "--with-bridge",
        action="store_true",
        help="Spawn ros_gz_bridge for IMX214 RGB (baylands path, then default-world fallback)",
    )
    parser.add_argument("--bridge-log", type=Path, default=Path("results/gazebo_demo/bridge_record.log"))
    parser.add_argument("--metrics", type=Path, default=Path("results/gazebo_demo/demo_metrics.json"))
    parser.add_argument("--report", type=Path, default=Path("results/gazebo_demo/record_report.json"))
    parser.add_argument("--overlay", action="store_true", help="Draw simple overlay on frames")
    parser.add_argument("--skip-rtf", action="store_true")
    args = parser.parse_args()

    if args.duration < 8 or args.duration > 20:
        print("WARN: duration outside 8–20s; proceeding.", file=sys.stderr)

    os.chdir(_repo_root())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    bridge_proc: subprocess.Popen | None = None
    log_fp = None
    if args.with_bridge:
        log_fp = open(args.bridge_log, "w", encoding="utf-8")
        gz_topic = f"/world/{args.world}/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
        cmd = [
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            f"{gz_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            "--ros-args",
            "-r",
            f"{gz_topic}:=/camera",
        ]
        bridge_proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT)
        print(f"Bridge PID={bridge_proc.pid} log={args.bridge_log}", flush=True)
        if not _wait_for_camera(timeout_s=120.0):
            print("WARN: /camera not on baylands path; retrying default world bridge…", flush=True)
            bridge_proc.terminate()
            try:
                bridge_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge_proc.kill()
            bridge_proc = _start_bridge_alt(log_fp)
            if not _wait_for_camera(timeout_s=60.0):
                print("ERROR: /camera never appeared. Is PX4 + Gazebo running with x500_depth?", file=sys.stderr)
                if bridge_proc:
                    bridge_proc.terminate()
                if log_fp:
                    log_fp.close()
                return 1

    rclpy.init()
    node = CameraRecorder(
        output=args.output,
        duration=float(args.duration),
        fps=float(args.fps),
        overlay=bool(args.overlay),
        metrics_path=args.metrics,
        width=int(args.width),
        height=int(args.height),
    )
    deadline = time.time() + float(args.duration) + 120.0
    try:
        while rclpy.ok() and time.time() < deadline and not node._finalized:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._finalized:
            node.get_logger().warn("Stopping early; finalizing recorder.")
            node._finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    rtf: float | None = None
    if not args.skip_rtf:
        rtf = _estimate_rtf(args.world)
        if rtf is None:
            rtf = _estimate_rtf("default")

    report = {
        "video_path": str(args.output.resolve()),
        "metrics_path": str(args.metrics.resolve()),
        "duration_requested_s": args.duration,
        "target_writer_fps": args.fps,
        "resize": [args.width, args.height],
        "gz_world": args.world,
        "rtf_estimate": rtf,
        "gpu_status": _gpu_status_line(),
        "libgl_always_software": os.environ.get("LIBGL_ALWAYS_SOFTWARE", ""),
    }
    if args.with_bridge:
        report["bridge_log"] = str(args.bridge_log.resolve())

    # Merge recorder metrics if present
    if args.metrics.is_file():
        try:
            report["recorder"] = json.loads(args.metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["recorder"] = None

    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(args.report.resolve()), "rtf_estimate": rtf}, indent=2), flush=True)

    if bridge_proc is not None:
        bridge_proc.send_signal(signal.SIGTERM)
        try:
            bridge_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            bridge_proc.kill()
    if log_fp is not None:
        log_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
