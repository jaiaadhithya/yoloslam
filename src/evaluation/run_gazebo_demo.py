"""
Single entry: PX4 SITL + Gazebo Baylands + x500_depth + /camera recording -> results/gazebo_demo/demo.mp4

Linux / WSL2 only (requires ROS 2, Gazebo Harmonic, PX4-Autopilot). On Windows hosts, delegates to WSL.
"""
from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _win_path_to_wsl(p: Path) -> str:
    p = p.resolve()
    if not p.drive:
        return str(p).replace("\\", "/")
    drive = p.drive.rstrip(":").lower()
    rest = "/".join(p.parts[1:])
    return f"/mnt/{drive}/{rest}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PX4 + Gazebo Baylands demo video (low-spec default)",
    )
    parser.add_argument(
        "--low-spec",
        dest="low_spec",
        action="store_true",
        default=True,
        help="Headless-friendly defaults, 640x480 @ ~12 FPS capture (default: on)",
    )
    parser.add_argument(
        "--no-low-spec",
        "--gui",
        dest="low_spec",
        action="store_false",
        help="Show the Gazebo window (HEADLESS=0; needs WSLg or an X server on WSL)",
    )
    parser.add_argument("--duration", type=int, default=15, help="Recording length seconds (10–20 typical)")
    parser.add_argument("--fps", type=int, default=12, help="Output video FPS")
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Skip px4_demo_motion; only bridge + record /camera (debug video pipeline)",
    )
    parser.add_argument(
        "--xvfb",
        action="store_true",
        help="Run PX4 SITL under xvfb-run (virtual framebuffer; helps some WSL headless setups)",
    )
    parser.add_argument(
        "--render-engine",
        default="",
        metavar="ENGINE",
        help="Set GZ_SIM_RENDER_ENGINE in WSL (e.g. ogre or ogre2)",
    )
    parser.add_argument(
        "--image-bridge",
        action="store_true",
        help="Use ros_gz_image (opt-in; needs ros_gz matching gz-harmonic or bridge.log fills with Unknown message type)",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    repo = _repo_root()
    script = repo / "scripts" / "gazebo_demo_lowspec.sh"

    if platform.system() == "Windows" and not args.native:
        wsl_root = _win_path_to_wsl(repo)
        low = "1" if args.low_spec else "0"
        exports = [
            f"export LOW_SPEC={low}",
            f"export DURATION={int(args.duration)}",
            f"export RECORD_FPS={int(args.fps)}",
        ]
        if args.camera_only:
            exports.append("export SKIP_MOTION=1")
        if args.xvfb:
            exports.append("export USE_XVFB=1")
        if args.render_engine:
            exports.append(f"export GZ_SIM_RENDER_ENGINE={shlex.quote(args.render_engine)}")
        if args.image_bridge:
            exports.append("export USE_IMAGE_BRIDGE=1")
        inner = f"cd {shlex.quote(wsl_root)} && {' && '.join(exports)} && bash scripts/gazebo_demo_lowspec.sh"
        print("Delegating to WSL:\n ", inner, flush=True)
        if args.low_spec:
            print(
                "Tip: default is headless (no Gazebo window). Add --gui for a visible sim.",
                flush=True,
            )
        return subprocess.call(["wsl", "-e", "bash", "-lc", inner])

    if not script.is_file():
        print(f"Missing {script}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["LOW_SPEC"] = "1" if args.low_spec else "0"
    env["DURATION"] = str(int(args.duration))
    env["RECORD_FPS"] = str(int(args.fps))
    if args.camera_only:
        env["SKIP_MOTION"] = "1"
    if args.xvfb:
        env["USE_XVFB"] = "1"
    if args.render_engine:
        env["GZ_SIM_RENDER_ENGINE"] = args.render_engine
    if args.image_bridge:
        env["USE_IMAGE_BRIDGE"] = "1"
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OUTPUT_MP4"] = str((repo / "results" / "gazebo_demo" / "demo.mp4").resolve())

    return subprocess.call(["bash", str(script)], cwd=str(repo), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
