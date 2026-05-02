"""
Launch PX4 + Gazebo Baylands (x500_depth) with the Gazebo UI only.

No ROS 2 bridge, no MP4 recording, no demo motion script — use QGroundControl or
joystick, or just watch the vehicle in the sim.

Linux/WSL2: requires PX4-Autopilot with gz Baylands. On Windows, delegates to WSL.
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="PX4 + Gazebo Baylands (GUI only, no recorder)")
    parser.add_argument(
        "--render-engine",
        default="",
        metavar="ENGINE",
        help="GZ_SIM_RENDER_ENGINE (default ogre2); try ogre if the window fails to open",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not kill other px4/gz; use if you manage SITL yourself (avoids 'already running') errors only if nothing is running)",
    )
    parser.add_argument(
        "--software-gl",
        action="store_true",
        help="Software rendering (LLVMpipe); use if the Gazebo window still does not show",
    )
    parser.add_argument(
        "--qt-wayland",
        action="store_true",
        help="Set QT_QPA_PLATFORM=wayland (some WSL setups need this instead of xcb)",
    )
    args = parser.parse_args()

    repo = _repo_root()
    script = repo / "scripts" / "gazebo_baylands_ui.sh"
    if not script.is_file():
        print(f"Missing {script}", file=sys.stderr)
        return 1

    exports = []
    if args.render_engine:
        exports.append(f"export GZ_SIM_RENDER_ENGINE={shlex.quote(args.render_engine)}")
    if args.keep_existing:
        exports.append("export AUTO_KILL_EXISTING=0")
    if args.software_gl:
        exports.append("export SOFTWARE_GL=1")
    if args.qt_wayland:
        exports.append("export QT_QPA_PLATFORM=wayland")

    if platform.system() == "Windows":
        wsl_root = _win_path_to_wsl(repo)
        inner_parts = [f"cd {shlex.quote(wsl_root)}"] + exports + ["bash scripts/gazebo_baylands_ui.sh"]
        inner = " && ".join(inner_parts)
        print("Delegating to WSL:\n ", inner, flush=True)
        return subprocess.call(["wsl", "-e", "bash", "-lc", inner])

    import os

    env = os.environ.copy()
    if args.render_engine:
        env["GZ_SIM_RENDER_ENGINE"] = args.render_engine
    if args.keep_existing:
        env["AUTO_KILL_EXISTING"] = "0"
    if args.software_gl:
        env["SOFTWARE_GL"] = "1"
    if args.qt_wayland:
        env["QT_QPA_PLATFORM"] = "wayland"
    cmd = ["bash", str(script)]
    return subprocess.call(cmd, cwd=str(repo), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
