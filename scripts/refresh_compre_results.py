#!/usr/bin/env python3
"""Run SLAM benchmark + PyBullet eval and refresh compre/ artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "src") + ((";" if sys.platform == "win32" else ":") + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    _run([sys.executable, str(ROOT / "scripts" / "run_slam_comparison.py"), "--max-frames", "792", "--frame-stride", "1"])
    _run([sys.executable, str(ROOT / "scripts" / "eval_pybullet_slam.py"), "--steps", "240"])
    print("compre/ refreshed (TUM benchmark + PyBullet SLAM eval).")


if __name__ == "__main__":
    main()
