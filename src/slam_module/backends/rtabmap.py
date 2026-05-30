"""RTAB-Map RGB-D visual SLAM via Python bindings when available."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from slam_module.backends.base import SlamBackend, SlamBackendInfo, SlamTrackResult
from slam_module.backends.opencv_orb_rgbd import _quat_from_R


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class RtabmapRgbdSlam(SlamBackend):
    info = SlamBackendInfo(
        name="rtabmap",
        github="https://github.com/introlab/rtabmap",
        description="Graph-based RGB-D SLAM with loop closure (RTAB-Map).",
        sensor_modes=["rgbd", "stereo", "lidar"],
    )

    def __init__(
        self,
        *,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self._rtabmap = None
        self._odom = None

    def _ensure(self) -> None:
        try:
            import rtabmap  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "rtabmap Python bindings not installed. "
                "Install RTAB-Map with Python support or use scripts/install_rtabmap_wsl.sh."
            ) from exc
        self._rtabmap = rtabmap
        if self._odom is None:
            params = rtabmap.Parameters()
            params.setRGBDEnabled(True)
            self._odom = rtabmap.Odometry(params)
            self._odom.init(f"{self.fx} {self.fy} {self.cx} {self.cy}")

    def reset(self) -> None:
        self._odom = None
        self._ensure()
        self._odom.reset()

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        self._ensure()
        if depth_m is None:
            raise ValueError("RTAB-Map RGB-D backend requires depth")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_mm = np.nan_to_num(depth_m, nan=0.0) * 1000.0
        depth_mm = depth_mm.astype(np.float32)
        odom_pose = self._odom.process(rgb, depth_mm, float(timestamp))
        T = np.asarray(odom_pose, dtype=np.float64).reshape(4, 4)
        return SlamTrackResult(
            position=T[:3, 3].copy(),
            quaternion_xyzw=_quat_from_R(T[:3, :3]),
            tracking_state="OK",
            timestamp=timestamp,
        )


class RtabmapCliSlam(SlamBackend):
    """Run RTAB-Map console RGB-D odometry through WSL if Python bindings missing."""

    info = RtabmapRgbdSlam.info

    def __init__(self) -> None:
        self._poses: list[np.ndarray] = []
        self._idx = 0

    def reset(self) -> None:
        self._poses = []
        self._idx = 0

    def is_available(self) -> bool:
        try:
            proc = subprocess.run(
                ["wsl", "bash", "-lc", "command -v rtabmap-rgbd_odometry"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return proc.returncode == 0 and proc.stdout.strip() != ""
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def run_dataset(
        self,
        dataset_dir: Path,
        *,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> tuple[list[float], np.ndarray]:
        from slam_module.backends.tum_loader import iter_tum_rgbd_frames, write_tum_poses

        if not self.is_available():
            raise RuntimeError("rtabmap-rgbd_odometry not found in WSL")

        dataset_dir = Path(dataset_dir).resolve()
        work = dataset_dir.parent / ".rtabmap_run"
        work.mkdir(parents=True, exist_ok=True)
        odom_db = work / "odom.db"
        if odom_db.exists():
            odom_db.unlink()

        rgb_list = work / "rgb_list.txt"
        depth_list = work / "depth_list.txt"
        rgb_rows = []
        depth_rows = []
        frames = iter_tum_rgbd_frames(dataset_dir)
        for i, sample in enumerate(frames):
            if i % frame_stride != 0:
                continue
            if max_frames is not None and len(rgb_rows) >= max_frames:
                break
            if sample.depth_m is None:
                continue
            rgb_rel = f"rgb_{len(rgb_rows):06d}.png"
            depth_rel = f"depth_{len(depth_rows):06d}.png"
            cv2.imwrite(str(work / rgb_rel), sample.rgb_bgr)
            depth_mm = np.nan_to_num(sample.depth_m, nan=0.0) * 1000.0
            cv2.imwrite(str(work / depth_rel), depth_mm.astype(np.uint16))
            rgb_rows.append((sample.timestamp, rgb_rel))
            depth_rows.append((sample.timestamp, depth_rel))

        with rgb_list.open("w", encoding="utf-8") as f:
            for t, rel in rgb_rows:
                f.write(f"{t:.9f} {rel}\n")
        with depth_list.open("w", encoding="utf-8") as f:
            for t, rel in depth_rows:
                f.write(f"{t:.9f} {rel}\n")

        wsl_work = subprocess.run(
            ["wsl", "wslpath", "-a", str(work)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        cmd = (
            f"cd '{wsl_work}' && rtabmap-rgbd_odometry "
            f"--rgb_list rgb_list.txt --depth_list depth_list.txt "
            f"--calibration '{525.0} {525.0} {319.5} {239.5}' "
            f"--output odom.txt --delete_db_on_start"
        )
        subprocess.run(["wsl", "bash", "-lc", cmd], check=True)
        odom_txt = work / "odom.txt"
        if not odom_txt.exists():
            raise RuntimeError(f"RTAB-Map did not produce {odom_txt}")

        timestamps: list[float] = []
        poses: list[np.ndarray] = []
        with odom_txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                timestamps.append(float(parts[0]))
                poses.append(np.array([float(x) for x in parts[1:8]], dtype=np.float64))
        return timestamps, np.stack(poses, axis=0)

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        _ = frame_bgr, depth_m
        if self._idx >= len(self._poses):
            return SlamTrackResult(
                position=np.zeros(3),
                quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                tracking_state="LOST",
                timestamp=timestamp,
            )
        row = self._poses[self._idx]
        self._idx += 1
        return SlamTrackResult(
            position=row[:3].copy(),
            quaternion_xyzw=row[3:7].copy(),
            tracking_state="OK",
            timestamp=timestamp,
        )
