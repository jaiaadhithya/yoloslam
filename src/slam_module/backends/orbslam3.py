"""ORB-SLAM3 external backend (https://github.com/UZ-SLAMLab/ORB_SLAM3)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from slam_module.backends.base import SlamBackend, SlamBackendInfo, SlamTrackResult
from slam_module.backends.opencv_orb_rgbd import _quat_from_R


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class OrbSlam3External(SlamBackend):
    info = SlamBackendInfo(
        name="orb_slam3",
        github="https://github.com/UZ-SLAMLab/ORB_SLAM3",
        description="ORB feature-based multi-map visual SLAM (mono/stereo/RGB-D/VIO).",
        sensor_modes=["mono", "stereo", "rgbd", "vio"],
    )

    def __init__(
        self,
        *,
        mode: str = "rgbd",
        orbslam3_root: Path | None = None,
        use_wsl: bool | None = None,
    ) -> None:
        self.mode = mode
        self.orbslam3_root = orbslam3_root or Path(
            os.environ.get("ORBSLAM3_ROOT", str(_repo_root() / "third_party" / "ORB_SLAM3"))
        )
        if use_wsl is None:
            use_wsl = os.name == "nt"
        self.use_wsl = use_wsl
        self._poses: list[np.ndarray] = []
        self._idx = 0

    def reset(self) -> None:
        self._poses = []
        self._idx = 0

    def _wsl_path(self, p: Path) -> str:
        proc = subprocess.run(
            ["wsl", "wslpath", "-a", str(p.resolve())],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    def _binary(self) -> Path:
        if self.mode == "rgbd":
            rel = Path("Examples/RGB-D/rgbd_tum")
        elif self.mode == "mono":
            rel = Path("Examples/Monocular/mono_tum")
        else:
            rel = Path("Examples/RGB-D/rgbd_tum")
        return self.orbslam3_root / rel

    def is_available(self) -> bool:
        exe = self._binary()
        if self.use_wsl:
            try:
                wsl_exe = self._wsl_path(exe)
                proc = subprocess.run(
                    ["wsl", "bash", "-lc", f"test -x '{wsl_exe}' && echo OK"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                return proc.returncode == 0 and "OK" in proc.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
        return exe.exists()

    def run_dataset(
        self,
        dataset_dir: Path,
        *,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> tuple[list[float], np.ndarray]:
        if not self.is_available():
            raise RuntimeError(
                f"ORB-SLAM3 binary missing at {self._binary()}. "
                "Run scripts/install_orbslam3_wsl.sh in WSL/Linux."
            )

        from slam_module.backends.tum_loader import write_tum_associations

        dataset_dir = Path(dataset_dir).resolve()
        assoc_path = dataset_dir / "associations.txt"
        if not assoc_path.exists():
            write_tum_associations(assoc_path, dataset_dir)
        out_dir = dataset_dir.parent / ".orbslam3_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = out_dir / "CameraTrajectory.txt"

        vocab = self.orbslam3_root / "Vocabulary" / "ORBvoc.txt"
        yaml_map = {
            "rgbd": self.orbslam3_root / "Examples/RGB-D/TUM1.yaml",
            "mono": self.orbslam3_root / "Examples/Monocular/TUM1.yaml",
        }
        settings = yaml_map.get(self.mode, yaml_map["rgbd"])
        exe = self._binary()

        if self.use_wsl:
            wsl_dataset = self._wsl_path(dataset_dir)
            wsl_vocab = self._wsl_path(vocab)
            wsl_settings = self._wsl_path(settings)
            wsl_exe = self._wsl_path(exe)
            wsl_out = self._wsl_path(out_dir)
            cmd = (
                f"cd '{wsl_out}' && "
                f"'{wsl_exe}' '{wsl_vocab}' '{wsl_settings}' '{wsl_dataset}' "
                f"associations.txt"
            )
            subprocess.run(["wsl", "bash", "-lc", cmd], check=True)
        else:
            subprocess.run(
                [str(exe), str(vocab), str(settings), str(dataset_dir), "associations.txt"],
                cwd=str(out_dir),
                check=True,
            )

        if not traj_path.exists():
            alt = out_dir / "KeyFrameTrajectory.txt"
            if alt.exists():
                traj_path = alt
            else:
                raise RuntimeError(f"ORB-SLAM3 did not write trajectory to {out_dir}")

        timestamps, poses = self._parse_tum_trajectory(traj_path)
        if frame_stride > 1:
            timestamps = timestamps[::frame_stride]
            poses = poses[::frame_stride]
        if max_frames is not None:
            timestamps = timestamps[:max_frames]
            poses = poses[:max_frames]
        return timestamps, poses

    @staticmethod
    def _parse_tum_trajectory(path: Path) -> tuple[list[float], np.ndarray]:
        timestamps: list[float] = []
        rows: list[list[float]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                timestamps.append(float(parts[0]))
                rows.append([float(x) for x in parts[1:8]])
        if not rows:
            raise RuntimeError(f"No poses parsed from {path}")
        return timestamps, np.asarray(rows, dtype=np.float64)

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

    def preload_trajectory(self, timestamps: list[float], poses: np.ndarray) -> None:
        self._poses = [poses[i] for i in range(len(poses))]
        self._timestamps = timestamps
        self._idx = 0

    def availability_report(self) -> dict:
        return {
            "available": self.is_available(),
            "binary": str(self._binary()),
            "orbslam3_root": str(self.orbslam3_root),
            "use_wsl": self.use_wsl,
        }
