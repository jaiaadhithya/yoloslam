"""DROID-SLAM backend (https://github.com/princeton-vl/DROID-SLAM)."""

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


def _droid_root() -> Path:
    env = os.environ.get("DROID_SLAM_PATH")
    if env:
        return Path(env)
    return _repo_root() / "third_party" / "DROID-SLAM"


class DroidSlam(SlamBackend):
    info = SlamBackendInfo(
        name="droid_slam",
        github="https://github.com/princeton-vl/DROID-SLAM",
        description="Deep recurrent optical-flow SLAM with differentiable bundle adjustment.",
        sensor_modes=["mono", "stereo", "rgbd"],
    )

    def __init__(
        self,
        *,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
        buffer: int = 512,
        upsample: bool = False,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.buffer = buffer
        self.upsample = upsample
        self._droid = None
        self._frame_idx = 0
        self._last_pose = np.eye(4, dtype=np.float64)
        self._ensure_import()

    def _ensure_import(self) -> None:
        root = _droid_root()
        if not root.exists():
            raise RuntimeError(
                f"DROID-SLAM not found at {root}. Run scripts/install_droid_slam.sh first."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from droid import Droid  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "DROID-SLAM Python package not importable. Install deps: "
                "torch, torchvision, lietorch, droid_backends (see third_party/DROID-SLAM)."
            ) from exc
        self._Droid = Droid

    def reset(self) -> None:
        self._droid = self._Droid(buffer=self.buffer, upsample=self.upsample)
        self._frame_idx = 0
        self._last_pose = np.eye(4, dtype=np.float64)

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        _ = depth_m
        if self._droid is None:
            self.reset()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t_ns = int(float(timestamp) * 1e9)
        intrinsics = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float32)
        self._droid.track(t_ns, rgb, intrinsics=intrinsics)
        self._frame_idx += 1

        if self._frame_idx < 8:
            return SlamTrackResult(
                position=self._last_pose[:3, 3].copy(),
                quaternion_xyzw=_quat_from_R(self._last_pose[:3, :3]),
                tracking_state="INIT",
                timestamp=timestamp,
            )

        traj = self._droid.terminate([rgb])
        if traj is None or len(traj) == 0:
            return SlamTrackResult(
                position=self._last_pose[:3, 3].copy(),
                quaternion_xyzw=_quat_from_R(self._last_pose[:3, :3]),
                tracking_state="LOST",
                timestamp=timestamp,
            )

        # Re-init streaming tracker after terminate snapshot.
        self.reset()
        T = np.asarray(traj[-1], dtype=np.float64)
        if T.shape == (4, 4):
            self._last_pose = T
        return SlamTrackResult(
            position=self._last_pose[:3, 3].copy(),
            quaternion_xyzw=_quat_from_R(self._last_pose[:3, :3]),
            tracking_state="OK",
            timestamp=timestamp,
        )


class DroidSlamStreaming(SlamBackend):
    """Streaming DROID tracker that keeps internal state between frames."""

    info = DroidSlam.info

    def __init__(self, **kwargs) -> None:
        self._cfg = kwargs
        self._droid = None
        self._frame_idx = 0
        self._poses: list[np.ndarray] = []
        self._Droid = None

    def _ensure(self) -> None:
        root = _droid_root()
        if str(root) not in sys.path and root.exists():
            sys.path.insert(0, str(root))
        from droid import Droid  # type: ignore

        self._Droid = Droid

    def reset(self) -> None:
        self._ensure()
        self._droid = self._Droid(
            buffer=int(self._cfg.get("buffer", 512)),
            upsample=bool(self._cfg.get("upsample", False)),
        )
        self._frame_idx = 0
        self._poses = []

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        _ = depth_m
        if self._droid is None:
            self.reset()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t_ns = int(float(timestamp) * 1e9)
        intrinsics = np.array(
            [
                float(self._cfg.get("fx", 525.0)),
                float(self._cfg.get("fy", 525.0)),
                float(self._cfg.get("cx", 319.5)),
                float(self._cfg.get("cy", 239.5)),
            ],
            dtype=np.float32,
        )
        self._droid.track(t_ns, rgb, intrinsics=intrinsics)
        self._frame_idx += 1
        state = "INIT" if self._frame_idx < 8 else "OK"
        T = np.eye(4, dtype=np.float64)
        if hasattr(self._droid, "video") and self._droid.video.poses_ is not None:
            n = len(self._droid.video.poses_)
            if n > 0:
                T = np.asarray(self._droid.video.poses_[n - 1].matrix().cpu().numpy(), dtype=np.float64)
        self._poses.append(T.copy())
        return SlamTrackResult(
            position=T[:3, 3].copy(),
            quaternion_xyzw=_quat_from_R(T[:3, :3]),
            tracking_state=state,
            timestamp=timestamp,
        )

    def run_dataset(
        self,
        dataset_dir: Path,
        *,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> tuple[list[float], np.ndarray]:
        from slam_module.backends.tum_loader import iter_tum_rgbd_frames

        self.reset()
        timestamps: list[float] = []
        poses: list[np.ndarray] = []
        frames = iter_tum_rgbd_frames(dataset_dir)
        selected = [frames[i] for i in range(0, len(frames), frame_stride)]
        if max_frames is not None:
            selected = selected[:max_frames]

        for sample in selected:
            self.track(sample.rgb_bgr, sample.timestamp, depth_m=sample.depth_m)
            timestamps.append(float(sample.timestamp))

        if self._droid is None:
            raise RuntimeError("DROID tracker not initialized")

        rgb_last = cv2.cvtColor(selected[-1].rgb_bgr, cv2.COLOR_BGR2RGB)
        traj = self._droid.terminate([rgb_last])
        if traj is None or len(traj) == 0:
            raise RuntimeError("DROID-SLAM produced empty trajectory")

        for T in traj:
            T = np.asarray(T, dtype=np.float64)
            q = _quat_from_R(T[:3, :3])
            poses.append(np.array([T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3]]))
        n = min(len(timestamps), len(poses))
        return timestamps[:n], np.stack(poses[:n], axis=0)


def install_droid_slam(dest: Path | None = None) -> Path:
    root = dest or _droid_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    if not (root / "droid").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/princeton-vl/DROID-SLAM.git", str(root)],
            check=True,
        )
    return root
