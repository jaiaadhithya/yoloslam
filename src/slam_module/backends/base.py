"""Common SLAM backend interface for benchmark + runtime integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SlamBackendInfo:
    name: str
    github: str
    description: str
    sensor_modes: list[str] = field(default_factory=list)


@dataclass
class SlamTrackResult:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    tracking_state: str = "OK"
    velocity: np.ndarray | None = None
    timestamp: float | None = None


class SlamBackend(ABC):
    info: SlamBackendInfo

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        raise NotImplementedError

    def run_dataset(
        self,
        dataset_dir: Path,
        *,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> tuple[list[float], np.ndarray]:
        """Run on TUM RGB-D layout; returns (timestamps, Nx7 TUM poses)."""
        from slam_module.backends.tum_loader import iter_tum_rgbd_frames

        self.reset()
        timestamps: list[float] = []
        poses: list[np.ndarray] = []
        for i, sample in enumerate(iter_tum_rgbd_frames(dataset_dir)):
            if i % frame_stride != 0:
                continue
            if max_frames is not None and len(timestamps) >= max_frames:
                break
            result = self.track(
                sample.rgb_bgr,
                sample.timestamp,
                depth_m=sample.depth_m,
            )
            timestamps.append(float(sample.timestamp))
            qx, qy, qz, qw = result.quaternion_xyzw
            px, py, pz = result.position
            poses.append(np.array([px, py, pz, qx, qy, qz, qw], dtype=np.float64))
        if not poses:
            raise RuntimeError(f"{self.info.name}: no poses produced")
        return timestamps, np.stack(poses, axis=0)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.info.name,
            "github": self.info.github,
            "description": self.info.description,
            "sensor_modes": self.info.sensor_modes,
        }
