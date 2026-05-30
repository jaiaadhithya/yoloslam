from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from slam_module.backends.base import SlamBackend, SlamTrackResult
from slam_module.slam_factory import create_backend, load_slam_config
from slam_module.synthetic_trajectory import SyntheticUavTrajectory


@dataclass
class SlamPose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    tracking_state: str
    velocity: np.ndarray | None = None
    timestamp: float | None = None


class SyntheticSlamWrapper:
    """Legacy synthetic trajectory (for unit tests / offline demos without vision)."""

    def __init__(
        self,
        *,
        dt: float = 0.05,
        traj_seed: int = 42,
        enable_glitch: bool = True,
    ) -> None:
        self.initialized = True
        self._dt = float(dt)
        self._traj = SyntheticUavTrajectory(dt=dt, seed=traj_seed, enable_glitch=enable_glitch)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def track(self, frame: np.ndarray, timestamp: float) -> SlamPose:
        _ = frame
        k = self._step
        self._step += 1
        pos, vel, quat = self._traj.state_at_step(k)
        return SlamPose(
            position=pos,
            quaternion_xyzw=quat,
            tracking_state="OK",
            velocity=vel.copy(),
            timestamp=float(timestamp),
        )


class SlamWrapper:
    """Unified SLAM wrapper: real vision backends or synthetic fallback."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        config: dict[str, Any] | None = None,
        use_synthetic: bool = False,
        dt: float = 0.05,
        traj_seed: int = 42,
        enable_glitch: bool = True,
    ) -> None:
        self.use_synthetic = use_synthetic
        self._synthetic: SyntheticSlamWrapper | None = None
        self._backend: SlamBackend | None = None
        self._trajectory_cache: list[SlamTrackResult] | None = None
        self._cache_idx = 0

        if use_synthetic or backend == "synthetic":
            self.use_synthetic = True
            self._synthetic = SyntheticSlamWrapper(
                dt=dt, traj_seed=traj_seed, enable_glitch=enable_glitch
            )
            self.initialized = True
            self.backend_name = "synthetic"
            return

        cfg = config or load_slam_config()
        if backend is not None:
            cfg = dict(cfg)
            cfg["backend"] = backend
        self._backend = create_backend(cfg=cfg)
        self.backend_name = str(cfg.get("backend", "opencv_orb_rgbd"))
        self.initialized = True

    def reset(self) -> None:
        self._cache_idx = 0
        if self._synthetic is not None:
            self._synthetic.reset()
        elif self._backend is not None:
            self._backend.reset()

    def preload_trajectory(self, results: list[SlamTrackResult]) -> None:
        self._trajectory_cache = results
        self._cache_idx = 0

    def track(
        self,
        frame: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamPose:
        if self._trajectory_cache is not None:
            if self._cache_idx >= len(self._trajectory_cache):
                cached = self._trajectory_cache[-1]
            else:
                cached = self._trajectory_cache[self._cache_idx]
                self._cache_idx += 1
            return SlamPose(
                position=cached.position.copy(),
                quaternion_xyzw=cached.quaternion_xyzw.copy(),
                tracking_state=cached.tracking_state,
                velocity=cached.velocity.copy() if cached.velocity is not None else None,
                timestamp=float(timestamp),
            )

        if self._synthetic is not None:
            return self._synthetic.track(frame, timestamp)

        if self._backend is None:
            raise RuntimeError("SLAM wrapper has no backend configured")

        result = self._backend.track(frame, timestamp, depth_m=depth_m)
        return SlamPose(
            position=result.position.copy(),
            quaternion_xyzw=result.quaternion_xyzw.copy(),
            tracking_state=result.tracking_state,
            velocity=result.velocity.copy() if result.velocity is not None else None,
            timestamp=float(timestamp),
        )


# Backward-compatible alias used across the codebase.
OrbSlamWrapper = SlamWrapper
