from dataclasses import dataclass

import numpy as np

from slam_module.synthetic_trajectory import SyntheticUavTrajectory


@dataclass
class SlamPose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    tracking_state: str
    velocity: np.ndarray | None = None
    timestamp: float | None = None


class OrbSlamWrapper:
    """Placeholder wrapper: realistic synthetic UAV trajectory until ORB-SLAM3 is integrated."""

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
