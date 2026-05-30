"""SLAM-compatible pose streams from PyBullet ground truth or simple noisy VO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


@dataclass
class PoseProviderConfig:
    use_noisy_vo: bool = False
    pos_sigma_m: float = 0.08
    yaw_sigma_rad: float = 0.03
    z_sigma_m: float = 0.05
    bias_rwalk_pos: float = 0.002
    bias_rwalk_yaw: float = 0.0005
    seed: int = 0


class PoseProvider:
    """Produces slam_pose dicts: x, y, z, yaw, state."""

    def __init__(self, cfg: PoseProviderConfig | None = None) -> None:
        self.cfg = cfg or PoseProviderConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._bias = np.zeros(3, dtype=float)
        self._yaw_bias = 0.0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)
        self._bias[:] = 0.0
        self._yaw_bias = 0.0

    def from_truth(self, x: float, y: float, z: float, yaw: float) -> dict:
        if not self.cfg.use_noisy_vo:
            return {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw), "state": "OK"}

        self._bias[:2] += self._rng.normal(0.0, self.cfg.bias_rwalk_pos, size=2)
        self._bias[2] += self._rng.normal(0.0, self.cfg.bias_rwalk_pos * 0.5)
        self._yaw_bias = _wrap_angle(self._yaw_bias + float(self._rng.normal(0.0, self.cfg.bias_rwalk_yaw)))

        nx = float(x + self._bias[0] + self._rng.normal(0.0, self.cfg.pos_sigma_m))
        ny = float(y + self._bias[1] + self._rng.normal(0.0, self.cfg.pos_sigma_m))
        nz = float(z + self._bias[2] + self._rng.normal(0.0, self.cfg.z_sigma_m))
        nyaw = _wrap_angle(float(yaw + self._yaw_bias + self._rng.normal(0.0, self.cfg.yaw_sigma_rad)))
        return {"x": nx, "y": ny, "z": nz, "yaw": nyaw, "state": "OK"}
