"""Simple visual-odometry-style pose estimate for offline demos (matches fusion dict shape)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisualOdometryConfig:
    vel_noise_std_m_s: float = 0.035
    yaw_noise_std_rad: float = 0.012
    seed: int = 0


class VisualOdometrySlam:
    """Integrates world-frame velocities with noise; yaw loosely tracks a reference (e.g. true yaw)."""

    def __init__(self, cfg: VisualOdometryConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(int(cfg.seed))
        self.x = 0.0
        self.y = 0.0
        self.z = 12.0
        self.yaw = 0.0

    def reset(self, x: float, y: float, z: float, yaw: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = float(yaw)

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw, "state": "OK"}

    def propagate(self, dt: float, vx: float, vy: float, vz: float, yaw_ref: float) -> None:
        if dt <= 0.0:
            return
        sn = float(self.cfg.vel_noise_std_m_s)
        nx = float(self._rng.normal(0.0, sn))
        ny = float(self._rng.normal(0.0, sn))
        nz = float(self._rng.normal(0.0, sn * 0.65))
        self.x += (float(vx) + nx) * dt
        self.y += (float(vy) + ny) * dt
        self.z += (float(vz) + nz) * dt
        self.z = max(0.05, self.z)
        yn = float(self._rng.normal(0.0, float(self.cfg.yaw_noise_std_rad)))
        self.yaw = float(yaw_ref + yn)
        self.yaw = float(np.arctan2(np.sin(self.yaw), np.cos(self.yaw)))
