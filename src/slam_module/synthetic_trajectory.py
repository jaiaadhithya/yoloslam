"""
Deterministic synthetic UAV trajectory for placeholder SLAM (no ORB-SLAM3).

Smooth forward motion, lateral survey sweep, gradual descent, bounded jitter,
and an optional brief tracking-offset event with exponential recovery.
"""
from __future__ import annotations

import numpy as np


def yaw_to_quaternion_xyzw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)


class SyntheticUavTrajectory:
    """
    World frame: x forward, y left, z up (NED-style vertical inverted from aviation
    altitude plots — fusion code uses z up; altitude figures plot z vs time showing descent as decreasing z).
    """

    def __init__(
        self,
        *,
        dt: float = 0.05,
        seed: int = 42,
        glitch_start_step: int = 118,
        glitch_hold_end_step: int = 128,
        glitch_recovery_tau_steps: float = 14.0,
        glitch_offset_m: tuple[float, float, float] = (0.45, -0.19, 0.07),
        enable_glitch: bool = True,
    ) -> None:
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self._noise_phases = self.rng.uniform(0.0, 2.0 * np.pi, size=3)
        self.glitch_start_step = int(glitch_start_step)
        self.glitch_hold_end_step = int(glitch_hold_end_step)
        self.glitch_recovery_tau_steps = float(glitch_recovery_tau_steps)
        self._glitch_amp = np.array(glitch_offset_m, dtype=float)
        self.enable_glitch = bool(enable_glitch)

    def _nominal_position_velocity(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        w1, w2 = 0.21, 0.56
        x = 1.38 * t + 0.048 * np.sin(w1 * t) + 0.022 * np.sin(w2 * t)
        dx = 1.38 + 0.048 * w1 * np.cos(w1 * t) + 0.022 * w2 * np.cos(w2 * t)
        y = (
            0.52 * np.sin(0.155 * t)
            + 0.24 * np.sin(0.405 * t)
            + 0.055 * np.sin(0.71 * t + 0.35)
        )
        dy = (
            0.52 * 0.155 * np.cos(0.155 * t)
            + 0.24 * 0.405 * np.cos(0.405 * t)
            + 0.055 * 0.71 * np.cos(0.71 * t + 0.35)
        )
        z = 23.0 - 0.090 * t + 0.038 * np.sin(0.125 * t)
        dz = -0.090 + 0.038 * 0.125 * np.cos(0.125 * t)
        pos = np.array([x, y, z], dtype=float)
        vel = np.array([dx, dy, dz], dtype=float)
        return pos, vel

    def _bounded_jitter(self, t: float) -> np.ndarray:
        ph = self._noise_phases
        return np.array(
            [
                0.011 * np.sin(2.19 * t + ph[0]) + 0.007 * np.sin(4.97 * t + 0.42),
                0.014 * np.sin(1.86 * t + ph[1]) + 0.006 * np.sin(4.05 * t + 1.05),
                0.009 * np.sin(2.38 * t + ph[2]),
            ],
            dtype=float,
        )

    def _slow_drift(self, t: float) -> np.ndarray:
        return np.array(
            [
                0.019 * np.sin(0.068 * t),
                0.023 * np.sin(0.053 * t + 0.75),
                0.014 * np.sin(0.060 * t),
            ],
            dtype=float,
        )

    def _jitter_velocity(self, t: float) -> np.ndarray:
        ph = self._noise_phases
        return np.array(
            [
                0.011 * 2.19 * np.cos(2.19 * t + ph[0]) + 0.007 * 4.97 * np.cos(4.97 * t + 0.42),
                0.014 * 1.86 * np.cos(1.86 * t + ph[1]) + 0.006 * 4.05 * np.cos(4.05 * t + 1.05),
                0.009 * 2.38 * np.cos(2.38 * t + ph[2]),
            ],
            dtype=float,
        )

    def _drift_velocity(self, t: float) -> np.ndarray:
        return np.array(
            [
                0.019 * 0.068 * np.cos(0.068 * t),
                0.023 * 0.053 * np.cos(0.053 * t + 0.75),
                0.014 * 0.060 * np.cos(0.060 * t),
            ],
            dtype=float,
        )

    def _glitch_offset(self, k: int) -> np.ndarray:
        if not self.enable_glitch:
            return np.zeros(3)
        t0, t1 = self.glitch_start_step, self.glitch_hold_end_step
        tau = self.glitch_recovery_tau_steps
        if k < t0:
            return np.zeros(3)
        if k < t1:
            return self._glitch_amp.copy()
        return self._glitch_amp * np.exp(-(k - t1) / tau)

    def position_at_step(self, k: int) -> np.ndarray:
        t = k * self.dt
        p0, _ = self._nominal_position_velocity(t)
        return p0 + self._bounded_jitter(t) + self._slow_drift(t) + self._glitch_offset(k)

    def velocity_at_step(self, k: int) -> np.ndarray:
        """Analytic velocity (nominal + smooth perturbations); glitch uses backward diff for simplicity."""
        t = k * self.dt
        _, v0 = self._nominal_position_velocity(t)
        v = v0 + self._jitter_velocity(t) + self._drift_velocity(t)
        if self.enable_glitch and k > 0:
            v_g = (self._glitch_offset(k) - self._glitch_offset(k - 1)) / self.dt
            v = v + v_g
        return v

    def quaternion_at_step(self, k: int) -> np.ndarray:
        v = self.velocity_at_step(k)
        planar = float(np.hypot(v[0], v[1]))
        yaw = float(np.arctan2(v[1], v[0])) if planar > 1e-4 else 0.0
        return yaw_to_quaternion_xyzw(yaw)

    def state_at_step(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.position_at_step(k), self.velocity_at_step(k), self.quaternion_at_step(k)


def precompute_trajectory(
    num_steps: int,
    *,
    dt: float = 0.05,
    seed: int = 42,
    t0: float = 0.0,
    enable_glitch: bool = True,
) -> dict[str, np.ndarray]:
    """Vectorized trajectory for CSV export and plots."""
    traj = SyntheticUavTrajectory(dt=dt, seed=seed, enable_glitch=enable_glitch)
    n = int(num_steps)
    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)
    for k in range(n):
        pos[k], vel[k], _ = traj.state_at_step(k)
    ts = t0 + np.arange(n, dtype=float) * dt
    speed = np.linalg.norm(vel, axis=1)
    return {
        "timestamp": ts,
        "x": pos[:, 0],
        "y": pos[:, 1],
        "z": pos[:, 2],
        "vx": vel[:, 0],
        "vy": vel[:, 1],
        "vz": vel[:, 2],
        "speed": speed,
    }
