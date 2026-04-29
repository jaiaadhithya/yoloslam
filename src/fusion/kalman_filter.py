from dataclasses import dataclass

import numpy as np


@dataclass
class EKFState:
    x: np.ndarray
    p: np.ndarray


class LandingTargetEKF:
    def __init__(self, dt: float = 1.0 / 30.0) -> None:
        self.dt = dt
        self.f = np.eye(6)
        self.f[0, 3] = dt
        self.f[1, 4] = dt
        self.f[2, 5] = dt
        self.h = np.zeros((3, 6))
        self.h[0, 0] = self.h[1, 1] = self.h[2, 2] = 1.0
        self.q = np.eye(6) * 1e-3
        self.r_base = np.eye(3) * 2e-2
        self.state = EKFState(x=np.zeros(6), p=np.eye(6))

    def predict(self) -> None:
        self.state.x = self.f @ self.state.x
        self.state.p = self.f @ self.state.p @ self.f.T + self.q

    def update(self, z: np.ndarray, confidence: float) -> None:
        conf = np.clip(confidence, 1e-3, 1.0)
        r = self.r_base / conf
        y = z - self.h @ self.state.x
        s = self.h @ self.state.p @ self.h.T + r
        m_dist = float(y.T @ np.linalg.inv(s) @ y)
        if m_dist > 16.0:
            return
        k = self.state.p @ self.h.T @ np.linalg.inv(s)
        self.state.x = self.state.x + k @ y
        i = np.eye(6)
        self.state.p = (i - k @ self.h) @ self.state.p

    def current_position(self) -> np.ndarray:
        return self.state.x[:3].copy()
