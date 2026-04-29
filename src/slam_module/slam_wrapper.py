from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SlamPose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    tracking_state: str


class OrbSlamWrapper:
    """Placeholder wrapper; replace with ORB-SLAM3 Python/C++ binding."""

    def __init__(self) -> None:
        self.initialized = True
        self._last_pos = np.zeros(3, dtype=float)

    def track(self, frame: np.ndarray, timestamp: float) -> SlamPose:
        _ = frame, timestamp
        # Lightweight synthetic pose for scaffold execution.
        self._last_pos = self._last_pos + np.array([0.01, 0.0, 0.0], dtype=float)
        return SlamPose(
            position=self._last_pos.copy(),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            tracking_state="OK",
        )
