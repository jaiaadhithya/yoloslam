import time
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from slam_module.slam_wrapper import OrbSlamWrapper


@dataclass
class PoseStampedLike:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    timestamp: float
    state: str


class SlamNode:
    def __init__(self) -> None:
        self.wrapper = OrbSlamWrapper()

    def process_frame(self, frame: np.ndarray) -> Dict:
        pose = self.wrapper.track(frame, time.time())
        msg = PoseStampedLike(
            x=float(pose.position[0]),
            y=float(pose.position[1]),
            z=float(pose.position[2]),
            qx=float(pose.quaternion_xyzw[0]),
            qy=float(pose.quaternion_xyzw[1]),
            qz=float(pose.quaternion_xyzw[2]),
            qw=float(pose.quaternion_xyzw[3]),
            timestamp=time.time(),
            state=pose.tracking_state,
        )
        return asdict(msg)


if __name__ == "__main__":
    node = SlamNode()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tick = 0
    while True:
        pose = node.process_frame(frame)
        if tick % 5 == 0:
            print(
                {
                    "x": round(pose["x"], 3),
                    "y": round(pose["y"], 3),
                    "z": round(pose["z"], 3),
                    "state": pose["state"],
                },
                flush=True,
            )
        tick += 1
        time.sleep(1.0)
