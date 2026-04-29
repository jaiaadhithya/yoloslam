from dataclasses import asdict, dataclass
from math import atan2, sqrt
from typing import Dict
import time

from landing_controller.pid import PID, PIDGains
from landing_controller.state_machine import LandingState, LandingStateMachine


@dataclass
class VelocityCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    state: str


class ControllerNode:
    def __init__(self) -> None:
        self.sm = LandingStateMachine()
        self.pid_x = PID(PIDGains(0.8, 0.02, 0.1))
        self.pid_y = PID(PIDGains(0.8, 0.02, 0.1))
        self.pid_z = PID(PIDGains(0.7, 0.01, 0.08))
        self.pid_yaw = PID(PIDGains(0.5, 0.01, 0.05))
        self.max_vxy = 1.5
        self.max_vz = 0.8
        self.max_yaw_rate = 0.8

    def update(self, target: Dict, current_pose: Dict, dt: float) -> Dict:
        ex = target["center_x"] - current_pose["x"]
        ey = target["center_y"] - current_pose["y"]
        ez = target["center_z"] - current_pose["z"]
        xy_error = sqrt(ex * ex + ey * ey)
        altitude = current_pose["z"]
        state = self.sm.update(
            target["has_valid_zone"],
            target["zone_score"],
            xy_error,
            altitude,
            dt,
            intrusion_detected=bool(target.get("intrusion_detected", False)),
        )

        vx = vy = vz = yaw_rate = 0.0
        if state in (LandingState.APPROACH, LandingState.ALIGN, LandingState.DESCEND):
            vx = self.pid_x.step(ex, dt)
            vy = self.pid_y.step(ey, dt)
        if state == LandingState.DESCEND:
            vz = -0.3
        elif state == LandingState.ABORT:
            vz = 0.5
        elif state in (LandingState.SURVEY, LandingState.SEARCH, LandingState.APPROACH):
            vz = self.pid_z.step(ez, dt)
        if state in (LandingState.SURVEY, LandingState.SEARCH):
            yaw_rate = 0.25
        elif state == LandingState.EVALUATE:
            yaw_rate = 0.0
        else:
            desired_yaw = atan2(ey, ex)
            yaw_rate = self.pid_yaw.step(desired_yaw, dt)

        vx = max(-self.max_vxy, min(self.max_vxy, vx))
        vy = max(-self.max_vxy, min(self.max_vxy, vy))
        vz = max(-self.max_vz, min(self.max_vz, vz))
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        return asdict(VelocityCommand(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate, state=state.value))


if __name__ == "__main__":
    node = ControllerNode()
    current_pose = {"x": 0.0, "y": 0.0, "z": 12.0}
    target = {
        "center_x": 1.0,
        "center_y": 1.0,
        "center_z": 0.0,
        "zone_score": 10.0,
        "has_valid_zone": True,
        "intrusion_detected": False,
    }
    tick = 0
    while True:
        cmd = node.update(target, current_pose, dt=0.1)
        if tick % 5 == 0:
            print(
                {
                    "state": cmd["state"],
                    "vx": round(cmd["vx"], 3),
                    "vy": round(cmd["vy"], 3),
                    "vz": round(cmd["vz"], 3),
                    "yaw_rate": round(cmd["yaw_rate"], 3),
                },
                flush=True,
            )
        tick += 1
        time.sleep(1.0)
