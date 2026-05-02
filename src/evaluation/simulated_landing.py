import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from landing_controller.controller_node import ControllerNode
from landing_controller.state_machine import LandingState


@dataclass
class SimulatedLandingResult:
    landing_error_m: float
    landing_time_s: float
    success: int
    final_state: str
    num_state_changes: int
    touched_safe_zone: int


def simulate_landing_from_fusion_log(
    fusion_rows: List[Dict],
    gt_zone_xy: tuple[float, float],
    dt: float = 0.2,
) -> tuple[SimulatedLandingResult, List[Dict]]:
    controller = ControllerNode()
    # Start above the scene with a mild offset.
    pose = {"x": 6.0, "y": -6.0, "z": 16.0}
    trajectory: list[Dict] = []
    state_changes = 0
    last_state = LandingState.SURVEY.value
    mission_time = 0.0

    for row in fusion_rows:
        target = {
            "center_x": float(row["zone_x"]),
            "center_y": float(row["zone_y"]),
            "center_z": 0.0,
            "zone_score": float(row["zone_score"]),
            "has_valid_zone": bool(row["has_valid_zone"]),
            "intrusion_detected": bool(row["intrusion_detected"]),
        }
        cmd = controller.update(target=target, current_pose=pose, dt=dt)
        pose["x"] += float(cmd["vx"]) * dt
        pose["y"] += float(cmd["vy"]) * dt
        pose["z"] = max(0.0, pose["z"] + float(cmd["vz"]) * dt)
        mission_time += dt

        if cmd["state"] != last_state:
            state_changes += 1
            last_state = cmd["state"]

        trajectory.append(
            {
                "time_s": mission_time,
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"],
                "vx": cmd["vx"],
                "vy": cmd["vy"],
                "vz": cmd["vz"],
                "state": cmd["state"],
                "transition_reason": cmd.get("transition_reason", ""),
            }
        )
        if cmd["state"] == LandingState.LANDED.value:
            break

    landing_error = float(np.hypot(pose["x"] - gt_zone_xy[0], pose["y"] - gt_zone_xy[1]))
    touched_safe = int(landing_error <= 3.0)
    success = int(last_state == LandingState.LANDED.value and touched_safe == 1)
    result = SimulatedLandingResult(
        landing_error_m=landing_error,
        landing_time_s=mission_time,
        success=success,
        final_state=last_state,
        num_state_changes=state_changes,
        touched_safe_zone=touched_safe,
    )
    return result, trajectory


def write_trajectory(rows: List[Dict], output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_result(result: SimulatedLandingResult, output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        writer.writeheader()
        writer.writerow(asdict(result))

