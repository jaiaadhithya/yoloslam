from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SurveyPlanConfig:
    width_m: float = 80.0
    height_m: float = 80.0
    lane_spacing_m: float = 8.0
    altitude_m: float = 18.0
    origin_x: float = 0.0
    origin_y: float = 0.0


def generate_lawnmower_waypoints(config: SurveyPlanConfig) -> List[Tuple[float, float, float]]:
    x0 = config.origin_x - config.width_m / 2.0
    y0 = config.origin_y - config.height_m / 2.0
    lanes = max(2, int(config.height_m / config.lane_spacing_m) + 1)
    waypoints: List[Tuple[float, float, float]] = []
    for lane in range(lanes):
        y = y0 + lane * config.lane_spacing_m
        if lane % 2 == 0:
            waypoints.append((x0, y, config.altitude_m))
            waypoints.append((x0 + config.width_m, y, config.altitude_m))
        else:
            waypoints.append((x0 + config.width_m, y, config.altitude_m))
            waypoints.append((x0, y, config.altitude_m))
    return waypoints
