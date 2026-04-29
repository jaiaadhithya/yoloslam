from dataclasses import asdict, dataclass
from typing import Dict, List
import time

import numpy as np

from fusion.safety_grid import SafetyGrid
from fusion.zone_selector import ZoneSelectionConfig, select_zone_from_grid


@dataclass
class LandingZone:
    center_x: float
    center_y: float
    center_z: float
    zone_score: float
    zone_area_cells: int
    zone_radius: float
    num_observations: int
    has_valid_zone: bool
    grid_snapshot: dict


class FusionNode:
    def __init__(self, fx: float = 320.0, fy: float = 320.0, cx: float = 320.0, cy: float = 240.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.grid = SafetyGrid(world_size_m=100.0, resolution_m=1.0)
        self.zone_selector_config = ZoneSelectionConfig()
        self.selected_zone_xy: tuple[float, float] | None = None

    def fuse(self, detections: List[Dict], slam_pose: Dict, nominal_depth_m: float = 10.0) -> Dict:
        self.grid.decay()
        if slam_pose.get("state") != "OK":
            return asdict(
                LandingZone(
                    center_x=0.0,
                    center_y=0.0,
                    center_z=0.0,
                    zone_score=0.0,
                    zone_area_cells=0,
                    zone_radius=0.0,
                    num_observations=0,
                    has_valid_zone=False,
                    grid_snapshot=self.grid.snapshot(),
                )
            )

        for det in detections:
            wx, wy, _ = self._project_detection(det, slam_pose, nominal_depth_m)
            self.grid.integrate(wx, wy, det["confidence"], det["is_safe"])

        zone = select_zone_from_grid(
            safety_map=self.grid.safety_map(),
            unsafe_map=self.grid.unsafe_score,
            resolution_m=self.grid.resolution_m,
            cell_to_world_fn=self.grid.cell_to_world,
            config=self.zone_selector_config,
        )
        if zone is None:
            return asdict(
                LandingZone(
                    center_x=0.0,
                    center_y=0.0,
                    center_z=0.0,
                    zone_score=0.0,
                    zone_area_cells=0,
                    zone_radius=0.0,
                    num_observations=0,
                    has_valid_zone=False,
                    grid_snapshot=self.grid.snapshot(),
                )
            )
        self.selected_zone_xy = zone.centroid_xy
        return asdict(
            LandingZone(
                center_x=zone.centroid_xy[0],
                center_y=zone.centroid_xy[1],
                center_z=0.0,
                zone_score=zone.zone_score,
                zone_area_cells=zone.num_observations,
                zone_radius=zone.zone_radius_m,
                num_observations=zone.num_observations,
                has_valid_zone=True,
                grid_snapshot=self.grid.snapshot(),
            )
        )

    def zone_is_still_safe(self) -> bool:
        if self.selected_zone_xy is None:
            return False
        return not self.grid.zone_is_contaminated(self.selected_zone_xy, radius_m=2.0)

    def _project_detection(self, det: Dict, pose: Dict, nominal_depth_m: float) -> np.ndarray:
        px = det["x_center"]
        py = det["y_center"]
        # A lightweight scaffold projection. Replace with depth/stereo when available.
        depth = nominal_depth_m
        xc = (px - self.cx) * depth / self.fx
        yc = (py - self.cy) * depth / self.fy
        zc = depth
        return np.array([pose["x"] + xc, pose["y"] + yc, pose["z"] + zc], dtype=float)


if __name__ == "__main__":
    node = FusionNode()
    sample_pose = {"x": 0.0, "y": 0.0, "z": 10.0, "state": "OK"}
    sample_detections = [
        {
            "x_center": 320.0,
            "y_center": 240.0,
            "width": 120.0,
            "height": 120.0,
            "confidence": 0.9,
            "is_safe": True,
        }
    ]
    tick = 0
    while True:
        zone = node.fuse(sample_detections, sample_pose)
        if tick % 5 == 0:
            print(
                {
                    "has_valid_zone": zone["has_valid_zone"],
                    "zone_score": round(zone["zone_score"], 3),
                    "center_x": round(zone["center_x"], 3),
                    "center_y": round(zone["center_y"], 3),
                    "zone_radius": round(zone["zone_radius"], 3),
                },
                flush=True,
            )
        tick += 1
        time.sleep(1.0)
