from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ZoneCandidate:
    centroid_xy: Tuple[float, float]
    score: float
    area_cells: int


class SafetyGrid:
    def __init__(self, world_size_m: float = 100.0, resolution_m: float = 1.0) -> None:
        self.world_size_m = world_size_m
        self.resolution_m = resolution_m
        self.side_cells = int(world_size_m / resolution_m)
        self.safe_score = np.zeros((self.side_cells, self.side_cells), dtype=float)
        self.unsafe_score = np.zeros_like(self.safe_score)

    def decay(self, factor: float = 0.995) -> None:
        self.safe_score *= factor
        self.unsafe_score *= factor

    def integrate(self, x: float, y: float, confidence: float, is_safe: bool) -> None:
        ix, iy = self.world_to_cell(x, y)
        if ix < 0 or iy < 0:
            return
        delta = max(0.05, confidence)
        if is_safe:
            self.safe_score[iy, ix] += delta
        else:
            self.unsafe_score[iy, ix] += delta

    def integrate_unsafe_footprint(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        per_cell_weight: float,
    ) -> None:
        """Spread unsafe mass across all grid cells overlapping the world XY footprint."""
        if per_cell_weight <= 0:
            return
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        ixs: list[int] = []
        iys: list[int] = []
        for wx, wy in corners:
            ix, iy = self.world_to_cell(wx, wy)
            if ix >= 0 and iy >= 0:
                ixs.append(ix)
                iys.append(iy)
        if not ixs:
            cx = 0.5 * (float(x_min) + float(x_max))
            cy = 0.5 * (float(y_min) + float(y_max))
            ix, iy = self.world_to_cell(cx, cy)
            if ix < 0 or iy < 0:
                return
            self.unsafe_score[iy, ix] += max(0.05, per_cell_weight)
            return
        ix0, ix1 = max(0, min(ixs)), min(self.side_cells - 1, max(ixs))
        iy0, iy1 = max(0, min(iys)), min(self.side_cells - 1, max(iys))
        w = max(0.05, per_cell_weight)
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                self.unsafe_score[iy, ix] += w

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        half = self.world_size_m / 2.0
        ix = int((x + half) / self.resolution_m)
        iy = int((y + half) / self.resolution_m)
        if ix < 0 or iy < 0 or ix >= self.side_cells or iy >= self.side_cells:
            return -1, -1
        return ix, iy

    def cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        half = self.world_size_m / 2.0
        x = (ix + 0.5) * self.resolution_m - half
        y = (iy + 0.5) * self.resolution_m - half
        return x, y

    def safety_map(self) -> np.ndarray:
        return self.safe_score - 1.5 * self.unsafe_score

    def select_best_zone(self, min_cells: int = 6, threshold: float = 1.0) -> ZoneCandidate | None:
        s_map = self.safety_map()
        mask = s_map > threshold
        visited = np.zeros_like(mask, dtype=bool)
        best: ZoneCandidate | None = None

        for y in range(self.side_cells):
            for x in range(self.side_cells):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(x, y)]
                cells: List[Tuple[int, int]] = []
                while stack:
                    cx, cy = stack.pop()
                    if cx < 0 or cy < 0 or cx >= self.side_cells or cy >= self.side_cells:
                        continue
                    if visited[cy, cx] or not mask[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    cells.append((cx, cy))
                    stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

                if len(cells) < min_cells:
                    continue
                score = float(sum(s_map[cy, cx] for cx, cy in cells))
                wx = float(np.mean([self.cell_to_world(cx, cy)[0] for cx, cy in cells]))
                wy = float(np.mean([self.cell_to_world(cx, cy)[1] for cx, cy in cells]))
                cand = ZoneCandidate(centroid_xy=(wx, wy), score=score, area_cells=len(cells))
                if best is None or cand.score > best.score:
                    best = cand
        return best

    def zone_is_contaminated(self, zone_center_xy: Tuple[float, float], radius_m: float = 2.0) -> bool:
        ix, iy = self.world_to_cell(zone_center_xy[0], zone_center_xy[1])
        if ix < 0 or iy < 0:
            return True
        r = max(1, int(radius_m / self.resolution_m))
        y0, y1 = max(0, iy - r), min(self.side_cells, iy + r + 1)
        x0, x1 = max(0, ix - r), min(self.side_cells, ix + r + 1)
        local_unsafe = np.sum(self.unsafe_score[y0:y1, x0:x1])
        local_safe = np.sum(self.safe_score[y0:y1, x0:x1]) + 1e-6
        return bool(local_unsafe > 0.6 * local_safe)

    def snapshot(self) -> Dict:
        return {
            "safe_score_sum": float(np.sum(self.safe_score)),
            "unsafe_score_sum": float(np.sum(self.unsafe_score)),
            "side_cells": self.side_cells,
            "resolution_m": self.resolution_m,
        }
