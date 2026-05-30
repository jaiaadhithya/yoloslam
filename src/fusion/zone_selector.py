from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ZoneSelectionConfig:
    min_zone_size_m: float = 3.0
    safety_threshold: float = 1.0
    unsafe_buffer_m: float = 1.5


@dataclass
class SelectedZone:
    centroid_xy: Tuple[float, float]
    zone_radius_m: float
    zone_score: float
    num_observations: int


def select_zone_from_grid(
    safety_map: np.ndarray,
    unsafe_map: np.ndarray,
    resolution_m: float,
    cell_to_world_fn,
    config: ZoneSelectionConfig,
) -> SelectedZone | None:
    min_cells = max(1, int((config.min_zone_size_m / resolution_m) ** 2))
    mask = safety_map > config.safety_threshold
    visited = np.zeros_like(mask, dtype=bool)
    best: SelectedZone | None = None

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if visited[y, x] or not mask[y, x]:
                continue
            cells = []
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cy < 0 or cx >= mask.shape[1] or cy >= mask.shape[0]:
                    continue
                if visited[cy, cx] or not mask[cy, cx]:
                    continue
                visited[cy, cx] = True
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

            if len(cells) < min_cells:
                continue

            # Allow zones that graze diagonal-only unsafe; still forbid cardinal neighbors of unsafe cells.
            h_um, w_um = unsafe_map.shape

            def _cardinal_touch_unsafe(ix: int, iy: int) -> bool:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < w_um and 0 <= ny < h_um and unsafe_map[ny, nx] > 0.01:
                        return True
                return False

            unsafe_hits = sum(1 for cx, cy in cells if _cardinal_touch_unsafe(cx, cy))
            if unsafe_hits > 0:
                continue

            zone_score = float(sum(safety_map[cy, cx] for cx, cy in cells))
            centroid_x = float(np.mean([cell_to_world_fn(cx, cy)[0] for cx, cy in cells]))
            centroid_y = float(np.mean([cell_to_world_fn(cx, cy)[1] for cx, cy in cells]))
            radius_m = float(np.sqrt(len(cells) / np.pi) * resolution_m)
            candidate = SelectedZone((centroid_x, centroid_y), radius_m, zone_score, len(cells))
            if best is None or candidate.zone_score > best.zone_score:
                best = candidate

    return best
