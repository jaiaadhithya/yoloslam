"""Lawnmower / square survey paths in world XY."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SurveyConfig:
    half_extent_m: float = 16.0
    altitude_m: float = 12.0
    speed_m_s: float = 2.2
    stripe_spacing_m: float = 3.0
    trajectory_type: str = "lawnmower"  # lawnmower | spiral


class LawnmowerSurvey:
    def __init__(self, cfg: SurveyConfig) -> None:
        self.cfg = cfg
        self._waypoints: list[tuple[float, float]] = []
        self._idx = 0
        self._rebuild()

    def _rebuild(self) -> None:
        h = self.cfg.half_extent_m
        spacing = max(0.8, self.cfg.stripe_spacing_m)
        if self.cfg.trajectory_type == "spiral":
            self._waypoints = _spiral_waypoints(h, spacing)
        else:
            self._waypoints = _lawnmower_waypoints(h, spacing)
        self._idx = 0

    def reset(self) -> None:
        self._rebuild()

    def velocity_toward_waypoint(self, x: float, y: float) -> tuple[float, float]:
        if not self._waypoints:
            return 0.0, 0.0
        tx, ty = self._waypoints[self._idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist < 0.35:
            self._idx = (self._idx + 1) % len(self._waypoints)
            tx, ty = self._waypoints[self._idx]
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0
        s = self.cfg.speed_m_s / dist
        return dx * s, dy * s


def _lawnmower_waypoints(h: float, spacing: float) -> list[tuple[float, float]]:
    ys = []
    y = -h + 1.0
    while y <= h - 1.0:
        ys.append(y)
        y += spacing
    wps: list[tuple[float, float]] = []
    forward = True
    for y in ys:
        if forward:
            wps.append((-h + 1.0, y))
            wps.append((h - 1.0, y))
        else:
            wps.append((h - 1.0, y))
            wps.append((-h + 1.0, y))
        forward = not forward
    return wps


def _spiral_waypoints(h: float, spacing: float) -> list[tuple[float, float]]:
    """Archimedean-ish square spiral inside [-h,h]."""
    wps: list[tuple[float, float]] = []
    x = y = 0.0
    direction = 0  # 0=E,1=N,2=W,3=S
    step = spacing
    limit = 1.0
    wps.append((x, y))
    max_pts = 800
    while len(wps) < max_pts and limit < 2.0 * h:
        for _ in range(2):
            for _ in range(int(max(1, round(limit / spacing)))):
                if direction == 0:
                    x += spacing
                elif direction == 1:
                    y += spacing
                elif direction == 2:
                    x -= spacing
                else:
                    y -= spacing
                x = max(-h + 1.0, min(h - 1.0, x))
                y = max(-h + 1.0, min(h - 1.0, y))
                wps.append((x, y))
            direction = (direction + 1) % 4
        limit += spacing
    return wps
