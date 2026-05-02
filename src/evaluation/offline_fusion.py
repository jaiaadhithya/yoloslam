import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from fusion.fusion_node import FusionNode


GROUND_TRUTH_SAFE_ZONES: list[tuple[float, float]] = [(-25.0, 25.0), (0.0, -35.0)]


@dataclass
class OfflineFusionConfig:
    condition: str
    num_steps: int = 320
    dt: float = 0.2
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 18.0
    seed: int = 0
    detection_noise_px: float = 5.0
    pose_noise_m: float = 0.12


def _closest_zone_distance(x: float, y: float) -> float:
    return float(min(np.hypot(x - zx, y - zy) for zx, zy in GROUND_TRUTH_SAFE_ZONES))


def _make_safe_detection(px: float, py: float, conf: float = 0.8) -> Dict:
    return {
        "x_center": float(px),
        "y_center": float(py),
        "width": 120.0,
        "height": 120.0,
        "confidence": float(conf),
        "class_id": 8,
        "class_name": "grass_field",
        "is_safe": True,
        "safety_label": "positive_safe",
        "safety_weight": 0.35,
        "fusion_unsafe_delta": 0.0,
    }


def _make_unsafe_detection(px: float, py: float, conf: float = 0.7) -> Dict:
    c = float(conf)
    return {
        "x_center": float(px),
        "y_center": float(py),
        "width": 90.0,
        "height": 90.0,
        "confidence": c,
        "class_id": 1,
        "class_name": "water",
        "is_safe": False,
        "safety_label": "unsafe",
        "safety_weight": -1.0,
        "fusion_unsafe_delta": max(0.05, c),
    }


def _simulate_detections(
    condition: str,
    step: int,
    rng: np.random.Generator,
    noise_px: float,
) -> List[Dict]:
    detections: list[Dict] = []
    n = 2 if condition != "slam_only" else 0
    for _ in range(n):
        detections.append(
            _make_safe_detection(
                px=320 + rng.normal(0, noise_px),
                py=240 + rng.normal(0, noise_px),
                conf=0.75 + 0.2 * rng.random(),
            )
        )
    if step % 8 == 0:
        detections.append(
            _make_unsafe_detection(
                px=460 + rng.normal(0, noise_px * 1.4),
                py=230 + rng.normal(0, noise_px * 1.4),
                conf=0.55 + 0.25 * rng.random(),
            )
        )
    if condition == "yolo_only":
        for det in detections:
            det["confidence"] = max(0.2, det["confidence"] - 0.12)
    return detections


def _simulate_pose(config: OfflineFusionConfig, step: int, rng: np.random.Generator) -> Dict:
    # Sweeping survey trajectory over map.
    t = step * config.dt
    sweep_x = config.start_x + 0.45 * t
    sweep_y = config.start_y + 10.0 * np.sin(0.08 * t)
    z = max(7.0, config.start_z - 0.01 * step)
    noise_scale = config.pose_noise_m
    if config.condition == "yolo_only":
        noise_scale *= 2.4
    elif config.condition == "fused":
        noise_scale *= 0.8
    return {
        "x": float(sweep_x + rng.normal(0, noise_scale)),
        "y": float(sweep_y + rng.normal(0, noise_scale)),
        "z": float(z + rng.normal(0, noise_scale * 0.5)),
        "state": "OK",
    }


def run_offline_fusion(config: OfflineFusionConfig) -> Tuple[List[Dict], tuple[float, float]]:
    rng = np.random.default_rng(config.seed)
    fusion = FusionNode()
    rows: list[Dict] = []
    for step in range(config.num_steps):
        pose = _simulate_pose(config, step, rng)
        detections = _simulate_detections(config.condition, step, rng, config.detection_noise_px)
        zone = fusion.fuse(detections, pose)
        zone["intrusion_detected"] = bool(
            not fusion.zone_is_still_safe() if zone.get("has_valid_zone") else False
        )
        zx = float(zone.get("center_x", 0.0))
        zy = float(zone.get("center_y", 0.0))
        rows.append(
            {
                "step": step,
                "time_s": float(step * config.dt),
                "condition": config.condition,
                "pose_x": pose["x"],
                "pose_y": pose["y"],
                "pose_z": pose["z"],
                "num_detections": len(detections),
                "has_valid_zone": int(bool(zone.get("has_valid_zone", False))),
                "zone_x": zx,
                "zone_y": zy,
                "zone_score": float(zone.get("zone_score", 0.0)),
                "zone_radius_m": float(zone.get("zone_radius", 0.0)),
                "intrusion_detected": int(bool(zone.get("intrusion_detected", False))),
                "distance_to_gt_safe_m": _closest_zone_distance(zx, zy) if zone.get("has_valid_zone") else -1.0,
            }
        )
    valid_rows = [r for r in rows if r["has_valid_zone"] == 1]
    if valid_rows:
        best = max(valid_rows, key=lambda r: r["zone_score"])
        selected = (best["zone_x"], best["zone_y"])
    else:
        selected = (0.0, 0.0)
    return rows, selected


def write_fusion_log(rows: List[Dict], output_csv: str) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

