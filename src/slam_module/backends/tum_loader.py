"""Load TUM RGB-D sequences (freiburg1_xyz layout)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TumRgbdSample:
    timestamp: float
    rgb_bgr: np.ndarray
    depth_m: np.ndarray | None


def _read_assoc(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), parts[1]))
    return rows


def _associate_rgb_depth(
    rgb_rows: list[tuple[float, str]],
    depth_rows: list[tuple[float, str]],
    max_delta_s: float = 0.02,
) -> list[tuple[float, str, str]]:
    assoc: list[tuple[float, str, str]] = []
    j = 0
    for t_rgb, rgb_rel in rgb_rows:
        while j + 1 < len(depth_rows) and depth_rows[j + 1][0] <= t_rgb:
            j += 1
        if not depth_rows:
            assoc.append((t_rgb, rgb_rel, ""))
            continue
        t_d, depth_rel = depth_rows[j]
        if j + 1 < len(depth_rows):
            t_next = depth_rows[j + 1][0]
            if abs(t_next - t_rgb) < abs(t_d - t_rgb):
                t_d, depth_rel = depth_rows[j + 1]
        if abs(t_d - t_rgb) <= max_delta_s:
            assoc.append((t_rgb, rgb_rel, depth_rel))
    return assoc


def iter_tum_rgbd_frames(
    dataset_dir: Path,
    *,
    depth_scale: float = 5000.0,
    max_delta_s: float = 0.02,
) -> list[TumRgbdSample]:
    dataset_dir = Path(dataset_dir)
    rgb_rows = _read_assoc(dataset_dir / "rgb.txt")
    depth_rows = _read_assoc(dataset_dir / "depth.txt")
    assoc = _associate_rgb_depth(rgb_rows, depth_rows, max_delta_s=max_delta_s)

    samples: list[TumRgbdSample] = []
    for t_rgb, rgb_rel, depth_rel in assoc:
        rgb_path = dataset_dir / rgb_rel
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        depth_m = None
        if depth_rel:
            depth_raw = cv2.imread(str(dataset_dir / depth_rel), cv2.IMREAD_UNCHANGED)
            if depth_raw is not None:
                depth_m = depth_raw.astype(np.float32) / float(depth_scale)
                depth_m[depth_m <= 0.0] = np.nan
        samples.append(TumRgbdSample(timestamp=t_rgb, rgb_bgr=rgb, depth_m=depth_m))
    return samples


def write_tum_associations(path: Path, dataset_dir: Path, *, max_delta_s: float = 0.02) -> Path:
    """Write ORB-SLAM3-style rgb-depth association file."""
    rgb_rows = _read_assoc(dataset_dir / "rgb.txt")
    depth_rows = _read_assoc(dataset_dir / "depth.txt")
    assoc = _associate_rgb_depth(rgb_rows, depth_rows, max_delta_s=max_delta_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t_rgb, rgb_rel, depth_rel in assoc:
            if not depth_rel:
                continue
            t_depth = next((td for td, dr in depth_rows if dr == depth_rel), t_rgb)
            f.write(f"{t_rgb:.9f} {t_depth:.9f}\n")
    return path


def write_tum_poses(path: Path, timestamps: list[float], poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t, row in zip(timestamps, poses):
            f.write(
                f"{t:.9f} {row[0]:.9f} {row[1]:.9f} {row[2]:.9f} "
                f"{row[3]:.9f} {row[4]:.9f} {row[5]:.9f} {row[6]:.9f}\n"
            )
