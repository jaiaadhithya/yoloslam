"""
Smoke test: FusionNode with synthetic detections and poses; saves safety grid heatmap.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


def _safe_det(px: float, py: float, conf: float = 0.82) -> dict[str, Any]:
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


def _grid_detections(rng: np.random.Generator) -> list[dict[str, Any]]:
    pts = [
        (280, 200),
        (320, 200),
        (360, 200),
        (280, 240),
        (320, 240),
        (360, 240),
        (280, 280),
        (320, 280),
        (360, 280),
    ]
    return [
        _safe_det(float(px + rng.normal(0, 2.0)), float(py + rng.normal(0, 2.0)), conf=float(0.78 + 0.1 * rng.random()))
        for px, py in pts
    ]


def run_fusion_smoke(out_dir: Path, seed: int = 7, num_steps: int = 120) -> dict[str, Any]:
    from fusion.fusion_node import FusionNode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    fusion = FusionNode()
    last_zone: dict[str, Any] = {}

    t0 = time.perf_counter()
    for step in range(num_steps):
        pose = {
            "x": float(0.5 + 0.02 * step),
            "y": float(-0.3 + 0.15 * np.sin(0.08 * step)),
            "z": float(12.0 - 0.01 * step),
            "state": "OK",
        }
        dets = _grid_detections(rng)
        last_zone = fusion.fuse(dets, pose)
    elapsed = time.perf_counter() - t0

    heat = fusion.grid.safety_map()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat, cmap="RdYlGn", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Fused safety map (smoke test)")
    ax.set_xlabel("cell x")
    ax.set_ylabel("cell y")
    fig.tight_layout()
    grid_path = out_dir / "grid.png"
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)

    return {
        "ok": True,
        "has_valid_zone": bool(last_zone.get("has_valid_zone", False)),
        "zone_score": float(last_zone.get("zone_score", 0.0)),
        "center_xy": (float(last_zone.get("center_x", 0.0)), float(last_zone.get("center_y", 0.0))),
        "runtime_s": elapsed,
        "num_steps": num_steps,
        "grid_png": str(grid_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/smoke_test/fusion", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    print(run_fusion_smoke(args.out_dir, seed=args.seed, num_steps=args.steps))


if __name__ == "__main__":
    main()
