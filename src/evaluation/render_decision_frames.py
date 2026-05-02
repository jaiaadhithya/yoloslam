"""Render sampled frames: synthetic YOLO view, safety grid, selected zone, state text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _parse_detections_json(cell: str) -> list[dict[str, Any]]:
    if not cell or not str(cell).strip():
        return []
    try:
        data = json.loads(cell)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _draw_detection_panel(width: int, height: int, detections: list[dict[str, Any]]) -> np.ndarray:
    img = np.ones((height, width, 3), dtype=float) * 0.92
    for d in detections:
        cx = float(d.get("x_center", width / 2))
        cy = float(d.get("y_center", height / 2))
        w = float(d.get("width", 80))
        h = float(d.get("height", 80))
        sl = (d.get("safety_label") or "").lower()
        if sl == "unsafe":
            color = (0.85, 0.2, 0.15)
        elif sl == "neutral":
            color = (0.95, 0.88, 0.2)
        elif sl == "positive_safe":
            color = (0.2, 0.72, 0.32)
        elif sl == "unknown":
            color = (0.75, 0.5, 0.9)
        else:
            safe = bool(d.get("is_safe", True))
            color = (0.2, 0.75, 0.35) if safe else (0.85, 0.25, 0.2)
        x0 = int(np.clip(cx - w / 2, 0, width - 1))
        y0 = int(np.clip(cy - h / 2, 0, height - 1))
        x1 = int(np.clip(cx + w / 2, 0, width - 1))
        y1 = int(np.clip(cy + h / 2, 0, height - 1))
        img[y0 : y1 + 1, x0 : x1 + 1, :] = color
    return np.clip(img, 0, 1)


def render_frames_from_artifacts(
    state_log_csv: str | Path,
    safety_npz: str | Path,
    output_dir: str | Path,
    sample_stride: int = 1,
) -> list[Path]:
    """
    Build frame PNGs using state_log.csv and safety_maps.npz (keys: times, maps, world_size_m, resolution_m).
    sample_stride subsamples rows in state_log (npz is expected aligned 1:1 with log rows for simplicity).
    """
    import csv

    log_path = Path(state_log_csv)
    npz_path = Path(safety_npz)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as data:
        maps = data["maps"]
        world_size_m = float(data["world_size_m"]) if "world_size_m" in data.files else 100.0
        res_m = float(data["resolution_m"]) if "resolution_m" in data.files else 1.0

    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = min(len(rows), len(maps))
    written: list[Path] = []
    W, H = 640, 480
    half = world_size_m / 2.0

    for i in range(0, n, sample_stride):
        row = rows[i]
        heat = np.asarray(maps[i])
        t = float(row["time_s"])
        state = row["state"]
        zx = float(row["zone_x"])
        zy = float(row["zone_y"])
        score = float(row["zone_score"])
        dets = _parse_detections_json(row.get("detections_json", ""))
        reason = row.get("transition_reason", "")

        cam = _draw_detection_panel(W, H, dets)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(cam)
        axes[0].set_title("Detections (image space)")
        axes[0].axis("off")
        overlay = axes[0].text(
            8,
            24,
            f"STATE: {state}",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 4},
        )
        overlay.set_zorder(10)

        im = axes[1].imshow(heat, cmap="RdYlGn", vmin=-8, vmax=20, origin="lower")
        ix = int((zx + half) / res_m)
        iy = int((zy + half) / res_m)
        side = heat.shape[0]
        if 0 <= ix < side and 0 <= iy < side:
            axes[1].plot(ix, iy, "c*", markersize=16, markeredgecolor="k", label="selected zone")
        axes[1].set_title("Projected safety map (fusion)")
        axes[1].set_xlabel("grid x")
        axes[1].set_ylabel("grid y")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        title = (
            f"t={t:.2f}s  score={score:.2f}  zone=({zx:.1f},{zy:.1f}) m\n"
            f"transition: {reason or '—'}"
        )
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fname = out_dir / f"frame_{i:05d}.png"
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        written.append(fname)

    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--state-log", default="results/full_pipeline/state_log.csv")
    parser.add_argument("--safety-npz", default="results/full_pipeline/frames/safety_maps.npz")
    parser.add_argument("--output-dir", default="results/full_pipeline/frames")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    paths = render_frames_from_artifacts(
        args.state_log,
        args.safety_npz,
        args.output_dir,
        sample_stride=args.stride,
    )
    print(f"Wrote {len(paths)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
