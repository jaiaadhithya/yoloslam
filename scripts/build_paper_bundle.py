#!/usr/bin/env python3
"""Assemble paper-ready figures and metrics into paper/figures and compre/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_FIG = REPO / "paper" / "figures"
COMPRE = REPO / "compre"
PAPER_RESULTS = REPO / "PAPER_RESULTS"
SLAM_BENCH = COMPRE / "slam_benchmark"

COPY_MAP = {
    SLAM_BENCH / "ate_rmse_bar.png": PAPER_FIG / "slam_ate_bar.png",
    SLAM_BENCH / "yoloslam_rkf_rgbd" / "trajectory_eval.png": PAPER_FIG / "slam_trajectory.png",
    PAPER_RESULTS / "grid_evolution.png": PAPER_FIG / "grid_evolution.png",
    PAPER_RESULTS / "decision_timeline.png": PAPER_FIG / "decision_timeline.png",
    PAPER_RESULTS / "grid_t05.png": PAPER_FIG / "grid_t05.png",
    PAPER_RESULTS / "grid_descent.png": PAPER_FIG / "grid_descent.png",
    PAPER_RESULTS / "pov_02.png": PAPER_FIG / "pov_survey.png",
    PAPER_RESULTS / "pov_04.png": PAPER_FIG / "pov_landed.png",
    PAPER_RESULTS / "abort_chase_030.png": PAPER_FIG / "abort_frame.png",
}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    for src, dst in COPY_MAP.items():
        if not src.is_file():
            print(f"[skip] missing {src}")
            continue
        shutil.copy2(src, dst)
        print(f"[copy] {dst.relative_to(REPO)}")

    slam = _load_json(SLAM_BENCH / "comparison.json")
    landing = _load_json(PAPER_RESULTS / "paper_metrics.json")
    pybullet = _load_json(COMPRE / "pybullet_slam_eval" / "metrics.json")

    figure_paths = {}
    for src, dst in COPY_MAP.items():
        if dst.is_file():
            figure_paths[src.name] = str(dst.relative_to(REPO))

    bundle = {
        "slam_benchmark": {
            "dataset": "TUM RGB-D freiburg1_xyz",
            "max_frames": slam.get("max_frames", 792),
            "best_backend": slam.get("best", {}).get("selected_backend", "keyframe_pose_graph_rgbd"),
            "results": [
                {
                    "backend": r["backend_key"],
                    "ate_rmse_m": (r.get("metrics") or {}).get("ate_rmse_m"),
                    "rpe_rmse_m": (r.get("metrics") or {}).get("rpe_rmse_m"),
                    "status": r.get("status"),
                }
                for r in slam.get("results", [])
                if r.get("status") == "ok"
            ],
        },
        "landing_pipeline": landing.get("summary", {}),
        "pybullet_slam_sanity": pybullet,
        "figure_paths": figure_paths,
    }
    out = COMPRE / "paper_results.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"[write] {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
