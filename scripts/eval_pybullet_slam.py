#!/usr/bin/env python3
"""Evaluate integrated SLAM on PyBullet nadir RGB-D vs ground-truth pose."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from pybullet_sim.scene import PyBulletLandingScene, TerrainSceneConfig  # noqa: E402
from pybullet_sim.survey_traj import LawnmowerSurvey, SurveyConfig  # noqa: E402
from slam_module.pybullet_slam_adapter import PyBulletSlamRunner  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _align_sim3_xy(gt: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Sim(3) alignment in XY (scale + rotation + translation)."""
    mu_gt = gt.mean(axis=0)
    mu_est = est.mean(axis=0)
    gt_c = gt - mu_gt
    est_c = est - mu_est
    cov = est_c.T @ gt_c / max(len(gt), 1)
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    var_est = np.sum(est_c * est_c) / max(len(est), 1)
    scale = float(np.trace(np.diag(s) @ r) / max(var_est, 1e-9))
    t = mu_gt - scale * (r @ mu_est)
    return (scale * (est @ r.T)) + t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pybullet_demo.yaml")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--out-dir", default="compre/pybullet_slam_eval")
    args = parser.parse_args()

    cfg = _load_yaml(_REPO / args.config)
    seed = int(cfg.get("seed", 42))
    cam = cfg.get("camera", {})
    W = int(cam.get("width", 640))
    H = int(cam.get("height", 480))
    fov_deg = float(cam.get("fov_deg", 72.0))

    sc = cfg.get("scene", {})
    scene_cfg = TerrainSceneConfig(
        seed=seed,
        terrain_half_extent_m=float(sc.get("terrain_half_extent_m", 20.0)),
        visual_ground_half_extent_m=float(sc.get("visual_ground_half_extent_m", 320.0)),
        safe_clearing_radius_m=float(sc.get("safe_clearing_radius_m", 9.0)),
        n_trees=int(sc.get("n_trees", 18)),
        n_rocks=int(sc.get("n_rocks", 28)),
        water_patch=bool(sc.get("water_patch", True)),
        water_half_size_m=float(sc.get("water_half_size_m", 4.5)),
        add_roads=bool(sc.get("add_roads", True)),
        road_half_width_m=float(sc.get("road_half_width_m", 1.7)),
        tree_trunk_radius_m=float(sc.get("tree_trunk_radius_m", 0.16)),
        tree_trunk_height_m=float(sc.get("tree_trunk_height_m", 1.35)),
        prefer_opengl_render=bool(sc.get("prefer_opengl_render", True)),
        grass_texture_size=int(sc.get("grass_texture_size", 2048)),
        ground_use_tiled_visuals=bool(sc.get("ground_use_tiled_visuals", False)),
        ground_tiles_per_axis=int(sc.get("ground_tiles_per_axis", 14)),
        n_buildings=int(sc.get("n_buildings", 5)),
        n_parked_cars=int(sc.get("n_parked_cars", 10)),
    )
    scene = PyBulletLandingScene.build(scene_cfg, use_gui=False)

    sv_cfg = cfg.get("survey", {})
    survey = LawnmowerSurvey(
        SurveyConfig(
            half_extent_m=float(sv_cfg.get("half_extent_m", 16.0)),
            altitude_m=float(sv_cfg.get("altitude_m", 12.0)),
            speed_m_s=float(sv_cfg.get("speed_m_s", 1.4)),
            stripe_spacing_m=float(sv_cfg.get("stripe_spacing_m", 3.0)),
            trajectory_type=str(sv_cfg.get("trajectory_type", "lawnmower")),
        )
    )

    backend = str((cfg.get("slam") or {}).get("backend", "keyframe_pose_graph_rgbd"))
    slam = PyBulletSlamRunner(backend=backend, camera=cam)

    dt = 1.0 / float(cfg.get("fps", 30))
    x = y = 0.0
    z = float(sv_cfg.get("altitude_m", 12.0))
    yaw = 0.0
    gt_xy: list[list[float]] = []
    est_xy: list[list[float]] = []

    for step in range(args.steps):
        svx, svy = survey.velocity_toward_waypoint(x, y)
        if abs(svx) + abs(svy) > 0.05:
            yaw = float(np.arctan2(svy, svx))
        x += float(svx) * dt
        y += float(svy) * dt
        scene.set_drone_pose(x, y, z, yaw)

        bgr, depth_m = scene.render_nadir_rgbd(x, y, z, yaw, W, H, fov_deg=fov_deg)
        est = slam.track_rgbd(bgr, depth_m, float(step) * dt)
        gt_xy.append([x, y])
        est_xy.append([float(est["x"]), float(est["y"])])

    gt_arr = np.asarray(gt_xy, dtype=np.float64)
    est_arr = np.asarray(est_xy, dtype=np.float64)
    raw_err = np.linalg.norm(est_arr - gt_arr, axis=1)
    aligned = _align_sim3_xy(gt_arr, est_arr)
    aligned_err = np.linalg.norm(aligned - gt_arr, axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(gt_arr[:, 0], gt_arr[:, 1], "g-", linewidth=2, label="Ground truth")
    ax.plot(est_arr[:, 0], est_arr[:, 1], "r--", alpha=0.7, label="SLAM raw")
    ax.plot(aligned[:, 0], aligned[:, 1], "b-.", linewidth=1.5, label="SLAM aligned")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"PyBullet SLAM ({backend})")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_xy.png", dpi=180)
    plt.close(fig)

    summary = {
        "backend": backend,
        "steps": args.steps,
        "raw_mean_xy_error_m": float(np.mean(raw_err)),
        "raw_rmse_xy_error_m": float(np.sqrt(np.mean(np.square(raw_err)))),
        "aligned_mean_xy_error_m": float(np.mean(aligned_err)),
        "aligned_rmse_xy_error_m": float(np.sqrt(np.mean(np.square(aligned_err)))),
        "max_aligned_xy_error_m": float(np.max(aligned_err)),
        "median_aligned_xy_error_m": float(np.median(aligned_err)),
        "plot": str((out_dir / "trajectory_xy.png").resolve()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
