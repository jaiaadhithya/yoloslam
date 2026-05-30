#!/usr/bin/env python3
"""
Offline PyBullet demo: YOLO/color vision + fusion + landing controller -> MP4.

Usage (from repo root):
  python scripts/run_pybullet_demo.py --config config/pybullet_demo.yaml

Requires a working `pybullet` install (prebuilt wheels on many Linux/macOS targets;
on Windows you may need MSVC build tools or a Python version with a published wheel).

Playback: OpenCV writes MPEG-4 Part 2 (`mp4v`) first; with `ffmpeg` on PATH the script
re-encodes to H.264 yuv420p (see `config/pybullet_demo.yaml` → `video`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

# Harmless when multiple matplotlib installs exist (common on Debian/Ubuntu + pip).
warnings.filterwarnings(
    "ignore",
    message=r"Unable to import Axes3D.*",
    category=UserWarning,
)

import cv2
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from fusion.fusion_node import FusionNode  # noqa: E402
from landing_controller.controller_node import ControllerNode  # noqa: E402
from landing_controller.state_machine import LandingSMConfig, LandingState  # noqa: E402
from pybullet_sim.overlay import annotate_chase_frame, composite_frame  # noqa: E402
from pybullet_sim.pose_provider import PoseProvider, PoseProviderConfig  # noqa: E402
from pybullet_sim.scene import PyBulletLandingScene, TerrainSceneConfig  # noqa: E402
from pybullet_sim.sim_vision import DemoDetector  # noqa: E402
from pybullet_sim.slam_vo import VisualOdometryConfig, VisualOdometrySlam  # noqa: E402
from pybullet_sim.survey_traj import LawnmowerSurvey, SurveyConfig  # noqa: E402
from slam_module.pybullet_slam_adapter import PyBulletSlamRunner  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_repo_path(p: str | None) -> str | None:
    if not p:
        return p
    pp = Path(p)
    if pp.is_file():
        return str(pp)
    cand = _REPO_ROOT / p
    return str(cand) if cand.is_file() else str(pp)


def _ffmpeg_reencode_h264(path: Path, *, fps: float, crf: int, preset: str) -> bool:
    """Replace file with CFR H.264 yuv420p + faststart (fixes many player glitches from OpenCV mp4v)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    tmp = path.with_suffix(path.suffix + ".h264tmp")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        err = (proc.stderr or proc.stdout or "").strip()
        if err:
            snippet = err[:480] + ("…" if len(err) > 480 else "")
            print(f"[warn] ffmpeg re-encode failed for {path.name}: {snippet}", file=sys.stderr)
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    tmp.replace(path)
    return True


def _compute_grid_convergence_time_s(safe_hist: list[float], fps: float) -> float | None:
    """First time (s) when |d(sum)/dt|/max(|sum|,1) stays under 0.01 (1%/s) on average over 0.5s."""
    if len(safe_hist) < max(30, int(2 * fps)):
        return None
    s = np.asarray(safe_hist, dtype=float)
    dt = 1.0 / fps
    ds_dt = np.diff(s) / dt
    denom = np.maximum(np.abs(s[1:]), 1.0)
    rel = np.abs(ds_dt) / denom
    win = max(3, int(0.5 * fps))
    if len(rel) < win:
        return None
    conv = np.convolve(rel, np.ones(win, dtype=float) / float(win), mode="valid")
    for i, val in enumerate(conv):
        if val < 0.01:
            return float(i + win) * dt
    return None


def _save_fusion_grid_png(fusion: FusionNode, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm = fusion.grid.safety_map()
    half = fusion.grid.world_size_m / 2.0
    extent = (-half, half, -half, half)
    vmax = max(4.0, float(np.nanmax(np.abs(sm)) + 1e-6))
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(
        sm,
        origin="lower",
        extent=extent,
        cmap="RdYlGn",
        aspect="equal",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Safety (safe − 1.5×unsafe)")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="PyBullet offline landing demo video")
    ap.add_argument("--config", type=str, default=str(_REPO_ROOT / "config" / "pybullet_demo.yaml"))
    ap.add_argument("--output", type=str, default=None, help="Override MP4 path from config")
    ap.add_argument("--gui", action="store_true", help="PyBullet GUI (slower)")
    ap.add_argument(
        "--state-log-csv",
        type=str,
        default=None,
        help="Append per-timestep FSM log for plot_decision_timeline.py",
    )
    ap.add_argument(
        "--grid-snapshots-dir",
        type=str,
        default=None,
        help="Directory for grid_t05.png, grid_t15.png, grid_t25.png, grid_descent.png",
    )
    ap.add_argument(
        "--headless-metrics-json",
        type=str,
        default=None,
        help="Skip MP4 encoding; run simulation only and write metrics JSON (for batch trials)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _REPO_ROOT / args.config
    cfg = _load_yaml(cfg_path)
    seed = int(cfg.get("seed", 42))
    fps = float(cfg.get("fps", 30))
    dt = 1.0 / max(1.0, fps)
    max_steps = int(float(cfg.get("max_duration_s", 90)) / dt)

    out_drone_name = args.output or cfg.get("output_video_drone") or cfg.get("output_video", "demo_landing_drone.mp4")
    out_drone = Path(out_drone_name)
    if not out_drone.is_absolute():
        out_drone = _REPO_ROOT / out_drone_name
    out_drone.parent.mkdir(parents=True, exist_ok=True)

    if args.output:
        p = Path(args.output)
        chase_path = p.parent / f"{p.stem}_chase{p.suffix}"
        out_chase = chase_path if chase_path.is_absolute() else _REPO_ROOT / chase_path
    else:
        out_chase_name = cfg.get("output_video_chase")
        if not out_chase_name:
            out_chase_name = out_drone.with_name(out_drone.stem + "_chase" + out_drone.suffix).name
        out_chase = Path(out_chase_name)
        if not out_chase.is_absolute():
            out_chase = _REPO_ROOT / out_chase_name
    out_chase.parent.mkdir(parents=True, exist_ok=True)

    cam = cfg.get("camera", {})
    W = int(cam.get("width", 640))
    H = int(cam.get("height", 480))
    fov_deg = float(cam.get("fov_deg", 90.0))
    fx = float(cam.get("fx", 320.0))
    fy = float(cam.get("fy", 320.0))
    cx = float(cam.get("cx", 320.0))
    cy = float(cam.get("cy", 240.0))

    sc = cfg.get("scene", {})
    terrain_cfg = TerrainSceneConfig(
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

    use_gui = bool(args.gui or cfg.get("use_gui", False))
    scene = PyBulletLandingScene.build(terrain_cfg, use_gui=use_gui)

    sv = cfg.get("survey", {})
    survey = LawnmowerSurvey(
        SurveyConfig(
            half_extent_m=float(sv.get("half_extent_m", 16.0)),
            altitude_m=float(sv.get("altitude_m", 12.0)),
            speed_m_s=float(sv.get("speed_m_s", 2.2)),
            stripe_spacing_m=float(sv.get("stripe_spacing_m", 3.0)),
            trajectory_type=str(sv.get("trajectory_type", "lawnmower")),
        )
    )

    pose_cfg = cfg.get("pose", {})
    pose_provider = PoseProvider(
        PoseProviderConfig(
            use_noisy_vo=bool(pose_cfg.get("use_noisy_vo", False)),
            pos_sigma_m=float(pose_cfg.get("pos_sigma_m", 0.08)),
            yaw_sigma_rad=float(pose_cfg.get("yaw_sigma_rad", 0.03)),
            z_sigma_m=float(pose_cfg.get("z_sigma_m", 0.05)),
            seed=seed,
        )
    )

    slam_cfg = cfg.get("slam", {})
    slam_mode = str(slam_cfg.get("mode", "visual_odometry")).lower()
    vo_cfg = VisualOdometryConfig(
        vel_noise_std_m_s=float(slam_cfg.get("vo_vel_noise_std_m_s", 0.035)),
        yaw_noise_std_rad=float(slam_cfg.get("vo_yaw_noise_std_rad", 0.012)),
        seed=int(slam_cfg.get("vo_seed", seed)),
    )
    vo = VisualOdometrySlam(vo_cfg)
    control_uses_slam = bool(slam_cfg.get("controller_uses_slam_estimate", True))
    rgbd_slam: PyBulletSlamRunner | None = None
    if slam_mode == "rgbd_slam":
        rgbd_slam = PyBulletSlamRunner(
            backend=str(slam_cfg.get("backend", "keyframe_pose_graph_rgbd")),
            camera=cam,
        )
        slam_mode = f"rgbd_slam({rgbd_slam.backend_name})"

    det_cfg = cfg.get("detector", {})
    safety_yaml = _resolve_repo_path(det_cfg.get("safety_mapping_yaml"))
    det_mode = str(det_cfg.get("mode", "scene_semantic")).lower()
    use_truth_geom = bool(det_cfg.get("use_truth_pose_for_geometry", True))
    detector = DemoDetector(
        mode=det_mode,
        model_path=str(det_cfg.get("model_path", "yolov8n.pt")),
        safety_yaml=safety_yaml,
    )

    fus = cfg.get("fusion", {})
    fusion = FusionNode(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        world_size_m=float(fus.get("world_size_m", 100.0)),
        zone_safety_threshold=float(fus.get("zone_safety_threshold", 1.0)),
        zone_min_size_m=float(fus.get("zone_min_size_m", 3.0)),
    )

    cc = cfg.get("controller", {})
    sm_cfg = LandingSMConfig(
        survey_duration_s=float(cc.get("survey_duration_s", 7.5)),
        zone_score_evaluate_commit=float(cc.get("zone_score_evaluate_commit", 4.0)),
        zone_score_search_commit=float(cc.get("zone_score_search_commit", 2.0)),
        approach_xy_tolerance_m=float(cc.get("approach_xy_tolerance_m", 1.0)),
        align_xy_tolerance_m=float(cc.get("align_xy_tolerance_m", 0.2)),
        align_stable_s=float(cc.get("align_stable_s", 1.2)),
        descend_xy_drift_m=float(cc.get("descend_xy_drift_m", 0.6)),
        touchdown_altitude_m=float(cc.get("touchdown_altitude_m", 0.25)),
        target_lost_timeout_s=float(cc.get("target_lost_timeout_s", 4.0)),
        abort_resurvey_s=float(cc.get("abort_resurvey_s", 2.0)),
    )
    controller = ControllerNode(sm_cfg)

    abort_cfg = cfg.get("abort_demo", {})
    abort_enabled = bool(abort_cfg.get("enabled", True))
    inject_after = float(abort_cfg.get("inject_after_descend_s", 1.1))

    chase_cfg = cfg.get("chase_camera", {})
    chase_dist = float(chase_cfg.get("distance_m", 14.0))
    chase_h = float(chase_cfg.get("height_above_drone_m", 5.0))
    chase_fov = float(chase_cfg.get("fov_deg", 58.0))
    chase_target_z_frac = float(chase_cfg.get("target_height_frac", 0.88))

    vid_cfg = cfg.get("video", {})
    reencode_h264 = bool(vid_cfg.get("reencode_h264", True))
    ffmpeg_crf = int(vid_cfg.get("ffmpeg_crf", 23))
    ffmpeg_preset = str(vid_cfg.get("ffmpeg_preset", "medium"))
    headless = bool(args.headless_metrics_json)
    if reencode_h264 and not shutil.which("ffmpeg") and not headless:
        print(
            "[demo] ffmpeg not on PATH — OpenCV mp4v can stutter in some players; "
            "install ffmpeg or open the MP4 in VLC.",
            flush=True,
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_drone: cv2.VideoWriter | None = None
    writer_chase: cv2.VideoWriter | None = None
    if not headless:
        writer_drone = cv2.VideoWriter(str(out_drone), fourcc, fps, (W, H))
        writer_chase = cv2.VideoWriter(str(out_chase), fourcc, fps, (W, H))
        if not writer_drone.isOpened():
            raise SystemExit(f"Could not open VideoWriter for {out_drone}")
        if not writer_chase.isOpened():
            raise SystemExit(f"Could not open VideoWriter for {out_chase}")

    progress_every = max(1, int(round(fps * 3.0)))
    print(
        f"[demo] Rendering up to {max_steps} frames (~{max_steps / fps:.0f} s @ {fps:.0f} fps)",
        flush=True,
    )
    if not headless:
        print(f"[demo] Drone POV: {out_drone}", flush=True)
        print(f"[demo] Chase POV: {out_chase}", flush=True)
    if headless:
        print("[demo] Headless metrics run (no MP4)", flush=True)
    print(
        "[demo] First lines may show PyBullet + Matplotlib warnings; then progress every ~3 s.",
        flush=True,
    )

    state_log_path = Path(args.state_log_csv) if args.state_log_csv else None
    state_log_fp = None
    state_writer: csv.DictWriter | None = None
    state_rows: list[dict] = []
    if state_log_path is not None:
        state_log_path.parent.mkdir(parents=True, exist_ok=True)
        state_log_fp = state_log_path.open("w", newline="", encoding="utf-8")
        state_writer = csv.DictWriter(
            state_log_fp, fieldnames=["time_s", "state", "altitude_m", "zone_score"]
        )
        state_writer.writeheader()

    snap_dir = Path(args.grid_snapshots_dir) if args.grid_snapshots_dir else None
    snap_done = {"t05": False, "t15": False, "t25": False, "descent": False}
    inject_time_s: float | None = None
    abort_time_s: float | None = None
    land_time_s: float | None = None
    zone_at_land: tuple[float, float] | None = None

    x, y, z, yaw = 0.0, 0.0, float(sv.get("altitude_m", 12.0)), 0.0
    scene.set_drone_pose(x, y, z, yaw)
    vo.reset(x, y, z, yaw)

    safe_hist: list[float] = []
    descend_timer = 0.0
    intrusion_fired = False
    abort_epilogue_s = float(abort_cfg.get("end_video_after_abort_s", 6.0))
    abort_epilogue_remaining: float | None = None
    last_valid_zone: dict | None = None
    traj_xyz: list[tuple[float, float, float]] = []
    max_traj_pts = int(cfg.get("trajectory", {}).get("max_points", 900))

    det_label = {"yolo": "YOLO+YAML", "color": "color+YAML", "scene_semantic": "scene-semantic", "hybrid": "YOLO+scene"}.get(det_mode, det_mode)
    vx_dyn = 0.0
    vy_dyn = 0.0
    vz_dyn = 0.0
    yaw_rate_dyn = 0.0
    accel_cfg = cfg.get("dynamics", {})
    a_xy = float(accel_cfg.get("max_accel_xy_m_s2", 1.2))
    a_z = float(accel_cfg.get("max_accel_z_m_s2", 0.9))
    a_yaw = float(accel_cfg.get("max_yaw_accel_rad_s2", 1.6))

    try:
        for step in range(max_steps):
            traj_xyz.append((float(x), float(y), float(z)))
            if len(traj_xyz) > max_traj_pts:
                traj_xyz = traj_xyz[-max_traj_pts:]

            sm_before = controller.sm.state
            if sm_before == LandingState.DESCEND:
                descend_timer += dt
            else:
                descend_timer = 0.0

            # SLAM / fusion pose (not ground truth): RGB-D SLAM, VO, noisy truth, or pure truth.
            if rgbd_slam is not None:
                rgb, depth_m = scene.render_nadir_rgbd(x, y, z, yaw, W, H, fov_deg=fov_deg)
            else:
                rgb = scene.render_nadir_rgb(x, y, z, yaw, W, H, fov_deg=fov_deg)
                depth_m = None

            if rgbd_slam is not None:
                slam_pose = rgbd_slam.track_rgbd(rgb, depth_m, float(step) * dt)
            elif slam_mode.startswith("ground_truth"):
                slam_pose = {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw), "state": "OK"}
            elif slam_mode.startswith("noisy_truth"):
                slam_pose = pose_provider.from_truth(x, y, z, yaw)
                slam_pose["z"] = float(slam_pose.get("z", z))
            else:
                slam_pose = vo.as_dict()

            chase_bgr = None
            if not headless:
                chase_bgr = scene.render_chase_rgb(
                    x,
                    y,
                    z,
                    yaw,
                    W,
                    H,
                    fov_deg=chase_fov,
                    distance_m=chase_dist,
                    height_above_drone_m=chase_h,
                    target_height_frac=chase_target_z_frac,
                )

            truth_pose = {"x": x, "y": y, "z": z, "yaw": yaw}
            ctrl_pose = slam_pose if control_uses_slam else truth_pose
            geom_pose = dict(truth_pose)
            geom_pose["state"] = "OK"
            if rgbd_slam is not None:
                geom_pose = dict(slam_pose)
                geom_pose["state"] = slam_pose.get("state", "OK")
            elif not (use_truth_geom and det_mode == "scene_semantic"):
                geom_pose = dict(slam_pose)
                geom_pose["state"] = slam_pose.get("state", "OK")

            dets = detector.detect(
                rgb,
                scene=scene,
                pose=geom_pose,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            zone = fusion.fuse(dets, geom_pose, nominal_depth_m=max(1.0, float(geom_pose["z"])))
            if bool(zone.get("has_valid_zone")):
                last_valid_zone = dict(zone)

            target = {
                "center_x": float(zone["center_x"]),
                "center_y": float(zone["center_y"]),
                "center_z": float(zone["center_z"]),
                "zone_score": float(zone["zone_score"]),
                "has_valid_zone": bool(zone["has_valid_zone"]),
                "intrusion_detected": False,
            }

            contaminated = False
            if fusion.selected_zone_xy is not None:
                contaminated = bool(fusion.grid.zone_is_contaminated(fusion.selected_zone_xy, radius_m=2.0))

            survey_alt = float(sv.get("altitude_m", 12.0))
            commit_thr = float(controller.sm.cfg.zone_score_evaluate_commit)
            if sm_before in (LandingState.SURVEY, LandingState.SEARCH):
                target["center_z"] = survey_alt
            elif sm_before == LandingState.EVALUATE:
                if not (target["has_valid_zone"] and target["zone_score"] > commit_thr):
                    target["center_z"] = survey_alt

            inject_now = (
                abort_enabled
                and not intrusion_fired
                and sm_before == LandingState.DESCEND
                and descend_timer >= inject_after
            )
            if inject_now:
                intrusion_fired = True
                if inject_time_s is None:
                    inject_time_s = float(step + 1) * dt
            target["intrusion_detected"] = bool(intrusion_fired or contaminated)

            # Keep an anchored target while approaching to avoid reverting to endless SEARCH
            # when detections momentarily drop from frame edges.
            if (
                not target["has_valid_zone"]
                and last_valid_zone is not None
                and sm_before in (LandingState.APPROACH, LandingState.ALIGN, LandingState.DESCEND)
            ):
                target["center_x"] = float(last_valid_zone["center_x"])
                target["center_y"] = float(last_valid_zone["center_y"])
                target["center_z"] = float(last_valid_zone["center_z"])
                target["zone_score"] = max(0.6, 0.9 * float(last_valid_zone.get("zone_score", 0.6)))
                target["has_valid_zone"] = True

            cmd = controller.update(target, ctrl_pose, dt)
            sm_after = controller.sm.state
            if sm_after == LandingState.ABORT and abort_time_s is None:
                abort_time_s = float(step + 1) * dt

            fusion_live = bool(zone.get("has_valid_zone"))
            zone_display = dict(zone)
            if target["has_valid_zone"]:
                zone_display["has_valid_zone"] = True
                zone_display["center_x"] = float(target["center_x"])
                zone_display["center_y"] = float(target["center_y"])
                zone_display["center_z"] = float(target["center_z"])
                zone_display["zone_score"] = float(target["zone_score"])
                if zone.get("has_valid_zone") and float(zone.get("zone_radius", 0.0)) > 0.05:
                    zone_display["zone_radius"] = float(zone["zone_radius"])
                elif last_valid_zone is not None:
                    zone_display["zone_radius"] = float(last_valid_zone.get("zone_radius", 3.5))
                else:
                    zone_display["zone_radius"] = 3.5

            if sm_before == LandingState.ABORT and sm_after == LandingState.SURVEY:
                intrusion_fired = False

            vx = float(cmd["vx"])
            vy = float(cmd["vy"])
            vz = float(cmd["vz"])
            yaw_rate = float(cmd["yaw_rate"])

            if sm_after in (LandingState.SURVEY, LandingState.EVALUATE, LandingState.SEARCH):
                svx, svy = survey.velocity_toward_waypoint(x, y)
                vx, vy = float(svx), float(svy)
                if abs(svx) + abs(svy) > 0.05:
                    yaw = float(np.arctan2(svy, svx))
                yaw_rate = 0.0
            elif sm_after in (LandingState.APPROACH, LandingState.ALIGN, LandingState.DESCEND):
                ex = float(target["center_x"]) - x
                ey = float(target["center_y"]) - y
                if abs(ex) + abs(ey) > 0.08:
                    yaw = float(np.arctan2(ey, ex))
                yaw_rate = 0.0

            # Smooth kinematic response to avoid abrupt panning/flicker.
            dvx = float(np.clip(vx - vx_dyn, -a_xy * dt, a_xy * dt))
            dvy = float(np.clip(vy - vy_dyn, -a_xy * dt, a_xy * dt))
            dvz = float(np.clip(vz - vz_dyn, -a_z * dt, a_z * dt))
            dyr = float(np.clip(yaw_rate - yaw_rate_dyn, -a_yaw * dt, a_yaw * dt))
            vx_dyn += dvx
            vy_dyn += dvy
            vz_dyn += dvz
            yaw_rate_dyn += dyr

            x += vx_dyn * dt
            y += vy_dyn * dt
            z += vz_dyn * dt
            yaw += yaw_rate_dyn * dt
            yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))

            z = max(0.05, z)
            half = terrain_cfg.terrain_half_extent_m - 0.6
            x = float(np.clip(x, -half, half))
            y = float(np.clip(y, -half, half))

            scene.set_drone_pose(x, y, z, yaw)

            if slam_mode == "visual_odometry" or slam_mode.startswith("visual_odometry"):
                vo.propagate(dt, vx_dyn, vy_dyn, vz_dyn, yaw)

            safe_hist.append(float(np.sum(fusion.grid.safe_score)))

            subtitle = []
            if inject_now or (intrusion_fired and sm_after == LandingState.ABORT):
                subtitle.append("Intrusion / ABORT (demo)")
            slam_line = f"SLAM: {slam_mode} | Fusion: SafetyGrid | Det: {det_label}"
            pipe_lines = [slam_line, f"Control pose: {'SLAM est.' if control_uses_slam else 'ground truth'}"]

            t_mission = float(step + 1) * dt
            row = {
                "time_s": round(t_mission, 4),
                "state": cmd["state"],
                "altitude_m": round(float(z), 4),
                "zone_score": round(float(zone_display.get("zone_score", 0.0)), 4),
            }
            state_rows.append(dict(row))
            if state_writer is not None:
                state_writer.writerow(row)

            if snap_dir is not None:
                if not snap_done["t05"] and t_mission >= 5.0:
                    snap_done["t05"] = True
                    _save_fusion_grid_png(fusion, snap_dir / "grid_t05.png", "Safety grid (~5 s)")
                if not snap_done["t15"] and t_mission >= 15.0:
                    snap_done["t15"] = True
                    _save_fusion_grid_png(fusion, snap_dir / "grid_t15.png", "Safety grid (~15 s)")
                if not snap_done["t25"] and t_mission >= 25.0:
                    snap_done["t25"] = True
                    _save_fusion_grid_png(fusion, snap_dir / "grid_t25.png", "Safety grid (~25 s)")
                if not snap_done["descent"] and sm_after == LandingState.DESCEND:
                    snap_done["descent"] = True
                    _save_fusion_grid_png(fusion, snap_dir / "grid_descent.png", "Safety grid (descent)")

            if not headless:
                frame_drone = composite_frame(
                    rgb,
                    dets,
                    fusion,
                    zone_display,
                    slam_pose,
                    cmd["state"],
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    frame_index=step,
                    zone_project_pose=geom_pose,
                    subtitle_lines=subtitle,
                    pipeline_lines=pipe_lines,
                    fusion_live=fusion_live,
                )
                assert writer_drone is not None
                writer_drone.write(frame_drone)

                chase_extra = [
                    slam_line,
                    f"Fusion zone valid={zone.get('has_valid_zone')} score={zone.get('zone_score', 0.0):.2f}",
                ]
                assert chase_bgr is not None and writer_chase is not None
                frame_chase = annotate_chase_frame(
                    chase_bgr,
                    controller_state=cmd["state"],
                    truth_xyz=(x, y, z),
                    truth_yaw=yaw,
                    n_detections=len(dets),
                    scene=scene,
                    trajectory_xyz=traj_xyz,
                    chase_fov_deg=chase_fov,
                    chase_distance_m=chase_dist,
                    chase_height_m=chase_h,
                    chase_target_z_frac=chase_target_z_frac,
                    extra_lines=chase_extra,
                    landing_zone=zone_display if zone_display.get("has_valid_zone") else None,
                    frame_index=step,
                )
                writer_chase.write(frame_chase)
            else:
                frame_drone = None
                frame_chase = None

            if step % progress_every == 0:
                print(
                    f"[demo] frame {step}/{max_steps}  state={cmd['state']}  z={z:.2f}m  slam=({slam_pose['x']:.1f},{slam_pose['y']:.1f})",
                    flush=True,
                )

            if sm_after == LandingState.ABORT and abort_epilogue_remaining is None:
                abort_epilogue_remaining = abort_epilogue_s

            if abort_epilogue_remaining is not None:
                abort_epilogue_remaining -= dt
                if abort_epilogue_remaining <= 0:
                    if not headless and frame_drone is not None and frame_chase is not None:
                        assert writer_drone is not None and writer_chase is not None
                        for _ in range(int(1.0 * fps)):
                            writer_drone.write(frame_drone)
                            writer_chase.write(frame_chase)
                    break

            if cmd["state"] == LandingState.LANDED:
                if land_time_s is None:
                    land_time_s = t_mission
                    if zone_display.get("has_valid_zone"):
                        zone_at_land = (
                            float(zone_display["center_x"]),
                            float(zone_display["center_y"]),
                        )
                if not headless and frame_drone is not None and frame_chase is not None:
                    assert writer_drone is not None and writer_chase is not None
                    for _ in range(int(2.0 * fps)):
                        writer_drone.write(frame_drone)
                        writer_chase.write(frame_chase)
                break
    finally:
        if state_log_fp is not None:
            state_log_fp.close()
        if writer_drone is not None:
            writer_drone.release()
        if writer_chase is not None:
            writer_chase.release()
        scene.disconnect()

    if reencode_h264 and not headless:
        for label, outp in (("drone", out_drone), ("chase", out_chase)):
            if _ffmpeg_reencode_h264(
                outp,
                fps=fps,
                crf=ffmpeg_crf,
                preset=ffmpeg_preset,
            ):
                print(f"[demo] Re-encoded {label} video for playback (H.264): {outp}", flush=True)
            else:
                print(
                    f"[demo] Kept OpenCV-encoded file for {label} (install ffmpeg for smooth H.264). "
                    f"Or open in VLC: {outp}",
                    flush=True,
                )

    art = cfg.get("artifacts", {})
    if bool(art.get("save_grid_evolution_png", False)):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_path = art.get("grid_plot_path", "results/pybullet_demo/grid_evolution.png")
            pp = Path(plot_path)
            if not pp.is_absolute():
                pp = _REPO_ROOT / pp
            pp.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(9, 3))
            plt.plot(np.arange(len(safe_hist)) / fps, safe_hist, color="#2a9d8f", lw=1.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Sum of safe_score")
            plt.title("Safety grid evolution (safe evidence accumulation)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(pp, dpi=150)
            plt.close()
        except Exception as exc:
            print(f"[warn] grid plot skipped: {exc}", file=sys.stderr)

    if headless and args.headless_metrics_json:
        clearing_r = float(sc.get("safe_clearing_radius_m", 9.0))
        thr = float(cfg.get("fusion", {}).get("zone_safety_threshold", 0.42))
        zone_accuracy: bool | None = None
        if zone_at_land is not None:
            smap = fusion.grid.safety_map()
            ix, iy = fusion.grid.world_to_cell(float(zone_at_land[0]), float(zone_at_land[1]))
            if 0 <= ix < smap.shape[1] and 0 <= iy < smap.shape[0]:
                zone_accuracy = bool(float(smap[iy, ix]) > thr)
        abort_detection_latency_s: float | None = None
        if inject_time_s is not None and abort_time_s is not None:
            abort_detection_latency_s = max(0.0, float(abort_time_s - inject_time_s))
        states_visited: list[str] = []
        seen_s: set[str] = set()
        for r in state_rows:
            st = str(r["state"])
            if st not in seen_s:
                seen_s.add(st)
                states_visited.append(st)
        payload = {
            "grid_convergence_time_s": _compute_grid_convergence_time_s(safe_hist, fps),
            "time_to_land_s": land_time_s,
            "zone_accuracy": zone_accuracy,
            "abort_detection_latency_s": abort_detection_latency_s,
            "inject_time_s": inject_time_s,
            "abort_time_s": abort_time_s,
            "states_visited": states_visited,
            "landed": land_time_s is not None,
            "zone_at_land_xy": [float(zone_at_land[0]), float(zone_at_land[1])] if zone_at_land else None,
            "ground_truth_note": (
                "zone_accuracy: fusion safety_map at landing centroid vs zone_safety_threshold "
                "(not offline_fusion.GROUND_TRUTH_SAFE_ZONES)."
            ),
            "safe_clearing_radius_m": clearing_r,
            "zone_safety_threshold": thr,
        }
        mj = Path(args.headless_metrics_json)
        if not mj.is_absolute():
            mj = _REPO_ROOT / mj
        mj.parent.mkdir(parents=True, exist_ok=True)
        mj.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[demo] Wrote headless metrics {mj}", flush=True)

    if not headless:
        print(f"Wrote {out_drone}")
        print(f"Wrote {out_chase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
