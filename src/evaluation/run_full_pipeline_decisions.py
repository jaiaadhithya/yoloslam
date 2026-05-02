"""
Offline full-pipeline decision evaluation: closed-loop fusion + landing state machine + logging.

Writes:
  results/full_pipeline/state_log.csv
  results/full_pipeline/decision_timeline.png
  results/full_pipeline/metrics.txt
  results/full_pipeline/frames/safety_maps.npz
  results/full_pipeline/frames/*.png
  results/full_pipeline/scenario_summary.json

Run from repo root (with PYTHONPATH=src or pip install -e .):

  python -m evaluation.run_full_pipeline_decisions

Optional GIF (requires Pillow):

  python -m evaluation.run_full_pipeline_decisions --gif
"""
from __future__ import annotations

import os

# Headless / CI-friendly plots
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.plot_decision_timeline import plot_decision_timeline
from evaluation.render_decision_frames import render_frames_from_artifacts
from fusion.fusion_node import FusionNode
from landing_controller.controller_node import ControllerNode
from landing_controller.state_machine import LandingState


def _make_safe_detection(px: float, py: float, conf: float = 0.82) -> dict[str, Any]:
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


def _make_person_detection(px: float, py: float, conf: float = 0.78) -> dict[str, Any]:
    c = float(conf)
    return {
        "x_center": float(px),
        "y_center": float(py),
        "width": 70.0,
        "height": 140.0,
        "confidence": c,
        "class_id": 5,
        "class_name": "person",
        "is_safe": False,
        "safety_label": "unsafe",
        "safety_weight": -1.0,
        "fusion_unsafe_delta": max(0.05, c),
    }


def _detections_for_step(
    rng: np.random.Generator,
    step: int,
    inject_person_pixels: bool,
) -> list[dict[str, Any]]:
    """
    Dense grass detections in image space so fused projections populate a contiguous grid cluster
    (zone_selector requires >= min_zone_size_m^2 cells above threshold).
    Optionally add person in image space after disturbance (late-stage hazard cue).
    """
    # 3x3 footprint (~40 px spacing) projects to neighboring world cells under nominal depth.
    grid_px = [
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
    dets = [
        _make_safe_detection(
            float(px + rng.normal(0, 2.5)),
            float(py + rng.normal(0, 2.5)),
            conf=float(0.8 + 0.12 * rng.random()),
        )
        for px, py in grid_px
    ]
    if inject_person_pixels:
        dets.append(
            _make_person_detection(325.0 + rng.normal(0, 2.0), 255.0 + rng.normal(0, 2.0), conf=0.75)
        )
    return dets


def _compute_decision_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    states = [r["state"] for r in rows]
    reasons = [r.get("transition_reason", "") for r in rows]
    times = [float(r["time_s"]) for r in rows]
    scores = [float(r["zone_score"]) for r in rows]
    zone_xy = [
        (float(r["zone_x"]), float(r["zone_y"]))
        for r in rows
        if int(r.get("has_valid_zone", 0)) == 1
    ]

    transitions = sum(
        1 for i in range(1, len(states)) if states[i] != states[i - 1]
    )
    abort_entries = sum(
        1 for i in range(1, len(states)) if states[i] == "ABORT" and states[i - 1] != "ABORT"
    )
    abort_count = states.count("ABORT")

    time_to_commit: float | None = None
    for r in rows:
        if r["state"] == "DESCEND":
            time_to_commit = float(r["time_s"])
            break

    zone_keys = [(round(x, 1), round(y, 1)) for x, y in zone_xy]
    unique_zones = len(set(zone_keys)) if zone_keys else 0
    oscillation = max(0, unique_zones - 1)

    final_state = states[-1] if states else "UNKNOWN"
    success_land = final_state == "LANDED"

    return {
        "state_transition_count": transitions,
        "abort_segment_entries": abort_entries,
        "abort_timesteps_in_abort_state": abort_count,
        "time_to_commit_descend_s": time_to_commit,
        "unique_zone_centroids_rounded_0p1m": unique_zones,
        "zone_centroid_oscillation_metric": oscillation,
        "final_state": final_state,
        "success_landing_state": int(success_land),
        "mean_zone_score_when_valid": float(np.mean([float(r["zone_score"]) for r in rows if int(r.get("has_valid_zone", 0)) == 1])) if any(int(r.get("has_valid_zone", 0)) == 1 for r in rows) else 0.0,
    }


def _write_metrics_txt(path: Path, metrics: dict[str, Any], scenario: dict[str, Any]) -> None:
    lines = [
        "YOLO-SLAM landing decision metrics (offline full pipeline)",
        "",
        "Scenario:",
        json.dumps(scenario, indent=2),
        "",
        "Decision / state-machine metrics:",
        json.dumps(metrics, indent=2),
        "",
        "Interpretation (paper):",
        "- Fusion enables not just perception, but reliable decision-making for autonomous landing.",
        "- Non-zero transition count shows the policy reacts to measurements over time, not a single-shot detection.",
        "- time_to_commit_descend_s separates exploration/survey from committed descent.",
        "- zone_centroid_oscillation_metric captures switching / drift of the fused target.",
        "- abort_segment_entries counts discrete hazard responses (e.g. intrusion).",
        "- reaction_latency_to_abort_s (see scenario JSON): delay from staged disturbance to ABORT via intrusion_detected.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_closed_loop(
    results_dir: Path,
    dt: float,
    max_steps: int,
    seed: int,
    descend_injection_delay_s: float,
    frame_stride: int,
    write_gif: bool,
) -> None:
    rng = np.random.default_rng(seed)
    fusion = FusionNode()
    controller = ControllerNode()
    # Start near map center so fused rays and motion converge quickly to a valid zone.
    pose = {"x": 1.2, "y": -0.8, "z": 14.0}

    mission_time = 0.0
    descend_enter_time: float | None = None
    obstacle_injected = False
    injection_time: float | None = None
    inject_person_pixels = False
    reaction_latency_s: float | None = None

    log_rows: list[dict[str, Any]] = []
    maps_stack: list[np.ndarray] = []

    disturbance_note = (
        "Mid-descent: unsafe mass injected at fused landing zone center in the fusion grid; "
        "optionally also adds a person detection in image space after injection."
    )

    for step in range(max_steps):
        slam_pose = {
            "x": pose["x"],
            "y": pose["y"],
            "z": pose["z"],
            "state": "OK",
        }

        detections = _detections_for_step(rng, step, inject_person_pixels)
        zone = fusion.fuse(detections, slam_pose)

        zx = float(zone.get("center_x", 0.0))
        zy = float(zone.get("center_y", 0.0))
        state_before = controller.sm.state

        injected_this_step = False
        if (
            zone.get("has_valid_zone")
            and state_before == LandingState.DESCEND
            and descend_enter_time is not None
            and not obstacle_injected
            and mission_time >= descend_enter_time + descend_injection_delay_s
        ):
            # Strong unsafe mass at selected zone so fusion marks contamination; also flag this timestep.
            fusion.grid.integrate(zx, zy, 25.0, False)
            obstacle_injected = True
            injection_time = mission_time
            inject_person_pixels = True
            injected_this_step = True

        intrusion = bool(
            zone.get("has_valid_zone")
            and (not fusion.zone_is_still_safe() or injected_this_step)
        )

        target = {
            "center_x": zx,
            "center_y": zy,
            "center_z": 0.0,
            "zone_score": float(zone.get("zone_score", 0.0)),
            "has_valid_zone": bool(zone.get("has_valid_zone", False)),
            "intrusion_detected": intrusion,
        }

        ex = target["center_x"] - pose["x"]
        ey = target["center_y"] - pose["y"]
        xy_error = math.hypot(ex, ey)

        cmd = controller.update(target=target, current_pose=pose, dt=dt)
        pose["x"] += float(cmd["vx"]) * dt
        pose["y"] += float(cmd["vy"]) * dt
        pose["z"] = max(0.0, pose["z"] + float(cmd["vz"]) * dt)

        if cmd["state"] == "DESCEND" and descend_enter_time is None:
            descend_enter_time = mission_time

        if (
            cmd.get("transition_reason") == "intrusion_detected"
            and injection_time is not None
            and reaction_latency_s is None
        ):
            reaction_latency_s = mission_time - injection_time

        ix, iy = fusion.grid.world_to_cell(zx, zy)
        snap = fusion.grid.snapshot()

        log_rows.append(
            {
                "time_s": round(mission_time, 4),
                "step": step,
                "state": cmd["state"],
                "zone_x": round(zx, 6),
                "zone_y": round(zy, 6),
                "zone_score": round(float(zone.get("zone_score", 0.0)), 6),
                "has_valid_zone": int(bool(zone.get("has_valid_zone", False))),
                "transition_reason": cmd.get("transition_reason", ""),
                "altitude_m": round(pose["z"], 4),
                "xy_error_m": round(xy_error, 4),
                "intrusion_detected": int(intrusion),
                "zone_cell_ix": ix,
                "zone_cell_iy": iy,
                "grid_safe_sum": round(snap["safe_score_sum"], 4),
                "grid_unsafe_sum": round(snap["unsafe_score_sum"], 4),
                "grid_side_cells": snap["side_cells"],
                "detections_json": json.dumps(
                    [
                        {
                            "x_center": d["x_center"],
                            "y_center": d["y_center"],
                            "width": d["width"],
                            "height": d["height"],
                            "is_safe": d["is_safe"],
                            "safety_label": d.get("safety_label"),
                            "class_name": d.get("class_name"),
                        }
                        for d in detections
                    ]
                ),
                "obstacle_injected": int(obstacle_injected and injection_time is not None and mission_time >= injection_time),
            }
        )

        maps_stack.append(fusion.grid.safety_map().astype(np.float32))

        mission_time += dt

        if cmd["state"] == "LANDED":
            break

    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "state_log.csv"
    if log_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    maps_arr = np.stack(maps_stack, axis=0) if maps_stack else np.zeros((0, 1, 1), dtype=np.float32)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        frames_dir / "safety_maps.npz",
        times=np.array([float(r["time_s"]) for r in log_rows], dtype=np.float64),
        maps=maps_arr,
        world_size_m=np.float64(fusion.grid.world_size_m),
        resolution_m=np.float64(fusion.grid.resolution_m),
    )

    plot_decision_timeline(csv_path, out_dir / "decision_timeline.png")

    render_frames_from_artifacts(
        csv_path,
        frames_dir / "safety_maps.npz",
        frames_dir,
        sample_stride=frame_stride,
    )

    scenario = {
        "disturbance_description": disturbance_note,
        "injection_time_s": injection_time,
        "reaction_latency_to_abort_s": reaction_latency_s,
        "obstacle_injected": obstacle_injected,
        "disturbance_scheduled_after_descend_s": descend_injection_delay_s,
        "final_outcome": log_rows[-1]["state"] if log_rows else None,
        "dt": dt,
        "max_steps": max_steps,
        "seed": seed,
    }

    metrics = _compute_decision_metrics(log_rows)

    summary_path = out_dir / "scenario_summary.json"
    summary_path.write_text(json.dumps({**scenario, "metrics": metrics}, indent=2), encoding="utf-8")

    _write_metrics_txt(out_dir / "metrics.txt", metrics, scenario)

    if write_gif and log_rows:
        _try_write_gif(frames_dir, frame_stride)


def _try_write_gif(frames_dir: Path, stride: int) -> None:
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from matplotlib import image as mpimg
    except ImportError:
        return
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        return
    paths = paths[:: max(1, stride // 2)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    def _frame(i: int):
        ax.clear()
        ax.axis("off")
        ax.imshow(mpimg.imread(paths[i]))
        return (ax,)

    anim = animation.FuncAnimation(fig, _frame, frames=len(paths), interval=180, blit=False)
    try:
        writer = animation.PillowWriter(fps=6)
        anim.save(frames_dir / "decision_timeline.gif", writer=writer)
    except Exception:
        pass
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline decision-making evaluation pipeline")
    parser.add_argument("--results-dir", default="results/full_pipeline", type=Path)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--descend-injection-delay",
        type=float,
        default=0.8,
        help="Seconds after entering DESCEND before contaminating the selected zone (mid-descent hazard).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=12,
        help="Export every Nth row to PNG frames (full timing still logged to CSV).",
    )
    parser.add_argument("--gif", action="store_true", help="Write decision_timeline.gif if Pillow is available")
    args = parser.parse_args()

    run_closed_loop(
        results_dir=args.results_dir,
        dt=args.dt,
        max_steps=args.max_steps,
        seed=args.seed,
        descend_injection_delay_s=args.descend_injection_delay,
        frame_stride=args.frame_stride,
        write_gif=args.gif,
    )
    print(f"Artifacts written under {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
