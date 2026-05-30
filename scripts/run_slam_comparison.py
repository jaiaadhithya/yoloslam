"""Run multi-SLAM benchmark on TUM RGB-D and write paper artifacts to compre/."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from slam_module.backends.base import SlamBackend
from slam_module.backends.droid_slam import DroidSlamStreaming
from slam_module.backends.opencv_orb_rgbd import OpenCvOrbRgbdSlam
from slam_module.backends.orbslam3 import OrbSlam3External
from slam_module.backends.keyframe_pose_graph_rgbd import KeyframePoseGraphRgbdSlam
from slam_module.backends.yoloslam_rkf_rgbd import YoloslamRkfRgbdSlam
from slam_module.backends.open3d_rgbd import Open3dRgbdOdometry
from slam_module.backends.rtabmap import RtabmapCliSlam, RtabmapRgbdSlam
from slam_module.backends.tum_loader import write_tum_poses


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset() -> Path:
    return _repo_root() / "datasets" / "tum_rgbd" / "rgbd_dataset_freiburg1_xyz"


def _ground_truth_path(dataset_dir: Path) -> Path:
    return dataset_dir / "groundtruth.txt"


def _candidate_backends() -> list[tuple[str, SlamBackend]]:
    return [
        ("opencv_orb_rgbd", OpenCvOrbRgbdSlam()),
        ("orb_slam3", OrbSlam3External(mode="rgbd", use_wsl=None)),
        ("droid_slam", DroidSlamStreaming()),
        ("yoloslam_rkf_rgbd", YoloslamRkfRgbdSlam()),
        ("keyframe_pose_graph_rgbd", KeyframePoseGraphRgbdSlam()),
        ("open3d_rgbd", Open3dRgbdOdometry()),
        ("rtabmap", RtabmapRgbdSlam()),
        ("rtabmap_cli", RtabmapCliSlam()),
    ]


def _backend_available(name: str, backend: SlamBackend) -> tuple[bool, str]:
    if hasattr(backend, "is_available"):
        ok = bool(backend.is_available())  # type: ignore[attr-defined]
        return ok, "binary/cli probe"
    try:
        backend.reset()
        return True, "python backend"
    except Exception as exc:
        return False, str(exc)


def _evo_metrics(gt_path: Path, est_path: Path, out_dir: Path) -> dict[str, Any]:
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface

    out_dir.mkdir(parents=True, exist_ok=True)
    gt = file_interface.read_tum_trajectory_file(str(gt_path))
    est = file_interface.read_tum_trajectory_file(str(est_path))
    gt, est = sync.associate_trajectories(gt, est, max_diff=0.02)
    est.align(gt, correct_scale=True)
    est_aligned = est
    from evo.core import metrics as evo_metrics

    ape = evo_metrics.APE(PoseRelation.translation_part)
    ape.process_data((gt, est_aligned))
    ape_stats = ape.get_all_statistics()

    rpe = evo_metrics.RPE(PoseRelation.translation_part, delta=1.0, delta_unit=Unit.meters)
    try:
        rpe.process_data((gt, est_aligned))
        rpe_stats = rpe.get_all_statistics()
    except Exception:
        rpe = evo_metrics.RPE(PoseRelation.translation_part, delta=0.5, delta_unit=Unit.meters)
        try:
            rpe.process_data((gt, est_aligned))
            rpe_stats = rpe.get_all_statistics()
        except Exception:
            rpe = evo_metrics.RPE(PoseRelation.translation_part, delta=1, delta_unit=Unit.frames)
            rpe.process_data((gt, est_aligned))
            rpe_stats = rpe.get_all_statistics()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    gt_pos = gt.positions_xyz
    est_pos = est_aligned.positions_xyz
    axes[0].plot(gt_pos[:, 0], gt_pos[:, 1], "g-", label="Ground truth", linewidth=2)
    axes[0].plot(est_pos[:, 0], est_pos[:, 1], "r--", label="Estimate", linewidth=1.8)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title("Top-down trajectory")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    err = ape.error
    axes[1].plot(err, color="#1f4e79")
    axes[1].set_title(f"ATE translation error (RMSE={ape_stats['rmse']:.4f} m)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Error (m)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "trajectory_eval.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "ate_rmse_m": float(ape_stats["rmse"]),
        "ate_mean_m": float(ape_stats["mean"]),
        "ate_median_m": float(ape_stats["median"]),
        "ate_std_m": float(ape_stats["std"]),
        "rpe_rmse_m": float(rpe_stats["rmse"]),
        "rpe_mean_m": float(rpe_stats["mean"]),
        "num_synced_poses": int(len(gt.timestamps)),
        "plot": str(plot_path.resolve()),
    }


def _run_one_backend(
    name: str,
    backend: SlamBackend,
    dataset_dir: Path,
    out_root: Path,
    *,
    max_frames: int | None,
    frame_stride: int,
) -> dict[str, Any]:
    backend_out = out_root / name
    backend_out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    timestamps, poses = backend.run_dataset(
        dataset_dir,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    runtime_s = time.perf_counter() - t0
    est_path = backend_out / "estimated.tum"
    write_tum_poses(est_path, timestamps, poses)
    metrics = _evo_metrics(_ground_truth_path(dataset_dir), est_path, backend_out)
    result = {
        "backend_key": name,
        "backend": backend.metadata(),
        "status": "ok",
        "runtime_s": runtime_s,
        "num_poses": int(len(timestamps)),
        "estimated_trajectory": str(est_path.resolve()),
        "metrics": metrics,
    }
    (backend_out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        raise RuntimeError("No SLAM backend completed successfully")
    best = min(ok, key=lambda r: float(r["metrics"]["ate_rmse_m"]))
    return {
        "selected_backend": best["backend_key"],
        "selected_github": best["backend"]["github"],
        "ate_rmse_m": best["metrics"]["ate_rmse_m"],
        "rpe_rmse_m": best["metrics"]["rpe_rmse_m"],
        "reason": "Lowest ATE RMSE on TUM freiburg1_xyz after timestamp association.",
    }


def _write_summary_table(results: list[dict[str, Any]], path: Path, best_key: str) -> None:
    lines = [
        "# SLAM Comparison (TUM RGB-D freiburg1_xyz)",
        "",
        "| Backend | GitHub | ATE RMSE (m) | RPE RMSE (m) | Poses | Runtime (s) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(results, key=lambda x: x.get("backend_key", "")):
        if r.get("status") == "ok":
            m = r["metrics"]
            mark = " **best**" if r["backend_key"] == best_key else ""
            lines.append(
                f"| {r['backend']['name']}{mark} | {r['backend']['github']} | "
                f"{m['ate_rmse_m']:.4f} | {m['rpe_rmse_m']:.4f} | {r['num_poses']} | "
                f"{r['runtime_s']:.1f} | ok |"
            )
        else:
            lines.append(
                f"| {r.get('backend_key', '?')} | {r.get('github', '-')} | - | - | - | - | "
                f"{r.get('status', 'failed')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_bar_chart(results: list[dict[str, Any]], path: Path) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return
    names = [r["backend"]["name"] for r in ok]
    ate = [r["metrics"]["ate_rmse_m"] for r in ok]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, ate, color="#1f4e79")
    best_idx = int(np.argmin(ate))
    bars[best_idx].set_color("#2ca02c")
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("SLAM benchmark on TUM freiburg1_xyz")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SLAM backends and write compre/ artifacts")
    parser.add_argument("--dataset-dir", type=Path, default=_default_dataset())
    parser.add_argument("--out-dir", type=Path, default=_repo_root() / "compre" / "slam_benchmark")
    parser.add_argument("--max-frames", type=int, default=300, help="Cap frames for faster runs")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument(
        "--backends",
        nargs="*",
        default=[
            "yoloslam_rkf_rgbd",
            "keyframe_pose_graph_rgbd",
            "opencv_orb_rgbd",
            "open3d_rgbd",
            "orb_slam3",
            "droid_slam",
            "rtabmap_cli",
        ],
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = _ground_truth_path(args.dataset_dir)
    if not gt.exists():
        raise FileNotFoundError(f"Missing ground truth: {gt}")

    registry = {k: v for k, v in _candidate_backends()}
    selected = [b for b in args.backends if b in registry]

    results: list[dict[str, Any]] = []
    for key in selected:
        backend = registry[key]
        available, reason = _backend_available(key, backend)
        if not available:
            results.append(
                {
                    "backend_key": key,
                    "github": backend.info.github,
                    "status": "skipped",
                    "reason": reason,
                }
            )
            print(f"[skip] {key}: {reason}")
            continue
        print(f"[run] {key} ...")
        try:
            result = _run_one_backend(
                key,
                backend,
                args.dataset_dir,
                args.out_dir,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
            )
            results.append(result)
            print(
                f"  ATE RMSE={result['metrics']['ate_rmse_m']:.4f} m, "
                f"runtime={result['runtime_s']:.1f}s"
            )
        except Exception as exc:
            results.append(
                {
                    "backend_key": key,
                    "github": backend.info.github,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[fail] {key}: {exc}")

    best = _select_best(results)
    comparison = {
        "dataset": str(args.dataset_dir.resolve()),
        "ground_truth": str(gt.resolve()),
        "max_frames": args.max_frames,
        "frame_stride": args.frame_stride,
        "results": results,
        "best": best,
    }
    (args.out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_summary_table(results, args.out_dir / "comparison_table.md", best["selected_backend"])
    _plot_bar_chart(results, args.out_dir / "ate_rmse_bar.png")

    best_cfg = {
        "backend": best["selected_backend"],
        "selection": best,
        "integrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (_repo_root() / "compre" / "best_slam.json").write_text(
        json.dumps(best_cfg, indent=2),
        encoding="utf-8",
    )

    # Update runtime config to use the winning backend.
    cfg_path = _repo_root() / "config" / "slam_backend.yaml"
    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    updated = []
    for line in lines:
        if line.startswith("backend:"):
            updated.append(f"backend: {best['selected_backend']}")
        else:
            updated.append(line)
    cfg_path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    readme = _repo_root() / "compre" / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# compre — paper comparison artifacts",
                "",
                "Generated by `scripts/run_slam_comparison.py`.",
                "",
                f"- Best SLAM backend: **{best['selected_backend']}** ({best['selected_github']})",
                f"- ATE RMSE: **{best['ate_rmse_m']:.4f} m**",
                f"- RPE RMSE: **{best['rpe_rmse_m']:.4f} m**",
                "",
                "## Contents",
                "",
                "- `slam_benchmark/comparison.json` — full metrics for all backends",
                "- `slam_benchmark/comparison_table.md` — paper-ready table",
                "- `slam_benchmark/ate_rmse_bar.png` — bar chart",
                "- `best_slam.json` — selected backend metadata",
                "",
                "Runtime integration: `config/slam_backend.yaml` points at the winning backend.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
