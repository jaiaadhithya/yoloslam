#!/usr/bin/env python3
"""
Collect all paper assets into PAPER_RESULTS/ (PyBullet demos, frames, metrics, plots).

Run from repo root; uses WSL-friendly subprocesses (ffmpeg/ffprobe optional).

  PYTHONPATH=src python3 scripts/collect_paper_results.py
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_PAPER = _REPO / "PAPER_RESULTS"
_FRAMES = _REPO / "results" / "paper_frames"
_DEMO_GRID = _REPO / "results" / "pybullet_demo" / "grid_evolution.png"


def _env() -> dict:
    e = os.environ.copy()
    sep = os.pathsep
    e["PYTHONPATH"] = f"{_REPO / 'src'}{sep}{e.get('PYTHONPATH', '')}"
    return e


def _run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(_REPO), env=_env(), **kw)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {cmd} ({r.returncode})")


def main() -> int:
    _PAPER.mkdir(parents=True, exist_ok=True)
    _FRAMES.mkdir(parents=True, exist_ok=True)
    (_REPO / "results" / "pybullet_demo").mkdir(parents=True, exist_ok=True)

    base_path = _REPO / "config" / "pybullet_demo.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    # --- 1) Normal landing demo + state log + grid snapshots ---
    cfg1 = copy.deepcopy(base)
    cfg1["abort_demo"] = cfg1.get("abort_demo") or {}
    cfg1["abort_demo"]["enabled"] = False
    cfg1["artifacts"] = {
        "save_grid_evolution_png": True,
        "grid_plot_path": "results/pybullet_demo/grid_evolution.png",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.dump(cfg1, tf, default_flow_style=False, sort_keys=False)
        tmp1 = Path(tf.name)
    try:
        _run(
            [
                sys.executable,
                str(_REPO / "scripts" / "run_pybullet_demo.py"),
                "--config",
                str(tmp1),
                "--state-log-csv",
                str(_FRAMES / "state_log_normal.csv"),
                "--grid-snapshots-dir",
                str(_FRAMES),
            ]
        )
    finally:
        tmp1.unlink(missing_ok=True)

    # Decision timeline
    _run(
        [
            sys.executable,
            "-m",
            "evaluation.plot_decision_timeline",
            "--state-log",
            str(_FRAMES / "state_log_normal.csv"),
            "--output",
            str(_FRAMES / "decision_timeline.png"),
        ]
    )

    # ffprobe + ffmpeg key frames
    drone = _REPO / (cfg1.get("output_video_drone") or "demo_landing_drone.mp4")
    chase = _REPO / (cfg1.get("output_video_chase") or "demo_landing_chase.mp4")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    nframes: int | None = None
    if ffprobe and drone.is_file():
        pr = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(drone),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
        )
        if pr.returncode == 0 and pr.stdout.strip().isdigit():
            nframes = int(pr.stdout.strip())
            print(f"[collect] {drone.name} frame count: {nframes}", flush=True)

    fps = float(cfg1.get("fps", 30))
    # Approximate indices: survey ~5s, approach mid, descend, landed hold
    candidates = [int(5 * fps), int(12 * fps), int(28 * fps), int(37 * fps)]
    if nframes is not None:
        candidates = [min(c, max(0, nframes - 1)) for c in candidates]
        # de-duplicate sorted
        seen = set()
        picks = []
        for c in sorted(candidates):
            if c not in seen:
                seen.add(c)
                picks.append(c)
        candidates = picks[:4]
    sel = "+".join(f"eq(n\\,{n})" for n in candidates[:4])
    if ffmpeg and drone.is_file():
        _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(drone),
                "-vf",
                f"select='{sel}'",
                "-vsync",
                "vfr",
                str(_FRAMES / "pov_%02d.png"),
            ]
        )
    if ffmpeg and chase.is_file():
        _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(chase),
                "-vf",
                f"select='{sel}'",
                "-vsync",
                "vfr",
                str(_FRAMES / "chase_%02d.png"),
            ]
        )

    # --- 2) Abort demo ---
    cfg2 = copy.deepcopy(base)
    cfg2["abort_demo"] = cfg2.get("abort_demo") or {}
    cfg2["abort_demo"]["enabled"] = True
    cfg2["max_duration_s"] = max(float(cfg2.get("max_duration_s", 120)), 160.0)
    cfg2["artifacts"] = {"save_grid_evolution_png": False}
    cfg2["output_video_drone"] = "demo_landing_abort_drone.mp4"
    cfg2["output_video_chase"] = "demo_landing_abort_chase.mp4"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.dump(cfg2, tf, default_flow_style=False, sort_keys=False)
        tmp2 = Path(tf.name)
    try:
        _run([sys.executable, str(_REPO / "scripts" / "run_pybullet_demo.py"), "--config", str(tmp2)])
    finally:
        tmp2.unlink(missing_ok=True)

    abort_chase = _REPO / "demo_landing_abort_chase.mp4"
    if ffmpeg and abort_chase.is_file():
        _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(abort_chase),
                "-vf",
                "fps=2",
                str(_FRAMES / "abort_chase_%03d.png"),
            ]
        )

    # --- 3) Batch metrics ---
    _run([sys.executable, str(_REPO / "scripts" / "run_paper_pybullet_trials.py")])

    # --- 4) Assemble PAPER_RESULTS ---
    if _PAPER.exists():
        shutil.rmtree(_PAPER)
    _PAPER.mkdir(parents=True, exist_ok=True)

    metrics_src = _REPO / "results" / "paper_metrics.json"
    if metrics_src.is_file():
        shutil.copy2(metrics_src, _PAPER / "paper_metrics.json")
    if _DEMO_GRID.is_file():
        shutil.copy2(_DEMO_GRID, _PAPER / "grid_evolution.png")

    for name in (
        "demo_landing_drone.mp4",
        "demo_landing_chase.mp4",
        "demo_landing_abort_drone.mp4",
        "demo_landing_abort_chase.mp4",
    ):
        src = _REPO / name
        if src.is_file():
            shutil.copy2(src, _PAPER / name)

    for pat in (
        "pov_*.png",
        "chase_*.png",
        "abort_chase_*.png",
        "grid_t05.png",
        "grid_t15.png",
        "grid_t25.png",
        "grid_descent.png",
        "decision_timeline.png",
        "state_log_normal.csv",
    ):
        for f in _FRAMES.glob(pat):
            shutil.copy2(f, _PAPER / f.name)

    print(f"[collect] Done. Outputs in {_PAPER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
