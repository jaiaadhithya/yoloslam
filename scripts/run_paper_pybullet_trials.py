#!/usr/bin/env python3
"""
Batch PyBullet headless runs for paper metrics (multiple seeds, abort on/off).

Usage (repo root):
  PYTHONPATH=src python3 scripts/run_paper_pybullet_trials.py

Writes results/paper_metrics.json (merged trials + summary statistics).
"""

from __future__ import annotations

import copy
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_SEEDS = [42, 123, 256, 512, 1024]


def _run_headless(repo: Path, cfg_path: Path, metrics_out: Path) -> None:
    env = os.environ.copy()
    sep = os.pathsep
    src = str(repo / "src")
    env["PYTHONPATH"] = f"{src}{sep}{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_pybullet_demo.py"),
        "--config",
        str(cfg_path),
        "--headless-metrics-json",
        str(metrics_out),
    ]
    r = subprocess.run(cmd, cwd=str(repo), env=env)
    if r.returncode != 0:
        raise SystemExit(f"run_pybullet_demo failed ({r.returncode}) for {cfg_path}")


def main() -> int:
    base_cfg_path = _REPO / "config" / "pybullet_demo.yaml"
    with base_cfg_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    out_dir = _REPO / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict] = []

    for seed in _SEEDS:
        for label, abort_flag in (("no_abort", False), ("with_abort", True)):
            cfg = copy.deepcopy(base)
            cfg["seed"] = int(seed)
            if "scene" not in cfg:
                cfg["scene"] = {}
            cfg["scene"]["seed"] = int(seed)
            if "slam" not in cfg:
                cfg["slam"] = {}
            cfg["slam"]["vo_seed"] = int(seed)
            if "abort_demo" not in cfg:
                cfg["abort_demo"] = {}
            cfg["abort_demo"]["enabled"] = bool(abort_flag)
            cfg["artifacts"] = {"save_grid_evolution_png": False}
            if abort_flag:
                cfg["max_duration_s"] = max(float(cfg.get("max_duration_s", 120)), 160.0)
                # DESCEND is often only a few frames before touchdown — inject on first DESCEND tick.
                ad = cfg.get("abort_demo") or {}
                ad["inject_after_descend_s"] = 0.0
                cfg["abort_demo"] = ad

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
                encoding="utf-8",
            ) as tf:
                yaml.dump(cfg, tf, default_flow_style=False, sort_keys=False)
                tmp_path = Path(tf.name)

            metrics_path = out_dir / f"paper_metrics_seed{seed}_{label}.json"
            try:
                _run_headless(_REPO, tmp_path, metrics_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            trials.append(
                {
                    "seed": seed,
                    "scenario": label,
                    "grid_convergence_time_s": data.get("grid_convergence_time_s"),
                    "time_to_land_s": data.get("time_to_land_s"),
                    "zone_accuracy": data.get("zone_accuracy"),
                    "abort_detection_latency_s": data.get("abort_detection_latency_s"),
                    "states_visited": data.get("states_visited", []),
                    "landed": data.get("landed"),
                    "inject_time_s": data.get("inject_time_s"),
                    "abort_time_s": data.get("abort_time_s"),
                }
            )

    def _mean_std(key: str, subset: list[dict]) -> tuple[float | None, float | None]:
        vals = [t[key] for t in subset if t.get(key) is not None and isinstance(t[key], (int, float))]
        if not vals:
            return None, None
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(statistics.mean(vals)), float(statistics.pstdev(vals))

    no_abort = [t for t in trials if t["scenario"] == "no_abort"]
    with_abort = [t for t in trials if t["scenario"] == "with_abort"]

    m_conv, s_conv = _mean_std("grid_convergence_time_s", no_abort)
    m_land, s_land = _mean_std("time_to_land_s", no_abort)
    m_land_ab, s_land_ab = _mean_std("time_to_land_s", with_abort)
    acc_vals = [t["zone_accuracy"] for t in no_abort if t.get("zone_accuracy") is not None]
    m_acc = float(sum(1.0 for v in acc_vals if v)) / len(acc_vals) if acc_vals else None
    ab_lat = [t["abort_detection_latency_s"] for t in with_abort if t.get("abort_detection_latency_s") is not None]
    m_ab = float(statistics.mean(ab_lat)) if ab_lat else None
    s_ab = float(statistics.pstdev(ab_lat)) if len(ab_lat) > 1 else (0.0 if ab_lat else None)

    summary = {
        "mean_convergence_time": m_conv,
        "std_convergence_time": s_conv,
        "mean_time_to_land": m_land,
        "std_time_to_land": s_land,
        "mean_time_to_land_with_abort": m_land_ab,
        "std_time_to_land_with_abort": s_land_ab,
        "mean_zone_accuracy": m_acc,
        "mean_abort_latency": m_ab,
        "std_abort_latency": s_ab,
        "seeds": _SEEDS,
        "note": (
            "zone_accuracy: fraction of no_abort trials where fusion safety_map value at "
            "landing (x,y) exceeds zone_safety_threshold. "
            "offline_fusion.GROUND_TRUTH_SAFE_ZONES apply only to offline fusion sim."
        ),
    }

    merged: dict[int, dict] = {}
    for t in trials:
        s = int(t["seed"])
        if s not in merged:
            merged[s] = {
                "seed": s,
                "grid_convergence_time_s": None,
                "time_to_land_s": None,
                "time_to_land_with_abort_s": None,
                "zone_accuracy": None,
                "abort_detection_latency_s": None,
                "states_visited": list(t.get("states_visited") or []),
            }
        if t["scenario"] == "no_abort":
            merged[s]["grid_convergence_time_s"] = t.get("grid_convergence_time_s")
            merged[s]["time_to_land_s"] = t.get("time_to_land_s")
            merged[s]["zone_accuracy"] = t.get("zone_accuracy")
            if t.get("states_visited"):
                merged[s]["states_visited"] = list(t["states_visited"])
        else:
            merged[s]["time_to_land_with_abort_s"] = t.get("time_to_land_s")
            merged[s]["abort_detection_latency_s"] = t.get("abort_detection_latency_s")

    trials_merged = [merged[k] for k in sorted(merged.keys())]
    payload_merged = {
        "trials": trials_merged,
        "summary": summary,
        "trials_raw": trials,
    }

    out_json = out_dir / "paper_metrics.json"
    out_json.write_text(json.dumps(payload_merged, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} ({len(trials)} runs, {len(trials_merged)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
