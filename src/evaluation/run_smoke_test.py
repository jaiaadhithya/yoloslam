"""
Sequential smoke tests (no ROS / Gazebo / training):

  PYTHONPATH=src python -m evaluation.run_smoke_test

Artifacts under results/smoke_test/
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

SMOKE_ROOT = Path("results/smoke_test")
PIPELINE_MAX_STEPS = 400
PIPELINE_DT = 0.1
PIPELINE_SEED = 42
PIPELINE_FRAME_STRIDE = 36
PIPELINE_INJECTION_DELAY = 0.8


def _verify_full_pipeline(fp_dir: Path) -> dict[str, Any]:
    """Check DESCEND, disturbance, ABORT ordering and transition_reason."""
    csv_path = fp_dir / "state_log.csv"
    summary_path = fp_dir / "scenario_summary.json"
    out: dict[str, Any] = {"ok": True, "checks": []}

    if not csv_path.is_file():
        out["ok"] = False
        out["error"] = f"Missing {csv_path}"
        return out

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    first_descend = next((r for r in rows if r.get("state") == "DESCEND"), None)
    first_abort = next((r for r in rows if r.get("state") == "ABORT"), None)
    inj_row = next((r for r in rows if str(r.get("obstacle_injected", "0")) == "1"), None)

    out["first_descend_time_s"] = float(first_descend["time_s"]) if first_descend else None
    out["disturbance_injection_time_s"] = float(inj_row["time_s"]) if inj_row else None
    out["abort_occurred"] = first_abort is not None
    out["abort_transition_reason"] = first_abort.get("transition_reason", "") if first_abort else ""

    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            out["summary_injection_s"] = summary.get("injection_time_s")
            out["summary_reaction_latency_s"] = summary.get("reaction_latency_to_abort_s")
        except json.JSONDecodeError:
            out["summary_warning"] = "Could not parse scenario_summary.json"

    reason = out.get("abort_transition_reason") or ""
    reason_ok = "intrusion_detected" in reason or "zone_unsafe" in reason or "unsafe" in reason.lower()
    out["abort_reason_matches_expected"] = bool(first_abort and reason_ok)

    inj_t = out.get("disturbance_injection_time_s")
    abort_t = float(first_abort["time_s"]) if first_abort else None
    if inj_t is not None and abort_t is not None:
        out["abort_after_disturbance"] = abort_t + 1e-9 >= inj_t
    else:
        out["abort_after_disturbance"] = None

    frames_dir = fp_dir / "frames"
    out["frames_dir_exists"] = frames_dir.is_dir()
    out["frame_png_count"] = len(list(frames_dir.glob("frame_*.png"))) if out["frames_dir_exists"] else 0
    out["decision_timeline_exists"] = (fp_dir / "decision_timeline.png").is_file()

    checks_pass = [
        first_descend is not None,
        inj_row is not None,
        first_abort is not None,
        out.get("abort_after_disturbance") is True,
        out.get("abort_reason_matches_expected") is True,
        out["decision_timeline_exists"],
        out["frame_png_count"] > 0,
    ]
    out["all_checks_pass"] = all(checks_pass)
    return out


def main() -> None:
    root = SMOKE_ROOT
    root.mkdir(parents=True, exist_ok=True)

    print("Smoke test root:", root.resolve())
    print()

    t_wall0 = time.perf_counter()
    phase_results: list[tuple[str, bool, str, dict[str, Any]]] = []

    # 1) YOLO
    try:
        from evaluation.test_yolo_inference import run_yolo_smoke

        t0 = time.perf_counter()
        r = run_yolo_smoke(root / "yolo")
        dt = time.perf_counter() - t0
        ok = bool(r.get("ok"))
        msg = (
            f"device={r.get('device')} cuda_avail={r.get('cuda_available')} "
            f"model={r.get('model_path')} infer={r.get('inference_time_s', 0):.3f}s wall={dt:.3f}s"
        )
        if not ok:
            msg = r.get("error", "failed")
        phase_results.append(("YOLO inference", ok, msg, r))
    except Exception as e:
        phase_results.append(("YOLO inference", False, repr(e), {}))

    # 2) SLAM
    try:
        from evaluation.test_slam import run_slam_smoke

        t0 = time.perf_counter()
        r = run_slam_smoke(root / "slam", num_steps=200, frame_seed=42)
        dt = time.perf_counter() - t0
        fwd = float(r.get("forward_progress_m", 0.0))
        des = float(r.get("total_descent_m", 0.0))
        ok = bool(r.get("ok")) and fwd > 0.8 and des > 0.3
        msg = f"forward_m={fwd:.2f} descent_m={des:.2f} runtime={r.get('runtime_s', 0):.4f}s wall={dt:.3f}s"
        phase_results.append(("SLAM wrapper", ok, msg, r))
    except Exception as e:
        phase_results.append(("SLAM wrapper", False, repr(e), {}))

    # 3) Fusion
    try:
        from evaluation.test_fusion import run_fusion_smoke

        t0 = time.perf_counter()
        r = run_fusion_smoke(root / "fusion", seed=7, num_steps=120)
        dt = time.perf_counter() - t0
        ok = bool(r.get("ok")) and bool(r.get("has_valid_zone"))
        msg = (
            f"valid_zone={r.get('has_valid_zone')} score={r.get('zone_score'):.2f} "
            f"loop={r.get('runtime_s', 0):.4f}s wall={dt:.3f}s"
        )
        phase_results.append(("Fusion grid", ok, msg, r))
    except Exception as e:
        phase_results.append(("Fusion grid", False, repr(e), {}))

    # 4) Full pipeline (closed-loop decisions)
    fp_dir = root / "full_pipeline"
    pipeline_steps = 0
    pipeline_s = 0.0
    steps_per_s = 0.0
    try:
        from evaluation.run_full_pipeline_decisions import run_closed_loop

        t0 = time.perf_counter()
        run_closed_loop(
            results_dir=fp_dir,
            dt=PIPELINE_DT,
            max_steps=PIPELINE_MAX_STEPS,
            seed=PIPELINE_SEED,
            descend_injection_delay_s=PIPELINE_INJECTION_DELAY,
            frame_stride=PIPELINE_FRAME_STRIDE,
            write_gif=False,
        )
        pipeline_s = time.perf_counter() - t0
        log_csv = fp_dir / "state_log.csv"
        timeline_png = fp_dir / "decision_timeline.png"
        if log_csv.is_file():
            with log_csv.open(encoding="utf-8") as f:
                pipeline_steps = max(0, sum(1 for _ in f) - 1)
        steps_per_s = pipeline_steps / pipeline_s if pipeline_s > 0 else 0.0
        pipe_ok = pipeline_steps > 0 and timeline_png.is_file()
        phase_results.append(
            (
                "Full pipeline (decisions)",
                pipe_ok,
                f"steps={pipeline_steps} wall={pipeline_s:.3f}s ({steps_per_s:.1f} steps/s)",
                {"pipeline_wall_s": pipeline_s, "steps": pipeline_steps},
            )
        )
    except Exception as e:
        phase_results.append(("Full pipeline (decisions)", False, repr(e), {}))

    t_wall = time.perf_counter() - t_wall0

    # Verification printout (state machine)
    print("--- State machine verification (full_pipeline) ---")
    verify = _verify_full_pipeline(fp_dir)
    print(f"  First DESCEND time (s): {verify.get('first_descend_time_s')}")
    print(f"  Disturbance injection time (s): {verify.get('disturbance_injection_time_s')}")
    print(f"  ABORT occurred: {verify.get('abort_occurred')}")
    print(f"  transition_reason at first ABORT: {verify.get('abort_transition_reason')!r}")
    print(f"  ABORT after disturbance: {verify.get('abort_after_disturbance')}")
    print(f"  Reason matches (intrusion/unsafe): {verify.get('abort_reason_matches_expected')}")
    print(f"  decision_timeline.png present: {verify.get('decision_timeline_exists')}")
    print(f"  frame PNGs written: {verify.get('frame_png_count')}")
    print(f"  All automated checks: {verify.get('all_checks_pass')}")
    print()

    # Performance summary
    print("--- Performance ---")
    print(f"  Full pipeline wall time: {pipeline_s:.3f} s")
    print(f"  Simulated steps (CSV rows): {pipeline_steps}")
    print(f"  Effective steps/s (loop + plots): {steps_per_s:.1f}")
    print(f"  Total smoke wall time: {t_wall:.3f} s")
    if pipeline_s > 45.0:
        print(
            "  Note: pipeline dominated by Matplotlib frame PNGs; increase PIPELINE_FRAME_STRIDE "
            "in run_smoke_test.py if needed.",
        )
    print()

    # Summary table
    print("--- Phase summary ---")
    for name, ok, msg, _ in phase_results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    print()

    overall = all(ok for _, ok, _, _ in phase_results) and verify.get("all_checks_pass", False)
    print("OVERALL:", "PASS" if overall else "FAIL")
    print()
    print("Outputs:")
    print(f"  {root.resolve() / 'yolo'}")
    print(f"  {root.resolve() / 'slam'}")
    print(f"  {root.resolve() / 'fusion'}")
    print(f"  {fp_dir.resolve()}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
