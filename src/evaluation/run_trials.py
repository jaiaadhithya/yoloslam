import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation.offline_fusion import (
    GROUND_TRUTH_SAFE_ZONES,
    OfflineFusionConfig,
    run_offline_fusion,
    write_fusion_log,
)
from evaluation.simulated_landing import simulate_landing_from_fusion_log, write_trajectory


def _condition_noise(condition: str) -> tuple[float, float]:
    if condition == "fused":
        return 3.0, 0.08
    if condition == "slam_only":
        return 9.0, 0.18
    return 7.0, 0.26


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    conditions = ["yolo_only", "slam_only", "fused"]
    rng = np.random.default_rng(2026)
    for condition in conditions:
        for trial_id in range(args.num_trials):
            start_x = float(rng.uniform(-10.0, 10.0))
            start_y = float(rng.uniform(-10.0, 10.0))
            det_noise, pose_noise = _condition_noise(condition)
            cfg = OfflineFusionConfig(
                condition=condition,
                seed=trial_id + abs(hash(condition)) % 10000,
                start_x=start_x,
                start_y=start_y,
                detection_noise_px=det_noise,
                pose_noise_m=pose_noise,
            )
            fusion_rows, selected_zone = run_offline_fusion(cfg)
            sim_result, trajectory_rows = simulate_landing_from_fusion_log(
                fusion_rows=fusion_rows,
                gt_zone_xy=selected_zone if selected_zone != (0.0, 0.0) else GROUND_TRUTH_SAFE_ZONES[0],
                dt=cfg.dt,
            )

            run_dir = out / "combined" / condition / f"trial_{trial_id:03d}"
            write_fusion_log(fusion_rows, str(run_dir / "fusion_log.csv"))
            write_trajectory(trajectory_rows, str(run_dir / "trajectory.csv"))

            all_rows.append(
                {
                    "trial_id": trial_id,
                    "condition": condition,
                    "landing_error_m": sim_result.landing_error_m,
                    "landing_time_s": sim_result.landing_time_s,
                    "success": sim_result.success,
                    "final_state": sim_result.final_state,
                    "num_state_changes": sim_result.num_state_changes,
                    "touched_safe_zone": sim_result.touched_safe_zone,
                    "selected_zone_x": selected_zone[0],
                    "selected_zone_y": selected_zone[1],
                }
            )

    csv_path = out / "comparison" / "ablation_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} trial rows to {csv_path}")


if __name__ == "__main__":
    main()
