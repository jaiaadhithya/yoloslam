import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    out = Path(args.results_dir) / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    table_path = out / "ablation_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing ablation data: {table_path}. Run evaluation.run_trials first.")

    by_condition: dict[str, list[float]] = {"yolo_only": [], "slam_only": [], "fused": []}
    with table_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_condition[row["condition"]].append(float(row["landing_error_m"]))

    yolo = np.array(by_condition["yolo_only"], dtype=float)
    slam = np.array(by_condition["slam_only"], dtype=float)
    fused = np.array(by_condition["fused"], dtype=float)

    plt.figure(figsize=(8, 4))
    plt.boxplot([yolo, slam, fused], tick_labels=["YOLO-only", "SLAM-only", "Fused"])
    plt.ylabel("Landing error (m)")
    plt.tight_layout()
    plt.savefig(out / "landing_accuracy_boxplot.png", dpi=300)

    means = [np.mean(yolo), np.mean(slam), np.mean(fused)]
    stds = [np.std(yolo), np.std(slam), np.std(fused)]
    plt.figure(figsize=(8, 4))
    plt.bar(["YOLO-only", "SLAM-only", "Fused"], means, yerr=stds, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("Mean landing error (m)")
    plt.tight_layout()
    plt.savefig(out / "ablation_bar_chart.png", dpi=300)


if __name__ == "__main__":
    main()
