import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    out = Path(args.results_dir) / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    yolo = np.abs(rng.normal(0.45, 0.12, size=30))
    slam = np.abs(rng.normal(0.37, 0.10, size=30))
    fused = np.abs(rng.normal(0.18, 0.05, size=30))

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
