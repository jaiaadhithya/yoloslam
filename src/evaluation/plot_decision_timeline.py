"""Plot state machine timeline, altitude, and zone score from state_log.csv."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


STATE_ORDER = [
    "SURVEY",
    "EVALUATE",
    "SEARCH",
    "APPROACH",
    "ALIGN",
    "DESCEND",
    "ABORT",
    "LANDED",
]
STATE_TO_Y = {s: float(i) for i, s in enumerate(STATE_ORDER)}


def plot_decision_timeline(
    state_log_csv: str | Path,
    output_png: str | Path,
) -> None:
    path = Path(state_log_csv)
    times: list[float] = []
    states: list[str] = []
    altitudes: list[float] = []
    scores: list[float] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            states.append(row["state"])
            altitudes.append(float(row["altitude_m"]))
            scores.append(float(row["zone_score"]))

    if not times:
        raise ValueError(f"No rows in {path}")

    y_numeric = [STATE_TO_Y.get(s, 0.0) for s in states]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(times, y_numeric, drawstyle="steps-post", color="C0", linewidth=1.2)
    axes[0].set_yticks(list(range(len(STATE_ORDER))))
    axes[0].set_yticklabels(STATE_ORDER)
    axes[0].set_ylabel("State")
    axes[0].set_title("Landing decision timeline")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, altitudes, color="C1", linewidth=1.2)
    axes[1].set_ylabel("Altitude (m)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, scores, color="C2", linewidth=1.2)
    axes[2].set_ylabel("Zone score")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decision timeline from state_log.csv")
    parser.add_argument("--state-log", default="results/full_pipeline/state_log.csv")
    parser.add_argument("--output", default="results/full_pipeline/decision_timeline.png")
    args = parser.parse_args()
    plot_decision_timeline(args.state_log, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
