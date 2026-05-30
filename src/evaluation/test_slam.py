"""
Synthetic SLAM trajectory eval: realistic UAV path (smooth survey + descent + optional glitch).

Writes ``results/slam_test/trajectory.csv`` and paper-style figures (XY, altitude vs time, 3D).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

os.environ.setdefault("MPLBACKEND", "Agg")

STEPS = 320
FRAME_SEED = 42
DT = 0.05


def run_slam_smoke(
    out_dir: Path,
    num_steps: int = STEPS,
    frame_seed: int = FRAME_SEED,
    dt: float = DT,
    enable_glitch: bool = True,
) -> dict[str, Any]:
    from slam_module.slam_wrapper import SlamWrapper
    from slam_module.synthetic_trajectory import precompute_trajectory

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(frame_seed)
    slam = SlamWrapper(use_synthetic=True, dt=dt, traj_seed=frame_seed, enable_glitch=enable_glitch)

    positions: list[np.ndarray] = []
    t0_wall = time.perf_counter()
    for i in range(num_steps):
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        pose = slam.track(frame, timestamp=float(i) * dt)
        positions.append(pose.position.copy())
    elapsed = time.perf_counter() - t0_wall

    traj = precompute_trajectory(
        num_steps, dt=dt, seed=frame_seed, t0=0.0, enable_glitch=enable_glitch
    )
    positions_arr = np.stack(positions, axis=0)
    if not np.allclose(positions_arr, np.column_stack([traj["x"], traj["y"], traj["z"]]), rtol=1e-5, atol=1e-4):
        raise RuntimeError("Wrapper trajectory diverged from precompute_trajectory (logic bug).")

    csv_path = out_dir / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
        for i in range(num_steps):
            w.writerow(
                [
                    f"{traj['timestamp'][i]:.6f}",
                    f"{traj['x'][i]:.6f}",
                    f"{traj['y'][i]:.6f}",
                    f"{traj['z'][i]:.6f}",
                    f"{traj['vx'][i]:.6f}",
                    f"{traj['vy'][i]:.6f}",
                    f"{traj['vz'][i]:.6f}",
                ]
            )

    ts = traj["timestamp"]
    x, y, z = traj["x"], traj["y"], traj["z"]

    # --- (a) Top-down XY ---
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(x, y, "-", color="#1f4e79", linewidth=2.0, alpha=0.92, label="Estimated path")
    ax.scatter([x[0]], [y[0]], c="#2ca02c", s=36, zorder=5, label="Start")
    ax.scatter([x[-1]], [y[-1]], c="#d62728", s=36, zorder=5, label="End")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, lateral)")
    ax.set_title("Synthetic SLAM trajectory (top-down)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    xy_path = out_dir / "trajectory_xy.png"
    fig.savefig(xy_path, dpi=200)
    plt.close(fig)

    # --- (b) Altitude / z vs time ---
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(ts, z, color="#1f4e79", linewidth=1.8, label="z (up axis)")
    if enable_glitch:
        g0, g1 = 118 * dt, 128 * dt
        ax.axvspan(g0, g1, alpha=0.18, color="tab:orange", label="Tracking offset (hold)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z (m)")
    ax.set_title("Altitude profile (descent toward landing)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    alt_path = out_dir / "altitude_profile.png"
    fig.savefig(alt_path, dpi=200)
    plt.close(fig)

    # --- (c) 3D ---
    fig = plt.figure(figsize=(6.5, 5.2))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot(x, y, z, color="#1f4e79", linewidth=1.6)
    ax3.scatter([x[0]], [y[0]], [z[0]], c="green", s=28, depthshade=True)
    ax3.scatter([x[-1]], [y[-1]], [z[-1]], c="red", s=28, depthshade=True)
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_title("3D trajectory (survey + descent)")
    fig.tight_layout()
    d3_path = out_dir / "trajectory_3d.png"
    fig.savefig(d3_path, dpi=200)
    plt.close(fig)

    forward_m = float(x[-1] - x[0])
    descent_m = float(z[0] - z[-1])
    path_len = float(np.sum(np.linalg.norm(np.diff(positions_arr, axis=0), axis=1)))

    return {
        "ok": True,
        "num_steps": num_steps,
        "dt_s": dt,
        "frame_seed": frame_seed,
        "enable_glitch": enable_glitch,
        "runtime_s": elapsed,
        "forward_progress_m": forward_m,
        "total_descent_m": descent_m,
        "path_length_m": path_len,
        "final_position": positions_arr[-1].tolist(),
        "trajectory_csv": str(csv_path.resolve()),
        "trajectory_xy": str(xy_path.resolve()),
        "altitude_profile": str(alt_path.resolve()),
        "trajectory_3d": str(d3_path.resolve()),
        "plot": str(xy_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic SLAM trajectory + paper figures")
    parser.add_argument("--out-dir", default="results/slam_test", type=Path)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=FRAME_SEED)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--no-glitch", action="store_true", help="Disable tracking-offset event")
    args = parser.parse_args()
    print(
        run_slam_smoke(
            args.out_dir,
            num_steps=args.steps,
            frame_seed=args.seed,
            dt=args.dt,
            enable_glitch=not args.no_glitch,
        )
    )


if __name__ == "__main__":
    main()
