import argparse
import json
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _extract_evo_stats(zip_path: Path) -> dict:
    # Keep parser lightweight: stats can be read from evo_res output later.
    return {"result_zip": str(zip_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth trajectory file")
    parser.add_argument("--est", required=True, help="Estimated trajectory file")
    parser.add_argument("--format", default="tum", choices=["tum", "kitti", "euroc"])
    parser.add_argument("--results-dir", default="results/slam_standalone")
    args = parser.parse_args()

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    ate_zip = out / "ate.zip"
    rpe_zip = out / "rpe.zip"

    _run(
        [
            "evo_ape",
            args.format,
            args.gt,
            args.est,
            "-vas",
            "--plot",
            "--plot_mode",
            "xy",
            "--save_results",
            str(ate_zip),
        ]
    )
    _run(
        [
            "evo_rpe",
            args.format,
            args.gt,
            args.est,
            "--delta",
            "1",
            "--delta_unit",
            "m",
            "-vas",
            "--plot",
            "--save_results",
            str(rpe_zip),
        ]
    )

    summary = {
        "ate": _extract_evo_stats(ate_zip),
        "rpe": _extract_evo_stats(rpe_zip),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SLAM standalone artifacts written to {out}")


if __name__ == "__main__":
    main()
