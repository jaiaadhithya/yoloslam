import argparse
import subprocess


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-slam", action="store_true")
    parser.add_argument("--yolo-weights", default="models/yolov8_terrain.pt")
    parser.add_argument("--yolo-data", default="datasets/terrain_dataset/data.yaml")
    parser.add_argument("--slam-gt", default="")
    parser.add_argument("--slam-est", default="")
    args = parser.parse_args()

    if not args.skip_yolo:
        _run(
            [
                "python",
                "-m",
                "evaluation.yolo_standalone_eval",
                "--weights",
                args.yolo_weights,
                "--data",
                args.yolo_data,
                "--results-dir",
                f"{args.results_dir}/yolo_standalone",
            ]
        )

    if not args.skip_slam:
        if not args.slam_gt or not args.slam_est:
            raise ValueError("Provide --slam-gt and --slam-est or pass --skip-slam.")
        _run(
            [
                "python",
                "-m",
                "evaluation.slam_standalone_eval",
                "--gt",
                args.slam_gt,
                "--est",
                args.slam_est,
                "--results-dir",
                f"{args.results_dir}/slam_standalone",
            ]
        )

    _run(
        [
            "python",
            "-m",
            "evaluation.run_trials",
            "--num-trials",
            str(args.num_trials),
            "--results-dir",
            args.results_dir,
        ]
    )
    _run(["python", "-m", "evaluation.plot_results", "--results-dir", args.results_dir])
    print(f"Paper pipeline complete. Artifacts in {args.results_dir}")


if __name__ == "__main__":
    main()
