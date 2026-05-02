import argparse
import csv
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve


MH01_URL = (
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/"
    "machine_hall/MH_01_easy/MH_01_easy.zip"
)


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Archive already exists: {out_path}")
        return
    print(f"Downloading {url} -> {out_path}")
    try:
        urlretrieve(url, out_path)  # nosec - trusted dataset host for this workflow
    except URLError as exc:
        raise RuntimeError(
            "EuRoC download failed (likely network/user-agent/proxy issue). "
            "Download MH_01_easy.zip manually via browser and place it at "
            f"{out_path} before rerunning this script."
        ) from exc


def _extract(archive: Path, out_dir: Path) -> Path:
    seq_dir = out_dir / "MH_01_easy"
    if seq_dir.exists():
        print(f"Sequence already extracted: {seq_dir}")
        return seq_dir
    print(f"Extracting {archive} -> {out_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out_dir)
    return seq_dir


def _ground_truth_csv(seq_dir: Path) -> Path:
    return seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"


def _write_tum_gt(csv_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        reader = csv.reader(f_in)
        _ = next(reader, None)  # header
        for row in reader:
            if len(row) < 8:
                continue
            t = float(row[0]) / 1e9
            px, py, pz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            f_out.write(f"{t:.9f} {px:.9f} {py:.9f} {pz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
    print(f"Wrote TUM ground truth to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/euroc")
    parser.add_argument("--results-dir", default="results/slam_standalone")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    archive = dataset_dir / "MH_01_easy.zip"
    _download(MH01_URL, archive)
    seq_dir = _extract(archive, dataset_dir)

    gt_csv = _ground_truth_csv(seq_dir)
    if not gt_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV missing: {gt_csv}")
    gt_tum = Path(args.results_dir) / "ground_truth.txt"
    _write_tum_gt(gt_csv, gt_tum)

    print("Next: run ORB-SLAM3 on MH_01_easy and export estimated.txt in TUM format.")


if __name__ == "__main__":
    main()
