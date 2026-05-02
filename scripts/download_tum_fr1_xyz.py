import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


TUM_URL = "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"


def _download(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 1024:
        print(f"Archive already exists: {out_file}")
        return
    print(f"Downloading {url} -> {out_file}")
    urlretrieve(url, out_file)  # nosec - dataset source URL


def _extract(archive: Path, out_dir: Path) -> Path:
    seq_dir = out_dir / "rgbd_dataset_freiburg1_xyz"
    if seq_dir.exists():
        print(f"Sequence already extracted: {seq_dir}")
        return seq_dir
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(out_dir)
    print(f"Extracted to {seq_dir}")
    return seq_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/tum_rgbd")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    archive = dataset_dir / "rgbd_dataset_freiburg1_xyz.tgz"
    _download(TUM_URL, archive)
    _extract(archive, dataset_dir)
    print("Next: run ORB-SLAM3 mono_tum or rgbd_tum and export estimated trajectory in TUM format.")


if __name__ == "__main__":
    main()
