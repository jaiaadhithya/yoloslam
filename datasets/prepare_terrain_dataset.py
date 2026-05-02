import argparse
import shutil
from pathlib import Path

import yaml


CLASS_NAMES = [
    "flat_ground",
    "water",
    "tree_canopy",
    "building_structure",
    "vehicle",
    "person",
    "debris_clutter",
    "road_surface",
    "grass_field",
    "rooftop_flat",
    "fence_pole",
]


def _write_data_yaml(dataset_root: Path) -> None:
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    with (dataset_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _ensure_layout(dataset_root: Path) -> None:
    for split in ["train", "val"]:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="datasets/terrain_dataset")
    parser.add_argument(
        "--semantic-drone-root",
        default="datasets/SemanticDroneDataset",
        help="If present, conversion can be added later; for now this script prepares target layout.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_layout(out_dir)
    _write_data_yaml(out_dir)

    # Optional bootstrap: reuse any already-labeled terrain dataset if provided.
    src_root = Path(args.semantic_drone_root)
    copied = False
    copied |= _copy_if_exists(src_root / "images", out_dir / "images")
    copied |= _copy_if_exists(src_root / "labels", out_dir / "labels")
    _ensure_layout(out_dir)
    _write_data_yaml(out_dir)

    if copied:
        print(f"Prepared terrain dataset from existing source at {src_root}")
    else:
        print(f"Prepared empty terrain dataset scaffold at {out_dir}")


if __name__ == "__main__":
    main()
