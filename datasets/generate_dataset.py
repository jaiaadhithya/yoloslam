import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="datasets/terrain_dataset")
    parser.add_argument("--num-images", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out_dir)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_defs = [
        ("flat_ground", (170, 170, 170), 0),
        ("water", (220, 60, 30), 1),
        ("tree_canopy", (40, 130, 40), 2),
        ("building_structure", (120, 120, 120), 3),
        ("vehicle", (40, 40, 180), 4),
        ("person", (20, 20, 230), 5),
        ("debris_clutter", (60, 80, 150), 6),
        ("road_surface", (90, 90, 90), 7),
        ("grass_field", (70, 180, 70), 8),
        ("rooftop_flat", (150, 150, 190), 9),
        ("fence_pole", (60, 130, 180), 10),
    ]

    rng = np.random.default_rng(123)
    n_train = int(args.num_images * 0.8)
    for i in range(args.num_images):
        split = "train" if i < n_train else "val"
        img = np.full((480, 640, 3), 170, dtype=np.uint8)

        img_path = out / "images" / split / f"{i:05d}.jpg"
        lbl_path = out / "labels" / split / f"{i:05d}.txt"
        label_lines: list[str] = []
        n_objs = int(rng.integers(3, 7))
        for _ in range(n_objs):
            cls_name, color, cls_id = class_defs[int(rng.integers(0, len(class_defs)))]
            _ = cls_name
            cx = int(rng.integers(50, 590))
            cy = int(rng.integers(50, 430))
            w = int(rng.integers(20, 200))
            h = int(rng.integers(20, 180))
            cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, -1)
            x = cx / 640.0
            y = cy / 480.0
            nw = w / 640.0
            nh = h / 480.0
            label_lines.append(f"{cls_id} {x:.6f} {y:.6f} {nw:.6f} {nh:.6f}")

        cv2.imwrite(str(img_path), img)
        lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    print(f"Generated synthetic dataset in {out}")


if __name__ == "__main__":
    main()
