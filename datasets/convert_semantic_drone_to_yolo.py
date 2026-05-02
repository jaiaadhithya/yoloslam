import argparse
import json
from pathlib import Path

import cv2
import numpy as np


TARGET_CLASS_TO_ID = {
    "flat_ground": 0,
    "water": 1,
    "tree_canopy": 2,
    "building_structure": 3,
    "vehicle": 4,
    "person": 5,
    "debris_clutter": 6,
    "road_surface": 7,
    "grass_field": 8,
    "rooftop_flat": 9,
    "fence_pole": 10,
}


def _bbox_to_yolo(x: int, y: int, w: int, h: int, iw: int, ih: int) -> tuple[float, float, float, float]:
    return ((x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih)


def _mask_for_rgb(mask_bgr: np.ndarray, rgb: list[int]) -> np.ndarray:
    bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
    return cv2.inRange(mask_bgr, bgr, bgr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--out-dir", default="datasets/terrain_dataset")
    parser.add_argument(
        "--mapping-json",
        required=True,
        help="JSON dict: source_class -> {'target':'grass_field','color_rgb':[R,G,B]}",
    )
    parser.add_argument("--min-area-px", type=int, default=120)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    mapping = json.loads(Path(args.mapping_json).read_text(encoding="utf-8"))

    paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    n_val = int(len(paths) * args.val_ratio)
    val_set = set(p.name for p in paths[:n_val])

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_path in paths:
        mask_path = masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if img is None or mask is None:
            continue
        ih, iw = img.shape[:2]
        split = "val" if img_path.name in val_set else "train"
        out_img = out_dir / "images" / split / img_path.name
        out_lbl = out_dir / "labels" / split / f"{img_path.stem}.txt"
        cv2.imwrite(str(out_img), img)

        lines: list[str] = []
        for spec in mapping.values():
            target = spec["target"]
            color_rgb = spec["color_rgb"]
            if target not in TARGET_CLASS_TO_ID:
                continue
            cls_id = TARGET_CLASS_TO_ID[target]
            binary = _mask_for_rgb(mask, color_rgb)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if area < args.min_area_px:
                    continue
                xc, yc, nw, nh = _bbox_to_yolo(int(x), int(y), int(w), int(h), iw, ih)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted += 1

    print(f"Converted {converted} images into YOLO labels at {out_dir}")


if __name__ == "__main__":
    main()
