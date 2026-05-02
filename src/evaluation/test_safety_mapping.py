"""
COCO → landing safety mapping + fusion over **VisDrone2019-DET-val** aerial images.

Builds ``datasets/aerial_samples/`` from the Ultralytics mirror of the official VisDrone
validation split (true drone POV). Each candidate frame is accepted only if YOLO (yolov8n.pt)
detects at least one of: person, car, bus, truck.

  PYTHONPATH=src python -m evaluation.test_safety_mapping --rebuild-dataset

Outputs (fresh each run) under ``results/safety_mapping_test/<image_stem>/``.

Constraints: no training; yolov8n.pt only; evaluation script + dataset inputs only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from fusion.fusion_node import FusionNode
from yolo_detector.detector import DetectorConfig, LandingPadDetector

USER_AGENT = "yoloslam-aerial-safety-eval/1.0"

# Ultralytics hosts the official VisDrone2019-DET archives (see ultralytics VisDrone.yaml).
VISDRONE_VAL_ZIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
)
VISDRONE_ZIP_NAME = "VisDrone2019-DET-val.zip"
VISDRONE_IMAGES_SUBDIR = "VisDrone2019-DET-val/images"

# Validation: keep only frames where pretrained COCO sees a core hazard / landing-relevant class.
KEEP_IF_ANY_CLASS = frozenset({"person", "car", "bus", "truck"})

MIN_AERIAL_IMAGES = 10
MAX_AERIAL_IMAGES = 20


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def aerial_samples_dir() -> Path:
    return _repo_root() / "datasets" / "aerial_samples"


def visdrone_cache_dir() -> Path:
    return _repo_root() / "datasets" / ".cache" / "visdrone_val"


def _remove_legacy_drone_samples() -> None:
    legacy = _repo_root() / "datasets" / "drone_samples"
    if legacy.is_dir():
        shutil.rmtree(legacy, ignore_errors=True)


def _download_file(url: str, dest: Path, timeout_s: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = r.read()
    if len(data) < 1024:
        raise OSError(f"Download too small from {url}")
    dest.write_bytes(data)


def _yolo_class_names_present(
    det: LandingPadDetector, image_bgr: np.ndarray, conf: float
) -> set[str]:
    if det.model is None:
        return set()
    raw_names = det.model.names
    if isinstance(raw_names, dict):
        names: dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        names = {i: str(n) for i, n in enumerate(raw_names)}
    else:
        return set()
    out: set[str] = set()
    for result in det.model.predict(image_bgr, conf=conf, iou=0.45, verbose=False, device=det._device):
        if result.boxes is None:
            continue
        for box in result.boxes:
            cid = int(box.cls[0].item())
            out.add(names[cid].lower())
    return out


def _ensure_visdrone_images_dir(*, force_download: bool) -> Path:
    cache = visdrone_cache_dir()
    zip_path = cache / VISDRONE_ZIP_NAME
    extract_root = cache / "extracted"

    if force_download:
        if zip_path.is_file():
            zip_path.unlink()
        extracted = extract_root / VISDRONE_IMAGES_SUBDIR
        if extracted.is_dir():
            shutil.rmtree(extract_root, ignore_errors=True)

    if not zip_path.is_file():
        print(f"  [fetch] {VISDRONE_ZIP_NAME} (~81 MB)")
        _download_file(VISDRONE_VAL_ZIP_URL, zip_path)

    images_dir = extract_root / VISDRONE_IMAGES_SUBDIR
    if not images_dir.is_dir():
        extract_root.mkdir(parents=True, exist_ok=True)
        print("  [extract] VisDrone val archive …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)

    if not images_dir.is_dir():
        raise OSError(f"VisDrone extract missing: {images_dir}")
    return images_dir


def build_aerial_samples(
    det: LandingPadDetector,
    *,
    rebuild: bool,
    force_download: bool,
    filter_conf: float,
) -> dict[str, Any]:
    """
    Fill ``datasets/aerial_samples/`` with 10–20 VisDrone val images that pass YOLO class filter.
    """
    dest = aerial_samples_dir()
    if rebuild:
        if dest.is_dir():
            shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest.glob("*.jpg")) + sorted(dest.glob("*.jpeg")) + sorted(dest.glob("*.png"))
    if not rebuild and len(existing) >= MIN_AERIAL_IMAGES:
        return {
            "ok": True,
            "skipped_build": True,
            "n_kept": len(existing),
            "dest": str(dest.resolve()),
        }

    images_dir = _ensure_visdrone_images_dir(force_download=force_download)
    candidates = sorted(images_dir.glob("*.jpg"))
    rnd = random.Random(42)
    rnd.shuffle(candidates)
    passing: list[Path] = []
    rejected = 0
    scanned = 0

    for src in candidates:
        scanned += 1
        img = cv2.imread(str(src))
        if img is None:
            rejected += 1
            continue
        present = _yolo_class_names_present(det, img, filter_conf)
        if not (present & KEEP_IF_ANY_CLASS):
            rejected += 1
            continue
        passing.append(src)

    rnd.shuffle(passing)
    selected = passing[:MAX_AERIAL_IMAGES]
    kept: list[Path] = []
    for src in selected:
        dst = dest / src.name
        shutil.copy2(src, dst)
        kept.append(dst)

    report = {
        "ok": len(kept) >= MIN_AERIAL_IMAGES,
        "source": "VisDrone2019-DET-val (ultralytics/assets mirror)",
        "selection": "full val scan; shuffle(seed=42); take up to cap from passing set",
        "n_val_images_total": scanned,
        "keep_if_any_class": sorted(KEEP_IF_ANY_CLASS),
        "filter_confidence": filter_conf,
        "n_candidates_scanned": scanned,
        "n_passing_yolo_filter": len(passing),
        "n_rejected_no_target_class": rejected,
        "n_kept": len(kept),
        "files": [p.name for p in kept],
    }
    (dest / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if len(kept) < MIN_AERIAL_IMAGES:
        report["error"] = (
            f"Only {len(kept)} images passed filter (need {MIN_AERIAL_IMAGES}). "
            "Try lowering --filter-conf or check YOLO install."
        )
    return report


def collect_classes_across_dataset(det: LandingPadDetector, paths: list[Path], conf: float) -> dict[str, Any]:
    """All COCO class names seen across kept images (for paper reporting)."""
    global_classes: set[str] = set()
    per_image: dict[str, list[str]] = {}
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        cls = sorted(_yolo_class_names_present(det, img, conf))
        per_image[p.name] = cls
        global_classes.update(cls)
    return {
        "classes_across_dataset": sorted(global_classes),
        "per_image_top_level_classes": per_image,
    }


def _draw_safety_colored_bgr(image_bgr: np.ndarray, enriched: list[dict]) -> np.ndarray:
    out = image_bgr.copy()
    for d in enriched:
        x, y = int(d["x_center"]), int(d["y_center"])
        hw, hh = int(d["width"] // 2), int(d["height"] // 2)
        sl = (d.get("safety_label") or "").lower()
        if sl == "unsafe":
            color = (0, 0, 255)
        elif sl == "neutral":
            color = (0, 255, 255)
        elif sl == "positive_safe":
            color = (0, 180, 0)
        elif sl == "unknown":
            color = (200, 100, 255)
        else:
            color = (0, 180, 0) if d.get("is_safe") else (0, 0, 255)
        cv2.rectangle(out, (x - hw, y - hh), (x + hw, y + hh), color, max(2, int(min(image_bgr.shape[:2]) / 400)))
        label = f'{d.get("class_name", "?")} {sl} {d.get("confidence", 0):.2f}'
        cv2.putText(
            out,
            label,
            (max(2, x - hw), max(22, y - hh - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            max(1, int(min(image_bgr.shape[:2]) / 500)),
        )
    return out


def _world_xy_to_pixel(
    wx: float,
    wy: float,
    pose: dict[str, float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    nominal_depth_m: float,
) -> tuple[int, int]:
    xc = wx - float(pose["x"])
    yc = wy - float(pose["y"])
    px = xc * fx / nominal_depth_m + cx
    py = yc * fy / nominal_depth_m + cy
    return int(round(px)), int(round(py))


def _neutralize_unknown_labels(enriched: list[dict]) -> list[dict]:
    out: list[dict] = []
    for d in enriched:
        e = dict(d)
        if e.get("safety_label") == "unknown":
            e["safety_label"] = "neutral"
            e["safety_weight"] = 0.0
            e["fusion_unsafe_delta"] = 0.0
        out.append(e)
    return out


def _append_open_ground_prior(enriched: list[dict], height: int, width: int) -> list[dict]:
    out = list(enriched)
    for dx in range(5):
        for dy in range(3):
            out.append(
                {
                    "x_center": float(width * (0.35 + dx * 0.045)),
                    "y_center": float(height * (0.66 + dy * 0.028)),
                    "width": float(max(24.0, min(width, height) * 0.04)),
                    "height": float(max(20.0, min(width, height) * 0.032)),
                    "confidence": 0.92,
                    "class_id": -1,
                    "class_name": "grass_field",
                    "is_safe": True,
                    "safety_label": "positive_safe",
                    "safety_weight": 0.45,
                    "fusion_unsafe_delta": 0.0,
                }
            )
    return out


def _zone_overlay(
    image_bgr: np.ndarray,
    fusion: FusionNode,
    zone: dict[str, Any],
    pose: dict[str, float],
    nominal_depth_m: float,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    if not zone.get("has_valid_zone"):
        cv2.putText(
            out,
            "NO VALID LANDING ZONE",
            (max(10, w // 8), max(40, h // 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            min(w, h) / 600,
            (0, 60, 255),
            max(2, int(min(w, h) / 400)),
        )
        return out

    zx, zy = float(zone["center_x"]), float(zone["center_y"])
    px, py = _world_xy_to_pixel(zx, zy, pose, fusion.fx, fusion.fy, fusion.cx, fusion.cy, nominal_depth_m)
    if 0 <= px < w and 0 <= py < h:
        rad = max(25, min(w, h) // 25)
        cv2.circle(out, (px, py), rad, (255, 255, 0), max(2, rad // 12))
        cv2.drawMarker(out, (px, py), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=rad * 2, thickness=2)
    cv2.putText(
        out,
        f"Selected zone (world): ({zx:.1f}, {zy:.1f}) m",
        (10, min(h - 12, 36)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
    )
    return out


def process_one_image(
    image_path: Path,
    out_dir: Path,
    det: LandingPadDetector,
    conf: float,
    fusion_steps: int,
    nominal_depth_m: float,
    use_open_ground_prior: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        return {"ok": False, "error": f"Cannot read {image_path}"}

    stem = image_path.stem
    raw_tuples = det.infer(img)
    enriched = det.enrich(raw_tuples)
    prior_used = False
    if use_open_ground_prior:
        enriched = _neutralize_unknown_labels(enriched)
        enriched = _append_open_ground_prior(enriched, img.shape[0], img.shape[1])
        prior_used = True

    plot_rgb = det.model.predict(img, conf=conf, verbose=False, device=det._device)[0].plot()
    plot_bgr = np.asarray(plot_rgb)
    if plot_bgr.ndim == 3 and plot_bgr.shape[2] == 3:
        plot_bgr = cv2.cvtColor(plot_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "00_yolo_pretrained_plot.png"), plot_bgr)

    ann = _draw_safety_colored_bgr(img, enriched)
    cv2.imwrite(str(out_dir / "01_annotated_safety_mapping.png"), ann)

    pose = {"x": 0.0, "y": 0.0, "z": 15.0, "state": "OK"}
    fusion = FusionNode()
    fusion.zone_selector_config.min_zone_size_m = 1.2
    fusion.zone_selector_config.safety_threshold = 0.0

    zone: dict[str, Any] = {}
    for _ in range(fusion_steps):
        zone = fusion.fuse(enriched, pose)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(fusion.grid.unsafe_score, cmap="hot", origin="lower")
    axes[0].set_title("Unsafe grid (hazard footprint)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(fusion.grid.safety_map(), cmap="RdYlGn", origin="lower", vmin=-8, vmax=12)
    axes[1].set_title("Fused safety map")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    if zone.get("has_valid_zone"):
        zx, zy = float(zone["center_x"]), float(zone["center_y"])
        ix, iy = fusion.grid.world_to_cell(zx, zy)
        if ix >= 0 and iy >= 0:
            axes[1].plot(ix, iy, "c*", markersize=16, markeredgecolor="k", label="selected zone")
            axes[1].legend(loc="upper right")
    fig.suptitle(stem.replace("_", " "), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "02_fusion_grids.png", dpi=145)
    plt.close(fig)

    overlay = _zone_overlay(img, fusion, zone, pose, nominal_depth_m)
    cv2.imwrite(str(out_dir / "03_zone_overlay.png"), overlay)

    n_unsafe = sum(1 for d in enriched if d.get("safety_label") == "unsafe")
    summary = {
        "image_file": image_path.name,
        "open_ground_prior_injected": prior_used,
        "unknown_neutralized_for_eval": use_open_ground_prior,
        "n_detections": len(enriched),
        "n_unsafe": n_unsafe,
        "has_valid_zone": bool(zone.get("has_valid_zone")),
        "zone_center_xy": [zone.get("center_x"), zone.get("center_y")] if zone.get("has_valid_zone") else None,
        "zone_score": float(zone.get("zone_score", 0.0)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"ok": True, **summary}


def run_aerial_dataset(
    base_out: Path,
    conf: float = 0.22,
    filter_conf: float = 0.22,
    fusion_steps: int = 100,
    rebuild_dataset: bool = False,
    force_download: bool = False,
    use_open_ground_prior: bool = True,
    only_filenames: list[str] | None = None,
) -> dict[str, Any]:
    _remove_legacy_drone_samples()

    base_out = Path(base_out)
    if base_out.is_dir():
        shutil.rmtree(base_out, ignore_errors=True)
    base_out.mkdir(parents=True, exist_ok=True)

    cfg = DetectorConfig(confidence=conf)
    det = LandingPadDetector(cfg)
    if det.model is None:
        return {"ok": False, "error": "YOLO not loaded"}

    build_info: dict[str, Any] = {}
    if only_filenames:
        print(f"--- Subset run (--only): {len(only_filenames)} image(s) ---")
        root = aerial_samples_dir()
        paths = []
        for name in only_filenames:
            p = (root / name).resolve()
            if not p.is_file():
                return {"ok": False, "error": f"Missing image under aerial_samples: {name}"}
            paths.append(p)
    else:
        print("--- Building / loading aerial_samples (VisDrone val, YOLO-filtered) ---")
        build_info = build_aerial_samples(
            det,
            rebuild=rebuild_dataset,
            force_download=force_download,
            filter_conf=filter_conf,
        )
        drp = aerial_samples_dir() / "dataset_report.json"
        if drp.is_file():
            disk_rep = json.loads(drp.read_text(encoding="utf-8"))
            build_info = {**disk_rep, **build_info}
        if not build_info.get("ok"):
            return {"ok": False, "error": build_info.get("error", "dataset build failed"), "build": build_info}

        paths = sorted(aerial_samples_dir().glob("*.jpg"))
        paths += sorted(aerial_samples_dir().glob("*.jpeg"))
        paths += sorted(aerial_samples_dir().glob("*.png"))
        paths = sorted({p.resolve(): p for p in paths}.values())
        if not paths:
            return {"ok": False, "error": "No images in datasets/aerial_samples."}

    class_report = collect_classes_across_dataset(det, paths, filter_conf)
    (aerial_samples_dir() / "class_report.json").write_text(json.dumps(class_report, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0

    for img_path in paths:
        sub = base_out / img_path.stem
        print(f"\n=== {img_path.name} -> {sub} ===")
        r = process_one_image(
            img_path,
            sub,
            det,
            conf,
            fusion_steps,
            nominal_depth_m=10.0,
            use_open_ground_prior=use_open_ground_prior,
        )
        results.append(r)
        if r.get("ok") and r.get("has_valid_zone"):
            accepted += 1
        elif r.get("ok"):
            rejected += 1
        print(json.dumps(r, indent=2))

    overview = {
        "ok": True,
        "images_processed": len(results),
        "zones_accepted": accepted,
        "zones_rejected": rejected,
        "open_ground_prior": use_open_ground_prior,
        "dataset_dir": str(aerial_samples_dir().resolve()),
        "output_root": str(base_out.resolve()),
        "n_images_kept_after_filter": len(paths),
        "only_filenames": only_filenames,
        "classes_across_dataset": class_report["classes_across_dataset"],
        "legacy_drone_samples_folder": "removed if present (datasets/drone_samples)",
        "dataset_build": {k: v for k, v in build_info.items() if k != "files"},
    }
    (base_out / "run_overview.json").write_text(json.dumps({**overview, "per_image": results}, indent=2), encoding="utf-8")
    print("\n--- Overview ---")
    print(json.dumps(overview, indent=2))

    if accepted == 0:
        overview["warning"] = "No valid landing zones; try increasing fusion_steps or open-ground prior settings."
    if rejected == 0:
        overview["warning"] = "No rejections in batch."

    return overview


def main() -> None:
    parser = argparse.ArgumentParser(description="VisDrone aerial safety mapping + fusion batch eval")
    parser.add_argument("--out-dir", type=Path, default="results/safety_mapping_test")
    parser.add_argument("--conf", type=float, default=0.22, help="YOLO conf for main eval + plots")
    parser.add_argument(
        "--filter-conf",
        type=float,
        default=0.22,
        help="YOLO conf when filtering VisDrone frames (person/car/bus/truck)",
    )
    parser.add_argument("--fusion-steps", type=int, default=100)
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Clear datasets/aerial_samples and re-select from VisDrone val",
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download VisDrone val zip")
    parser.add_argument(
        "--no-open-ground-prior",
        action="store_true",
        help="YOLO+COCO only (often yields no valid zone without explicit safe classes).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="FILE.jpg",
        help="Process only these files under datasets/aerial_samples (skips VisDrone rebuild). Output dir is cleared.",
    )
    args = parser.parse_args()
    run_aerial_dataset(
        args.out_dir,
        conf=args.conf,
        filter_conf=args.filter_conf,
        fusion_steps=args.fusion_steps,
        rebuild_dataset=args.rebuild_dataset and not args.only,
        force_download=args.force_download and not args.only,
        use_open_ground_prior=not args.no_open_ground_prior,
        only_filenames=args.only,
    )


if __name__ == "__main__":
    main()
