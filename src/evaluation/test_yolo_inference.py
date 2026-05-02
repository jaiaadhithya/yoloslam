"""
Smoke test: pretrained Ultralytics YOLO (yolov8n.pt) on real images with COCO-like content.

Downloads sample images, saves raw inputs, runs inference with a low confidence threshold,
prints raw detection counts, and saves side-by-side original | Ultralytics plot() output.
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

# Public demo images (people, vehicles, typical COCO categories).
SAMPLE_URLS: list[tuple[str, str]] = [
    ("https://ultralytics.com/images/bus.jpg", "bus.jpg"),
    ("https://ultralytics.com/images/zidane.jpg", "zidane.jpg"),
    ("https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg", "bus_assets.jpg"),
    ("https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg", "zidane_assets.jpg"),
    ("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", "bus_yolov5_data.jpg"),
    ("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg", "zidane_yolov5_data.jpg"),
]

SMOKE_CONFIDENCE = 0.1
USER_AGENT = "yoloslam-smoke-test/1.0 (evaluation; +https://github.com/ultralytics/ultralytics)"


def _download(url: str, dest: Path, timeout_s: float = 45.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        if len(data) < 2048:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _load_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return img


def _top_k_boxes(result: Any, k: int = 5) -> list[dict[str, Any]]:
    """Raw Ultralytics result — before project enrich()."""
    out: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes
    n = len(boxes)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    order = np.argsort(-confs.ravel())[:k]
    names = getattr(result, "names", None)
    for idx in order:
        cls_id = int(boxes.cls[idx].item())
        cf = float(boxes.conf[idx].item())
        xyxy = boxes.xyxy[idx].tolist()
        if isinstance(names, dict):
            name = names.get(cls_id, str(cls_id))
        elif isinstance(names, (list, tuple)) and cls_id < len(names):
            name = str(names[cls_id])
        else:
            name = str(cls_id)
        out.append(
            {
                "class_id": cls_id,
                "class_name": str(name),
                "confidence": cf,
                "xyxy": [float(x) for x in xyxy],
            }
        )
    return out


def _side_by_side(original_bgr: np.ndarray, plotted_bgr: np.ndarray) -> np.ndarray:
    h0, w0 = original_bgr.shape[:2]
    if plotted_bgr.shape[0] != h0 or plotted_bgr.shape[1] != w0:
        plotted_bgr = cv2.resize(plotted_bgr, (w0, h0), interpolation=cv2.INTER_LINEAR)
    gap = np.full((h0, 8, 3), 255, dtype=np.uint8)
    return np.hstack([original_bgr, gap, plotted_bgr])


def run_yolo_smoke(
    out_dir: Path,
    sample_seed: int = 2026,
    confidence: float = SMOKE_CONFIDENCE,
) -> dict[str, Any]:
    """
    Download real images, run YOLO with given confidence, save inputs + side-by-side annotated outputs.
    """
    from yolo_detector.detector import DetectorConfig, LandingPadDetector

    _ = sample_seed  # reserved for future augmentation; URLs are fixed

    out_dir = Path(out_dir)
    raw_dir = out_dir / "inputs_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = DetectorConfig(confidence=confidence)
    det = LandingPadDetector(cfg)

    device_info = getattr(det, "_device", "unknown")
    if det.model is None:
        return {
            "ok": False,
            "error": "YOLO model not loaded (ultralytics missing or load failed)",
            "device": str(device_info),
            "out_dir": str(out_dir),
            "images_written": 0,
        }

    # Collect downloaded paths
    image_paths: list[Path] = []
    print("--- Downloading / verifying sample images ---")
    for url, fname in SAMPLE_URLS:
        dest = raw_dir / fname
        if dest.is_file() and dest.stat().st_size > 2048:
            print(f"  [cache] {fname}")
            image_paths.append(dest)
            continue
        print(f"  [fetch] {fname} <- {url}")
        if _download(url, dest):
            image_paths.append(dest)
        else:
            print(f"  [skip] failed: {fname}")

    if not image_paths:
        return {
            "ok": False,
            "error": "No sample images could be downloaded. Check network or firewall.",
            "device": str(device_info),
            "out_dir": str(out_dir),
            "images_written": 0,
        }

    try:
        import torch

        cuda_avail = bool(torch.cuda.is_available())
    except ImportError:
        cuda_avail = False

    t0 = time.perf_counter()
    written = 0
    total_dets = 0
    log_lines: list[str] = []

    print("\n--- Per-image: input stats + raw YOLO (before enrich) ---")
    for i, path in enumerate(image_paths):
        img = _load_bgr(path)
        if img is None:
            print(f"  [{i}] ERROR: could not decode {path}")
            continue

        # Verify input is a real image
        mean_px = float(img.mean())
        print(
            f"  [{i}] {path.name}: shape={img.shape} dtype={img.dtype} "
            f"mean={mean_px:.1f} min={int(img.min())} max={int(img.max())}"
        )
        cv2.imwrite(str(out_dir / f"sample_{i:02d}_original.png"), img)

        # Raw Ultralytics predict (same conf as smoke test — no extra filtering here)
        results = det.model.predict(
            img,
            conf=confidence,
            iou=cfg.iou,
            verbose=False,
            device=det._device,
        )
        r0 = results[0]
        n_det = len(r0.boxes) if r0.boxes is not None else 0
        total_dets += n_det
        print(f"       raw detections count: {n_det}")

        top5 = _top_k_boxes(r0, k=5)
        for j, row in enumerate(top5):
            print(
                f"       top{j+1}: {row['class_name']} conf={row['confidence']:.3f} "
                f"xyxy={[round(x, 1) for x in row['xyxy']]}"
            )
        log_lines.append(f"{path.name}: n={n_det} top={top5[:3]}")

        # Built-in annotation (standard YOLO demo look); Ultralytics returns BGR uint8 for OpenCV workflows.
        plotted = np.asarray(r0.plot())
        if plotted is None or plotted.size == 0:
            plotted_bgr = img.copy()
        elif plotted.ndim == 3 and plotted.shape[2] == 3:
            plotted_bgr = plotted.astype(np.uint8, copy=False)
        else:
            plotted_bgr = img.copy()

        cv2.imwrite(str(out_dir / f"sample_{i:02d}_annotated_plot.png"), plotted_bgr)

        combo = _side_by_side(img, plotted_bgr)
        cv2.imwrite(str(out_dir / f"sample_{i:02d}_side_by_side.png"), combo)
        written += 1

    elapsed = time.perf_counter() - t0
    print(f"\n--- Done: {written} image(s), {total_dets} total raw detections, {elapsed:.3f}s inference wall ---")

    return {
        "ok": True,
        "device": str(device_info),
        "cuda_available": cuda_avail,
        "model_path": cfg.model_path,
        "confidence": confidence,
        "num_images": written,
        "total_raw_detections": total_dets,
        "inference_time_s": elapsed,
        "out_dir": str(out_dir.resolve()),
        "detection_log": log_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/smoke_test/yolo", type=Path)
    parser.add_argument("--conf", type=float, default=SMOKE_CONFIDENCE, help="YOLO confidence threshold (default 0.1)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    r = run_yolo_smoke(args.out_dir, sample_seed=args.seed, confidence=args.conf)
    print(r)


if __name__ == "__main__":
    main()
