import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def _write_per_class_table(names: dict, metrics, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    p = metrics.box.p.tolist() if hasattr(metrics.box, "p") else []
    r = metrics.box.r.tolist() if hasattr(metrics.box, "r") else []
    f1 = metrics.box.f1.tolist() if hasattr(metrics.box, "f1") else []
    ap50 = metrics.box.ap50.tolist() if hasattr(metrics.box, "ap50") else []
    ap = metrics.box.ap.tolist() if hasattr(metrics.box, "ap") else []
    rows = []
    for class_id, class_name in names.items():
        idx = int(class_id)
        rows.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "precision": float(p[idx]) if idx < len(p) else 0.0,
                "recall": float(r[idx]) if idx < len(r) else 0.0,
                "f1": float(f1[idx]) if idx < len(f1) else 0.0,
                "ap50": float(ap50[idx]) if idx < len(ap50) else 0.0,
                "ap50_95": float(ap[idx]) if idx < len(ap) else 0.0,
            }
        )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["class_id", "class_name"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _write_summary(metrics, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map50": float(getattr(metrics.box, "map50", 0.0)),
        "map50_95": float(getattr(metrics.box, "map", 0.0)),
        "mp": float(getattr(metrics.box, "mp", 0.0)),
        "mr": float(getattr(metrics.box, "mr", 0.0)),
        "speed_ms": dict(getattr(metrics, "speed", {})),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_reliability_plot(per_class_csv: Path, out_png: Path) -> None:
    rows = []
    with per_class_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    names = [r["class_name"] for r in rows]
    recalls = [float(r["recall"]) for r in rows]
    precisions = [float(r["precision"]) for r in rows]
    x = np.arange(len(names))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, precisions, width=0.4, label="Precision")
    plt.bar(x + 0.2, recalls, width=0.4, label="Recall")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolov8_terrain.pt")
    parser.add_argument("--data", default="datasets/terrain_dataset/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results-dir", default="results/yolo_standalone")
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split, plots=True, save_json=True)

    per_class_csv = out_dir / "per_class_metrics.csv"
    summary_json = out_dir / "summary_metrics.json"
    reliability_png = out_dir / "precision_recall_by_class.png"
    _write_per_class_table(metrics.names, metrics, per_class_csv)
    _write_summary(metrics, summary_json)
    _write_reliability_plot(per_class_csv, reliability_png)
    print(f"YOLO standalone artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
