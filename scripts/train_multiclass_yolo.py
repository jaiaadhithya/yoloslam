import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seg", action="store_true", help="Train segmentation variant")
    parser.add_argument("--name", default="terrain_multiclass")
    args = parser.parse_args()

    base = "yolov8n-seg.pt" if args.seg else "yolov8n.pt"
    model = YOLO(base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)

    run_subdir = "segment" if args.seg else "detect"
    best = Path("runs") / run_subdir / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected best model at: {best}")

    out = Path("models") / "yolov8_terrain_multiclass.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(best.read_bytes())
    print(f"Saved trained weights to {out}")


if __name__ == "__main__":
    main()
