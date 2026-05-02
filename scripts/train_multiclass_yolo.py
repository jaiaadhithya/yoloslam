import argparse
import os
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    # Torch >=2.6 defaults to weights_only=True, which breaks older Ultralytics checkpoints.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/terrain_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seg", action="store_true", help="Train segmentation variant")
    parser.add_argument("--name", default="terrain_multiclass")
    args = parser.parse_args()

    base = "yolov8n-seg.pt" if args.seg else "yolov8n.pt"
    model = YOLO(base)
    data_file = Path(args.data).resolve()
    data_cfg = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    data_cfg["path"] = str(data_file.parent.resolve()).replace("\\", "/")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        yaml.safe_dump(data_cfg, tf, sort_keys=False)
        temp_data_path = tf.name
    model.train(data=temp_data_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)

    run_subdir = "segment" if args.seg else "detect"
    best = Path("runs") / run_subdir / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected best model at: {best}")

    out = Path("models") / "yolov8_terrain.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(best.read_bytes())
    print(f"Saved trained weights to {out}")


if __name__ == "__main__":
    main()
