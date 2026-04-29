import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import cv2
import numpy as np

from yolo_detector.detector import DetectorConfig, LandingPadDetector


@dataclass
class Detection:
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str
    is_safe: bool
    timestamp: float


class YoloNode:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.detector = LandingPadDetector(config or DetectorConfig())
        self.last_latency_ms = 0.0

    def process_frame(self, image_bgr: np.ndarray) -> List[Dict]:
        t0 = time.perf_counter()
        raw = self.detector.infer(image_bgr)
        enriched = self.detector.enrich(raw)
        detections = [
            Detection(
                d["x_center"],
                d["y_center"],
                d["width"],
                d["height"],
                d["confidence"],
                d["class_id"],
                d["class_name"],
                d["is_safe"],
                time.time(),
            )
            for d in enriched
        ]
        self.last_latency_ms = (time.perf_counter() - t0) * 1e3
        return [asdict(d) for d in detections]

    @staticmethod
    def annotate(image_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
        out = image_bgr.copy()
        for d in detections:
            x1 = int(d["x_center"] - d["width"] / 2)
            y1 = int(d["y_center"] - d["height"] / 2)
            x2 = int(d["x_center"] + d["width"] / 2)
            y2 = int(d["y_center"] + d["height"] / 2)
            color = (0, 180, 0) if d["is_safe"] else (0, 0, 220)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f'{d["class_name"]} {d["confidence"]:.2f} safe={int(d["is_safe"])}',
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return out


if __name__ == "__main__":
    node = YoloNode()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    tick = 0
    while True:
        detections = node.process_frame(img)
        if tick % 5 == 0:
            print(
                {"num_detections": len(detections), "latency_ms": round(node.last_latency_ms, 3)},
                flush=True,
            )
        tick += 1
        time.sleep(1.0)
