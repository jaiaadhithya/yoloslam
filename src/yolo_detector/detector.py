from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


DetectionTuple = Tuple[float, float, float, float, float, int]


@dataclass
class DetectorConfig:
    model_path: str = "models/yolov8_terrain.pt"
    confidence: float = 0.5
    iou: float = 0.45
    class_map: Dict[int, str] = field(
        default_factory=lambda: {
            0: "flat_ground",
            1: "water",
            2: "tree_canopy",
            3: "building_structure",
            4: "vehicle",
            5: "person",
            6: "debris_clutter",
            7: "road_surface",
            8: "grass_field",
            9: "rooftop_flat",
            10: "fence_pole",
        }
    )
    safe_label_map: Dict[str, bool] = field(
        default_factory=lambda: {
            "flat_ground": True,
            "grass_field": True,
            "rooftop_flat": True,
            "road_surface": False,
            "water": False,
            "tree_canopy": False,
            "building_structure": False,
            "vehicle": False,
            "person": False,
            "debris_clutter": False,
            "fence_pole": False,
        }
    )


class LandingPadDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.model = None
        model_path = Path(config.model_path)
        if YOLO is not None and model_path.exists():
            self.model = YOLO(str(model_path))

    def infer(self, image_bgr: np.ndarray) -> List[DetectionTuple]:
        if self.model is None:
            return self._fallback_detect(image_bgr)

        results = self.model.predict(
            image_bgr,
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=False,
        )
        detections: List[DetectionTuple] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                x_c = (x1 + x2) / 2.0
                y_c = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1
                detections.append((x_c, y_c, w, h, conf, class_id))
        return detections

    def enrich(self, detections: List[DetectionTuple]) -> List[Dict]:
        enriched: List[Dict] = []
        for x, y, w, h, conf, class_id in detections:
            class_name = self.config.class_map.get(class_id, f"class_{class_id}")
            enriched.append(
                {
                    "x_center": x,
                    "y_center": y,
                    "width": w,
                    "height": h,
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name,
                    "is_safe": bool(self.config.safe_label_map.get(class_name, False)),
                }
            )
        return enriched

    def _fallback_detect(self, image_bgr: np.ndarray) -> List[DetectionTuple]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 220], dtype=np.uint8)
        upper = np.array([180, 60, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 300:
            return []
        return [(x + w / 2, y + h / 2, float(w), float(h), 0.6, 0)]
