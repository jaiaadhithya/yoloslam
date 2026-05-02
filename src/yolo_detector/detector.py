from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from yolo_detector.safety_mapping import SafetyMapper

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


DetectionTuple = Tuple[float, float, float, float, float, int]


@dataclass
class DetectorConfig:
    # Ultralytics pretrained weights (e.g. yolov8n.pt); auto-downloads on first load if missing.
    model_path: str = "yolov8n.pt"
    confidence: float = 0.5
    iou: float = 0.45
    # COCO → landing semantics (unsafe / neutral / positive_safe). None disables YAML.
    safety_mapping_yaml: str | None = "config/safety_mapping.yaml"
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
        self._dynamic_names: Dict[int, str] | None = None
        self._safety_mapper: SafetyMapper | None = None
        try:
            import torch

            self._device: str | int = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"
        if YOLO is not None:
            # Local file or hub name (e.g. yolov8n.pt) — Ultralytics fetches weights if needed.
            try:
                self.model = YOLO(str(config.model_path))
                names = getattr(self.model, "names", None)
                if isinstance(names, dict):
                    self._dynamic_names = {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, (list, tuple)):
                    self._dynamic_names = {i: str(n) for i, n in enumerate(names)}
            except Exception:
                self.model = None

    def _mapping_path(self) -> Path | None:
        raw = self.config.safety_mapping_yaml
        if not raw:
            return None
        p = Path(raw)
        if p.is_file():
            return p
        root = Path(__file__).resolve().parents[2]
        cand = root / raw
        return cand if cand.is_file() else None

    def _get_safety_mapper(self) -> SafetyMapper | None:
        if not self.config.safety_mapping_yaml:
            return None
        if self._safety_mapper is None:
            mp = self._mapping_path()
            self._safety_mapper = SafetyMapper.from_yaml(mp) if mp is not None else SafetyMapper.default()
        return self._safety_mapper

    def infer(self, image_bgr: np.ndarray) -> List[DetectionTuple]:
        if self.model is None:
            return self._fallback_detect(image_bgr)

        results = self.model.predict(
            image_bgr,
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=False,
            device=self._device,
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
        mapper = self._get_safety_mapper() if self.config.safety_mapping_yaml else None
        enriched: List[Dict] = []
        for x, y, w, h, conf, class_id in detections:
            if self._dynamic_names is not None:
                class_name = self._dynamic_names.get(class_id, f"class_{class_id}")
            else:
                class_name = self.config.class_map.get(class_id, f"class_{class_id}")

            if mapper is not None:
                safety_label, safety_weight = mapper.lookup(class_name)
                if safety_label == "neutral":
                    fusion_unsafe_delta = 0.0
                elif safety_label == "unsafe":
                    fusion_unsafe_delta = max(
                        0.05,
                        float(conf) * abs(float(safety_weight)) * mapper.unsafe_confidence_scale,
                    )
                elif safety_label == "positive_safe":
                    fusion_unsafe_delta = 0.0
                else:
                    fusion_unsafe_delta = max(0.05, float(conf) * abs(float(safety_weight)))
            else:
                legacy_safe = bool(self.config.safe_label_map.get(class_name, False))
                safety_label = "positive_safe" if legacy_safe else "unsafe"
                safety_weight = 0.35 if legacy_safe else -1.0
                fusion_unsafe_delta = max(0.05, float(conf))

            is_safe = safety_label == "positive_safe"

            row = {
                "x_center": x,
                "y_center": y,
                "width": w,
                "height": h,
                "confidence": conf,
                "class_id": class_id,
                "class_name": class_name,
                "is_safe": is_safe,
                "safety_label": safety_label,
                "safety_weight": float(safety_weight),
                "fusion_unsafe_delta": float(fusion_unsafe_delta),
            }
            enriched.append(row)
        return enriched

    def _fallback_detect(self, image_bgr: np.ndarray) -> List[DetectionTuple]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, w = image_bgr.shape[:2]
        detections: List[DetectionTuple] = []

        # Safe-ish terrain proxy: green/grass regions.
        safe_mask = cv2.inRange(hsv, np.array([30, 35, 35], dtype=np.uint8), np.array([100, 255, 255], dtype=np.uint8))
        safe_contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(safe_contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(contour)
            if area < 0.005 * (h * w):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            conf = min(0.8, 0.45 + 0.35 * (area / float(h * w)))
            detections.append((x + bw / 2.0, y + bh / 2.0, float(bw), float(bh), float(conf), 8))

        # Unsafe water proxy: saturated blue regions.
        water_mask = cv2.inRange(hsv, np.array([90, 60, 40], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8))
        water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(water_contours, key=cv2.contourArea, reverse=True)[:2]:
            area = cv2.contourArea(contour)
            if area < 0.004 * (h * w):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            conf = min(0.75, 0.4 + 0.35 * (area / float(h * w)))
            detections.append((x + bw / 2.0, y + bh / 2.0, float(bw), float(bh), float(conf), 1))

        # Keep the stack alive for demos if nothing was segmented.
        if not detections:
            detections.append((w / 2.0, h / 2.0, w * 0.25, h * 0.25, 0.51, 8))

        return detections
