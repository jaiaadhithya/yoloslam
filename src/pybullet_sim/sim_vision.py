"""Detection adapters for synthetic terrain demos."""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from yolo_detector.detector import DetectorConfig, LandingPadDetector


def detect_color_terrain(image_bgr: np.ndarray) -> List[Dict]:
    """Heuristic blobs for grass (safe), water (unsafe), rocks/debris (unsafe), trees (unsafe)."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    out: List[Dict] = []

    # Grass / safe landing surface
    grass = cv2.inRange(hsv, np.array([32, 28, 35], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8))
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    for cnt in sorted(cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(cnt)
        if area < 0.0025 * (h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        conf = float(min(0.92, 0.55 + 0.4 * (area / float(h * w))))
        out.append(_det(x + bw * 0.5, y + bh * 0.5, float(bw), float(bh), conf, "grass_field", "positive_safe", 0.35, 0.0))

    # Water
    water = cv2.inRange(hsv, np.array([90, 55, 45], dtype=np.uint8), np.array([125, 255, 255], dtype=np.uint8))
    for cnt in sorted(cv2.findContours(water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(cnt)
        if area < 0.002 * (h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        conf = float(min(0.9, 0.5 + 0.35 * (area / float(h * w))))
        out.append(_det(x + bw * 0.5, y + bh * 0.5, float(bw), float(bh), conf, "water", "unsafe", -1.0, max(0.05, conf)))

    # Rocks / soil (brown, low hue)
    rock = cv2.inRange(hsv, np.array([0, 40, 35], dtype=np.uint8), np.array([22, 255, 210], dtype=np.uint8))
    rock &= ~grass
    for cnt in sorted(cv2.findContours(rock, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < 0.0018 * (h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        conf = float(min(0.88, 0.48 + 0.35 * (area / float(h * w))))
        out.append(_det(x + bw * 0.5, y + bh * 0.5, float(bw), float(bh), conf, "debris_clutter", "unsafe", -1.0, max(0.05, conf)))

    # Tree canopy (dark green, tighter)
    trees = cv2.inRange(hsv, np.array([32, 60, 25], dtype=np.uint8), np.array([85, 255, 160], dtype=np.uint8))
    trees &= cv2.bitwise_not(grass)
    for cnt in sorted(cv2.findContours(trees, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(cnt)
        if area < 0.0015 * (h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = float(bw) / float(max(1, bh))
        if ar > 2.8:  # skip elongated grass strips misclassified
            continue
        conf = float(min(0.9, 0.5 + 0.35 * (area / float(h * w))))
        out.append(_det(x + bw * 0.5, y + bh * 0.5, float(bw), float(bh), conf, "tree_canopy", "unsafe", -1.0, max(0.05, conf)))

    return out


def _det(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    conf: float,
    class_name: str,
    safety_label: str,
    safety_weight: float,
    fusion_unsafe_delta: float,
) -> Dict:
    is_safe = safety_label == "positive_safe"
    return {
        "x_center": float(xc),
        "y_center": float(yc),
        "width": float(bw),
        "height": float(bh),
        "confidence": float(conf),
        "class_id": -1,
        "class_name": class_name,
        "is_safe": bool(is_safe),
        "safety_label": safety_label,
        "safety_weight": float(safety_weight),
        "fusion_unsafe_delta": float(fusion_unsafe_delta),
    }


class DemoDetector:
    """Runs Ultralytics YOLO, color segmentation, or stable scene-semantic detections."""

    def __init__(self, *, mode: str, model_path: str, safety_yaml: str | None) -> None:
        self.mode = str(mode).lower()
        self._yolo: LandingPadDetector | None = None
        if self.mode in {"yolo", "hybrid"}:
            cfg = DetectorConfig(model_path=model_path, safety_mapping_yaml=safety_yaml, confidence=0.28, iou=0.45)
            self._yolo = LandingPadDetector(cfg)

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        scene=None,
        pose: Dict | None = None,
        fx: float = 320.0,
        fy: float = 320.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ) -> List[Dict]:
        if self.mode == "yolo" and self._yolo is not None:
            raw = self._yolo.infer(image_bgr)
            return self._yolo.enrich(raw)
        if self.mode == "color":
            return detect_color_terrain(image_bgr)
        scene_dets = (
            detect_scene_semantic(scene, image_bgr.shape[1], image_bgr.shape[0], pose, fx=fx, fy=fy, cx=cx, cy=cy)
            if scene is not None and pose is not None
            else detect_color_terrain(image_bgr)
        )
        if self.mode == "hybrid" and self._yolo is not None:
            raw = self._yolo.infer(image_bgr)
            yolo_dets = self._yolo.enrich(raw)
            return yolo_dets + scene_dets
        return scene_dets


def detect_scene_semantic(
    scene,
    width: int,
    height: int,
    pose: Dict,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> List[Dict]:
    """Stable detections from known scene geometry: no flicker from color aliasing."""
    z = max(0.6, float(pose.get("z", 10.0)))
    yaw = float(pose.get("yaw", 0.0))
    px = float(pose.get("x", 0.0))
    py = float(pose.get("y", 0.0))
    dets: List[Dict] = []

    def world_to_pix(wx: float, wy: float) -> tuple[float, float]:
        dx = wx - px
        dy = wy - py
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        cam_x = c * dx + s * dy
        cam_y = -s * dx + c * dy
        u = fx * cam_x / z + cx
        v = fy * (-cam_y) / z + cy
        return float(u), float(v)

    def push_circle(
        wx: float,
        wy: float,
        r: float,
        class_name: str,
        safe: bool,
        conf: float,
        unsafe_delta_scale: float = 0.32,
    ) -> None:
        u, v = world_to_pix(wx, wy)
        rad = max(6.0, abs(fx * (2.0 * r) / z))
        if u + rad < 0 or u - rad >= width or v + rad < 0 or v - rad >= height:
            return
        bw = float(min(width * 0.95, 1.45 * rad))
        bh = float(min(height * 0.95, 1.45 * rad))
        label = "positive_safe" if safe else "unsafe"
        dets.append(
            _det(
                u,
                v,
                bw,
                bh,
                conf,
                class_name,
                label,
                0.35 if safe else -1.0,
                0.0 if safe else max(0.03, conf * unsafe_delta_scale),
            )
        )

    # Known unsafe objects (tight fusion footprint so urban clutter does not smother the whole grid).
    for tx, ty, tr in getattr(scene, "tree_crowns", []):
        push_circle(tx, ty, tr * 1.05, "tree_canopy", False, 0.90, unsafe_delta_scale=0.22)
    for rx, ry, rr in getattr(scene, "rock_patches", []):
        push_circle(rx, ry, rr, "debris_clutter", False, 0.82, unsafe_delta_scale=0.2)
    for wx, wy, wh in getattr(scene, "water_patches", []):
        push_circle(wx, wy, wh * 1.1, "water", False, 0.88, unsafe_delta_scale=0.22)
    for bx, by, br in getattr(scene, "building_footprints", []):
        push_circle(bx, by, br * 0.82, "building", False, 0.86, unsafe_delta_scale=0.1)
    for cx, cy, cr in getattr(scene, "car_patches", []):
        push_circle(cx, cy, cr * 0.78, "car", False, 0.88, unsafe_delta_scale=0.1)

    # Safe road regions.
    for rx, ry, hx, hy in getattr(scene, "road_patches", []):
        push_circle(rx, ry, max(hx, hy) * 0.65, "flat_ground", True, 0.76)

    # Primary safe clearing around origin -> dense positive-safe evidence (open landing field).
    clear_r = float(getattr(getattr(scene, "cfg", None), "safe_clearing_radius_m", 8.0))
    push_circle(0.0, 0.0, clear_r * 0.92, "grass_field", True, 0.96)
    push_circle(0.0, 0.0, clear_r * 0.65, "flat_ground", True, 0.95)
    push_circle(0.0, 0.0, clear_r * 0.38, "flat_ground", True, 0.96)
    for ox, oy in [(-0.35, 0.0), (0.35, 0.0), (0.0, -0.35), (0.0, 0.35), (-0.25, -0.25), (0.25, 0.25)]:
        push_circle(ox * clear_r, oy * clear_r, clear_r * 0.22, "flat_ground", True, 0.9)

    # Keep pipeline alive with a safe hint if everything is out of frame.
    if not dets:
        dets.append(_det(width * 0.5, height * 0.5, width * 0.35, height * 0.35, 0.6, "flat_ground", "positive_safe", 0.35, 0.0))
    return dets
