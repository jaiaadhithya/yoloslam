"""Composite visualization for paper-style demo frames."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _world_delta_to_cam(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    cam_x = c * dx + s * dy
    cam_y = -s * dx + c * dy
    return cam_x, float(-cam_y)


def _project_world_to_pixel(
    wx: float,
    wy: float,
    pose: Dict,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[int, int]:
    """Nadir camera: world XY offset to image coords with yaw."""
    yaw = float(pose.get("yaw", 0.0))
    z = max(0.5, float(pose.get("z", 10.0)))
    dx = wx - float(pose["x"])
    dy = wy - float(pose["y"])
    cam_x, cam_y = _world_delta_to_cam(dx, dy, yaw)
    u = int(round(fx * cam_x / z + cx))
    v = int(round(fy * cam_y / z + cy))
    return u, v


def _safety_heatmap(safe: np.ndarray, unsafe: np.ndarray, size: int = 220) -> np.ndarray:
    s = safe.astype(float)
    u = unsafe.astype(float)
    net = s - 1.2 * u
    m = float(np.max(np.abs(net)) + 1e-6)
    net = (np.clip(net / m, -1.0, 1.0) + 1.0) * 0.5
    colored = cv2.applyColorMap((net * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)


def _draw_pov_landing_signalling(
    frame: np.ndarray,
    *,
    controller_state: str,
    zone: Dict,
    pose: Dict,
    fusion_live: bool,
    z_sc: float,
    frame_index: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    """Bottom status strip, signal lights, and approach cue on drone POV (in-place)."""
    h, w = frame.shape[:2]
    has_zone = bool(zone.get("has_valid_zone"))
    committed = controller_state in ("APPROACH", "ALIGN", "DESCEND", "LANDED")

    dist_xy = 0.0
    dz = 0.0
    u_tgt: int | None = None
    v_tgt: int | None = None
    if has_zone:
        dist_xy = math.hypot(
            float(zone["center_x"]) - float(pose["x"]),
            float(zone["center_y"]) - float(pose["y"]),
        )
        dz = float(zone.get("center_z", pose.get("z", 0.0))) - float(pose.get("z", 0.0))
        u_tgt, v_tgt = _project_world_to_pixel(
            float(zone["center_x"]),
            float(zone["center_y"]),
            pose,
            fx,
            fy,
            cx,
            cy,
        )

    if controller_state in ("ALIGN", "DESCEND", "LANDED"):
        pulse_top = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(frame_index * 0.25))
        tc = (0, int(230 * pulse_top), int(255 * pulse_top))
        cv2.rectangle(frame, (0, 0), (w, 4), tc, -1)

    if controller_state == "ABORT":
        title, detail = "ABORT", "Unsafe — re-survey / climb"
        bar_bgr = (55, 55, 200)
    elif controller_state == "LANDED":
        title, detail = "TOUCHDOWN", "Landing complete — systems safe"
        bar_bgr = (70, 200, 90)
    elif has_zone and not committed:
        title = "SAFE ZONE SIGNAL"
        detail = (
            ("Fusion: live | " if fusion_live else "Track: held | ")
            + f"score {z_sc:.2f} — awaiting commit to approach"
        )
        bar_bgr = (90, 150, 255)
    elif controller_state == "DESCEND":
        pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(frame_index * 0.35))
        title = "DESCEND — CLEARED TO LAND"
        detail = f"Δ horizontal {dist_xy:.2f} m | Δ vertical {dz:+.2f} m | signal: land"
        g = int(180 + 75 * pulse)
        bar_bgr = (60, g, 80)
    elif controller_state == "ALIGN":
        title = "ALIGN — FINAL POSITION"
        detail = f"Δ horizontal {dist_xy:.2f} m | hold over pad center"
        bar_bgr = (90, 200, 200)
    elif controller_state == "APPROACH":
        title = "APPROACH — CLOSING ON PAD"
        detail = f"Δ horizontal {dist_xy:.2f} m | follow cue to touchdown marker"
        bar_bgr = (110, 190, 180)
    elif controller_state == "SEARCH":
        title, detail = "SEARCH", "Locating certified safe landing zone"
        bar_bgr = (100, 120, 200)
    elif controller_state == "EVALUATE":
        title, detail = "EVALUATE", "Fusing detections — landing map update"
        bar_bgr = (95, 115, 165)
    else:
        title, detail = "SURVEY", "Aerial mapping — landing pipeline active"
        bar_bgr = (85, 100, 140)

    bar_h = 72
    y0 = h - bar_h
    roi = frame[y0:h, 0:w].astype(np.float32)
    tint = np.array(bar_bgr, dtype=np.float32).reshape(1, 1, 3)
    blended = roi * 0.38 + tint * 0.62
    frame[y0:h, 0:w] = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.line(frame, (0, y0), (w, y0), (240, 240, 250), 1, cv2.LINE_AA)

    led_y = y0 + 36
    led_off = (38, 42, 48)
    led1 = (230, 180, 100) if controller_state in ("SURVEY", "EVALUATE", "SEARCH") else led_off
    led2 = (80, 200, 255) if (has_zone and not committed) else led_off
    led3 = (100, 255, 120) if committed else led_off
    for i, col in enumerate((led1, led2, led3)):
        lx = 22 + i * 34
        cv2.circle(frame, (lx, led_y), 11, (20, 20, 22), -1, cv2.LINE_AA)
        cv2.circle(frame, (lx, led_y), 9, col, -1, cv2.LINE_AA)

    cv2.putText(frame, "MAP", (8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "ZONE", (40, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, "LDG", (74, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1, cv2.LINE_AA)

    tx = 118
    cv2.putText(frame, title, (tx, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (18, 18, 20), 3, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 250, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (tx, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (18, 18, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (tx, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 235), 1, cv2.LINE_AA)

    if has_zone and committed and controller_state != "LANDED" and u_tgt is not None and v_tgt is not None:
        icx, icy = int(round(cx)), int(round(cy))
        du, dv = u_tgt - icx, v_tgt - icy
        if math.hypot(du, dv) > 28.0:
            tip = (u_tgt, v_tgt)
            cv2.arrowedLine(frame, (icx, icy), tip, (210, 255, 255), 2, cv2.LINE_AA, 0, 0.22)
            cv2.circle(frame, (icx, icy), 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "APPROACH CUE",
                (min(w - 140, icx + 12), max(icy - 10, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "APPROACH CUE",
                (min(w - 140, icx + 12), max(icy - 10, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 255, 255),
                1,
                cv2.LINE_AA,
            )


def composite_frame(
    image_bgr: np.ndarray,
    detections: List[Dict],
    fusion,
    zone: Dict,
    pose: Dict,
    controller_state: str,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frame_index: int = 0,
    zone_project_pose: Dict | None = None,
    subtitle_lines: List[str] | None = None,
    pipeline_lines: List[str] | None = None,
    fusion_live: bool = True,
) -> np.ndarray:
    frame = image_bgr.copy()
    h, w = frame.shape[:2]

    zp = zone_project_pose if zone_project_pose is not None else pose

    for d in detections:
        x, yc, bw, bh = float(d["x_center"]), float(d["y_center"]), float(d["width"]), float(d["height"])
        x0 = int(round(x - bw / 2))
        y0 = int(round(yc - bh / 2))
        x1 = int(round(x + bw / 2))
        y1 = int(round(yc + bh / 2))
        color = (0, 200, 0) if d.get("safety_label") == "positive_safe" else (40, 40, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        label = str(d.get("class_name", ""))
        cv2.putText(frame, label, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    heat = _safety_heatmap(fusion.grid.safe_score, fusion.grid.unsafe_score)
    hs, ws = heat.shape[:2]
    x0 = w - ws - 12
    y0 = 12
    roi = frame[y0 : y0 + hs, x0 : x0 + ws]
    blended = cv2.addWeighted(roi, 0.25, heat, 0.75, 0)
    frame[y0 : y0 + hs, x0 : x0 + ws] = blended
    cv2.rectangle(frame, (x0, y0), (x0 + ws, y0 + hs), (240, 240, 240), 1)
    cv2.putText(frame, "Safety grid", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    z_sc = float(zone.get("zone_score", 0.0))
    live_tag = "live" if fusion_live else "held"
    lines = [
        f"State: {controller_state}",
        f"SLAM pose x,y,z: {pose['x']:.1f}, {pose['y']:.1f}, {pose['z']:.1f} m",
        f"Zone score: {z_sc:.2f} | target OK={zone.get('has_valid_zone')} ({live_tag})",
    ]
    if pipeline_lines:
        lines.extend(pipeline_lines)
    if subtitle_lines:
        lines.extend(subtitle_lines)

    y = 26
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        y += 26

    if zone.get("has_valid_zone"):
        u, v = _project_world_to_pixel(float(zone["center_x"]), float(zone["center_y"]), zp, fx, fy, cx, cy)
        r = int(max(18, min(w, h) * 0.085))
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(frame_index * 0.22))
        committed = controller_state in ("APPROACH", "ALIGN", "DESCEND", "LANDED")
        if committed:
            col = (0, int(220 * pulse), int(255 * pulse))
            cross = (40, 255, 255)
            label_main = "SAFE — LAND HERE"
            sub = "Committed landing target" + ("" if fusion_live else " (track held)")
        else:
            col = (0, int(140 * pulse), int(255 * pulse))
            cross = (180, 220, 255)
            label_main = "SAFE ZONE DETECTED"
            sub = "Fusion: mapping / confirming before approach"

        thick = max(2, int(2 + pulse * 2)) if committed else max(2, int(1 + pulse))
        cv2.circle(frame, (u, v), r, col, thick, cv2.LINE_AA)
        cv2.circle(frame, (u, v), r + 6, (255, 255, 255), 1, cv2.LINE_AA)
        if not committed:
            cv2.circle(frame, (u, v), r + 10, col, 1, cv2.LINE_AA)
        br = int(r * 0.75)
        for sgn_x, sgn_y in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            ox, oy = u + sgn_x * br, v + sgn_y * br
            lx, ly = u + sgn_x * (br // 2), v + sgn_y * (br // 2)
            cv2.line(frame, (ox, oy), (lx, oy), col, 2, cv2.LINE_AA)
            cv2.line(frame, (ox, oy), (ox, ly), col, 2, cv2.LINE_AA)
        cv2.line(frame, (u - 14, v), (u + 14, v), cross, 2, cv2.LINE_AA)
        cv2.line(frame, (u, v - 14), (u, v + 14), cross, 2, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(label_main, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        box_w = max(tw, sw) + 16
        bx0 = u - box_w // 2
        by0 = v - r - th - sh - 30
        bx1, by1 = bx0 + box_w, by0 + th + sh + 14
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (25, 25, 25), -1)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), col, 2, cv2.LINE_AA)
        cv2.putText(
            frame, label_main, (bx0 + 8, by0 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, sub, (bx0 + 8, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA
        )

    _draw_pov_landing_signalling(
        frame,
        controller_state=controller_state,
        zone=zone,
        pose=zp,
        fusion_live=fusion_live,
        z_sc=z_sc,
        frame_index=frame_index,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    return frame


def _draw_landing_zone_chase(
    frame: np.ndarray,
    scene,
    zone: Dict,
    truth_xyz: Tuple[float, float, float],
    truth_yaw: float,
    controller_state: str,
    frame_index: int,
    *,
    chase_fov_deg: float,
    chase_distance_m: float,
    chase_height_m: float,
    chase_target_z_frac: float,
) -> None:
    """Ground disk + marker + cue line in chase view (in-place)."""
    if scene is None or not hasattr(scene, "project_world_to_chase_pixel"):
        return
    h, w = frame.shape[:2]
    committed = controller_state in ("APPROACH", "ALIGN", "DESCEND", "LANDED")
    pulse = 0.65 + 0.35 * (0.5 + 0.5 * np.sin(frame_index * 0.2))
    fill_bgr = (int(90 * pulse), int(220 * pulse), int(90 * pulse)) if committed else (int(100 * pulse), int(200 * pulse), int(255 * pulse))
    edge_bgr = (60, 255, 120) if committed else (70, 180, 255)

    cx = float(zone["center_x"])
    cy = float(zone["center_y"])
    rad_m = max(1.2, float(zone.get("zone_radius", 3.5)))
    z0 = 0.1

    def proj(wx: float, wy: float, wz: float) -> Tuple[int | None, int | None]:
        return scene.project_world_to_chase_pixel(
            wx,
            wy,
            wz,
            float(truth_xyz[0]),
            float(truth_xyz[1]),
            float(truth_xyz[2]),
            float(truth_yaw),
            w,
            h,
            fov_deg=chase_fov_deg,
            distance_m=chase_distance_m,
            height_above_drone_m=chase_height_m,
            target_height_frac=chase_target_z_frac,
        )

    n_ring = 56
    ring_pts: List[Tuple[int, int]] = []
    for i in range(n_ring + 1):
        ang = (i / float(n_ring)) * 2.0 * np.pi
        wx = cx + rad_m * float(np.cos(ang))
        wy = cy + rad_m * float(np.sin(ang))
        u, v = proj(wx, wy, z0)
        if u is not None and v is not None and 0 <= u < w and 0 <= v < h:
            ring_pts.append((u, v))
    if len(ring_pts) >= 6:
        arr = np.array(ring_pts, dtype=np.int32).reshape((-1, 1, 2))
        layer = np.zeros_like(frame)
        cv2.fillPoly(layer, [arr], fill_bgr)
        cv2.addWeighted(frame, 1.0, layer, 0.22, 0.0, frame)
        cv2.polylines(frame, [arr], True, edge_bgr, 2, cv2.LINE_AA)

    pillar: List[Tuple[int, int]] = []
    for z in np.linspace(0.05, 2.4, 14):
        u, v = proj(cx, cy, float(z))
        if u is not None and v is not None:
            pillar.append((u, v))
    for i in range(1, len(pillar)):
        cv2.line(frame, pillar[i - 1], pillar[i], edge_bgr, 2, cv2.LINE_AA)

    uc, vc = proj(cx, cy, z0)
    if uc is not None and vc is not None:
        pr = max(6, int(min(w, h) * 0.018))
        cv2.circle(frame, (uc, vc), pr, edge_bgr, 2, cv2.LINE_AA)
        cv2.line(frame, (uc - pr - 4, vc), (uc + pr + 4, vc), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (uc, vc - pr - 4), (uc, vc + pr + 4), (255, 255, 255), 1, cv2.LINE_AA)
        tag = "LANDING TARGET (safe)" if committed else "LANDING ZONE (detected)"
        cv2.putText(frame, tag, (uc + pr + 10, vc + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(frame, tag, (uc + pr + 10, vc + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, edge_bgr, 2, cv2.LINE_AA)

    uu, vv = float(truth_xyz[0]), float(truth_xyz[1])
    u0, v0 = proj(uu, vv, max(0.12, float(truth_xyz[2])))
    if (
        u0 is not None
        and v0 is not None
        and uc is not None
        and vc is not None
        and committed
    ):
        cv2.line(frame, (u0, v0), (uc, vc), (200, 200, 255), 1, cv2.LINE_AA)


def annotate_chase_frame(
    image_bgr: np.ndarray,
    *,
    controller_state: str,
    truth_xyz: Tuple[float, float, float],
    truth_yaw: float,
    n_detections: int,
    scene=None,
    trajectory_xyz: Sequence[Tuple[float, float, float]] | None = None,
    chase_fov_deg: float = 54.0,
    chase_distance_m: float = 10.5,
    chase_height_m: float = 2.25,
    chase_target_z_frac: float = 0.88,
    extra_lines: List[str] | None = None,
    landing_zone: Dict | None = None,
    frame_index: int = 0,
) -> np.ndarray:
    """Chase HUD + 3D flight path projected into this view."""
    frame = image_bgr.copy()
    h, w = frame.shape[:2]

    if (
        trajectory_xyz is not None
        and len(trajectory_xyz) > 1
        and scene is not None
        and hasattr(scene, "project_world_to_chase_pixel")
    ):
        step = max(1, len(trajectory_xyz) // 650)
        pts2: List[Optional[Tuple[int, int]]] = []
        for i in range(0, len(trajectory_xyz), step):
            px, py, pz = trajectory_xyz[i]
            u, v = scene.project_world_to_chase_pixel(
                float(px),
                float(py),
                float(pz),
                float(truth_xyz[0]),
                float(truth_xyz[1]),
                float(truth_xyz[2]),
                float(truth_yaw),
                w,
                h,
                fov_deg=chase_fov_deg,
                distance_m=chase_distance_m,
                height_above_drone_m=chase_height_m,
                target_height_frac=chase_target_z_frac,
            )
            pts2.append((u, v) if u is not None and v is not None else None)
        lx, ly, lz = trajectory_xyz[-1]
        u, v = scene.project_world_to_chase_pixel(
            float(lx),
            float(ly),
            float(lz),
            float(truth_xyz[0]),
            float(truth_xyz[1]),
            float(truth_xyz[2]),
            float(truth_yaw),
            w,
            h,
            fov_deg=chase_fov_deg,
            distance_m=chase_distance_m,
            height_above_drone_m=chase_height_m,
            target_height_frac=chase_target_z_frac,
        )
        pts2.append((u, v) if u is not None and v is not None else None)
        for i in range(1, len(pts2)):
            a, b = pts2[i - 1], pts2[i]
            if a is None or b is None:
                continue
            cv2.line(frame, a, b, (60, 255, 255), 4, cv2.LINE_AA)
            cv2.line(frame, a, b, (255, 255, 255), 2, cv2.LINE_AA)

    if landing_zone is not None and bool(landing_zone.get("has_valid_zone")):
        _draw_landing_zone_chase(
            frame,
            scene,
            landing_zone,
            truth_xyz,
            truth_yaw,
            controller_state,
            frame_index,
            chase_fov_deg=chase_fov_deg,
            chase_distance_m=chase_distance_m,
            chase_height_m=chase_height_m,
            chase_target_z_frac=chase_target_z_frac,
        )

    lines = [
        f"Chase view | FSM: {controller_state}",
        f"True UAV x,y,z: {truth_xyz[0]:.1f}, {truth_xyz[1]:.1f}, {truth_xyz[2]:.1f} m",
        f"Detections (frame): {n_detections}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (250, 250, 250), 2, cv2.LINE_AA)
        y += 26
    return frame
