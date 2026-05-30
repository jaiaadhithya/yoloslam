"""Helpers for running integrated SLAM inside PyBullet demos."""

from __future__ import annotations

import math

import numpy as np

from slam_module.slam_factory import load_slam_config
from slam_module.slam_wrapper import SlamWrapper


def _slam_config_with_camera(
    backend: str | None,
    camera: dict | None,
) -> dict:
    cfg = dict(load_slam_config())
    if backend is not None:
        cfg["backend"] = backend
    if not camera:
        return cfg
    backend_key = str(cfg.get("backend", "opencv_orb_rgbd"))
    params = dict(cfg.get(backend_key, {}) or {})
    for key in ("fx", "fy", "cx", "cy", "max_depth_m", "min_depth_m"):
        if key in camera:
            params[key] = float(camera[key])
    if "max_features" in camera:
        params["max_features"] = int(camera["max_features"])
    cfg[backend_key] = params
    return cfg

def slam_pose_to_dict(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    *,
    tracking_state: str = "OK",
) -> dict:
    qx, qy, qz, qw = [float(v) for v in quaternion_xyzw]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "yaw": float(yaw),
        "state": str(tracking_state),
    }


class PyBulletSlamRunner:
    """Frame-by-frame SLAM using the configured backend from slam_backend.yaml."""

    def __init__(
        self,
        *,
        backend: str | None = None,
        camera: dict | None = None,
    ) -> None:
        cfg = _slam_config_with_camera(backend, camera)
        self.wrapper = SlamWrapper(config=cfg)
        self.backend_name = self.wrapper.backend_name

    def reset(self) -> None:
        self.wrapper.reset()

    def track_rgbd(
        self,
        frame_bgr: np.ndarray,
        depth_m: np.ndarray,
        timestamp: float,
    ) -> dict:
        pose = self.wrapper.track(frame_bgr, timestamp, depth_m=depth_m)
        return slam_pose_to_dict(
            pose.position,
            pose.quaternion_xyzw,
            tracking_state=pose.tracking_state,
        )
