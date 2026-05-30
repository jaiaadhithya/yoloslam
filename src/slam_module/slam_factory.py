"""SLAM backend registry and factory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from slam_module.backends.base import SlamBackend
from slam_module.backends.droid_slam import DroidSlamStreaming
from slam_module.backends.keyframe_pose_graph_rgbd import KeyframePoseGraphRgbdSlam
from slam_module.backends.yoloslam_rkf_rgbd import YoloslamRkfRgbdSlam
from slam_module.backends.opencv_orb_rgbd import OpenCvOrbRgbdSlam
from slam_module.backends.orbslam3 import OrbSlam3External
from slam_module.backends.rtabmap import RtabmapCliSlam, RtabmapRgbdSlam
from slam_module.synthetic_trajectory import SyntheticUavTrajectory


def _open3d_backend():
    from slam_module.backends.open3d_rgbd import Open3dRgbdOdometry

    return Open3dRgbdOdometry


BACKEND_REGISTRY: dict[str, type | callable] = {
    "opencv_orb_rgbd": OpenCvOrbRgbdSlam,
    "droid_slam": DroidSlamStreaming,
    "orb_slam3": OrbSlam3External,
    "rtabmap": RtabmapRgbdSlam,
    "rtabmap_cli": RtabmapCliSlam,
    "open3d_rgbd": _open3d_backend,
    "keyframe_pose_graph_rgbd": KeyframePoseGraphRgbdSlam,
    "yoloslam_rkf_rgbd": YoloslamRkfRgbdSlam,
}


def load_slam_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or Path(__file__).resolve().parents[2] / "config" / "slam_backend.yaml"
    if not cfg_path.exists():
        return {"backend": "opencv_orb_rgbd", "opencv_orb_rgbd": {}}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_backend(name: str | None = None, cfg: dict[str, Any] | None = None) -> SlamBackend:
    config = cfg or load_slam_config()
    backend_name = name or str(config.get("backend", "opencv_orb_rgbd"))
    if backend_name == "synthetic":
        raise ValueError("Use SyntheticSlamWrapper for synthetic backend")
    cls = BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        raise KeyError(f"Unknown SLAM backend '{backend_name}'. Options: {sorted(BACKEND_REGISTRY)}")
    if callable(cls) and not isinstance(cls, type):
        cls = cls()
    params = dict(config.get(backend_name, {}) or {})
    return cls(**params)  # type: ignore[call-arg]


def list_backends() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name, cls in BACKEND_REGISTRY.items():
        factory = cls() if callable(cls) and not isinstance(cls, type) else cls
        info = factory.info  # type: ignore[attr-defined]
        items.append(
            {
                "key": name,
                "name": info.name,
                "github": info.github,
                "description": info.description,
                "sensor_modes": info.sensor_modes,
            }
        )
    return items
