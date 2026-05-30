"""Open3D RGB-D odometry backend (https://github.com/isl-org/Open3D)."""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d

from slam_module.backends.base import SlamBackend, SlamBackendInfo, SlamTrackResult
from slam_module.backends.opencv_orb_rgbd import _quat_from_R


class Open3dRgbdOdometry(SlamBackend):
    info = SlamBackendInfo(
        name="open3d_rgbd",
        github="https://github.com/isl-org/Open3D",
        description="Open3D multi-scale RGB-D visual odometry with robust kernels.",
        sensor_modes=["rgbd"],
    )

    def __init__(
        self,
        *,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
        depth_scale: float = 5000.0,
        depth_trunc_m: float = 4.0,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.depth_trunc_m = depth_trunc_m
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        self.reset()

    def reset(self) -> None:
        self._prev_rgbd: o3d.geometry.RGBDImage | None = None
        self._pose = np.eye(4, dtype=np.float64)
        self._initialized = False

    def _to_rgbd(self, frame_bgr: np.ndarray, depth_m: np.ndarray) -> o3d.geometry.RGBDImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_mm = np.nan_to_num(depth_m, nan=0.0) * 1000.0
        depth_o3d = o3d.geometry.Image(depth_mm.astype(np.float32))
        color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=float(self.depth_trunc_m),
            convert_rgb_to_intensity=False,
        )

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        _ = timestamp
        if depth_m is None:
            raise ValueError("Open3D RGB-D backend requires depth")
        rgbd = self._to_rgbd(frame_bgr, depth_m)
        state = "OK"
        if not self._initialized or self._prev_rgbd is None:
            self._prev_rgbd = rgbd
            self._initialized = True
            return SlamTrackResult(
                position=self._pose[:3, 3].copy(),
                quaternion_xyzw=_quat_from_R(self._pose[:3, :3]),
                tracking_state="INIT",
            )

        option = o3d.pipelines.odometry.OdometryOption()
        success, delta, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            self._prev_rgbd,
            rgbd,
            self.intrinsic,
            np.eye(4, dtype=np.float64),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
        if success:
            self._pose = self._pose @ delta
        else:
            state = "LOST"
        self._prev_rgbd = rgbd
        return SlamTrackResult(
            position=self._pose[:3, 3].copy(),
            quaternion_xyzw=_quat_from_R(self._pose[:3, :3]),
            tracking_state=state,
        )
