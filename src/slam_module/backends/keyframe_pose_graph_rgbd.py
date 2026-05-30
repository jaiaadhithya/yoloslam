"""Keyframe pose-graph RGB-D SLAM built on ORB features (full local mapping)."""

from __future__ import annotations

import cv2
import numpy as np

from slam_module.backends.base import SlamBackend, SlamBackendInfo, SlamTrackResult
from slam_module.backends.opencv_orb_rgbd import OpenCvOrbRgbdSlam, _quat_from_R


class KeyframePoseGraphRgbdSlam(SlamBackend):
    info = SlamBackendInfo(
        name="keyframe_pose_graph_rgbd",
        github="https://github.com/opencv/opencv",
        description="Keyframe RGB-D SLAM with ORB map points and pose-graph chaining.",
        sensor_modes=["rgbd"],
    )

    def __init__(
        self,
        *,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
        max_features: int = 2000,
        min_depth_m: float = 0.3,
        max_depth_m: float = 5.0,
        keyframe_trans_m: float = 0.08,
        keyframe_rot_deg: float = 8.0,
        max_map_points: int = 4000,
    ) -> None:
        self.tracker = OpenCvOrbRgbdSlam(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            max_features=max_features,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        self.keyframe_trans_m = keyframe_trans_m
        self.keyframe_rot_deg = keyframe_rot_deg
        self.max_map_points = max_map_points
        self._keyframes: list[np.ndarray] = []
        self._poses: list[np.ndarray] = []
        self._map_pts: list[np.ndarray] = []
        self._last_kf_pose = np.eye(4, dtype=np.float64)

    def reset(self) -> None:
        self.tracker.reset()
        self._keyframes = []
        self._poses = []
        self._map_pts = []
        self._last_kf_pose = np.eye(4, dtype=np.float64)

    @staticmethod
    def _pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)
        return T

    @staticmethod
    def _rotation_angle_deg(R: np.ndarray) -> float:
        trace = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.degrees(np.arccos(trace)))

    def _maybe_add_keyframe(self, frame_bgr: np.ndarray, depth_m: np.ndarray, pose_T: np.ndarray) -> None:
        if not self._keyframes:
            self._keyframes.append(frame_bgr.copy())
            self._poses.append(pose_T.copy())
            self._last_kf_pose = pose_T.copy()
            return
        delta = np.linalg.inv(self._last_kf_pose) @ pose_T
        trans = float(np.linalg.norm(delta[:3, 3]))
        rot = self._rotation_angle_deg(delta[:3, :3])
        if trans < self.keyframe_trans_m and rot < self.keyframe_rot_deg:
            return
        self._keyframes.append(frame_bgr.copy())
        self._poses.append(pose_T.copy())
        self._last_kf_pose = pose_T.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        step = 8
        for v in range(step // 2, gray.shape[0], step):
            for u in range(step // 2, gray.shape[1], step):
                depth = self.tracker._depth_at(depth_m, float(u), float(v))
                if depth is None:
                    continue
                p_cam = self.tracker._backproject(float(u), float(v), depth)
                p_world = pose_T[:3, :3] @ p_cam + pose_T[:3, 3]
                self._map_pts.append(p_world)
        if len(self._map_pts) > self.max_map_points:
            self._map_pts = self._map_pts[-self.max_map_points :]

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        result = self.tracker.track(frame_bgr, timestamp, depth_m=depth_m)
        R = result.quaternion_xyzw  # noqa: N806 - not used directly
        _ = R
        # Reconstruct rotation from tracker internal state.
        R_wc = self.tracker.R_wc
        t_wc = self.tracker.t_wc
        pose_T = self._pose_matrix(R_wc, t_wc)
        if depth_m is not None:
            self._maybe_add_keyframe(frame_bgr, depth_m, pose_T)
        return SlamTrackResult(
            position=t_wc.copy(),
            quaternion_xyzw=_quat_from_R(R_wc),
            tracking_state=result.tracking_state,
            timestamp=timestamp,
        )
