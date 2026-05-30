"""OpenCV ORB + RGB-D keyframe visual SLAM (classical baseline)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from slam_module.backends.base import SlamBackend, SlamBackendInfo, SlamTrackResult


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s
    q /= np.linalg.norm(q) + 1e-12
    return q


class OpenCvOrbRgbdSlam(SlamBackend):
    info = SlamBackendInfo(
        name="opencv_orb_rgbd",
        github="https://github.com/opencv/opencv",
        description="ORB feature tracking with RGB-D PnP and local keyframe map.",
        sensor_modes=["rgbd"],
    )

    def __init__(
        self,
        *,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
        max_features: int = 1500,
        min_depth_m: float = 0.3,
        max_depth_m: float = 5.0,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.dist = np.zeros(5, dtype=np.float64)
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.R_wc = np.eye(3, dtype=np.float64)
        self.t_wc = np.zeros(3, dtype=np.float64)
        self.prev_gray: np.ndarray | None = None
        self.prev_kp: list[cv2.KeyPoint] | None = None
        self.prev_desc: np.ndarray | None = None
        self.initialized = False
        self.lost_count = 0

    def reset(self) -> None:
        self.R_wc = np.eye(3, dtype=np.float64)
        self.t_wc = np.zeros(3, dtype=np.float64)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.initialized = False
        self.lost_count = 0

    def _backproject(self, u: float, v: float, depth: float) -> np.ndarray:
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth], dtype=np.float64)

    def _depth_at(self, depth_m: np.ndarray, u: float, v: float) -> float | None:
        h, w = depth_m.shape[:2]
        ui = int(round(u))
        vi = int(round(v))
        if ui < 1 or vi < 1 or ui >= w - 1 or vi >= h - 1:
            return None
        patch = depth_m[vi - 1 : vi + 2, ui - 1 : ui + 2]
        valid = patch[np.isfinite(patch) & (patch > self.min_depth_m) & (patch < self.max_depth_m)]
        if valid.size < 3:
            return None
        return float(np.median(valid))

    def _solve_pnp(
        self,
        pts3d: np.ndarray,
        pts2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(pts3d) < 8:
            return None
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            self.K,
            self.dist,
            iterationsCount=120,
            reprojectionError=4.0,
            confidence=0.995,
        )
        if not ok or inliers is None or len(inliers) < 6:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3)

    def _track_monocular(
        self,
        gray: np.ndarray,
        kp: list[cv2.KeyPoint],
        desc: np.ndarray,
        state: str,
    ) -> SlamTrackResult:
        """Monocular VO fallback when depth is unavailable (ROS RGB-only mode)."""
        matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        good: list[cv2.DMatch] = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 8 or self.prev_kp is None:
            self.lost_count += 1
            state = "LOST" if self.lost_count > 8 else "MONO"
            return SlamTrackResult(
                position=self.t_wc.copy(),
                quaternion_xyzw=_quat_from_R(self.R_wc),
                tracking_state=state,
            )

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            self.lost_count += 1
            state = "LOST" if self.lost_count > 8 else "MONO"
        else:
            _, R_rel, t_rel, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
            scale = 0.05  # nominal monocular scale (m per frame)
            t_wc_new = self.t_wc + self.R_wc @ (t_rel.reshape(3) * scale)
            self.R_wc = self.R_wc @ R_rel
            self.t_wc = t_wc_new
            self.lost_count = 0
            state = "MONO"

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_desc = desc
        return SlamTrackResult(
            position=self.t_wc.copy(),
            quaternion_xyzw=_quat_from_R(self.R_wc),
            tracking_state=state,
        )

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        _ = timestamp
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        state = "OK"

        if not self.initialized:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_desc = desc
            self.initialized = True
            return SlamTrackResult(
                position=self.t_wc.copy(),
                quaternion_xyzw=_quat_from_R(self.R_wc),
                tracking_state=state,
            )

        if desc is None or self.prev_desc is None or self.prev_kp is None:
            self.lost_count += 1
            state = "LOST" if self.lost_count > 5 else "OK"
            return SlamTrackResult(
                position=self.t_wc.copy(),
                quaternion_xyzw=_quat_from_R(self.R_wc),
                tracking_state=state,
            )

        if depth_m is None:
            return self._track_monocular(gray, kp, desc, state)

        matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        good: list[cv2.DMatch] = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        pts3d: list[np.ndarray] = []
        pts2d: list[np.ndarray] = []
        for m in good:
            prev_pt = self.prev_kp[m.queryIdx].pt
            cur_pt = kp[m.trainIdx].pt
            depth = self._depth_at(depth_m, prev_pt[0], prev_pt[1])
            if depth is None:
                continue
            p_cam = self._backproject(prev_pt[0], prev_pt[1], depth)
            p_world = self.R_wc @ p_cam + self.t_wc
            pts3d.append(p_world)
            pts2d.append(np.array([cur_pt[0], cur_pt[1]], dtype=np.float64))

        solved = self._solve_pnp(np.asarray(pts3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64))
        if solved is not None:
            R_cw, t_cw = solved
            self.R_wc = R_cw.T
            self.t_wc = -R_cw.T @ t_cw
            self.lost_count = 0
        else:
            self.lost_count += 1
            if self.lost_count > 8:
                state = "LOST"

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_desc = desc
        return SlamTrackResult(
            position=self.t_wc.copy(),
            quaternion_xyzw=_quat_from_R(self.R_wc),
            tracking_state=state,
        )
