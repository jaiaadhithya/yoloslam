"""YOLOSLAM-RKF: Robust Keyframe Fusion RGB-D SLAM (proposed backend).

Extends the keyframe pose-graph baseline with:
1) Denser keyframe map (shorter translation/rotation thresholds).
2) Map-anchored recovery PnP when frame-to-frame ORB matches fall below a threshold.
Frame-to-frame tracking uses the same RGB-D PnP chain as the baseline (no fusion drift).
"""

from __future__ import annotations

import cv2
import numpy as np

from slam_module.backends.base import SlamBackendInfo, SlamTrackResult
from slam_module.backends.keyframe_pose_graph_rgbd import KeyframePoseGraphRgbdSlam
from slam_module.backends.opencv_orb_rgbd import _quat_from_R


class YoloslamRkfRgbdSlam(KeyframePoseGraphRgbdSlam):
    """YOLOSLAM-RKF: landing-oriented keyframe RGB-D SLAM with map recovery."""

    info = SlamBackendInfo(
        name="yoloslam_rkf_rgbd",
        github="yoloslam (this repository)",
        description=(
            "Proposed YOLOSLAM-RKF: denser keyframe RGB-D map + map-anchored "
            "recovery when frame-to-frame ORB matches are sparse."
        ),
        sensor_modes=["rgbd"],
    )

    def __init__(
        self,
        *,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
        max_features: int = 2200,
        min_depth_m: float = 0.3,
        max_depth_m: float = 5.0,
        keyframe_trans_m: float = 0.06,
        keyframe_rot_deg: float = 6.0,
        max_map_points: int = 6000,
        recovery_match_threshold: int = 7,
    ) -> None:
        super().__init__(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            max_features=max_features,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            keyframe_trans_m=keyframe_trans_m,
            keyframe_rot_deg=keyframe_rot_deg,
            max_map_points=max_map_points,
        )
        self.recovery_match_threshold = recovery_match_threshold

    def _try_map_recovery(
        self,
        kp: list[cv2.KeyPoint],
        depth_m: np.ndarray,
    ) -> bool:
        if len(self._map_pts) < 12 or not kp:
            return False
        pts3d: list[np.ndarray] = []
        pts2d: list[np.ndarray] = []
        kp_xy = np.array([p.pt for p in kp], dtype=np.float64)
        for p_world in self._map_pts[-1000:]:
            p_cam = self.tracker.R_wc.T @ (p_world - self.tracker.t_wc)
            if p_cam[2] <= self.tracker.min_depth_m:
                continue
            u = self.tracker.fx * p_cam[0] / p_cam[2] + self.tracker.cx
            v = self.tracker.fy * p_cam[1] / p_cam[2] + self.tracker.cy
            if u < 0 or v < 0 or u >= depth_m.shape[1] or v >= depth_m.shape[0]:
                continue
            j = int(np.argmin(np.linalg.norm(kp_xy - np.array([u, v]), axis=1)))
            if float(np.linalg.norm(kp_xy[j] - np.array([u, v]))) > 14.0:
                continue
            depth = self.tracker._depth_at(depth_m, kp[j].pt[0], kp[j].pt[1])
            if depth is None:
                continue
            pts3d.append(p_world)
            pts2d.append(np.array(kp[j].pt, dtype=np.float64))
        if len(pts3d) < 10:
            return False
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(pts3d, dtype=np.float64),
            np.asarray(pts2d, dtype=np.float64),
            self.tracker.K,
            self.tracker.dist,
            iterationsCount=100,
            reprojectionError=3.5,
            confidence=0.995,
        )
        if not ok or inliers is None or len(inliers) < 8:
            return False
        R_cw, _ = cv2.Rodrigues(rvec)
        self.tracker.R_wc = R_cw.T
        self.tracker.t_wc = -self.tracker.R_wc @ tvec.reshape(3)
        self.tracker.lost_count = 0
        return True

    def track(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        *,
        depth_m: np.ndarray | None = None,
    ) -> SlamTrackResult:
        if depth_m is None:
            return self.tracker.track(frame_bgr, timestamp, depth_m=None)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self.tracker.orb.detectAndCompute(gray, None)
        n_matches = 0
        if (
            self.tracker.initialized
            and desc is not None
            and self.tracker.prev_desc is not None
            and self.tracker.prev_kp is not None
        ):
            matches = self.tracker.matcher.knnMatch(self.tracker.prev_desc, desc, k=2)
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    n_matches += 1

        result = self.tracker.track(frame_bgr, timestamp, depth_m=depth_m)
        state = result.tracking_state

        if n_matches < self.recovery_match_threshold and self._try_map_recovery(kp or [], depth_m):
            state = "RELOC"
            result = SlamTrackResult(
                position=self.tracker.t_wc.copy(),
                quaternion_xyzw=_quat_from_R(self.tracker.R_wc),
                tracking_state=state,
                timestamp=timestamp,
            )

        R_wc = self.tracker.R_wc
        t_wc = self.tracker.t_wc
        pose_T = self._pose_matrix(R_wc, t_wc)
        self._maybe_add_keyframe(frame_bgr, depth_m, pose_T)
        return SlamTrackResult(
            position=t_wc.copy(),
            quaternion_xyzw=_quat_from_R(R_wc),
            tracking_state=state,
            timestamp=timestamp,
        )
