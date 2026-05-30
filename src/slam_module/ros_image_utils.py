"""ROS Image message helpers for SLAM nodes."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    if msg.height == 0 or msg.width == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    step = int(msg.step) if msg.step else 0
    if step > 0 and len(arr) >= step * msg.height:
        row = arr[: step * msg.height].reshape((msg.height, step))
        channels = max(1, step // msg.width)
        img = row[:, : msg.width * channels].reshape((msg.height, msg.width, channels))
    else:
        channels = max(1, int(len(arr) / (msg.height * msg.width)))
        img = arr.reshape((msg.height, msg.width, channels))

    enc = (msg.encoding or "").lower()
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    if "bgr" in enc:
        return img[:, :, :3][:, :, ::-1].copy()
    return img[:, :, :3].copy()


def image_msg_to_depth_m(msg: Image) -> np.ndarray | None:
    """Convert depth Image to float32 meters (NaN for invalid)."""
    if msg.height == 0 or msg.width == 0:
        return None
    enc = (msg.encoding or "").lower()

    if enc in {"32fc1", "32fc"}:
        arr = np.frombuffer(msg.data, dtype=np.float32)
        depth = arr.reshape((msg.height, msg.width)).astype(np.float32)
        depth[~np.isfinite(depth)] = np.nan
        depth[depth <= 0.0] = np.nan
        return depth

    if enc in {"16uc1", "mono16"}:
        arr = np.frombuffer(msg.data, dtype=np.uint16)
        depth = arr.reshape((msg.height, msg.width)).astype(np.float32) / 1000.0
        depth[depth <= 0.0] = np.nan
        return depth

    # Fallback: treat as float32 buffer.
    try:
        arr = np.frombuffer(msg.data, dtype=np.float32)
        if arr.size >= msg.height * msg.width:
            depth = arr[: msg.height * msg.width].reshape((msg.height, msg.width)).astype(np.float32)
            depth[~np.isfinite(depth)] = np.nan
            depth[depth <= 0.0] = np.nan
            return depth
    except ValueError:
        return None
    return None
