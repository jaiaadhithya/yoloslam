# Run YOLOSLAM-RKF Anywhere

Quick guide to run the **YOLOSLAM-RKF** RGB-D SLAM backend (`yoloslam_rkf_rgbd`) on a normal laptop — no CUDA, no ORB-SLAM3 build, no WSL required.

---

## What you need

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ recommended |
| **OS** | Linux, Windows, or macOS |
| **GPU** | Not required |
| **Input** | Synchronized **RGB + depth** (not monocular-only) |
| **Core deps** | OpenCV, NumPy, PyYAML (`pip install -r requirements.txt`) |

YOLOSLAM-RKF uses **OpenCV ORB + RGB-D PnP** plus map-anchored recovery. It is the default backend in `config/slam_backend.yaml`.

---

## 1. One-time setup

From the repo root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Set Python path (every session):

```bash
# Linux/macOS / Git Bash
export PYTHONPATH=src

# Windows PowerShell
$env:PYTHONPATH = "src"
```

---

## 2. Verify the backend loads

```bash
PYTHONPATH=src python -c "
from slam_module.slam_factory import create_backend
b = create_backend('yoloslam_rkf_rgbd')
print(b.info.name, b.info.description)
b.reset()
print('OK')
"
```

Expected: `yoloslam_rkf_rgbd` and `OK`.

---

## 3. Run TUM benchmark (paper numbers)

**Dataset:** [TUM RGB-D freiburg1_xyz](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)

Extract to:

```
datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz/
├── rgb/
├── depth/
├── groundtruth.txt
└── associations.txt
```

**Full benchmark (792 frames, ~1–2 min on a typical CPU):**

```bash
PYTHONPATH=src python scripts/run_slam_comparison.py --max-frames 792 --frame-stride 1
```

**Faster smoke test (100 frames):**

```bash
PYTHONPATH=src python scripts/run_slam_comparison.py --max-frames 100 --frame-stride 2
```

**Outputs:**

| Path | Content |
|------|---------|
| `compre/slam_benchmark/comparison.json` | All metrics |
| `compre/slam_benchmark/yoloslam_rkf_rgbd/trajectory_eval.png` | Trajectory plot |
| `compre/slam_benchmark/ate_rmse_bar.png` | Bar chart |
| `compre/best_slam.json` | Winner metadata |

**Expected winner:** `yoloslam_rkf_rgbd` — ATE RMSE **~0.057 m** on full sequence.

**WSL tip (Windows):** If the dataset is on OneDrive and I/O is slow, use:

```bash
bash scripts/run_slam_comparison_wsl.sh
```

---

## 4. Run PyBullet landing demo (RGB-D SLAM + nadir camera)

Requires `pybullet` (in `requirements.txt`).

```bash
PYTHONPATH=src python scripts/run_pybullet_demo.py --config config/pybullet_demo.yaml
```

Config already sets:

```yaml
slam:
  mode: rgbd_slam
  backend: yoloslam_rkf_rgbd
  controller_uses_slam_estimate: true
```

**Important:** `camera.max_depth_m: 20.0` in `config/pybullet_demo.yaml` must match survey altitude (~12 m). Default 5 m rejects all depth samples.

**Outputs:** `demo_landing_drone.mp4`, `demo_landing_chase.mp4` (repo root).

---

## 5. Run in ROS 2 (RGB + depth topics)

Prerequisites: ROS 2, camera publishing `/camera` (RGB) and `/camera/depth` (depth).

```bash
# Terminal 1 — launch integrated demo (includes depth bridge)
ros2 launch launch/final_integrated_demo.launch.py

# SLAM node reads config/slam_backend.yaml → yoloslam_rkf_rgbd
```

If depth is missing, the node falls back to monocular ORB (`tracking_state=MONO`).

---

## 6. Use in your own Python code

```python
import numpy as np
from slam_module.slam_factory import create_backend

slam = create_backend("yoloslam_rkf_rgbd")  # or None → reads slam_backend.yaml
slam.reset()

# rgb: H×W×3 uint8 BGR or RGB (match your pipeline)
# depth_m: H×W float32, depths in **meters**
result = slam.track(rgb, depth_m, timestamp=0.0)

print(result.position)           # [x, y, z] world frame
print(result.quaternion_xyzw)    # orientation
print(result.tracking_state)     # e.g. OK, LOST
```

### Tune intrinsics (`config/slam_backend.yaml`)

```yaml
backend: yoloslam_rkf_rgbd

yoloslam_rkf_rgbd:
  fx: 525.0
  fy: 525.0
  cx: 319.5
  cy: 239.5
  max_features: 2200
  keyframe_trans_m: 0.06
  keyframe_rot_deg: 6.0
  max_map_points: 6000
  recovery_match_threshold: 7
  max_depth_m: 5.0   # increase for outdoor / high altitude (PyBullet uses 20)
```

**Rule:** `fx`, `fy`, `cx`, `cy`, and `max_depth_m` must match your camera. Wrong values → poor tracking even if the code runs.

---

## 7. Copy figures for LaTeX paper

```bash
python scripts/build_paper_bundle.py
```

Copies SLAM plots to `paper/figures/` and refreshes `compre/paper_results.json`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: slam_module` | Set `PYTHONPATH=src` |
| `Unknown SLAM backend` | Use `yoloslam_rkf_rgbd`; check `config/slam_backend.yaml` |
| TUM dataset not found | Download/extract to `datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz/` |
| PyBullet SLAM drift / no depth | Set `max_depth_m: 20` in `config/pybullet_demo.yaml` |
| Very slow benchmark on Windows | Run via WSL + native FS copy (`run_slam_comparison_wsl.sh`) |
| `evo` errors on short runs | Use ≥100 frames; full paper eval uses 792 |
| Poor accuracy on new camera | Recalibrate fx/fy/cx/cy; verify depth is in meters |

---

## What is portable vs what is not

| Works on most machines | Needs extra setup |
|------------------------|-------------------|
| YOLOSLAM-RKF SLAM core | ROS 2 full stack |
| TUM benchmark script | Real drone flight (not flight-tested) |
| PyBullet rgbd_slam demo | ORB-SLAM3 / DROID / RTAB-Map backends |
| `pip install -r requirements.txt` | Reproducing paper landing batch (uses scene_semantic + truth pose) |

---

## Reference

| Item | Location |
|------|----------|
| Implementation | `src/slam_module/backends/yoloslam_rkf_rgbd.py` |
| Default config | `config/slam_backend.yaml` |
| PyBullet config | `config/pybullet_demo.yaml` |
| Benchmark harness | `scripts/run_slam_comparison.py` |
| Paper metrics | `compre/slam_benchmark/comparison.json` |

**Benchmark result (TUM freiburg1_xyz, 792 frames):** ATE RMSE **0.0573 m**, RPE RMSE **0.0409 m**.
