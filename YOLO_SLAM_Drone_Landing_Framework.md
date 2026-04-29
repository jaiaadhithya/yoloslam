# YOLO-SLAM Integrated Framework for Autonomous Drone Landing

## Conference Paper Project — Complete Build Specification

---

## 1. Project Overview

**Title (Working):** *Vision-Guided Autonomous UAV Landing via Integrated YOLO Detection and Visual SLAM in Simulated Environments*

**Core Idea:** Combine YOLOv8 object detection (to identify and localize a landing pad) with ORB-SLAM3 (to maintain spatial awareness and trajectory estimation) on a simulated PX4-powered quadrotor in Gazebo, demonstrating that the fused pipeline achieves more accurate, robust, and repeatable autonomous landings than either subsystem alone.

**Simulation Stack:**

| Layer | Tool |
|---|---|
| Physics / World | Gazebo Harmonic (or Gazebo Classic 11) |
| Autopilot | PX4-Autopilot (v1.15+) via SITL |
| Middleware | ROS 2 Humble |
| Bridge | px4_ros_com + px4_msgs (micro-XRCE-DDS) |
| Drone Model | PX4 default Iris quadrotor (SDF) |
| Camera Sensor | Gazebo RGB camera plugin (640×480 @ 30 Hz) |
| Depth (optional) | Gazebo depth camera plugin for SLAM bootstrapping |

> **Why PX4 + Gazebo instead of CARLA?** CARLA is vehicle-centric and lacks native multirotor dynamics. PX4 SITL + Gazebo is the industry-standard UAV simulation stack with realistic IMU, GPS, barometer, and camera sensor plugins. It also publishes ground-truth poses for quantitative evaluation — critical for a paper.

---

## 2. Repository Structure to Generate

```
yolo_slam_landing/
├── README.md                         # Setup, build, run instructions
├── requirements.txt                  # Python deps (ultralytics, opencv, numpy, matplotlib, scipy)
├── docker/
│   ├── Dockerfile                    # Full env: ROS2 + PX4 + Gazebo + Python deps
│   └── docker-compose.yml
│
├── worlds/
│   ├── landing_pad_world.sdf         # Gazebo world: flat ground + helipad + obstacles + lighting
│   └── models/
│       └── landing_pad/
│           ├── model.sdf
│           ├── model.config
│           └── meshes/
│               └── helipad.dae       # Textured landing pad mesh (H-circle)
│
├── config/
│   ├── yolo_params.yaml              # Confidence threshold, NMS, target class, model path
│   ├── slam_params.yaml              # ORB-SLAM3 vocabulary, camera intrinsics, feature count
│   ├── landing_controller.yaml       # PID gains, descent rate, alignment tolerance
│   └── camera_intrinsics.yaml        # fx, fy, cx, cy, distortion coeffs (match Gazebo plugin)
│
├── launch/
│   ├── full_pipeline.launch.py       # Launches everything: Gazebo, PX4, YOLO, SLAM, Controller
│   ├── yolo_only.launch.py           # YOLO detection standalone (for ablation)
│   ├── slam_only.launch.py           # SLAM standalone (for ablation)
│   └── evaluation.launch.py          # Runs N trials, records bags, dumps metrics
│
├── src/
│   ├── yolo_detector/
│   │   ├── __init__.py
│   │   ├── yolo_node.py              # ROS2 node: subscribes to /camera/image_raw, publishes detections
│   │   ├── detector.py               # YOLOv8 inference wrapper (ultralytics)
│   │   └── msg/
│   │       └── Detection.msg         # bbox (x,y,w,h), confidence, class_id, timestamp
│   │
│   ├── slam_module/
│   │   ├── __init__.py
│   │   ├── slam_node.py              # ROS2 node: runs ORB-SLAM3 mono/stereo, publishes pose
│   │   ├── slam_wrapper.py           # Python/C++ binding to ORB-SLAM3
│   │   ├── trajectory_logger.py      # Logs estimated vs ground-truth trajectory to CSV
│   │   └── msg/
│   │       └── SLAMPose.msg          # 6-DOF pose + covariance + tracking state
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── fusion_node.py            # Fuses YOLO bbox + SLAM pose → landing-pad 3D position
│   │   ├── kalman_filter.py          # EKF for fusing detection + pose estimates
│   │   └── msg/
│   │       └── LandingTarget.msg     # 3D position of pad in world frame + uncertainty ellipse
│   │
│   ├── landing_controller/
│   │   ├── __init__.py
│   │   ├── controller_node.py        # Consumes LandingTarget, sends velocity/position cmds to PX4
│   │   ├── pid.py                    # PID controller (x, y, z, yaw)
│   │   └── state_machine.py          # SEARCH → APPROACH → ALIGN → DESCEND → LANDED
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                # Computes ATE, RPE, landing error, precision, recall
│       ├── run_trials.py             # Orchestrates N repeated landings, varies conditions
│       ├── plot_results.py           # Generates publication-quality figures (matplotlib)
│       └── statistical_tests.py      # Paired t-tests, Wilcoxon, confidence intervals
│
├── models/
│   └── yolov8_landing_pad.pt         # Fine-tuned YOLOv8n weights (or download script)
│
├── datasets/
│   ├── generate_dataset.py           # Script to fly drone, capture frames, auto-label via ground truth
│   ├── landing_pad_dataset/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── data.yaml                     # Ultralytics dataset config
│
├── results/                          # AUTO-GENERATED after evaluation runs
│   ├── yolo_standalone/
│   │   ├── detection_log.csv
│   │   ├── precision_recall.png
│   │   └── confusion_matrix.png
│   ├── slam_standalone/
│   │   ├── trajectory_est.csv
│   │   ├── trajectory_gt.csv
│   │   ├── ate_plot.png
│   │   └── rpe_plot.png
│   ├── combined/
│   │   ├── landing_errors.csv
│   │   ├── landing_accuracy_boxplot.png
│   │   ├── trajectory_with_detections.png
│   │   └── state_machine_timeline.png
│   └── comparison/
│       ├── ablation_table.csv        # YOLO-only vs SLAM-only vs Fused
│       ├── ablation_bar_chart.png
│       └── statistical_significance.txt
│
├── paper/
│   ├── figures/                      # Symlinks or copies from results/
│   ├── main.tex                      # LaTeX source (IEEE conference format)
│   ├── references.bib
│   └── Makefile                      # `make pdf` to compile
│
└── scripts/
    ├── install_px4.sh                # Clone + build PX4-Autopilot
    ├── install_orbslam3.sh           # Clone + build ORB-SLAM3 with ROS2 wrapper
    ├── download_yolo_weights.sh      # Fetch pretrained YOLOv8n from Ultralytics
    └── record_bag.sh                 # Record relevant topics to rosbag2
```

---

## 3. Module-by-Module Build Instructions

### 3.1 — Gazebo World & Landing Pad Model

**Goal:** A reproducible simulation environment with a clearly textured landing pad.

**What to generate:**

1. **`worlds/landing_pad_world.sdf`** — A Gazebo SDF world containing:
   - A 50 m × 50 m ground plane with asphalt texture.
   - The landing pad model placed at a known world coordinate (e.g., `x=0, y=0, z=0`). Record this as ground-truth.
   - At least 3 static obstacle objects (boxes, cylinders) placed around the pad to give SLAM visual features and add realism.
   - A directional sun light + one point light to simulate outdoor conditions.
   - A wind plugin (optional, for robustness experiments).

2. **`worlds/models/landing_pad/`** — A Gazebo model:
   - A 1.5 m × 1.5 m flat square with an "H" or concentric-circle helipad texture (use a .png applied to a plane mesh or a .dae Collada file).
   - The texture must be distinctive enough for YOLO to detect, and provide ORB features for SLAM.
   - Include `model.config` with proper naming.

3. **Drone camera setup** — In the Iris SDF (or via an overlay model), attach:
   - A downward-facing RGB camera at `roll=0, pitch=-90°, yaw=0` relative to body frame.
   - Resolution: 640×480, FOV: 80°, update rate: 30 Hz.
   - Publishes to `/drone/camera/image_raw` (sensor_msgs/Image).
   - A downward-facing depth camera (same pose) publishing to `/drone/depth/image_raw` (optional, for dense SLAM or depth-aided detection).

**Verification:** Launch the world with `ros2 launch`, spawn the Iris, take off to 10 m via QGroundControl or MAVSDK, and confirm the camera feed shows the pad clearly via `rqt_image_view`.

---

### 3.2 — YOLO-Based Landing Pad Detection (Module 1 — Standalone Results)

**Goal:** Detect and localize the landing pad in each camera frame with bounding box, confidence, and pixel-center coordinates.

**What to generate:**

1. **`datasets/generate_dataset.py`**
   - Automate data collection: script the drone to fly a lawnmower pattern over the pad at altitudes 3–25 m, varied yaw, capturing ~2000 frames.
   - Auto-label using ground-truth pad position + camera intrinsics + drone pose (from `/mavros/local_position/pose` or PX4 ground truth) to project the pad's 3D corners into pixel coordinates → YOLO-format bounding box.
   - Split 80/20 train/val. Write `data.yaml`.

2. **`src/yolo_detector/detector.py`**
   - Load YOLOv8n (nano, for real-time) via `ultralytics` Python API.
   - Fine-tune on the generated dataset: `model.train(data='data.yaml', epochs=100, imgsz=640)`.
   - Save best weights to `models/yolov8_landing_pad.pt`.
   - Inference method: takes an OpenCV image, returns list of `(x_center, y_center, width, height, confidence, class_id)`.

3. **`src/yolo_detector/yolo_node.py`** — ROS 2 Node:
   - Subscribes to `/drone/camera/image_raw`.
   - Runs `detector.py` inference per frame.
   - Publishes `Detection.msg` on `/yolo/detections`.
   - Publishes annotated image (bounding box drawn) on `/yolo/annotated_image` for visualization.
   - Log per-frame latency.

4. **Standalone evaluation script** (in `src/evaluation/`):
   - Run the YOLO node while the drone hovers at various altitudes.
   - Compute and save:
     - **Precision and Recall** at IoU thresholds 0.5 and 0.75.
     - **mAP@0.5** and **mAP@0.5:0.95**.
     - **Inference FPS** (must be > 15 Hz for real-time claim).
     - **Detection range** — max altitude where recall > 0.9.
     - **Pixel-error** — Euclidean distance between predicted bbox center and ground-truth projected center.
   - Generate: `precision_recall.png`, `confusion_matrix.png`, a CSV log, and a markdown summary.

**Paper Figures from this module:**
- Fig 1: Sample detection frames at altitudes 5 m, 10 m, 20 m.
- Fig 2: Precision-Recall curve.
- Table I: Detection metrics (mAP, FPS, detection range).

---

### 3.3 — Visual SLAM (Module 2 — Standalone Results)

**Goal:** Use ORB-SLAM3 to estimate the drone's 6-DOF trajectory in real time from the onboard camera.

**What to generate:**

1. **`scripts/install_orbslam3.sh`**
   - Clone ORB-SLAM3 (https://github.com/UZ-SLAMLab/ORB_SLAM3).
   - Build with ROS 2 wrapper (use community `orbslam3_ros2` or write a minimal one).
   - Download ORB vocabulary (`ORBvoc.txt`).

2. **`config/slam_params.yaml`**
   - Camera intrinsics matching the Gazebo camera plugin exactly.
   - ORB feature count: 2000.
   - Scale factor: 1.2, nLevels: 8.
   - Monocular mode (or stereo if using depth camera).

3. **`src/slam_module/slam_node.py`** — ROS 2 Node:
   - Subscribes to `/drone/camera/image_raw`.
   - Feeds frames to ORB-SLAM3 via `slam_wrapper.py`.
   - Publishes estimated pose as `geometry_msgs/PoseStamped` on `/slam/pose`.
   - Publishes tracking state (OK, LOST, INITIALIZING) on `/slam/state`.
   - Publishes sparse point cloud on `/slam/map_points` (for visualization in RViz).

4. **`src/slam_module/trajectory_logger.py`**
   - Records timestamped estimated poses and ground-truth poses (from PX4 SITL `/mavros/local_position/pose`) to CSV files aligned by timestamp.

5. **Standalone evaluation** (in `src/evaluation/`):
   - Fly a predefined trajectory (e.g., figure-8 at 10 m altitude, then descend to 3 m).
   - Compute:
     - **Absolute Trajectory Error (ATE)** — RMSE of position error after SE(3) alignment (use `evo` Python package).
     - **Relative Pose Error (RPE)** — drift per meter traveled.
     - **Tracking loss rate** — % of frames where SLAM state != OK.
     - **Scale accuracy** (for monocular: compare estimated scale factor vs true).
     - **Map point density** over time.
   - Generate: `ate_plot.png`, `rpe_plot.png`, trajectory overlay plot (estimated vs ground truth), CSV logs.

**Paper Figures from this module:**
- Fig 3: Estimated vs ground-truth trajectory (3D or top-down view).
- Fig 4: ATE over time.
- Table II: SLAM metrics (ATE RMSE, RPE, tracking loss %, FPS).

---

### 3.4 — Sensor Fusion: YOLO + SLAM (Combined Pipeline)

**Goal:** Fuse YOLO detections (2D pixel target) with SLAM pose (6-DOF camera state) to produce a 3D world-frame estimate of the landing pad position, filtered over time.

**What to generate:**

1. **`src/fusion/fusion_node.py`** — ROS 2 Node:
   - Subscribes to `/yolo/detections` and `/slam/pose`.
   - Time-synchronizes using `message_filters.ApproximateTimeSynchronizer`.
   - On each synchronized pair:
     - Backprojects the YOLO bbox center from pixel to a 3D ray using camera intrinsics.
     - If depth is available: use depth at bbox center to get 3D point directly.
     - If monocular only: use the known pad size + bbox size to estimate depth via pinhole model (`Z = (f × real_size) / pixel_size`).
     - Transforms the 3D point from camera frame → world frame using the SLAM pose.
   - Feeds the 3D position into an Extended Kalman Filter.
   - Publishes `LandingTarget.msg` on `/fusion/landing_target` containing:
     - `position` (x, y, z in world frame).
     - `covariance` (3×3 matrix).
     - `confidence` (from YOLO).
     - `is_valid` (true only if SLAM tracking OK and YOLO confidence above threshold).

2. **`src/fusion/kalman_filter.py`**
   - State: `[x, y, z, vx, vy, vz]` of the landing pad in world frame.
   - Constant-velocity motion model (pad is static, so velocity should converge to 0 — this handles noise).
   - Measurement model: 3D position from backprojection.
   - Measurement noise: derived from YOLO confidence and SLAM covariance.
   - Process noise: small (pad doesn't move).
   - Handle measurement rejection (Mahalanobis distance gating) for outlier detections.

3. **Fusion evaluation**:
   - Compare fused landing-pad position estimate vs ground truth over time.
   - Compute:
     - **3D position RMSE** of pad estimate.
     - **Convergence time** — time until estimate is within 0.2 m of truth.
     - **Stability** — variance of estimate over a 5-second hover window.
   - Compare against:
     - YOLO-only baseline (pixel center converted to position using altitude from barometer, no SLAM).
     - SLAM-only baseline (pad position from map points, no YOLO semantic identification).

**Paper Figures from this module:**
- Fig 5: System architecture block diagram (the main figure).
- Fig 6: Fused pad position estimate vs ground truth over time (x, y, z subplots).
- Fig 7: Estimation error comparison — YOLO-only vs SLAM-only vs Fused (line plot or boxplot).

---

### 3.5 — Landing Controller & State Machine

**Goal:** Use the fused landing target to autonomously guide the drone from cruise altitude to touchdown.

**What to generate:**

1. **`src/landing_controller/state_machine.py`** — Finite State Machine:

   ```
   SEARCH ──→ APPROACH ──→ ALIGN ──→ DESCEND ──→ LANDED
     ↑            │           │          │
     └────────────┴───────────┴──────────┘  (target lost → SEARCH)
   ```

   - **SEARCH:** Drone hovers at 15 m, rotates slowly (yaw scan). Transitions to APPROACH when YOLO detects pad with confidence > 0.7.
   - **APPROACH:** Fly toward the pad's estimated (x, y) at current altitude. Transition to ALIGN when horizontal distance < 1.0 m.
   - **ALIGN:** Hold position, fine-tune (x, y) alignment using fused estimate. Transition to DESCEND when horizontal error < 0.15 m for 2 consecutive seconds.
   - **DESCEND:** Descend at 0.3 m/s while maintaining (x, y) alignment. If horizontal error exceeds 0.5 m, return to ALIGN. Transition to LANDED when altitude < 0.2 m.
   - **LANDED:** Disarm motors. Log final position error.
   - **Any state → SEARCH:** If target is lost (no valid detection for 3 seconds).

2. **`src/landing_controller/controller_node.py`** — ROS 2 Node:
   - Subscribes to `/fusion/landing_target` and `/mavros/local_position/pose`.
   - Runs the state machine.
   - Publishes velocity setpoints to PX4 via `/mavros/setpoint_velocity/cmd_vel` (or OFFBOARD position control).
   - Publishes current state on `/controller/state` for logging.

3. **`src/landing_controller/pid.py`**
   - Independent PID controllers for x, y, z, yaw.
   - Configurable gains via `config/landing_controller.yaml`.
   - Anti-windup on integrator.

---

### 3.6 — Evaluation & Ablation Study

**Goal:** Generate all quantitative data needed for the paper's results section.

**What to generate:**

1. **`src/evaluation/run_trials.py`**
   - Orchestrates automated experiments:
     - **Condition A — YOLO-only landing:** Uses YOLO bbox center + barometer altitude for position estimate. No SLAM.
     - **Condition B — SLAM-only landing:** Uses SLAM trajectory + map-point-based pad identification. No YOLO.
     - **Condition C — Fused (proposed method):** Full pipeline.
   - For each condition, run **N = 30 trials** with randomized:
     - Initial drone position (within 10 m radius, altitude 12–18 m).
     - Lighting conditions (sun angle varied in Gazebo).
     - Wind disturbance (if wind plugin enabled).
   - Each trial records:
     - Landing position error (Euclidean distance from pad center at touchdown).
     - Landing time (from SEARCH start to LANDED).
     - Whether landing succeeded (touched down within 0.5 m of center).
     - SLAM tracking loss events.
     - YOLO detection loss events.
     - Full trajectory (rosbag).

2. **`src/evaluation/metrics.py`**
   - Functions to compute from trial logs:
     - **Mean / Median / Std landing error** per condition.
     - **Success rate** (% landings within 0.5 m).
     - **Mean landing time.**
     - **Precision / Recall** of YOLO at different altitudes.
     - **ATE / RPE** of SLAM per trial.

3. **`src/evaluation/statistical_tests.py`**
   - Paired t-test (or Wilcoxon signed-rank if non-normal) comparing:
     - Fused vs YOLO-only landing error.
     - Fused vs SLAM-only landing error.
   - Report p-values and 95% confidence intervals.
   - Effect size (Cohen's d).

4. **`src/evaluation/plot_results.py`**
   - Generate publication-quality figures (300 DPI, IEEE column width):
     - **Ablation bar chart:** Mean landing error ± std for each condition.
     - **Box plot:** Landing error distributions per condition.
     - **Trajectory overlay:** Top-down view of all 30 landing trajectories per condition, color-coded.
     - **State machine timeline:** Example timeline showing state transitions during one landing.
     - **Cumulative success plot:** Success rate vs allowed error threshold.
   - Use `matplotlib` with `pgf` backend or `seaborn` for clean styling.
   - All plots must use consistent colors: e.g., YOLO-only = blue, SLAM-only = orange, Fused = green.

**Paper Figures and Tables from this module:**
- Fig 8: Landing trajectory comparison (top-down, 3 conditions).
- Fig 9: Landing error box plots.
- Fig 10: State machine timeline for a sample trial.
- Table III: Ablation results (mean error, success rate, time, p-values).

---

## 4. Paper Structure (IEEE Conference Format)

Generate `paper/main.tex` in IEEE conference format (`\documentclass[conference]{IEEEtran}`) with the following section structure and content guidance:

### Abstract (~150 words)
- Problem: Autonomous precision landing remains challenging due to perceptual drift and lack of semantic awareness.
- Approach: Integrated YOLOv8 detection + ORB-SLAM3 via EKF fusion for robust 3D landing-pad localization.
- Results: Cite key numbers from ablation (e.g., "reduces landing error by X% over detection-only baseline, achieving Y cm mean accuracy in Z simulated trials").
- Significance: Demonstrates that semantic–geometric fusion outperforms either modality in isolation.

### I. Introduction
- Motivation: growing need for autonomous drone landing in GPS-denied or precision-critical scenarios.
- Problem statement: neither detection alone (no depth, no drift correction) nor SLAM alone (no semantic target identification) is sufficient.
- Contribution summary (3 bullet points): (1) fused YOLO+SLAM pipeline, (2) EKF-based 3D target estimation, (3) comprehensive ablation in realistic simulation.

### II. Related Work
- Visual servoing for landing (cite ArUco/AprilTag-based approaches).
- YOLO-family detectors on UAVs.
- Visual SLAM for UAVs (ORB-SLAM, VINS-Mono, etc.).
- Fusion approaches in autonomous landing.
- Gap: few works rigorously ablate detection-only vs SLAM-only vs fused in a controlled simulation with statistical significance.

### III. System Architecture
- Block diagram (Fig 5).
- Subsections: A. YOLO Detection, B. ORB-SLAM3, C. EKF Fusion, D. Landing Controller.
- Mathematical formulation of the backprojection and EKF.

### IV. Experimental Setup
- Simulation environment description (Gazebo, PX4, Iris, camera specs).
- Dataset generation for YOLO training.
- Trial design (N=30, randomized conditions).
- Metrics definitions.

### V. Results
- A. YOLO standalone results (Table I, Fig 1–2).
- B. SLAM standalone results (Table II, Fig 3–4).
- C. Fused pipeline + landing results (Table III, Fig 6–10).
- D. Statistical significance analysis.
- E. Failure mode analysis (when does fusion still fail?).

### VI. Discussion
- Why fusion helps: SLAM corrects for camera motion / drift; YOLO provides semantic grounding.
- Limitations: monocular scale ambiguity, simulation-to-real gap, computational load.
- Future work: real-world transfer, moving landing pads, multi-target scenarios, onboard deployment (Jetson).

### VII. Conclusion
- Restate contribution and key result in 3–4 sentences.

### References
- Populate `references.bib` with ~20–25 relevant citations (the coder should add proper BibTeX entries from: original YOLO papers, ORB-SLAM3 paper, PX4 paper, Gazebo, relevant landing papers, EKF/sensor-fusion textbooks).

---

## 5. Build & Run Instructions (for README.md)

Generate a `README.md` that includes:

1. **Prerequisites:** Ubuntu 22.04, ROS 2 Humble, Gazebo Harmonic, PX4-Autopilot (SITL), Python 3.10+, CUDA 11.8+ (for YOLO GPU inference).
2. **One-command Docker setup** (preferred path): `docker compose up` should bring up the full stack.
3. **Manual setup** steps referencing the install scripts.
4. **How to run each experiment:**
   ```bash
   # Full pipeline demo (single landing)
   ros2 launch yolo_slam_landing full_pipeline.launch.py

   # YOLO-only ablation
   ros2 launch yolo_slam_landing yolo_only.launch.py

   # SLAM-only ablation
   ros2 launch yolo_slam_landing slam_only.launch.py

   # Run 30-trial evaluation for all conditions
   ros2 launch yolo_slam_landing evaluation.launch.py num_trials:=30

   # Generate plots and tables
   python3 src/evaluation/plot_results.py --results-dir results/
   ```
5. **How to compile the paper:** `cd paper && make pdf`

---

## 6. Key Technical Specifications & Constraints

These values should be used consistently across all modules:

| Parameter | Value | Rationale |
|---|---|---|
| YOLO model | YOLOv8n (nano) | Real-time on drone hardware |
| YOLO input size | 640×640 | Standard Ultralytics default |
| YOLO confidence threshold | 0.5 | Balance precision/recall |
| YOLO NMS IoU threshold | 0.45 | Standard |
| Camera resolution | 640×480 | Reasonable for onboard compute |
| Camera FPS | 30 | Standard |
| Camera FOV | 80° horizontal | Typical drone camera |
| SLAM mode | Monocular (primary), stereo (secondary) | Mono is more challenging = better paper; stereo for comparison |
| ORB features | 2000 | ORB-SLAM3 default |
| EKF state | [x, y, z, vx, vy, vz] | 6D state for pad position |
| EKF update rate | 30 Hz (synced with camera) | Real-time |
| Landing pad size | 1.5 m × 1.5 m | Standard helipad size |
| Descent rate | 0.3 m/s | Safe, controllable |
| Alignment tolerance | 0.15 m | Precision landing threshold |
| Number of trials | 30 per condition | Enough for statistical significance |
| Success threshold | 0.5 m from center | Standard in landing literature |

---

## 7. Dependency Versions (Pin These)

```txt
# Python
ultralytics==8.2.0
opencv-python==4.9.0.80
numpy==1.26.4
matplotlib==3.8.3
scipy==1.12.0
seaborn==0.13.2
evo==1.28.0
transforms3d==0.4.1
filterpy==1.4.5

# ROS 2
ros-humble-desktop
px4-ros-com
px4-msgs

# System
PX4-Autopilot v1.15.x
ORB-SLAM3 (latest main branch)
Gazebo Harmonic (or gazebo-classic-11)
```

---

## 8. Execution Order for the Agentic Coder

Follow this sequence strictly — each step builds on the previous:

1. **Environment** — Generate `Dockerfile` and `docker-compose.yml`. Verify ROS 2 + Gazebo + PX4 SITL boots and the Iris drone hovers.
2. **World** — Generate the Gazebo world and landing pad model. Verify the pad is visible from the drone camera.
3. **YOLO training data** — Generate `generate_dataset.py`. Fly the drone, collect and label frames. Verify labels are correct by visualizing a few.
4. **YOLO training** — Fine-tune YOLOv8n. Verify mAP > 0.9 on val set.
5. **YOLO ROS node** — Generate `yolo_node.py`. Verify detections publish at > 15 Hz.
6. **YOLO standalone eval** — Run and save metrics + plots. These go into the paper.
7. **ORB-SLAM3 setup** — Build and wrap. Verify it tracks in real time on the Gazebo camera feed.
8. **SLAM ROS node** — Generate `slam_node.py`. Verify pose publishes and roughly matches ground truth.
9. **SLAM standalone eval** — Run and save ATE/RPE + plots. These go into the paper.
10. **Fusion node** — Generate `fusion_node.py` + `kalman_filter.py`. Verify fused pad position converges to ground truth.
11. **Landing controller** — Generate state machine + PID controller. Verify the drone lands on the pad in a single demo run.
12. **Full evaluation** — Run 30 trials × 3 conditions. Save all data.
13. **Plots and tables** — Generate all paper figures.
14. **Statistical tests** — Run significance tests. Save results.
15. **LaTeX paper** — Generate `main.tex` with all figures and tables referenced. Compile to PDF.
16. **README** — Generate final README with complete instructions.

---

## 9. Common Pitfalls to Avoid

- **PX4 time sync**: Use `timesync` bridge between PX4 and ROS 2, or timestamps will be off and message_filters will drop everything.
- **SLAM scale in monocular mode**: ORB-SLAM3 monocular has scale ambiguity. Initialize scale using the first barometer reading + known takeoff height, or use stereo/depth mode.
- **YOLO on GPU in Docker**: Pass `--gpus all` in `docker-compose.yml` and install `nvidia-container-toolkit`.
- **Gazebo rendering in headless mode**: Use `LIBGL_ALWAYS_SOFTWARE=1` or `gzserver` (no GUI) for batch trials. Camera plugin still renders.
- **EKF divergence**: If YOLO gives false positives, the EKF can diverge. Implement Mahalanobis gating and require N consecutive detections before updating.
- **OFFBOARD mode in PX4**: Must send setpoints at > 2 Hz before switching to OFFBOARD, or PX4 rejects the mode switch.
- **Frame conventions**: PX4 uses NED (North-East-Down). ROS uses ENU (East-North-Up). ORB-SLAM3 uses a camera convention. Document and handle all transforms explicitly via `tf2_ros`.

---

## 10. Deliverables Checklist

When the project is complete, the following must exist and work:

- [ ] Docker environment boots in one command
- [ ] Gazebo world loads with visible landing pad
- [ ] Drone takes off and camera feed is accessible
- [ ] YOLO detects landing pad in real-time (>15 FPS)
- [ ] YOLO standalone metrics and plots are generated
- [ ] ORB-SLAM3 tracks drone pose in real-time
- [ ] SLAM standalone metrics and plots are generated
- [ ] Fusion node produces 3D landing-pad estimates
- [ ] Landing controller flies drone to pad and lands
- [ ] 30 trials × 3 conditions run successfully
- [ ] All evaluation plots and tables are generated
- [ ] Statistical significance tests pass (p < 0.05 for fused vs baselines)
- [ ] LaTeX paper compiles to a clean PDF
- [ ] README documents everything needed to reproduce
