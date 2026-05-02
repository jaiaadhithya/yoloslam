# NEXT STEPS PROMPT — Getting Paper-Ready Results From Current State

## Immediate Run Block (Do This First)

```bash
# 1) Try headless PX4+Gazebo first
cd ~/PX4-Autopilot
HEADLESS=1 PX4_GZ_WORLD=baylands make px4_sitl gz_x500_depth

# 2) If headless works at usable rate, record once and go offline forever
cd /mnt/c/Users/jaiaa/OneDrive/Desktop/stuff/study_project/yoloslam
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 bag record /camera /mavros/local_position/pose /mavros/imu/data -o flight_dataset_1
```

If this does not produce usable FPS quickly, stop tuning Gazebo and move to public datasets (Semantic Drone + EuRoC), then continue offline evaluation.

---

## The Problem

Gazebo + PX4 SITL is running at frames-per-hour on this WSL setup. We cannot wait for real-time simulation to produce results. We need to restructure the workflow so every module can be developed, tested, and evaluated **independently and offline**, then integrated for a final demo.

The paper needs three categories of results: (1) YOLO standalone, (2) SLAM standalone, (3) fused landing — and none of them require live real-time Gazebo to produce publishable data.

---

## Phase 1: Fix or Bypass the Gazebo Performance Problem

Try these in order. Stop at whichever one works:

### Option A — Headless Gazebo + Record a Dataset (Preferred)

The camera plugin renders even without the GUI. The GUI is what kills WSL.

```bash
# Run PX4 SITL with Gazebo in headless/server-only mode
HEADLESS=1 PX4_GZ_WORLD=baylands make px4_sitl gz_x500_depth
```

If `HEADLESS=1` is not supported by the PX4 GZ integration, try:

```bash
# Launch gz server only (no GUI rendering)
export GZ_SIM_RENDER_ENGINE_SERVER=ogre2
export DISPLAY=  # unset display to prevent GUI
```

Or use the older Gazebo Classic path if Harmonic is too heavy:

```bash
PX4_SIM_SPEED_FACTOR=1 HEADLESS=1 make px4_sitl gazebo-classic_iris
```

**Goal:** Get PX4 flying and the camera topic publishing at >=5 FPS in headless mode. Record a rosbag of the flight:

```bash
ros2 bag record /camera /mavros/local_position/pose /mavros/imu/data -o flight_dataset_1
```

Then all YOLO/SLAM/fusion work happens offline against this bag. No more waiting for live Gazebo frames.

### Option B — Use a Pre-Recorded Aerial Dataset (If Gazebo Won't Cooperate At All)

If we absolutely cannot get Gazebo running at usable speed, use publicly available aerial/drone datasets. This is common and accepted in conference papers — many SLAM and detection papers evaluate on benchmark datasets, not their own simulator.

**For YOLO terrain classification, use any of:**

| Dataset | Why It Works |
|---|---|
| **Semantic Drone Dataset** (TU Graz) | 400 high-res images, 20+ classes (grass, trees, dirt, pavement, person, car, water, etc.), taken from 5–30 m altitude. Already has pixel-level annotations. |
| **UAVid** | Urban aerial video dataset with semantic segmentation labels. |
| **Aerial Semantic Segmentation Dataset (FloodNet / RescueNet)** | Disaster-response aerial imagery, relevant to unknown-terrain landing narrative. |

Download one. Convert annotations to YOLO bbox format. Train YOLOv8n. This gives publishable detection metrics.

**For SLAM trajectory evaluation, use:**

| Dataset | Why It Works |
|---|---|
| **EuRoC MAV Dataset** | Standard visual-inertial benchmark with high-quality trajectory ground truth. |
| **TUM RGB-D** (selected sequences) | Standard SLAM benchmark, acceptable for trajectory error analysis. |

Run ORB-SLAM3, compute ATE/RPE with `evo`, and use those metrics directly in the paper.

### Option C — Dual Boot / Native Ubuntu

If Options A and B are not acceptable, install Ubuntu 22.04 natively (dual boot or separate drive). Gazebo performance on native Linux is significantly better than WSL.

---

## Phase 2: YOLO Terrain Classification — Get Real Results

This is independent of Gazebo. Do this now regardless of path.

### Step 2.1 — Get a labeled dataset

**Path A (from Gazebo bag):**
- Extract frames from rosbag and label based on known world structure or semantic labels.

**Path B (public dataset):**
- Download Semantic Drone Dataset.
- Convert masks -> YOLO boxes.
- Map classes to project labels (e.g., `grass` -> `grass_field`, `car` -> `vehicle`, `water` -> `water`).

### Step 2.2 — Train YOLOv8n

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(
    data='datasets/terrain_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='terrain_detector'
)
```

Save best weights to `models/yolov8_terrain.pt`.

### Step 2.3 — Generate standalone YOLO results for paper

Produce:
1. Per-class precision/recall/F1
2. mAP@0.5 and mAP@0.5:0.95
3. Confusion matrix
4. Precision-recall curves
5. Detection sample images
6. Inference speed (FPS)
7. Optional recall-vs-altitude plot

Store in `results/yolo_standalone/`.

---

## Phase 3: SLAM — Get Real Results

Also independent of Gazebo.

### Step 3.1 — Get trajectory data

Path A: rosbag replay + SLAM output trajectory CSV + GT pose topic.  
Path B: EuRoC sequences + provided GT.

### Step 3.2 — Generate standalone SLAM results

```bash
pip install evo --break-system-packages

evo_ape tum ground_truth.txt estimated.txt -vas --plot --plot_mode xy --save_results results/slam_standalone/ate.zip
evo_rpe tum ground_truth.txt estimated.txt --delta 1 --delta_unit m -vas --plot --save_results results/slam_standalone/rpe.zip
```

Produce:
1. ATE statistics
2. RPE statistics
3. Trajectory overlays
4. ATE over time
5. Tracking success ratio
6. Optional map visualization

Store in `results/slam_standalone/`.

---

## Phase 4: Fused Pipeline + Landing Results — Offline Simulation

### Step 4.1 — Offline fusion (`src/evaluation/offline_fusion.py`)

Pipeline:
1. Read rosbag or frame+pose dataset.
2. Run YOLO inference per frame.
3. Run SLAM per frame.
4. Feed YOLO+SLAM into fusion logic as a library.
5. Log grid state, selected zone, and distance to GT safe zone.
6. Save `results/combined/fusion_log.csv`.

### Step 4.2 — Simulated landing (`src/evaluation/simulated_landing.py`)

1. Consume fusion log.
2. Simulate controller state machine and velocity output over time.
3. Integrate simple kinematics (`position += velocity * dt`).
4. Compute landing metrics:
   - landing error,
   - safety success,
   - mission time,
   - state transition timeline.

### Step 4.3 — Ablation (30 trials x 3 conditions)

Conditions:
- **A YOLO-only**
- **B SLAM-only**
- **C Fused**

Randomize:
- start offsets,
- frame subsets/shifts,
- detection/pose noise.

### Step 4.4 — Combined paper outputs

Generate:
1. Ablation table (+ p-values)
2. Landing error boxplot
3. Trajectory comparison plot
4. Safety-grid evolution panels
5. State-machine timeline
6. System architecture diagram

Store in `results/combined/` and `results/comparison/`.

---

## Phase 5: Paper Compilation

### Figures

- Fig 1: architecture
- Fig 2: YOLO detections
- Fig 3: PR curves
- Fig 4: confusion matrix
- Fig 5: SLAM trajectory
- Fig 6: ATE over time
- Fig 7: safety grid evolution
- Fig 8: landing trajectories comparison
- Fig 9: landing error boxplot
- Fig 10: state machine timeline

### Tables

- Table I: YOLO per-class metrics
- Table II: SLAM metrics
- Table III: ablation metrics (+ p-values)

Use IEEE conference format and cite YOLOv8, ORB-SLAM3, PX4, dataset references, and safe landing literature.

---

## Execution Order

1. Try headless Gazebo and record a short bag if usable.
2. If not usable quickly, switch to Semantic Drone + EuRoC.
3. Complete YOLO standalone metrics.
4. Complete SLAM standalone metrics.
5. Complete offline fusion + simulated landing + ablation.
6. Generate all paper figures/tables.
7. Compile LaTeX paper draft.

---

## Important Notes

1. Offline evaluation is standard and methodologically valid.
2. The key contribution is fusion quality, not live simulator framerate.
3. Live Gazebo can remain a demo artifact after quantitative offline results are complete.
4. Do not over-invest in WSL Gazebo optimization if headless mode is still too slow.

