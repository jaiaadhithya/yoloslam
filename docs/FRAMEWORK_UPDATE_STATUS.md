# Framework Update Status

This file tracks compliance against `framework_update_prompt`.

## 1) YOLO multi-class safe/unsafe shift

- Done: detector class map changed to terrain/hazard classes.
- Done: `Detection.msg` includes `class_name` and `is_safe`.
- Done: safety-overlay topic name updated in config (`/yolo/safety_map`).
- Done: dataset defaults moved to terrain naming (`datasets/terrain_dataset`, `models/yolov8_terrain.pt`).
- Pending: true Gazebo semantic-label based auto-labeling (current generator is synthetic placeholder).
- Pending: segmentation model training/evaluation pipeline.

## 2) Gazebo world replacement

- Done: new `worlds/safe_landing_world.sdf` with 100m-style mixed zones.
- Done: includes clearings, water, trees, buildings, vehicles, pedestrians, debris analogs.
- Done: GT zone config added (`config/ground_truth_zones.yaml`).
- Pending: high-fidelity textured meshes (current world is structured but still primitive geometry).
- Pending: overcast world variant.

## 3) Fusion scoring-grid architecture

- Done: `src/fusion/safety_grid.py` with score accumulation + decay.
- Done: `src/fusion/zone_selector.py` for cluster-based zone selection.
- Done: fusion publishes zone score, radius, and observation count fields in runtime dict + message schema update.
- Pending: full ROS message wiring + EKF smoothing of selected centroid in runtime path.

## 4) Landing controller survey-first logic

- Done: state machine includes `SURVEY`, `EVALUATE`, abort behavior.
- Done: `src/landing_controller/survey_planner.py` added (lawnmower waypoints).
- Pending: real PX4 offboard trajectory execution of survey planner waypoints.

## 5) Evaluation and ablation

- Partial: scaffold scripts exist.
- Pending: all requested new metrics (hazard avoidance, zone selection stability, false-safe rate) and dynamic obstacle trial automation.

## 6) Paper framing

- Done: title/abstract/method framing shifted to unknown-terrain safe landing.
- Pending: full related-work and discussion expansions with complete citation coverage.

## 7) File rename/delete/add checklist

- Done: added `worlds/safe_landing_world.sdf`.
- Done: added `config/ground_truth_zones.yaml`, `config/survey_pattern.yaml`.
- Done: added `src/fusion/zone_selector.py`, `src/landing_controller/survey_planner.py`.
- Done: `worlds/models/landing_pad/model.sdf` and `model.config` deleted.
- Pending: remove/replace remaining legacy references and fully migrate any old dataset directories.
