#!/usr/bin/env bash
# PX4 SITL + Gazebo Baylands (gz_x500_depth) + ros_gz_image|ros_gz_bridge + camera recorder + optional motion.
# Target: Ubuntu 22.04 / WSL2 with ROS 2 Humble, Gazebo Harmonic, PX4-Autopilot (gz).
#
# Env:
#   SKIP_MOTION=1       — only PX4+gz+bridge+recorder (debug video pipeline).
#   CAMERA_ONLY_DELAY   — seconds to wait before recorder when SKIP_MOTION=1 (default 18).
#   CAMERA_DIAG_SLEEP   — seconds after /camera exists before diagnostics (default 10).
#   USE_IMAGE_BRIDGE=1 — use ros_gz_image (requires apt ros_gz matching your gz-harmonic; else bridge.log shows
#                        "Unknown message type" and /camera stays empty — default is parameter_bridge).
#   GZ_SIM_RENDER_ENGINE — default ogre2; try ogre on WSL if RGB camera stays at 0 Hz.
#   USE_XVFB=1          — prefix PX4 with xvfb-run -a (install: sudo apt install xvfb).
#   EKF_WAIT            — seconds before px4_demo_motion tries to arm (default 24).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PX4_AUTOPILOT_PATH="${PX4_AUTOPILOT_PATH:-${HOME}/PX4-Autopilot}"
LOW_SPEC="${LOW_SPEC:-1}"
DURATION="${DURATION:-15}"
RECORD_FPS="${RECORD_FPS:-12}"
OUTPUT_MP4="${OUTPUT_MP4:-${REPO_ROOT}/results/gazebo_demo/demo.mp4}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
SKIP_MOTION="${SKIP_MOTION:-0}"
CAMERA_ONLY_DELAY="${CAMERA_ONLY_DELAY:-18}"
CAMERA_DIAG_SLEEP="${CAMERA_DIAG_SLEEP:-10}"
EKF_WAIT="${EKF_WAIT:-24}"

IMX_GZ_BAYLANDS="/world/baylands/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
IMX_GZ_DEFAULT="/world/default/model/x500_depth_0/link/camera_link/sensor/IMX214/image"

BRIDGE_PID=""
MOTION_PID=""
REC_PID=""
HZ_PID=""
PX4_PID=""

if [[ ! -d "${PX4_AUTOPILOT_PATH}" ]]; then
  echo "ERROR: PX4-Autopilot not found at ${PX4_AUTOPILOT_PATH}"
  echo "  export PX4_AUTOPILOT_PATH=/path/to/PX4-Autopilot"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_MP4}")"
mkdir -p "${REPO_ROOT}/results/gazebo_demo"

if [[ "${LOW_SPEC}" == "1" ]]; then
  export HEADLESS="${HEADLESS:-1}"
  export PX4_SIM_SPEED_FACTOR="${PX4_SIM_SPEED_FACTOR:-1}"
  export QT_QUICK_BACKEND="${QT_QUICK_BACKEND:-software}"
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
  # Without GL, gz/Ogre2 often registers the IMX214 topic but publishes 0 Hz → empty /camera.
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
else
  # GUI: let Gazebo/Qt use the normal display (WSLg on Win11, or host X11).
  export HEADLESS=0
  unset QT_QUICK_BACKEND 2>/dev/null || true
  unset QT_QPA_PLATFORM 2>/dev/null || true
fi

cleanup() {
  [[ -n "${HZ_PID}" ]] && kill "${HZ_PID}" 2>/dev/null || true
  [[ -n "${REC_PID}" ]] && kill "${REC_PID}" 2>/dev/null || true
  [[ -n "${BRIDGE_PID}" ]] && kill "${BRIDGE_PID}" 2>/dev/null || true
  [[ -n "${MOTION_PID}" ]] && kill "${MOTION_PID}" 2>/dev/null || true
  [[ -n "${PX4_PID}" ]] && kill "${PX4_PID}" 2>/dev/null || true
  pkill -f "ros_gz_bridge parameter_bridge" 2>/dev/null || true
  pkill -f "ros_gz_image image_bridge" 2>/dev/null || true
  pkill -f "evaluation.px4_demo_motion" 2>/dev/null || true
  pkill -f "gz sim" 2>/dev/null || true
  pkill -f "px4" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "== Gazebo demo: Baylands + x500_depth (low_spec=${LOW_SPEC}) =="
echo "    PX4_AUTOPILOT_PATH=${PX4_AUTOPILOT_PATH}"
echo "    Output: ${OUTPUT_MP4}  duration=${DURATION}s  fps=${RECORD_FPS}"
echo "    SKIP_MOTION=${SKIP_MOTION}  GZ_SIM_RENDER_ENGINE=${GZ_SIM_RENDER_ENGINE:-ogre2}  EKF_WAIT=${EKF_WAIT}"
if [[ "${HEADLESS:-0}" == "1" ]]; then
  echo ""
  echo "    NOTE: HEADLESS=1 — no Gazebo window. To see the sim, run: python -m evaluation.run_gazebo_demo --gui"
  echo "          or: python -m evaluation.run_gazebo_ui"
fi
if [[ "${SKIP_MOTION}" == "0" ]]; then
  echo "    NOTE: PX4 console 'Ready for takeoff!' is normal until px4_demo_motion arms. Watch: tail -f results/gazebo_demo/motion.log"
fi

PX4_CMD=(make px4_sitl gz_x500_depth)
if [[ "${USE_XVFB:-0}" == "1" ]] && command -v xvfb-run >/dev/null 2>&1; then
  PX4_CMD=(xvfb-run -a "${PX4_CMD[@]}")
  echo "    USE_XVFB=1 (xvfb-run)"
elif [[ "${USE_XVFB:-0}" == "1" ]]; then
  echo "WARN: USE_XVFB=1 but xvfb-run not found; install xvfb"
fi

(
  cd "${PX4_AUTOPILOT_PATH}"
  export PX4_GZ_WORLD=baylands
  export GZ_SIM_RENDER_ENGINE="${GZ_SIM_RENDER_ENGINE:-ogre2}"
  exec "${PX4_CMD[@]}"
) &
PX4_PID=$!

echo "    Waiting for Gazebo camera topic (up to 180s)..."
if ! timeout 180 bash -c '
  until command -v gz >/dev/null 2>&1 && gz topic -l 2>/dev/null | grep -qE "IMX214|camera_link.*image"; do sleep 2; done
'; then
  echo "ERROR: Timed out waiting for Gazebo camera image topic."
  exit 1
fi

if [[ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  echo "ERROR: /opt/ros/${ROS_DISTRO}/setup.bash not found. Install ROS 2 ${ROS_DISTRO}."
  exit 1
fi

# ROS setup scripts use unset optional vars (e.g. AMENT_TRACE_SETUP_FILES); this script uses
# set -u, so nounset must be off while sourcing.
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ -f "${REPO_ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/install/setup.bash"
fi
set -u
# Colcon install prepends its paths; force repo src so evaluation/* edits are used.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

BRIDGE_LOG="${REPO_ROOT}/results/gazebo_demo/bridge.log"
BRIDGE_KIND="parameter_bridge"

_start_image_bridge() {
  local gz_topic="$1"
  # Default parameter_bridge: ros_gz_image often breaks with "Unknown message type [N]" in bridge.log when
  # Humble ros_gz debs are not built against the same gz-msgs / sim as PX4's Gazebo Harmonic.
  if [[ "${USE_IMAGE_BRIDGE:-0}" == "1" ]] && ros2 pkg prefix ros_gz_image >/dev/null 2>&1; then
    BRIDGE_KIND="ros_gz_image"
    ros2 run ros_gz_image image_bridge "${gz_topic}" \
      --ros-args \
      -r "${gz_topic}:=/camera" \
      -p qos:=sensor_data \
      >"${BRIDGE_LOG}" 2>&1 &
  else
    BRIDGE_KIND="parameter_bridge"
    ros2 run ros_gz_bridge parameter_bridge \
      "${gz_topic}@sensor_msgs/msg/Image[gz.msgs.Image" \
      --ros-args -r "${gz_topic}:=/camera" \
      >"${BRIDGE_LOG}" 2>&1 &
  fi
  BRIDGE_PID=$!
  echo "    Bridge: ${BRIDGE_KIND}  log=${BRIDGE_LOG}"
}

_start_image_bridge "${IMX_GZ_BAYLANDS}"

echo "    Waiting for ROS /camera (up to 60s)..."
if ! timeout 60 bash -c '
  until ros2 topic list 2>/dev/null | grep -q "^/camera$"; do sleep 1; done
'; then
  echo "WARN: /camera missing on baylands bridge; trying default world name..."
  kill "${BRIDGE_PID}" 2>/dev/null || true
  wait "${BRIDGE_PID}" 2>/dev/null || true
  _start_image_bridge "${IMX_GZ_DEFAULT}"
  timeout 45 bash -c '
    until ros2 topic list 2>/dev/null | grep -q "^/camera$"; do sleep 1; done
  ' || { echo "ERROR: bridge did not expose /camera."; exit 1; }
fi

echo "    Camera settle + diagnostics delay ${CAMERA_DIAG_SLEEP}s…"
sleep "${CAMERA_DIAG_SLEEP}"

CAMERA_DIAG="${REPO_ROOT}/results/gazebo_demo/camera_diagnostics.txt"
echo "    Writing camera diagnostics -> ${CAMERA_DIAG}"
{
  echo "=== $(date -u +"%Y-%m-%dT%H:%M:%SZ")  LOW_SPEC=${LOW_SPEC} HEADLESS=${HEADLESS:-unset} BRIDGE=${BRIDGE_KIND} ==="
  echo "--- gz topic -i (baylands IMX214) ---"
  gz topic -i -t "${IMX_GZ_BAYLANDS}" 2>&1 || true
  echo "--- gz topic -i (default world IMX214) ---"
  gz topic -i -t "${IMX_GZ_DEFAULT}" 2>&1 || true
  echo "--- gz topic -e (one IMX214 frame, 10s timeout) — OK means gz renders/sends Images ---"
  if timeout 10 gz topic -e -t "${IMX_GZ_BAYLANDS}" -n 1 >/dev/null 2>&1; then
    echo "OK: gz transport delivered >=1 Image on baylands IMX214 topic"
  else
    echo "FAIL: no Image from gz in 10s (rendering/lockstep); try USE_XVFB=1, GZ_SIM_RENDER_ENGINE=ogre, or --gui"
  fi
  echo "--- gz topic -l | grep -iE IMX214|camera|image (first 50) ---"
  gz topic -l 2>/dev/null | grep -iE 'IMX214|camera|image' | head -50 || true
  echo "--- ros2 topic info /camera -v ---"
  ros2 topic info /camera -v 2>&1 || true
  echo "--- ros2 topic list | grep -E '^/camera' ---"
  ros2 topic list 2>/dev/null | grep -E '^/camera' || true
  echo "--- ros2 topic echo /camera --once (best_effort, 25s) ---"
  timeout 25 ros2 topic echo /camera --once --qos-reliability best_effort 2>&1 | head -25 || true
  echo "--- ros2 topic echo /camera --once (reliable, 25s) ---"
  timeout 25 ros2 topic echo /camera --once 2>&1 | head -25 || true
  echo "--- ros2 topic hz /camera (45s; IMX214 can be <<1Hz in lockstep/WSL) ---"
  timeout 45 ros2 topic hz /camera 2>&1 || true
  echo "--- bridge.log (last 25 lines; look for Unknown message type = ros_gz vs gz-harmonic mismatch) ---"
  tail -25 "${BRIDGE_LOG}" 2>/dev/null || true
  echo "--- note ---"
  echo "gz -e OK + empty hz: often ultra-low FPS; check SDF <update_rate> / lockstep / WSL GL."
  echo "bridge.log 'Unknown message type': use default parameter_bridge or align ros_gz with gz-harmonic (apt/source)."
} | tee "${CAMERA_DIAG}"

# ros_gz_bridge often forwards only when something subscribes. Do not block on
# `topic echo --once` here: in gz↔PX4 lockstep that can stall for a long time with
# no user feedback and people Ctrl+C (which tears down PX4 via trap cleanup).
echo "    Subscribing /camera so gz_bridge forwards (PX4 may show brief EKF/attitude warnings — normal)."
echo "    Leave this terminal alone until \"== Done\" (avoid Ctrl+C unless aborting)."
(
  ros2 topic hz /camera --qos-reliability best_effort >/dev/null 2>&1 \
    || ros2 topic hz /camera >/dev/null 2>&1
) &
HZ_PID=$!
sleep 5

METRICS="${REPO_ROOT}/results/gazebo_demo/demo_metrics.json"
MOTION_LOG="${REPO_ROOT}/results/gazebo_demo/motion.log"

MOTION_PID=""
if [[ "${SKIP_MOTION}" == "1" ]]; then
  echo "    SKIP_MOTION=1 — skipping px4_demo_motion; waiting ${CAMERA_ONLY_DELAY}s before recorder"
  sleep "${CAMERA_ONLY_DELAY}"
else
  # Motion needs ~ekf-wait + arm before interesting flight; start it first, then record.
  python3 -m evaluation.px4_demo_motion --duration "$((DURATION + 50))" --low-spec "${LOW_SPEC}" \
    --ekf-wait "${EKF_WAIT}" \
    >"${MOTION_LOG}" 2>&1 &
  MOTION_PID=$!
  echo "    Motion log: ${MOTION_LOG}  PID=${MOTION_PID}"
  echo "    Delaying recorder ~24s (keep /camera subscribed during arm + takeoff)…"
  sleep 24
fi

python3 -m evaluation.gazebo_camera_recorder \
  --output "${OUTPUT_MP4}" \
  --duration "${DURATION}" \
  --fps "${RECORD_FPS}" \
  --overlay \
  --metrics "${METRICS}" &
REC_PID=$!
echo "    Recorder PID=${REC_PID}"
# Overlap with hz subscriber: if we kill hz before the recorder's subscription is active,
# some bridge setups go idle and the recorder never sees a writer match.
sleep 4
kill "${HZ_PID}" 2>/dev/null || true
HZ_PID=""

wait "${REC_PID}"

echo "== Done. Video: ${OUTPUT_MP4} =="
