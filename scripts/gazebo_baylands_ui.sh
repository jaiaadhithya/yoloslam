#!/usr/bin/env bash
# PX4 SITL + Gazebo Baylands + x500_depth with GUI only.
# No ROS 2, no camera recorder, no MAVLink motion script — just fly / inspect in the sim.
#
# Env:
#   AUTO_KILL_EXISTING=0 — do not stop other px4/gz (default 1: free instance 0 before start).
#   SOFTWARE_GL=1 — LLVMpipe / software GL (try if the window never appears or Ogre crashes).
#   QT_QPA_PLATFORM — default xcb for WSLg X11; try wayland if you use native Wayland.
set -euo pipefail

PX4_AUTOPILOT_PATH="${PX4_AUTOPILOT_PATH:-${HOME}/PX4-Autopilot}"
AUTO_KILL_EXISTING="${AUTO_KILL_EXISTING:-1}"

# Non-interactive WSL (e.g. Cursor → wsl bash -lc) often has no DISPLAY → no Gazebo window.
export DISPLAY="${DISPLAY:-:0}"
if [[ -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
elif [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
  export XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
  mkdir -p "${XDG_RUNTIME_DIR}" 2>/dev/null || true
fi
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

if [[ "${SOFTWARE_GL:-0}" == "1" ]]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
else
  unset LIBGL_ALWAYS_SOFTWARE 2>/dev/null || true
fi
unset QT_QUICK_BACKEND 2>/dev/null || true

if [[ ! -d "${PX4_AUTOPILOT_PATH}" ]]; then
  echo "ERROR: PX4-Autopilot not found at ${PX4_AUTOPILOT_PATH}"
  echo "  export PX4_AUTOPILOT_PATH=/path/to/PX4-Autopilot"
  exit 1
fi

cd "${PX4_AUTOPILOT_PATH}"
export PX4_GZ_WORLD=baylands
export HEADLESS=0
export GZ_SIM_RENDER_ENGINE="${GZ_SIM_RENDER_ENGINE:-ogre2}"

echo "== Gazebo UI only: Baylands + gz_x500_depth =="
echo "    PX4_AUTOPILOT_PATH=${PX4_AUTOPILOT_PATH}"
echo "    DISPLAY=${DISPLAY}  QT_QPA_PLATFORM=${QT_QPA_PLATFORM}  XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}"
echo "    GZ_SIM_RENDER_ENGINE=${GZ_SIM_RENDER_ENGINE}  SOFTWARE_GL=${SOFTWARE_GL:-0}"
echo "    NOTE: 'Ready for takeoff!' means the stack is up — this script does NOT arm or climb. Use QGroundControl"
echo "          (UDP 127.0.0.1:14550) or in another terminal: python3 -m evaluation.px4_demo_motion --conn udp:127.0.0.1:14540"
echo "    Close the Gazebo window or press Ctrl+C in this terminal to stop."

if [[ "${AUTO_KILL_EXISTING}" == "1" ]]; then
  echo "    Stopping prior PX4 SITL / Gazebo (frees instance 0)…"
  pkill -f 'px4_sitl_default/bin/px4' 2>/dev/null || true
  pkill -f 'gz sim' 2>/dev/null || true
  pkill -f 'ruby.*simulation-gazebo' 2>/dev/null || true
  sleep 2
fi

exec make px4_sitl gz_x500_depth
