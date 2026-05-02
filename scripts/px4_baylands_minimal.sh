#!/usr/bin/env bash
# Minimal PX4 SITL + Gazebo Baylands + x500 (no ROS nodes from this repo).
#
# Default vehicle: gz_x500_depth — includes onboard RGB (IMX214) for later camera recording.
#   Plain gz_x500 has no monocular camera in upstream PX4 Gazebo models; use:
#     PX4_MAKE_TARGET=gz_x500 ./scripts/px4_baylands_minimal.sh
#
# Performance (smooth sim, prefer GPU):
#   - Unsets software GL overrides unless VIDEO_USE_SOFTWARE_GL=1 (WSL software fallback).
#   - ogre2 render engine, real-time factor target 1.
#
# Optional env:
#   PX4_AUTOPILOT_PATH  — PX4 checkout (default ~/PX4-Autopilot)
#   PX4_MAKE_TARGET     — make goal (default gz_x500_depth)
#   HEADLESS            — must be exactly 1 to hide Gazebo GUI (default: show GUI)
#   VIDEO_USE_SOFTWARE_GL — 1 = force LIBGL_ALWAYS_SOFTWARE (slower; can fix broken GL)
#   AUTO_KILL_EXISTING  — 1 = pkill old px4/gz before start (default 1)
#   GZ_SIM_RENDER_ENGINE — default ogre2
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_AUTOPILOT_PATH="${PX4_AUTOPILOT_PATH:-${HOME}/PX4-Autopilot}"
PX4_MAKE_TARGET="${PX4_MAKE_TARGET:-gz_x500_depth}"
# Many dev shells export HEADLESS=1 from other scripts; only headless when explicitly 1.
if [[ "${HEADLESS:-}" == "1" ]]; then
  HEADLESS=1
else
  HEADLESS=0
fi
AUTO_KILL_EXISTING="${AUTO_KILL_EXISTING:-1}"
export GZ_SIM_RENDER_ENGINE="${GZ_SIM_RENDER_ENGINE:-ogre2}"
export PX4_SIM_SPEED_FACTOR="${PX4_SIM_SPEED_FACTOR:-1}"
# Prefer repo baylands.sdf (shadows off, lower physics rate, no sky clouds) when resolving worlds.
export GZ_SIM_RESOURCE_PATH="${REPO_ROOT}/yoloslam_worlds:${GZ_SIM_RESOURCE_PATH:-}"

# GPU-friendly defaults (contrast with gazebo_demo_lowspec.sh).
if [[ "${VIDEO_USE_SOFTWARE_GL:-0}" == "1" ]]; then
  export LIBGL_ALWAYS_SOFTWARE=1
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
else
  unset LIBGL_ALWAYS_SOFTWARE 2>/dev/null || true
  unset GALLIUM_DRIVER 2>/dev/null || true
fi
unset QT_QUICK_BACKEND 2>/dev/null || true

if [[ -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
elif [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
  export XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
  mkdir -p "${XDG_RUNTIME_DIR}" 2>/dev/null || true
fi

# WSL2 + WSLg: GUI needs DISPLAY (and often Wayland) — Cursor/SSH shells sometimes omit these.
if grep -qi microsoft /proc/version 2>/dev/null; then
  export DISPLAY="${DISPLAY:-:0}"
  if [[ -z "${WAYLAND_DISPLAY:-}" && -S "${XDG_RUNTIME_DIR}/wayland-0" ]]; then
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
  fi
  # Prefer Wayland on modern WSLg when available; fall back to xcb.
  if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
  else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
  fi
else
  export DISPLAY="${DISPLAY:-:0}"
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
fi

if [[ ! -d "${PX4_AUTOPILOT_PATH}" ]]; then
  echo "ERROR: PX4-Autopilot not found at ${PX4_AUTOPILOT_PATH}"
  echo "  export PX4_AUTOPILOT_PATH=/path/to/PX4-Autopilot"
  exit 1
fi

if [[ "${AUTO_KILL_EXISTING}" == "1" ]]; then
  echo "Stopping prior PX4 SITL / Gazebo (best-effort)…"
  pkill -f 'px4_sitl_default/bin/px4' 2>/dev/null || true
  pkill -f 'gz sim' 2>/dev/null || true
  pkill -f 'ruby.*simulation-gazebo' 2>/dev/null || true
  sleep 2
fi

echo "== Minimal Baylands + ${PX4_MAKE_TARGET} (YOLOSLAM) =="
echo "    PX4_AUTOPILOT_PATH=${PX4_AUTOPILOT_PATH}"
echo "    HEADLESS=${HEADLESS}  GZ_SIM_RENDER_ENGINE=${GZ_SIM_RENDER_ENGINE}"
echo "    DISPLAY=${DISPLAY:-unset}  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-unset}  QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-unset}"
echo "    LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-unset}"
if [[ "${HEADLESS}" != "1" ]] && [[ -z "${DISPLAY:-}" ]]; then
  echo "WARN: DISPLAY is empty — Gazebo GUI will not open. Try: export DISPLAY=:0"
fi
echo ""
echo "    After 'Ready for takeoff', optional smooth motion (separate terminal):"
echo "      cd ${REPO_ROOT} && PYTHONPATH=src python3 -m evaluation.px4_demo_motion --gentle --conn udp:127.0.0.1:14540"
echo ""
echo "    Record monocular video (starts ros_gz bridge + recorder; separate terminal):"
echo "      cd ${REPO_ROOT} && PYTHONPATH=src python3 -m evaluation.record_gazebo_video --with-bridge"
echo ""
echo "    If sim runs but NO Gazebo window: second terminal →  ${REPO_ROOT}/scripts/gazebo_attach_gui.sh"
echo "      (or:  gz sim -g   with same DISPLAY / WSLg as above)"
echo ""
echo "    OakD-Lite patch (640x480 @ 12 Hz, drop depth_camera): YOLOSLAM_PATCH_OAKD=1 (default)."
echo ""

if [[ "${YOLOSLAM_PATCH_OAKD:-1}" == "1" ]]; then
  if python3 "${REPO_ROOT}/scripts/patch_oakd_lite_demo.py" --px4 "${PX4_AUTOPILOT_PATH}"; then
    :
  else
    echo "WARN: OakD-Lite patch failed or skipped — continuing with stock sensors."
  fi
fi

cd "${PX4_AUTOPILOT_PATH}"
export PX4_GZ_WORLD=baylands
export HEADLESS

exec make px4_sitl "${PX4_MAKE_TARGET}"
