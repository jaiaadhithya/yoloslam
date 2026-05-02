#!/usr/bin/env bash
# Attach Gazebo **GUI** to an already-running gz sim started by PX4 (no extra server).
#
# Use when: PX4 logs show "Gazebo world is ready" but no window appears (WSLg / Qt / split client).
#
# Run from the SAME machine/WSL session, after SITL is up:
#   ./scripts/gazebo_attach_gui.sh
#
# Env (optional, match your WSL GUI setup):
#   GZ_PARTITION — must match the running sim if you set it when starting PX4 (rare)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "/run/user/$(id -u)" ]]; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
elif [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
  export XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
  mkdir -p "${XDG_RUNTIME_DIR}" 2>/dev/null || true
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
  export DISPLAY="${DISPLAY:-:0}"
  if [[ -z "${WAYLAND_DISPLAY:-}" && -S "${XDG_RUNTIME_DIR}/wayland-0" ]]; then
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
  fi
  if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
  else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
  fi
else
  export DISPLAY="${DISPLAY:-:0}"
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
fi

unset QT_QUICK_BACKEND 2>/dev/null || true

echo "Attaching Gazebo GUI (gz sim -g)…"
echo "  DISPLAY=${DISPLAY} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-} QT_QPA_PLATFORM=${QT_QPA_PLATFORM}"
echo "  GZ_PARTITION=${GZ_PARTITION:-<default>}"
echo ""
echo "If the window is empty or errors, try:  export QT_QPA_PLATFORM=xcb && unset WAYLAND_DISPLAY"
echo ""

exec gz sim -g
