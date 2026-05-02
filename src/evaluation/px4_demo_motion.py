"""
Smooth MAVLink OFFBOARD velocity for PX4 SITL (gz_x500) — survey + gentle descent.

Applies typical SITL-friendly parameters so arming is not blocked by missing
battery / supply checks, then waits for estimators before arming.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence


def _param_set(master, name: str, value: float, ptype: int) -> None:
    from pymavlink import mavutil

    b = name.encode("ascii")[:16].ljust(16, b"\0")
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        b,
        value,
        ptype,
    )


def _sitl_relax_preflight(master) -> None:
    """Best-effort params so Baylands SITL can arm without real power / GPS / GCS."""
    from pymavlink import mavutil

    # CBRK_SUPPLY_CHK = 894281 — skip power supply checks in simulation (INT32 in PX4)
    _param_set(master, "CBRK_SUPPLY_CHK", 894281.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    time.sleep(0.2)
    # Allow arming without global position / GPS where supported
    _param_set(master, "COM_ARM_WO_GPS", 1.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    time.sleep(0.2)
    # Be tolerant of long “GCS loss” in bench sim (INT32 seconds in PX4, not float)
    _param_set(master, "COM_DL_LOSS_T", 5000.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    time.sleep(0.2)
    # NAV_DLL_ACT = 0 — no action on data link loss (if param exists)
    _param_set(master, "NAV_DLL_ACT", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
    time.sleep(0.2)
    # Float timeout [s] in PX4 (REAL32, not INT32) — large value avoids early auto-disarm in SITL.
    _param_set(master, "COM_DISARM_PRFLT", 7200.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.2)


def _custom_mode_id(mode_val) -> int:
    """pymavlink mode_mapping() may nest (base_mode, custom_mode); unwrap to a scalar."""
    if isinstance(mode_val, (str, bytes, bytearray)):
        return int(float(mode_val))
    if isinstance(mode_val, Sequence) and not isinstance(mode_val, (str, bytes, bytearray)):
        if len(mode_val) == 0:
            raise ValueError("empty mode mapping value")
        return _custom_mode_id(mode_val[-1])
    try:
        return int(mode_val)
    except (TypeError, ValueError):
        return int(float(mode_val))


def _try_set_custom_mode(master, mode_name: str) -> bool:
    from pymavlink import mavutil

    mapping = master.mode_mapping()
    if not mapping or mode_name not in mapping:
        return False
    cid = _custom_mode_id(mapping[mode_name])
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        cid,
    )
    return True


def _try_set_offboard(master) -> bool:
    from pymavlink import mavutil

    if _try_set_custom_mode(master, "OFFBOARD"):
        return True
    try:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6,
            0,
            0,
            0,
            0,
            0,
        )
        return True
    except Exception:
        return False


def _velocity_only_type_mask() -> int:
    from pymavlink import mavutil

    return (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )


def _send_body_vel(master, vx: float, vy: float, vz: float, mask: int) -> None:
    from pymavlink import mavutil

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        mask,
        0.0,
        0.0,
        0.0,
        vx,
        vy,
        vz,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _try_arm(master) -> None:
    from pymavlink import mavutil

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def _armed_from_heartbeat(master, wait_s: float = 2.5) -> bool:
    from pymavlink import mavutil

    t0 = time.time()
    while time.time() - t0 < wait_s:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
        if msg is None:
            continue
        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="PX4 SITL smooth demo motion")
    parser.add_argument("--conn", default="udp:127.0.0.1:14540")
    parser.add_argument("--duration", type=int, default=50)
    parser.add_argument("--low-spec", type=str, default="1")
    parser.add_argument(
        "--ekf-wait",
        type=float,
        default=24.0,
        help="Seconds to wait after params before arming (attitude/EKF in gz sim)",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for first MAVLink HEARTBEAT (then exit with error)",
    )
    parser.add_argument(
        "--gentle",
        action="store_true",
        help="Low-amplitude OFFBOARD velocities for smooth demo video (minimal oscillation).",
    )
    args = parser.parse_args()

    try:
        from pymavlink import mavutil
    except ImportError:
        print("Install pymavlink: pip install pymavlink", file=sys.stderr)
        return 1

    print(f"Connecting {args.conn} …", flush=True)
    master = mavutil.mavlink_connection(args.conn, source_system=255)
    hb_deadline = time.time() + float(args.heartbeat_timeout)
    print(
        f"Waiting for first HEARTBEAT (up to {args.heartbeat_timeout:.0f}s). "
        "SITL uses onboard UDP 14540 by default; QGroundControl uses 14550.",
        flush=True,
    )
    got_hb = False
    while time.time() < hb_deadline:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg is not None:
            got_hb = True
            break
    if not got_hb:
        print(
            "ERROR: No MAVLink heartbeat — PX4 may not be running, or use --conn udp:127.0.0.1:14540 "
            "(see mavlink start lines in the PX4 console).",
            file=sys.stderr,
        )
        return 1
    print(f"Heartbeat sys={master.target_system} comp={master.target_component}", flush=True)

    print("Applying SITL preflight relax params…", flush=True)
    _sitl_relax_preflight(master)
    print(f"Waiting {args.ekf_wait:.0f}s for EKF / attitude (progress every ~6s)…", flush=True)
    remaining = float(args.ekf_wait)
    while remaining > 0:
        step = min(6.0, remaining)
        time.sleep(step)
        remaining -= step
        if remaining > 1.0:
            print(f"  … {remaining:.0f}s left before arm attempts", flush=True)

    armed = False
    for attempt in range(12):
        _try_arm(master)
        if _armed_from_heartbeat(master, wait_s=3.0):
            armed = True
            print(f"Armed on attempt {attempt + 1}", flush=True)
            break
        print(f"Arm attempt {attempt + 1}/12 (not armed yet)…", flush=True)

    if not armed:
        print("WARN: Could not confirm arm; continuing without takeoff.", flush=True)

    # gz x500_depth often spawns already airborne; NAV_TAKEOFF to a low AMSL target
    # triggers "Already higher than takeoff altitude" → preflight disarm → OFFBOARD fails.
    print("Skipping NAV_TAKEOFF (SITL spawn is already in flight).", flush=True)
    time.sleep(2.0)

    # Auto-takeoff / altitude-hold states reject OFFBOARD; move to a manual-style mode first.
    print("Leaving auto-takeoff (POSCTL / ALTCTL / STABILIZED) before OFFBOARD…", flush=True)
    for mode_name in ("STABILIZED", "MANUAL", "ALTCTL", "POSCTL"):
        if _try_set_custom_mode(master, mode_name):
            print(f"Requested {mode_name}", flush=True)
            time.sleep(2.5)
            break

    mask = _velocity_only_type_mask()
    # PX4 requires a steady OFFBOARD stream (~2 s) before accepting OFFBOARD mode.
    print("Priming OFFBOARD setpoint stream…", flush=True)
    for i in range(120):
        if i % 25 == 0:
            _try_arm(master)
        _send_body_vel(master, 0.0, 0.0, 0.0, mask)
        time.sleep(0.05)

    ok = _try_set_offboard(master)
    print(f"OFFBOARD mode request ok={ok}", flush=True)
    time.sleep(0.5)

    t0 = time.time()
    period = float(args.duration)
    while time.time() - t0 < period:
        t = time.time() - t0
        if args.gentle:
            # BODY_NED: small forward + mild descent (vz > 0 ≈ down in body frame).
            vx = 0.32 + 0.04 * math.sin(0.11 * t)
            vy = 0.06 * math.sin(0.07 * t)
            vz = 0.05 + 0.02 * math.sin(0.09 * t)
        else:
            vx = 0.95 + 0.12 * math.sin(0.18 * t)
            vy = 0.28 * math.sin(0.085 * t)
            vz = 0.10 + 0.04 * math.sin(0.12 * t)
        _send_body_vel(master, vx, vy, vz, mask)
        time.sleep(0.05)

    print("Motion loop finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
