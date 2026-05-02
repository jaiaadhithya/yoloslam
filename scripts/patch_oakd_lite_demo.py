"""
Patch PX4-gazebo-models OakD-Lite in a local PX4 checkout for lightweight demo video:

  - IMX214 RGB: 640x480, update_rate 12
  - Remove StereoOV7251 depth_camera sensor

Creates model.sdf.bak_yoloslam next to model.sdf on first patch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def _find_oakd(px4_root: Path) -> Path | None:
    rels = (
        Path("Tools/simulation/gz/models/OakD-Lite/model.sdf"),
        Path("Tools/simulation/gz/models/PX4-gazebo-models/models/OakD-Lite/model.sdf"),
    )
    for rel in rels:
        p = px4_root / rel
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    try:
        for p in px4_root.rglob("OakD-Lite/model.sdf"):
            try:
                if p.is_file():
                    return p
            except OSError:
                continue
    except OSError:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--px4", type=Path, required=True, help="PX4-Autopilot root")
    args = ap.parse_args()
    px4 = args.px4.expanduser().resolve()
    if not px4.is_dir():
        print(f"ERROR: not a directory: {px4}", file=sys.stderr)
        return 1
    oakd = _find_oakd(px4)
    if oakd is None:
        print("WARN: OakD-Lite/model.sdf not found under PX4 tree; skip patch.", file=sys.stderr)
        return 0
    bak = oakd.with_name("model.sdf.bak_yoloslam")
    text = oakd.read_text(encoding="utf-8")
    if not bak.is_file():
        bak.write_text(text, encoding="utf-8")
        print(f"Backup -> {bak}", flush=True)

    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        print(f"ERROR: parse {oakd}: {e}", file=sys.stderr)
        return 1

    changed = False
    for link in root.iter("link"):
        if link.get("name") != "camera_link":
            continue
        for sensor in list(link.findall("sensor")):
            name = sensor.get("name", "")
            stype = sensor.get("type", "")
            if name == "StereoOV7251" or stype == "depth_camera":
                link.remove(sensor)
                changed = True
                print(f"Removed sensor {name!r} ({stype})", flush=True)
            elif name == "IMX214":
                ur = sensor.find("update_rate")
                if ur is not None:
                    ur.text = "12"
                cam = sensor.find("camera")
                if cam is not None:
                    img = cam.find("image")
                    if img is not None:
                        w = img.find("width")
                        h = img.find("height")
                        if w is not None:
                            w.text = "640"
                        if h is not None:
                            h.text = "480"
                changed = True
                print("IMX214 -> 640x480 @ 12 Hz", flush=True)

    if not changed:
        print("No changes applied (already patched or unexpected SDF structure).", flush=True)
        return 0

    out = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
    oakd.write_text(out, encoding="utf-8")
    print(f"Patched {oakd}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
