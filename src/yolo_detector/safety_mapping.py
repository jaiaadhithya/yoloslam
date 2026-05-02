"""Load YAML-driven class → landing safety semantics for pretrained COCO (or custom) detectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _norm(name: str) -> str:
    return str(name).strip().lower()


@dataclass
class SafetyMapper:
    """Maps normalized class names to safety_label and numeric safety_weight (negative = hazard)."""

    unsafe_classes: frozenset[str]
    neutral_classes: frozenset[str]
    positive_safe_classes: frozenset[str]
    unsafe_base_weight: float
    positive_safe_weight: float
    unsafe_confidence_scale: float

    @classmethod
    def _builtin_minimal(cls) -> SafetyMapper:
        """Fallback if config file is missing."""
        return cls(
            unsafe_classes=frozenset(
                {
                    "person",
                    "bicycle",
                    "car",
                    "motorcycle",
                    "bus",
                    "train",
                    "truck",
                    "bird",
                    "cat",
                    "dog",
                    "horse",
                    "fire hydrant",
                    "bench",
                }
            ),
            neutral_classes=frozenset({"stop sign", "traffic light", "backpack", "handbag"}),
            positive_safe_classes=frozenset({"grass_field", "flat_ground"}),
            unsafe_base_weight=-1.0,
            positive_safe_weight=0.35,
            unsafe_confidence_scale=1.0,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SafetyMapper:
        p = Path(path)
        if not p.is_file():
            return cls._builtin_minimal()
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        unsafe_block = raw.get("unsafe") or {}
        neutral_block = raw.get("neutral") or {}
        ps_block = raw.get("positive_safe") or {}

        unsafe_names = {_norm(x) for x in (unsafe_block.get("classes") or [])}
        neutral_names = {_norm(x) for x in (neutral_block.get("classes") or [])}
        ps_names = {_norm(x) for x in (ps_block.get("classes") or [])}

        return cls(
            unsafe_classes=frozenset(unsafe_names),
            neutral_classes=frozenset(neutral_names),
            positive_safe_classes=frozenset(ps_names),
            unsafe_base_weight=float(unsafe_block.get("weight", -1.0)),
            positive_safe_weight=float(ps_block.get("weight", 0.35)),
            unsafe_confidence_scale=float(unsafe_block.get("confidence_scale", 1.0)),
        )

    @classmethod
    def default(cls) -> SafetyMapper:
        """Load repo config/safety_mapping.yaml when present."""
        root = Path(__file__).resolve().parents[2]
        return cls.from_yaml(root / "config" / "safety_mapping.yaml")

    def lookup(self, class_name: str) -> tuple[str, float]:
        """
        Returns (safety_label, safety_weight).
        Labels: neutral | unsafe | positive_safe | unknown
        """
        key = _norm(class_name)
        if key in self.neutral_classes:
            return "neutral", 0.0
        if key in self.unsafe_classes:
            return "unsafe", self.unsafe_base_weight
        if key in self.positive_safe_classes:
            return "positive_safe", self.positive_safe_weight
        return "unknown", -0.25

    def describe(self) -> dict[str, Any]:
        return {
            "unsafe_n": len(self.unsafe_classes),
            "neutral_n": len(self.neutral_classes),
            "positive_safe_n": len(self.positive_safe_classes),
            "unsafe_weight": self.unsafe_base_weight,
        }
