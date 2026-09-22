"""Configuration for the cross-section labeler GUI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def _default_labels() -> List[str]:
    return ["channel", "ridge1", "floodplain1", "ridge2", "floodplain2"]


def _default_colors() -> Dict[str, str]:
    # Per-label and per-feature colors. The labeler indexes by full label
    # (e.g. ``ridge1``) and, for predicted-point overlays, by the feature
    # base name (``ridge`` from ``ridge1`` via ``label.split('_')[0]``).
    return {
        "channel": "tab:blue",
        "ridge1": "tab:red",
        "ridge2": "tab:orange",
        "ridge": "tab:red",
        "floodplain1": "tab:green",
        "floodplain2": "tab:olive",
        "floodplain": "tab:green",
    }


@dataclass
class GUIConfig:
    """Settings for the interactive labeling GUI."""

    labels: List[str] = field(default_factory=_default_labels)
    colors: Dict[str, str] = field(default_factory=_default_colors)
    # Savitzky–Golay profile smoothing for the labeler overlay
    show_smoothed_profile: bool = True
    smooth_window_m: float = 25.0  # approximate filter length along-track
    smooth_polyorder: int = 2
