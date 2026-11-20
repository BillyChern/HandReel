"""A principled successor of C-SMOOTH: error-state RTS smoothing of the object's pose trajectory on SE(3) with
anisotropic camera-frame noise, BASE's window gauge held exactly.

Opt-in (`smooth_stage: null` = off).
"""

from fastdo.smooth.alongray import JointRanges, alongray_joint, alongray_ranges, rts_cv_1d
from fastdo.smooth.config import ARMS, SMOOTH_KEY, SmootherConfig, smoother_from_mapping
from fastdo.smooth.rts import Noise, SmoothResult, smooth_se3
from fastdo.smooth.window import SmoothOutput, gate, row_spacing, smooth_window

__all__ = [
    "ARMS",
    "SMOOTH_KEY",
    "JointRanges",
    "Noise",
    "SmoothOutput",
    "SmoothResult",
    "SmootherConfig",
    "alongray_joint",
    "alongray_ranges",
    "gate",
    "row_spacing",
    "rts_cv_1d",
    "smooth_se3",
    "smooth_window",
    "smoother_from_mapping",
]
