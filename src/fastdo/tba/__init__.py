"""TBA / TBA-V: temporal object bundle adjustment + hand-aware carved template, BASE's window gauge held.

Opt-in (`tba_stage: null` = off).
"""

from fastdo.tba.config import TBA_KEY, TbaConfig, tba_from_mapping
from fastdo.tba.window import (
    FrameEvidence,
    TbaOutput,
    Window,
    finish,
    load_window,
    prepare,
    run_tba,
)

__all__ = [
    "TBA_KEY",
    "FrameEvidence",
    "TbaConfig",
    "TbaOutput",
    "Window",
    "finish",
    "load_window",
    "prepare",
    "run_tba",
    "tba_from_mapping",
]
