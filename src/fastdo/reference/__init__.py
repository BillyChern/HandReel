"""Opt-in reference conditioning for the physics and robot REFERENCE paths (post-reconstruction).

Physics-certain priors only: an object no hand touches does not move in the world (conditional on every contact being
observed), and poses related by a shape symmetry are the same physical state. Nothing here changes a reconstruction
output, a production default or the E1-scored trajectory.
"""

from fastdo.reference.conditioner import (
    ConditionedReference,
    SegmentReport,
    WindowInputs,
    condition_window,
    consolidate_rows,
)
from fastdo.reference.config import ReferenceConfig, constants_hash
from fastdo.reference.frames import WorldSource, declared_static, no_world, world_from_ego
from fastdo.reference.motion import MotionTest, motion_test, motion_test_residuals
from fastdo.reference.segmentation import HandInput
from fastdo.reference.start_pose import StartPose, start_pose_at_onset
from fastdo.reference.symmetry_group import SymmetryGroup, compute_group, trivial_group

__all__ = [
    "ConditionedReference", "HandInput", "MotionTest", "ReferenceConfig", "SegmentReport", "StartPose",
    "SymmetryGroup", "WindowInputs", "WorldSource", "compute_group", "condition_window", "consolidate_rows",
    "constants_hash", "declared_static", "motion_test", "motion_test_residuals", "no_world", "start_pose_at_onset",
    "trivial_group", "world_from_ego",
]
