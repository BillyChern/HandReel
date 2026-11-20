# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""nullaudit: check whether a physics success test for hand-object trajectories passes trivial null trajectories.

Model-free core (numpy only): build the standard nulls, score them with any success-test callable, and report each
null's pass rate beside the Z0 base rate (clips where the static-at-start null passes). The simulated nulls
(hand-absent, plan-then-freeze) need a simulator callback; `nullaudit.mujoco_adapter` provides one for MuJoCo.
"""

from .audit import (
                    KINEMATIC_NULLS,
                    SIMULATED_NULLS,
                    STANDARD_NULLS,
                    AuditReport,
                    ClipAudit,
                    NullOutcome,
                    NullRate,
                    Rollout,
                    Simulator,
                    audit_clip,
                    summarize,
                    verdict,
                    wilson95,
)
from .doasido import (
                    DAID_POS_M,
                    DAID_ROT_RAD,
                    DoAsIDoObjectTest,
                    ObjectTestResult,
                    object_errors,
                    reference_on_executor_grid,
)
from .nulls import move_hand_away, plan_then_freeze_controls, reversed_reference, static_at_start, time_shifted
from .poses import as_pose7, matrix_to_quat, quat_angle

__version__ = "0.1.0"

__all__ = [
    "DAID_POS_M",
    "DAID_ROT_RAD",
    "KINEMATIC_NULLS",
    "SIMULATED_NULLS",
    "STANDARD_NULLS",
    "AuditReport",
    "ClipAudit",
    "DoAsIDoObjectTest",
    "NullOutcome",
    "NullRate",
    "ObjectTestResult",
    "Rollout",
    "Simulator",
    "as_pose7",
    "audit_clip",
    "matrix_to_quat",
    "move_hand_away",
    "object_errors",
    "plan_then_freeze_controls",
    "quat_angle",
    "reference_on_executor_grid",
    "reversed_reference",
    "static_at_start",
    "summarize",
    "time_shifted",
    "verdict",
    "wilson95",
]
