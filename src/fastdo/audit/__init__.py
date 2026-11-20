# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""fastdo.audit: null tests for physics certificates of video-to-robot trajectories.

For each certified run the audit reports (a) whether the object frozen at its first reference pose passes the
certificate, (b) whether the run's controls still pass it with every hand-object contact removed, and (c) the grip
census, the median hand-object normal force over the object's weight. Presets reproduce DO AS I DO's certificate
C_D and the object half of our registered C_289. The kinematic nulls and the report come from `nullaudit`
(packages/nullaudit); this package adds the presets, the census, the scene edits and the CLI
(`python -m fastdo.audit`). `python -m fastdo.audit.example` is a worked example on the synthetic scene of
`fastdo.audit.toy` (CPU, a few seconds). The README states what each check does and does not certify.
"""

from .census import F_RATED_N, GripCensus, check_object_mask, grip_census, object_weight
from .certificates import (
    C289,
    CD,
    PRESETS,
    C289Certificate,
    C289Result,
    C289Thresholds,
    CDCertificate,
    CDThresholds,
    load_certificate,
    make_certificate,
    motion_ok,
    mt_null,
    plaus_ok,
    plausibility,
    resolve_thresholds,
)
from .errors import InputError
from .match import executor_diff, match_executor, missing_pairs
from .runner import (
    QUAT_TOL,
    REMOVALS,
    REPLAY_TOL,
    Executed,
    PositiveControl,
    RunAudit,
    audit_run,
    report_dict,
    summarize_runs,
)
from .scene import (
    classify_pairs,
    compile_without_pairs,
    contact_paths,
    free_object_geom,
    geom_mask,
    hand_object_pair_names,
    ka_h,
    load_model,
    remove_pairs,
    start_pedestal,
)

__version__ = "0.1.0"

__all__ = [
    "C289",
    "CD",
    "F_RATED_N",
    "PRESETS",
    "QUAT_TOL",
    "REMOVALS",
    "REPLAY_TOL",
    "C289Certificate",
    "C289Result",
    "C289Thresholds",
    "CDCertificate",
    "CDThresholds",
    "Executed",
    "GripCensus",
    "InputError",
    "PositiveControl",
    "RunAudit",
    "audit_run",
    "check_object_mask",
    "classify_pairs",
    "compile_without_pairs",
    "contact_paths",
    "executor_diff",
    "free_object_geom",
    "geom_mask",
    "grip_census",
    "hand_object_pair_names",
    "ka_h",
    "load_certificate",
    "load_model",
    "make_certificate",
    "match_executor",
    "missing_pairs",
    "motion_ok",
    "mt_null",
    "object_weight",
    "plaus_ok",
    "plausibility",
    "remove_pairs",
    "report_dict",
    "resolve_thresholds",
    "start_pedestal",
    "summarize_runs",
]
