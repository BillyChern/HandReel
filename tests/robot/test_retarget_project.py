"""Known answers for fastdo.robot.retarget_project: a
feasible grasp is returned unchanged; the `test_grasp_r6` sphere grasp pushed 20 mm into the sphere is projected back to
penetration <= pen_m with its keypoint deviation reported; the projection never exceeds a joint limit or its bound."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fastdo.robot import g1_constants as C
from fastdo.robot.retarget_project import ProjectConfig, project_grasp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grasp_r6 as T  # the R6-1 sphere scene and grasp pose


@pytest.fixture(scope="module")
def setup(g1):
    _, r, cc, clr = T.sphere.__wrapped__(g1)
    base_obj = np.eye(4)
    base_obj[:3, 3] = -cc                              # the Dex3 base in the sphere-centre frame (clears by 0.5 mm)
    return clr, base_obj, r


def _push(base_obj: np.ndarray, d: float) -> np.ndarray:
    b = base_obj.copy()
    b[:3, 3] *= 1.0 - d / np.linalg.norm(b[:3, 3])      # the base d closer to the sphere centre
    return b


@pytest.fixture(scope="module", params=[0.020, 0.040])
def pushed(setup, request):
    clr, base_obj, _ = setup
    return project_grasp(clr, _push(base_obj, request.param), T._G)


def test_a_feasible_grasp_is_returned_unchanged(setup) -> None:
    clr, base_obj, _ = setup
    res = project_grasp(clr, base_obj, T._G)
    assert res.feasible and res.rounds == 0 and res.keypoint_max_mm == 0.0
    assert np.array_equal(res.base_obj, base_obj) and np.allclose(res.hand_q, T._G, atol=1e-12)


def test_a_penetrating_grasp_is_projected_back_with_its_deviation_reported(setup, pushed) -> None:
    clr, _, _ = setup
    res = pushed
    cfg = ProjectConfig()
    assert res.clearance_start_m < -cfg.pen_m                             # the push penetrates beyond the target
    assert res.feasible and res.clearance_m >= -cfg.pen_m - 1e-9
    assert np.isclose(clr.clearance(res.base_obj, res.hand_q), res.clearance_m)
    assert 0.0 < res.keypoint_rms_mm                                      # reported (R6-9: deviation from the retarget)
    assert res.shift_m <= cfg.max_shift_m + 1e-9 and res.rot_deg <= 20.0 + 1e-9


def test_the_projection_never_exceeds_a_joint_limit(pushed) -> None:
    res = pushed
    lim = np.asarray(C.RIGHT_HAND_COMMAND_LIMITS)
    m = ProjectConfig().limit_margin_rad
    assert (res.hand_q >= lim[:, 0] + m - 1e-12).all() and (res.hand_q <= lim[:, 1] - m + 1e-12).all()


def test_a_deep_push_is_solved_at_the_bound_and_never_beyond_it(setup) -> None:
    """80 mm push (~39 mm deep, the depth class of the S3 clips): solved, with the base shift ON the 60 mm ball."""
    clr, base_obj, _ = setup
    res = project_grasp(clr, _push(base_obj, 0.080), T._G)
    assert res.clearance_start_m < -0.03 and res.feasible
    assert res.shift_m <= ProjectConfig().max_shift_m + 1e-9 and res.rot_deg <= 20.0 + 1e-9


def test_an_unreachable_projection_is_reported_infeasible(setup) -> None:
    """With a 2 mm / 0.3 deg bound the 40 mm push cannot be undone: `feasible` is False, never claimed."""
    import dataclasses

    clr, base_obj, _ = setup
    cfg = dataclasses.replace(ProjectConfig(), max_shift_m=0.002, max_rot_rad=0.005)
    res = project_grasp(clr, _push(base_obj, 0.040), T._G, cfg)
    assert not res.feasible and res.clearance_m < -cfg.pen_m and res.rounds == cfg.rounds


def test_the_result_does_not_depend_on_evaluation_history(g1) -> None:
    """2026-10-17: an unquantised Powell run on the 80 mm push flipped between solved and stuck depending on which
    projections had filled HandClearance's cache first. Fresh and warm caches must give the same result."""
    _, _, cc, clr_a = T.sphere.__wrapped__(g1)
    _, _, _, clr_b = T.sphere.__wrapped__(g1)
    base_obj = np.eye(4)
    base_obj[:3, 3] = -cc
    for d in (0.02, 0.06):
        project_grasp(clr_b, _push(base_obj, d), T._G)          # warm clr_b's cache with other projections
    a, b = project_grasp(clr_a, _push(base_obj, 0.08), T._G), project_grasp(clr_b, _push(base_obj, 0.08), T._G)
    assert a.feasible == b.feasible and np.array_equal(a.hand_q, b.hand_q) and np.array_equal(a.base_obj, b.base_obj)


def test_the_back_out_starts_solve_what_the_retarget_start_cannot(setup) -> None:
    """120 mm push (40 mm deep): from the retarget alone Powell ends infeasible (-6.75 mm, measured 2026-10-17); the
    back-out starts reach the target inside the bound."""
    clr, base_obj, _ = setup
    res = project_grasp(clr, _push(base_obj, 0.120), T._G)
    assert res.clearance_start_m < -0.03 and res.feasible and res.shift_m <= ProjectConfig().max_shift_m + 1e-9


def test_returned_hand_poses_are_on_the_cache_grid(pushed) -> None:
    """The order-independence contract: every evaluated (and the returned) hand pose is quantised to HandClearance's
    1e-9 cache key, so a cached geom set is always the exact one for its input."""
    res = pushed
    assert np.array_equal(res.hand_q, np.round(res.hand_q, 9))
