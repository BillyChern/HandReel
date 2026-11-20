"""Known answers for fastdo.robot.grasp_refine: a one-sided grasp of the `test_grasp_r6` sphere (the thumb opened)
has epsilon 0 and the refinement brings it into force closure inside every bound; a grasp that cannot be improved is
returned unchanged; the refinement never exceeds a joint limit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fastdo.robot import g1_constants as C
from fastdo.robot.approach import open_hand
from fastdo.robot.grasp_refine import GraspContactModel, RefineConfig, refine_grasp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grasp_r6 as T  # the R6-1 sphere scene and grasp pose


@pytest.fixture(scope="module")
def setup(g1):
    _, r, cc, clr = T.sphere.__wrapped__(g1)
    base_obj = np.eye(4)
    base_obj[:3, 3] = -cc                              # the Dex3 base in the sphere-centre frame
    return GraspContactModel(clr, np.zeros(3), r, RefineConfig()), base_obj


def _one_sided() -> np.ndarray:
    q = T._G.copy()
    q[1:3] = open_hand(T._G)[1:3]                      # thumb flexion opened: index + middle on one side only
    return q


@pytest.fixture(scope="module")
def refined(setup):
    gm, base_obj = setup
    return refine_grasp(gm, base_obj, _one_sided())


def test_the_one_sided_grasp_is_not_in_force_closure(setup) -> None:
    gm, base_obj = setup
    s = gm.measure(base_obj, _one_sided())
    assert s["epsilon"] == 0.0 and "thumb" not in s["fingers"]


def test_refinement_brings_it_into_force_closure_inside_every_bound(setup, refined) -> None:
    gm, _ = setup
    cfg = RefineConfig()
    assert refined.improved and refined.epsilon_start == 0.0 and refined.epsilon > 0.0
    assert "thumb" in refined.fingers and len(refined.fingers) >= 2          # the opposition the start lacked
    assert refined.shift_m <= cfg.max_shift_m + 1e-9 and refined.rot_deg <= 10.0 + 1e-9
    s = gm.measure(refined.base_obj, refined.hand_q)
    assert s["min_gap"] >= -cfg.pen_m - 1e-6                       # no penetration beyond the R6-1 (b) target
    assert np.isclose(s["epsilon"], refined.epsilon)


def test_refinement_never_exceeds_a_joint_limit(refined) -> None:
    lim = np.asarray(C.RIGHT_HAND_COMMAND_LIMITS)
    m = RefineConfig().limit_margin_rad
    assert (refined.hand_q >= lim[:, 0] + m - 1e-12).all() and (refined.hand_q <= lim[:, 1] - m + 1e-12).all()


def test_a_grasp_that_cannot_be_improved_is_returned_unchanged(setup) -> None:
    gm, base_obj = setup
    far = base_obj.copy()
    far[:3, 3] += np.array([0.0, 0.0, 0.2])            # 20 cm away: no bounded refinement reaches the sphere
    res = refine_grasp(gm, far, T._G)
    assert not res.improved and res.epsilon == res.epsilon_start == 0.0
    assert np.array_equal(res.base_obj, far) and res.dq_norm_rad == 0.0 and res.reason == "no epsilon gain"


def test_the_wrist_bound_is_a_projection_never_a_refusal(setup) -> None:
    """Spheres moved 25 and 40 mm along the palm normal tempt the optimiser to shift the wrist further than the bound:
    the projection keeps every candidate feasible, so no result reports a shift beyond the bound or is refused for
    leaving it (whatever the epsilon outcome); a wider bound is honoured as given."""
    import dataclasses

    gm, base_obj = setup
    for d in (0.025, 0.04):
        off = base_obj.copy()
        off[:3, 3] += d * np.array([0.0, 0.0, 1.0])
        res = refine_grasp(gm, off, _one_sided())
        assert res.shift_m <= RefineConfig().max_shift_m + 1e-9 and res.rot_deg <= 10.0 + 1e-9, d
        assert res.reason != "wrist offset beyond the 15 mm / 10 deg bound", d
        wide = refine_grasp(gm, off, _one_sided(), dataclasses.replace(RefineConfig(), max_shift_m=0.03))
        assert wide.shift_m <= 0.03 + 1e-9 and wide.reason != "wrist offset beyond the 15 mm / 10 deg bound", d


class _FlatEpsilonModel:
    """Every finger touches (reach satisfied), nothing penetrates, and the set is out of closure until joint 4 passes
    1.6 rad (near its 1.745 limit): epsilon is FLAT at 0 below and Q = q4 - 1.6 is not."""

    cfg = RefineConfig()

    def measure(self, base_obj, hand_q):
        q4 = float(hand_q[4])
        return {"epsilon": max(0.0, q4 - 1.6), "quality": q4 - 1.6, "min_gap": 0.0,
                "finger_gap": {"thumb": 0.0, "index": 0.0, "middle": 0.0}, "contacts": 3,
                "fingers": ("thumb", "index", "middle")}


def test_a_touching_set_without_closure_is_brought_into_closure_by_the_signed_quality() -> None:
    q = np.array([0.0, 0.0, -0.5, 0.5, 0.5, 0.5, 0.5])
    res = refine_grasp(_FlatEpsilonModel(), np.eye(4), q)
    assert res.epsilon_start == 0.0 and res.improved and res.epsilon > 0.0
