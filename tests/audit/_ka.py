"""Known-answer suite for fastdo.audit, parameterised by the implementation under test.

`run_ka(Impl())` must return no failures. Every mutant in test_mutants.py must make at least one check FAIL (an
AssertionError); a mutant that only crashes a check is not counted as killed. Each answer is closed form or a
property of the synthetic scene in `_scene.py`:

- census: REST -> 0; PRESS -> |servo force| / weight (equilibrium of a gravity-compensated plate); CARRY -> 1;
  warm-up rows are never sampled; median, not mean; the stride sets the sampled rows; gravity is the run's.
- C_289: on a reference moving 1/64 m per row along x after W rows, an executed copy offset by 2^-5 m in y has
  window error exactly 2^-5 m, which passes an inclusive threshold of 2^-5 m and fails a strict one; the grace rows,
  M_i, the informative rule and the full-run rule each decide one case.
- scene edits: only hand|object pairs are removed (KA-H), the start pedestal is classified as such.
- runner: the motionless-object null and the contact-removed null on the synthetic scene, N0-verified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import fastdo.audit as A
from tests.audit import _scene as S


@dataclass(frozen=True)
class Impl:
    grip_census: Callable[..., Any] = A.grip_census
    make_certificate: Callable[..., Any] = A.make_certificate
    classify_pairs: Callable[..., Any] = A.classify_pairs
    hand_object_pair_names: Callable[..., Any] = A.hand_object_pair_names
    remove_pairs: Callable[..., Any] = A.remove_pairs
    audit_run: Callable[..., Any] = A.audit_run


def _census(impl: Impl, m: Any, states: list[S.State], W: int = 0, stride: int = 1, **kw: Any) -> Any:
    h, o = S.masks(m)
    q, v, c = S.rows(states)
    return impl.grip_census(m, q, v, c, hand_geom=h, object_geom=o, release_row=W, stride=stride,
                            gravity=np.array([0.0, 0.0, -S.G]), **kw)


def ka_census_rest(impl: Impl) -> None:
    g = _census(impl, S.model(), [S.rest(S.model())] * 4)
    assert g.grip_over_weight_p50 == 0.0 and g.contacts_p50 == 0.0 and g.rows_sampled == 4


def ka_census_press(impl: Impl) -> None:
    m = S.model()
    st = S.press(m)
    g = _census(impl, m, [st] * 3)
    want = st.actuator_force / S.WEIGHT
    assert abs(g.grip_over_weight_p50 - want) <= 1e-9 * want, (g.grip_over_weight_p50, want)
    assert abs(g.weight_n - S.WEIGHT) <= 1e-12


def ka_census_carry(impl: Impl) -> None:
    m = S.carry_model()
    g = _census(impl, m, [S.carry(m)])
    assert abs(g.grip_over_weight_p50 - 1.0) <= 1e-9, g.grip_over_weight_p50


def ka_census_skips_warmup(impl: Impl) -> None:
    m = S.model()
    g = _census(impl, m, [S.press(m)] * 5 + [S.rest(m)] * 5, W=5)
    assert g.grip_over_weight_p50 == 0.0 and g.rows_sampled == 5


def ka_census_median(impl: Impl) -> None:
    m = S.model()
    st = S.press(m)
    g = _census(impl, m, [st, st, st, S.rest(m), S.rest(m)])
    assert abs(g.grip_over_weight_p50 - st.actuator_force / S.WEIGHT) <= 1e-9 * st.actuator_force


def ka_census_stride(impl: Impl) -> None:
    m = S.model()
    states = [S.rest(m) if i % 2 else S.press(m) for i in range(10)]
    g = _census(impl, m, states, W=2, stride=2)          # rows 2, 4, 6, 8: all PRESS
    assert g.rows_sampled == 4 and g.grip_over_weight_p50 > 1.0


def ka_census_uses_run_gravity(impl: Impl) -> None:
    m = S.carry_model()
    st = S.carry(m)
    m.opt.gravity[:] = 0.0                               # e.g. left at the warm-up value by an executor
    g = _census(impl, m, [st])
    assert abs(g.grip_over_weight_p50 - 1.0) <= 1e-9, g.grip_over_weight_p50


def ka_census_uses_ctrl(impl: Impl) -> None:
    m = S.model()
    st = S.press(m, depth=0.02)
    g = _census(impl, m, [st])
    assert abs(g.grip_over_weight_p50 - st.actuator_force / S.WEIGHT) <= 1e-9 * st.actuator_force


# ------------------------------------------------------------------------------------------------ C_289
W, T, N = 10, 200, 200
TH = replace(A.C289, obj_pos_m=2.0 ** -5)


def _ref(step: float = 2.0 ** -6) -> np.ndarray:
    k = np.concatenate([np.zeros(W), np.arange(1, N - W + 1)]).astype(np.float64)
    r = np.zeros((N, 7))
    r[:, 0] = k * step
    r[:, 3] = 1.0
    return r


def _c289(impl: Impl, ref: np.ndarray, th: Any = TH, t: int = T) -> Any:
    return impl.make_certificate(th, ref, warmup_rows=W, required_rows=t)


def ka_c289_inclusive(impl: Impl) -> None:
    ref = _ref()
    ex = ref.copy()
    ex[:, 1] = 2.0 ** -5
    r = _c289(impl, ref)(ex)
    assert r.obj_pos_m == 2.0 ** -5 and r.obj and r.passed and r.informative and r.motion


def ka_c289_over_threshold(impl: Impl) -> None:
    ref = _ref()
    ex = ref.copy()
    ex[:, 1] = 2.0 ** -5 + 2.0 ** -20
    assert not _c289(impl, ref)(ex).passed


def ka_c289_grace(impl: Impl) -> None:
    ref = _ref()
    ex = ref.copy()
    ex[W:W + 50, 1] = 0.5                                # off-track only inside the 50 grace rows
    r = _c289(impl, ref)(ex)
    assert r.obj_pos_m == 0.0 and r.passed


def ka_c289_motion(impl: Impl) -> None:
    ref = _ref()
    ref[W + 25:, 0] += 20.0                              # a 20 m jump inside the grace rows keeps the clip informative
    ex = ref.copy()
    ex[W + 50:, 0] = ref[W + 50, 0]                      # object stops at the window start: 1.1 m mean error
    r = _c289(impl, ref, replace(TH, obj_pos_m=10.0))(ex)
    assert r.obj and r.informative and not r.motion and not r.passed


def ka_c289_mt_window(impl: Impl) -> None:
    ref = _ref(step=0.0)
    ref[W:W + 50, 0] = 1.0                               # an excursion inside the grace rows only
    r = _c289(impl, ref)(ref.copy())
    assert not r.informative and not r.passed


def ka_c289_informative(impl: Impl) -> None:
    ref = _ref(step=1e-5)                                # 1.9 mm in total: the static null meets the object half
    r = _c289(impl, ref)(ref.copy())
    assert r.obj and not r.informative and not r.passed
    assert A.mt_null(ref, W, T) and not A.mt_null(_ref(), W, T)


def ka_c289_full(impl: Impl) -> None:
    ref = _ref()
    assert not _c289(impl, ref)(ref[:T - 1]).passed
    assert _c289(impl, ref)(ref).passed


def ka_cd_preset(impl: Impl) -> None:
    ref = _ref()
    cd = impl.make_certificate("CD", ref, warmup_rows=W, required_rows=T)
    static = np.repeat(ref[:1], N, 0)
    assert not cd(static).passed and cd(ref).passed


# ------------------------------------------------------------------------------------------------ scene edits
def ka_pairs(impl: Impl) -> None:
    m = S.model()
    pairs = impl.classify_pairs(m)
    names = impl.hand_object_pair_names(pairs)
    assert names == ["hand_object"], names
    assert A.start_pedestal(pairs)
    edited, removed = impl.remove_pairs(S.XML, names)
    assert A.ka_h(S.XML, edited, removed, len(names))["pass"]
    m2 = S.model(edited)
    assert A.contact_paths(m2, *S.masks(m2)) == {"explicit_pairs": 0, "default_collision_pairs": 0,
                                                 "outside_mask_explicit_pairs": 0,
                                                 "outside_mask_default_collision_pairs": 0}
    assert A.start_pedestal(impl.classify_pairs(m2))


def ka_pairs_refuse_missing(impl: Impl) -> None:
    try:
        impl.remove_pairs(S.XML, ["hand_object", "no_such_pair"])
    except ValueError:
        return
    raise AssertionError("a missing pair name was accepted")


# ------------------------------------------------------------------------------------------------ runner
def _toy_run(impl: Impl, moving: bool, removal: str, **kw: Any) -> Any:
    m = S.model()
    h, o = S.masks(m)
    st = S.press(m)                                      # the certified run: the plate presses the cube down
    n = 120
    u = np.full((n, 1), st.ctrl[0])
    ref = np.repeat(S.object_rows(st.qpos[None]), n, 0)
    if moving:
        ref[20:, 2] += np.linspace(0.0, 0.5, n - 20)
    cert = impl.make_certificate("CD", ref, warmup_rows=20, required_rows=n)
    return impl.audit_run(ref, cert, warmup_rows=20, model=m, hand_geom=h, object_geom=o, controls=u, q0=st.qpos,
                          ctrl0=st.ctrl, weld_eq_ids=(), removal=removal, **kw)


def ka_run_static_pass(impl: Impl) -> None:
    r = _toy_run(impl, moving=False, removal="displace")
    d = r.to_dict()
    assert d["z0"] and d["motionless_object_null"]["passed"]
    assert d["contact_removed_null"]["status"] == "pass" and d["contact_removed_null"]["verified_n0"] is True
    assert d["grip_census"]["grip_over_weight_p50"] > 1.0          # the replayed run presses


def ka_run_moving_fails(impl: Impl) -> None:
    """The pressing controls do not produce the lifted reference: the positive control fails and the null is
    invalid. The certificate's rejection is waived by `allow_uncertified_run` only, not by `allow_replay_mismatch`;
    waived, the null is scored and fails."""
    d = _toy_run(impl, moving=True, removal="displace").to_dict()
    assert not d["z0"] and d["contact_removed_null"]["status"] == "invalid"
    assert d["positive_control"]["replay_passes"] is False and d["grip_census"]["status"] == "not_run"
    d = _toy_run(impl, moving=True, removal="displace", allow_replay_mismatch=True).to_dict()
    assert d["contact_removed_null"]["status"] == "invalid" and d["positive_control"]["overridden"] is False
    d = _toy_run(impl, moving=True, removal="displace", allow_uncertified_run=True).to_dict()
    assert d["contact_removed_null"]["status"] == "fail" and d["positive_control"]["overridden"] is True


FINGER = ('<geom name="collision_hand_finger" type="box" size="0.02 0.02 0.005" pos="0 0 -0.012" mass="0.01" '
          'contype="1" conaffinity="1"/>')
DEFAULT_COLLIDING = S.XML.replace(
    '<geom name="collision_hand_palm" type="box" size="0.05 0.05 0.01" mass="1"/>',
    '<geom name="collision_hand_palm" type="box" size="0.05 0.05 0.01" mass="1"/>' + FINGER).replace(
    f'<geom name="right_object_cube" type="box" size="{S.HALF} {S.HALF} {S.HALF}" mass="{S.MASS}"/>',
    f'<geom name="right_object_cube" type="box" size="{S.HALF} {S.HALF} {S.HALF}" mass="{S.MASS}" conaffinity="1"/>')


def ka_run_displace_removes_default_collisions(impl: Impl) -> None:
    """A hand geom with no `<pair>` (a finger under the plate) presses the cube through MuJoCo's default contype /
    conaffinity collision, a path that no pair edit removes on any engine version (MuJoCo 3.4.0 honours the pair
    gap, 3.13.0 does not, so a known answer on the paired palm alone depends on the engine). Displacing the hand
    removes it: the motionless pressed cube is a verified pass; a pair edit leaves it and the null is invalid."""
    assert DEFAULT_COLLIDING.count("collision_hand_finger") == 1 and DEFAULT_COLLIDING.count('conaffinity="1"') == 2
    m = S.model(DEFAULT_COLLIDING)
    h, o = S.masks(m)
    st = S.press(m)
    n = 120
    ref = np.repeat(S.object_rows(st.qpos[None]), n, 0)
    cert = impl.make_certificate("CD", ref, warmup_rows=20, required_rows=n)
    d = impl.audit_run(ref, cert, warmup_rows=20, model=m, hand_geom=h, object_geom=o, controls=np.full((n, 1),
                       st.ctrl[0]), q0=st.qpos, ctrl0=st.ctrl, weld_eq_ids=(), removal="displace").to_dict()
    assert d["contact_removed_null"]["status"] == "pass" and d["contact_removed_null"]["verified_n0"] is True


def ka_run_pairs_moves_hand(impl: Impl) -> None:
    d = _toy_run(impl, moving=False, removal="pairs", hand_away=(0, 2.0)).to_dict()
    assert d["contact_removed_null"]["status"] == "pass" and d["contact_removed_null"]["verified_n0"] is True


def ka_run_prerem_model(impl: Impl) -> None:
    edited, _ = A.remove_pairs(S.XML, ["hand_object"])
    d = _toy_run(impl, moving=False, removal="model", removed_model=S.model(edited)).to_dict()
    assert d["contact_removed_null"]["status"] == "pass" and d["contact_removed_null"]["verified_n0"] is True


def ka_run_prerem_needs_executor_options(impl: Impl) -> None:
    edited, _ = A.remove_pairs(S.XML, ["hand_object"])
    other = S.model(edited.replace('timestep="0.005"', 'timestep="0.002"'))
    try:
        _toy_run(impl, moving=False, removal="model", removed_model=other)
    except ValueError as e:
        assert "timestep" in str(e), e
    else:
        raise AssertionError("a removed model with other solver options was accepted")
    assert A.match_executor(other, S.model()) == ["opt.timestep"]
    assert A.executor_diff(S.model(), other) == []
    d = _toy_run(impl, moving=False, removal="model", removed_model=other).to_dict()
    assert d["contact_removed_null"]["status"] == "pass" and d["contact_removed_null"]["verified_n0"] is True


def ka_run_leak_is_invalid(impl: Impl) -> None:
    d = _toy_run(impl, moving=False, removal="model", removed_model=S.model()).to_dict()   # pairs NOT removed
    assert d["contact_removed_null"]["status"] == "invalid", d["contact_removed_null"]


def ka_run_census_after_release(impl: Impl) -> None:
    m = S.model()
    h, o = S.masks(m)
    q, v, c = S.rows([S.press(m)] * 20 + [S.rest(m)] * 20)
    ref = S.object_rows(q)
    cert = impl.make_certificate("CD", ref, warmup_rows=20, required_rows=40)
    r = impl.audit_run(ref, cert, warmup_rows=20, model=m, hand_geom=h, object_geom=o,
                       executed=A.Executed(q, v, c), census_stride=1)
    assert r.census.grip_over_weight_p50 == 0.0 and r.census.rows_sampled == 20


CHECKS = {k: v for k, v in globals().items() if k.startswith("ka_") and callable(v)}


def run_ka(impl: Impl) -> list[str]:
    """Names of the checks that FAIL (AssertionError). Any other exception propagates: it is not a kill."""
    failed = []
    for name, check in CHECKS.items():
        try:
            check(impl)
        except AssertionError:
            failed.append(name)
    return failed
