# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""One audited run and the cross-run report.

`audit_run` runs, for one certified run:
- (a) the motionless-object null: the object frozen at its first reference pose, scored by the certificate
  (`nullaudit.static_at_start`; a pass makes the clip Z0);
- the POSITIVE CONTROL, whenever controls are given: the run's controls replayed once with contact on the model the
  audit simulates. It must end normally (no solver warning, finite state, bounded |qvel|) and pass the certificate;
  when the executed qpos rows are stored, the replay must match them to `REPLAY_TOL`; when the run's `sim_dt` is
  given, it must equal the model's timestep. A failed control means the audit is not simulating the executor's
  certified run: the contact-removed null is then `invalid` and the census is not taken, unless every reason is
  waived: `allow_uncertified_run` waives only the certificate's rejection of the replay (to apply the null to
  uncertified runs), `allow_replay_mismatch` the invalid replay, the departure from the
  executed rows and the timestep (the report lists what was waived). A control checked against neither executed
  rows nor `sim_dt`, on a model not declared the executor's own (`model_is_executor`), is certificate-only, and
  the report lists it as a problem;
- (b) the contact-removed null: the run's controls executed with the hand-object contact removed, scored by the
  certificate, and counted only if the N0 ledger (hand mask plus every other dynamic geom) shows no post-release
  force on the object (`nullaudit`'s N0 rule); the ledger part from outside the hand mask is `other_object_force`;
- (c) the grip census of the certified run (`census.grip_census`) on the executed rows, else on the replay.
Every executor setting that was not given and took a default is listed in `defaulted`, and the certificate's
thresholds are recorded in `settings`. A certificate with PLAUS / hand hooks (a complete C289) is refused: the hooks
see only object rows, so on a null they would score the certified run's hand. `summarize_runs` pools runs:
each null's pass rate beside the Z0 base rate (`nullaudit.summarize`), the certificate's completeness label, the
census spread and the list of problems the CLI turns into a non-zero exit code. `report_dict` is the JSON report.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import nullaudit as na
import numpy as np
from nullaudit.poses import as_pose7

from .census import F_RATED_N, GripCensus, _mask, check_object_mask, grip_census
from .errors import InputError

__all__ = ["QUAT_TOL", "REMOVALS", "REPLAY_TOL", "Executed", "PositiveControl", "RunAudit", "audit_run",
           "report_dict", "summarize_runs"]

REMOVALS = ("displace", "pairs", "model")
REPLAY_TOL = 1e-6        # max |replay qpos - executed qpos| (m, rad) for the positive control
TIMESTEP_TOL = 1e-12
QUAT_TOL = 1e-3          # | |q| - 1 | allowed in reference object quaternions
CERT_ONLY = "positive control is certificate-only (no executed rows, no executor_model)"
QUOTE_RULES = (
    "Quote every null-pass rate beside its Z0 base rate (clips where the motionless object passes).",
    "Z0 is a property of the clip and the certificate, not of any method's run.",
    "Grip multiples are simulator forces under the scene's contact model, not the grip the task needs.",
    ("A C289[...] label names the terms that were checked; without PLAUS and the hand half it is a necessary "
     "condition only."),
)


@dataclass(frozen=True)
class Executed:
    """Executed rows of one run: post-step qpos (n, nq), qvel (n, nv) and the applied controls (n, nu)."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray


@dataclass(frozen=True)
class PositiveControl:
    """The with-contact replay of the run's controls (module docstring). `reasons` lists every failed condition,
    `waived` the ones an override waived (`overridden` = all of them); `anchors` what ties the replay to the
    executor's run beyond the certificate: "executed_rows" (qpos, qvel and ctrl compared) or "executor_model". A
    matching `sim_dt` is checked but is not an anchor: another model with the same timestep passes it."""

    replay_status: str
    replay_rows: int
    replay_passes: bool
    max_dev: float | None
    tol: float
    timestep: float
    sim_dt: float | None
    outside_mask_force_n_max: float
    reasons: tuple[str, ...]
    overridden: bool = False
    waived: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()
    max_dev_qvel: float | None = None
    max_dev_ctrl: float | None = None

    @property
    def passed(self) -> bool:
        return not self.reasons

    @property
    def certificate_only(self) -> bool:
        """Nothing but the certificate's verdict ties the replay to the executor's run."""
        return not self.anchors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed, "certificate_only": self.certificate_only}


@dataclass(frozen=True)
class RunAudit:
    clip_id: str
    certificate: str
    clip: na.ClipAudit
    census: GripCensus | None
    census_note: str = ""
    removal: str = ""
    positive_control: PositiveControl | None = None
    defaulted: tuple[str, ...] = ()
    certificate_checked: tuple[str, ...] | None = None
    certificate_complete: bool | None = None
    reference_informative: bool | None = None
    other_object_force: float | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def invalid(self) -> bool:
        return any(o.status == "invalid" for o in self.clip.outcomes.values())

    def to_dict(self) -> dict[str, Any]:
        o = self.clip.outcomes
        static, absent = o["static_at_start"], o["hand_absent"]
        return {
            "clip": self.clip_id, "certificate": self.certificate,
            "certificate_checked": None if self.certificate_checked is None else list(self.certificate_checked),
            "certificate_complete": self.certificate_complete,
            "reference_passes": self.clip.reference_passes, "reference_informative": self.reference_informative,
            "motionless_object_null": {"passed": static.passed, **_clean(static.detail)},
            "z0": self.clip.z0,
            "positive_control": None if self.positive_control is None else self.positive_control.to_dict(),
            "contact_removed_null": {"status": absent.status, "verified_n0": absent.verified_n0, "note": absent.note,
                                     "removal": self.removal, "other_object_force": self.other_object_force,
                                     **_clean(absent.detail)},
            "grip_census": (self.census.to_dict() if self.census is not None
                            else {"status": "not_run", "note": self.census_note or "needs a scene and executed rows"}),
            "defaulted": list(self.defaulted), "settings": self.settings, "invalid": self.invalid,
        }


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}


def _check_rows(name: str, x: np.ndarray, T: int) -> None:
    if x.ndim < 1 or x.shape[0] < T:
        raise InputError(f"{name} has {x.shape[0] if x.ndim else 0} rows, the clip needs T = {T}")


def _check_pose_rows(name: str, p7: np.ndarray, tol: float | None) -> tuple[int, float]:
    """Refuse empty or non-finite rows, exact-zero quaternions (no rotation; refused whatever `tol`), and
    quaternions whose norm is off 1 by more than `tol`. Returns the count of rows off unit norm (at QUAT_TOL when
    `tol=None`, which audits them and reports the count) and the minimum quaternion norm."""
    if p7.shape[0] == 0:
        raise InputError(f"{name} has no rows")
    if not np.isfinite(p7).all():
        raise InputError(f"{name} has non-finite rows")
    norm = np.linalg.norm(p7[:, 3:], axis=1)
    zero = int((norm == 0.0).sum())
    if zero:
        raise InputError(f"{name}: {zero} quaternions are exactly zero, which is no rotation (refused even with "
                         "quat_tol=None)")
    bad = int((np.abs(norm - 1.0) > (QUAT_TOL if tol is None else tol)).sum())
    if bad and tol is not None:
        raise InputError(f"{name}: {bad} quaternions off unit norm by > {tol} (min norm {float(norm.min()):.3g}); "
                         "pass quat_tol=None to audit such a reference anyway (the count is reported)")
    return bad, float(norm.min())


def _array(name: str, x: Any, shape: tuple[int | None, ...]) -> np.ndarray:
    """`x` as a finite float array of `shape` (None = any length); refused otherwise."""
    try:
        a = np.asarray(x, np.float64)
    except (TypeError, ValueError) as e:
        raise InputError(f"{name} must be numeric ({e})") from e
    if a.ndim != len(shape) or any(w is not None and a.shape[i] != w for i, w in enumerate(shape)):
        want = "(" + ", ".join("n" if w is None else str(w) for w in shape) + ("," if len(shape) == 1 else "") + ")"
        raise InputError(f"{name} has shape {a.shape}, expected {want}")
    if not np.isfinite(a).all():
        raise InputError(f"{name} has non-finite values")
    return a


def _executed(executed: Executed, model: Any, T: int) -> Executed:
    """The executed rows, finite, with one row per step for qpos, qvel and ctrl and at least T rows."""
    if model is None:
        q = _array("executed qpos", executed.qpos, (None, None))
        return Executed(q, _array("executed qvel", executed.qvel, (q.shape[0], None)),
                        _array("executed ctrl", executed.ctrl, (q.shape[0], None)))
    q = _array("executed qpos", executed.qpos, (None, model.nq))
    return Executed(q, _array("executed qvel", executed.qvel, (q.shape[0], model.nv)),
                    _array("executed ctrl", executed.ctrl, (q.shape[0], model.nu)))


def _hand_away(hand_away: Any, nu: int) -> tuple[int, float]:
    """(control column, offset), refused unless it is two finite numbers with an integral column in [0, nu)."""
    try:
        ha = np.asarray(hand_away, np.float64).ravel()
    except (TypeError, ValueError) as e:
        raise InputError(f"hand_away {hand_away!r}: expected (control column, offset)") from e
    if ha.shape != (2,) or not np.isfinite(ha).all() or ha[0] != np.round(ha[0]) or not 0 <= ha[0] < nu:
        raise InputError(f"hand_away {np.asarray(hand_away).tolist()!r}: expected two numbers (control column, "
                         f"an integer in [0, {nu}), and a finite offset)")
    return int(ha[0]), float(ha[1])


def _check_removed(model: Any, removed: Any, hand: np.ndarray, obj: np.ndarray) -> None:
    """The removed model must be the executor's scene minus hand|object pairs: equal outside names and those
    pairs (`match.executor_diff`; a layout mismatch raises there)."""
    from .match import executor_diff, missing_pairs

    bad = executor_diff(model, removed)
    if bad:
        raise InputError(f"removed_model is not the executor's scene minus pairs (differs in {', '.join(bad[:8])}); "
                         "compile it with compile_without_pairs(..., executor_model=model)")
    gone = [(a, b) for a, b in missing_pairs(model, removed) if not ((hand[a] and obj[b]) or (obj[a] and hand[b]))]
    if gone:
        raise InputError(f"removed_model also lacks {len(gone)} pair(s) that are not hand|object, e.g. {gone[:3]}")


class _PreRemoved:
    """A simulator on a scene whose hand-object pairs were deleted beforehand: the hand-absent call runs that scene
    as is, and its force ledger still decides whether the run is N0."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True) -> na.Rollout:
        return self.inner(controls, hand_object_contact=True)


def audit_run(reference: np.ndarray, certificate: Callable[[np.ndarray], Any], *, clip_id: str = "",
              certificate_name: str = "", warmup_rows: int, required_rows: int | None = None, model: Any = None,
              hand_geom: Any = None, object_geom: Any = None, executed: Executed | None = None,
              controls: np.ndarray | None = None, q0: np.ndarray | None = None, ctrl0: np.ndarray | None = None,
              weld_eq_ids: Sequence[int] | None = None, gravity: np.ndarray | None = None,
              ctrl_steps: int | None = None, terminate_z: float | None = None, removal: str = "displace",
              removed_model: Any = None, hand_away: tuple[int, float] | None = None, census_stride: int = 5,
              ref_pos: np.ndarray | None = None, sim_dt: float | None = None, replay_tol: float = REPLAY_TOL,
              allow_replay_mismatch: bool = False, allow_uncertified_run: bool = False,
              model_is_executor: bool = False, census: bool = True,
              quat_tol: float | None = QUAT_TOL, allow_short_controls: bool = False) -> RunAudit:
    """Audit one certified run (module docstring). Every refused input raises `InputError`.

    Args:
        reference: (n, 7) or (n, 4, 4) reference object rows on the executor grid, warm-up rows included; finite,
            with unit quaternions (to `QUAT_TOL`), and at least T rows.
        certificate: a certificate bound to this clip (`certificates.make_certificate`). One with PLAUS or hand
            hooks (`plaus`, `hand_errors`) is refused.
        warmup_rows: W, the release row.
        required_rows: T (default: the certificate's `required_rows`; needed when controls are given).
        model, hand_geom, object_geom: the MuJoCo scene the audit simulates and its hand / object geom masks. The
            object mask defaults to the geoms of the free joint's body; a given mask must be consistent with that
            body (`census.check_object_mask`).
        executed: the certified run's stored rows: the census uses them, and the positive control compares to them.
        controls: (>= T, nu) the run's controls; with them the positive control and (b) run.
        q0, weld_eq_ids: required with controls (pass `weld_eq_ids=()` for none). ctrl0, gravity, ctrl_steps,
            terminate_z: default to zeros, the model's gravity, 1 and -inf, and each default used is listed in
            `defaulted`.
        removal: "displace" (hand geoms moved 1 km), "pairs" (hand-object pairs given a 10 m gap AND the hand
            moved by `hand_away` = (control column, offset), required), or "model" (`removed_model` is the
            executor's scene with the hand-object pairs deleted, `scene.compile_without_pairs(...,
            executor_model=model)`).
        sim_dt: the run's simulator timestep, checked against the model's.
        replay_tol: the positive control's tolerance on |replay - executed qpos|.
        allow_replay_mismatch: waive an invalid replay, a departure from the executed rows and a timestep mismatch.
        allow_uncertified_run: waive only the certificate's rejection of the replay (the run need not be certified).
        model_is_executor: `model` is the executor's own compiled model (e.g. its saved `.mjb`), not a recompile of
            its scene file; it anchors the positive control like executed rows or `sim_dt` do.
        census: take the grip census (default True).
        quat_tol: refuse reference quaternions off unit norm by more than this; None audits them and reports the
            count (`reference_nonunit_quaternions`). The minimum norm is always reported; an exact-zero quaternion
            is always refused.
        allow_short_controls: audit a run whose stored controls end before T (a run that stopped early, as some
            study arms did) instead of refusing it; the row count is reported (`short_controls`).
    """
    if removal not in REMOVALS:
        raise InputError(f"removal {removal!r}: expected one of {REMOVALS}")
    if getattr(certificate, "plaus", None) is not None or getattr(certificate, "hand_errors", None) is not None:
        raise InputError("the certificate has PLAUS / hand hooks (a complete C289): the hooks see only object rows, "
                         "so on a null they would score the certified run's hand, not the null's; audit with the "
                         "object half (make_certificate without plaus and hand_errors) and score the certified run "
                         "with the hooks separately")
    try:
        ref = as_pose7(reference)
    except ValueError as e:
        raise InputError(f"reference: {e}") from e
    bad_quat, min_norm = _check_pose_rows("reference", ref, quat_tol)
    defaulted: list[str] = []
    t_req = required_rows if required_rows is not None else getattr(certificate, "required_rows", None)
    if t_req is None:
        if controls is not None:
            raise InputError("required_rows (T) is needed when controls are given")
        t_req = ref.shape[0]
        defaulted.append("required_rows")
    T, W = int(t_req), int(warmup_rows)
    if T < 1 or not 0 <= W < T:
        raise InputError(f"need 0 <= warmup_rows < required_rows and required_rows >= 1, got W = {W}, T = {T}")
    if not (np.isfinite(replay_tol) and replay_tol >= 0):
        raise InputError(f"replay_tol {replay_tol!r} must be finite and >= 0")
    if int(census_stride) != census_stride or census_stride < 1:
        raise InputError(f"census_stride {census_stride!r} must be an integer >= 1")
    if sim_dt is not None and not (np.isfinite(sim_dt) and sim_dt > 0):
        raise InputError(f"sim_dt {sim_dt!r} must be finite and > 0")
    if removal == "model" and controls is not None and removed_model is None:
        raise InputError("removal='model' needs removed_model")
    _check_rows("reference", ref, T)
    obj = hand = None
    if model is not None:
        from .scene import free_object_geom

        obj = free_object_geom(model)
        if hand_geom is not None:
            hand = _mask(hand_geom, model.ngeom)
            if object_geom is not None:
                obj = check_object_mask(model, object_geom, hand)
            if not hand.any() or (hand & obj).any():
                raise InputError("the hand mask is empty or overlaps the object")
    if executed is not None:
        executed = _executed(executed, model, T)
        _check_rows("executed rows", np.asarray(executed.qpos), T)
    from nullaudit.mujoco_adapter import QVEL_MAX

    th = getattr(certificate, "thresholds", None)
    settings: dict[str, Any] = {"T": T, "W": W, "reference_nonunit_quaternions": bad_quat,
                                "reference_min_quaternion_norm": min_norm,
                                "thresholds": asdict(th) if is_dataclass(th) and not isinstance(th, type) else None,
                                "quat_tol": quat_tol, "replay_tol": float(replay_tol),
                                "census_stride": int(census_stride),
                                "qvel_max": QVEL_MAX, "timestep_tol": TIMESTEP_TOL, "f_rated_n": F_RATED_N}
    wids: tuple[int, ...] = ()
    sim: Any = None
    absent_u = pc = replay = None
    if controls is not None:
        if model is None or hand is None or obj is None:
            raise InputError("controls need the model and the hand mask")
        if q0 is None:
            raise InputError("q0 (the executor's start state) is required with controls")
        if weld_eq_ids is None:
            raise InputError("weld_eq_ids is required with controls (pass () if the executor has no warm-up weld)")
        u = _array("controls", controls, (None, model.nu))
        if u.shape[0] < T and allow_short_controls:
            settings["short_controls"] = int(u.shape[0])
        else:
            _check_rows("controls", u, T)
        q0 = _array("q0", q0, (model.nq,))
        if ctrl0 is None:
            ctrl0 = np.zeros(model.nu)
            defaulted.append("ctrl0")
        else:
            ctrl0 = _array("ctrl0", ctrl0, (model.nu,))
            if u.shape[0] and not np.array_equal(ctrl0.astype(np.float32), u[0].astype(np.float32)):
                settings["ctrl0_overwritten"] = True          # the executor sets row 0's control before any step
        if gravity is None:
            gravity = np.array(model.opt.gravity)
            defaulted.append("gravity")
        gravity = _array("gravity", gravity, (3,))
        if ctrl_steps is None:
            ctrl_steps = 1
            defaulted.append("ctrl_steps")
        if int(ctrl_steps) != ctrl_steps or ctrl_steps < 1:
            raise InputError(f"ctrl_steps {ctrl_steps!r} must be an integer >= 1")
        if terminate_z is None:
            terminate_z = -np.inf
            defaulted.append("terminate_z")
        if np.isnan(terminate_z) or terminate_z == np.inf:
            raise InputError(f"terminate_z {terminate_z!r} must be a number or -inf (no drop check)")
        bad_w = [int(i) for i in weld_eq_ids if int(i) != i or not 0 <= int(i) < model.neq]
        if bad_w:
            raise InputError(f"weld_eq_ids {bad_w} outside [0, neq = {model.neq})")
        if removal == "pairs" and hand_away is None:
            raise InputError("removal='pairs' leaks force unless the hand is moved: pass hand_away")
        away = None if hand_away is None else _hand_away(hand_away, model.nu)
        wids = tuple(int(i) for i in weld_eq_ids)
        ex_kw: dict[str, Any] = {"q0": q0, "hand_geom": hand,
                                 "ctrl0": np.asarray(ctrl0, np.float64), "warmup_rows": W,
                                 "gravity": np.asarray(gravity, np.float64), "weld_eq_ids": wids,
                                 "ctrl_steps": int(ctrl_steps), "terminate_z": float(terminate_z)}
        settings.update(weld_eq_ids=list(wids), ctrl_steps=int(ctrl_steps),
                        terminate_z=float(terminate_z), gravity=[float(x) for x in ex_kw["gravity"]],
                        hand_away=None if away is None else list(away))
        pc, replay = _positive_control(model, u, ex_kw, certificate, executed, sim_dt, replay_tol, W,
                                       allow_replay_mismatch, allow_uncertified_run, model_is_executor)
        sim = _absent_simulator(model, removal, removed_model, ex_kw, hand, obj)
        if removal == "pairs":
            assert away is not None
            absent_u = na.move_hand_away(u, column=away[0], offset=away[1])
    clip = na.audit_clip(ref, certificate, clip_id=clip_id, simulator=sim,
                         controls=None if sim is None else np.asarray(controls, np.float64),
                         release_row=W, absent_controls=absent_u, nulls=("static_at_start", "hand_absent"))
    other = None
    if sim is not None:
        inner = sim.inner if isinstance(sim, _PreRemoved) else sim
        if inner.last_other_force is not None:
            other = float(np.sum(inner.last_other_force[W:]))
    if pc is not None and not pc.passed and not pc.overridden:
        bad = na.NullOutcome("hand_absent", "invalid", "positive control failed: " + "; ".join(pc.reasons), False,
                             dict(clip.outcomes["hand_absent"].detail))
        clip = na.ClipAudit(clip.clip_id, clip.z0, clip.reference_passes, {**clip.outcomes, "hand_absent": bad})
    gc, note = _census(model, hand, obj, executed, replay, pc, census, W, gravity, census_stride, ref_pos,
                       wids if controls is not None else tuple(int(i) for i in (weld_eq_ids or ())))
    label = getattr(certificate, "label", None) or certificate_name or type(certificate).__name__
    checked = getattr(certificate, "checked", None)
    informative = certificate.informative_clip() if hasattr(certificate, "informative_clip") else None
    return RunAudit(clip_id=clip_id, certificate=str(label), clip=clip, census=gc, census_note=note,
                    removal=removal if sim is not None else "", positive_control=pc, defaulted=tuple(defaulted),
                    certificate_checked=None if checked is None else tuple(checked),
                    certificate_complete=getattr(certificate, "complete", None), reference_informative=informative,
                    other_object_force=other, settings=settings)


def _positive_control(model: Any, u: np.ndarray, ex_kw: dict[str, Any], certificate: Callable[[np.ndarray], Any],
                      executed: Executed | None, sim_dt: float | None, tol: float, W: int, allow_mismatch: bool,
                      allow_uncertified: bool, model_is_executor: bool) -> tuple[PositiveControl, Executed]:
    from nullaudit.mujoco_adapter import MujocoExecutor

    ex = MujocoExecutor(model=model, **ex_kw)
    ro = ex(u)
    assert ex.last_qpos is not None and ex.last_qvel is not None and ex.last_ctrl is not None
    replay = Executed(ex.last_qpos, ex.last_qvel, ex.last_ctrl)
    passes = na.verdict(certificate(as_pose7(ro.object_traj)))
    reasons: list[tuple[str, str]] = []               # (what waives it, text)
    if ro.status == "invalid":
        reasons.append(("mismatch", f"the replay is invalid ({ex.last_invalid or 'non-finite state'})"))
    if not passes:
        reasons.append(("uncertified", f"the certificate rejects the with-contact replay (status {ro.status})"))
    dev = None
    devs: dict[str, float] = {}
    if executed is not None:
        q = np.asarray(executed.qpos, np.float64)
        n = min(q.shape[0], replay.qpos.shape[0])
        dev = float(np.abs(replay.qpos[:n] - q[:n]).max()) if n else float("inf")
        if replay.qpos.shape[0] < min(q.shape[0], u.shape[0]):
            dev = float("inf")                       # the replay stopped before the executed run did
        if not dev <= tol:
            reasons.append(("mismatch", f"the replay departs from the executed rows by {dev:.3g} > {tol:g}"))
        for name, a, b in (("qvel", replay.qvel, executed.qvel), ("ctrl", replay.ctrl, executed.ctrl)):
            k = min(a.shape[0], b.shape[0])
            dk = float(np.abs(a[:k] - np.asarray(b, np.float64)[:k]).max()) if k else float("inf")
            devs[name] = dk
            if not dk <= tol:
                text = f"the replay's {name} departs from the executed rows by {dk:.3g} > {tol:g}"
                reasons.append(("mismatch", text))
    if sim_dt is not None and abs(float(model.opt.timestep) - float(sim_dt)) > TIMESTEP_TOL:
        reasons.append(("mismatch", f"model timestep {model.opt.timestep:g} != the run's sim_dt {float(sim_dt):g}"))
    allowed = {"mismatch": allow_mismatch, "uncertified": allow_uncertified}
    texts = tuple(t for _, t in reasons)
    waived = tuple(t for k, t in reasons if allowed[k])
    anchors = tuple(k for k, on in (("executed_rows", executed is not None), ("executor_model", model_is_executor))
                    if on)
    of = ex.last_other_force
    outside = float(of[W:].max()) if of is not None and of.shape[0] > W else 0.0
    pc = PositiveControl(replay_status=ro.status, replay_rows=int(replay.qpos.shape[0]), replay_passes=passes,
                         max_dev=dev, tol=tol, timestep=float(model.opt.timestep),
                         sim_dt=None if sim_dt is None else float(sim_dt), outside_mask_force_n_max=outside,
                         reasons=texts, overridden=bool(texts) and len(waived) == len(texts), waived=waived,
                         anchors=anchors, max_dev_qvel=devs.get("qvel"), max_dev_ctrl=devs.get("ctrl"))
    return pc, replay


def _absent_simulator(model: Any, removal: str, removed_model: Any, ex_kw: dict[str, Any], hand: np.ndarray,
                      obj: np.ndarray) -> Any:
    from nullaudit.mujoco_adapter import MujocoExecutor

    if removal == "model":
        if removed_model is None:
            raise InputError("removal='model' needs removed_model")
        _check_removed(model, removed_model, hand, obj)
        return _PreRemoved(MujocoExecutor(model=removed_model, **ex_kw))
    return MujocoExecutor(model=model, absent_mode="displace" if removal == "displace" else "pairs", **ex_kw)


def _census(model: Any, hand: Any, obj: Any, executed: Executed | None, replay: Executed | None,
            pc: PositiveControl | None, want: bool, W: int, gravity: Any, stride: int, ref_pos: Any,
            weld_eq_ids: Sequence[int] | None) -> tuple[GripCensus | None, str]:
    if not want:
        return None, "not requested"
    if model is None or hand is None:
        return None, "needs a scene and a hand mask"
    if pc is not None and not pc.passed and not pc.overridden:
        return None, "positive control failed"
    rows, status = (executed, "executed") if executed is not None else (replay, pc.replay_status if pc else "")
    if rows is None:
        return None, "needs executed rows or controls"
    gc = grip_census(model, rows.qpos, rows.qvel, rows.ctrl, hand_geom=hand, object_geom=obj, release_row=W,
                     gravity=gravity, stride=stride, ref_pos=ref_pos, weld_eq_ids=tuple(weld_eq_ids or ()),
                     replay_status=status)
    return gc, ""


def _names(runs: Sequence[RunAudit]) -> list[str]:
    """Each run's clip id, or "<run i>" (its position) when the id is empty. Two runs with one id are refused:
    the pooled report keys its per-run entries by this name, and a repeated name would drop one of them."""
    names = [r.clip_id or f"<run {i}>" for i, r in enumerate(runs)]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"runs share an id: {', '.join(repeated)}; give each run its own clip id")
    return names


def summarize_runs(runs: Sequence[RunAudit]) -> dict[str, Any]:
    """Pooled report: null rates beside the Z0 base rate, the certificate label, the census spread and problems.
    Runs are named by clip id, or by "<run i>" when the id is empty."""
    rep = na.summarize([r.clip for r in runs])
    named = list(zip(_names(runs), runs, strict=True))
    by_design = [c for c, r in named if not r.clip.reference_passes and r.reference_informative is False]
    rejected = [c for c, r in named if not r.clip.reference_passes and c not in by_design]
    labels = sorted({r.certificate for r in runs})
    lines = [f"Certificate: {', '.join(labels)}" + (" (incomplete: a necessary condition only)"
                                                    if any(r.certificate_complete is False for r in runs) else "")]
    lines += [ln for ln in rep.headline().splitlines() if not ln.startswith("WARNING: the test rejects")]
    if rejected:
        more = "..." if len(rejected) > 5 else ""
        lines.append(f"WARNING: the certificate rejects its own reference on {len(rejected)} clips "
                     f"({', '.join(rejected[:5])}{more})")
    if by_design:
        lines.append(f"Note: {len(by_design)} clips are MT-null; C_289 rejects every trajectory on them, its own "
                     "reference included (not informative).")
    pcs = [(c, r.positive_control) for c, r in named if r.positive_control is not None]
    pc_failed = [c for c, p in pcs if not p.passed]
    pc_over = [c for c, p in pcs if p.overridden]
    pc_cert_only = [c for c, p in pcs if p.certificate_only]
    if pcs:
        lines.append(f"Positive control (with-contact replay): {len(pcs) - len(pc_failed)}/{len(pcs)} pass"
                     + (f"; {len(pc_over)} failures overridden" if pc_over else "")
                     + (f"; {len(pc_cert_only)} certificate-only" if pc_cert_only else ""))
    outside = [c for c, p in pcs if p.outside_mask_force_n_max > 0]
    g = [r.census.grip_over_weight_p50 for r in runs if r.census is not None and
         r.census.grip_over_weight_p50 is not None]
    excluded = {c: (r.census_note if r.census is None else "no post-release row sampled")
                for c, r in named if r.census is None or r.census.grip_over_weight_p50 is None}
    grip: dict[str, Any] = {"runs": len(g), "excluded": len(excluded), "excluded_runs": excluded}
    if g:
        a = np.asarray(g, np.float64)
        grip.update({"median_of_run_medians": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
                     "runs_over_10x": int((a > 10).sum()), "runs_with_contact": int((a > 0).sum())})
    invalid = [c for c, r in named if r.invalid]
    defaulted = {c: list(r.defaulted) for c, r in named if r.defaulted}
    nonunit = {c: r.settings["reference_nonunit_quaternions"] for c, r in named
               if r.settings.get("reference_nonunit_quaternions")}
    norms = [r.settings.get("reference_min_quaternion_norm") for r in runs]
    min_norm = min((x for x in norms if x is not None), default=None)
    short = {c: r.settings["short_controls"] for c, r in named if "short_controls" in r.settings}
    tols = {c: r.settings["replay_tol"] for c, r in named if r.settings.get("replay_tol", REPLAY_TOL) != REPLAY_TOL}
    ctrl0 = [c for c, r in named if r.settings.get("ctrl0_overwritten")]
    truncated = {c: r.census.truncated_at for c, r in named
                 if r.census is not None and r.census.truncated_at is not None}
    unsampled = [c for c, r in named if r.census is not None and r.census.grip_over_weight_p50 is None]
    problems = []
    if tols:
        problems.append(f"non-default replay tolerance on {len(tols)} runs: "
                        + ", ".join(f"{k} ({v:g})" for k, v in list(tols.items())[:5]))
    if ctrl0:
        problems.append(f"ctrl0 differs from the first control row on {len(ctrl0)} runs and has no effect (the "
                        f"executor sets row 0's control before the first step): {', '.join(ctrl0[:5])}")
    if truncated:
        problems.append(f"grip census stopped at a non-finite state on {len(truncated)} runs: "
                        + ", ".join(f"{k} (row {v})" for k, v in list(truncated.items())[:5]))
    if unsampled:
        problems.append(f"grip census sampled no post-release row on {len(unsampled)} runs: {', '.join(unsampled[:5])}")
    if short:
        problems.append(f"{len(short)} runs have controls shorter than T (audited with allow_short_controls)")
    if nonunit:
        problems.append(f"{len(nonunit)} references have quaternions off unit norm by > {QUAT_TOL} (min norm "
                        f"{min_norm:.3g}; audited with quat_tol=None)")
    if rejected:
        problems.append(f"the certificate rejects its own reference on {len(rejected)} clips")
    if invalid:
        problems.append(f"{len(invalid)} runs invalid: {', '.join(invalid[:5])}")
    if pc_failed:
        problems.append(f"positive control failed on {len(pc_failed)} runs: {', '.join(pc_failed[:5])}")
    if pc_cert_only:
        problems.append(f"{CERT_ONLY} on {len(pc_cert_only)} runs: {', '.join(pc_cert_only[:5])}")
    if defaulted:
        problems.append(f"defaults used on {len(defaulted)} runs: "
                        + "; ".join(f"{k}: {', '.join(v)}" for k, v in list(defaulted.items())[:3]))
    if outside:
        problems.append(f"object contact from geoms outside the hand mask in the certified replay of {len(outside)} "
                        f"runs: {', '.join(outside[:5])}")
    d = rep.to_dict()
    d.update(reference_rejected=rejected, reference_rejected_by_design=by_design)
    return {"headline": "\n".join(lines), **d, "certificates": labels, "grip_census": grip,
            "positive_control": {"tested": len(pcs), "failed": pc_failed, "overridden": pc_over,
                                 "certificate_only": pc_cert_only},
            "invalid_runs": invalid, "defaulted": defaulted, "outside_mask_contact": outside,
            "reference_nonunit_quaternions": nonunit, "reference_min_quaternion_norm": min_norm,
            "short_controls": short, "non_default_replay_tol": tols, "census_truncated": truncated,
            "problems": problems}


def report_dict(runs: Sequence[RunAudit], *, tool_version: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """The JSON report: per-run results, the pooled summary and the quoting rules."""
    return {"tool": "fastdo.audit", "version": tool_version, "inputs": inputs or {},
            "runs": [r.to_dict() for r in runs], "summary": summarize_runs(runs), "quote_rules": list(QUOTE_RULES)}
