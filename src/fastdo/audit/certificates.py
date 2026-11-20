# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Certificate presets: DO AS I DO's C_D and our registered C_289, plus loaders for a user's own certificate.

A certificate is any callable that scores an EXECUTED object trajectory ((n, 7) rows x, y, z, qw, qx, qy, qz on the
executor grid, warm-up rows included) and returns a bool, an object with a boolean `passed`, or a mapping with
"passed" (the contract of `nullaudit.verdict`).

- **C_D** (`CD`): DO AS I DO's object test (arXiv 2606.19333, section 4.1), our PRIMARY reading. Mean position
  error < 0.1 m and mean angle < 0.5 rad over the whole executed trace, warm-up included, on a run that executed the
  clip in full. The implementation is `nullaudit.DoAsIDoObjectTest`, itself checked against the registered
  scorer's `daid` to 1e-12.
- **C_289** (`C289`): the object half of our registered certificate (the scorer's `score_trajectory`
  and `informative_success`), ported line for line. On the window [W + 50, T): mean position error <= 0.03 m and mean
  angle <= 30 deg (SR_MT's object half), the object-must-move rule M_i, and the clip must be MT-informative (the
  object held at its first reference pose does NOT meet the object half). Thresholds are inclusive. PLAUS needs the
  scene and the executed qpos rows (`plausibility`); the hand half of SR_MT (fingertip <= 0.06 m and
  0.5 (L1 + L3) <= 0.08 m against MANO) needs hand targets this module cannot build, so the caller passes per-row
  hand errors. When either is missing the verdict is the object half only, a necessary condition, and the result
  says so (`checked`). The hooks see only the object rows they are called on, so `audit_run` refuses a C_289 with
  hooks: on a null they would score the certified run's hand, not the null's.

Threshold configs (`make_certificate` with a mapping or a JSON file) name their preset, `{"preset": "C289", ...}`,
and every value must have its field's type (a bool is `true`, not `"false"`); the applied thresholds are recorded
in the audit report (`thresholds`).
"""

from __future__ import annotations

import importlib
import json
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
from nullaudit import DoAsIDoObjectTest
from nullaudit.poses import as_pose7, quat_angle

from .errors import InputError

__all__ = [
    "C289",
    "CD",
    "PRESETS",
    "C289Certificate",
    "C289Result",
    "C289Thresholds",
    "CDCertificate",
    "CDThresholds",
    "load_certificate",
    "make_certificate",
    "motion_ok",
    "mt_null",
    "plaus_ok",
    "plausibility",
    "resolve_thresholds",
]


@dataclass(frozen=True)
class CDThresholds:
    """C_D: strict `<` on whole-trace means; `require_full` = our PRIMARY reading (n >= T)."""

    name: str = "C_D"
    pos_m: float = 0.1
    rot_rad: float = 0.5
    require_full: bool = True


@dataclass(frozen=True)
class C289Thresholds:
    """C_289 (the registered scorer's module constants). Inclusive `<=` throughout."""

    name: str = "C_289"
    grace_rows: int = 50
    obj_pos_m: float = 0.03
    obj_rot_rad: float = math.radians(30.0)
    tip_m: float = 0.06
    joint_m: float = 0.08
    motion_min_m: float = 0.03
    motion_frac: float = 0.5
    pen_mm: float = 10.0
    carry_frac: float = 0.5
    acc_max: float = 50.0
    plaus_stride: int = 2
    move_speed: float = 0.05
    touch_dist: float = 1e-3
    speed_half: int = 5
    acc_half: int = 10


CD = CDThresholds()
C289 = C289Thresholds()
PRESETS: dict[str, CDThresholds | C289Thresholds] = {"CD": CD, "C_D": CD, "C289": C289, "C_289": C289}


def motion_ok(p_ex: np.ndarray, q_ex: np.ndarray, p_ref: np.ndarray, q_ref: np.ndarray,
              th: C289Thresholds = C289) -> bool:
    """M_i on window rows (index 0 = row W + g): the executed object must move at least half as far as the
    reference, by position when the reference moves >= 3 cm on average, else by angle (the physics scorer's motion test)."""
    dpr = float(np.linalg.norm(p_ref - p_ref[0], axis=1).mean())
    if dpr >= th.motion_min_m:
        return float(np.linalg.norm(p_ex - p_ex[0], axis=1).mean()) >= th.motion_frac * dpr
    rr = float(quat_angle(q_ref, np.repeat(q_ref[:1], len(q_ref), 0)).mean())
    re_ = float(quat_angle(q_ex, np.repeat(q_ex[:1], len(q_ex), 0)).mean())
    return re_ >= th.motion_frac * rr


def mt_null(reference: np.ndarray, warmup_rows: int, required_rows: int, th: C289Thresholds = C289) -> bool:
    """MT-null (the physics scorer's null flags): the object held at reference row 0 meets SR_MT's object half on the
    window. A clip that is MT-null is not informative, and C_289 never counts a success on it."""
    ref = as_pose7(reference)
    seg = ref[warmup_rows + th.grace_rows:required_rows]
    pos = float(np.linalg.norm(seg[:, :3] - ref[0, :3], axis=1).mean())
    rot = float(quat_angle(np.repeat(ref[:1, 3:], len(seg), 0), seg[:, 3:]).mean())
    return bool(pos <= th.obj_pos_m and rot <= th.obj_rot_rad)


def plausibility(model: Any, qpos: np.ndarray, obj_body: int, hand_body: np.ndarray, a: int, b: int,
                 sim_dt: float, obj_qadr: int | None = None, th: C289Thresholds = C289) -> dict[str, Any]:
    """PLAUS statistics on rows [a, b) every `plaus_stride` rows (the physics scorer's plausibility statistics). `hand_body` is a
    (nbody,) bool mask; the study's rule is every body named `<side>_*` except `<side>_object`. `obj_qadr` is the
    qpos address of the object's free joint (default: the scene's only free joint; the study's scenes put it last)."""
    import mujoco

    data = mujoco.MjData(model)
    rows = np.arange(a, b, th.plaus_stride)
    pen = np.zeros(len(rows))
    touch = np.zeros(len(rows), dtype=bool)
    for k, t in enumerate(rows):
        data.qpos[:] = qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for c in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if (b1 == obj_body and hand_body[b2]) or (b2 == obj_body and hand_body[b1]):
                touch[k] = touch[k] or c.dist <= th.touch_dist
                pen[k] = max(pen[k], -c.dist * 1e3)
    if obj_qadr is None:
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise InputError(f"expected exactly one free joint (the object), found {len(free)}")
        obj_qadr = int(model.jnt_qposadr[free[0]])
    p = qpos[:, obj_qadr:obj_qadr + 3]
    last = len(p) - 1
    lo, hi = np.clip(rows - th.speed_half, 0, last), np.clip(rows + th.speed_half, 0, last)
    speed = np.linalg.norm(p[hi] - p[lo], axis=1) / np.maximum(hi - lo, 1) / sim_dt
    moving = speed >= th.move_speed
    l2, h2 = np.clip(rows - th.acc_half, 0, last), np.clip(rows + th.acc_half, 0, last)
    acc = np.linalg.norm(p[h2] - 2 * p[rows] + p[l2], axis=1) / (th.acc_half * sim_dt) ** 2
    return {"pen_mm_p99": float(np.percentile(pen, 99)), "pen_mm_max": float(pen.max()),
            "moving_rows": int(moving.sum()),
            "touch_frac_on_moving": float(touch[moving].mean()) if moving.any() else None,
            "acc_p99": float(np.percentile(acc, 99)), "acc_max": float(acc.max())}


def plaus_ok(st: Mapping[str, Any], th: C289Thresholds = C289) -> bool:
    """PLAUS verdict (the physics scorer's plausibility test): penetration, carry (vacuous when nothing moves) and acceleration."""
    if "pen_mm_p99" not in st:
        return False
    carry = st["touch_frac_on_moving"]
    return bool(st["pen_mm_p99"] <= th.pen_mm and (carry is None or carry >= th.carry_frac)
                and st["acc_p99"] <= th.acc_max)


@dataclass(frozen=True)
class C289Result:
    """One C_289 verdict. `passed` is the conjunction of every term in `checked`; `complete` = all of C_289's terms
    were checked (object half, M_i, informative, PLAUS, hand half)."""

    passed: bool
    full: bool
    obj: bool
    motion: bool
    informative: bool
    obj_pos_m: float
    obj_rot_rad: float
    executed_rows: int
    plaus: bool | None = None
    hand: bool | None = None
    checked: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.plaus is not None and self.hand is not None

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class C289Certificate:
    """C_289 bound to one clip's object reference (on the executor grid, warm-up rows included).

    Args:
        reference: (n, 7) reference rows.
        warmup_rows: W.
        required_rows: T, the clip's simulator rows.
        thresholds: `C289` or a replaced copy.
        plaus: optional callable(executed rows (n, 7)) -> PLAUS statistics dict, e.g. a closure over
            `plausibility` and the run's full qpos rows. Without it PLAUS is not checked.
        hand_errors: optional callable(executed rows) -> (tip_m, joint_m) window means. Without it the hand half
            is not checked.
    Both hooks receive only the object rows, so they score whatever hand rows they close over: use the hooks to
    score one certified run, not nulls (`audit_run` refuses a certificate with hooks).
    """

    reference: np.ndarray
    warmup_rows: int
    required_rows: int
    thresholds: C289Thresholds = C289
    plaus: Callable[[np.ndarray], Mapping[str, Any]] | None = None
    hand_errors: Callable[[np.ndarray], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        lo = self.warmup_rows + self.thresholds.grace_rows
        if self.required_rows <= lo:
            raise InputError(f"C_289 needs T > W + grace ({self.required_rows} <= {lo}): the window is empty")
        if as_pose7(self.reference).shape[0] < self.required_rows:
            raise InputError(f"the reference has {as_pose7(self.reference).shape[0]} rows, T = {self.required_rows}")

    @property
    def checked(self) -> tuple[str, ...]:
        """The terms this certificate checks: always the object half, M_i and the informative rule."""
        return ("obj", "motion", "informative") + (("plaus",) if self.plaus is not None else ()) + (
            ("hand",) if self.hand_errors is not None else ())

    @property
    def complete(self) -> bool:
        return self.plaus is not None and self.hand_errors is not None

    @property
    def label(self) -> str:
        """"C289" when complete, else the checked terms, e.g. "C289[obj+M_i+informative]"."""
        names = {"obj": "obj", "motion": "M_i", "informative": "informative", "plaus": "PLAUS", "hand": "hand"}
        return "C289" if self.complete else "C289[" + "+".join(names[k] for k in self.checked) + "]"

    def informative_clip(self) -> bool:
        """False on an MT-null clip, where C_289 rejects every trajectory, its own reference included."""
        return not mt_null(self.reference, self.warmup_rows, self.required_rows, self.thresholds)

    def __call__(self, executed: np.ndarray) -> C289Result:
        th, ex, ref = self.thresholds, as_pose7(executed), as_pose7(self.reference)
        W, T, n = self.warmup_rows, self.required_rows, int(as_pose7(executed).shape[0])
        a, b = W + th.grace_rows, min(T, n)
        full = n >= T and bool(np.isfinite(ex[:T]).all())
        informative = not mt_null(ref, W, T, th)
        checked = ["obj", "motion", "informative"]
        if b <= a or not np.isfinite(ex[a:b]).all():
            return C289Result(False, bool(full), False, False, informative, math.nan, math.nan, n,
                              None, None, tuple(checked))
        p_err = float(np.linalg.norm(ex[a:b, :3] - ref[a:b, :3], axis=1).mean())
        r_err = float(quat_angle(ex[a:b, 3:], ref[a:b, 3:]).mean())
        obj = bool(full and p_err <= th.obj_pos_m and r_err <= th.obj_rot_rad)
        motion = bool(motion_ok(ex[a:b, :3], ex[a:b, 3:], ref[a:b, :3], ref[a:b, 3:], th))
        ok, detail = obj and motion and informative, {}
        pl = hd = None
        if self.plaus is not None:
            st = dict(self.plaus(ex))
            pl, detail = plaus_ok(st, th), st
            ok, checked = ok and pl, checked + ["plaus"]
        if self.hand_errors is not None:
            tip, joint = self.hand_errors(ex)
            hd = bool(tip <= th.tip_m and joint <= th.joint_m)
            ok, checked = ok and hd, checked + ["hand"]
            detail = {**detail, "tip_m": float(tip), "joint_m": float(joint)}
        return C289Result(bool(ok), bool(full), obj, motion, informative, p_err, r_err, n, pl, hd,
                          tuple(checked), detail)


@dataclass(frozen=True)
class CDCertificate(DoAsIDoObjectTest):
    """DO AS I DO's object test (`nullaudit.DoAsIDoObjectTest`) with the thresholds it was built from."""

    thresholds: CDThresholds = CD


def _typed(base: CDThresholds | C289Thresholds, doc: Mapping[str, Any]) -> dict[str, Any]:
    """`doc`'s values checked against the field types of `base` (bool, int, float, str); refused otherwise."""
    hints = get_type_hints(type(base))
    out: dict[str, Any] = {}
    for k, v in doc.items():
        t = hints[k]
        is_bool = isinstance(v, (bool, np.bool_))
        cast: Callable[[Any], Any]
        if t is bool:
            ok, cast = is_bool, bool
        elif t is int:
            ok, cast = isinstance(v, numbers.Integral) and not is_bool, int
        elif t is float:
            ok, cast = isinstance(v, numbers.Real) and not is_bool and not math.isnan(float(v)), float
        else:
            ok, cast = isinstance(v, str), str
        if not ok:
            raise InputError(f"{base.name} threshold {k!r} must be {t.__name__}, got {type(v).__name__} {v!r}")
        out[k] = cast(v)
    return out


def resolve_thresholds(spec: str | Mapping[str, Any] | CDThresholds | C289Thresholds
                       ) -> CDThresholds | C289Thresholds:
    """The thresholds a certificate spec applies: a thresholds object, a preset name ("CD", "C289"), or a JSON
    config file or mapping that names its preset and overrides fields of it (`{"preset": "C289", "obj_pos_m": 0.05}`).
    A config without "preset", an unknown field or a value of the wrong type (e.g. "false" for a bool) is refused."""
    if isinstance(spec, (CDThresholds, C289Thresholds)):
        return spec
    if isinstance(spec, str):
        if spec in PRESETS:
            return PRESETS[spec]
        try:
            doc = json.loads(Path(spec).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise InputError(f"certificate {spec!r}: not a preset ({', '.join(PRESETS)}) or a readable JSON "
                             f"config ({e})") from e
        if not isinstance(doc, dict):
            raise InputError(f"certificate config {spec}: expected a JSON object")
    else:
        doc = dict(spec)
    preset = doc.pop("preset", None)
    if not isinstance(preset, str) or preset not in PRESETS:
        raise InputError(f'a certificate config must name its "preset" ({", ".join(PRESETS)}); got {preset!r}')
    base = PRESETS[preset]
    unknown = set(doc) - set(asdict(base))
    if unknown:
        raise InputError(f"unknown {base.name} threshold(s): {sorted(unknown)}")
    return replace(base, **_typed(base, doc))


def _pose7(x: np.ndarray) -> np.ndarray:
    try:
        return as_pose7(x)
    except ValueError as e:
        raise InputError(f"reference: {e}") from e


def make_certificate(spec: str | Mapping[str, Any] | CDThresholds | C289Thresholds, reference: np.ndarray, *,
                     warmup_rows: int, required_rows: int | None,
                     plaus: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
                     hand_errors: Callable[[np.ndarray], tuple[float, float]] | None = None) -> Callable[..., Any]:
    """A certificate bound to one clip, from a preset name ("CD", "C289"), a JSON config file or mapping
    ({"preset": "C289", "obj_pos_m": 0.05, ...}, `resolve_thresholds`) or a thresholds object. The result carries
    the applied thresholds as `.thresholds`."""
    th = resolve_thresholds(spec)
    if isinstance(th, CDThresholds):
        return CDCertificate(_pose7(reference), required_rows=required_rows if th.require_full else None,
                             pos_m=th.pos_m, rot_rad=th.rot_rad, thresholds=th)
    if required_rows is None:
        raise InputError("C_289 needs required_rows (T)")
    return C289Certificate(_pose7(reference), int(warmup_rows), int(required_rows), th, plaus, hand_errors)


def load_certificate(target: str) -> Callable[..., Any]:
    """A user's certificate factory from "package.module:function". The factory is called as
    `factory(reference, warmup_rows=W, required_rows=T)` and must return a certificate callable."""
    mod, _, attr = target.partition(":")
    if not mod or not attr:
        raise InputError(f"expected a preset ({', '.join(PRESETS)}), a .json config or 'module:function', "
                         f"got {target!r}")
    try:
        module = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name is None or not (mod == e.name or mod.startswith(e.name + ".")):
            raise                                       # a module the factory's module imports: not our input
        raise InputError(f"certificate factory {target!r}: no module {e.name!r}") from e
    if not callable(getattr(module, attr, None)):
        raise InputError(f"certificate factory {target!r}: {mod} has no callable {attr!r}")
    return getattr(module, attr)
