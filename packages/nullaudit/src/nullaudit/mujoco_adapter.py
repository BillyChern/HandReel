# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""Optional MuJoCo simulator adapter: a `Simulator` for `audit.audit_clip`.

Install with `pip install "nullaudit[mujoco]"`; import with `from nullaudit.mujoco_adapter import MujocoExecutor`.

The executor loop is the one our registered certifier uses:
- reset to (q0, qvel 0, ctrl0 as float32);
- row s: the warm-up welds on and gravity 0 while s < W, otherwise welds off and the model's gravity; ctrl = the
  row's control rounded to float32; one `mj_step`;
- stop on a BADQACC / BADQPOS / BADQVEL / BADCTRL warning (MuJoCo zeroes a NaN, inf or huge control and goes
  on, so the applied control is not the stored one), a CONTACTFULL / CNSTRFULL warning (contacts or constraints dropped
  for lack of memory), a non-finite state, or any |qvel| above `qvel_max` (a diverging solve long before MuJoCo's
  own 1e10 bound) ("invalid"; the reason is kept in `last_invalid`);
- stop at the end of a control block with the object below `terminate_z` ("drop");
- otherwise "clip_end" after T rows.
Row t's hand-object force (the N0 ledger) is the sum of normal forces (`mj_contactForce`) over active contacts
between a geom of the free object's body and any LEDGER geom, from the step that executed row t. The ledger is the
hand mask plus every geom on a non-static body other than the object (`contact_ledger_geoms`: a body welded to the
world is static, a mocap body is not), so a hand geom left out of the mask still shows up. The part from geoms
outside the hand mask is kept separately (`last_other_force`).

Hand-absent (`hand_object_contact=False`) edits a COPY of the model, in one of two modes:
- `absent_mode="pairs"` (default): every explicit hand-object `<pair>` gets margin 0 and gap 10 m.
  The edit alone does NOT remove contact: MuJoCo 3.13 still produced hand-object force through such a pair in our
  synthetic test, the gap was seen to be ignored for box-box pairs, and default (contype) collisions are untouched.
  Move the hand away as well, e.g. `nulls.move_hand_away(controls, column=0)`;
- `absent_mode="displace"`: every hand geom is moved 1 km along its body's z axis (body inertias are compiled and
  do not change), so no collision path can reach the object.
Either way the force ledger is returned, and the audit marks a run with any post-release hand force `invalid`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .audit import Rollout

__all__ = ["QVEL_MAX", "MujocoExecutor", "contact_ledger_geoms", "dynamic_geoms"]

_BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADCTRL, mujoco.mjtWarning.mjWARN_CONTACTFULL, mujoco.mjtWarning.mjWARN_CNSTRFULL)
QVEL_MAX = 1e3        # |qvel| (m/s, rad/s) above which a run counts as diverged


def dynamic_geoms(model: Any) -> np.ndarray:
    """(ngeom,) bool: geoms on a body that can move (not welded to the world, or a mocap body)."""
    body = np.asarray(model.geom_bodyid)
    return np.asarray((model.body_weldid[body] != 0) | (model.body_mocapid[body] >= 0))


def contact_ledger_geoms(model: Any, hand_geom: np.ndarray, object_geom: np.ndarray) -> np.ndarray:
    """(ngeom,) bool: the hand mask plus every dynamic geom outside the object (module docstring)."""
    return (np.asarray(hand_geom, bool) | dynamic_geoms(model)) & ~np.asarray(object_geom, bool)


@dataclass(eq=False)
class MujocoExecutor:
    """Executes control rows on one MuJoCo scene with exactly one free (object) joint.

    Args:
        model: the scene's MjModel (never mutated by a hand-absent run; a normal run restores gravity).
        q0: start qpos (nq,).
        hand_geom: (ngeom,) bool mask, or a sequence of geom ids, of the hand's geoms.
        ctrl0: initial ctrl (default zeros).
        warmup_rows: W; the welds in `weld_eq_ids` are active and gravity is 0 on rows < W.
        gravity: post-release gravity (default: the model's).
        weld_eq_ids: equality constraints that hold the object during the warm-up.
        ctrl_steps: rows per control block (the drop check runs at block ends).
        terminate_z: drop height of the object's z (default: never).
        absent_mode: "pairs" or "displace", see the module docstring.
        qvel_max: a run whose |qvel| exceeds this on any row is "invalid" (default `QVEL_MAX`).
    """

    model: Any
    q0: np.ndarray
    hand_geom: Any
    ctrl0: np.ndarray | None = None
    warmup_rows: int = 0
    gravity: np.ndarray | None = None
    weld_eq_ids: tuple[int, ...] = ()
    ctrl_steps: int = 1
    terminate_z: float = -np.inf
    absent_mode: str = "pairs"
    qvel_max: float = QVEL_MAX
    last_qpos: np.ndarray | None = field(default=None, init=False)
    last_qvel: np.ndarray | None = field(default=None, init=False)
    last_ctrl: np.ndarray | None = field(default=None, init=False)
    last_hand_force: np.ndarray | None = field(default=None, init=False)
    last_other_force: np.ndarray | None = field(default=None, init=False)
    last_invalid: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        m = self.model
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"expected exactly one free joint (the object), found {len(free)}")
        self.obj_body = int(m.jnt_bodyid[free[0]])
        self.obj_qadr = int(m.jnt_qposadr[free[0]])
        self.obj_geom = np.asarray(m.geom_bodyid == self.obj_body)
        hg = np.asarray(self.hand_geom)
        if hg.dtype != bool:
            mask = np.zeros(m.ngeom, bool)
            mask[hg.astype(int)] = True
            hg = mask
        self.hand_geom = hg
        if not hg.any() or not self.obj_geom.any() or (hg & self.obj_geom).any():
            raise ValueError("hand / object geom masks are empty or overlap")
        self.ledger_geom = contact_ledger_geoms(m, hg, self.obj_geom)
        self.gravity = np.array(m.opt.gravity if self.gravity is None else self.gravity, dtype=np.float64)
        self.ctrl0 = np.zeros(m.nu) if self.ctrl0 is None else np.asarray(self.ctrl0, dtype=np.float64)
        self.weld_eq_ids = tuple(int(e) for e in self.weld_eq_ids)
        if any(not 0 <= e < m.neq for e in self.weld_eq_ids):
            raise ValueError(f"weld_eq_ids {list(self.weld_eq_ids)} outside [0, neq = {m.neq})")
        if np.asarray(self.q0).shape != (m.nq,) or self.ctrl0.shape != (m.nu,) or self.gravity.shape != (3,):
            raise ValueError(f"q0 / ctrl0 / gravity have shapes {np.asarray(self.q0).shape} / {self.ctrl0.shape} / "
                             f"{self.gravity.shape}, expected ({m.nq},) / ({m.nu},) / (3,)")
        if int(self.ctrl_steps) < 1:
            raise ValueError(f"ctrl_steps {self.ctrl_steps} must be >= 1")

    def _absent_model(self) -> Any:
        m = copy.deepcopy(self.model)
        for i in range(m.npair):
            g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
            if (self.hand_geom[g1] and self.obj_geom[g2]) or (self.obj_geom[g1] and self.hand_geom[g2]):
                m.pair_margin[i] = 0.0
                m.pair_gap[i] = 10.0
        if self.absent_mode == "displace":
            m.geom_sameframe[self.hand_geom] = 0      # else the geom frame is taken from its body, ignoring pos
            m.geom_pos[self.hand_geom, 2] += 1000.0
        elif self.absent_mode != "pairs":
            raise ValueError(f"absent_mode {self.absent_mode!r}: expected 'pairs' or 'displace'")
        return m

    def _object_forces(self, m: Any, d: Any, f6: np.ndarray) -> tuple[float, float]:
        """(hand-mask, other ledger) normal force on the object at the current step."""
        hand = other = 0.0
        for i in range(int(d.ncon)):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            o1, o2 = self.obj_geom[g1], self.obj_geom[g2]
            if o1 == o2:
                continue
            g = g2 if o1 else g1
            if not self.ledger_geom[g]:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if self.hand_geom[g]:
                hand += float(f6[0])
            else:
                other += float(f6[0])
        return hand, other

    def _invalid(self, d: Any) -> str | None:
        """Why the state after a step is not a valid run (module docstring), or None."""
        for w in _BAD:
            if d.warning[w].number > 0:
                return f"MuJoCo warning {w.name}"
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            return "non-finite state"
        v = float(np.abs(d.qvel).max()) if d.qvel.size else 0.0
        if v > self.qvel_max:
            return f"|qvel| {v:.3g} > {self.qvel_max:g}"
        return None

    def __call__(self, controls: np.ndarray, *, hand_object_contact: bool = True,
                 rows: int | None = None) -> Rollout:
        """Execute `controls` (T, nu) for at most `rows` rows (default T). The full post-step qpos and qvel rows, the
        applied controls (float32-rounded; zero on a row where MuJoCo refused a bad control) and the two ledger
        parts of the last run are kept in `last_qpos`, `last_qvel`, `last_ctrl`, `last_hand_force` and
        `last_other_force`; the rollout's force is their sum, and an invalid run's reason is in `last_invalid`. The
        model's gravity is restored afterwards."""
        m = self.model if hand_object_contact else self._absent_model()
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.q0
        d.qvel[:] = 0.0
        d.ctrl[:] = np.asarray(self.ctrl0, np.float64).astype(np.float32)
        T = int(controls.shape[0] if rows is None else rows)
        u32 = np.asarray(controls, np.float64)[:T].astype(np.float32).astype(np.float64)
        if u32.ndim != 2 or u32.shape[1] != m.nu:
            raise ValueError(f"controls have shape {np.asarray(controls).shape}, expected (T, nu = {m.nu})")
        n = u32.shape[0]
        qp = np.zeros((n, m.nq))
        qv = np.zeros((n, m.nv))
        ct = np.zeros((n, m.nu))
        hand_n = np.zeros(n)
        other_n = np.zeros(n)
        f6 = np.zeros(6)
        g_model = np.array(m.opt.gravity)
        z_adr = self.obj_qadr + 2
        end, k, why = "incomplete", 0, None
        try:
            for s in range(n):
                warm = s < self.warmup_rows
                m.opt.gravity[:] = 0.0 if warm else self.gravity
                for e in self.weld_eq_ids:
                    d.eq_active[e] = 1 if warm else 0
                d.ctrl[:] = u32[s]
                bad_ctrl = d.warning[mujoco.mjtWarning.mjWARN_BADCTRL].number
                mujoco.mj_step(m, d)
                qp[s] = d.qpos
                qv[s] = d.qvel
                # the applied control: MuJoCo replaces a NaN, inf or huge control vector by zeros (BADCTRL)
                ct[s] = 0.0 if d.warning[mujoco.mjtWarning.mjWARN_BADCTRL].number > bad_ctrl else d.ctrl
                hand_n[s], other_n[s] = self._object_forces(m, d, f6)
                k = s + 1
                why = self._invalid(d)
                if why is not None:
                    end = "invalid"
                    break
                if (s + 1) % self.ctrl_steps == 0 and d.qpos[z_adr] < self.terminate_z:
                    end = "drop"
                    break
            else:
                if n >= T:
                    end = "clip_end"
        finally:
            m.opt.gravity[:] = g_model
        self.last_qpos = qp[:k]
        self.last_qvel = qv[:k]
        self.last_ctrl = ct[:k]
        self.last_hand_force, self.last_other_force = hand_n[:k], other_n[:k]
        self.last_invalid = why
        obj = qp[:k, self.obj_qadr:self.obj_qadr + 7]
        return Rollout(object_traj=obj, hand_object_force=hand_n[:k] + other_n[:k], status=end)
