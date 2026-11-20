"""CPU MuJoCo backend: many independent worlds over many clips, with DO AS I DO's executor semantics.

Each world is one `mujoco.MjData` on its clip's model (worlds of a clip share the model; the per-row phase is set
before every step). The semantics are `fastdo.physics.fd_problem.rollout` with `terminate=True`, row by row:

- reset = the clip's start state (q0, qvel 0, ctrl0 as float32; `mj_resetData` first, so warnings, warm start and
  equality activations are the fresh-data defaults, exactly `fd_problem.start_data`);
- row s: weld on and gravity zero while s < W, else weld off and the clip's gravity; ctrl = the row's control
  rounded to float32 (the GPU executor's control dtype); `mj_step`;
- after the row: contact terms of the step's collision pass (`reward.contact_terms`); a non-finite state or a
  BADQACC/BADQPOS/BADQVEL warning ends the run ("invalid"); at a block end ((s+1) % ctrl_steps == 0) an object
  below `terminate_z` ends it ("drop") -- the executor's only termination rule (its penetration rule is inactive at
  the registered threshold 1e9, which `clip.build_from_ref_run` asserts).

Reference-state initialisation (`reset(i, row > 0)`, training only): the world starts at a reference row with the
object welded at its reference pose for a settle window (per-world weld targets are written into the shared model's
`eq_data` before every welded step; worlds that start at row 0 always use the executor's own target).

Training-only assist (`alpha > 0`, `fastdo.transductive.config.EnvConfig`): an object wrench toward the reference
from the release on; worlds at alpha = 0 run the executor exactly (their `xfrc_applied` is never written).

The worlds are independent, so a world's trajectory does not depend on which other worlds exist or on how they are
split across processes; MuJoCo on CPU is deterministic, so a replayed control sequence reproduces the trajectory bit
for bit (`get_state`/`set_state` snapshot a world exactly).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from fastdo.physics.fd_problem import _relpose, quat_sub
from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.reward import contact_terms

__all__ = ["STATE_SPEC", "CpuWorlds", "RowBlock"]


def _object_body(sp: ClipSpec) -> tuple[int, int, float, float]:
    """(body id, dof address, mass, largest principal inertia) of the clip's free object."""
    m = sp.ex.model
    j = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    b = int(m.jnt_bodyid[j])
    return b, int(m.jnt_dofadr[j]), float(m.body_mass[b]), float(np.max(m.body_inertia[b]))

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
_BAD = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL)


@dataclass(eq=False)
class RowBlock:
    """What one world did over a block of control rows (`n` executed, possibly fewer than requested)."""

    rows: np.ndarray       # (n,) control rows executed
    qpos: np.ndarray       # (n, nq) state after each row
    qvel: np.ndarray       # (n, nv)
    pen: np.ndarray        # (n,)
    ho: np.ndarray         # (n,) bool
    ped: np.ndarray        # (n,) bool
    ctrl: np.ndarray       # (n, nu) applied controls (float32 values)
    end: str | None        # None (still running), "drop", "invalid"


class CpuWorlds:
    """`len(world_clip)` worlds; world i runs clip `specs[world_clip[i]]` (module docstring)."""

    def __init__(self, specs: Sequence[ClipSpec], world_clip: Sequence[int], assist_wn: float = 10.0,
                 assist_track: bool = True) -> None:
        self.specs = list(specs)
        self.world_clip = np.asarray(world_clip, np.int64)
        self.data = [mujoco.MjData(self.specs[c].ex.model) for c in self.world_clip]
        self.row = np.zeros(len(self.data), np.int64)
        self.alpha = np.zeros(len(self.data))           # training-only assist level per world (0 = the executor)
        self.assist_wn = float(assist_wn)
        self.assist_track = bool(assist_track)
        self._obj = [_object_body(sp) for sp in self.specs]
        self._weld0 = [np.array(sp.ex.model.eq_data[list(sp.ex.weld_ids), 3:10]) for sp in self.specs]
        self.weld_until = np.zeros(len(self.data), np.int64)    # RSI settle: weld held through row weld_until - 1
        self.weld_rel = [w.copy() for w in (self._weld0[c] for c in self.world_clip)]
        for i in range(len(self.data)):
            self.reset(i)

    @property
    def n_worlds(self) -> int:
        return len(self.data)

    def spec(self, i: int) -> ClipSpec:
        return self.specs[int(self.world_clip[i])]

    def rebind(self, i: int, clip: int) -> None:
        """Bind world i to clip `clip` (an index into `specs`); call `reset` next."""
        if int(self.world_clip[i]) != int(clip):
            self.world_clip[i] = int(clip)
            self.data[i] = mujoco.MjData(self.specs[int(clip)].ex.model)

    def reset(self, i: int, row: int = 0, settle: int = 20) -> None:
        """Start world i at row 0 (the executor's start state) or, for reference-state initialisation, at `row` > 0:
        qpos = the reference row (object at the start pose before the release W), qvel = the reference velocities
        (object at rest before W), ctrl = ctrl_ref[row]; from the release on, the object is welded at its
        reference pose for `settle` rows (gravity on), so the hand can close on it before it is free."""
        sp = self.spec(i)
        m, d = sp.ex.model, self.data[i]
        mujoco.mj_resetData(m, d)
        c = int(self.world_clip[i])
        self.weld_until[i] = 0
        self.weld_rel[i] = self._weld0[c].copy()
        self.row[i] = int(row)
        if row == 0:
            d.qpos[:] = sp.ex.q0
            d.qvel[:] = 0.0
            d.ctrl[:] = np.asarray(sp.ex.ctrl0, np.float64).astype(np.float32)
            return
        a0, dof = sp.ex.obj_qadr, self._obj[c][1]
        d.qpos[:] = sp.rw.qpos_ref[row]
        d.qvel[:] = sp.rw.qvel_ref[row]
        if row < sp.W:
            d.qpos[a0:a0 + 7] = sp.ex.q0[a0:a0 + 7]
            d.qvel[dof:dof + 6] = 0.0
        d.ctrl[:] = np.asarray(sp.ctrl_ref[row], np.float64).astype(np.float32)
        if row >= sp.W and settle > 0:
            mujoco.mj_kinematics(m, d)
            self.weld_rel[i] = np.stack([np.concatenate(_relpose(m, d, e)) for e in sp.ex.weld_ids])
            self.weld_until[i] = row + settle

    def qpos(self, i: int) -> np.ndarray:
        return self.data[i].qpos

    def qvel(self, i: int) -> np.ndarray:
        return self.data[i].qvel

    def step_rows(self, i: int, ctrls: np.ndarray) -> RowBlock:
        """Apply control rows `ctrls` (k, nu) to world i from its current row; stops early on a termination."""
        sp = self.spec(i)
        ex, rw = sp.ex, sp.rw
        m, d = ex.model, self.data[i]
        k = int(ctrls.shape[0])
        nq, nv = m.nq, m.nv
        qp, qv = np.zeros((k, nq)), np.zeros((k, nv))
        pen, ho, ped = np.zeros(k), np.zeros(k, bool), np.zeros(k, bool)
        u32 = np.asarray(ctrls, np.float64).astype(np.float32).astype(np.float64)
        z_adr = ex.obj_qadr + 2
        s0 = int(self.row[i])
        end: str | None = None
        n = 0
        for j in range(k):
            s = s0 + j
            warm = s < ex.release_step
            welded = warm or s < self.weld_until[i]
            m.opt.gravity[:] = 0.0 if warm else ex.gravity
            rel = self.weld_rel[i]
            for k, e in enumerate(ex.weld_ids):
                d.eq_active[e] = 1 if welded else 0
                if welded:
                    m.eq_data[e, 3:10] = rel[k]
            d.ctrl[:] = u32[j]
            if self.alpha[i] > 0.0 and not warm:
                self._assist(i, sp, d, s)
            mujoco.mj_step(m, d)
            qp[j], qv[j] = d.qpos, d.qvel
            pen[j], ho[j], ped[j] = contact_terms(d, rw)
            n = j + 1
            if any(d.warning[w].number > 0 for w in _BAD) or not (np.isfinite(d.qpos).all()
                                                                   and np.isfinite(d.qvel).all()):
                end = "invalid"
                break
            if (s + 1) % sp.ctrl_steps == 0 and d.qpos[z_adr] < sp.terminate_z:
                end = "drop"
                break
        self.row[i] = s0 + n
        w0 = self._weld0[int(self.world_clip[i])]          # leave the shared model with the executor's own target
        for k, e in enumerate(ex.weld_ids):
            m.eq_data[e, 3:10] = w0[k]
        return RowBlock(rows=np.arange(s0, s0 + n), qpos=qp[:n], qvel=qv[:n], pen=pen[:n], ho=ho[:n],
                        ped=ped[:n], ctrl=u32[:n], end=end)

    def _assist(self, i: int, sp: ClipSpec, d: mujoco.MjData, s: int) -> None:
        """The assist wrench (`EnvConfig` docstring) for row s, written to the object's `xfrc_applied` (world frame,
        at the COM). Training only: worlds with alpha = 0 never call this, so their xfrc_applied stays zero."""
        body, dof, mass, inertia = self._obj[int(self.world_clip[i])]
        a0, wn, al = sp.ex.obj_qadr, self.assist_wn, float(self.alpha[i])
        ref, vref = sp.rw.qpos_ref[s + 1], sp.rw.qvel_ref[s + 1]
        q = d.qpos[a0 + 3:a0 + 7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, q)
        rot = rot.reshape(3, 3)
        e_rot = rot @ quat_sub(ref[a0 + 3:a0 + 7], q)
        w, w_ref = rot @ d.qvel[dof + 3:dof + 6], rot @ vref[dof + 3:dof + 6]
        f = al * mass * (-sp.ex.gravity + wn * wn * (ref[a0:a0 + 3] - d.qpos[a0:a0 + 3])
                         + 2.0 * wn * (vref[dof:dof + 3] - d.qvel[dof:dof + 3]))
        tau = al * inertia * (wn * wn * e_rot + 2.0 * wn * (w_ref - w))
        if not self.assist_track:                       # gravity compensation only
            f, tau = al * mass * -sp.ex.gravity, np.zeros(3)
        d.xfrc_applied[body, :3] = f
        d.xfrc_applied[body, 3:] = tau

    def get_state(self, i: int) -> np.ndarray:
        """Exact snapshot of world i: [clip, row, weld_until, weld targets (7 per weld), integration state]."""
        sp = self.spec(i)
        st = np.zeros(mujoco.mj_stateSize(sp.ex.model, STATE_SPEC))
        mujoco.mj_getState(sp.ex.model, self.data[i], st, STATE_SPEC)
        head = [float(self.world_clip[i]), float(self.row[i]), float(self.weld_until[i])]
        return np.concatenate([head, self.weld_rel[i].ravel(), st])

    def set_state(self, i: int, state: np.ndarray) -> None:
        self.rebind(i, int(state[0]))
        sp = self.spec(i)
        nw = len(sp.ex.weld_ids)
        mujoco.mj_resetData(sp.ex.model, self.data[i])
        mujoco.mj_setState(sp.ex.model, self.data[i], np.asarray(state[3 + 7 * nw:], np.float64), STATE_SPEC)
        self.row[i], self.weld_until[i] = int(state[1]), int(state[2])
        self.weld_rel[i] = np.asarray(state[3:3 + 7 * nw], np.float64).reshape(nw, 7)
