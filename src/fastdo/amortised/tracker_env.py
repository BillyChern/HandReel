"""Environment v2 on the CPU backend.

Worlds run DO AS I DO's executor exactly (`fastdo.transductive.sim_cpu.CpuWorlds`: start state, welded zero-gravity
warmup of W rows, release, block-end drop rule). Per policy step a world applies k control rows
u_r = ctrl_ref[r] + residual_r, the residual ramping linearly (first-order hold) from the previous target to the new
target delta = delta_max (.) tanh(a); a zero action reproduces the warm start. Reward, termination, observation and
the compressed record are `fastdo.amortised.tracker_obs`. Fingertips come from forward kinematics of each executed
row (a scratch `MjData` per clip; the worlds' own data are never touched), their SDF distance from the clip's object
grid (`fastdo.amortised.contact_ref.GridSDF`, body frame).

`step` returns the records of the PRE-step states it was given, so a training loop stores exactly
(`HostReplay.add(rec_pre, action, reward, term, trunc, final)`) and observes `observe(table, record)` -- the same
function the replay's rebuild calls. A world whose episode ended is reset at once (row 0, the executor's start
state, at its next assist level); its post-step record is handed back as `final` (the replay's final record).
Nothing here is a time-out: `trunc` is always 0.

Training loop (`fastdo.amortised.trainer`): an optional `reset_hook(i) -> (clip, alpha, start_row)` chooses each
new episode of world i at the moment it starts (the clip sampler, the assist schedule and reference-state
initialisation, RSI, at `start_row` > 0); without it every episode restarts world i's clip at row 0 with
`next_alpha[i]`. `StepOut.qpos_rows` / `ctrl_rows` are the rows each world executed in the step (an ended episode's
last rows included), which the success detector scores and the certificate reproduces. `get_state`/`set_state`
snapshot every world exactly (the simulator state of `CpuWorlds.get_state` plus this class's per-world arrays), for
resume.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mujoco
import numpy as np

from fastdo.amortised.contact_ref import GridSDF, field_on_rows
from fastdo.amortised.se3 import quat_conj, quat_rotate
from fastdo.amortised.tracker_obs import (
    RefTable,
    TermState,
    TrackerConfig,
    _group,
    record_fields,
    row_rewards,
    step_penalty,
)
from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.sim_cpu import CpuWorlds, _object_body

__all__ = ["KAPPA_UNSUPPORTED", "SHARPA_FINGERS", "StepOut", "TipSpec", "TrackerEnv", "clip_table_entry", "fk_hand",
           "kappa_rows", "sharpa_tip_spec", "tip_sdf"]

SHARPA_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
KAPPA_UNSUPPORTED = 0.25            # kappa on rows the reconstruction marks unsupported


@dataclass(frozen=True, eq=False)
class TipSpec:
    """Hand bodies of a clip's model: the wrist body and, per finger, the tip body and the tip's offset in it."""

    wrist_body: int
    tip_body: tuple[int, ...]
    tip_offset: np.ndarray     # (F, 3)


def sharpa_tip_spec(model: mujoco.MjModel, side: str) -> TipSpec:
    """The Sharpa hand's `TipSpec`: wrist = the body of the `<side>_palm` site (DO AS I DO's IK wrist target), tips =
    the `<side>_<finger>_tip` sites (the physics scorer's fingertip sites for the Sharpa hand) as (body,
    offset in that body) in `SHARPA_FINGERS` order. Raises if a site is missing."""
    def site(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise ValueError(f"the model has no site {name!r}")
        return i

    tips = [site(f"{side}_{f}_tip") for f in SHARPA_FINGERS]
    return TipSpec(wrist_body=int(model.site_bodyid[site(f"{side}_palm")]),
                   tip_body=tuple(int(model.site_bodyid[i]) for i in tips),
                   tip_offset=np.asarray(model.site_pos[tips], np.float64).copy())


def kappa_rows(frame_valid: np.ndarray, ref_dt: float, sim_dt: float, W: int, n_rows: int,
               low: float = KAPPA_UNSUPPORTED) -> np.ndarray:
    """kappa on the executor rows: 1 on supported reference frames, `low` on the frames the retarget
    stage marks invalid (the retarget metadata's `frame_valid`: `retarget_row_valid` AND the reference's own frame
    validity; all 1 on OakInk2 MoCap), sampled onto the rows exactly as the reference is (`field_on_rows`: frame 0
    held through the warmup, then linear between frames, so a row between a supported and an unsupported frame gets
    an intermediate value)."""
    f = np.where(np.asarray(frame_valid, bool), 1.0, float(low))
    return field_on_rows(f, ref_dt, sim_dt, W, n_rows)


def fk_hand(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, tip: TipSpec
            ) -> tuple[np.ndarray, np.ndarray]:
    """(wrist pose (n, 7) p + wxyz, fingertips (n, F, 3)) of qpos rows (n, nq) by forward kinematics on `data`."""
    qpos = np.atleast_2d(np.asarray(qpos, np.float64))
    wr = np.zeros((qpos.shape[0], 7))
    tips = np.zeros((qpos.shape[0], len(tip.tip_body), 3))
    b = np.asarray(tip.tip_body)
    for j, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        wr[j, :3], wr[j, 3:] = data.xpos[tip.wrist_body], data.xquat[tip.wrist_body]
        tips[j] = data.xpos[b] + np.einsum("fij,fj->fi", data.xmat[b].reshape(-1, 3, 3), tip.tip_offset)
    return wr, tips


def tip_sdf(sdf: GridSDF, tips: np.ndarray, obj_p: np.ndarray, obj_q: np.ndarray) -> np.ndarray:
    """(..., F, 4): SDF distance and unit gradient (WORLD frame) of fingertips (..., F, 3) at object poses (..., 3/4)."""
    qi = quat_conj(obj_q)[..., None, :]
    local = quat_rotate(qi, tips - obj_p[..., None, :])
    phi, g = sdf.query(local)
    g = g / np.maximum(np.sqrt((g * g).sum(-1, keepdims=True)), 1e-12)
    return np.concatenate([phi[..., None], quat_rotate(obj_q[..., None, :], g)], -1)


def clip_table_entry(spec: ClipSpec, tip: TipSpec, sdf: GridSDF, points: np.ndarray,
                     kappa: np.ndarray | None = None) -> dict:
    """One clip's `RefTable` entry from its `ClipSpec`: the robot/object reference rows of the executor problem
    (`rw.qpos_ref`, `rw.qvel_ref`, `ctrl_ref`), fingertips of the kinematic reference by FK, their SDF distance."""
    R = spec.n_robot
    m = spec.ex.model
    n = min(spec.rw.qpos_ref.shape[0], spec.ctrl_ref.shape[0])
    qref = np.asarray(spec.rw.qpos_ref[:n], np.float64)
    _, tips = fk_hand(m, mujoco.MjData(m), qref, tip)
    d = tip_sdf(sdf, tips, qref[:, R:R + 3], qref[:, R + 3:R + 7])[..., 0]
    body = _object_body(spec)[0]
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
    return {"obj_p": qref[:, R:R + 3], "obj_q": qref[:, R + 3:R + 7],
            "obj_v": np.asarray(spec.rw.qvel_ref[:n, R:R + 6], np.float64), "q_ref": qref[:, :R],
            "ctrl_ref": np.asarray(spec.ctrl_ref[:n, :R], np.float64), "tips_ref": tips, "dref": d,
            "kappa": np.ones(n) if kappa is None else np.asarray(kappa[:n], np.float64), "W": spec.W, "T": spec.T,
            "points": points, "mass": _object_body(spec)[2], "friction": float(m.geom_friction[geoms[0], 0])}


@dataclass(eq=False)
class StepOut:
    reward: np.ndarray        # (n,) float32
    term: np.ndarray          # (n,) float32
    trunc: np.ndarray         # (n,) float32 (always 0)
    end: list[str | None]     # per world: the end reason or None
    final: dict               # post-step records (meaningful for ended worlds)
    record: dict              # records of the worlds' current states (after auto-resets)
    rows: np.ndarray          # (n,) rows executed this step
    qpos_rows: list[np.ndarray] = field(default_factory=list)   # per world: (rows, nq) post-row states this step
    ctrl_rows: list[np.ndarray] = field(default_factory=list)   # per world: (rows, nu) applied controls (float32 values)


class TrackerEnv:
    """Worlds `world_clip[i]` over `specs` (module docstring). `tips[c]`, `sdfs[c]` belong to clip c; `table` was
    stacked from the same clips in the same order."""

    def __init__(self, cfg: TrackerConfig, specs: Sequence[ClipSpec], table: RefTable, tips: Sequence[TipSpec],
                 sdfs: Sequence[GridSDF], world_clip: Sequence[int], evaluation: bool = False,
                 reset_hook: Callable[[int], tuple[int, float, int]] | None = None) -> None:
        self.cfg, self.specs, self.table, self.tips, self.sdfs = cfg, list(specs), table, list(tips), list(sdfs)
        self.evaluation = bool(evaluation)
        self.reset_hook = reset_hook
        self.sim = CpuWorlds(self.specs, world_clip)
        self.n = self.sim.n_worlds
        self.R = table.R
        m0 = self.specs[0].ex.model
        self.fields = record_fields(m0.nq, m0.nv, self.R, table.F)
        self._scratch = [mujoco.MjData(sp.ex.model) for sp in self.specs]
        self._obj_geom = [np.isin(np.arange(sp.ex.model.ngeom),
                                  [g for g in range(sp.ex.model.ngeom)
                                   if sp.ex.model.geom_bodyid[g] == _object_body(sp)[0]]) for sp in self.specs]
        self.delta_prev = np.zeros((self.n, self.R))
        self.contact = np.zeros((self.n, 3))
        self.alpha = np.zeros(self.n)
        self.next_alpha = np.zeros(self.n)
        self.start_row = np.zeros(self.n, np.int64)
        self.term = TermState(self.n)
        self._ds = _group(self.R, cfg.delta_max)

    # ---------------------------------------------------------------------------------------------- records
    def _tip_force(self, i: int, c: int) -> np.ndarray:
        """Per-finger hand-object normal force (N) of world i's current contacts."""
        m, d = self.specs[c].ex.model, self.sim.data[i]
        out = np.zeros(len(self.tips[c].tip_body))
        f6 = np.zeros(6)
        for k in range(int(d.ncon)):
            g1, g2 = int(d.contact.geom1[k]), int(d.contact.geom2[k])
            other = g2 if self._obj_geom[c][g1] else (g1 if self._obj_geom[c][g2] else -1)
            if other < 0:
                continue
            b = int(m.geom_bodyid[other])
            if b in self.tips[c].tip_body:
                mujoco.mj_contactForce(m, d, k, f6)
                out[self.tips[c].tip_body.index(b)] += f6[0]
        return out

    def _record(self, i: int) -> dict:
        c = int(self.sim.world_clip[i])
        sp, d = self.specs[c], self.sim.data[i]
        R = self.R
        q = np.array(d.qpos, np.float64)
        wr, tips = fk_hand(sp.ex.model, self._scratch[c], q, self.tips[c])
        ts = tip_sdf(self.sdfs[c], tips[0], q[R:R + 3], q[R + 3:R + 7])
        raw = {"qpos": q, "qvel": np.array(d.qvel), "wrist": wr[0], "tips": tips[0], "tip_sdf": ts,
               "contact": self.contact[i], "prev_delta": self.delta_prev[i], "alpha": self.alpha[i], "clip": c,
               "row": int(self.sim.row[i]), "tip_force": self._tip_force(i, c), "obj_vel": np.array(d.qvel[R:R + 6])}
        return {k: np.asarray(raw[k]).astype(dt) for k, (_, dt) in self.fields.items()}

    def records(self) -> dict:
        rows = [self._record(i) for i in range(self.n)]
        return {k: np.stack([r[k] for r in rows]) for k in self.fields}

    # ---------------------------------------------------------------------------------------------- episodes
    def _reset(self, i: int) -> None:
        if self.reset_hook is not None:
            clip, alpha, row = self.reset_hook(i)
            self.sim.rebind(i, int(clip))
            self.next_alpha[i] = float(alpha)
        else:
            row = 0
        self.sim.reset(i, int(row))
        self.start_row[i] = int(row)
        self.delta_prev[i] = 0.0
        self.contact[i] = 0.0
        self.alpha[i] = self.next_alpha[i]
        self.sim.alpha[i] = 0.0 if self.evaluation else self.alpha[i]
        self.term.reset(i)

    def reset_all(self) -> dict:
        for i in range(self.n):
            self._reset(i)
        return self.records()

    def _execute(self, i: int, c: int, u: np.ndarray) -> tuple[object, np.ndarray, int, str | None]:
        """World i (clip c) executes control rows `u` from its current row: the executor block, the per-row reward
        terms, and the training termination (`TermState.rows`, then the executor's own end). Returns (block, row
        rewards, rows used, end reason or None). `step` and `execute_controls` share it."""
        cfg, R, tab, sp = self.cfg, self.R, self.table, self.specs[c]
        blk = self.sim.step_rows(i, u)
        n = blk.rows.shape[0]
        _, tips = fk_hand(sp.ex.model, self._scratch[c], blk.qpos, self.tips[c])
        d_tip = tip_sdf(self.sdfs[c], tips, blk.qpos[:, R:R + 3], blk.qpos[:, R + 3:R + 7])[..., 0]
        rew, rn, th = row_rewards(tab, cfg, c, blk.rows, blk.qpos, tips, d_tip, blk.pen)
        p_err = np.sqrt(((blk.qpos[:, R:R + 3] - tab.obj_p[tab.idx(np.full(n, c), blk.rows)]) ** 2).sum(-1))
        used, reason = self.term.rows(i, cfg, blk.rows, rn, th, p_err, sp.W, sp.T, self.evaluation)
        if reason is None and blk.end is not None:
            reason = blk.end
        return blk, rew, used, reason

    def execute_controls(self, i: int, u: np.ndarray) -> tuple[int, str | None]:
        """KNOWN-ANSWER path: world i executes the given
        FULL control rows `u` (k, nu) (e.g. a stored MPC run's EXACT controls, `ref_exec_ctrl`) from its current row
        through the same executor, row convention and training termination as `step` (`_execute`), with no residual
        and no reward. Returns (rows executed, end reason or None); an ended world is NOT reset, so the caller reads
        where and why the env ended the expert."""
        c = int(self.sim.world_clip[i])
        sp = self.specs[c]
        k = min(int(u.shape[0]), sp.T - int(self.sim.row[i]))
        _, _, used, reason = self._execute(i, c, np.asarray(u[:k], np.float64))
        return int(used), reason

    def step(self, actions: np.ndarray) -> StepOut:
        cfg, R = self.cfg, self.R
        reward, term = np.zeros(self.n, np.float32), np.zeros(self.n, np.float32)
        ends: list[str | None] = [None] * self.n
        finals: dict[int, dict] = {}
        nrows = np.zeros(self.n, np.int64)
        qrows: list[np.ndarray] = []
        crows: list[np.ndarray] = []
        for i in range(self.n):
            c = int(self.sim.world_clip[i])
            sp = self.specs[c]
            s0 = int(self.sim.row[i])
            k = min(cfg.k, sp.T - s0)
            target = self._ds * np.tanh(np.asarray(actions[i], np.float64))
            res = self.delta_prev[i][None] + (np.arange(1, k + 1) / cfg.k)[:, None] * (target - self.delta_prev[i])
            u = np.array(sp.ctrl_ref[s0:s0 + k], np.float64)
            u[:, :R] += res
            blk, rew, used, reason = self._execute(i, c, u)
            n = blk.rows.shape[0]
            reward[i] = float(rew[:used].sum() + step_penalty(cfg, target, self.delta_prev[i]))
            self.delta_prev[i] = target
            self.contact[i] = (blk.pen[n - 1], blk.ho[n - 1], blk.ped[n - 1]) if n else (0.0, 0.0, 0.0)
            nrows[i] = n
            qrows.append(np.asarray(blk.qpos[:used] if reason is not None else blk.qpos))
            crows.append(np.asarray(blk.ctrl[:used] if reason is not None else blk.ctrl))
            if reason is not None:
                finals[i] = self._record(i)
                term[i], ends[i] = 1.0, reason
                self._reset(i)
        final = {k: np.zeros((self.n, *shape), dt) for k, (shape, dt) in self.fields.items()}
        for i, r in finals.items():
            for k, a in final.items():
                a[i] = r[k]
        return StepOut(reward=reward, term=term, trunc=np.zeros(self.n, np.float32), end=ends, final=final,
                       record=self.records(), rows=nrows, qpos_rows=qrows, ctrl_rows=crows)

    # ---------------------------------------------------------------------------------------------- resume
    _ARRAYS = ("delta_prev", "contact", "alpha", "next_alpha", "start_row")

    def get_state(self) -> dict[str, np.ndarray]:
        """Exact per-world state (module docstring): simulator snapshots padded into one array, and the arrays."""
        sims = [self.sim.get_state(i) for i in range(self.n)]
        width = max(len(x) for x in sims)
        pad = np.full((self.n, width), np.nan)
        for i, x in enumerate(sims):
            pad[i, :len(x)] = x
        out = {k: np.array(getattr(self, k)) for k in self._ARRAYS}
        out.update(sim=pad, sim_len=np.array([len(x) for x in sims], np.int64), sim_alpha=np.array(self.sim.alpha),
                   env_count=np.array(self.term.env_count), pos_sum=np.array(self.term.pos_sum))
        return out

    def set_state(self, st: dict[str, np.ndarray]) -> None:
        for i in range(self.n):
            self.sim.set_state(i, np.asarray(st["sim"][i, :int(st["sim_len"][i])], np.float64))
        for k in self._ARRAYS:
            getattr(self, k)[:] = st[k]
        self.sim.alpha[:] = st["sim_alpha"]
        self.term.env_count[:] = st["env_count"]
        self.term.pos_sum[:] = st["pos_sum"]
