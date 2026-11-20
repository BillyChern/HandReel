"""One clip of the transductive corpus: DO AS I DO's executor problem as plain arrays, cached per clip.

A `ClipSpec` is everything the training environment and the certifier need for one clip, with DO AS I DO's
executor semantics as reproduced on CPU MuJoCo 3.4.0 (`fastdo.physics.fd_problem`):

- `ex`: the executor (`ExecutorSpec`: their `setup_mj_model` scene with the warmup weld's solver override and its
  target written at the placed start pose, start state q0/ctrl0, release row W, default gravity);
- `rw`: their registered per-step reward (`RewardSpec`: weights, velocity scale, penetration hinge, pedestal gates);
- `ctrl_ref`: their kinematic warm start (the control reference after the warmup interpolation) -- the base the
  residual policy adds to, and the T0 candidate when the residual is zero;
- the time base of their executor: blocks of `ctrl_steps` rows until the executed row count reaches
  `max_sim_steps` (`full_steps` = that count), termination checked at block ends;
- bookkeeping: task, side, corpus and REF's stored outcome / executed controls (REF = the reference MPC run).

`build_from_ref_run` turns a `RefRun` (retargeting env only) into a spec;
`save_clip`/`load_clip` cache it as `<task>.mjb.z` (zlib-compressed MuJoCo binary model, version-specific; a plain
`<task>.mjb` is read too) + `<task>.npz`, so worker processes load a clip in well under a second without importing
DO AS I DO.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from fastdo.physics.fd_problem import ExecutorSpec, RewardSpec

__all__ = ["ClipSpec", "build_from_ref_run", "full_steps", "load_clip", "ref_reward_spec", "save_clip"]

_RW_ARRAYS = ("qpos_ref", "qvel_ref", "w_dq", "hand_geom", "object_geom", "side_object_geom", "pedestal_geom",
              "gate_in_hand", "gate_near_ped")
_RW_SCALARS = ("vel_scale", "pen_scale", "pen_margin", "terminal_scale", "warmup_mult", "ped_scale")


def full_steps(max_sim_steps: int, ctrl_steps: int) -> int:
    """Rows the executor runs on a clip it does not terminate: whole blocks until the count reaches max_sim_steps."""
    return int(math.ceil(max_sim_steps / ctrl_steps) * ctrl_steps)


@dataclass(eq=False)
class ClipSpec:
    """One clip's executor problem (module docstring). Arrays are float64 unless noted."""

    task: str
    side: str
    corpus: str
    ex: ExecutorSpec
    rw: RewardSpec
    ctrl_ref: np.ndarray                    # (Tref, nu) warm-start control reference
    max_sim_steps: int
    ctrl_steps: int                         # executor block length (termination checked at block ends)
    terminate_z: float = -1.0
    ref_exec_ctrl: np.ndarray | None = None  # (T_ref_exec, nu) REF's executed controls (known answer)
    ref_outcome: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = self.T
        if self.rw.qpos_ref.shape[0] < T + 1 or self.ctrl_ref.shape[0] < T + 1:
            raise ValueError(f"{self.task}: reference shorter than the executed rows + 1 ({T + 1})")
        if self.ex.model.nu != self.ctrl_ref.shape[1]:
            raise ValueError(f"{self.task}: ctrl_ref width {self.ctrl_ref.shape[1]} != nu {self.ex.model.nu}")

    @property
    def T(self) -> int:
        """Executed rows of a non-terminated run (`full_steps`)."""
        return full_steps(self.max_sim_steps, self.ctrl_steps)

    @property
    def W(self) -> int:
        return self.ex.release_step

    @property
    def nu(self) -> int:
        return int(self.ex.model.nu)

    @property
    def n_robot(self) -> int:
        """Robot qpos dims (= robot dofs; the object free joint follows)."""
        return int(self.ex.obj_qadr)


def ref_reward_spec(run: Any, ex: Any) -> RewardSpec:
    """DO AS I DO's registered reward for a `RefRun` as a `fastdo.physics.fd_problem.RewardSpec`.

    Vendored verbatim from the study helper's `reward_spec`, which does not ship;
    `tests/run/test_transductive_reward_vendored.py` checks that the two agree field by field."""
    c, m = run.cfg, ex.model
    nv, nu = m.nv, m.nu
    w = np.ones(nv)
    w[:3] = c["base_pos_rew_scale"]
    w[3:6] = c["base_rot_rew_scale"]
    w[6:nu] = c["joint_rew_scale"]
    w[-6:-3] = c["pos_rew_scale"]
    w[-3:] = c["rot_rew_scale"]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    side = run.side
    for k in ("contact_rew_scale", "drop_penalty_scale", "floor_penalty_scale"):
        if float(c[k]) != 0.0:
            raise ValueError(f"{k} != 0 is not implemented")
    g = run.gates
    return RewardSpec(
        qpos_ref=run.qpos_ref, qvel_ref=run.qvel_ref, w_dq=w, vel_scale=float(c["vel_rew_scale"]),
        pen_scale=float(c["penetration_penalty_scale"]), pen_margin=float(c["penetration_margin"]),
        terminal_scale=float(c["terminal_rew_scale"]), warmup_mult=float(c["warmup_rew_multiplier"]),
        hand_geom=np.array(["collision_hand_" in n for n in names]),
        object_geom=np.array(["object" in n for n in names]),
        ped_scale=float(c["pedestal_penalty_scale"]) if g.get("has_gate") else 0.0,
        side_object_geom=np.array([n.startswith((f"{side}_object", f"{side}_support_")) for n in names]),
        pedestal_geom=np.array([n.startswith(("right_pedestal_", "left_pedestal_")) for n in names]),
        gate_in_hand=g.get("in_hand"), gate_near_ped=g.get("near_ped"))


def build_from_ref_run(run: Any, ref_outcome: dict[str, Any] | None = None) -> ClipSpec:
    """A `ClipSpec` from a `RefRun` (retargeting env; the reward is `ref_reward_spec`, vendored so this module
    imports no study code)."""
    from fastdo.physics.fd_problem import build_executor

    cfg = run.cfg
    if float(cfg["terminate_penetration_threshold"]) < 1e6:
        raise NotImplementedError("a finite terminate_penetration_threshold is not part of the registered executor")
    ctrl_steps = int(cfg["ctrl_steps"])
    T = full_steps(run.max_sim_steps, ctrl_steps)
    ex = build_executor(run.model, run.q0, run.ctrl0, run.warmup_steps, run.gravity, run.weld_ids, T)
    rw = ref_reward_spec(run, ex)
    return ClipSpec(task=run.task, side=run.side, corpus=run.corpus, ex=ex, rw=rw, ctrl_ref=run.ctrl_ref,
                    max_sim_steps=int(run.max_sim_steps), ctrl_steps=ctrl_steps,
                    terminate_z=float(cfg["terminate_z_threshold"]), ref_exec_ctrl=run.exec_ctrl,
                    ref_outcome=dict(ref_outcome or {}))


def save_clip(spec: ClipSpec, root: Path) -> Path:
    """Write `<root>/<task>.mjb.z` and `<task>.npz`; returns the npz path. Atomic per file (tmp + rename)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mjb = root / f"{spec.task}.mjb.z"
    raw = np.zeros(mujoco.mj_sizeModel(spec.ex.model), np.uint8)
    mujoco.mj_saveModel(spec.ex.model, None, raw)
    tmp = mjb.with_name(f".{mjb.name}.tmp")
    tmp.write_bytes(zlib.compress(raw.tobytes(), 1))
    tmp.replace(mjb)
    arrays: dict[str, np.ndarray] = {"q0": spec.ex.q0, "ctrl0": spec.ex.ctrl0, "gravity": spec.ex.gravity,
                                     "weld_ids": np.asarray(spec.ex.weld_ids, np.int64), "ctrl_ref": spec.ctrl_ref}
    for name in _RW_ARRAYS:
        val = getattr(spec.rw, name)
        if val is not None:
            arrays[f"rw_{name}"] = np.asarray(val)
    if spec.ref_exec_ctrl is not None:
        arrays["ref_exec_ctrl"] = spec.ref_exec_ctrl
    meta = {"task": spec.task, "side": spec.side, "corpus": spec.corpus, "release_step": spec.ex.release_step,
            "n_steps": spec.ex.n_steps, "obj_qadr": spec.ex.obj_qadr, "max_sim_steps": spec.max_sim_steps,
            "ctrl_steps": spec.ctrl_steps, "terminate_z": spec.terminate_z, "ref_outcome": spec.ref_outcome,
            **{f"rw_{k}": float(getattr(spec.rw, k)) for k in _RW_SCALARS}}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    npz = root / f"{spec.task}.npz"
    tmp = npz.with_name(f".{spec.task}.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(npz)
    return npz


def load_clip(root: Path, task: str) -> ClipSpec:
    """Inverse of `save_clip` (same MuJoCo version as the writer)."""
    root = Path(root)
    packed = root / f"{task}.mjb.z"
    if packed.is_file():
        model = mujoco.MjModel.from_binary_path("model.mjb", {"model.mjb": zlib.decompress(packed.read_bytes())})
    else:
        model = mujoco.MjModel.from_binary_path(str(root / f"{task}.mjb"))
    with np.load(root / f"{task}.npz") as z:
        a = {k: z[k] for k in z.files}
    meta = json.loads(a.pop("meta_json").tobytes().decode())
    ex = ExecutorSpec(model=model, q0=a["q0"], ctrl0=a["ctrl0"], release_step=int(meta["release_step"]),
                      gravity=a["gravity"], weld_ids=tuple(int(i) for i in a["weld_ids"]),
                      n_steps=int(meta["n_steps"]), obj_qadr=int(meta["obj_qadr"]))
    rw = RewardSpec(**{k: a.get(f"rw_{k}") for k in _RW_ARRAYS}, **{k: meta[f"rw_{k}"] for k in _RW_SCALARS})
    return ClipSpec(task=meta["task"], side=meta["side"], corpus=meta["corpus"], ex=ex, rw=rw,
                    ctrl_ref=a["ctrl_ref"], max_sim_steps=int(meta["max_sim_steps"]),
                    ctrl_steps=int(meta["ctrl_steps"]), terminate_z=float(meta["terminate_z"]),
                    ref_exec_ctrl=a.get("ref_exec_ctrl"), ref_outcome=dict(meta["ref_outcome"]))
