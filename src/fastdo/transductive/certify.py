"""Certification of a stored rollout: deterministic re-execution and the physics scorer's run-directory artifacts.

`reexecute` runs a control sequence from the clip's start state in a FRESH single world of the CPU backend (DO AS I
DO's executor semantics, `fastdo.transductive.sim_cpu`), in the executor's blocks of `ctrl_steps` rows, until the
controls end, the clip ends (T rows) or the executor terminates. CPU MuJoCo is deterministic: the same controls give
the same trajectory bit for bit, so the certificate IS the deliverable trajectory.

`write_run_dir` writes what `fastdo.retarget.isolation_artifacts.load_arm_artifacts` and the physics scorer
(unchanged) read from a DO AS I DO run directory: `trajectory_mjwp.npz` (executed qpos
/ qvel / ctrl per block; the MPC diagnostics `opt_steps`, `improvement`, `rew_max` as zeros: no MPC ran),
`isolation_errors.npz` (the per-frame and whole-trace object errors, from DO AS I DO's own
`compute_object_tracking_error` when `tracking_fn` is given), `arm_meta.json` (REF's run settings with this run's
executed steps) and `physics_record.json` (provenance; no GPU wall time). The scene, kinematic reference and config
are REF's own files, linked read-only by the caller; this function refuses to write
through a symlink. `fastdo.transductive.sim_warp.reexecute_warp` is the GPU backend's re-execution.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.reward import row_tracking
from fastdo.transductive.sim_cpu import CpuWorlds

__all__ = ["Execution", "reexecute", "write_run_dir"]


@dataclass(eq=False)
class Execution:
    """A re-executed rollout: executed rows' post-step states and applied controls, and how the run ended."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    end: str            # "clip_end", "drop", "invalid", "incomplete" (controls ran out before the clip's end)

    @property
    def rows(self) -> int:
        return int(self.qpos.shape[0])


def reexecute(spec: ClipSpec, ctrl: np.ndarray) -> Execution:
    """Execute `ctrl` (n <= T rows, nu) on a fresh world (module docstring)."""
    sim = CpuWorlds([spec], [0])
    ctrl = np.asarray(ctrl, np.float64)[:spec.T]
    qp, qv, uc = [], [], []
    end = "incomplete"
    s = 0
    while s < ctrl.shape[0]:
        blk = sim.step_rows(0, ctrl[s:s + spec.ctrl_steps])
        qp.append(blk.qpos)
        qv.append(blk.qvel)
        uc.append(blk.ctrl)
        s += blk.rows.shape[0]
        if blk.end is not None:
            end = blk.end
            break
    else:
        if s >= spec.T:
            end = "clip_end"
    m = spec.ex.model
    cat = lambda xs, w: np.concatenate(xs) if xs else np.zeros((0, w))
    return Execution(qpos=cat(qp, m.nq), qvel=cat(qv, m.nv), ctrl=cat(uc, m.nu), end=end)


def _blocks(x: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[0]
    return x.reshape(n // k, k, *x.shape[1:]) if n and n % k == 0 else x.reshape(1, n, *x.shape[1:])


def write_run_dir(pdir: Path, spec: ClipSpec, exe: Execution, ref_meta: dict[str, Any], provenance: dict[str, Any],
                  tracking_fn: Callable[[np.ndarray, np.ndarray], dict] | None = None) -> dict[str, Any]:
    """Write the scorer's run artifacts for `exe` into `pdir` (module docstring). Returns the whole-trace errors."""
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    for name in ("trajectory_mjwp.npz", "isolation_errors.npz", "arm_meta.json", "physics_record.json"):
        if (pdir / name).is_symlink():
            raise RuntimeError(f"refusing to write through a symlink: {pdir / name}")
    n, k = exe.rows, spec.ctrl_steps
    nb = max(n // k if n % k == 0 else 1, 1)
    np.savez(pdir / "trajectory_mjwp.npz", qpos=_blocks(exe.qpos, k), qvel=_blocks(exe.qvel, k),
             ctrl=_blocks(exe.ctrl, k), opt_steps=np.zeros((nb, 1)), improvement=np.zeros((nb, 1)),
             rew_max=np.zeros((nb, 1)), sim_step=np.arange(1, nb + 1) * (n // nb if nb else 0))
    if tracking_fn is not None:
        err = tracking_fn(exe.qpos, spec.rw.qpos_ref[:n].astype(np.float32))
        pos, rot = np.asarray(err["pos_err_frames"], np.float64), np.asarray(err["quat_err_frames"], np.float64)
        whole = (float(err["obj_pos_err"]), float(err["obj_quat_err"]))
    else:
        pos, rot = row_tracking(spec.rw, spec.ex.obj_qadr, np.arange(n), exe.qpos)
        whole = (float(pos.mean()), float(rot.mean()))
    np.savez(pdir / "isolation_errors.npz", obj_pos_err=np.float64(whole[0]), obj_quat_err=np.float64(whole[1]),
             pos_err_frames=pos, quat_err_frames=rot)
    meta = {**ref_meta, "executed_steps": n, "wall_seconds": None, "num_samples": "policy",
            "num_perturb_samples": None, "output_dir": str(pdir), "provenance": provenance}
    (pdir / "arm_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    phys = {"task": spec.task, "arm": "ours", "side": spec.side, "rc": 0, "attempts": [], "gpu_wall_s": None,
            "gpu_wall_s_all_attempts": None, "hold_wall_s_all_attempts": None, "optimize_s": None,
            "arm_meta": meta, "provenance": provenance}
    (pdir / "physics_record.json").write_text(json.dumps(phys, indent=1, default=float))
    return {"whole_pos_m": whole[0], "whole_rot_rad": whole[1], "rows": n, "end": exe.end}
