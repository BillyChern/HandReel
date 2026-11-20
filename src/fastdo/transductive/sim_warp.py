"""GPU backend: one clip's worlds on DO AS I DO's own simulator, MuJoCo Warp 0.0.1 (`retargeting` env only).

`WarpPool` is a `fastdo.transductive.batch_group.Pool`: `mujoco_warp.put_model` / `put_data` of the clip's executor
model (the CPU MjModel of `ClipSpec.ex`: DO AS I DO's `setup_mj_model` scene, the warmup weld's solver override and
target), with n worlds, DO AS I DO's contact/constraint buffers (nconmax 120, njmax 500 per world). What differs
from DO AS I DO's executor (`optimize_physics`), which steps all its worlds in one phase, is that worlds here sit at
their own rows, so the phase is per world:
- gravity: `opt.gravity` is replaced by an (n,) array BEFORE the step graph is captured (mujoco_warp 0.0.1 indexes
  it `gravity[worldid % len]`), zero while the world's row < W, else the clip's gravity;
- the weld: `Data.eq_active[:, weld]` per world (row < W, or inside an RSI settle), and `Model.eq_data` as an
  (n, neq) array (batched before the capture, like gravity) holding each world's weld target;
- the training-only assist wrench (`EnvConfig`): `Data.xfrc_applied` of the object body, alpha > 0 worlds only.
Controls are written as float32 (the executor's dtype) and one captured `mujoco_warp.step` advances every world
(CUDA; warp runs on torch's current stream, which must not be the default stream, so the torch writes and the step are
ordered and the graph is capturable); on warp's CPU device the step is called directly (slow; tests and the known-answer probe). Contact terms come
from the flat contact buffer (`torch_terms.contact_terms_t`). A reset copies the start state into the masked worlds:
qpos q0, qvel / qacc / warm start / applied forces 0, ctrl ctrl0, time 0, weld on.

`reexecute_warp` replays one control sequence in a fresh 1-world pool (deterministic re-execution on the training
simulator) and returns the executed qpos rows, for certification by the unchanged scorer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from fastdo.transductive.clip import ClipSpec
from fastdo.transductive.torch_terms import PoolRef, contact_terms_t, quat_sub_t

__all__ = ["WarpPool", "reexecute_warp"]

NCONMAX, NJMAX = 120, 500          # DO AS I DO's nconmax_per_env / njmax_per_env (registered config)


def _quat2mat(q: Tensor) -> Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


class WarpPool:
    """n worlds of `spec` on mujoco_warp (module docstring)."""

    def __init__(self, spec: ClipSpec, n: int, device: str = "cuda:0", assist_wn: float = 10.0,
                 nconmax: int = NCONMAX, njmax: int = NJMAX, capture: bool | None = None,
                 assist: bool = False, settle: int = 20, assist_track: bool = True) -> None:
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        wp.init()                                   # idempotent; stream_from_torch needs the runtime
        self.spec, self.n, self.device, self.wn = spec, int(n), device, float(assist_wn)
        self.assist = bool(assist)          # compute the assist wrench every row (no host sync); off: never
        self.assist_track = bool(assist_track)
        self._wp, self._mjw = wp, mjw
        m = spec.ex.model
        d0 = mujoco.MjData(m)
        d0.qpos[:] = spec.ex.q0
        d0.ctrl[:] = np.asarray(spec.ex.ctrl0, np.float64).astype(np.float32)
        for e in spec.ex.weld_ids:
            d0.eq_active[e] = 1
        cuda = str(device).startswith("cuda")
        self.wps = None
        if cuda:                                   # warp runs on torch's (non-default) current stream: ordered with
            ts = torch.cuda.current_stream(device)  # the torch writes that feed every step, and capturable
            if ts.cuda_stream == 0:
                raise RuntimeError("WarpPool needs a non-default torch stream (torch.cuda.set_stream(Stream()))")
            self.wps = wp.stream_from_torch(ts)
        with wp.ScopedDevice(device), self._scope():
            self.mw = mjw.put_model(m)
            self.mw.opt.gravity = wp.array(np.tile(np.zeros(3, np.float32), (self.n, 1)), dtype=wp.vec3)
            # per-world equality data (weld targets; RSI settles), batched like gravity before any capture
            eq0 = self.mw.eq_data.numpy()
            self.mw.eq_data = wp.array(np.tile(eq0[:1], (self.n, 1, 1)), dtype=self.mw.eq_data.dtype)
            self.dw = mjw.put_data(m, d0, nworld=self.n, nconmax=nconmax, njmax=njmax)
        t = wp.to_torch
        self.qpos_t, self.qvel_t, self.ctrl_t = t(self.dw.qpos), t(self.dw.qvel), t(self.dw.ctrl)
        self.qacc_t, self.warm_t, self.time_t = t(self.dw.qacc), t(self.dw.qacc_warmstart), t(self.dw.time)
        self.act_t, self.qfrc_t = t(self.dw.act), t(self.dw.qfrc_applied)
        self.xfrc_t, self.eq_t, self.grav_t = t(self.dw.xfrc_applied), t(self.dw.eq_active), t(self.mw.opt.gravity)
        c = self.dw.contact
        self.cdist, self.cgeom, self.cwid, self.nacon = t(c.dist), t(c.geom), t(c.worldid), t(self.dw.nacon)
        tdev = self.qpos_t.device
        self.q0 = torch.as_tensor(spec.ex.q0, dtype=torch.float32, device=tdev)
        self.c0 = torch.as_tensor(np.asarray(spec.ex.ctrl0, np.float64).astype(np.float32), device=tdev)
        self.g = torch.as_tensor(spec.ex.gravity, dtype=torch.float32, device=tdev)
        self.weld = list(spec.ex.weld_ids)
        j = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        self.body, self.dof = int(m.jnt_bodyid[j]), int(m.jnt_dofadr[j])
        self.mass, self.inertia = float(m.body_mass[self.body]), float(np.max(m.body_inertia[self.body]))
        self.ref = PoolRef.build(spec, tdev, torch.float32)
        self.settle = int(settle)
        self.eqd_t = t(self.mw.eq_data)                               # (n, neq, 11)
        self.weld0 = self.eqd_t[0, self.weld, 3:10].clone()           # the executor's own targets (nweld, 7)
        self.weld_rel = self.weld0[None].repeat(self.n, 1, 1)
        self.weld_until = torch.zeros(self.n, dtype=torch.long, device=tdev)
        self.graph = None
        if capture is None:
            capture = cuda
        if capture:
            self._step()                                   # module loading must happen before the capture
            self._step()
            wp.synchronize()
            with wp.ScopedDevice(device), self._scope(), wp.ScopedCapture(device=device, stream=self.wps) as cap:
                mjw.step(self.mw, self.dw)
            self.graph = cap.graph
        self.reset(torch.ones(self.n, dtype=torch.bool, device=tdev))

    def _scope(self):
        import contextlib

        return self._wp.ScopedStream(self.wps) if self.wps is not None else contextlib.nullcontext()

    def _step(self) -> None:
        with self._wp.ScopedDevice(self.device), self._scope():
            if self.graph is not None:
                self._wp.capture_launch(self.graph)
            else:
                self._mjw.step(self.mw, self.dw)

    def reset(self, mask: Tensor, row: Tensor | None = None) -> None:
        """Masked worlds to the executor's start (row 0) or, where `row` > 0, to the reference state of that row
        (`sim_cpu.CpuWorlds.reset`: object welded at its reference pose for `settle` rows after the release; the
        weld target is computed on the host with CPU MuJoCo kinematics, once per reset)."""
        m = mask.to(self.qpos_t.device)
        self.qpos_t[m] = self.q0
        self.ctrl_t[m] = self.c0
        for x in (self.qvel_t, self.qacc_t, self.warm_t, self.act_t, self.qfrc_t, self.xfrc_t):
            x[m] = 0.0
        self.time_t[m] = 0.0
        self.weld_until[m] = 0
        self.weld_rel[m] = self.weld0
        for e in self.weld:
            self.eq_t[m, e] = True
        if row is None or not bool((row[mask] > 0).any()):
            return
        import mujoco

        from fastdo.physics.fd_problem import _relpose

        sp, a0 = self.spec, self.spec.ex.obj_qadr
        md, dd = sp.ex.model, None
        for w in torch.nonzero(mask & (row > 0)).flatten().tolist():
            r = int(row[w])
            q = sp.rw.qpos_ref[r].copy()
            v = sp.rw.qvel_ref[r].copy()
            if r < sp.W:
                q[a0:a0 + 7] = sp.ex.q0[a0:a0 + 7]
                v[self.dof:self.dof + 6] = 0.0
            self.qpos_t[w] = torch.as_tensor(q, dtype=torch.float32)
            self.qvel_t[w] = torch.as_tensor(v, dtype=torch.float32)
            self.ctrl_t[w] = torch.as_tensor(np.asarray(sp.ctrl_ref[r], np.float32))
            if r >= sp.W and self.settle > 0:
                dd = dd or mujoco.MjData(md)
                dd.qpos[:] = q
                mujoco.mj_kinematics(md, dd)
                rel = np.stack([np.concatenate(_relpose(md, dd, e)) for e in self.weld])
                self.weld_rel[w] = torch.as_tensor(rel, dtype=torch.float32)
                self.weld_until[w] = r + self.settle

    def state(self) -> tuple[Tensor, Tensor]:
        return self.qpos_t, self.qvel_t

    def _assist(self, rows: Tensor, alpha: Tensor, on: Tensor) -> None:
        a0, dof, wn = self.spec.ex.obj_qadr, self.dof, self.wn
        ref, vref = self.ref.qpos_ref[rows + 1], self.ref.qvel_ref[rows + 1]
        q = self.qpos_t[:, a0 + 3:a0 + 7]
        rot = _quat2mat(q)
        e = (rot @ quat_sub_t(ref[:, a0 + 3:a0 + 7], q)[..., None])[..., 0]
        w = (rot @ self.qvel_t[:, dof + 3:dof + 6, None])[..., 0]
        w_ref = (rot @ vref[:, dof + 3:dof + 6, None])[..., 0]
        al = (alpha * on.to(alpha.dtype))[:, None]
        f = al * self.mass * (-self.g + wn * wn * (ref[:, a0:a0 + 3] - self.qpos_t[:, a0:a0 + 3])
                              + 2.0 * wn * (vref[:, dof:dof + 3] - self.qvel_t[:, dof:dof + 3]))
        tau = al * self.inertia * (wn * wn * e + 2.0 * wn * (w_ref - w))
        if not self.assist_track:                       # gravity compensation only
            f, tau = al * self.mass * -self.g, torch.zeros_like(tau)
        self.xfrc_t[:, self.body, :3] = f
        self.xfrc_t[:, self.body, 3:] = tau

    def step_row(self, ctrl: Tensor, rows: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        warm = rows < self.spec.ex.release_step
        welded = warm | (rows < self.weld_until)
        self.grav_t[:] = torch.where(warm[:, None], torch.zeros_like(self.g), self.g)
        settling = (welded & ~warm)[:, None]
        for k, e in enumerate(self.weld):
            self.eq_t[:, e] = welded
            self.eqd_t[:, e, 3:10] = torch.where(settling, self.weld_rel[:, k], self.weld0[k])
        self.ctrl_t[:] = ctrl.to(torch.float32)
        if self.assist:
            self._assist(rows, alpha.to(torch.float32), (alpha > 0) & ~warm)
        self._step()
        pen, ho, ped = contact_terms_t(self.ref, self.cdist, self.cgeom, self.cwid, self.nacon, self.n)
        # non-finite, or beyond MuJoCo's mjMAXVAL (1e10; the CPU executor's BADQPOS/BADQVEL warnings)
        bad = ~((self.qpos_t.abs() <= 1e10).all(1) & (self.qvel_t.abs() <= 1e10).all(1))
        return pen, ho, ped, bad


def reexecute_warp(spec: ClipSpec, ctrl: np.ndarray, device: str = "cuda:0") -> tuple[np.ndarray, str]:
    """Replay `ctrl` (n <= T rows) in a fresh single-world pool with the executor's rules; returns (qpos rows, end)."""
    if str(device).startswith("cuda") and torch.cuda.current_stream(device).cuda_stream == 0:
        with torch.cuda.stream(torch.cuda.Stream(torch.device(device))):
            return reexecute_warp(spec, ctrl, device)
    pool = WarpPool(spec, 1, device=device)
    dev = pool.qpos_t.device
    u = torch.as_tensor(np.asarray(ctrl, np.float32)[:spec.T], device=dev)
    zero = torch.zeros(1, device=dev)
    out, end = [], "incomplete"
    a0 = spec.ex.obj_qadr
    for s in range(u.shape[0]):
        _, _, _, bad = pool.step_row(u[s:s + 1], torch.tensor([s], device=dev), zero)
        out.append(pool.qpos_t[0].double().cpu().numpy())
        if bool(bad[0]):
            end = "invalid"
            break
        if (s + 1) % spec.ctrl_steps == 0 and float(pool.qpos_t[0, a0 + 2]) < spec.terminate_z:
            end = "drop"
            break
    else:
        if u.shape[0] >= spec.T:
            end = "clip_end"
    return np.stack(out), end
