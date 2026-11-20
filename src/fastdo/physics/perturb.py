# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""Sync-free replay of DO AS I DO's random object perturbation (`mjwp.apply_perturbation`).

Per rollout step DO AS I DO draws `rand_start`, `rand_continue` (K values each, tiled over the N_ctrl candidates:
world n*K+k gets seed k) and, only `if starting.any()`, four more tensors for the new force/torque. That branch is a
host sync per step, and it also decides how far the CUDA generator advances, so the draws cannot simply be moved.

Two facts make the whole schedule computable before the rollout, with the SAME generator consumption:
1. The perturbation state (active flag, stored force/torque) depends only on the random draws and on the reference's
   per-frame gate -- never on the simulated state. It is also identical across the N_ctrl candidates of one seed k
   (common random numbers), so it can be carried at (K, .) size and broadcast.
2. The branch needs only `rand_start < perturb_prob`, which the host can evaluate WITHOUT reading the device by
   emulating Philox4x32-10 (curand's engine, as torch's `uniform_` uses it) at the generator offset of that draw.

So `build_schedule` issues exactly DO AS I DO's draws, in its order (sample count, shapes, generator), decides each
branch on the host, and returns a (rows, K, 6) wrench table plus a per-step row index; the rollout then writes one
row per object per step. The emulated decisions are VERIFIED after every rollout against the drawn device values
(`Schedule.verify`, one readback); a mismatch raises. `decide="device"` reads the values back instead (exact, slow;
also what the CPU tests use).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = ["ObjPlan", "PerturbPlan", "Rng", "Schedule", "apply_row", "build_schedule", "philox4x32_10", "plan_from_env",
           "torch_uniform_at"]

_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF


def philox4x32_10(ctr: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, int, int, int]:
    """Random123 / curand Philox4x32 with 10 rounds (9 key bumps)."""
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(10):
        p0 = _M0 * c0
        p1 = _M1 * c2
        hi0, lo0 = p0 >> 32, p0 & _MASK
        hi1, lo1 = p1 >> 32, p1 & _MASK
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _MASK, lo1, (hi0 ^ c3 ^ k1) & _MASK, lo0
        if r < 9:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
    return c0, c1, c2, c3


def torch_uniform_at(seed: int, offset: int, n: int) -> np.ndarray:
    """The first n values torch's CUDA `uniform_` (e.g. `torch.rand(n, 1)`) returns at generator (seed, offset).

    Small tensors: element i comes from thread i: `curand_init(seed, subsequence=i, offset)` then the `.x` lane of
    `curand_uniform4` = Philox(ctr=(offset/4, i), key=seed), mapped to float32 by x*2^-32 + 2^-33 (exact in float32
    either fused or not, because x*2^-32 is a power-of-two scaling) and `rand == 1 ? 0 : rand`.
    """
    assert offset % 4 == 0, "torch advances the Philox offset in multiples of 4"
    q = offset // 4
    key = (seed & _MASK, (seed >> 32) & _MASK)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = philox4x32_10((q & _MASK, (q >> 32) & _MASK, i & _MASK, (i >> 32) & _MASK), key)[0]
        u = np.float32(np.float64(np.float32(x)) * 2.0 ** -32 + 2.0 ** -33)
        out[i] = np.float32(0.0) if u == np.float32(1.0) else u
    return out


@dataclass
class ObjPlan:
    """One object body as `apply_perturbation` iterates it (right then left)."""

    body_id: int
    has_mesh: bool
    gate: np.ndarray | None       # host copy of `_perturb_gate_mask[body_id]` (None/empty = ungated)
    mass: float                   # `model_cpu.body_mass[body_id]` (numpy float64 scalar, kept as is)


@dataclass
class PerturbPlan:
    """Per-clip constants of the perturbation (config values are read once, as `apply_perturbation` reads them)."""

    objs: list[ObjPlan]
    K: int
    n_ctrl: int
    enabled: bool                 # step_env's "any perturbation setting non-zero"
    random: bool                  # perturb_force_scale != 0 or perturb_torque_scale != 0
    force: float
    torque: float
    force_scale: float
    torque_scale: float
    prob: float
    cont_prob: float
    gravity_mag: float
    warmup_steps: int


@dataclass
class Schedule:
    """Wrench rows per object and, per rollout step, the row to write (or None: step_env did not perturb)."""

    rows: dict[int, torch.Tensor]                      # body_id -> (R, K, 6)
    step_row: list[int | None]                         # rollout t -> row index (None = untouched)
    drawn: list[torch.Tensor] = field(default_factory=list)     # rand_start / rand_continue tensors, in order
    predicted: list[np.ndarray] = field(default_factory=list)   # host values used for the decisions

    def verify(self) -> None:
        """Compare the host-decided draws with the device values (one readback); raise on any difference."""
        if not self.drawn:
            return
        dev = torch.cat([d.reshape(-1) for d in self.drawn]).cpu().numpy()
        host = np.concatenate([p.reshape(-1) for p in self.predicted])
        if dev.shape != host.shape or not np.array_equal(dev, host):
            bad = int(np.flatnonzero(dev != host)[0]) if dev.shape == host.shape else -1
            raise RuntimeError(f"perturbation RNG emulation diverged from the device draws (first index {bad})")


class Rng:
    """The default CUDA (or CPU) generator as `apply_perturbation` uses it, plus host decision of each draw."""

    def __init__(self, device: torch.device | str, decide: str = "emulate") -> None:
        self.device = torch.device(device)
        self.decide = decide
        if decide == "emulate" and self.device.type != "cuda":
            raise ValueError("decide='emulate' reproduces the CUDA Philox engine; use decide='device' on CPU")
        self._gen = None
        if self.device.type == "cuda":
            # the SAME generator `RngSlot` swaps and `torch.rand(device=...)` draws from (index None = current device)
            idx = torch.cuda.current_device() if self.device.index is None else self.device.index
            self.device = torch.device("cuda", idx)
            self._gen = torch.cuda.default_generators[idx]

    def rand_decided(self, k: int) -> tuple[torch.Tensor, np.ndarray]:
        """`torch.rand(k, 1, device=...)` and its values on the host (emulated, or read back)."""
        if self.decide == "emulate":
            seed, off = self._gen.initial_seed(), self._gen.get_offset()
            t = torch.rand(k, 1, device=self.device)
            return t, torch_uniform_at(seed, off, k).reshape(k, 1)
        t = torch.rand(k, 1, device=self.device)
        return t, t.cpu().numpy()

    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=self.device)

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=self.device)


def _new_wrench(plan: PerturbPlan, obj: ObjPlan, rng: Rng) -> tuple[torch.Tensor, torch.Tensor]:
    """The `if starting.any()` block of `apply_perturbation` at (K, .) size: same draws, same op order."""
    K = plan.K
    force_dir = rng.randn(K, 3)
    force_dir = force_dir / (force_dir.norm(dim=1, keepdim=True) + 1e-8)
    force_mag_scale = rng.rand(K, 1)
    new_force = force_dir * force_mag_scale * plan.force_scale * obj.mass * plan.gravity_mag
    torque_dir = rng.randn(K, 3)
    torque_dir = torque_dir / (torque_dir.norm(dim=1, keepdim=True) + 1e-8)
    torque_mag_scale = rng.rand(K, 1)
    new_torque = torque_dir * torque_mag_scale * plan.torque_scale
    return new_force, new_torque


def build_schedule(plan: PerturbPlan, rng: Rng, horizon: int, warmup_sim_step: int) -> Schedule:
    """Issue the rollout's perturbation draws in DO AS I DO's order and tabulate the per-step object wrenches.

    Mirrors `rollout` (abs_step = warmup_sim_step + t + 1, perturb iff abs_step >= warmup_steps) and
    `apply_perturbation` (module docstring). State starts at zero each rollout, as `rollout` zeroes it.
    """
    K, dev = plan.K, rng.device
    state = {o.body_id: {"active": np.zeros(K, bool), "src": np.full(K, -1)} for o in plan.objs}
    events: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {o.body_id: [] for o in plan.objs}
    # per object: list of row specs ("zero",) | ("const",) | ("rand", src (K,), active (K,))
    specs: dict[int, list[tuple]] = {o.body_id: [] for o in plan.objs}
    step_row: list[int | None] = []
    sched = Schedule(rows={}, step_row=step_row)
    for t in range(horizon):
        abs_step = warmup_sim_step + t + 1
        if not (plan.enabled and abs_step >= plan.warmup_steps):
            step_row.append(None)
            continue
        step_row.append(len(specs[plan.objs[0].body_id]) if plan.objs else None)
        for obj in plan.objs:
            st = state[obj.body_id]
            if not obj.has_mesh:
                specs[obj.body_id].append(("zero",))
                continue
            if obj.gate is not None and obj.gate.size > 0 and not bool(obj.gate[min(abs_step, obj.gate.shape[0] - 1)]):
                st["active"][:] = False
                specs[obj.body_id].append(("zero",))
                continue
            if not plan.random:
                specs[obj.body_id].append(("const",))
                continue
            rs_t, rs = rng.rand_decided(K)
            rc_t, rc = rng.rand_decided(K)
            sched.drawn += [rs_t, rc_t]
            sched.predicted += [rs, rc]
            starting = rs[:, 0] < np.float32(plan.prob)
            continuing = st["active"] & ~starting & (rc[:, 0] < np.float32(plan.cont_prob))
            if starting.any():
                events[obj.body_id].append(_new_wrench(plan, obj, rng))
                st["src"] = np.where(starting, len(events[obj.body_id]) - 1, st["src"])
            st["active"] = continuing | starting
            specs[obj.body_id].append(("rand", st["src"].copy(), st["active"].copy()))
    for obj in plan.objs:
        sched.rows[obj.body_id] = _tabulate(plan, specs[obj.body_id], events[obj.body_id], dev)
    return sched


def _tabulate(plan: PerturbPlan, specs: list[tuple], events: list[tuple[torch.Tensor, torch.Tensor]],
              dev: torch.device) -> torch.Tensor:
    """(R, K, 6) rows: zero rows are 0.0; others are (0.0 + force) + stored_force * active, as in the reference."""
    K, R = plan.K, len(specs)
    out = torch.zeros(R, K, 6, device=dev)
    if R == 0:
        return out
    nz = [i for i, s in enumerate(specs) if s[0] != "zero"]
    if not nz:
        return out
    f = torch.zeros(len(nz), K, 3, device=dev)
    tq = torch.zeros(len(nz), K, 3, device=dev)
    f += plan.force
    tq += plan.torque
    rand_rows = [j for j, i in enumerate(nz) if specs[i][0] == "rand"]
    if rand_rows:
        zero = torch.zeros(1, K, 3, device=dev)
        bank_f = torch.cat([zero] + [e[0].unsqueeze(0) for e in events], dim=0)     # (E+1, K, 3)
        bank_t = torch.cat([zero] + [e[1].unsqueeze(0) for e in events], dim=0)
        src = torch.as_tensor(np.stack([specs[nz[j]][1] for j in rand_rows]) + 1, device=dev)   # (Rr, K)
        act = torch.as_tensor(np.stack([specs[nz[j]][2] for j in rand_rows]), device=dev).unsqueeze(-1)
        kk = torch.arange(K, device=dev).unsqueeze(0).expand_as(src)
        sel = torch.as_tensor(rand_rows, device=dev)
        f[sel] = f[sel] + bank_f[src, kk] * act
        tq[sel] = tq[sel] + bank_t[src, kk] * act
    out[torch.as_tensor(nz, device=dev)] = torch.cat([f, tq], dim=-1)
    return out


def apply_row(xfrc: torch.Tensor, sched: Schedule, plan: PerturbPlan, t: int) -> None:
    """Write step t's wrench rows into `xfrc_applied` (N, nbody, 6); world n*K+k gets row[k]."""
    r = sched.step_row[t]
    if r is None:
        return
    x4 = xfrc.view(plan.n_ctrl, plan.K, xfrc.shape[1], 6)
    for obj in plan.objs:
        x4[:, :, obj.body_id, :] = sched.rows[obj.body_id][r]


def plan_from_env(config: Any, env: Any, body_has_mesh_geom: Callable[[Any, int], bool],
                  name2id: Callable[[str], int]) -> PerturbPlan:
    """Resolve the per-clip `PerturbPlan` once (DO AS I DO re-resolves names and meshes every step)."""
    objs = []
    for side in ("right", "left"):
        bid = name2id(f"{side}_object")
        if bid == -1:
            continue
        gate = env._perturb_gate_mask.get(bid)
        objs.append(ObjPlan(body_id=bid, has_mesh=bool(body_has_mesh_geom(env.model_cpu, bid)),
                            gate=None if gate is None else gate.detach().cpu().numpy().astype(bool),
                            mass=env.model_cpu.body_mass[bid]))
    K = int(config.num_perturb_samples)
    grav = abs(config.default_gravity[2]) if len(config.default_gravity) > 2 else 9.81
    return PerturbPlan(objs=objs, K=K, n_ctrl=env.num_worlds // K,
                       enabled=bool(config.perturb_force != 0 or config.perturb_torque != 0
                                    or config.perturb_force_scale != 0 or config.perturb_torque_scale != 0),
                       random=not (config.perturb_force_scale == 0 and config.perturb_torque_scale == 0),
                       force=config.perturb_force, torque=config.perturb_torque,
                       force_scale=config.perturb_force_scale, torque_scale=config.perturb_torque_scale,
                       prob=config.perturb_prob, cont_prob=config.perturb_continue_prob, gravity_mag=grav,
                       warmup_steps=int(config.warmup_steps))
