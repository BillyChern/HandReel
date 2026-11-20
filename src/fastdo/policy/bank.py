"""Per-world reference storage for the batched tracking environments.

Every world tracks its own (possibly perturbed) copy of a clip, stored padded to the longest clip seen:
`obj (N, Tmax, 4, 4)` object poses, `wrist (N, Tmax, 4, 4)` reference wrist poses (from the wrist block of
`q_ref`), `q (N, Tmax, nq_robot)` robot-only joint references, `plan (N, Tmax, L)` contact plans and `length
(N,)`; indices past a world's last frame are clamped to it. All live on the environment's device in float32
(the contact plan as bool). The object free-joint block of `PolicyReference.q_ref`, when present, is dropped
here: it is interpolated linearly by `resample_reference` (not renormalized) and `obj_pose` is the authority.

`packed` gathers the H + 1 frames `k..k+H` of every world into one flat `PolicyReference` with stride H + 1, so
`fastdo.policy.reference.future_window` and `fastdo.policy.rewards.step_reward`/`terminated`, which take one
reference and integer indices, evaluate per-world references unchanged: world i's frame k_i + j sits at flat
index i (H + 1) + j, and `base = i (H + 1)` addresses its current frame.

`rows`/`set_rows` copy worlds' stored references (the world-state copy of `fastdo.policy.env_state`).

`ClipRegistry` holds the clips the worlds draw from, by handle: each is validated once, when registered, and
dropped once no world tracks it and no installed reference set holds it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fastdo.policy.reference import PolicyReference
from fastdo.policy.scene import wrist_pose

__all__ = ["BankRows", "ClipRecord", "ClipRegistry", "PackedReference", "ReferenceBank"]


@dataclass(frozen=True)
class PackedReference:
    """Frames `k..k+H` of every world, flattened with stride H + 1 (module docstring)."""

    obj: PolicyReference      # obj_pose = object poses, q_ref = robot joints, contact_plan = plan
    wrist: PolicyReference    # obj_pose = reference wrist poses (same q_ref and plan)
    base: Tensor              # (N,) long, flat index of each world's frame k


@dataclass(frozen=True)
class BankRows:
    """Copies of M worlds' stored references (`ReferenceBank.rows`), padded to the bank's length at the time."""

    obj: Tensor               # (M, Tmax, 4, 4)
    wrist: Tensor             # (M, Tmax, 4, 4)
    q: Tensor                 # (M, Tmax, nq_robot)
    plan: Tensor              # (M, Tmax, L) bool
    length: Tensor            # (M,) long
    plan_fraction: Tensor     # (M, L)


class ReferenceBank:
    """Padded per-world references on `device` (module docstring)."""

    def __init__(self, n_worlds: int, nq_robot: int, n_labels: int, hz: float, device: torch.device) -> None:
        self.n_worlds, self.nq_robot, self.n_labels, self.hz, self.device = n_worlds, nq_robot, n_labels, hz, device
        self.length = torch.zeros(n_worlds, dtype=torch.long, device=device)
        self.obj = torch.zeros(n_worlds, 0, 4, 4, device=device)
        self.wrist = torch.zeros(n_worlds, 0, 4, 4, device=device)
        self.q = torch.zeros(n_worlds, 0, nq_robot, device=device)
        self.plan = torch.zeros(n_worlds, 0, n_labels, dtype=torch.bool, device=device)
        self.plan_fraction = torch.zeros(n_worlds, n_labels, device=device)

    @property
    def t_max(self) -> int:
        return int(self.obj.shape[1])

    def validate(self, ref: PolicyReference) -> None:
        """Raise ValueError if `ref` cannot be tracked by this bank's worlds."""
        T = ref.num_frames
        if abs(ref.hz - self.hz) > 1e-9 * self.hz:
            raise ValueError(f"reference hz {ref.hz} != environment hz {self.hz}; resample it first")
        if T < 2 or ref.obj_pose.shape != (T, 4, 4) or ref.contact_plan.shape != (T, self.n_labels):
            raise ValueError(f"reference needs T >= 2 frames with obj_pose (T,4,4) and contact_plan (T,{self.n_labels})"
                             f", got {tuple(ref.obj_pose.shape)} and {tuple(ref.contact_plan.shape)}")
        if ref.q_ref.ndim != 2 or ref.q_ref.shape[0] != T or ref.q_ref.shape[1] < self.nq_robot:
            raise ValueError(f"q_ref must be (T, >= {self.nq_robot}), got {tuple(ref.q_ref.shape)}")
        if not bool(torch.isfinite(ref.obj_pose).all()) or not bool(torch.isfinite(ref.q_ref[:, :self.nq_robot]).all()):
            raise ValueError("reference object poses and robot joints must be finite")

    def _grow(self, t_max: int) -> None:
        pad = t_max - self.t_max
        if pad <= 0:
            return
        n = self.n_worlds
        self.obj = torch.cat([self.obj, torch.eye(4, device=self.device).expand(n, pad, 4, 4)], dim=1)
        self.wrist = torch.cat([self.wrist, torch.eye(4, device=self.device).expand(n, pad, 4, 4)], dim=1)
        self.q = torch.cat([self.q, self.q.new_zeros(n, pad, self.nq_robot)], dim=1)
        self.plan = torch.cat([self.plan, self.plan.new_zeros(n, pad, self.n_labels)], dim=1)

    def write(self, world: int, ref: PolicyReference) -> None:
        """Store `ref` as world `world`'s reference, moved to the bank's device and dtype. `ref` must have passed
        `validate` (the environments validate a clip once, in `ClipRegistry.register`; its randomized draws only
        move the object poses)."""
        T = ref.num_frames
        self._grow(T)
        q = ref.q_ref[:, :self.nq_robot].to(device=self.device, dtype=torch.float32)
        plan = ref.contact_plan.to(device=self.device, dtype=torch.bool)
        self.obj[world, :T] = ref.obj_pose.to(device=self.device, dtype=torch.float32)
        self.wrist[world, :T] = wrist_pose(ref.q_ref[:, :6].to(device=self.device, dtype=torch.float64)).float()
        self.q[world, :T] = q
        self.plan[world, :T] = plan
        self.length[world] = T
        self.plan_fraction[world] = plan.float().mean(0)

    def rows(self, ids: Tensor) -> BankRows:
        """Copies of the stored references of worlds `ids` (M,) long."""
        return BankRows(obj=self.obj[ids], wrist=self.wrist[ids], q=self.q[ids], plan=self.plan[ids],
                        length=self.length[ids], plan_fraction=self.plan_fraction[ids])

    def set_rows(self, ids: Tensor, rows: BankRows) -> None:
        """Store `rows` as the references of the distinct worlds `ids`, growing the bank to the rows' length or padding
        the rows (identity poses, zero joints, no contact) to the bank's."""
        self._grow(int(rows.obj.shape[1]))
        m, pad = int(ids.numel()), self.t_max - int(rows.obj.shape[1])
        obj, wrist, q, plan = rows.obj, rows.wrist, rows.q, rows.plan
        if pad > 0:
            eye = torch.eye(4, device=self.device).expand(m, pad, 4, 4)
            obj, wrist = torch.cat([obj, eye], dim=1), torch.cat([wrist, eye], dim=1)
            q = torch.cat([q, q.new_zeros(m, pad, self.nq_robot)], dim=1)
            plan = torch.cat([plan, plan.new_zeros(m, pad, self.n_labels)], dim=1)
        self.obj[ids], self.wrist[ids], self.q[ids], self.plan[ids] = obj, wrist, q, plan
        self.length[ids], self.plan_fraction[ids] = rows.length, rows.plan_fraction

    def reference(self, world: int) -> PolicyReference:
        """World `world`'s stored reference (robot-only `q_ref`, on the bank's device; no covariance)."""
        T = int(self.length[world])
        return PolicyReference(q_ref=self.q[world, :T].clone(), obj_pose=self.obj[world, :T].clone(), log_scale=0.0,
                               contact_plan=self.plan[world, :T].clone(), covariance=None, hz=self.hz)

    def last_index(self) -> Tensor:
        return self.length - 1

    def packed(self, k: Tensor, horizon: int) -> PackedReference:
        """Frames `k..k+horizon` of every world (clamped to each world's last frame), flattened."""
        n, h1 = self.n_worlds, horizon + 1
        rows = torch.arange(n, device=self.device)[:, None].expand(n, h1)
        cols = torch.minimum(k[:, None] + torch.arange(h1, device=self.device), self.last_index()[:, None])
        q = self.q[rows, cols].reshape(n * h1, self.nq_robot)
        plan = self.plan[rows, cols].reshape(n * h1, self.n_labels)
        obj = PolicyReference(q_ref=q, obj_pose=self.obj[rows, cols].reshape(n * h1, 4, 4), log_scale=0.0,
                              contact_plan=plan, covariance=None, hz=self.hz)
        wrist = PolicyReference(q_ref=q, obj_pose=self.wrist[rows, cols].reshape(n * h1, 4, 4), log_scale=0.0,
                                contact_plan=plan, covariance=None, hz=self.hz)
        return PackedReference(obj=obj, wrist=wrist, base=torch.arange(n, device=self.device) * h1)


@dataclass
class ClipRecord:
    """A registered clip and data derived from it (e.g. the warm-up hand cloud), dropped together with it."""

    ref: PolicyReference
    cache: dict[str, Any] = field(default_factory=dict)


class ClipRegistry:
    """Clips by integer handle (module docstring); `register` looks a clip up by identity."""

    def __init__(self, bank: ReferenceBank) -> None:
        self._bank = bank
        self._records: dict[int, ClipRecord] = {}
        self._next = 0

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, handle: int) -> ClipRecord:
        return self._records[handle]

    def register(self, ref: PolicyReference) -> int:
        """Handle of `ref`, validating (`ReferenceBank.validate`) and adding it if it is not registered yet."""
        for handle, record in self._records.items():
            if record.ref is ref:
                return handle
        self._bank.validate(ref)
        handle, self._next = self._next, self._next + 1
        self._records[handle] = ClipRecord(ref)
        return handle

    def adopt(self, record: ClipRecord) -> int:
        """Handle of `record`'s clip: the registered handle if its reference is registered, otherwise `record` itself
        (validated when it was first registered, its cache included) under a new handle."""
        for handle, existing in self._records.items():
            if existing.ref is record.ref:
                return handle
        handle, self._next = self._next, self._next + 1
        self._records[handle] = record
        return handle

    def prune(self, keep: Iterable[int]) -> None:
        """Drop every clip whose handle is not in `keep`."""
        alive = set(keep)
        for handle in [h for h in self._records if h not in alive]:
            del self._records[handle]
