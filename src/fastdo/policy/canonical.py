"""Object-centric gravity-aligned canonical frame (policy design, Section 3).

The task is invariant under G_g = SE(2) x R (rotations about gravity, all translations). Expressing every
observation in

    F = (p, R),  R = [x, u x x, u],  u = -g/|g|,  x = normalize(P_perp(R_obj e_1)),

with P_perp the projection onto the plane orthogonal to u, makes the policy exactly invariant to that group:
for h in G_g, `canonical_frame(h T) = h canonical_frame(T)`. The columns are right-handed:
x x (u x x) = u (x.x) - x (x.u) = u for x orthogonal to u.

Degeneracies, in order: if the object's first axis is within 5 deg of vertical, x is the horizontal
wrist-to-object direction (object position minus wrist position); if that is within 5 deg of vertical too (or
the wrist sits on the object origin), the previous frame's x is kept (re-projected onto the horizontal plane).
Without a usable previous x (the first frame), the object's second axis is used: it is within 5 deg of
horizontal whenever the first axis is within 5 deg of vertical, so it is always well defined. Every fallback
is built from G_g-equivariant quantities, so the equivariance holds in all branches.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["DEGENERATE_ANGLE_DEG", "canonical_frame"]

DEGENERATE_ANGLE_DEG = 5.0
_EPS = 1e-12


def _horizontal(v: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """Unit horizontal direction of v (N,3) and a mask of where it is > 5 deg away from vertical."""
    norm = v.norm(dim=-1)
    proj = v - (v @ u)[:, None] * u
    proj_norm = proj.norm(dim=-1)
    ok = (norm > _EPS) & (proj_norm > math.sin(math.radians(DEGENERATE_ANGLE_DEG)) * norm)
    return proj / proj_norm.clamp_min(_EPS)[:, None], ok


def canonical_frame(obj_pose: Tensor, wrist_pos: Tensor, gravity: Tensor, prev_x: Tensor | None) -> Tensor:
    """Canonical frames (N,4,4): rotation columns `[x, u x x, u]` with `u = -gravity/|gravity|`, translation
    the object position. `obj_pose (N,4,4)`, `wrist_pos (N,3)`, `gravity (3,)`, `prev_x (N,3) | None` (the
    previous call's `F[:, :3, 0]`). See the module docstring for the degeneracy fallbacks."""
    if obj_pose.ndim != 3 or obj_pose.shape[1:] != (4, 4):
        raise ValueError(f"obj_pose must be (N,4,4), got {tuple(obj_pose.shape)}")
    n = int(obj_pose.shape[0])
    if wrist_pos.shape != (n, 3) or gravity.shape != (3,):
        raise ValueError(f"shapes: obj_pose {tuple(obj_pose.shape)} (N,4,4), wrist_pos "
                         f"{tuple(wrist_pos.shape)} (N,3), gravity {tuple(gravity.shape)} (3,)")
    if prev_x is not None and prev_x.shape != (n, 3):
        raise ValueError(f"prev_x must be (N,3), got {tuple(prev_x.shape)}")
    g = gravity.to(dtype=obj_pose.dtype, device=obj_pose.device)   # normalize in the pose dtype
    g_norm = g.norm()
    if float(g_norm) <= _EPS:
        raise ValueError("gravity must be non-zero")
    u = -g / g_norm
    p = obj_pose[:, :3, 3]

    x, ok = _horizontal(obj_pose[:, :3, 0], u)
    wrist_x, wrist_ok = _horizontal(p - wrist_pos.to(p), u)
    candidates = [(wrist_x, wrist_ok)]
    if prev_x is not None:
        candidates.append(_horizontal(prev_x.to(p), u))
    second_x, _ = _horizontal(obj_pose[:, :3, 1], u)           # always usable once the first axis degenerates
    for cand, cand_ok in candidates:
        x = torch.where(ok[:, None], x, cand)
        ok = ok | cand_ok
    x = torch.where(ok[:, None], x, second_x)

    uu = u.expand_as(x)
    F = torch.zeros_like(obj_pose)
    F[:, :3, 0] = x
    F[:, :3, 1] = torch.linalg.cross(uu, x)
    F[:, :3, 2] = uu
    F[:, :3, 3] = p
    F[:, 3, 3] = 1.0
    return F
