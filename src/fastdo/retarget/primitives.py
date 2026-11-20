"""Analytic minimum signed distance of the hand's collision primitives to the object SDF.

Why: C4 showed the baseline physics MPC drops the object when the kinematic reference puts fingers on
or inside the surface. Sampled surface points miss the parts of a primitive between samples, so the
penetration term measures each primitive as a whole.

Math. For an exact distance function phi (1-Lipschitz, |grad phi| = 1 a.e.), the minimum of phi over
the closed ball B(c, r) is phi(c) - r whenever phi(c) >= r (move from c by r along -grad phi), and
phi(c) - r is a lower bound otherwise. A capsule with axis a..b and radius r is the union of the balls
B(a + t (b - a), r), t in [0, 1], so

    d_min = min_t phi(a + t (b - a)) - r,

found by 16 uniformly spaced samples of t (batched trilinear lookups), a bracket around the argmin,
3 golden-section steps, and the best evaluated t kept as t* (optionally warm-started from the previous
LM iteration's t*, added as one more candidate). The search can only overestimate the minimum over t,
by second order in the final bracket (< 1e-4 canonical units for finger-sized capsules). The object's
grid SDF is trilinear, so it is only approximately a distance function: phi(axis) - r stays within
one grid cell of the grid's own minimum over a capsule that clears the surface, and is a (possibly
much deeper) lower bound for a penetrating one -- conservative exactly where the penalty is active.

Gradient (envelope theorem, t* held fixed): d d_min / dq = grad phi(p*) . ((1 - t*) dA/dq + t* dB/dq),
exact for body-fixed endpoints because a rigid body's point Jacobian is affine in the local point.

Covers. Every primitive is replaced by capsules (a sphere is a capsule with a == b) containing it, so
d_min over the covers never exceeds the primitive's true distance; the only effect is clearance
slightly larger than requested near the over-covered parts:
- capsule: itself (exact).
- cylinder (radius r, half-length h): the capsule of the same axis and radius. It bulges up to r past
  each flat end (at the end centre, tapering to 0 at the rim). The acceptance spec also lists a
  radius-r sphere at each flat-end centre; those spheres lie inside that capsule (and t = 0, 1 are
  sampled), so they never change d_min and are omitted.
- box (half-extents e1 <= e2 <= e3): capsules parallel to the longest axis, stacked along the middle
  axis in the mid-plane of the thinnest axis, n = ceil(e2 / sqrt((e1 + tol)^2 - e1^2)) of them, radius
  sqrt(e1^2 + (e2 / n)^2) <= e1 + tol. They bulge at most `tol` (default 0.5 mm) past the two largest
  faces, up to ~e1 past the four rim faces. This replaces the addendum's "4 edge capsules of radius
  min(e) + centre sphere", which does not contain a thin box: on the Sharpa palm (16 x 35 x 35 mm
  half-extents) most of each large face lies outside those covers (see
  `tests/retarget/test_primitives.py`).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from torch import Tensor

from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.transport import robot_canonical_points
from fastdo.solver.sdf import SdfGrid, sdf_lookup, sdf_values

__all__ = [
    "CapsuleCovers",
    "CapsuleDistance",
    "PhiFn",
    "ValuesFn",
    "as_phi_fn",
    "as_values_fn",
    "capsule_min_distance",
    "geom_capsule_covers",
    "hand_capsule_covers",
    "primitive_distances",
]

PhiFn = Callable[[Tensor], tuple[Tensor, Tensor]]
"""`y (T, N, 3) -> (phi (T, N), grad (T, N, 3))`, e.g. `functools.partial(sdf_lookup, sdf)`."""

ValuesFn = Callable[[Tensor], Tensor]
"""`y (T, N, 3) -> phi (T, N)`, e.g. `functools.partial(sdf_values, sdf)` (no gradient)."""

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
Cover = tuple[np.ndarray, np.ndarray, float]


def as_phi_fn(sdf: SdfGrid | PhiFn) -> PhiFn:
    """An `SdfGrid` becomes its trilinear lookup; a callable is used as-is (analytic test SDFs)."""
    if isinstance(sdf, SdfGrid):
        grid = sdf
        return lambda y: sdf_lookup(grid, y)
    return sdf


def as_values_fn(sdf: SdfGrid | PhiFn) -> ValuesFn | None:
    """An `SdfGrid` becomes its value-only lookup (bitwise the values of `sdf_lookup`); a callable gives None."""
    if isinstance(sdf, SdfGrid):
        return functools.partial(sdf_values, sdf)
    return None


@dataclass(frozen=True)
class CapsuleDistance:
    """Per frame and capsule: `d_min (T, P)`, the minimizing axis parameter `t_star (T, P)`, the axis
    point `point (T, P, 3)` and `grad (T, P, 3)` = grad phi at that point (canonical units)."""

    d_min: Tensor
    t_star: Tensor
    point: Tensor
    grad: Tensor


def _axis_phi(phi_fn: PhiFn, a: Tensor, b: Tensor, t: Tensor, values_fn: ValuesFn | None = None) -> Tensor:
    """phi at `a + t (b - a)` for `a, b (T, P, 3)` and `t (T, P, K)` -> `(T, P, K)`, one batched lookup."""
    T, P, K = t.shape
    pts = (a[..., None, :] + t[..., None] * (b - a)[..., None, :]).reshape(T, P * K, 3)
    phi = values_fn(pts) if values_fn is not None else phi_fn(pts)[0]
    return phi.reshape(T, P, K)


def capsule_min_distance(phi_fn: PhiFn, end_a: Tensor, end_b: Tensor, radius: Tensor,
                         t_init: Tensor | None = None, n_samples: int = 16, n_golden: int = 3,
                         values_fn: ValuesFn | None = None) -> CapsuleDistance:
    """`d_min = min_t phi(a + t (b - a)) - r` per capsule (module docstring), all in one unit system.

    Args:
        phi_fn: Signed distance and gradient, see `PhiFn`.
        end_a, end_b: `(T, P, 3)` axis endpoints (equal for a sphere).
        radius: `(P,)` capsule radii.
        t_init: `(T, P)` warm start (the previous iteration's `t_star`), added as a candidate.
        n_samples: Uniform samples of t in [0, 1], endpoints included.
        n_golden: Golden-section steps on the bracket around the best candidate.
        values_fn: Value-only lookup for the candidate scan and the golden-section steps; the final
            lookup at `p*` always calls `phi_fn`. Results are bitwise equal with or without it.
    """
    dtype = torch.float64
    a, b = end_a.to(dtype), end_b.to(dtype)
    T, P, _ = a.shape
    cand = torch.linspace(0.0, 1.0, n_samples, dtype=dtype).expand(T, P, n_samples)
    if t_init is not None:
        cand = torch.cat([cand, t_init.to(dtype).clamp(0.0, 1.0)[..., None]], dim=-1).sort(dim=-1).values
    f = _axis_phi(phi_fn, a, b, cand, values_fn)
    k = cand.shape[-1]
    i = f.argmin(dim=-1, keepdim=True)
    best_t, best_f = cand.gather(-1, i).squeeze(-1), f.gather(-1, i).squeeze(-1)
    lo = cand.gather(-1, (i - 1).clamp_min(0)).squeeze(-1)
    hi = cand.gather(-1, (i + 1).clamp_max(k - 1)).squeeze(-1)

    def _keep_best(t: Tensor, ft: Tensor) -> None:
        nonlocal best_t, best_f
        better = ft < best_f
        best_t, best_f = torch.where(better, t, best_t), torch.where(better, ft, best_f)

    if n_golden > 0:
        c, d = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
        fcd = _axis_phi(phi_fn, a, b, torch.stack([c, d], dim=-1), values_fn)
        fc, fd = fcd[..., 0], fcd[..., 1]
        _keep_best(c, fc)
        _keep_best(d, fd)
        for _ in range(n_golden):
            left = fc < fd                                     # minimum in [lo, d]: drop (d, hi]
            hi, lo = torch.where(left, d, hi), torch.where(left, lo, c)
            x = torch.where(left, hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo))
            fx = _axis_phi(phi_fn, a, b, x[..., None], values_fn)[..., 0]
            _keep_best(x, fx)
            c, d, fc, fd = (torch.where(left, x, d), torch.where(left, c, x),
                            torch.where(left, fx, fd), torch.where(left, fc, fx))

    point = a + best_t[..., None] * (b - a)
    phi, grad = phi_fn(point)
    return CapsuleDistance(d_min=phi - radius.to(dtype), t_star=best_t, point=point, grad=grad)


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def _box_stack(half: np.ndarray, tol: float) -> list[Cover]:
    """Box-local capsules containing the box (module docstring), before the geom pose is applied."""
    k1, k2, k3 = np.argsort(half, kind="stable")
    e1, e2, e3 = float(half[k1]), float(half[k2]), float(half[k3])
    n = max(1, math.ceil(e2 / math.sqrt((e1 + tol) ** 2 - e1 ** 2) - 1e-9))
    spacing = 2.0 * e2 / n
    radius = math.sqrt(e1 ** 2 + (spacing / 2.0) ** 2)
    covers: list[Cover] = []
    for j in range(n):
        a, b = np.zeros(3), np.zeros(3)
        a[k2] = b[k2] = -e2 + spacing * (j + 0.5)
        a[k3], b[k3] = -e3, e3
        covers.append((a, b, radius))
    return covers


def geom_capsule_covers(geom_type: int, size: np.ndarray, pos: np.ndarray, quat: np.ndarray,
                        box_face_tol: float = 5e-4) -> list[Cover]:
    """Capsule covers `(a, b, radius)` of one MuJoCo geom, in its parent body's frame.

    MuJoCo's parameterization: box `size` = half-extents; capsule/cylinder `size = (radius,
    half-length)` along local +z; sphere `size[0]` = radius; `pos`/`quat` (wxyz) place the geom.
    """
    size = np.asarray(size, dtype=np.float64)
    mat, pos = _quat_matrix(quat), np.asarray(pos, dtype=np.float64)
    if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        local = [(np.array([0.0, 0.0, -size[1]]), np.array([0.0, 0.0, size[1]]), float(size[0]))]
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        local = [(np.zeros(3), np.zeros(3), float(size[0]))]
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        local = _box_stack(size[:3], box_face_tol)
    else:
        raise NotImplementedError(f"no capsule cover for geom type {geom_type}; box/cylinder/capsule/sphere only")
    return [(pos + mat @ a, pos + mat @ b, r) for a, b, r in local]


@dataclass(frozen=True)
class CapsuleCovers:
    """All hand collision covers: body-local endpoints `local_a`/`local_b (P, 3)` on `body_ids (P,)`,
    `radius (P,)` in metres, and the collision geom `geom_ids (P,)` each cover belongs to."""

    body_ids: np.ndarray
    geom_ids: np.ndarray
    local_a: np.ndarray
    local_b: np.ndarray
    radius: np.ndarray

    @property
    def n(self) -> int:
        return int(self.radius.shape[0])


def hand_capsule_covers(scene: RobotScene, box_face_tol: float = 5e-4) -> CapsuleCovers:
    """Covers of every `RobotScene.hand_collision_geoms()` primitive (15 capsules and the 9-capsule
    palm-box stack, P = 24, on the Sharpa whisking scene; its arm cylinder has no contact pair)."""
    model = scene.model
    body_ids, geom_ids, ends_a, ends_b, radii = [], [], [], [], []
    for g in scene.hand_collision_geoms():
        for a, b, r in geom_capsule_covers(int(model.geom_type[g]), model.geom_size[g], model.geom_pos[g],
                                           model.geom_quat[g], box_face_tol):
            body_ids.append(int(model.geom_bodyid[g]))
            geom_ids.append(g)
            ends_a.append(a)
            ends_b.append(b)
            radii.append(r)
    return CapsuleCovers(body_ids=np.asarray(body_ids, dtype=np.int64), geom_ids=np.asarray(geom_ids, dtype=np.int64),
                         local_a=np.stack(ends_a), local_b=np.stack(ends_b), radius=np.asarray(radii))


def primitive_distances(q: np.ndarray | Tensor, scene: RobotScene, covers: CapsuleCovers,
                        sdf: SdfGrid | PhiFn, mesh_scale: float, n_samples: int = 16, n_golden: int = 3,
                        ) -> Tensor:
    """Per-frame, per-cover `d_min (T, P)` in metres for a qpos trajectory `(T, nq)`, the same function
    the penetration term optimizes (cold start), for the clearance/penetration acceptance statistics."""
    q_t = torch.as_tensor(q, dtype=torch.float64)
    ids = np.concatenate([covers.body_ids, covers.body_ids])
    world, _ = scene.point_kinematics(q_t.numpy(), ids, np.concatenate([covers.local_a, covers.local_b]),
                                      jacobian=False)
    world_t = torch.from_numpy(world)
    canon, _ = robot_canonical_points(q_t, world_t, torch.zeros(*world_t.shape, 1, dtype=torch.float64), mesh_scale)
    P = covers.n
    dist = capsule_min_distance(as_phi_fn(sdf), canon[:, :P], canon[:, P:],
                                torch.from_numpy(covers.radius) / mesh_scale, n_samples=n_samples, n_golden=n_golden,
                                values_fn=as_values_fn(sdf))
    return dist.d_min * mesh_scale
