"""Approximate rotational symmetry GROUP of a template (design 5.1 as amended by R1.7), on top of
`fastdo.robot.symmetry.rotational_symmetries`.

Per principal axis (through c*, the surface-sample centroid): if every non-zero 15 deg step passes the mismatch test
the axis is continuous and represented by C_24; otherwise it gets the largest C_n (n in 2, 3, 4, 6, 8, 12, 24) whose
every multiple of 360/n deg passes. The per-axis generators are closed under products (breadth-first, at most 48
elements, rotations equal within 1 deg merged); every element is re-tested with the same mismatch statistic and the set
is accepted only if all pass and the closure did not overflow. Otherwise G falls back to the single axis with the
largest n (ties: larger extent). Applied tier: tol = min(3 mm, 0.05 x the largest principal extent); the 10 mm tier is
a diagnostic only.

An element g is a rotation R_g of the METRIC object frame about c*; it acts on a record pose as R <- R R_g,
a <- a + R (c* - R_g c*) (centre unchanged) and on the world rotation as O <- O R_g.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from fastdo.reference.config import ReferenceConfig
from fastdo.reference.so3 import angle_deg
from fastdo.robot.symmetry import rotational_symmetries

__all__ = ["SymmetryGroup", "apply_element", "compute_group", "trivial_group"]

_ORDERS = (24, 12, 8, 6, 4, 3, 2)


@dataclass(frozen=True)
class SymmetryGroup:
    elements: np.ndarray                  # (K,3,3), elements[0] = identity
    c_star: np.ndarray                    # (3,) metric centre (sample centroid)
    tol_m: float
    kind: str                             # "trivial" | "C<n>" | "C<n>-continuous" | "closure<K>" | fallback note
    mismatch_p95_m: np.ndarray = field(default_factory=lambda: np.zeros(1))   # (K,)
    mismatch_max_m: np.ndarray = field(default_factory=lambda: np.zeros(1))   # (K,)
    continuous: bool = False
    extents_m: tuple[float, ...] = ()
    axis_orders: tuple[int, ...] = ()

    @property
    def trivial(self) -> bool:
        return len(self.elements) == 1

    def to_dict(self) -> dict:
        return {"kind": self.kind, "order": len(self.elements), "tol_mm": round(self.tol_m * 1e3, 3),
                "continuous": self.continuous, "axis_orders": list(self.axis_orders),
                "extents_mm": [round(e * 1e3, 1) for e in self.extents_m],
                "mismatch_p95_mm": np.round(self.mismatch_p95_m * 1e3, 3).tolist(),
                "mismatch_max_mm": np.round(self.mismatch_max_m * 1e3, 3).tolist()}


def trivial_group(c_star: np.ndarray, tol_m: float = 0.0) -> SymmetryGroup:
    return SymmetryGroup(np.eye(3)[None], np.asarray(c_star, dtype=np.float64), tol_m, "trivial")


def apply_element(pose: np.ndarray, rot_g: np.ndarray, c_star: np.ndarray) -> np.ndarray:
    """Record pose `(4,4)` relabelled by g (module docstring); the centre R c* + a is unchanged."""
    out = np.array(pose, dtype=np.float64, copy=True)
    R = out[:3, :3]
    out[:3, 3] = out[:3, 3] + R @ (c_star - rot_g @ c_star)
    out[:3, :3] = R @ rot_g
    return out


def _samples(V: np.ndarray, F: np.ndarray, scale: float, cfg: ReferenceConfig) -> np.ndarray:
    mesh = trimesh.Trimesh(scale * np.asarray(V, dtype=np.float64), np.asarray(F), process=False)
    return np.asarray(trimesh.sample.sample_surface(mesh, cfg.sym_samples, seed=cfg.sym_seed)[0], dtype=np.float64)


def _mismatch(pts: np.ndarray, tree: cKDTree, centre: np.ndarray, rot: np.ndarray) -> tuple[float, float]:
    y = (pts - centre) @ rot.T + centre
    d = np.concatenate([tree.query(y)[0], cKDTree(y).query(pts)[0]])
    return float(np.percentile(d, 95)), float(d.max())


def _closure(gens: list[np.ndarray], cap: int) -> list[np.ndarray] | None:
    elems = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                c = a @ g
                if not any(angle_deg(c, e) < 1.0 for e in elems):
                    elems.append(c)
                    nxt.append(c)
                    if len(elems) > cap:
                        return None
        frontier = nxt
    return elems


def _axis_order(allowed: tuple[int, ...], step: int) -> tuple[int, bool]:
    nonzero = set(range(step, 360, step))
    if nonzero <= set(allowed):
        return 360 // step, True
    for n in _ORDERS:
        if (360 // n) % step:
            continue
        if all((k * 360 // n) in allowed for k in range(1, n)):
            return n, False
    return 1, False


def compute_group(V: np.ndarray, F: np.ndarray, scale: float, cfg: ReferenceConfig, tol_m: float | None = None
                  ) -> SymmetryGroup:
    """G of the metric template (module docstring). `tol_m=None` -> the applied tier."""
    axes = rotational_symmetries(V, F, scale, tol_m=1e9, step_deg=cfg.sym_step_deg, n_samples=cfg.sym_samples,
                                 seed=cfg.sym_seed)
    extents = tuple(float(a.extent_m) for a in axes)
    tol = min(cfg.sym_tol_abs_m, cfg.sym_tol_rel * max(extents)) if tol_m is None else float(tol_m)
    c_star = np.asarray(axes[0].centre, dtype=np.float64)
    orders, conts, gens = [], [], []
    for a in axes:
        allowed = (0,) + tuple(d for d, v in a.mismatch_m.items() if v <= tol)
        n, cont = _axis_order(allowed, cfg.sym_step_deg)
        orders.append(n)
        conts.append(cont)
        if n > 1:
            gens.append(Rotation.from_rotvec(np.radians(360.0 / n) * a.axis).as_matrix())
    if not gens:
        return SymmetryGroup(np.eye(3)[None], c_star, tol, "trivial", extents_m=extents, axis_orders=tuple(orders))
    pts = _samples(V, F, scale, cfg)
    tree = cKDTree(pts)
    elems = _closure(gens, cfg.sym_max_elements)
    kind = f"closure{len(elems)}" if elems is not None else ""
    if elems is not None:
        mm = [_mismatch(pts, tree, c_star, e) for e in elems[1:]]
        if not all(m[0] <= tol for m in mm):
            elems = None
    if elems is None:                               # fall back to the best single axis
        best = max(range(3), key=lambda k: (orders[k], extents[k]))
        n = orders[best]
        g = Rotation.from_rotvec(np.radians(360.0 / n) * axes[best].axis).as_matrix()
        elems = [np.linalg.matrix_power(g, k) for k in range(n)]
        mm = [_mismatch(pts, tree, c_star, e) for e in elems[1:]]
        keep = [0] + [k + 1 for k, m in enumerate(mm) if m[0] <= tol]
        if len(keep) != n:                          # the axis's own powers must all pass, else trivial
            return SymmetryGroup(np.eye(3)[None], c_star, tol, "trivial(fallback_failed)", extents_m=extents,
                                 axis_orders=tuple(orders))
        kind = f"C{n}" + ("-continuous" if conts[best] else "") + "(fallback)"
        conts = [conts[best]]
    p95 = np.array([0.0] + [m[0] for m in mm])
    mx = np.array([0.0] + [m[1] for m in mm])
    return SymmetryGroup(np.stack(elems), c_star, tol, kind, p95, mx, any(conts), extents, tuple(orders))
