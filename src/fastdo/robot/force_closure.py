"""Force-closure measures of a hand-object contact set.

A contact i at point p_i with unit normal n_i (pointing from the hand INTO the object) and friction mu pushes on the
object with any force in its friction cone, linearised by the 4 pyramid edges e_ik = n_i +- mu t (as in
`fastdo.physics.grasp_preload.pyramid_edges`). Each edge gives the unit-force object wrench about the centre of mass c

    w_ik = [ e_ik ; (p_i - c) x e_ik / L ]                    (L: characteristic length, so torque is in force units)

Two measures, both descriptive (no threshold is set here):
- `epsilon_metric`: the Ferrari-Canny epsilon, the radius of the largest origin-centred ball inside conv{w_ik}. It is
  > 0 iff the set is in force closure (it can resist ANY wrench); 0 when the hull is not full-dimensional (rank < 6)
  or the origin is outside or on its boundary.
- `static_hold_residual`: whether nonnegative edge forces can supply one given wrench (the static hold against
  gravity): min ||W^T a - w_req|| over a >= 0 (NNLS), as a fraction of ||w_req||. A set can hold a static load
  without being in force closure (a palm cradle); it cannot pass epsilon > 0 without being able to hold every load.
- `signed_quality`: epsilon when in force closure, otherwise minus the
  distance from the origin to conv{w_ik} (the hull's min-norm point: min ||W^T l|| over l >= 0, sum l = 1, solved as
  an NNLS with the simplex row weighted by `SIMPLEX_WEIGHT`). It is continuous across the closure boundary, so an
  optimiser has a signal where epsilon is flat at 0.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import ConvexHull, QhullError

from fastdo.physics.grasp_preload import pyramid_edges

__all__ = ["contact_wrenches", "epsilon_metric", "hull_distance", "object_contacts", "signed_quality",
           "static_hold_residual"]

SIMPLEX_WEIGHT = 1e3


def contact_wrenches(points: np.ndarray, normals: np.ndarray, com: np.ndarray, mu: float, char_len: float) -> np.ndarray:
    """`(4n, 6)` unit-force object wrenches of the pyramid edges of every contact (module docstring)."""
    out = []
    for p, n in zip(np.asarray(points, float), np.asarray(normals, float), strict=True):
        for e in pyramid_edges(n, mu):
            e = e / np.linalg.norm(e)
            out.append(np.concatenate([e, np.cross(p - com, e) / char_len]))
    return np.asarray(out).reshape(-1, 6)


def epsilon_metric(wrenches: np.ndarray) -> float:
    """Ferrari-Canny epsilon of conv(wrenches) about the origin (module docstring); 0 when not in force closure."""
    w = np.asarray(wrenches, float)
    if len(w) < 7 or np.linalg.matrix_rank(w, tol=1e-9) < 6:
        return 0.0
    try:
        hull = ConvexHull(w)
    except QhullError:
        return 0.0
    # facets: a . x + b <= 0 inside, |a| = 1, so the origin's distance to a facet is -b (> 0 iff strictly inside)
    return max(0.0, float(np.min(-hull.equations[:, -1])))


def static_hold_residual(wrenches: np.ndarray, w_req: np.ndarray) -> float:
    """min over edge forces a >= 0 of ||wrenches^T a - w_req|| / ||w_req|| (0 = the set can supply w_req exactly)."""
    w = np.asarray(wrenches, float)
    target = np.asarray(w_req, float)
    if len(w) == 0:
        return 1.0
    _, res = nnls(w.T, target)
    return float(res / np.linalg.norm(target))


def hull_distance(wrenches: np.ndarray) -> float:
    """Distance from the origin to conv(wrenches) (0 if the origin is inside or on it); 1 for an empty set."""
    w = np.asarray(wrenches, float)
    if len(w) == 0:
        return 1.0
    a = np.vstack([w.T, SIMPLEX_WEIGHT * np.ones((1, len(w)))])
    b = np.concatenate([np.zeros(w.shape[1]), [SIMPLEX_WEIGHT]])
    lam, _ = nnls(a, b)
    lam = lam / max(float(lam.sum()), 1e-12)
    return float(np.linalg.norm(w.T @ lam))


def signed_quality(wrenches: np.ndarray) -> float:
    """epsilon in force closure, else minus the origin's distance to the wrench hull (module docstring)."""
    eps = epsilon_metric(wrenches)
    return eps if eps > 0.0 else -hull_distance(wrenches)


def object_contacts(model: mujoco.MjModel, data: mujoco.MjData, robot_geoms: np.ndarray, object_body: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points `(n,3)`, normals `(n,3)` from the robot into the object, robot geom ids `(n,)`) of the current
    robot-object contacts. MuJoCo's contact frame normal points from geom1 to geom2."""
    pts, nrm, geo = [], [], []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        n = np.asarray(c.frame[:3], float)
        if robot_geoms[g1] and b2 == object_body:
            pts.append(np.asarray(c.pos, float).copy()), nrm.append(n), geo.append(g1)
        elif robot_geoms[g2] and b1 == object_body:
            pts.append(np.asarray(c.pos, float).copy()), nrm.append(-n), geo.append(g2)
    return np.asarray(pts).reshape(-1, 3), np.asarray(nrm).reshape(-1, 3), np.asarray(geo, dtype=int)
