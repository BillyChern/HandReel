"""World sources, the record <-> world map and the covariance transport of the reference conditioner.

Record convention (pre-ego, as `solve.npz` / `contact.npz` / `ClipRecord.object_pose` store it): a pose `(R, a)` maps
METRIC template points y (mesh vertices x scale) into the camera frame, x = R y + a. A world source gives per row
`(Q_t, u_t, sigma_t)`; `world_frame.apply_ego_motion` multiplies the object translation (and the hand points) by
sigma_t and then applies the rigid `[Q_t | u_t]`, so the world centre of the template point c* is
p_t = Q_t (R_t c* + sigma_t a_t) + u_t and the world rotation is O_t = Q_t R_t. The map back is exact.

Covariance: the solver's left perturbation xi = [rho; theta] of (R, a) in the camera frame moves p by
Q [sigma rho + theta x (R c* + sigma a)] and O by the world-left rotation Q theta, so
Sigma^w = B J Sigma^cam J^T B^T with J = [[sigma I, -hat(R c* + sigma a)], [0, I]] and B = blockdiag(Q, Q).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastdo.reference.so3 import hat

__all__ = ["WorldSource", "declared_static", "no_world", "round_trip", "sanitize_covariance", "scaled_points",
           "to_record", "to_world", "transport_covariance", "world_from_ego"]

KINDS = ("ego_motion", "declared_static", "none")


@dataclass(frozen=True)
class WorldSource:
    kind: str                     # "ego_motion" | "declared_static" | "none"
    Q: np.ndarray                 # (T,3,3) rigid rotation camera -> world
    u: np.ndarray                 # (T,3)
    sigma: np.ndarray             # (T,) per-row scale applied to camera-frame translations / hand points
    reason: str = ""              # why `none` (or the ego file)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"world source kind {self.kind!r}")

    @property
    def can_consolidate(self) -> bool:
        return self.kind in ("ego_motion", "declared_static")


def declared_static(n: int) -> WorldSource:
    """Camera fixed by DATASET metadata (DexYCB rig serial, OakInk2 MoCap world): identity."""
    return WorldSource("declared_static", np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), np.ones(n))


def no_world(n: int, reason: str) -> WorldSource:
    """No certified world: the camera frame is used for flip steps only; no consolidation."""
    return WorldSource("none", np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), np.ones(n), reason)


def world_from_ego(path: str | Path | None, frame_idx: np.ndarray) -> WorldSource:
    """The ego-motion as a world source, accepted exactly as `world_frame.apply_ego_motion` accepts it (same
    rows, self-check `ok`, every row on the trajectory); otherwise `none` with the reason."""
    n = len(frame_idx)
    if path is None or not Path(path).is_file():
        return no_world(n, "no_ego_motion_file")
    with np.load(path, allow_pickle=False) as z:
        c2w = np.asarray(z["cam_to_world"], dtype=np.float64)
        valid = np.asarray(z["valid"], dtype=bool)
        rows = np.asarray(z["frame_idx"], dtype=np.int64)
        ok = bool(json.loads(str(z["summary"])).get("ok", False))
    if not np.array_equal(rows, np.asarray(frame_idx, dtype=np.int64)):
        return no_world(n, "ego_rows_differ")
    if not ok:
        return no_world(n, "ego_self_check_failed")
    if not valid.all():
        return no_world(n, "ego_rows_off_trajectory")
    s = np.cbrt(np.linalg.det(c2w[:, :3, :3]))
    return WorldSource("ego_motion", c2w[:, :3, :3] / s[:, None, None], c2w[:, :3, 3].copy(), s, str(path))


def to_world(poses: np.ndarray, c_star: np.ndarray, ws: WorldSource) -> tuple[np.ndarray, np.ndarray]:
    """`(p (T,3), O (T,3,3))` world centre and rotation of record poses (module docstring)."""
    P = np.asarray(poses, dtype=np.float64)
    R, a = P[:, :3, :3], P[:, :3, 3]
    x = R @ np.asarray(c_star, dtype=np.float64) + ws.sigma[:, None] * a
    return np.einsum("tij,tj->ti", ws.Q, x) + ws.u, ws.Q @ R


def to_record(p: np.ndarray, O: np.ndarray, c_star: np.ndarray, ws: WorldSource) -> np.ndarray:
    """Record poses `(T,4,4)` (pre-ego camera frame) of world centres `p` and rotations `O` (exact inverse)."""
    R = np.einsum("tji,tjk->tik", ws.Q, np.asarray(O, dtype=np.float64))
    xc = np.einsum("tji,tj->ti", ws.Q, np.asarray(p, dtype=np.float64) - ws.u)
    a = (xc - R @ np.asarray(c_star, dtype=np.float64)) / ws.sigma[:, None]
    out = np.tile(np.eye(4), (len(R), 1, 1))
    out[:, :3, :3] = R
    out[:, :3, 3] = a
    return out


def round_trip(poses: np.ndarray, c_star: np.ndarray, ws: WorldSource) -> np.ndarray:
    """record -> world -> record on every row (KA-WRAP-RT); rows without a finite pose are returned unchanged."""
    P = np.asarray(poses)
    out = np.array(P, copy=True)
    ok = np.isfinite(P).all(axis=(1, 2))
    if ok.any():
        sub = WorldSource(ws.kind, ws.Q[ok], ws.u[ok], ws.sigma[ok], ws.reason)
        p, O = to_world(P[ok], c_star, sub)
        out[ok] = to_record(p, O, c_star, sub).astype(P.dtype)
    return out


def sanitize_covariance(cov: np.ndarray | None, n: int, eig_min: float, eig_max: float) -> np.ndarray:
    """Symmetrised, eigenvalue-clipped `(n,6,6)`; a non-finite block becomes `eig_max * I` (no weight); `None` ->
    zeros (the floor alone)."""
    if cov is None:
        return np.zeros((n, 6, 6))
    c = np.asarray(cov, dtype=np.float64)
    out = np.empty_like(c)
    for t in range(n):
        if not np.isfinite(c[t]).all():
            out[t] = eig_max * np.eye(6)
            continue
        s = 0.5 * (c[t] + c[t].T)
        w, v = np.linalg.eigh(s)
        out[t] = (v * np.clip(w, eig_min, eig_max)) @ v.T
    return out


def transport_covariance(cov_cam: np.ndarray, poses: np.ndarray, c_star: np.ndarray, ws: WorldSource) -> np.ndarray:
    """`Sigma^w (T,6,6)` in the [centre; world-left rotation] order (module docstring)."""
    P = np.asarray(poses, dtype=np.float64)
    R, a = P[:, :3, :3], P[:, :3, 3]
    lever = R @ np.asarray(c_star, dtype=np.float64) + ws.sigma[:, None] * a
    n = len(P)
    J = np.zeros((n, 6, 6))
    J[:, :3, :3] = ws.sigma[:, None, None] * np.eye(3)
    J[:, :3, 3:] = -hat(lever)
    J[:, 3:, 3:] = np.eye(3)
    B = np.zeros((n, 6, 6))
    B[:, :3, :3] = ws.Q
    B[:, 3:, 3:] = ws.Q
    M = B @ J
    return M @ np.asarray(cov_cam, dtype=np.float64) @ np.transpose(M, (0, 2, 1))


def scaled_points(points_cam: np.ndarray, ws: WorldSource) -> np.ndarray:
    """Hand points in the ego-SCALED camera frame (sigma_t x): distances to the object posed with (R, sigma a) equal
    the world distances, because the rigid part [Q | u] preserves them."""
    return np.asarray(points_cam, dtype=np.float64) * ws.sigma[:, None, None]
