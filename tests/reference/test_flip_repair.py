"""Flip repair: an injected 180 deg relabelling is restored exactly; otherwise byte-identity."""

from __future__ import annotations

import dataclasses

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.reference import condition_window, no_world
from fastdo.reference.symmetry_group import apply_element, compute_group

from .synth import window


def _traj(n=100):
    P = np.tile(np.eye(4), (n, 1, 1))
    t = np.arange(n) / 10
    P[:, :3, :3] = Rotation.from_rotvec(np.stack([0.3 * np.sin(t), 0.2 * t, 0.1 + 0 * t], 1)).as_matrix()
    P[:, :3, 3] = np.stack([0.05 * t, 0.02 * np.cos(t), 0.6 + 0 * t], 1)
    return P


def _box_window(P, cfg, n):
    m = trimesh.creation.box(extents=[0.12, 0.08, 0.05])
    G = compute_group(m.vertices, m.faces, 1.0, cfg)
    w = window(P, np.full(n, 0.005), mesh=m, group=G, world=no_world(n, "test"))
    return w, G


def _inject(P, G, rows, k=1):
    Q = P.copy()
    for r in rows:
        Q[r] = apply_element(P[r], G.elements[k], G.c_star)
    return Q


def test_injected_flip_run_is_restored(cfg) -> None:
    n = 100
    P = _traj(n)
    w, G = _box_window(P, cfg, n)
    Q = _inject(P, G, range(30, 46))
    ref = condition_window(dataclasses.replace(w, poses=Q), cfg)
    assert ref.flip["active"] and ref.flip["relabelled_rows"] == 16
    assert np.allclose(ref.poses, P, atol=1e-9)
    assert np.array_equal(ref.poses[~ref.changed], Q[~ref.changed])
    assert np.all(G.mismatch_p95_m <= cfg.sym_tol_abs_m)


def test_flip_at_the_end_relabels_the_short_run(cfg) -> None:
    """Only the unary breaks the global ambiguity: rows 85-99 flipped -> relabel those 15 rows, never rows 0-84."""
    n = 100
    P = _traj(n)
    w, G = _box_window(P, cfg, n)
    Q = _inject(P, G, range(85, 100))
    ref = condition_window(dataclasses.replace(w, poses=Q), cfg)
    assert np.array_equal(ref.poses[:85], Q[:85])
    assert np.allclose(ref.poses, P, atol=1e-9)


def test_no_large_step_is_byte_identical(cfg) -> None:
    n = 100
    P = _traj(n).astype(np.float32)
    w, _ = _box_window(P, cfg, n)
    ref = condition_window(w, cfg)
    assert not ref.flip["active"] and not ref.changed.any()
    assert ref.poses.dtype == np.float32 and ref.poses.tobytes() == P.tobytes()


def test_trivial_group_is_byte_identical(cfg) -> None:
    n = 60
    P = _traj(n)
    g = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
    P[20:30, :3, :3] = P[20:30, :3, :3] @ g
    w = window(P, np.full(n, 0.005), world=no_world(n, "test"))
    ref = condition_window(w, cfg)
    assert w.group.trivial and not ref.changed.any() and ref.poses.tobytes() == P.tobytes()
