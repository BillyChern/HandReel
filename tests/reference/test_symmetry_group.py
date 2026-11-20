"""Symmetry group: known answers on analytic meshes."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from fastdo.reference.so3 import angle_deg
from fastdo.reference.symmetry_group import _closure, apply_element, compute_group

from .synth import asym_mesh


def test_cylinder_is_continuous_with_a_flip(cfg) -> None:
    m = trimesh.creation.cylinder(radius=0.03, height=0.15, sections=96)
    G = compute_group(m.vertices, m.faces, 1.0, cfg)
    assert G.continuous and len(G.elements) == 48, G.kind      # D_24: C_24 about the axis x C_2 flips


def test_box_with_distinct_extents_is_d2(cfg) -> None:
    m = trimesh.creation.box(extents=[0.12, 0.08, 0.05])
    G = compute_group(m.vertices, m.faces, 1.0, cfg)
    assert len(G.elements) == 4 and not G.continuous, G.kind
    for a in G.elements[1:]:
        assert abs(angle_deg(a, np.eye(3)) - 180.0) < 1e-6


def test_asymmetric_mesh_is_trivial(cfg) -> None:
    m = asym_mesh()
    G = compute_group(m.vertices, m.faces, 1.0, cfg)
    assert G.trivial


def test_near_symmetric_mesh_is_trivial_at_the_applied_tier(cfg) -> None:
    """A box with a 6 mm bump is symmetric to ~6 mm only: the 3 mm applied tier must NOT accept the flips."""
    a = trimesh.creation.box(extents=[0.12, 0.08, 0.05])
    b = trimesh.creation.box(extents=[0.03, 0.03, 0.006])
    b.apply_translation([0.03, 0.0, 0.028])
    m = trimesh.util.concatenate([a, b])
    G = compute_group(m.vertices, m.faces, 1.0, cfg)
    assert G.trivial, G.kind
    near = compute_group(m.vertices, m.faces, 1.0, cfg, tol_m=cfg.sym_near_tol_m)
    assert not near.trivial


def test_closure_overflow_is_detected() -> None:
    gens = [Rotation.from_rotvec([np.radians(15), 0, 0]).as_matrix(),
            Rotation.from_rotvec([0, np.radians(15), 0]).as_matrix()]
    assert _closure(gens, 48) is None
    c2 = [Rotation.from_rotvec([0, 0, np.pi]).as_matrix(), Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()]
    assert len(_closure(c2, 48)) == 4


def test_relabelling_keeps_the_centre() -> None:
    c = np.array([0.01, 0.02, -0.03])
    P = np.eye(4)
    P[:3, :3] = Rotation.from_rotvec([0.3, 0.1, 0.2]).as_matrix()
    P[:3, 3] = [0.1, 0.2, 0.5]
    g = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
    Q = apply_element(P, g, c)
    assert np.allclose(Q[:3, :3] @ c + Q[:3, 3], P[:3, :3] @ c + P[:3, 3], atol=1e-12)
