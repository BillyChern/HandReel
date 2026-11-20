"""Known answers of T1, the contact-set stage of the transfer, for
fastdo.robot.contact_transfer: geodesic W1 between contact sets on the object surface, W1 quantisation of a certified
contact measure onto k Dex3 sites, and the site-level acceptance test against the object's wrench demand."""
from __future__ import annotations

import numpy as np
import trimesh

from fastdo.robot.contact_transfer import (
    SiteConfig,
    accept_sites,
    geodesic_matrix,
    quantise,
    surface_graph,
    w1_uniform,
)

R = 0.04


def _sphere():
    m = trimesh.creation.icosphere(subdivisions=4, radius=R)
    return m, surface_graph(m.vertices, m.faces)


def _vid(m, p):
    return int(np.argmin(np.linalg.norm(m.vertices - np.asarray(p), axis=1)))


def test_identical_contact_sets_transport_at_zero_cost() -> None:
    m, g = _sphere()
    ids = [_vid(m, R * np.array(v)) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    assert w1_uniform(geodesic_matrix(g, ids, ids)) == 0.0


def test_a_rotated_set_on_a_sphere_costs_r_theta() -> None:
    """Rotate 3 equatorial contacts by theta about z: the W1 is the mean true geodesic between the snapped vertex
    pairs (r * angle), to 3% (the 2-ring graph's measured error is <= 1.4% at this resolution; an edge-only graph is
    16-18% high at every resolution)."""
    m, g = _sphere()
    th = np.radians(40.0)
    az = np.radians([0.0, 120.0, 240.0])
    src = [_vid(m, R * np.array([np.cos(a), np.sin(a), 0.0])) for a in az]
    tgt = [_vid(m, R * np.array([np.cos(a + th), np.sin(a + th), 0.0])) for a in az]
    w = w1_uniform(geodesic_matrix(g, src, tgt[::-1]))            # target listed in another order: OT must re-pair
    v = m.vertices
    true = np.mean([R * np.arccos(np.clip(v[s] @ v[t] / R ** 2, -1.0, 1.0)) for s, t in zip(src, tgt, strict=True)])
    assert abs(w / true - 1.0) <= 0.03, (w, true)
    assert abs(true / (R * th) - 1.0) < 0.05                       # the snapped pairs are a 40 deg rotation


def test_quantisation_keeps_where_the_load_is_carried() -> None:
    """Two tight clusters on opposite poles: 2 north contacts carrying 5 units of force each, 6 south carrying 1. They
    quantise onto 2 sites, one per cluster. With k = 1 the single site sits in the HEAVY (north) cluster, although the
    south has more contacts, so the force weights decide, not the count."""
    m, g = _sphere()
    rng = np.random.default_rng(0)
    north = [_vid(m, R * np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.004, 3)) for _ in range(2)]
    south = [_vid(m, R * np.array([0.0, 0.0, -1.0]) + rng.normal(0, 0.004, 3)) for _ in range(6)]
    src, w = north + south, np.array([5.0] * 2 + [1.0] * 6)
    sites, cost = quantise(g, src, w, k=2)
    z = m.vertices[sites, 2]
    assert sorted(np.sign(z)) == [-1.0, 1.0] and cost >= 0.0
    one, _ = quantise(g, src, w, k=1)
    assert m.vertices[one[0], 2] > 0.0


def _pad(m, centre, axes):
    """A 1 x 1 cm finger pad: 4 surface points around `centre` along the two in-face `axes` (a patch, so it resists
    torsion about its normal; two opposing POINT contacts never can, so they are never in force closure)."""
    c = np.asarray(centre, float)
    return [_vid(m, c + 0.005 * (a * np.asarray(axes[0]) + b * np.asarray(axes[1]))) for a in (-1, 1) for b in (-1, 1)]


def test_opposing_pads_give_accepted_sites_and_same_face_pads_are_refused() -> None:
    """A 6 x 6 x 6 cm box carried (static, 0.2 kg) by two 1 cm pads on OPPOSITE faces: the sites can supply the
    weight and are in force closure (Q > 0). The same pads side by side on ONE face: Q < 0 and T1 refuses."""
    box = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
    sub = box.subdivide().subdivide().subdivide().subdivide()
    g = surface_graph(sub.vertices, sub.faces)
    weight = np.array([0.0, 0.0, 0.2 * 9.81, 0.0, 0.0, 0.0])
    cfg = SiteConfig()
    yz = ([0, 1, 0], [0, 0, 1])
    opp = _pad(sub, [0.03, 0.0, 0.0], yz) + _pad(sub, [-0.03, 0.0, 0.0], yz)
    n_opp = np.array([[-1.0, 0, 0]] * 4 + [[1.0, 0, 0]] * 4)                  # face normals, pointing INTO the box
    ok = accept_sites(sub.vertices[opp], n_opp, np.zeros(3), weight, cfg)
    assert ok["accepted"] and ok["quality"] > 0.0 and ok["static_hold_residual"] < 1e-6
    same = _pad(sub, [0.03, 0.012, 0.0], yz) + _pad(sub, [0.03, -0.012, 0.0], yz)
    bad = accept_sites(sub.vertices[same], np.array([[-1.0, 0, 0]] * 8), np.zeros(3), weight, cfg)
    assert not bad["accepted"] and bad["quality"] < 0.0
    assert g.shape[0] == len(sub.vertices)


def test_load_share_weights_follow_the_load_not_the_squeeze() -> None:
    """A box held by two side pads (4 points each) and touched by one contact pressing DOWN on its top: the minimum-norm
    balancing forces split the weight equally between the two side pads, and the top contact (which can only push the
    box down) carries ~nothing. Squeeze never enters: the shares sum to what the load needs."""
    from fastdo.robot.contact_transfer import load_share_weights

    pts = np.array([[0.03, y, z] for y in (-0.005, 0.005) for z in (-0.005, 0.005)]
                   + [[-0.03, y, z] for y in (-0.005, 0.005) for z in (-0.005, 0.005)] + [[0.0, 0.0, 0.03]])
    nrm = np.array([[-1.0, 0, 0]] * 4 + [[1.0, 0, 0]] * 4 + [[0.0, 0, -1.0]])
    weight = np.array([0.0, 0.0, 0.2 * 9.81, 0.0, 0.0, 0.0])
    w = load_share_weights(pts, nrm, np.zeros(3), weight, SiteConfig())
    left, right, top = w[:4].sum(), w[4:8].sum(), w[8]
    assert np.isclose(left, right, rtol=1e-3) and top < 1e-3 * (left + right)
    assert (w >= 0).all()


def test_quantisation_on_a_disconnected_mesh() -> None:
    """Object meshes can be several geoms concatenated (disconnected graph): unreachable pairs are charged a finite
    cost, so quantisation still works, and k = 2 puts one site on each part."""
    a = trimesh.creation.icosphere(subdivisions=3, radius=R)
    b = a.copy()
    b.apply_translation([0.2, 0.0, 0.0])
    m = trimesh.util.concatenate([a, b])
    g = surface_graph(m.vertices, m.faces)
    src = [_vid(m, [R, 0, 0]), _vid(m, [0, R, 0]), _vid(m, [0.2 + R, 0, 0]), _vid(m, [0.2, R, 0])]
    sites, cost = quantise(g, src, np.ones(4), k=2)
    assert sorted(m.vertices[sites, 0] > 0.1) == [False, True] and np.isfinite(cost)


def test_sites_are_real_contacts_not_mesh_vertices() -> None:
    """Regression (oak5e99, 2026-10-17): sites must be the ORIGINAL contact coordinates. Contacts placed between
    vertices keep their exact position; the vertex id serves only the transport cost."""
    from fastdo.robot.contact_transfer import site_contacts

    m, g = _sphere()
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(6, 3))
    pts = R * pts / np.linalg.norm(pts, axis=1, keepdims=True)          # on the sphere, generally between vertices
    vid = np.array([_vid(m, p) for p in pts])
    sites, _ = quantise(g, vid, np.ones(6), k=3)
    rep = site_contacts(g, vid, sites)
    chosen = pts[rep]
    assert all(any(np.array_equal(c, p) for p in pts) for c in chosen)       # exactly source points
    gap = np.linalg.norm(m.vertices[sites] - chosen, axis=1)
    assert (gap > 1e-6).all()                       # and NOT their snapped vertices (random points lie between vertices)
