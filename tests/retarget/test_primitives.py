"""Tests for `fastdo.retarget.primitives`: capsule minimum signed distance and primitive covers.

The math the module rests on: for an exact distance function phi (1-Lipschitz, |grad phi| = 1 a.e.),
min over the closed ball B(c, r) of phi is phi(c) - r whenever phi(c) >= r, and phi(c) - r is a lower
bound otherwise. A capsule is the union of balls of radius r centred on its axis, so its minimum is
min_t phi(a + t (b - a)) - r. Analytic SDFs (sphere, box) check the formula exactly; the real whisk
SDF grid (trilinear, only approximately a distance function) is checked against dense sampling to
within one grid cell.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import trimesh
from torch import Tensor

from fastdo.retarget.mj_model import RobotScene, _object_contact_geoms
from fastdo.retarget.primitives import (
    capsule_min_distance,
    geom_capsule_covers,
    hand_capsule_covers,
)
from fastdo.solver.sdf import mesh_sdf_grid, sdf_lookup

DT = torch.float64
PhiFn = Callable[[Tensor], tuple[Tensor, Tensor]]
REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "sharpa"
             / "right" / "whisking" / "0" / "scene.xml")
WHISK_OBJ = (REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "whisking"
             / "video_segmentation" / "masks" / "frame_000125_masks" / "whisk" / "whisk.obj")
scene_missing = pytest.mark.skipif(not SCENE_XML.exists(), reason="baseline scene.xml not checked out")
whisk_missing = pytest.mark.skipif(not WHISK_OBJ.exists(), reason="shipped whisk.obj not available")
_PALM_HALF_EXTENTS = np.array([0.016, 0.035, 0.035])   # the Sharpa palm collision box


def _sphere_phi(center: Tensor, radius: float) -> PhiFn:
    def phi(y: Tensor) -> tuple[Tensor, Tensor]:
        d = y - center
        n = d.norm(dim=-1)
        return n - radius, d / n.clamp_min(1e-15)[..., None]
    return phi


def _box_phi(half: Tensor) -> PhiFn:
    """Exact signed distance of an axis-aligned box centred at the origin (inside and outside)."""
    def phi(y: Tensor) -> tuple[Tensor, Tensor]:
        yy = y.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            q = yy.abs() - half
            out = q.clamp_min(0.0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0.0)
            (grad,) = torch.autograd.grad(out.sum(), yy)
        return out.detach(), grad
    return phi


def _point_segment_distance(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-30)
    closest = a + t.clamp(0.0, 1.0)[..., None] * ab
    return (p - closest).norm(dim=-1)


def _uniform_ball(n: int, gen: torch.Generator) -> Tensor:
    d = torch.randn(n, 3, generator=gen, dtype=DT)
    d = d / d.norm(dim=-1, keepdim=True)
    return d * torch.rand(n, 1, generator=gen, dtype=DT) ** (1.0 / 3.0)


@pytest.mark.parametrize("which", ["sphere", "box"])
def test_min_of_an_exact_distance_function_over_a_ball_is_phi_at_centre_minus_radius(which: str) -> None:
    """Lower bound on every sampled ball point (1-Lipschitz), attained at c - r grad phi(c) when phi(c) >= r."""
    gen = torch.Generator().manual_seed(0)
    phi_fn = _sphere_phi(torch.zeros(3, dtype=DT), 0.5) if which == "sphere" else \
        _box_phi(torch.tensor([0.2, 0.3, 0.5], dtype=DT))
    centres = (torch.rand(40, 3, generator=gen, dtype=DT) - 0.5) * 3.0
    radii = torch.rand(40, generator=gen, dtype=DT) * 0.3 + 0.01
    phi_c, grad_c = phi_fn(centres[None])
    ball = _uniform_ball(4000, gen)
    samples = centres[:, None, :] + radii[:, None, None] * torch.cat([ball, ball / ball.norm(dim=-1, keepdim=True)])
    phi_s, _ = phi_fn(samples)
    formula = phi_c[0] - radii
    assert torch.all(phi_s.min(dim=1).values >= formula - 1e-12)          # lower bound, always
    outside = phi_c[0] >= radii
    assert outside.sum() >= 10
    attained, _ = phi_fn((centres - radii[:, None] * grad_c[0])[None])
    assert torch.allclose(attained[0][outside], formula[outside], atol=1e-12)


def test_capsule_min_distance_matches_the_closed_form_for_a_sphere_sdf() -> None:
    """Exact search (many golden steps) reproduces dist(o, segment) - R - r to 1e-9, including capsules
    partly inside the sphere (axis outside its r-neighbourhood of the centre). The 1e-6
    closed-form tolerance therefore holds for the exact search only. The shipped default (16 samples +
    3 golden steps) can only overestimate the minimum, and is bounded here at
    < 1e-4 canonical units (~0.02 mm); on the real whisk grid it differs from an exhaustive search by up
    to ~0.035 mm."""
    gen = torch.Generator().manual_seed(1)
    R, T, P = 0.3, 3, 40
    centre = torch.zeros(3, dtype=DT)
    mid = torch.randn(T, P, 3, generator=gen, dtype=DT)
    mid = mid / mid.norm(dim=-1, keepdim=True) * (R + (torch.rand(T, P, 1, generator=gen, dtype=DT) - 0.3) * 0.2)
    axis = torch.randn(T, P, 3, generator=gen, dtype=DT)
    axis = axis / axis.norm(dim=-1, keepdim=True) * 0.13
    a, b = mid - axis, mid + axis
    radius = torch.full((P,), 0.05, dtype=DT)
    closed = _point_segment_distance(centre.expand(T, P, 3), a, b) - R - radius
    assert (closed < 0).any() and (closed > 0).any()

    exact = capsule_min_distance(_sphere_phi(centre, R), a, b, radius, n_golden=60)
    assert torch.allclose(exact.d_min, closed, atol=1e-9)
    default = capsule_min_distance(_sphere_phi(centre, R), a, b, radius)
    assert torch.all(default.d_min >= closed - 1e-12)
    assert float((default.d_min - closed).max()) < 1e-4
    point = a + exact.t_star[..., None] * (b - a)
    assert torch.allclose(exact.point, point) and torch.all((exact.t_star >= 0) & (exact.t_star <= 1))

    warm = capsule_min_distance(_sphere_phi(centre, R), a, b, radius, t_init=exact.t_star)
    assert torch.all(warm.d_min <= default.d_min + 1e-15)                   # warm start never hurts
    assert torch.allclose(warm.d_min, closed, atol=1e-9)


def _covers_as_tensors(covers: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[Tensor, Tensor, Tensor]:
    a = torch.as_tensor(np.stack([c[0] for c in covers]), dtype=DT)
    b = torch.as_tensor(np.stack([c[1] for c in covers]), dtype=DT)
    r = torch.as_tensor([c[2] for c in covers], dtype=DT)
    return a, b, r


def _random_pose(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return rng.normal(size=3) * 0.02, quat, mat.reshape(3, 3)


@pytest.mark.parametrize(("gtype", "size"), [
    (mujoco.mjtGeom.mjGEOM_BOX, _PALM_HALF_EXTENTS),
    (mujoco.mjtGeom.mjGEOM_BOX, np.array([0.03, 0.012, 0.02])),
    (mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.032, 0.046, 0.0])),
    (mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([0.01, 0.0235, 0.0])),
])
def test_covers_contain_their_primitive_and_match_the_closed_form(gtype: int, size: np.ndarray) -> None:
    """Every point of the solid primitive lies in some cover (conservative); against an analytic sphere
    SDF the covers' d_min is min_i dist(o, segment_i) - R - r_i, never above the primitive's exact
    distance dist(o, primitive) - R; a box stack bulges at most `box_face_tol` past its largest faces."""
    pos, quat, mat = _random_pose(int(gtype) + int(size[0] * 1e4))
    covers = geom_capsule_covers(int(gtype), size, pos, quat, box_face_tol=5e-4)
    a, b, r = _covers_as_tensors(covers)
    gen = torch.Generator().manual_seed(2)
    u = torch.rand(20000, 3, generator=gen, dtype=DT) * 2.0 - 1.0
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        corners = torch.cartesian_prod(*[torch.tensor([-1.0, 0.0, 1.0], dtype=DT)] * 3)  # corners, edges, faces
        local = torch.cat([u, corners]) * torch.as_tensor(size, dtype=DT)
    else:   # cylinder/capsule core: disc radius size[0] x half-length size[1] along local z
        rho = u[:, 0].abs().sqrt() * size[0]
        ang = u[:, 1] * torch.pi
        local = torch.stack([rho * torch.cos(ang), rho * torch.sin(ang), u[:, 2] * size[1]], dim=-1)
    world = local @ torch.as_tensor(mat, dtype=DT).T + torch.as_tensor(pos, dtype=DT)
    slack = _point_segment_distance(world[:, None, :], a[None], b[None]) - r[None]
    assert float(slack.min(dim=1).values.max()) <= 1e-12

    centre = torch.as_tensor(pos, dtype=DT) + torch.tensor([0.05, -0.03, 0.04], dtype=DT)
    sphere_r = 0.02
    got = capsule_min_distance(_sphere_phi(centre, sphere_r), a[None], b[None], r, n_golden=60)
    closed = _point_segment_distance(centre.expand(1, len(covers), 3), a[None], b[None]) - sphere_r - r
    assert torch.allclose(got.d_min, closed, atol=1e-9)
    exact_local = (centre - torch.as_tensor(pos, dtype=DT)) @ torch.as_tensor(mat, dtype=DT)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        exact = (exact_local.abs() - torch.as_tensor(size, dtype=DT)).clamp_min(0.0).norm() - sphere_r
        e_sorted = np.sort(size)
        assert float(r.max()) <= e_sorted[0] + 5e-4 + 1e-12
    else:
        radial = exact_local[:2].norm() - size[0]
        axial = exact_local[2].abs() - size[1]
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            exact = _point_segment_distance(exact_local, torch.tensor([0, 0, -size[1]], dtype=DT),
                                            torch.tensor([0, 0, size[1]], dtype=DT)) - size[0] - sphere_r
        else:
            exact = torch.stack([radial, axial]).clamp_min(0.0).norm() - sphere_r
    assert float(got.d_min.min()) <= float(exact) + 1e-12


def test_box_edge_capsules_of_the_smallest_half_extent_miss_the_palm_face() -> None:
    """Why the box cover is a mid-plane stack and not 4 edge capsules of radius min(e) plus a centre
    sphere: for the palm box (16 x 35 x 35 mm half-extents) a point on its large face lies 23.7 mm from
    the centre and >= 35 mm from every edge axis, outside all of those radius-16 mm covers."""
    e1, e2, e3 = _PALM_HALF_EXTENTS
    face_point = torch.tensor([e1, 0.0, e3 / 2], dtype=DT)
    edges = [(torch.tensor([sx * e1, sy * e2, -e3], dtype=DT), torch.tensor([sx * e1, sy * e2, e3], dtype=DT))
             for sx in (-1, 1) for sy in (-1, 1)]
    edge_slack = min(float(_point_segment_distance(face_point, ea, eb)) - e1 for ea, eb in edges)
    assert edge_slack > 0.0 and float(face_point.norm()) - e1 > 0.0
    covers = geom_capsule_covers(int(mujoco.mjtGeom.mjGEOM_BOX), _PALM_HALF_EXTENTS, np.zeros(3),
                                 np.array([1.0, 0.0, 0.0, 0.0]), box_face_tol=5e-4)
    a, b, r = _covers_as_tensors(covers)
    assert float((_point_segment_distance(face_point.expand(len(covers), 3), a, b) - r).min()) <= 0.0


@scene_missing
def test_hand_capsule_covers_cover_every_hand_collision_geom_on_the_whisking_scene() -> None:
    """Only geometry the simulator collides with the object is covered: the scene's contacts are explicit
    pairs (contype = conaffinity = 0 everywhere), and the arm cylinder on the palm body is in none of them."""
    scene = RobotScene.from_xml(SCENE_XML)
    covers = hand_capsule_covers(scene)
    geoms = scene.hand_collision_geoms()
    model = scene.model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms]
    assert len(geoms) == 16 and "collision_hand_right_arm_cyl" not in names
    hand_bodies = {int(model.geom_bodyid[g]) for g in geoms}
    for g in geoms:
        partners = np.r_[model.pair_geom2[model.pair_geom1 == g], model.pair_geom1[model.pair_geom2 == g]]
        assert any(int(model.geom_bodyid[p]) not in hand_bodies for p in partners), names[geoms.index(g)]
    assert set(covers.geom_ids.tolist()) == set(geoms) and covers.n == 24
    types = {int(model.geom_type[g]) for g in geoms}
    assert types == {int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}
    assert covers.local_a.shape == covers.local_b.shape == (covers.n, 3)
    for g in geoms:
        sel = covers.geom_ids == g
        assert np.all(covers.body_ids[sel] == model.geom_bodyid[g])
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            assert sel.sum() == 1
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, model.geom_quat[g])
            z = mat.reshape(3, 3)[:, 2] * model.geom_size[g][1]
            np.testing.assert_allclose(covers.local_a[sel][0], model.geom_pos[g] - z, atol=1e-12)
            np.testing.assert_allclose(covers.local_b[sel][0], model.geom_pos[g] + z, atol=1e-12)
            assert covers.radius[sel][0] == pytest.approx(model.geom_size[g][0])


_CONTACT_XML = """<mujoco><worldbody>
  <body name="obj"><geom name="obj_g" type="sphere" size="0.05" contype="0" conaffinity="0"/></body>
  <body name="hand">
    <geom name="paired" type="capsule" size="0.01 0.02" contype="0" conaffinity="0"/>
    <geom name="hand_only_pair" type="capsule" size="0.01 0.02" contype="0" conaffinity="0"/>
    <geom name="bitmask" type="capsule" size="0.01 0.02" contype="1" conaffinity="0"/>
    <geom name="inert" type="cylinder" size="0.03 0.04" contype="0" conaffinity="0"/>
  </body></worldbody>
  <contact><pair geom1="obj_g" geom2="paired"/><pair geom1="paired" geom2="hand_only_pair"/></contact>
</mujoco>"""


def test_object_contact_geoms_keeps_only_geometry_the_simulator_can_collide_with_the_object() -> None:
    """A hand geom counts when an explicit pair joins it to a non-hand geom, or when its contype /
    conaffinity bits let MuJoCo collide it dynamically; a pair with another hand geom does not count."""
    model = mujoco.MjModel.from_xml_string(_CONTACT_XML)
    gid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
           for n in ("paired", "hand_only_pair", "bitmask", "inert")}
    hand = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")]
    kept = _object_contact_geoms(model, list(gid.values()), hand)
    assert kept == [gid["paired"], gid["bitmask"]]


@scene_missing
def test_axis_point_jacobian_is_the_endpoint_blend_for_a_rigid_body() -> None:
    """dp*/dq = (1 - t*) dA/dq + t* dB/dq holds exactly for body-fixed endpoints (the envelope-theorem
    Jacobian's only kinematic ingredient), checked against `point_kinematics` on the blended point."""
    scene = RobotScene.from_xml(SCENE_XML)
    covers = hand_capsule_covers(scene)
    rng = np.random.default_rng(3)
    q = np.zeros((2, scene.model.nq))
    q[:, 31] = 1.0
    q[:, :28] = rng.normal(size=(2, 28)) * 0.2
    t = rng.uniform(size=covers.n)
    ids = np.concatenate([covers.body_ids, covers.body_ids, covers.body_ids])
    pts = np.concatenate([covers.local_a, covers.local_b, (1 - t)[:, None] * covers.local_a
                          + t[:, None] * covers.local_b])
    world, jac = scene.point_kinematics(q, ids, pts)
    assert jac is not None
    P = covers.n
    np.testing.assert_allclose((1 - t)[None, :, None, None] * jac[:, :P] + t[None, :, None, None] * jac[:, P:2 * P],
                               jac[:, 2 * P:], atol=1e-10)
    np.testing.assert_allclose((1 - t)[None, :, None] * world[:, :P] + t[None, :, None] * world[:, P:2 * P],
                               world[:, 2 * P:], atol=1e-12)


@whisk_missing
def test_capsule_min_distance_on_the_whisk_sdf_grid_is_within_one_cell_of_dense_sampling() -> None:
    """The trilinear grid is only approximately a distance function, so phi(axis) - r can differ from
    the grid's own minimum over the capsule volume. In the formula's regime (phi(axis) >= r, i.e.
    d_min >= 0, where the clearance band constrains the hand) it stays within one grid cell. For a
    penetrating capsule it is a lower bound up to one cell, and can be far deeper: the whisk's wires
    are ~2 mm thick, so a 9 mm ball around a wire reports ~ -8 mm while the true depth is ~1 mm."""
    mesh = trimesh.load(WHISK_OBJ, process=False)
    assert isinstance(mesh, trimesh.Trimesh)
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64)
    scale = 0.18086324855404615
    pts, face_idx = trimesh.sample.sample_surface(mesh, 24, seed=np.random.default_rng(4))
    normals = np.asarray(mesh.face_normals[face_idx])
    gen = torch.Generator().manual_seed(4)
    P = pts.shape[0]
    radius = torch.full((P,), 0.009 / scale, dtype=DT)
    offset = (torch.rand(P, 1, generator=gen, dtype=DT) * 3.0 + 0.2) * radius[:, None]
    mid = torch.as_tensor(pts, dtype=DT) + offset * torch.as_tensor(normals, dtype=DT)
    axis = torch.randn(P, 3, generator=gen, dtype=DT)
    axis = axis / axis.norm(dim=-1, keepdim=True) * (0.016 / scale)
    a, b = (mid - axis)[None], (mid + axis)[None]

    got = capsule_min_distance(lambda y: sdf_lookup(sdf, y), a, b, radius)
    ts = torch.linspace(0.0, 1.0, 61, dtype=DT)
    ball = _uniform_ball(800, gen)
    ball = torch.cat([ball, ball / ball.norm(dim=-1, keepdim=True)])        # interior + surface samples
    dense = (a[0][:, None, None, :] + ts[None, :, None, None] * (b - a)[0][:, None, None, :]
             + radius[:, None, None, None] * ball[None, None, :, :])         # (P, 61, 1600, 3)
    phi, _ = sdf_lookup(sdf, dense.reshape(1, -1, 3))
    truth = phi.reshape(P, -1).min(dim=1).values
    clear = got.d_min[0] >= 0.0
    assert clear.sum() >= 5 and (~clear).sum() >= 3, f"test setup: {int(clear.sum())} clear capsules"
    err = (got.d_min[0] - truth)[clear].abs()
    assert float(err.max()) <= sdf.step, f"max |d_min - dense| = {float(err.max()):.4g} > cell {sdf.step:.4g}"
    assert float((got.d_min[0] - truth)[~clear].max()) <= sdf.step


@whisk_missing
def test_value_only_axis_search_is_bitwise_identical() -> None:
    """Value-only lookups in the candidate scan and golden-section steps change nothing."""
    from fastdo.retarget.primitives import as_phi_fn, as_values_fn

    mesh = trimesh.load(WHISK_OBJ, process=False)
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64)
    gen = torch.Generator().manual_seed(4)
    end_a = (torch.rand(10, 24, 3, generator=gen, dtype=DT) * 2.0 - 1.0) * 0.6
    end_b = end_a + 0.1 * torch.randn(10, 24, 3, generator=gen, dtype=DT)
    radius = torch.full((24,), 0.02, dtype=DT)
    values_fn = as_values_fn(sdf)
    assert values_fn is not None and as_values_fn(_sphere_phi(torch.zeros(3, dtype=DT), 0.3)) is None
    t_init = None
    for _ in range(2):
        ref = capsule_min_distance(as_phi_fn(sdf), end_a, end_b, radius, t_init=t_init)
        fast = capsule_min_distance(as_phi_fn(sdf), end_a, end_b, radius, t_init=t_init, values_fn=values_fn)
        for field in ("d_min", "t_star", "point", "grad"):
            assert torch.equal(getattr(ref, field), getattr(fast, field)), field
        t_init = ref.t_star
