import math

import torch

from fastdo.geometry.lie import apply_left, se3_exp
from fastdo.solver.sdf import SdfGrid
from fastdo.solver.terms import project, transform_points
from fastdo.solver.terms_image import (
    boundary_pixels,
    contact_term,
    distance_transform,
    pad_boundaries,
    prefilter_distance_transforms,
    silhouette_term,
)

torch.manual_seed(0)
DT = torch.float64


def _disc_mask(hw: tuple[int, int], cu: float, cv: float, r: float) -> torch.Tensor:
    H, W = hw
    vv, uu = torch.meshgrid(torch.arange(H, dtype=DT), torch.arange(W, dtype=DT), indexing="ij")
    return ((uu - cu) ** 2 + (vv - cv) ** 2) <= r * r


def _perturb(poses, log_scale, delta):
    T = poses.shape[0]
    return torch.stack([apply_left(poses[k], delta[6 * k:6 * k + 6]) for k in range(T)]), log_scale + delta[-1]


def _sphere_points(m: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.randn(m, 3, dtype=DT)
    n = v / v.norm(dim=1, keepdim=True)
    return 0.5 * n, n


def test_distance_transform_and_boundary() -> None:
    mask = _disc_mask((64, 64), 32.0, 32.0, 10.0)
    dt = distance_transform(mask)
    assert dt[32, 32] == 0.0 and abs(float(dt[32, 52]) - 10.0) < 1.5
    excl = torch.zeros(64, 64, dtype=torch.bool)
    excl[:, 40:] = True
    b = boundary_pixels(mask, excl)
    assert b.shape[1] == 2 and b.shape[0] > 20
    assert (b[:, 0] < 38).all()                          # boundary near the excluded region is dropped
    bb, valid = pad_boundaries([b, b[:5]])
    assert bb.shape == (2, b.shape[0], 2) and valid[1].sum() == 5


def test_silhouette_out_term_matches_finite_difference() -> None:
    pts, nrm = _sphere_points()
    poses = se3_exp(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=DT))
    poses[0, 2, 3] = 1.0
    log_scale = torch.tensor(math.log(0.2), dtype=DT)
    K = torch.tensor([[200.0, 0.0, 64.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]], dtype=DT)
    mask = _disc_mask((128, 128), 70.0, 60.0, 18.0)       # deliberately offset from the object
    dt = distance_transform(mask)[None]
    bnd, bval = pad_boundaries([boundary_pixels(mask, torch.zeros_like(mask))])
    hands = torch.zeros(1, 128, 128, dtype=torch.bool)

    def res(p, s):
        return silhouette_term(pts, nrm, dt, bnd, bval, hands, p, s, K, 1.0, 0.0).residual[0, : pts.shape[0]]

    out = silhouette_term(pts, nrm, dt, bnd, bval, hands, poses, log_scale, K, 1.0, 0.0)
    J = out.jacobian[0, : pts.shape[0]]                   # (M,1,7)
    w = out.weight[0, : pts.shape[0]]
    assert (w > 0).sum() > 100
    h = 1e-5
    for i in range(7):
        d = torch.zeros(7, dtype=DT); d[i] = h
        num = (res(*_perturb(poses, log_scale, d)) - res(*_perturb(poses, log_scale, -d))) / (2 * h)
        sel = w > 0
        assert torch.allclose(J[sel, :, i], num[sel], atol=2e-2), f"column {i}"


def test_silhouette_cost_decreases_toward_truth() -> None:
    pts, nrm = _sphere_points()
    K = torch.tensor([[200.0, 0.0, 64.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]], dtype=DT)
    poses_gt = torch.eye(4, dtype=DT)[None].clone(); poses_gt[0, 2, 3] = 1.0
    log_scale = torch.tensor(math.log(0.2), dtype=DT)
    uv = project(transform_points(pts, poses_gt, log_scale.exp()), K)[0]
    mask = _disc_mask((128, 128), float(uv[:, 0].mean()), float(uv[:, 1].mean()), 0.1 * 200.0)
    dt = distance_transform(mask)[None]
    bnd, bval = pad_boundaries([boundary_pixels(mask, torch.zeros_like(mask))])
    hands = torch.zeros(1, 128, 128, dtype=torch.bool)

    def cost(p):
        o = silhouette_term(pts, nrm, dt, bnd, bval, hands, p, log_scale, K, 1.0, 1.0)
        return float((o.weight * (o.residual[..., 0] ** 2)).sum())

    shifted = poses_gt.clone(); shifted[0, 0, 3] += 0.02
    assert cost(poses_gt) < cost(shifted)


def test_silhouette_term_matches_with_cached_prefilter() -> None:
    pts, nrm = _sphere_points()
    poses = se3_exp(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=DT))
    poses[0, 2, 3] = 1.0
    log_scale = torch.tensor(math.log(0.2), dtype=DT)
    K = torch.tensor([[200.0, 0.0, 64.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]], dtype=DT)
    mask = _disc_mask((128, 128), 70.0, 60.0, 18.0)
    dt = distance_transform(mask)[None]
    bnd, bval = pad_boundaries([boundary_pixels(mask, torch.zeros_like(mask))])
    hands = torch.zeros(1, 128, 128, dtype=torch.bool)

    out_raw = silhouette_term(pts, nrm, dt, bnd, bval, hands, poses, log_scale, K, 1.0, 1.0)
    splines = prefilter_distance_transforms(dt)
    out_cached = silhouette_term(pts, nrm, splines, bnd, bval, hands, poses, log_scale, K, 1.0, 1.0)
    assert torch.allclose(out_raw.residual, out_cached.residual, atol=1e-10)
    assert torch.allclose(out_raw.jacobian, out_cached.jacobian, atol=1e-10)
    assert torch.equal(out_raw.weight, out_cached.weight)


def test_contact_term_penetration_and_jacobian() -> None:
    G = 64
    ax = torch.linspace(-1.0, 1.0, G, dtype=DT)
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
    sphere = SdfGrid(values=torch.sqrt(gx**2 + gy**2 + gz**2) - 0.5, origin=torch.full((3,), -1.0, dtype=DT),
                     step=float(ax[1] - ax[0]))
    poses = se3_exp(torch.tensor([[0.1, -0.05, 0.02, 0.2, 0.1, -0.3]], dtype=DT))
    poses[0, 2, 3] += 1.0
    log_scale = torch.tensor(math.log(0.3), dtype=DT)
    canon = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, 0.9], [0.3, 0.3, 0.0]]], dtype=DT)   # inside, outside, inside
    hand = transform_points(canon[0], poses, log_scale.exp())                             # (1,3,3) camera frame
    fingertips = torch.tensor([1, 2])
    in_hand = torch.tensor([True])
    out = contact_term(sphere, hand, fingertips, in_hand, poses, log_scale, 1.0, 1.0, delta=0.005)
    r_pen = out.residual[0, :3, 0]
    assert r_pen[0] > 0 and r_pen[2] > 0 and r_pen[1] == 0
    r_con = out.residual[0, 3:, 0]
    assert r_con[0] > 0 and r_con[1] == 0                                                   # only the outside fingertip

    def res(p, s):
        return contact_term(sphere, hand, fingertips, in_hand, p, s, 1.0, 1.0, 0.005).residual[0]

    h = 1e-5
    for i in range(7):
        d = torch.zeros(7, dtype=DT); d[i] = h
        num = (res(*_perturb(poses, log_scale, d)) - res(*_perturb(poses, log_scale, -d))) / (2 * h)
        assert torch.allclose(out.jacobian[0, :, :, i], num, atol=2e-3), f"column {i}"
