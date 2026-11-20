import math

import numpy as np
import torch
import trimesh
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.qc.features import (
    QcFeatures,
    laplace_std_ok_frac,
    ot_residual_rms_frac,
    qc_features,
    render_silhouettes,
    reprojection_iou,
    track_inlier_ratio,
)
from fastdo.solver.api import MeshSamples
from fastdo.solver.sdf import mesh_sdf_grid

_H, _W = 480, 640
_F = 600.0
_R_CANON = 0.04          # canonical sphere radius (m); metric radius = scale * _R_CANON
_SCALE = 1.5


def _K() -> Tensor:
    return torch.tensor([[_F, 0.0, _W / 2], [0.0, _F, _H / 2], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _sphere_mesh() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=4, radius=_R_CANON)


def _mesh_samples(n: int = 2048, seed: int = 0) -> MeshSamples:
    """Random-area surface samples (the sampler dexycb_eval uses), at the solver's default density."""
    mesh = _sphere_mesh()
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = np.asarray(mesh.face_normals)[np.asarray(face_idx)]
    return MeshSamples(points=torch.as_tensor(np.asarray(pts), dtype=torch.float32),
                       normals=torch.as_tensor(normals, dtype=torch.float32))


def _poses(seed: int = 0) -> Tensor:
    """Random rotations (irrelevant to a sphere's silhouette, relevant to the splat) at 0.5-0.65 m."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.tensor([[0.0, 0.0, 0.50], [0.05, -0.03, 0.55], [-0.06, 0.04, 0.60], [0.02, 0.05, 0.65]],
                     dtype=torch.float64)
    xi = torch.cat([torch.zeros(4, 3, dtype=torch.float64),
                    torch.randn(4, 3, generator=gen, dtype=torch.float64)], dim=1)
    poses = torch.stack([se3_exp(x) for x in xi])
    poses[:, :3, 3] = t
    return poses


def _analytic_sphere_masks(centres: Tensor, radius: float, K: Tensor) -> Tensor:
    """Exact silhouette: pixel centre (u,v) is inside iff its camera ray meets the sphere."""
    vv, uu = torch.meshgrid(torch.arange(_H, dtype=torch.float64), torch.arange(_W, dtype=torch.float64),
                            indexing="ij")
    d = torch.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], torch.ones_like(uu)], dim=-1)
    d = d / d.norm(dim=-1, keepdim=True)
    out = []
    for c in centres:
        along = (d * c).sum(-1)
        out.append((c.dot(c) - along**2 <= radius * radius) & (along > 0))
    return torch.stack(out)


def _call(poses_est: Tensor, masks: Tensor, hand_verts: Tensor | None = None, with_sdf: bool = False,
          covariance: Tensor | None = None, per_frame_cost: Tensor | None = None,
          track_weight: Tensor | None = None) -> QcFeatures:
    T = poses_est.shape[0]
    mesh = _sphere_mesh()
    sdf = mesh_sdf_grid(np.asarray(mesh.vertices), np.asarray(mesh.faces), resolution=64) if with_sdf else None
    return qc_features(
        masks=masks, poses=poses_est, scale=_SCALE, mesh=_mesh_samples(), K=_K(),
        covariance=covariance if covariance is not None else torch.eye(6, dtype=torch.float64).expand(T, 6, 6) * 1e-6,
        per_frame_cost=per_frame_cost if per_frame_cost is not None else torch.full((T,), 0.001),
        diameter=2.0 * _SCALE * _R_CANON, hand_verts=hand_verts, sdf=sdf,
        track_weight=track_weight if track_weight is not None else torch.ones(T, 8),
    )


def test_qc_features_reprojection_iou_perfect_pose() -> None:
    poses = _poses()
    masks = _analytic_sphere_masks(poses[:, :3, 3], _SCALE * _R_CANON, _K())
    qc = _call(poses, masks)
    assert qc.reprojection_iou > 0.98


def test_qc_features_reprojection_iou_drops_with_pose_error() -> None:
    poses = _poses()
    masks = _analytic_sphere_masks(poses[:, :3, 3], _SCALE * _R_CANON, _K())
    shifted = poses.clone()
    shifted[:, 0, 3] += 0.03                          # 3 cm lateral error: ~half the metric radius
    perfect = _call(poses, masks).reprojection_iou
    wrong = _call(shifted, masks).reprojection_iou
    assert wrong < 0.7
    assert wrong < perfect - 0.25


def test_reprojection_iou_skips_frames_with_empty_union() -> None:
    poses = _poses()
    masks = _analytic_sphere_masks(poses[:, :3, 3], _SCALE * _R_CANON, _K())
    behind = poses.clone()
    behind[1, 2, 3] = -0.5                            # object behind the camera: nothing rendered
    masks[1] = False                                  # ... and nothing segmented
    pts = _mesh_samples().points
    iou_all = reprojection_iou(masks, behind, _SCALE, pts, _K())
    keep = torch.tensor([0, 2, 3])
    iou_kept = reprojection_iou(masks[keep], behind[keep], _SCALE, pts, _K())
    assert math.isclose(iou_all, iou_kept, rel_tol=0.0, abs_tol=1e-12)
    silhouettes = render_silhouettes(pts, behind, _SCALE, _K(), (_H, _W))
    assert silhouettes.shape == (4, _H, _W) and silhouettes.dtype == torch.bool
    assert not bool(silhouettes[1].any())


def test_qc_features_penetration_zero_without_hand() -> None:
    poses = _poses()
    masks = _analytic_sphere_masks(poses[:, :3, 3], _SCALE * _R_CANON, _K())
    assert _call(poses, masks, hand_verts=None, with_sdf=True).penetration_depth_m == 0.0
    verts = poses[:, None, :3, 3].expand(-1, 5, -1).clone()     # deep inside, but no SDF supplied
    assert _call(poses, masks, hand_verts=verts, with_sdf=False).penetration_depth_m == 0.0


def test_qc_features_penetration_depth_is_metric() -> None:
    poses = _poses()
    masks = _analytic_sphere_masks(poses[:, :3, 3], _SCALE * _R_CANON, _K())
    # canonical-frame hand vertices: one 1 cm inside the canonical surface, one outside, one NaN (missing)
    y = torch.tensor([[_R_CANON - 0.010, 0.0, 0.0], [0.0, _R_CANON + 0.02, 0.0], [math.nan] * 3],
                     dtype=torch.float64)
    R, t = poses[:, :3, :3], poses[:, :3, 3]
    verts = _SCALE * torch.einsum("tij,vj->tvi", R, y) + t[:, None, :]
    verts[:2, 0] = t[:2] + torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64)   # frames 0,1: hand away
    pen = _call(poses, masks, hand_verts=verts, with_sdf=True).penetration_depth_m
    assert abs(pen - _SCALE * 0.010) < 1e-3          # metric depth: scale * canonical depth = 15 mm


def test_qc_features_scalar_definitions() -> None:
    T = 4
    stds = torch.tensor([0.002, 0.009, 0.011, 0.05], dtype=torch.float64)
    cov6 = torch.eye(6, dtype=torch.float64).repeat(T, 1, 1) * 1e-3
    cov6[:, :3, :3] = torch.eye(3, dtype=torch.float64) * (stds**2 / 3.0)[:, None, None]
    cov7 = torch.eye(7, dtype=torch.float64).repeat(T, 1, 1)
    cov7[:, :6, :6] = cov6
    assert laplace_std_ok_frac(cov6) == 0.5
    assert laplace_std_ok_frac(cov7) == 0.5
    assert laplace_std_ok_frac(cov6, threshold=0.02) == 0.75
    cost = torch.tensor([0.001, 0.002, 0.003, 0.004])
    expected_rms = math.sqrt((1 + 4 + 9 + 16) / 4) * 1e-3 / 0.12
    assert math.isclose(ot_residual_rms_frac(cost, 0.12), expected_rms, rel_tol=1e-6)
    weights = torch.tensor([[1.0, 0.9, 0.5, 0.0], [0.51, 0.2, 0.0, 0.0], [0.7, 0.7, 0.7, 0.7],
                            [0.0, 0.0, 0.0, 0.0]])
    assert track_inlier_ratio(weights) == 7 / 16    # strictly above 0.5; 0.5 itself is not an inlier
    poses = _poses()
    masks = _analytic_sphere_masks(poses[:, :3, 3], _SCALE * _R_CANON, _K())
    qc = _call(poses, masks, covariance=cov6, per_frame_cost=cost, track_weight=weights)
    assert qc.laplace_std_ok_frac == 0.5
    assert math.isclose(qc.ot_residual_rms_frac, expected_rms * 0.12 / (2.0 * _SCALE * _R_CANON), rel_tol=1e-6)
    assert qc.track_inlier_ratio == 7 / 16
    assert qc.to_array().shape == (5,) and qc.to_array()[0] == qc.reprojection_iou
