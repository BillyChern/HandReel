"""Template mesh clusters for the tracker's column-mass features.

`mesh_cluster_ids` runs once per template (host reads allowed; the caller caches the result). `mesh_cluster_mass`
runs inside device-side loops: it never reads back to the host and has no data-dependent Python branch.

Batched clips (padded templates): pad each clip's `(M_b,)` ids to the batch width `M` with id 0
(`torch.cat([ids_b, ids_b.new_zeros(M - M_b)])`, valid because `batch._pad_stack` pads at the end), stack to `(B,M)`
and gather per frame with `ids_padded[db.layout.clip_index]` -> `(T,M)`. Padded columns carry zero plan mass, so id 0
adds nothing. Never pass an unpadded `(M_b,)` vector: a kNN plan's `indices` may name padded columns.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from fastdo.solver.ot_sparse import SparsePlan

__all__ = ["mesh_cluster_ids", "mesh_cluster_mass"]

logger = logging.getLogger(__name__)

_ASSIGN_CHUNK = 8192   # template rows per distance chunk in the Lloyd assignment (bounds the (M,k,3) transient)


def _assign(points: Tensor, centres: Tensor) -> Tensor:
    """(M,) index of the nearest centre by squared distance; ties go to the lowest index (argmin's first hit)."""
    parts = [((points[s:s + _ASSIGN_CHUNK, None, :] - centres[None, :, :]) ** 2).sum(-1).argmin(1)
             for s in range(0, points.shape[0], _ASSIGN_CHUNK)]
    return torch.cat(parts)


def _update(points: Tensor, ids: Tensor, centres: Tensor) -> Tensor:
    """Member means; an empty cluster keeps its previous centre."""
    sums = torch.zeros_like(centres).index_add_(0, ids, points)
    counts = torch.bincount(ids, minlength=centres.shape[0]).to(points.dtype)[:, None]
    return torch.where(counts > 0, sums / counts.clamp_min(1.0), centres)


def _farthest_point_sampling(points: Tensor, n_clusters: int, start: int) -> Tensor:
    """(n_clusters,3) greedy farthest-point centres starting from `points[start]`."""
    chosen = [start]
    d2 = ((points - points[start]) ** 2).sum(-1)
    for _ in range(1, n_clusters):
        nxt = int(torch.argmax(d2))
        chosen.append(nxt)
        d2 = torch.minimum(d2, ((points - points[nxt]) ** 2).sum(-1))
    return points[chosen].clone()


def mesh_cluster_ids(points: Tensor, n_clusters: int = 64, seed: int = 0, n_iter: int = 25) -> Tensor:
    """(M,) int64 Lloyd k-means labels on the template samples the plan's columns index.

    Pass exactly those samples: for clip b of a solver batch, `DeviceBatch.Y[k][DeviceBatch.Y_valid[k]]` (M_b,3)
    for any frame k of the clip. Runs on a CPU float64 copy: farthest-point initialization from a seeded random
    start, `n_iter` Lloyd iterations (assign, then member means; an empty cluster keeps its centre), then a final
    assignment (`n_iter=0` returns the farthest-point assignment). Returns int64 ids on `points.device`.
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (M,3), got {tuple(points.shape)}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    M = points.shape[0]
    if M < n_clusters:
        raise ValueError(f"need at least n_clusters={n_clusters} points, got {M}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter}")
    pts = points.detach().to(device="cpu", dtype=torch.float64)
    start = int(torch.randint(M, (1,), generator=torch.Generator().manual_seed(seed)))
    centres = _farthest_point_sampling(pts, n_clusters, start)
    for _ in range(n_iter):
        centres = _update(pts, _assign(pts, centres), centres)
    ids = _assign(pts, centres)
    if logger.isEnabledFor(logging.DEBUG):
        sizes = torch.bincount(ids, minlength=n_clusters)
        logger.debug("mesh_cluster_ids: M=%d n_clusters=%d n_iter=%d cluster size min %d max %d", M, n_clusters,
                     n_iter, int(sizes.min()), int(sizes.max()))
    return ids.to(points.device)


def mesh_cluster_mass(plan: Tensor | SparsePlan, cluster_id: Tensor, n_clusters: int = 64) -> Tensor:
    """(T,n_clusters) float64 plan column mass pooled per cluster; each row sums to the frame's plan mass.

    `plan` is dense (T,N,M) or a kNN `SparsePlan`; `cluster_id` is (M,) shared by every frame or (T,M) per frame,
    padded to the plan's width M (module docstring). A dense plan's width is checked; a sparse plan's cannot be
    without a host read.
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if cluster_id.dim() not in (1, 2):
        raise ValueError(f"cluster_id must be (M,) or (T,M), got {tuple(cluster_id.shape)}")
    values = plan.values if isinstance(plan, SparsePlan) else plan
    if values.dim() != 3:
        raise ValueError(f"plan must be (T,N,M) or a SparsePlan with (T,N,k) values, got {tuple(values.shape)}")
    T = values.shape[0]
    if cluster_id.dim() == 2 and cluster_id.shape[0] != T:
        raise ValueError(f"per-frame cluster_id has {cluster_id.shape[0]} frames, the plan {T}")
    out = torch.zeros(T, n_clusters, dtype=torch.float64, device=values.device)
    if isinstance(plan, SparsePlan):
        if cluster_id.dim() == 1:
            ids = cluster_id[plan.indices]
        else:
            ids = torch.gather(cluster_id, 1, plan.indices.flatten(1))
        return out.scatter_add_(1, ids.flatten(1), plan.values.to(torch.float64).flatten(1))
    if cluster_id.shape[-1] != plan.shape[-1]:
        raise ValueError(f"cluster_id has {cluster_id.shape[-1]} columns, the dense plan {plan.shape[-1]}")
    col = plan.to(torch.float64).sum(1)                                   # (T,M) column mass
    if cluster_id.dim() == 1:
        return out.index_add_(1, cluster_id, col)
    return out.scatter_add_(1, cluster_id, col)
