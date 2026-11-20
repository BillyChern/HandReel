"""Feature-token CPU tests 16-20: template k-means cluster ids and plan column mass pooled per cluster."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.synthetic import SyntheticConfig, make_synthetic_clip
from fastdo.tracker.clusters import mesh_cluster_ids, mesh_cluster_mass
from tests.tracker.synthetic_terms import make_terms_fixture

F64 = torch.float64


def _wcss(points: Tensor, ids: Tensor, n_clusters: int) -> float:
    """Within-cluster sum of squares with member means (empty clusters contribute nothing)."""
    total = 0.0
    for c in range(n_clusters):
        members = points[ids == c]
        if members.shape[0] > 0:
            total += float(((members - members.mean(0)) ** 2).sum())
    return total


def _plan_total(plan: Tensor | SparsePlan) -> Tensor:
    return plan.values.sum((1, 2)) if isinstance(plan, SparsePlan) else plan.sum((1, 2))


def test_mesh_cluster_ids_deterministic_and_covering() -> None:
    Y = make_synthetic_clip(SyntheticConfig(n_frames=2, image_hw=(120, 160), focal=150.0, shape="torus",
                                            n_surface=2048), 0).surface_points.double()
    ids = mesh_cluster_ids(Y, 64, seed=0)
    assert torch.equal(ids, mesh_cluster_ids(Y, 64, seed=0))
    assert ids.dtype == torch.int64 and ids.shape == (2048,)
    assert int(ids.min()) >= 0 and int(ids.max()) < 64
    assert int(torch.bincount(ids, minlength=64).min()) >= 1
    assert torch.equal(mesh_cluster_ids(Y.float(), 64, seed=0), ids)
    ids0 = mesh_cluster_ids(Y, 64, seed=0, n_iter=0)
    assert _wcss(Y, ids, 64) <= _wcss(Y, ids0, 64) + 1e-12


def test_mesh_cluster_ids_rejects_bad_inputs() -> None:
    pts = torch.rand(128, 3, dtype=F64, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        mesh_cluster_ids(pts[:63], 64)
    with pytest.raises(ValueError):
        mesh_cluster_ids(pts[:, :2], 8)
    with pytest.raises(ValueError):
        mesh_cluster_ids(pts, 0)
    with pytest.raises(ValueError):
        mesh_cluster_ids(pts, 8, n_iter=-1)


@pytest.mark.parametrize("knn", [None, 16], ids=["dense", "sparse"])
def test_mesh_cluster_mass_conserves_total_mass(knn: int | None) -> None:
    fx = make_terms_fixture(knn=knn)
    cluster_id = mesh_cluster_ids(fx.Y, 64, 0)
    mass = mesh_cluster_mass(fx.plan, cluster_id)
    assert mass.shape == (6, 64) and mass.dtype == F64
    assert bool((mass >= 0).all())
    torch.testing.assert_close(mass.sum(1), _plan_total(fx.plan), rtol=1e-10, atol=1e-15)
    torch.testing.assert_close(mesh_cluster_mass(fx.plan, cluster_id.expand(6, 256)), mass, rtol=0.0, atol=0.0)
    ids_pad = torch.cat([cluster_id, cluster_id.new_zeros(8)])
    plan_pad: Tensor | SparsePlan = (fx.plan if isinstance(fx.plan, SparsePlan)
                                     else torch.cat([fx.plan, fx.plan.new_zeros(6, 96, 8)], -1))
    for ids in (ids_pad, ids_pad.expand(6, 264)):
        torch.testing.assert_close(mesh_cluster_mass(plan_pad, ids), mass, rtol=1e-10, atol=1e-15)


def test_mesh_cluster_mass_sparse_equals_dense() -> None:
    fx = make_terms_fixture()
    assert isinstance(fx.plan, Tensor)
    cluster_id = mesh_cluster_ids(fx.Y, 64, 0)
    sparse = SparsePlan(indices=torch.arange(256).expand(6, 96, 256).contiguous(), values=fx.plan)
    torch.testing.assert_close(mesh_cluster_mass(sparse, cluster_id), mesh_cluster_mass(fx.plan, cluster_id),
                               rtol=1e-10, atol=1e-15)


def test_mesh_cluster_mass_rejects_wrong_id_length() -> None:
    plan = torch.rand(2, 5, 256, dtype=F64, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        mesh_cluster_mass(plan, torch.zeros(255, dtype=torch.int64))
