"""Evaluating a subset of clips (`BatchEvaluate`'s `frames`) equals the matching rows of a full evaluation.

The oracle is the same evaluate callback over every frame, on its own cache that went through the same refreshes;
the two differ only in how many frames share a kernel call (round-off). A subset refresh must also leave the
cached plan, potentials and visibility mask of the clips it does not name bit-identical. Cases cover the dense
coarse plan, the kNN-truncated fine plan, nearest-neighbour correspondences, and refreshes split into chunks.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver import batch_eval
from fastdo.solver.api import SolverConfig, _levels
from fastdo.solver.banded import Priors
from fastdo.solver.batch import pack_clips, prepare_batch
from fastdo.solver.batch_eval import EvalCache, make_evaluate
from fastdo.solver.moments import PlanMoments
from fastdo.solver.ot_sparse import SparsePlan
from tests.solver.test_batch import DEVICE, _clip, _inputs, _unzip

N_OBS, N_MESH = 96, 192


def _cached(cache: EvalCache, rows: Tensor) -> list[Tensor]:
    """The cached plan (dense, sparse indices and values, or plan moments), Sinkhorn potentials (none for
    nearest-neighbour correspondences) and visibility mask at frames `rows`."""
    plan = cache.plan
    assert plan is not None and cache.mask is not None
    if isinstance(plan, PlanMoments):
        parts = [plan.bary, plan.mass] + ([] if plan.normal is None else [plan.normal])
    else:
        parts = [plan.indices, plan.values] if isinstance(plan, SparsePlan) else [plan]
    potentials = cache.potentials if cache.potentials is not None else ()
    return [x.index_select(0, rows).clone() for x in (*parts, *potentials, cache.mask)]


@pytest.mark.parametrize(("correspondence", "level_index", "chunk_frames"),
                         [("ot", 0, None), ("ot", 2, None), ("ot", 0, 4), ("ot", 2, 4), ("nn", 1, None)])
def test_subset_evaluation_matches_full_rows(monkeypatch: pytest.MonkeyPatch, correspondence: str,
                                             level_index: int, chunk_frames: int | None) -> None:
    if chunk_frames is not None:
        monkeypatch.setattr(batch_eval, "_REFRESH_ELEMENTS", chunk_frames * N_OBS * N_MESH)
    cfg = SolverConfig(n_obs=N_OBS, n_mesh=N_MESH, device=DEVICE, sinkhorn_float32=False, correspondence=correspondence)
    db = prepare_batch(pack_clips(*_unzip([_inputs(_clip(T, N_MESH, seed=130 + T), seed=T) for T in (5, 6, 7)])), cfg)
    level = _levels(cfg)[level_index][0]
    gen = torch.Generator().manual_seed(level_index)
    xi = (0.01 * torch.randn(db.layout.n_frames, 6, generator=gen, dtype=torch.float64)).to(DEVICE)
    full_cache, sub_cache = EvalCache(), EvalCache()
    full = make_evaluate(db, cfg, level, full_cache)
    sub = make_evaluate(db, cfg, level, sub_cache)
    full(db.poses0, db.log_scale0, True)
    sub(db.poses0, db.log_scale0, True)
    frames = db.layout.frames_of([0, 2])
    kept = db.layout.clip_frames[1]
    assert frames is not None and db.layout.frames_of([0, 1, 2]) is None
    assert frames.tolist() == list(range(5)) + list(range(11, 18))
    untouched = _cached(sub_cache, kept)
    ls = db.log_scale0 + 0.01
    moved = se3_exp(xi) @ db.poses0
    for refresh, poses in ((True, moved), (False, se3_exp(0.5 * xi) @ moved)):
        terms_full, priors_full = full(poses, ls, refresh)
        terms_sub, priors_sub = sub(poses, ls, refresh, frames)
        assert len(terms_sub) == len(terms_full) == 2
        for got, want in zip(terms_sub, terms_full, strict=True):
            assert got.residual.shape[0] == frames.shape[0]
            for name in ("residual", "jacobian", "weight"):
                a, b = getattr(got, name), getattr(want, name).index_select(0, frames)
                assert torch.allclose(a, b, rtol=1e-7, atol=1e-12), (refresh, name, (a - b).abs().max())
        assert isinstance(priors_sub, Priors) and isinstance(priors_full, Priors)
        for pair in ((priors_sub.smooth, priors_full.smooth), (priors_sub.velocity, priors_full.velocity)):
            assert (pair[0] is None) == (pair[1] is None)
            for a, b in zip(pair[0] or (), pair[1] or (), strict=True):   # priors always cover every frame
                assert torch.equal(a, b)
    for a, b in zip(_cached(sub_cache, kept), untouched, strict=True):          # clip 1 kept its poses0 refresh
        assert torch.equal(a, b)
    for a, b in zip(_cached(sub_cache, frames), _cached(full_cache, frames), strict=True):
        assert torch.allclose(a.double(), b.double(), rtol=1e-7, atol=1e-12)


def test_subset_refresh_needs_a_full_refresh_first() -> None:
    cfg = SolverConfig(n_obs=N_OBS, n_mesh=N_MESH, device=DEVICE)
    db = prepare_batch(pack_clips(*_unzip([_inputs(_clip(T, N_MESH, seed=150 + T), seed=T) for T in (3, 4)])), cfg)
    evaluate = make_evaluate(db, cfg, _levels(cfg)[0][0], EvalCache())
    with pytest.raises(ValueError, match="previous refresh of every frame"):
        evaluate(db.poses0, db.log_scale0, True, db.layout.frames_of([1]))
