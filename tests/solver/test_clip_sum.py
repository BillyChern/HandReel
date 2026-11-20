"""Per-clip sums in the LM must be deterministic -- CUDA index_add sums floats in a different order each run."""
from __future__ import annotations

import pytest
import torch

from fastdo.solver.banded import clip_sum


def test_clip_sum_equals_index_add_on_cpu() -> None:
    g = torch.Generator().manual_seed(0)
    values = torch.randn(1000, generator=g, dtype=torch.float64)
    index = torch.randint(0, 7, (1000,), generator=g)
    want = torch.zeros(7, dtype=torch.float64).index_add(0, index, values)
    assert torch.allclose(clip_sum(values, index, 7), want, rtol=0, atol=1e-12)


def test_an_infinite_frame_does_not_poison_other_clips() -> None:
    values = torch.tensor([1.0, float("inf"), 2.0, 3.0], dtype=torch.float64)
    index = torch.tensor([0, 1, 2, 2])
    out = clip_sum(values, index, 4)
    assert out.tolist() == [1.0, float("inf"), 5.0, 0.0]     # a one-hot matmul would give NaN everywhere


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_clip_sum_is_bit_identical_across_runs_on_cuda() -> None:
    g = torch.Generator().manual_seed(1)
    values = (torch.randn(200_000, generator=g, dtype=torch.float64) * 1e3).cuda()
    index = torch.randint(0, 16, (200_000,), generator=g).cuda()
    first = clip_sum(values, index, 16)
    assert all(torch.equal(clip_sum(values, index, 16), first) for _ in range(20))


def _scatter_reference(values: torch.Tensor, idx: torch.Tensor, m: int) -> torch.Tensor:
    return torch.zeros(values.shape[0], m, dtype=values.dtype, device=values.device).scatter_add_(1, idx, values)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"))])
def test_segment_sum_matches_scatter_add_including_empty_segments(device: str) -> None:
    from fastdo.solver.ot_sparse import _segment_plan, _segment_sum

    g = torch.Generator().manual_seed(2)
    idx = torch.randint(0, 40, (3, 500), generator=g).to(device)     # segments 40..49 are never touched
    values = torch.rand(3, 500, generator=g, dtype=torch.float64).to(device)
    got = _segment_sum(values, _segment_plan(idx, 50), 50)
    assert torch.allclose(got, _scatter_reference(values, idx, 50), rtol=0, atol=1e-12)
    assert (got[:, 40:] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_segment_sum_is_bit_identical_across_runs_on_cuda() -> None:
    from fastdo.solver.ot_sparse import _segment_plan, _segment_sum

    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, 2048, (25, 1024 * 32), generator=g).cuda()
    values = torch.rand(25, 1024 * 32, generator=g).cuda()
    plan = _segment_plan(idx, 2048)
    first = _segment_sum(values, plan, 2048)
    assert all(torch.equal(_segment_sum(values, plan, 2048), first) for _ in range(20))
