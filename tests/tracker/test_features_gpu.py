"""Feature-token GPU tests 21-22 (GPU run): CUDA tokens equal CPU tokens; cost and allocator peak at T=100."""

from __future__ import annotations

import logging
import math
import statistics
import time

import pytest
import torch

from fastdo.geometry.lie import se3_exp
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import TermOutput
from fastdo.tracker.clusters import mesh_cluster_ids
from fastdo.tracker.features import build_frame_tokens, track_tokens
from tests.tracker.synthetic_terms import make_terms_fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (GPU run)")

logger = logging.getLogger(__name__)
MIB = 1024.0 ** 2


def _to(t: TermOutput, device: str) -> TermOutput:
    return TermOutput(t.residual.to(device), t.jacobian.to(device), t.weight.to(device))


def test_frame_tokens_cuda_matches_cpu() -> None:
    fx = make_terms_fixture(knn=16)
    assert isinstance(fx.plan, SparsePlan)
    cluster_id = mesh_cluster_ids(fx.Y, 64, 0)
    tok = build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, cluster_id, fx.template_diameter, fx.poses, fx.log_scale)
    track = track_tokens(fx.anchors, fx.tr_raw, fx.template_diameter)
    dev = "cuda"
    plan = SparsePlan(indices=fx.plan.indices.to(dev), values=fx.plan.values.to(dev))
    tok_c = build_frame_tokens(_to(fx.pm, dev), _to(fx.tr_raw, dev), plan, cluster_id.to(dev),
                               fx.template_diameter, fx.poses.to(dev), fx.log_scale.to(dev))
    track_c = track_tokens(fx.anchors.to(dev), _to(fx.tr_raw, dev), fx.template_diameter)
    assert tok_c.device.type == "cuda" and track_c.device.type == "cuda"
    torch.testing.assert_close(tok_c.cpu(), tok, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(track_c.cpu(), track, rtol=1e-9, atol=1e-12)


@pytest.mark.timing
def test_frame_tokens_cuda_cost_at_t100() -> None:
    dev, f64 = "cuda", torch.float64
    T, N, M, J, k = 100, 1024, 2048, 128, 32
    gen = torch.Generator(device=dev).manual_seed(0)
    pm = TermOutput(torch.randn(T, N, 3, generator=gen, device=dev, dtype=f64),
                    torch.randn(T, N, 3, 7, generator=gen, device=dev, dtype=f64),
                    torch.bernoulli(torch.full((T, N), 0.8, device=dev, dtype=f64), generator=gen))
    tr_w = (torch.rand(T, J, generator=gen, device=dev, dtype=f64)
            * torch.bernoulli(torch.full((T, J), 0.5, device=dev, dtype=f64), generator=gen))
    tr_raw = TermOutput(torch.randn(T, J, 2, generator=gen, device=dev, dtype=f64),
                        torch.randn(T, J, 2, 7, generator=gen, device=dev, dtype=f64), tr_w)
    plan = SparsePlan(indices=torch.randint(0, M, (T, N, k), generator=gen, device=dev),
                      values=torch.rand(T, N, k, generator=gen, device=dev, dtype=f64) * 1e-3)
    cluster_id = torch.randint(0, 64, (M,), generator=gen, device=dev)
    xi = 0.1 * torch.randn(T, 6, generator=gen, device=dev, dtype=f64)
    poses = se3_exp(xi)
    poses[:, 2, 3] += 0.6
    log_scale = torch.tensor(math.log(0.12), dtype=f64, device=dev)
    anchors = torch.randn(J, 3, generator=gen, device=dev, dtype=f64)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()

    def call() -> None:
        build_frame_tokens(pm, tr_raw, plan, cluster_id, 1.0, poses, log_scale)
        track_tokens(anchors, tr_raw, 1.0)

    for _ in range(3):
        call()
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    peak_delta = (torch.cuda.max_memory_allocated() - allocated_before) / MIB
    median = statistics.median(times)
    p90 = sorted(times)[math.ceil(0.9 * len(times)) - 1]
    logger.info("d8 cuda T=%d N=%d M=%d J=%d k=%d: median %.2f ms p90 %.2f ms peak_alloc_delta %.1f MiB",
                T, N, M, J, k, median, p90, peak_delta)
    assert median <= 20.0
    assert peak_delta <= 512.0
