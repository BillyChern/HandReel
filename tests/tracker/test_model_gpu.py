"""Tracker-model GPU tests G1-G3 (GPU run): CUDA equals CPU in float64; forward cost at T=100; backward memory."""

from __future__ import annotations

import copy
import dataclasses
import logging
import statistics
import time

import pytest
import torch

from fastdo.tracker.config import TrackerConfig
from fastdo.tracker.heads import TrackerOutput
from fastdo.tracker.model import AmortizedTracker
from fastdo.tracker.pack import TrackerInputs, build_tracker_inputs
from tests.tracker.tracker_fixtures import (
    SMALL_CONFIG,
    random_tracker_inputs,
    randomize_parameters,
    tracker_loss,
    two_clip_args,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (GPU run)")

logger = logging.getLogger(__name__)
MIB = 1024.0 ** 2
SIZES = "B=1 W=100 J=128 N=1024 C=64"


def _to(inputs: TrackerInputs, device: str) -> TrackerInputs:
    return TrackerInputs(*(getattr(inputs, f.name).to(device) for f in dataclasses.fields(TrackerInputs)))


def _t100_inputs(device: str) -> TrackerInputs:
    return _to(random_tracker_inputs(1, (100,), 128, 1024, 64, seed=0, dtype=torch.float32), device)


def _median_p90(values: list[float]) -> tuple[float, float]:
    return statistics.median(values), statistics.quantiles(values, n=10)[-1]


@pytest.mark.parametrize("config", ["small", "default"])
def test_tracker_cuda_matches_cpu_float64(config: str) -> None:
    cfg = SMALL_CONFIG if config == "small" else TrackerConfig()
    model = AmortizedTracker(cfg).double()
    randomize_parameters(model, std=0.05, seed=0)
    inputs = build_tracker_inputs(**two_clip_args(6))
    with torch.no_grad():
        ref = model(inputs)
        out = copy.deepcopy(model).to("cuda")(_to(inputs, "cuda"))
    for name in TrackerOutput._fields:
        torch.testing.assert_close(getattr(out, name).cpu(), getattr(ref, name), rtol=1e-9, atol=1e-12)


def _kernel_count_and_cpu_ms(model: AmortizedTracker, inputs: TrackerInputs) -> tuple[int, float]:
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.no_grad(), torch.profiler.profile(activities=acts) as prof:
        model(inputs)
        torch.cuda.synchronize()
    events = prof.key_averages()
    launches = [e.count for e in events if e.key == "cudaLaunchKernel"]
    if launches:
        kernels = int(launches[0])
    else:
        kernels = int(sum(e.count for e in events if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA))
    cpu_ms = sum(e.self_cpu_time_total for e in events) / 1e3
    return kernels, cpu_ms


@pytest.mark.timing
def test_tracker_cuda_forward_cost_at_t100() -> None:
    torch.manual_seed(0)
    model = AmortizedTracker().to("cuda").eval()
    inputs = _t100_inputs("cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    times, host = [], []
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
        torch.cuda.synchronize()
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eager = model(inputs)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            host.append((t1 - t0) * 1e3)
    eager_peak = (torch.cuda.max_memory_allocated() - before) / MIB
    kernels, cpu_ms = _kernel_count_and_cpu_ms(model, inputs)
    med, p90 = _median_p90(times)
    logger.info("d9 cuda fwd %s: median %.2f ms p90 %.2f ms host median %.2f ms kernels %d cpu_total %.2f ms "
                "peak_alloc_delta %.1f MiB", SIZES, med, p90, statistics.median(host), kernels, cpu_ms, eager_peak)
    # CUDA-graph variant: static inputs, side-stream warm-up, capture, replay.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_g = torch.cuda.memory_allocated()
    side = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.stream(side):
            for _ in range(3):
                model(inputs)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            static_out = model(inputs)
    graph.replay()
    torch.cuda.synchronize()
    for name in TrackerOutput._fields:
        diff = (getattr(static_out, name) - getattr(eager, name)).abs().max()
        assert float(diff) <= 1e-5, name
    for _ in range(3):
        graph.replay()
    gtimes = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        gtimes.append((time.perf_counter() - t0) * 1e3)
    graph_peak = (torch.cuda.max_memory_allocated() - before_g) / MIB
    gmed, gp90 = _median_p90(gtimes)
    logger.info("d9 cuda graph fwd %s: median %.2f ms p90 %.2f ms peak_alloc_delta %.1f MiB", SIZES, gmed, gp90,
                graph_peak)
    assert gmed <= 10.0
    assert eager_peak <= 512.0 and graph_peak <= 1024.0


def test_tracker_cuda_backward_memory_at_t100() -> None:
    torch.manual_seed(0)
    model = AmortizedTracker().to("cuda")
    inputs = _t100_inputs("cuda")
    tracker_loss(model(inputs)).backward()                     # warm-up
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    tracker_loss(model(inputs)).backward()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    peak = (torch.cuda.max_memory_allocated() - before) / MIB
    logger.info("d9 cuda fwd+bwd %s: %.1f ms peak_alloc_delta %.1f MiB", SIZES, ms, peak)
    for name, p in model.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
    assert peak <= 1536.0
