"""The throughput benchmark's harness (`fastdo.bench.solver_throughput`) on tiny clips, CPU-sized.

The timing claim itself is `tests/solver/test_batch.py::test_throughput_curve` (CUDA, timing-marked); this checks
that the harness builds the documented fixture, reports one row per batch size with the per-level LM iteration
counts of every clip, and rejects a batch size larger than the clip pool.
"""

from __future__ import annotations

import pytest
import torch

from fastdo.bench import solver_throughput
from fastdo.bench.solver_throughput import measure_throughput, synthetic_dexycb_like
from fastdo.solver import batch_eval
from fastdo.solver.api import SolverConfig
from fastdo.solver.memory import BatchPlan, ClipDims, plan_batches


def test_synthetic_dexycb_like_fixture() -> None:
    clips = synthetic_dexycb_like(2, n_frames=3, n_surface=64, image_hw=(48, 64))
    assert len(clips) == 2
    mesh, obs, init, scale = clips[1]
    assert mesh.points.shape == (64, 3) and obs.masks.shape == (3, 48, 64) and init.shape == (3, 4, 4)
    assert scale > 0 and obs.tracks.shape == (3, 64, 2)
    assert not torch.equal(clips[0][2], clips[1][2])                       # distinct seeds per clip


def test_measure_throughput_reports_every_batch_size() -> None:
    clips = synthetic_dexycb_like(2, n_frames=3, n_surface=64, image_hw=(48, 64))
    cfg = SolverConfig(n_obs=32, n_mesh=64, iters_coarse=2, iters_mid=1, iters_fine=2, device="cpu")
    rows = measure_throughput(clips, (1, 2), cfg, repeats=1)
    assert [r.batch_size for r in rows] == [1, 2]
    for r in rows:
        assert not r.error and len(r.walls) == 1 and r.clips_per_second > 0 and r.peak_gb == 0.0
        assert len(r.iterations) == r.batch_size and all(len(levels) == 3 for levels in r.iterations)
        assert r.sub_batches == (r.batch_size,) and r.estimate_gb > 0          # no CUDA budget on CPU: one batch
        assert r.reserved_gb == 0.0 and r.nvml_peak_gb is None and r.nvml_rows == 0 and not r.fused_sinkhorn
        assert not r.card_limited
    with pytest.raises(ValueError, match="needs that many clips"):
        measure_throughput(clips, (3,), cfg)


def test_measure_throughput_reports_a_budget_error_as_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch whose clips cannot fit the CUDA budget becomes a row with the error, not an aborted curve."""
    clips = synthetic_dexycb_like(2, n_frames=3, n_surface=64, image_hw=(48, 64))
    cfg = SolverConfig(n_obs=32, n_mesh=64, iters_coarse=2, iters_mid=1, iters_fine=2, device="cpu")

    def over_budget(dims: list[ClipDims], config: SolverConfig) -> list[BatchPlan]:
        """The harness's planning under a 1 KiB budget; the warm-up solve is left unconstrained."""
        return plan_batches(dims, config, 1 << 10, batch_eval._REFRESH_ELEMENTS)

    monkeypatch.setattr(solver_throughput, "plan_solve", over_budget)
    (row,) = measure_throughput(clips, (2,), cfg, repeats=1)
    assert row.error.startswith("GpuBudgetError: clip 0") and row.clips_per_second == 0.0 and not row.walls
