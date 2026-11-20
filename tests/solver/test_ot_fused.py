"""Fused CUDA Sinkhorn kernels (`ot_fused`) against the eager expressions of `ot.py`.

The oracle is the eager PyTorch expression itself (broadcast subtract/divide/add, `torch.logsumexp`,
`torch.exp`), on shapes that are not powers of two, with exactly-masked template columns (log-mass -1e30)
and per-frame eps. The fused kernels reorder the row sums, so agreement is to float round-off.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch
from torch import Tensor

from fastdo.solver import ot_fused
from fastdo.solver.api import SolverConfig
from fastdo.solver.ot import _LOG_ZERO, sinkhorn_unbalanced_potentials

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernels are CUDA-only")
# Round-off tolerance relative to max(1, |value|): a float32 sum over ~1e3 terms accumulates ~1e-4 relative.
RTOL = {torch.float32: 2e-4, torch.float64: 1e-11}


def _problem(T: int, N: int, M: int, dtype: torch.dtype, seed: int = 0) -> dict[str, Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b = torch.rand(T, M, dtype=dtype, device="cuda", generator=gen)
    b = torch.where(b < 0.3, torch.zeros_like(b), b)                    # exactly masked template columns
    return {
        "C": torch.rand(T, N, M, dtype=dtype, device="cuda", generator=gen) * 1e-2,
        "f": torch.randn(T, N, dtype=dtype, device="cuda", generator=gen) * 1e-4,
        "g": torch.randn(T, M, dtype=dtype, device="cuda", generator=gen) * 1e-4,
        "log_a": torch.log(torch.rand(T, N, dtype=dtype, device="cuda", generator=gen)),
        "log_b": torch.where(b > 0, torch.log(b.clamp_min(1e-300)), torch.full_like(b, _LOG_ZERO)),
        "eps": torch.rand(T, dtype=torch.float64, device="cuda", generator=gen) * 1e-4 + 1e-5,
    }


def _close(a: Tensor, b: Tensor, rtol: float) -> float:
    return float(((a - b).abs() / b.abs().clamp_min(1.0)).max())


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
# Memory-light (merge-gate rule: a solver unit test stays far below 1 GB on a shared GPU): the largest case is
# 3 x 300 x 1025 = 0.9 M entries (the eager oracle's temporaries peak near 60 MB in float64), yet it still
# covers row blocks that are not powers of two (1025 -> BLOCK 2048 for rows and the plan, 1024 x 97 -> 128) and
# frames of many rows (the plan kernel runs one program per frame and row).
@pytest.mark.parametrize("shape", [(3, 5, 7), (2, 300, 1025), (3, 1024, 97)])
def test_fused_kernels_match_eager_expressions(dtype: torch.dtype, shape: tuple[int, int, int]) -> None:
    x = _problem(*shape, dtype=dtype)
    C, f, g, log_a, log_b = x["C"], x["f"], x["g"], x["log_a"], x["log_b"]
    e3 = x["eps"].to(dtype).view(-1, 1, 1)
    rows = torch.logsumexp((g[:, None, :] - C) / e3 + log_b[:, None, :], dim=2)
    cols = torch.logsumexp((f[:, :, None] - C) / e3 + log_a[:, :, None], dim=1)
    plan = torch.exp((f[:, :, None] + g[:, None, :] - C) / e3 + log_a[:, :, None] + log_b[:, None, :])
    assert _close(ot_fused.lse_rows(C, g, log_b, x["eps"]), rows, RTOL[dtype]) < RTOL[dtype]
    assert _close(ot_fused.lse_cols(C, f, log_a, x["eps"]), cols, RTOL[dtype]) < RTOL[dtype]
    fused_plan = ot_fused.plan_from_potentials(C, f, g, log_a, log_b, x["eps"])
    assert torch.equal(fused_plan == 0, plan == 0)                        # masked columns stay exactly zero
    assert torch.allclose(fused_plan, plan, rtol=RTOL[dtype] * 10, atol=0.0)


@cuda_only
def test_fused_kernels_accept_a_scalar_eps_and_strided_inputs() -> None:
    x = _problem(4, 33, 65, torch.float32, seed=3)
    C = x["C"].transpose(1, 2).contiguous().transpose(1, 2)            # same values, (T, 1, M) strides
    g = x["g"][:, ::1]
    assert C.stride() != x["C"].stride()
    eager = torch.logsumexp((g[:, None, :] - C) / 1e-4 + x["log_b"][:, None, :], dim=2)
    assert _close(ot_fused.lse_rows(C, g, x["log_b"], 1e-4), eager, 2e-4) < 2e-4


@cuda_only
def test_sinkhorn_fused_path_matches_eager_path() -> None:
    """`sinkhorn_unbalanced_potentials` end to end: fused (`fused=True`) vs eager (the default)."""
    x = _problem(6, 200, 300, torch.float32, seed=5)
    a = torch.full((6, 200), 1.0 / 200, device="cuda")
    b = torch.exp(x["log_b"]).clamp_min(0.0)
    b = b / b.sum(1, keepdim=True)
    eps = x["eps"] * 20
    fused = sinkhorn_unbalanced_potentials(x["C"], a, b, eps, 0.01, 1e-3, 50, fused=True)
    eager = sinkhorn_unbalanced_potentials(x["C"], a, b, eps, 0.01, 1e-3, 50)
    for got, want in zip(fused, eager, strict=True):
        assert torch.allclose(got, want, rtol=1e-3, atol=1e-7), (got - want).abs().max()


@cuda_only
def test_fused_path_is_skipped_for_autograd_and_cpu() -> None:
    C = torch.rand(2, 3, 4, device="cuda", requires_grad=True)
    assert not ot_fused.fused_available(C)
    with torch.no_grad():
        assert ot_fused.fused_available(C)
    assert not ot_fused.fused_available(torch.rand(2, 3, 4))


def test_fused_kernels_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU: the solver never enters the fused kernels by default, even where they apply; `fused=True`
    does. The kernels are replaced by recording stand-ins, so this runs without CUDA."""
    calls: list[str] = []

    def rows(C: Tensor, g: Tensor, log_b: Tensor, eps: object) -> Tensor:
        calls.append("rows")
        return torch.zeros(C.shape[0], C.shape[1], dtype=C.dtype)

    def cols(C: Tensor, f: Tensor, log_a: Tensor, eps: object) -> Tensor:
        calls.append("cols")
        return torch.zeros(C.shape[0], C.shape[2], dtype=C.dtype)

    def plan(C: Tensor, f: Tensor, g: Tensor, log_a: Tensor, log_b: Tensor, eps: object) -> Tensor:
        calls.append("plan")
        return torch.zeros_like(C)

    monkeypatch.setattr(ot_fused, "fused_available", lambda *tensors: True)
    monkeypatch.setattr(ot_fused, "lse_rows", rows)
    monkeypatch.setattr(ot_fused, "lse_cols", cols)
    monkeypatch.setattr(ot_fused, "plan_from_potentials", plan)
    C = torch.rand(2, 3, 4, dtype=torch.float64)
    a, b = torch.full((2, 3), 1.0 / 3, dtype=torch.float64), torch.full((2, 4), 0.25, dtype=torch.float64)
    sinkhorn_unbalanced_potentials(C, a, b, 0.1, 1.0, 1.0, 3)
    # The FUNCTION's default is still the reference path; only SolverConfig's default changed.
    assert not calls and SolverConfig().fused_sinkhorn
    sinkhorn_unbalanced_potentials(C, a, b, 0.1, 1.0, 1.0, 3, fused=True)
    assert calls == ["rows", "cols"] * 3 + ["plan"]


def test_importing_the_solver_does_not_import_triton() -> None:
    """CPU: `ot_fused` imports triton at import time, so `ot.py` imports it only inside a Sinkhorn call
    that opts in. In a fresh interpreter, importing `fastdo.solver` and running a default Sinkhorn solve leaves both
    triton and `ot_fused` out of `sys.modules`; the opt-in call afterwards shows the probe sees the lazy import."""
    code = textwrap.dedent("""
        import sys
        import torch
        import fastdo.solver
        from fastdo.solver.ot import sinkhorn_unbalanced_potentials
        C = torch.rand(2, 3, 4, dtype=torch.float64)
        a, b = torch.full((2, 3), 1.0 / 3, dtype=torch.float64), torch.full((2, 4), 0.25, dtype=torch.float64)
        sinkhorn_unbalanced_potentials(C, a, b, 0.1, 1.0, 1.0, 3)
        print("loaded:", sorted(m for m in ("triton", "fastdo.solver.ot_fused") if m in sys.modules))
        sinkhorn_unbalanced_potentials(C, a, b, 0.1, 1.0, 1.0, 3, fused=True)
        print("opt-in loads ot_fused:", "fastdo.solver.ot_fused" in sys.modules)
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.startswith(("loaded:", "opt-in"))]
    assert lines == ["loaded: []", "opt-in loads ot_fused: True"], proc.stdout + proc.stderr
