"""The batched (multi-clip) banded system equals B separate systems and its dense form.

The oracle for each clip is the single-clip `BlockTridiagonal` path (layout-free, scalar scale) and, for
the whole batch, `torch.linalg.solve`/`torch.linalg.inv` on `densify`'s (6T+B)-dimensional matrix, whose
per-clip scale columns are placed by the layout. Clip lengths mix odd/even and 1-/2-frame clips, so the
identity-padded frame pairs and the empty-velocity case are exercised.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from fastdo.bench.solver_throughput import synthetic_dexycb_like
from fastdo.solver import _block_thomas, api, banded
from fastdo.solver._block_thomas import cholesky_solve, prefer_cusolver
from fastdo.solver.api import SolverConfig, solve_trajectories, solve_trajectory
from fastdo.solver.banded import (
    ClipLayout,
    Priors,
    assemble_block_tridiagonal,
    densify,
    marginal_covariances,
    solve_block_tridiagonal,
)
from fastdo.solver.terms import TermOutput

DT = torch.float64
LENGTHS = (1, 4, 7, 2, 5)


def _random_clip(T: int, velocity: bool, gen: torch.Generator) -> tuple[TermOutput, Priors]:
    term = TermOutput(torch.randn(T, 5, 2, dtype=DT, generator=gen), torch.randn(T, 5, 2, 7, dtype=DT, generator=gen),
                      torch.rand(T, 5, dtype=DT, generator=gen))
    smooth = None
    if T > 1:
        smooth = tuple(torch.randn(T - 1, *shape, dtype=DT, generator=gen) for shape in ((6,), (6, 6), (6, 6)))
    vel = None
    if velocity and T > 2:
        vel = tuple(torch.randn(T - 2, *shape, dtype=DT, generator=gen) for shape in ((6,), (6, 6), (6, 6), (6, 6)))
    return term, Priors(smooth=smooth, velocity=vel)  # type: ignore[arg-type]


def _concat(clips: list[tuple[TermOutput, Priors]], velocity: bool) -> tuple[TermOutput, Priors]:
    """Concatenate per-clip terms/priors; prior rows straddling two clips are zero, as `batch_priors` makes them."""
    T = sum(LENGTHS)
    term = TermOutput(*(torch.cat([getattr(c[0], f) for c in clips]) for f in ("residual", "jacobian", "weight")))
    smooth = [torch.zeros(T - 1, *s, dtype=DT) for s in ((6,), (6, 6), (6, 6))]
    vel = [torch.zeros(T - 2, *s, dtype=DT) for s in ((6,), (6, 6), (6, 6), (6, 6))]
    off = 0
    for (_, pri), n in zip(clips, LENGTHS, strict=True):
        if pri.smooth is not None:
            for dst, src in zip(smooth, pri.smooth, strict=True):
                dst[off:off + n - 1] = src
        if pri.velocity is not None:
            for dst, src in zip(vel, pri.velocity, strict=True):
                dst[off:off + n - 2] = src
        off += n
    return term, Priors(smooth=tuple(smooth), velocity=tuple(vel) if velocity else None)  # type: ignore[arg-type]


def _damped_dense(H: Tensor, lam_rows: Tensor, floor: float) -> Tensor:
    return H + torch.diag(lam_rows * torch.diagonal(H)) + floor * torch.eye(H.shape[0], dtype=DT)


@pytest.mark.parametrize("velocity", [False, True])
def test_batched_banded_solve_matches_per_clip_and_dense(velocity: bool) -> None:
    gen = torch.Generator().manual_seed(20)
    clips = [_random_clip(n, velocity, gen) for n in LENGTHS]
    term, priors = _concat(clips, velocity)
    T, B = sum(LENGTHS), len(LENGTHS)
    layout = ClipLayout.build(LENGTHS, "cpu")
    sys = assemble_block_tridiagonal([term], priors, T, layout)
    assert sys.h_ss.shape == (B,) and (sys.U2 is not None) == velocity
    lam = torch.tensor([0.1, 0.3, 1e-3, 1.0, 0.05], dtype=DT)
    delta_pose, delta_sigma = solve_block_tridiagonal(sys, lam)
    cov = marginal_covariances(sys)
    assert delta_pose.shape == (T, 6) and delta_sigma.shape == (B,) and cov.shape == (T, 6, 6)

    for b, sl in enumerate(layout.frame_slices()):                    # per-clip oracle
        clip_sys = assemble_block_tridiagonal([clips[b][0]], clips[b][1], LENGTHS[b])
        dp, ds = solve_block_tridiagonal(clip_sys, float(lam[b]))
        assert torch.allclose(delta_pose[sl], dp, rtol=1e-10, atol=1e-10), (b, (delta_pose[sl] - dp).abs().max())
        assert torch.allclose(delta_sigma[b], ds, rtol=1e-10, atol=1e-10), b
        assert torch.allclose(cov[sl], marginal_covariances(clip_sys), rtol=1e-10, atol=1e-10), b

    H, g = densify(sys)                                                 # dense oracle, per-clip damping
    clip_of = torch.repeat_interleave(torch.arange(B), torch.tensor(LENGTHS))
    lam_rows = torch.cat([lam[clip_of].repeat_interleave(6), lam])
    dense = torch.linalg.solve(_damped_dense(H, lam_rows, 1e-9), -g)
    assert torch.allclose(delta_pose.reshape(-1), dense[:6 * T], rtol=1e-8, atol=1e-8)
    assert torch.allclose(delta_sigma, dense[6 * T:], rtol=1e-8, atol=1e-8)
    inv = torch.linalg.inv(_damped_dense(H, torch.zeros_like(lam_rows), 1e-9))
    blocks = torch.stack([inv[6 * k:6 * k + 6, 6 * k:6 * k + 6] for k in range(T)])
    assert torch.allclose(cov, blocks, rtol=1e-8, atol=1e-8)


def test_clip_layout_indexes_frames_pairs_and_padding() -> None:
    layout = ClipLayout.build((3, 2, 4), "cpu")
    assert layout.width == 4 and layout.n_clips == 3 and layout.n_frames == 9
    assert layout.clip_index.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert layout.pair_rows.tolist() == [0, 1, 3, 5, 6, 7]               # rows 2 and 4 straddle clips
    assert layout.triple_rows.tolist() == [0, 5, 6]
    assert layout.slot.tolist() == [0, 1, 2, 4, 5, 8, 9, 10, 11]
    assert layout.frame_slices() == (slice(0, 3), slice(3, 5), slice(5, 9))
    single = ClipLayout.build((5,), "cpu")
    assert single.width == 6 and single.pair_rows.tolist() == [0, 1, 2, 3] and single.u2_slot.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="at least one frame"):
        ClipLayout.build((3, 0), "cpu")


# Operators whose CUDA kernel is chosen by torch's cuSOLVER/MAGMA heuristic, or that solve or factor a matrix.
_LINALG = re.compile(r"linalg|cholesky|solve|inverse|(^|_)lu(_|$)|eig|svd|(^|_)qr|geqrf|ormqr|householder")


class _LinalgOps(TorchDispatchMode):
    """Records the linear-algebra operators dispatched while active, with the preferred CUDA backend at each."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        name = func._schema.name.removeprefix("aten::")
        if _LINALG.search(name):
            self.ops.add(name)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("velocity", [False, True])
def test_banded_solve_avoids_magma_routed_operators(velocity: bool) -> None:
    """The LM loop's banded solve dispatches only `linalg_cholesky_ex` and `linalg_solve_triangular`.
    It never dispatches `cholesky_solve`, which torch's default heuristic sends to MAGMA for batched CUDA inputs,
    nor any other solver. The triangular pair equals `torch.cholesky_solve` to round-off (bitwise on this host's
    LAPACK)."""
    gen = torch.Generator().manual_seed(21)
    clips = [_random_clip(n, velocity, gen) for n in LENGTHS]
    term, priors = _concat(clips, velocity)
    sys = assemble_block_tridiagonal([term], priors, sum(LENGTHS), ClipLayout.build(LENGTHS, "cpu"))
    with _LinalgOps() as seen:
        solve_block_tridiagonal(sys, torch.full((len(LENGTHS),), 0.1, dtype=DT))
    assert seen.ops == {"linalg_cholesky_ex", "linalg_solve_triangular"}, seen.ops
    A = torch.randn(4, 3, 12, 12, dtype=DT, generator=gen)
    L = torch.linalg.cholesky(A @ A.mT + 48.0 * torch.eye(12, dtype=DT))
    rhs = torch.randn(4, 3, 12, 2, dtype=DT, generator=gen)
    torch.testing.assert_close(cholesky_solve(L, rhs), torch.cholesky_solve(rhs, L), rtol=1e-12, atol=1e-14)


def _backend_setter_calls(monkeypatch: pytest.MonkeyPatch) -> list[str | torch._C._LinalgBackend]:
    """Spy on `torch.backends.cuda.preferred_linalg_library`: records every call that sets a backend (reads pass)."""
    sets: list[str | torch._C._LinalgBackend] = []
    real = torch.backends.cuda.preferred_linalg_library

    def spy(backend: str | torch._C._LinalgBackend | None = None) -> torch._C._LinalgBackend:
        if backend is not None:
            sets.append(backend)
        return real(backend)

    monkeypatch.setattr(torch.backends.cuda, "preferred_linalg_library", spy)
    return sets


def test_prefer_cusolver_pins_cuda_only_shares_overlaps_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin sets cuSOLVER for a CUDA device only and restores the caller's preference on exit, also
    when the block raises. Two solves that overlap on two threads share one pin: one set and one restore, and the
    solve that finishes first does not restore the flag under the other. Setting the flag needs no GPU."""
    sets = _backend_setter_calls(monkeypatch)
    before = torch.backends.cuda.preferred_linalg_library()
    with prefer_cusolver(torch.device("cpu")):
        assert torch.backends.cuda.preferred_linalg_library() == before
    assert sets == []
    with pytest.raises(RuntimeError, match="inside"), prefer_cusolver(torch.device("cuda")):
        assert torch.backends.cuda.preferred_linalg_library() == torch._C._LinalgBackend.Cusolver
        raise RuntimeError("inside")
    assert torch.backends.cuda.preferred_linalg_library() == before and sets == ["cusolver", before]

    sets.clear()
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()
    seen: list[torch._C._LinalgBackend] = []

    def first() -> None:
        with prefer_cusolver("cuda"):
            first_in.set()
            second_in.wait(30)
        first_out.set()

    def second() -> None:
        first_in.wait(30)
        with prefer_cusolver("cuda"):
            second_in.set()
            first_out.wait(30)
            seen.append(torch.backends.cuda.preferred_linalg_library())

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert seen == [torch._C._LinalgBackend.Cusolver], seen
    assert torch.backends.cuda.preferred_linalg_library() == before and sets == ["cusolver", before]


def test_solve_sets_the_backend_once_per_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """`solve_trajectories` sets the backend flag once at its entry and restores it once at its exit,
    however many LM iterations and banded factorizations the solve runs, and restores it when the solve raises.
    `_block_thomas._is_cuda` is patched so a CPU solve takes the CUDA branch; no factorization may set the flag."""
    sets = _backend_setter_calls(monkeypatch)
    monkeypatch.setattr(_block_thomas, "_is_cuda", lambda device: True)
    before = torch.backends.cuda.preferred_linalg_library()
    factorizations: list[int] = []
    real_forward = banded.forward_schur

    def counting_forward(D: Tensor, U: Tensor, lam: Tensor, floor: float) -> tuple[Tensor, Tensor, Tensor]:
        n_sets = len(sets)
        out = real_forward(D, U, lam, floor)
        assert len(sets) == n_sets, "forward_schur set the backend flag"
        factorizations.append(n_sets)
        return out

    monkeypatch.setattr(banded, "forward_schur", counting_forward)
    clips = synthetic_dexycb_like(2, n_frames=3, n_surface=64, image_hw=(48, 64))
    meshes, obs, inits, scales = ([c[0] for c in clips], [c[1] for c in clips], [c[2] for c in clips],
                                  [c[3] for c in clips])
    counts: list[int] = []
    for iters in (1, 4):
        sets.clear()
        factorizations.clear()
        cfg = SolverConfig(n_obs=32, n_mesh=64, iters_coarse=iters, iters_mid=iters, iters_fine=iters, device="cpu")
        solve_trajectories(meshes, obs, inits, scales, cfg)
        assert sets == ["cusolver", before], (iters, sets)
        assert torch.backends.cuda.preferred_linalg_library() == before
        counts.append(len(factorizations))
    assert counts[1] > counts[0] >= 4, counts        # one per LM iteration over 3 levels, plus the covariance pass

    sets.clear()
    inside: list[torch._C._LinalgBackend] = []

    def failing_batch(*args: Any) -> list[api.TrajectorySolution]:
        inside.append(torch.backends.cuda.preferred_linalg_library())
        raise RuntimeError("solve failed")

    monkeypatch.setattr(api, "_solve_batch", failing_batch)
    with pytest.raises(RuntimeError, match="solve failed"):
        solve_trajectory(meshes[0], obs[0], inits[0], scales[0], cfg)
    assert inside == [torch._C._LinalgBackend.Cusolver] and sets == ["cusolver", before]
    assert torch.backends.cuda.preferred_linalg_library() == before
