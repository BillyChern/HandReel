"""The sparse visibility z-buffer (`SolverConfig.sparse_zbuffer`) is bit-identical to the dense one."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from fastdo.geometry.lie import se3_exp
from fastdo.solver.api import SolverConfig, solve_trajectories
from fastdo.solver.memory import ClipDims, batch_peak_bytes
from fastdo.solver.visibility import _sparse_zbuffer_at, visible_samples
from tests.solver.test_batch import _clip, _inputs, _unzip

DT = torch.float64


def _dense_at(lin: torch.Tensor, z: torch.Tensor, size: int) -> torch.Tensor:
    buf = torch.full((size,), float("inf"), dtype=z.dtype)
    buf.scatter_reduce_(0, lin, z, reduce="amin")
    return buf[lin]


@pytest.mark.parametrize("seed", range(6))
def test_sparse_zbuffer_at_equals_the_dense_buffer_with_heavy_collisions(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    n, size = 5000, [7, 300, 100_000][seed % 3]                  # few pixels -> most entries collide
    lin = torch.randint(0, size, (n,), generator=g)
    z = torch.rand(n, generator=g, dtype=DT)
    z[torch.rand(n, generator=g) < 0.3] = float("inf")           # non-candidates
    z[:10] = z[10:20]                                             # exact ties
    assert torch.equal(_sparse_zbuffer_at(lin, z), _dense_at(lin, z, size))


def _scene(seed: int, T: int = 5, M: int = 700, hw: tuple[int, int] = (96, 128)):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(M, 3, generator=g, dtype=DT) * 0.05
    nrm = torch.nn.functional.normalize(torch.randn(M, 3, generator=g, dtype=DT), dim=-1)
    xi = torch.zeros(T, 6, dtype=DT)
    xi[:, :3] = torch.randn(T, 3, generator=g, dtype=DT) * 0.02
    xi[:, 2] += 0.4                                                # in front of the camera
    xi[:, 3:] = torch.randn(T, 3, generator=g, dtype=DT) * 0.8
    poses = se3_exp(xi)
    K = torch.tensor([[150.0, 0, hw[1] / 2], [0, 150.0, hw[0] / 2], [0, 0, 1]], dtype=DT)
    return pts, nrm, poses, K, hw


@pytest.mark.parametrize("downsample", [1, 2, 5])
@pytest.mark.parametrize("seed", range(4))
def test_visible_samples_sparse_is_identical(seed: int, downsample: int) -> None:
    pts, nrm, poses, K, hw = _scene(seed)
    ls = torch.tensor(0.1 * seed - 0.1, dtype=DT)
    for delta in (0.0, 0.3):
        dense = visible_samples(pts, nrm, poses, ls, K, hw, downsample, 0.02, delta)
        sparse = visible_samples(pts, nrm, poses, ls, K, hw, downsample, 0.02, delta, sparse=True)
        assert torch.equal(dense, sparse)
        assert 0 < int(dense.sum()) < dense.numel()               # the z-buffer actually rejected something


def test_visible_samples_sparse_is_identical_packed_per_frame() -> None:
    """Batched packing: per-frame templates, K, scale, frame sizes and padded samples."""
    pts, nrm, poses, K, hw = _scene(11, T=6)
    T, M = poses.shape[0], pts.shape[0]
    g = torch.Generator().manual_seed(3)
    pts_t = pts.expand(T, M, 3) * (1 + 0.1 * torch.rand(T, 1, 1, generator=g, dtype=DT))
    nrm_t = nrm.expand(T, M, 3).clone()
    K_t = K.expand(T, 3, 3).clone()
    K_t[::2, 0, 0] *= 0.7
    ls = torch.linspace(-0.2, 0.2, T, dtype=DT)
    frame_hw = torch.tensor([hw, (80, 100)] * (T // 2))
    valid = torch.rand(T, M, generator=g) > 0.2
    diam = torch.full((T,), 0.2, dtype=DT)
    kw = dict(frame_hw=frame_hw, sample_valid=valid, diameter=diam)
    dense = visible_samples(pts_t, nrm_t, poses, ls, K_t, hw, 1, 0.02, 0.1, **kw)
    sparse = visible_samples(pts_t, nrm_t, poses, ls, K_t, hw, 1, 0.02, 0.1, sparse=True, **kw)
    assert torch.equal(dense, sparse)


def test_solve_with_sparse_zbuffer_is_bit_identical() -> None:
    inputs = [_inputs(_clip(T, 256, seed=206 + T), seed=T) for T in (5, 7)]
    cfg = SolverConfig(n_obs=128, n_mesh=256, iters_coarse=4, iters_mid=2, iters_fine=3, device="cpu",
                       sparse_zbuffer=False)
    off = solve_trajectories(*_unzip(inputs), cfg)
    on = solve_trajectories(*_unzip(inputs), dataclasses.replace(cfg, sparse_zbuffer=True))
    for a, b in zip(off, on, strict=True):
        assert torch.equal(a.poses, b.poses) and a.scale == b.scale and a.accepted == b.accepted


def test_memory_estimate_drops_the_per_pixel_zbuffer() -> None:
    cfg = SolverConfig(device="cpu", plan_moments=True, sparse_zbuffer=False)
    d = [ClipDims(n_frames=100, n_mesh=2048, image_hw=(1440, 1920), n_tracks=96)] * 5
    dense = batch_peak_bytes(d, cfg, 1 << 28)
    sparse = batch_peak_bytes(d, dataclasses.replace(cfg, sparse_zbuffer=True), 1 << 28)
    # the dense buffer (8 bytes x 1440 x 1920 x 500 frames = 11 GB) set the peak; without it the Sinkhorn chunk does
    assert dense >= 500 * 8 * 1440 * 1920 and sparse < 0.5 * dense
    assert batch_peak_bytes(d, dataclasses.replace(cfg, sparse_zbuffer=True), 1 << 26) < 0.5 * sparse
    small = [ClipDims(n_frames=6, n_mesh=256, image_hw=(120, 160), n_tracks=16)]
    c = SolverConfig(device="cpu", sparse_zbuffer=True)
    assert batch_peak_bytes(small, c, 1 << 28) > 0
