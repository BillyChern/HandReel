"""The memory of keyframe registration (`fastdo.solver.keyframe`), checked on CPU.

An H100 run at 4x counts (FASTDO_GPU_MEM_GB=20) died with "Tried to allocate 7.50 GiB": exactly one (120, 2048,
4096) float64 hypothesis x observed x mesh distance tensor, before the solve and its memory plan. The residuals held
the distance tensor and a masked copy of it at once (a measured CUDA peak of 2.03x one tensor), and a scaled replay
(3x counts under 12 GiB) failed the same way on the copy's allocation. The residuals now read the mesh-side minimum
first and mask the tensor in place, so one distance tensor is alive at a time, with identical outputs.

Peaks are the torch.profiler CPU allocator curve (as in `test_memory.py`).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from fastdo.solver import keyframe
from tests.solver.test_batch import _clip

_H = 120   # keyframe._rotation_hypotheses


def _two_copy_residuals(obs: Tensor, Y: Tensor, Yn: Tensor, R: Tensor, t: Tensor, s: Tensor
                        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The residuals as they were before this lane: the obs->mesh minimum over a masked copy of the distances."""
    P = s[:, None, None] * torch.einsum("hij,mj->hmi", R, Y) + t[:, None]
    front = (torch.einsum("hij,mj->hmi", R, Yn) * -P).sum(-1) > 0
    D = torch.cdist(obs.expand(R.shape[0], -1, -1), P)
    d_obs, j_obs = D.masked_fill(~front[:, None, :], math.inf).min(2)
    d_mesh, i_mesh = D.min(1)
    return d_obs, j_obs, d_mesh, i_mesh, front


def _peak_bytes(fn: Callable[[], object]) -> int:
    """Peak live bytes above the start of `fn`, from the profiler's allocation and free events."""
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    events = sorted((e.start_ns(), e.nbytes()) for e in prof.profiler.kineto_results.events() if e.name() == "[memory]")
    live = peak = 0
    for _t_ns, n_bytes in events:
        live += n_bytes
        peak = max(peak, live)
    return peak


def _hypothesis_inputs(n_obs: int, n_mesh: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(3)
    dt = torch.float64
    Y = 0.05 * torch.randn(n_mesh, 3, generator=gen, dtype=dt)
    Yn = F.normalize(torch.randn(n_mesh, 3, generator=gen, dtype=dt), dim=-1)
    obs = 0.05 * torch.randn(n_obs, 3, generator=gen, dtype=dt) + torch.tensor([0.0, 0.0, 0.6], dtype=dt)
    R = keyframe._rotation_hypotheses(dt)
    t = torch.tensor([0.0, 0.0, 0.62], dtype=dt).expand(R.shape[0], 3).contiguous()
    s = torch.ones(R.shape[0], dtype=dt)
    return obs, Y, Yn, R, t, s


def test_residuals_hold_one_distance_tensor_with_identical_outputs() -> None:
    args = _hypothesis_inputs(96, 384)
    one_tensor = _H * 96 * 384 * 8
    for new, old in zip(keyframe._residuals(*args), _two_copy_residuals(*args), strict=True):
        assert new.dtype == old.dtype and torch.equal(new, old)
    assert _peak_bytes(lambda: _two_copy_residuals(*args)) >= 1.9 * one_tensor     # what the R4b run paid
    peak = _peak_bytes(lambda: keyframe._residuals(*args))
    assert peak <= 1.25 * one_tensor, (peak, one_tensor, peak / one_tensor)


def test_register_keyframe_pose_peaks_at_one_distance_tensor() -> None:
    clip = _clip(1, 1024, seed=5, image_hw=(120, 160))
    valid = clip.masks[0] & torch.isfinite(clip.pointmaps[0]).all(-1)
    observed = clip.pointmaps[0][valid]
    n_obs, n_mesh = 128, 1024
    assert observed.shape[0] >= n_obs
    one_tensor = _H * n_obs * n_mesh * 8
    peak = _peak_bytes(lambda: keyframe.register_keyframe_pose(
        clip.surface_points, clip.surface_normals, observed, clip.intrinsics, clip.masks[0], scale=clip.scale_gt,
        n_obs=n_obs, n_mesh=n_mesh, n_iters=2, seed=0))
    assert peak <= 1.3 * one_tensor, (peak, one_tensor, peak / one_tensor)
