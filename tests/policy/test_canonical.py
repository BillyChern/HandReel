"""Object-centric gravity-aligned canonical frame (design Section 3)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp, so3_exp
from fastdo.policy.canonical import canonical_frame

DT = torch.float64
GRAVITIES = [
    torch.tensor([0.0, 0.0, -9.81], dtype=DT),
    torch.tensor([1.3, -2.1, -9.5], dtype=DT),                  # tilted world frame (camera-frame clips)
]


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _up(gravity: Tensor) -> Tensor:
    return -gravity / gravity.norm()


def _horizontal_basis(u: Tensor) -> tuple[Tensor, Tensor]:
    a = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    e1 = a - (a @ u) * u
    e1 = e1 / e1.norm()
    return e1, torch.linalg.cross(u, e1)


def _apply(h: Tensor, pose: Tensor, wrist: Tensor) -> tuple[Tensor, Tensor]:
    return h @ pose, wrist @ h[:3, :3].T + h[:3, 3]


def _group_element(gen: torch.Generator, u: Tensor) -> Tensor:
    yaw = (torch.rand((), dtype=DT, generator=gen) * 2.0 - 1.0) * math.pi
    h = torch.eye(4, dtype=DT)
    h[:3, :3] = so3_exp(yaw * u)
    h[:3, 3] = torch.randn(3, dtype=DT, generator=gen) * 2.0
    return h


def _check_frame(F: Tensor, pose: Tensor, u: Tensor) -> None:
    R = F[:, :3, :3]
    eye = torch.eye(3, dtype=DT).expand_as(R)
    assert (R.transpose(-1, -2) @ R - eye).abs().amax() < 1e-12
    assert (torch.linalg.det(R) - 1.0).abs().amax() < 1e-12
    assert (R[:, :, 2] - u).abs().amax() < 1e-12
    assert (R[:, :, 0] @ u).abs().amax() < 1e-12
    assert torch.equal(F[:, :3, 3], pose[:, :3, 3])
    assert torch.equal(F[:, 3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DT).expand(F.shape[0], 4))


@pytest.mark.parametrize("gravity", GRAVITIES)
def test_canonical_frame_gravity_aligned_and_equivariant(gravity: Tensor) -> None:
    gen = _gen(0)
    N = 256
    u = _up(gravity)
    xi = torch.randn(N, 6, dtype=DT, generator=gen)
    pose = se3_exp(xi)
    wrist = pose[:, :3, 3] + 0.1 * torch.randn(N, 3, dtype=DT, generator=gen)

    F = canonical_frame(pose, wrist, gravity, None)
    assert F.shape == (N, 4, 4)
    _check_frame(F, pose, u)
    # Regular case: x is the normalized horizontal projection of the object's first axis.
    a = pose[:, :3, 0]
    proj = a - (a @ u)[:, None] * u
    regular = proj.norm(dim=-1) > math.sin(math.radians(5.0))
    assert regular.sum() > N // 2
    assert torch.allclose(F[regular, :3, 0], proj[regular] / proj[regular].norm(dim=-1, keepdim=True),
                          atol=1e-12)

    for _ in range(4):
        h = _group_element(gen, u)
        pose_h, wrist_h = _apply(h, pose, wrist)
        F_h = canonical_frame(pose_h, wrist_h, gravity, None)
        err = (F_h - h @ F).abs().amax()
        assert err < 1e-9, f"canonical_frame(h T) != h canonical_frame(T): {err:.3e}"


def _axis_pose(axis_dir: Tensor, position: Tensor) -> Tensor:
    """A pose whose first rotation column is `axis_dir` (unit), completed to a right-handed frame."""
    helper = torch.tensor([0.3, 0.9, 0.1], dtype=DT)
    y = helper - (helper @ axis_dir) * axis_dir
    y = y / y.norm()
    T = torch.eye(4, dtype=DT)
    T[:3, 0] = axis_dir
    T[:3, 1] = y
    T[:3, 2] = torch.linalg.cross(axis_dir, y)
    T[:3, 3] = position
    return T


@pytest.mark.parametrize("gravity", GRAVITIES)
def test_canonical_frame_degenerate_axis_uses_wrist_direction(gravity: Tensor) -> None:
    gen = _gen(3)
    u = _up(gravity)
    e1, e2 = _horizontal_basis(u)
    p = torch.tensor([0.4, -0.2, 1.1], dtype=DT)
    wrist_to_obj = 0.07 * e1 - 0.04 * e2 + 0.05 * u                         # has a vertical component too
    for sign in (1.0, -1.0):                                                # axis parallel or anti-parallel
        pose = _axis_pose(sign * u, p)[None]
        wrist = (p - wrist_to_obj)[None]
        F = canonical_frame(pose, wrist, gravity, None)
        _check_frame(F, pose, u)
        horiz = wrist_to_obj - (wrist_to_obj @ u) * u
        assert torch.allclose(F[0, :3, 0], horiz / horiz.norm(), atol=1e-12)
        # The wrist branch is equivariant under yaw about u plus translation.
        for _ in range(3):
            h = _group_element(gen, u)
            pose_h, wrist_h = _apply(h, pose, wrist)
            F_h = canonical_frame(pose_h, wrist_h, gravity, None)
            _check_frame(F_h, pose_h, u)
            assert (F_h - h @ F).abs().amax() < 1e-9


@pytest.mark.parametrize("gravity", GRAVITIES)
def test_canonical_frame_wrist_direction_takes_precedence_over_previous_axis(gravity: Tensor) -> None:
    u = _up(gravity)
    e1, e2 = _horizontal_basis(u)
    p = torch.tensor([-0.3, 0.5, 0.8], dtype=DT)
    pose = _axis_pose(u, p)[None]                                           # object axis parallel to up
    wrist = (p - 0.1 * e1)[None]                                            # horizontal wrist direction e1
    F = canonical_frame(pose, wrist, gravity, e2[None])                     # previous frame's x = e2
    _check_frame(F, pose, u)
    assert torch.allclose(F[0, :3, 0], e1, atol=1e-12)


def test_canonical_frame_casts_gravity_and_validates_shapes() -> None:
    gen = _gen(7)
    N = 64
    pose = se3_exp(torch.randn(N, 6, dtype=DT, generator=gen))
    wrist = pose[:, :3, 3] + 0.1 * torch.randn(N, 3, dtype=DT, generator=gen)
    g32 = torch.tensor([1.3, -2.1, -9.5], dtype=torch.float32)
    F = canonical_frame(pose, wrist, g32, None)                             # normalized in float64, not float32
    assert F.dtype == DT
    _check_frame(F, pose, _up(g32.to(DT)))

    g = GRAVITIES[0]
    bad_calls = [
        (torch.tensor(1.0, dtype=DT), wrist, g, None),                      # 0-d pose
        (pose[:, :3], wrist, g, None),                                      # (N,3,4)
        (pose[0], wrist[:4], g, None),                                      # one unbatched (4,4) pose
        (pose, wrist[:, :2], g, None),
        (pose, wrist[:-1], g, None),
        (pose, wrist, g[None], None),
        (pose, wrist, torch.zeros(3, dtype=DT), None),
        (pose, wrist, g, wrist[:, :2]),
        (pose, wrist, g, wrist[:-1]),
    ]
    for args in bad_calls:
        with pytest.raises(ValueError):
            canonical_frame(*args)


def test_canonical_frame_five_degree_threshold() -> None:
    gravity = GRAVITIES[0]
    u = _up(gravity)
    e1, e2 = _horizontal_basis(u)
    p = torch.zeros(3, dtype=DT)
    wrist = (p - 0.1 * e2)[None]                                            # wrist direction = +e2
    for deg, expect in ((4.9, e2), (5.1, e1)):
        th = math.radians(deg)
        pose = _axis_pose(math.cos(th) * u + math.sin(th) * e1, p)[None]
        F = canonical_frame(pose, wrist, gravity, None)
        assert torch.allclose(F[0, :3, 0], expect, atol=1e-12), deg


@pytest.mark.parametrize("gravity", GRAVITIES)
def test_canonical_frame_keeps_previous_axis_when_wrist_degenerate(gravity: Tensor) -> None:
    gen = _gen(5)
    u = _up(gravity)
    e1, e2 = _horizontal_basis(u)
    p = torch.tensor([0.1, 0.2, 0.3], dtype=DT)
    pose = _axis_pose(u, p)[None]
    wrist = (p - 0.08 * u + 0.001 * e1)[None]                               # wrist ~straight below
    prev_x = (0.6 * e1 + 0.8 * e2)[None]

    F = canonical_frame(pose, wrist, gravity, prev_x)
    _check_frame(F, pose, u)
    assert torch.allclose(F[0, :3, 0], prev_x[0], atol=1e-12)

    # No previous frame: the object's second axis (horizontal within 5 deg whenever the first is
    # vertical within 5 deg) keeps the frame well defined and still equivariant.
    F0 = canonical_frame(pose, wrist, gravity, None)
    _check_frame(F0, pose, u)
    b = pose[0, :3, 1]
    b = b - (b @ u) * u
    assert torch.allclose(F0[0, :3, 0], b / b.norm(), atol=1e-12)

    h = _group_element(gen, u)
    pose_h, wrist_h = _apply(h, pose, wrist)
    for px in (prev_x, None):
        px_h = None if px is None else px @ h[:3, :3].T
        F_h = canonical_frame(pose_h, wrist_h, gravity, px_h)
        assert (F_h - h @ canonical_frame(pose, wrist, gravity, px)).abs().amax() < 1e-9
