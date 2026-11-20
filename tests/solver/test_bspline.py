import torch

from fastdo.solver._bspline import bspline_prefilter_2d, bspline_sample_with_grad

torch.manual_seed(0)
DT = torch.float64


def _random_grid(h: int, w: int) -> torch.Tensor:
    return torch.randn(1, 1, h, w, dtype=DT)


def test_exact_interpolation_1d_including_borders() -> None:
    """A degenerate H=1 grid exercises the 1D code path end to end, borders included."""
    row = torch.randn(9, dtype=DT)
    img = row[None, None, None, :]
    coef = bspline_prefilter_2d(img)
    uv = torch.stack([torch.arange(9, dtype=DT), torch.zeros(9, dtype=DT)], dim=-1)[None]
    val, _ = bspline_sample_with_grad(coef, uv)
    assert torch.allclose(val[0], row, atol=1e-8)


def test_exact_interpolation_2d_including_borders() -> None:
    h, w = 6, 7
    img = _random_grid(h, w)
    coef = bspline_prefilter_2d(img)
    vv, uu = torch.meshgrid(torch.arange(h, dtype=DT), torch.arange(w, dtype=DT), indexing="ij")
    uv = torch.stack([uu.flatten(), vv.flatten()], dim=-1)[None]
    val, _ = bspline_sample_with_grad(coef, uv)
    assert torch.allclose(val[0], img[0, 0].flatten(), atol=1e-8)
    # borders/corners specifically, since a boundary-convention bug shows up only there
    corners = torch.tensor([[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]], dtype=DT)
    val_c, _ = bspline_sample_with_grad(coef, corners[None])
    expected_c = torch.tensor([img[0, 0, 0, 0], img[0, 0, 0, -1], img[0, 0, -1, 0], img[0, 0, -1, -1]], dtype=DT)
    assert torch.allclose(val_c[0], expected_c, atol=1e-8)


def test_gradient_matches_central_difference_interior_and_near_border() -> None:
    h, w = 6, 7
    img = _random_grid(h, w)
    coef = bspline_prefilter_2d(img)
    points = torch.tensor(
        [[3.2, 2.7], [0.3, 0.5], [w - 1.2, 1.4], [2.0, h - 1.1], [0.0, 0.0], [w - 1.0, h - 1.0]], dtype=DT
    )
    uv = points[None]
    _, grad = bspline_sample_with_grad(coef, uv)
    eps = 1e-5
    for i in range(points.shape[0]):
        for axis in range(2):
            plus = points.clone()
            plus[i, axis] += eps
            minus = points.clone()
            minus[i, axis] -= eps
            v_plus, _ = bspline_sample_with_grad(coef, plus[None])
            v_minus, _ = bspline_sample_with_grad(coef, minus[None])
            num = (v_plus[0, i] - v_minus[0, i]) / (2 * eps)
            assert torch.allclose(grad[0, i, axis], num, atol=1e-6), (i, axis)


def test_second_derivative_continuous_across_a_knot() -> None:
    """C2 sanity: the analytic gradient's own finite difference (an estimate of the second
    derivative) should converge to the *same* value approaching an interior knot from either
    side -- for a merely C1 interpolant (e.g. cubic convolution) it would not, converging
    instead to two different one-sided limits as delta shrinks.
    """
    h, w = 6, 7
    img = _random_grid(h, w)
    coef = bspline_prefilter_2d(img)

    def grad_u(u: float, v: float) -> float:
        uv = torch.tensor([[[u, v]]], dtype=DT)
        _, grad = bspline_sample_with_grad(coef, uv)
        return float(grad[0, 0, 0])

    knot, v0 = 3.0, 2.7
    diffs = []
    for delta in (1e-2, 1e-3, 1e-4):
        d_minus = (grad_u(knot, v0) - grad_u(knot - delta, v0)) / delta
        d_plus = (grad_u(knot + delta, v0) - grad_u(knot, v0)) / delta
        diffs.append(abs(d_minus - d_plus))
    assert diffs[-1] < 1e-3
    # the mismatch should shrink roughly linearly with delta (truncation error of a smooth
    # function), not plateau at an O(1) value the way a genuine second-derivative jump would.
    assert diffs[1] < diffs[0] / 3
    assert diffs[2] < diffs[1] / 3
