"""The differentiable weighted LM step and the tracker-to-solver glue (tests S1-S14).

S10's import guard is static (an `ast` scan of the step module's own imports), not a `sys.modules` check: importing
any `fastdo.solver.*` submodule runs `fastdo/solver/__init__.py`, which imports `.lm` and with it scipy
(solver/__init__.py:42-52), so scipy is in `sys.modules` whatever the step module itself imports."""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import apply_left, se3_adjoint, se3_exp
from fastdo.solver import banded
from fastdo.solver._block_thomas import damped_diag_blocks
from fastdo.solver.banded import (
    BlockTridiagonal,
    ClipLayout,
    Priors,
    Smooth,
    assemble_block_tridiagonal,
    assemble_from_blocks,
    densify,
    term_blocks,
)
from fastdo.solver.lm import lm_step, total_cost
from fastdo.solver.terms import TermOutput, smoothness_term, velocity_term
from fastdo.tracker import (
    TrackerOutput,
    clip_frame_mask,
    clips_to_frames,
    frames_to_clips,
    tracker_increment_poses,
    tracker_sinkhorn_init,
    tracker_weight_overrides,
    weighted_lm_step,
    weighted_term_blocks,
)

DT = torch.float64
TRACKER_SRC = Path(__file__).resolve().parents[2] / "src" / "fastdo" / "tracker"
LAM = 0.05
T5 = 5


def _close(x: Tensor, ref: Tensor, rel: float) -> None:
    err = float((x - ref).detach().abs().max())
    assert err <= rel * max(float(ref.detach().abs().max()), 1e-300), (err, float(ref.detach().abs().max()))


def _problem(T: int = T5) -> tuple[TermOutput, TermOutput, Tensor]:
    """The shared S problem: pointmap-like (T,10,3) and track-like (T,6,2) terms, seeds 1, 2, 0."""
    g1, g2, g0 = (torch.Generator().manual_seed(s) for s in (1, 2, 0))
    w_pm = torch.rand(T, 10, dtype=DT, generator=g1) * 0.9 + 0.1
    w_pm[[1, 3], [2, 0]] = 0.0
    pm = TermOutput(torch.randn(T, 10, 3, dtype=DT, generator=g1),
                    torch.randn(T, 10, 3, 7, dtype=DT, generator=g1), w_pm)
    w_tr = torch.rand(T, 6, dtype=DT, generator=g2) * 0.9 + 0.1
    w_tr[2, 5] = 0.0
    tr = TermOutput(torch.randn(T, 6, 2, dtype=DT, generator=g2),
                    torch.randn(T, 6, 2, 7, dtype=DT, generator=g2), w_tr)
    poses = se3_exp(0.2 * torch.randn(T, 6, dtype=DT, generator=g0))
    return pm, tr, poses


def _overrides(T: int = T5, seed: int = 3) -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(T, 10, dtype=DT, generator=gen) * 0.9 + 0.1,
            torch.rand(T, 6, dtype=DT, generator=gen) * 0.9 + 0.1)


def _setup(T: int = T5) -> tuple[TermOutput, TermOutput, Smooth, Tensor, Tensor]:
    pm, tr, poses = _problem(T)                  # (pm, tr, smooth, o_pm, o_tr): constant-pose prior, seed-3 overrides
    o_pm, o_tr = _overrides(T)
    return pm, tr, smoothness_term(poses, 0.1), o_pm, o_tr


def _flat(delta: tuple[Tensor, Tensor]) -> Tensor:
    return torch.cat([delta[0].flatten(), delta[1].reshape(-1)])


def test_weighted_term_blocks_match_term_blocks() -> None:
    pm, tr, _ = _problem()
    gen = torch.Generator().manual_seed(3)
    for term in (pm, tr):
        o = torch.rand(term.weight.shape, dtype=DT, generator=gen) * 0.9 + 0.1
        o[0, :2] = 0.0
        ref = term_blocks(TermOutput(term.residual, term.jacobian, term.weight.clamp_min(0) * o))
        for got, want in (*zip(weighted_term_blocks(term, o), ref, strict=True),
                          *zip(weighted_term_blocks(term), term_blocks(term), strict=True)):
            _close(got, want, 1e-12)
        H32, g32 = weighted_term_blocks(TermOutput(term.residual.float(), term.jacobian.float(),
                                                   term.weight.float()), o.float())
        assert H32.dtype == DT and g32.dtype == DT
        negative, zero = term.weight.clone(), term.weight.clone()
        negative[4, 1], zero[4, 1] = -0.5, 0.0
        Hn, gn = weighted_term_blocks(TermOutput(term.residual, term.jacobian, negative), o)
        Hz, gz = weighted_term_blocks(TermOutput(term.residual, term.jacobian, zero), o)
        assert torch.equal(Hn, Hz) and torch.equal(gn, gz)


@pytest.mark.parametrize("use_velocity", [False, True])
def test_weighted_lm_step_matches_dense_reference_at_uniform_weights(use_velocity: bool) -> None:
    pm, tr, poses = _problem(T5)
    smooth = smoothness_term(poses, 0.1)
    priors = Priors(smooth=smooth, velocity=velocity_term(poses, 0.5)) if use_velocity else smooth
    ones = [torch.ones_like(pm.weight), torch.ones_like(tr.weight)]
    dp, ds = weighted_lm_step([pm, tr], priors, ones, 1e-2, T5)
    assert dp.shape == (T5, 6) and ds.shape == ()
    ref = lm_step(*densify(assemble_block_tridiagonal([pm, tr], priors, T5)), 1e-2)
    _close(_flat((dp, ds)), ref, 1e-10)


def test_weighted_lm_step_is_differentiable() -> None:
    pm, tr, smooth, o_pm, o_tr = _setup()

    def step(o_pm: Tensor, o_tr: Tensor, lam: Tensor) -> Tensor:
        return _flat(weighted_lm_step([pm, tr], smooth, [o_pm, o_tr], lam, T5))

    lam = torch.tensor([LAM], dtype=DT)
    assert torch.autograd.gradcheck(step, (o_pm.requires_grad_(), o_tr.requires_grad_(), lam.requires_grad_()))


def _embed(J7: Tensor, frame: int, T: int) -> Tensor:
    """(d,7) term Jacobian row -> (d, 6T+1): pose columns at 6*frame, scale column at 6T."""
    out = torch.zeros(J7.shape[0], 6 * T + 1, dtype=DT)
    out[:, 6 * frame:6 * frame + 6] = J7[:, :6]
    out[:, 6 * T] = J7[:, 6]
    return out


def test_weighted_lm_step_gradient_matches_implicit_formula() -> None:
    pm, tr, smooth, o_pm, o_tr = _setup()
    lam = torch.tensor([LAM], dtype=DT)

    def step(o: Tensor, damping: Tensor) -> Tensor:
        return _flat(weighted_lm_step([pm, tr], smooth, [o, o_tr], damping, T5))

    delta = step(o_pm, lam)
    blocks = [weighted_term_blocks(pm, o_pm), weighted_term_blocks(tr, o_tr)]
    H, g = densify(assemble_from_blocks(blocks, smooth, T5))
    diag_H = torch.diag(torch.diagonal(H))
    A_inv = torch.linalg.inv(H + LAM * diag_H + 1e-9 * torch.eye(6 * T5 + 1, dtype=DT))
    _close(delta, -A_inv @ g, 1e-10)
    jac_o = torch.autograd.functional.jacobian(lambda o: step(o, lam), o_pm)      # (n,T,10)
    ref = torch.zeros_like(jac_o)
    for i in range(T5):
        for k in range(10):
            Jk = _embed(pm.jacobian[i, k], i, T5)
            omega = pm.weight[i, k].clamp_min(0)
            JtJ = Jk.T @ Jk
            dA = omega * (JtJ + LAM * torch.diag(torch.diagonal(JtJ)))
            ref[:, i, k] = -A_inv @ (dA @ delta + omega * (Jk.T @ pm.residual[i, k]))
    _close(jac_o, ref, 1e-10)
    jac_lam = torch.autograd.functional.jacobian(lambda d: step(o_pm, d), lam)[:, 0]
    _close(jac_lam, -A_inv @ (diag_H @ delta), 1e-10)


def test_weighted_lm_step_zero_weight_matches_term_removed() -> None:
    pm, tr, smooth, _, o_tr = _setup()
    zero = _flat(weighted_lm_step([pm, tr], smooth, [torch.zeros_like(pm.weight), o_tr], LAM, T5))
    removed = _flat(weighted_lm_step([tr], smooth, [o_tr], LAM, T5))
    assert float((zero - removed).abs().max()) <= 1e-14 * float(removed.abs().max())


def test_weighted_lm_step_gradients_finite_at_zero_weights() -> None:
    pm, tr, smooth, o_pm, o_tr = _setup()
    zero_base = pm.weight == 0
    assert int(zero_base.sum()) == 2
    op = o_pm.clone().requires_grad_(True)                  # precondition: the sqrt path is NaN there
    sys_sqrt = assemble_block_tridiagonal([TermOutput(pm.residual, pm.jacobian, pm.weight * op), tr],
                                          smooth, T5)
    (g_sqrt,) = torch.autograd.grad((_flat(banded.solve_block_tridiagonal(sys_sqrt, LAM)) ** 2).sum(), op)
    assert not torch.isfinite(g_sqrt[zero_base]).any()
    (o_zero := o_pm.clone())[0, 0] = 0.0
    assert float(pm.weight[0, 0]) > 0
    op, ot = o_zero.requires_grad_(True), o_tr.clone().requires_grad_(True)
    lam = torch.tensor([LAM], dtype=DT, requires_grad=True)
    loss = (_flat(weighted_lm_step([pm, tr], smooth, [op, ot], lam, T5)) ** 2).sum()
    gp, gt, gl = torch.autograd.grad(loss, (op, ot, lam))
    assert all(bool(torch.isfinite(x).all()) for x in (gp, gt, gl))
    assert torch.all(gp[zero_base] == 0) and torch.all(gt[tr.weight == 0] == 0)
    assert bool(torch.isfinite(gp[0, 0])) and float(gp[0, 0]) != 0.0
    w_pm, w_tr = pm.weight.clone(), tr.weight.clone()
    w_pm[3], w_tr[3] = 0.0, 0.0                              # frame 3 has no data, the prior holds it
    masked = [TermOutput(pm.residual, pm.jacobian, w_pm), TermOutput(tr.residual, tr.jacobian, w_tr)]
    op, ot = o_pm.clone().requires_grad_(True), o_tr.clone().requires_grad_(True)
    loss = (_flat(weighted_lm_step(masked, smooth, [op, ot], lam, T5)) ** 2).sum()
    assert all(bool(torch.isfinite(x).all()) for x in torch.autograd.grad(loss, (op, ot, lam)))


def test_weighted_lm_step_batched_per_clip_damping() -> None:
    T, layout = 9, ClipLayout.build((5, 4), "cpu")
    pm, tr, poses = _problem(T)
    pair = torch.zeros(T - 1, dtype=DT).index_fill_(0, layout.pair_rows, 1.0)
    triple = torch.zeros(T - 2, dtype=DT).index_fill_(0, layout.triple_rows, 1.0)
    r, Jk, Jk1 = smoothness_term(poses, 0.1)
    rv, Ja, Jb, Jc = velocity_term(poses, 0.5)
    priors = Priors(smooth=(r * pair[:, None], Jk * pair[:, None, None], Jk1 * pair[:, None, None]),
                    velocity=(rv * triple[:, None], *(x * triple[:, None, None] for x in (Ja, Jb, Jc))))
    o_pm, o_tr = _overrides(T)
    lam = torch.tensor([1e-2, 3e-1], dtype=DT, requires_grad=True)
    dp, ds = weighted_lm_step([pm, tr], priors, [o_pm, o_tr], lam, T, layout)
    assert dp.shape == (T, 6) and ds.shape == (2,)
    blocks = [weighted_term_blocks(pm, o_pm), weighted_term_blocks(tr, o_tr)]
    H, g = densify(assemble_from_blocks(blocks, priors, T, layout))
    lam_rows = torch.cat([lam.detach()[layout.clip_index].repeat_interleave(6), lam.detach()])
    A = H + torch.diag(lam_rows * torch.diagonal(H)) + 1e-9 * torch.eye(6 * T + 2, dtype=DT)
    _close(_flat((dp, ds)), torch.linalg.solve(A, -g), 1e-10)

    def clip0(o: Tensor) -> Tensor:
        d = weighted_lm_step([pm, tr], priors, [o, o_tr], lam.detach(), T, layout)
        return torch.cat([d[0][:5].flatten(), d[1][:1]])

    jac = torch.autograd.functional.jacobian(clip0, o_pm)
    assert torch.all(jac[:, 5:] == 0) and float(jac[:, :5].abs().max()) > 0
    (g_lam,) = torch.autograd.grad((_flat((dp, ds)) ** 2).sum(), lam)
    assert bool(torch.isfinite(g_lam).all()) and bool((g_lam != 0).all())


@pytest.mark.parametrize("lam", [0.0, 1e-3, 10.0])
def test_weighted_step_decreases_weighted_quadratic_model(lam: float) -> None:
    T, (pm, tr, smooth, o_pm, o_tr) = T5, _setup(T5)
    dp, ds = weighted_lm_step([pm, tr], smooth, [o_pm, o_tr], lam, T)

    def model(dp: Tensor, ds: Tensor) -> Tensor:
        step7 = torch.cat([dp, ds.reshape(1, 1).expand(T, 1)], dim=-1)            # (T,7)
        terms = [TermOutput(t.residual + torch.einsum("tnda,ta->tnd", t.jacobian, step7), t.jacobian,
                            t.weight.clamp_min(0) * o) for t, o in ((pm, o_pm), (tr, o_tr))]
        r, Jk, Jk1 = smooth
        prior = (r + (Jk @ dp[:-1, :, None]).squeeze(-1) + (Jk1 @ dp[1:, :, None]).squeeze(-1), Jk, Jk1)
        return total_cost(terms, prior)

    q0, q1 = model(torch.zeros(T, 6, dtype=DT), torch.zeros((), dtype=DT)), model(dp, ds)
    blocks = [weighted_term_blocks(pm, o_pm), weighted_term_blocks(tr, o_tr)]
    H, g = densify(assemble_from_blocks(blocks, smooth, T))
    A = H + lam * torch.diag(torch.diagonal(H)) + 1e-9 * torch.eye(6 * T + 1, dtype=DT)
    decrease = g @ torch.linalg.solve(A, g)
    assert float(decrease) > 0
    assert float(q1 - q0) <= -(1 - 1e-9) * float(decrease) + 1e-12 * abs(float(q0))


def test_solve_block_tridiagonal_contract() -> None:
    params = inspect.signature(banded.solve_block_tridiagonal).parameters
    assert list(params) == ["sys", "lam", "floor"] and params["floor"].default == 1e-9
    pm, tr, poses = _problem(5)
    out = banded.solve_block_tridiagonal(assemble_block_tridiagonal([pm, tr], smoothness_term(poses, 0.1), 5), 0.1)
    assert isinstance(out, tuple) and len(out) == 2 and out[0].shape == (5, 6) and out[1].shape == ()
    out = banded.solve_block_tridiagonal(assemble_block_tridiagonal([pm, tr], None, 5, ClipLayout.build((3, 2), "cpu")),
                                         torch.tensor([0.1, 0.2], dtype=DT))
    assert len(out) == 2 and out[0].shape == (5, 6) and out[1].shape == (2,)
    Jm = torch.randn(9, 6, dtype=DT, generator=(gen := torch.Generator().manual_seed(4)))
    D, g = Jm.T @ Jm, torch.randn(1, 6, dtype=DT, generator=gen)
    ref = D + 0.3 * torch.diag(torch.diagonal(D)) + 1e-9 * torch.eye(6, dtype=DT)
    _close(damped_diag_blocks(D[None, None], torch.tensor([0.3], dtype=DT), 1e-9)[0, 0], ref, 1e-12)
    sys = BlockTridiagonal(D=D[None], U=torch.zeros(0, 6, 6, dtype=DT), c=torch.zeros(1, 6, dtype=DT),
                           h_ss=torch.tensor(2.0, dtype=DT), g=g, g_s=torch.tensor(0.5, dtype=DT))
    dp, ds = banded.solve_block_tridiagonal(sys, 0.3)
    _close(dp[0], torch.linalg.solve(ref, -g[0]), 1e-12)
    _close(ds, torch.tensor(-0.5 / (2.0 * 1.3 + 1e-9), dtype=DT), 1e-12)


FORBIDDEN = ("scipy", "fastdo.solver.lm", "fastdo.solver.lm_blocks")


def _imported(tree: ast.AST) -> list[tuple[str, tuple[str, ...]]]:
    nodes = list(ast.walk(tree))
    return [*((alias.name, ()) for node in nodes if isinstance(node, ast.Import) for alias in node.names),
            *(("." * node.level + (node.module or ""), tuple(a.name for a in node.names))
              for node in nodes if isinstance(node, ast.ImportFrom))]


def _forbidden(module: str, names: tuple[str, ...]) -> bool:
    candidates = [module, *(f"{module}.{name}" for name in names)]
    return any(c == bad or c.startswith(bad + ".") for c in candidates for bad in FORBIDDEN)


def test_step_modules_import_no_scipy() -> None:
    assert _forbidden("fastdo.solver", ("lm",)) and _forbidden("scipy.linalg", ("solveh_banded",))
    assert not _forbidden("fastdo.solver.banded", ("solve_block_tridiagonal",))
    for name in ("step.py", "sinkhorn_check.py", "gradient_check_report.py"):
        imports = _imported(ast.parse((TRACKER_SRC / name).read_text()))
        assert not [imp for imp in imports if _forbidden(*imp)], name
        solve_from = [m for m, names in imports if "solve_block_tridiagonal" in names]
        assert solve_from in ([], ["fastdo.solver.banded"]), (name, solve_from)
    step_imports = _imported(ast.parse((TRACKER_SRC / "step.py").read_text()))
    assert ("fastdo.solver.banded" in [m for m, names in step_imports
                                       if {"solve_block_tridiagonal", "assemble_from_blocks"} <= set(names)])


def _tracker_output(seed: int = 0) -> tuple[TrackerOutput, ClipLayout]:
    layout = ClipLayout.build((5, 3), "cpu")
    B, W, N, J, C = 2, layout.width, 7, 4, 64
    rnd = functools.partial(torch.randn, dtype=DT, generator=torch.Generator().manual_seed(seed))
    out = TrackerOutput(point_weights=rnd(B, W, N).sigmoid(), track_weights=rnd(B, W, J).sigmoid(),
                        damping=rnd(B).exp(), xi_hat=rnd(B, W, 6), sigma_hat=rnd(B),
                        precision_diag=rnd(B, W, 6).exp(), dual_g=rnd(B, W, C), dual_f=rnd(B, W, N))
    return out, layout


def test_tracker_weight_overrides_and_sinkhorn_init() -> None:
    out, layout = _tracker_output()
    T, M, N = layout.n_frames, 80, 7
    o_point, o_track = tracker_weight_overrides(out, layout)
    assert torch.equal(o_point, clips_to_frames(out.point_weights, layout)) and o_point.shape == (T, N)
    assert torch.equal(o_track, clips_to_frames(out.track_weights, layout))
    real = clip_frame_mask(layout)
    assert torch.equal(frames_to_clips(o_point, layout)[real], out.point_weights[real])
    gen = torch.Generator().manual_seed(7)
    ids = torch.randint(64, (M,), generator=gen)
    eps = torch.rand(T, dtype=DT, generator=gen) * 1e-3 + 1e-4
    f, g = tracker_sinkhorn_init(out, layout, ids, eps)
    assert torch.equal(f, torch.zeros(T, N, dtype=DT)) and g.shape == (T, M) and g.dtype == DT
    dual_g = clips_to_frames(out.dual_g, layout)
    ref = torch.stack([torch.stack([eps[t] * dual_g[t, int(ids[j])] for j in range(M)]) for t in range(T)])
    _close(g, ref, 1e-15)
    assert torch.equal(tracker_sinkhorn_init(out, layout, ids.expand(T, M).clone(), eps)[1], g)
    g_float = tracker_sinkhorn_init(out, layout, ids, 3e-4)[1]
    assert torch.equal(g_float, tracker_sinkhorn_init(out, layout, ids, torch.full((T,), 3e-4, dtype=DT))[1])
    C = torch.rand(T, N, M, dtype=DT, generator=gen) * 1e-2
    a = torch.rand(T, N, dtype=DT, generator=gen)
    a[0, 1] = 0.0
    f, g = tracker_sinkhorn_init(out, layout, None, eps, drive="f", C=C, a=a, rho_b=1e-3)
    assert torch.equal(f, eps[:, None] * clips_to_frames(out.dual_f, layout))
    log_a = torch.log(a.clamp_min(1e-30))
    ref = torch.stack([torch.stack([-(1e-3 / (1e-3 + eps[t])) * eps[t] * torch.logsumexp(
        (f[t] - C[t, :, j]) / eps[t] + log_a[t], 0) for j in range(M)]) for t in range(T)])
    _close(g, ref, 1e-13)
    assert torch.equal(tracker_sinkhorn_init(out, layout, None, eps, drive="f", C=C, a=a,
                                             rho_b=torch.full((T,), 1e-3, dtype=DT))[1], g)
    g_hard = tracker_sinkhorn_init(out, layout, None, eps, drive="f", C=C, a=a, rho_b=float("inf"))[1]
    _close(g_hard, g * (1e-3 + eps[:, None]) / 1e-3, 1e-13)                      # tau_b = 1 at rho = inf


def test_tracker_increment_poses() -> None:
    out, layout = _tracker_output()
    T = layout.n_frames
    gen = torch.Generator().manual_seed(8)
    poses = se3_exp(0.3 * torch.randn(T, 6, dtype=DT, generator=gen))
    log_scale = torch.randn(2, dtype=DT, generator=gen)
    L = 0.1 * torch.arange(1, T + 1, dtype=DT)
    new_poses, new_log_scale = tracker_increment_poses(poses, log_scale, out, layout, L)
    xi = clips_to_frames(out.xi_hat, layout)
    scaled = torch.cat([xi[:, :3] * L[:, None], xi[:, 3:]], dim=-1)
    ref = apply_left(poses, (se3_adjoint(poses) @ scaled[..., None]).squeeze(-1))
    assert float((new_poses - ref).abs().max()) <= 1e-12
    assert torch.equal(new_log_scale, log_scale + out.sigma_hat)
    zero_xi = out._replace(xi_hat=torch.zeros_like(out.xi_hat))
    assert torch.equal(tracker_increment_poses(poses, log_scale, zero_xi, layout, L)[0], poses)
    unit, _ = tracker_increment_poses(poses, log_scale, out, layout, 1.0)
    assert float((unit - new_poses).abs().max()) > 1e-3
    zero_dim, _ = tracker_increment_poses(poses, log_scale, out, layout, torch.tensor(1.0, dtype=DT))
    assert torch.equal(zero_dim, unit)


def test_weighted_lm_step_reads_nothing_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    pm, tr, smooth, o_pm, o_tr = _setup(9)
    out, layout = _tracker_output()
    ids, eps = torch.arange(80) % 64, torch.full((8,), 1e-4, dtype=DT)
    C, a = torch.rand(8, 7, 80, dtype=DT), torch.full((8, 7), 1 / 7, dtype=DT)
    batch = ClipLayout.build((5, 4), "cpu")
    lam = torch.tensor([1e-2, 3e-1], dtype=DT)

    def run() -> list[Tensor]:
        res = [*weighted_term_blocks(pm, o_pm), *weighted_lm_step([pm, tr], smooth, [o_pm, None], LAM, 9)]
        res += [*weighted_lm_step([pm, tr], None, [o_pm, o_tr], lam, 9, batch)]
        res += [*tracker_weight_overrides(out, layout), *tracker_sinkhorn_init(out, layout, ids, eps)]
        return res + [*tracker_sinkhorn_init(out, layout, None, eps, drive="f", C=C, a=a, rho_b=1e-3)]

    ref = run()

    def host_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("host read")

    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__float__", "__int__"):
            m.setattr(torch.Tensor, name, host_read)
        got = run()
    assert len(got) == len(ref) and all(torch.equal(x, y) for x, y in zip(got, ref, strict=True))


def test_weighted_lm_step_rejects_bad_shapes() -> None:
    pm, tr, smooth, _, _ = _setup()
    out, layout = _tracker_output()
    two_clips = ClipLayout.build((3, 2), "cpu")
    calls = [lambda: weighted_lm_step([pm, tr], smooth, [None], LAM, T5),                  # override count
             lambda: weighted_lm_step([pm], smooth, [torch.ones(T5, 11, dtype=DT)], LAM, T5),  # override shape
             lambda: weighted_lm_step([_problem(T5 + 1)[0]], smooth, [None], LAM, T5),     # frame count
             lambda: weighted_lm_step([pm], None, [None], torch.ones(3, dtype=DT), T5, two_clips),
             lambda: weighted_lm_step([pm], smooth, [None], torch.ones(2, dtype=DT), T5),
             lambda: tracker_sinkhorn_init(out, layout, None, 1e-4),                        # g drive, no ids
             lambda: tracker_sinkhorn_init(out, layout, None, 1e-4, drive="f")]             # f drive, no C
    for call in calls:
        with pytest.raises(ValueError):
            call()
