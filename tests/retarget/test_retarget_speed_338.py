"""Retarget speed levers: the exact data-movement levers A1-A5 are bit-identical to the code they replace, the
opt-in scaling-domain Sinkhorn (B1) matches the log-domain iteration to round-off, and every new check fires.

Reference implementations of the replaced code are copied here verbatim (origin/main 4a40535a), so a regression in
the new code shows as a bit difference rather than as a change in a downstream tolerance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch
import trimesh
from torch import Tensor

from fastdo.geometry.lie import quat_wxyz_to_matrix
from fastdo.retarget.batched_kinematics import BatchedKinematics, canonical_points
from fastdo.retarget.contact import ContactMeasure
from fastdo.retarget.kinematic import RetargetConfig, retarget_kinematic
from fastdo.retarget.mj_model import RobotScene
from fastdo.retarget.transport import InbandMeasure, contact_transport_term_inband, inband_measure
from fastdo.solver import lm_blocks, ot
from fastdo.solver.lm_blocks import BlockLMConfig, BlockTerm, SmoothTerm, lm_optimize_blocks
from fastdo.solver.sdf import SdfGrid, mesh_sdf_grid, sdf_lookup
from tests.retarget.whisking_data import (
    MESH_SCALE,
    build_scene,
    build_whisk_sdf_and_surface,
    load_whisking_inputs,
    scene_missing,
    whisking_data_missing,
)

DT = torch.float64


def _box_sdf() -> SdfGrid:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return mesh_sdf_grid(np.asarray(box.vertices), np.asarray(box.faces), resolution=24)


# ---------------------------------------------------------------- B1: scaling-domain Sinkhorn


def _ot_problem(seed: int, B: int = 12, N: int = 15, M: int = 96) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    C = torch.rand(B, N, M, generator=gen, dtype=DT) * 3.0
    C = C + 5.0 * (torch.rand(B, N, M, generator=gen) < 0.7).to(DT)              # the label-mismatch term
    a = torch.rand(B, N, generator=gen, dtype=DT) * 0.85 + 0.135                  # band weights in [e^-2, 1]
    valid = torch.rand(B, N, generator=gen) < 0.6
    valid[:, 0] = True
    return C, a, valid


@pytest.mark.parametrize("eps", [0.1, 0.01, 0.002])
def test_scaling_sinkhorn_matches_the_log_domain_iteration(eps: float) -> None:
    C, a, valid = _ot_problem(0)
    ref = ot.sinkhorn_unbalanced_uniform_b(C, a, valid, eps, 2.0, 0.02, 25)
    got = ot.sinkhorn_unbalanced_uniform_b_scaling(C, a, valid, eps, 2.0, 0.02, 25)
    assert float((got - ref).abs().max()) <= 1e-12 * float(ref.abs().max())
    assert torch.equal(got[~valid], torch.zeros_like(got[~valid]))               # invalid rows carry no mass


def test_scaling_sinkhorn_absorbs_and_still_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a tiny drift threshold every iteration re-absorbs; with a huge one none does: both match."""
    C, a, valid = _ot_problem(1)
    ref = ot.sinkhorn_unbalanced_uniform_b(C, a, valid, 0.005, 2.0, 0.02, 25)
    for drift in (1e-3, 1e9):
        monkeypatch.setattr(ot, "_ABSORB_DRIFT", drift)
        before = dict(ot._SINKHORN_STATS)
        got = ot.sinkhorn_unbalanced_uniform_b_scaling(C, a, valid, 0.005, 2.0, 0.02, 25)
        absorbed = ot._SINKHORN_STATS["absorptions"] - before["absorptions"]
        assert absorbed > C.shape[0] if drift < 1.0 else absorbed == C.shape[0]
        assert float((got - ref).abs().max()) <= 1e-11 * float(ref.abs().max())


def test_scaling_sinkhorn_falls_back_to_the_log_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch element whose sums leave the safe range is re-solved by the log-domain solver itself (bit-equal)."""
    C, a, valid = _ot_problem(2)
    ref = ot.sinkhorn_unbalanced_uniform_b(C, a, valid, 0.01, 2.0, 0.02, 25)
    monkeypatch.setattr(ot, "_SCALING_RANGE", 0.9)                              # every sum is "out of range"
    before = ot._SINKHORN_STATS["fallbacks"]
    got = ot.sinkhorn_unbalanced_uniform_b_scaling(C, a, valid, 0.01, 2.0, 0.02, 25)
    assert ot._SINKHORN_STATS["fallbacks"] - before == C.shape[0]
    assert torch.equal(got, ref)


def test_scaling_sums_range_predicate() -> None:
    x = torch.tensor([[1.0, 1e-300], [1.0, 1e290], [1.0, math.inf], [1.0, math.nan], [1e-270, 1e270]], dtype=DT)
    assert ot._sums_in_range(x).tolist() == [False, False, False, False, True]


def test_scaling_sinkhorn_row_without_valid_atom_falls_back() -> None:
    C, a, valid = _ot_problem(3, B=3)
    valid[1] = False
    ref = ot.sinkhorn_unbalanced_uniform_b(C, a, valid, 0.01, 2.0, 0.02, 25)
    got = ot.sinkhorn_unbalanced_uniform_b_scaling(C, a, valid, 0.01, 2.0, 0.02, 25)
    assert torch.equal(got[1], ref[1])
    assert torch.isfinite(got).all()


def test_retarget_config_validates_sinkhorn() -> None:
    with pytest.raises(ValueError, match="sinkhorn"):
        RetargetConfig(sinkhorn="exp")
    assert RetargetConfig().sinkhorn == "log"


def test_legacy_path_refuses_the_scaling_sinkhorn() -> None:
    T = 2
    cfg = RetargetConfig(fast_path=False, sinkhorn="scaling")
    eye = torch.eye(4, dtype=DT).expand(T, 4, 4)
    with pytest.raises(NotImplementedError, match="scaling"):
        retarget_kinematic(eye, eye, 1.0, torch.zeros(T, 778, 3, dtype=DT), torch.zeros(778, dtype=torch.int8),
                           None, None, None, None, 30.0, cfg)                     # type: ignore[arg-type]


# ---------------------------------------------------------------- A1, A3, A5: the in-band transport term


def _reference_inband(measure: InbandMeasure, robot_points: Tensor, robot_jac: Tensor | None, robot_labels: Tensor,
                      eps: float, rho_h: float, rho_r: float, lambda_label: float, n_sinkhorn: int,
                      plan: Tensor | None, contact_margin: float, sdf: SdfGrid, n_dof: int = 28,
                      ) -> tuple[BlockTerm, Tensor]:
    """`contact_transport_term_inband` as of origin/main 4a40535a, verbatim."""
    dtype = torch.float64
    T, U, _ = robot_points.shape
    robot_points = robot_points.to(dtype)
    if plan is None:
        atoms = measure.points if contact_margin == 0.0 else measure.points + contact_margin * measure.normals
        cost = torch.cdist(atoms, robot_points, compute_mode="donot_use_mm_for_euclid_dist") ** 2
        mismatch = measure.labels.to(torch.int64)[:, :, None] != robot_labels.to(torch.int64)[None, None, :]
        cost = cost + lambda_label * mismatch.to(dtype)
        plan = torch.zeros(T, measure.index.shape[1], U, dtype=dtype)
        if bool(measure.nonempty.any()):
            ne = measure.nonempty
            plan[ne] = ot.sinkhorn_unbalanced_uniform_b(cost[ne], measure.weights[ne], measure.valid[ne], eps, rho_h,
                                                        rho_r, n_sinkhorn)
    plan_t = plan.transpose(1, 2)
    w_u = plan_t.sum(dim=2)
    bary = (plan_t @ torch.cat([measure.points, measure.normals], dim=2)) / w_u.clamp_min(1e-12)[..., None]
    pbar = bary[..., :3]
    if contact_margin != 0.0:
        nbar_raw = bary[..., 3:]
        nbar_norm = nbar_raw.norm(dim=-1, keepdim=True)
        degenerate = nbar_norm.squeeze(-1) < 1e-6
        nbar = nbar_raw / nbar_norm.clamp_min(1e-12)
        if degenerate.any():
            _, grad = sdf_lookup(sdf, pbar)
            nbar = torch.where(degenerate[..., None], grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-12), nbar)
        pbar = pbar + contact_margin * nbar
    mask = w_u >= (1e-6 / U)
    scale_w = torch.sqrt((float(U) * w_u).clamp_min(0.0))
    zero = torch.zeros((), dtype=dtype)
    residual = torch.where(mask[..., None], scale_w[..., None] * (robot_points - pbar), zero)
    if robot_jac is None:
        jac = zero.expand(T, U * 3, n_dof)
    else:
        jac = torch.where(mask[..., None, None], scale_w[..., None, None] * robot_jac.to(dtype), zero)
        jac = jac.reshape(T, U * 3, n_dof)
    weight = torch.repeat_interleave(mask.to(dtype), repeats=3, dim=1)
    return BlockTerm(residual=residual.reshape(T, U * 3), jacobian=jac, weight=weight), plan


def _transport_case(seed: int) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    T, V, U, M = 9, 64, 96, 256
    human_points = torch.rand(T, V, 3, generator=gen, dtype=DT) - 0.5
    weights = torch.exp(-2.0 * torch.rand(T, V, generator=gen, dtype=DT))
    weights = torch.where(torch.rand(T, V, generator=gen, dtype=DT) < 0.8, torch.zeros_like(weights), weights)
    weights[[2, 5, 6]] = 0.0                                                       # frames without contact
    normals = torch.randn(M, 3, generator=gen, dtype=DT)
    human = ContactMeasure(points=human_points, raw_points=human_points, weights=weights,
                           labels=torch.randint(0, 6, (V,), generator=gen).to(torch.int8),
                           sample_idx=torch.randint(0, M, (T, V), generator=gen),
                           sdf_values=torch.zeros(T, V, dtype=DT))
    return {"measure": inband_measure(human, normals / normals.norm(dim=-1, keepdim=True)),
            "robot_points": torch.rand(T, U, 3, generator=gen, dtype=DT) - 0.5,
            "robot_jac": torch.randn(T, U, 3, 28, generator=gen, dtype=DT),
            "robot_labels": torch.randint(0, 6, (U,), generator=gen).to(torch.int8)}


@pytest.mark.parametrize("margin", [0.0, 0.003 / 0.18])
def test_inband_transport_is_bit_identical_to_the_replaced_code(margin: float) -> None:
    case, sdf = _transport_case(0), _box_sdf()
    m = case["measure"]
    assert bool(m.nonempty.any()) and not bool(m.nonempty.all())
    for eps in (0.055, 0.0055):
        args = (case["robot_labels"], eps, 1.1, 0.011, 5.0, 25)
        ref, ref_plan = _reference_inband(m, case["robot_points"], case["robot_jac"], *args, None, margin, sdf)
        got, plan = contact_transport_term_inband(m, case["robot_points"], case["robot_jac"], *args, None, margin, sdf)
        assert torch.equal(plan, ref_plan)
        for f in ("residual", "jacobian", "weight"):
            assert torch.equal(getattr(got, f), getattr(ref, f)), f
        trial_ref, _ = _reference_inband(m, case["robot_points"], None, *args, plan, margin, sdf)
        trial, same = contact_transport_term_inband(m, case["robot_points"], None, *args, plan, margin, sdf)
        assert same is plan and torch.equal(trial.residual, trial_ref.residual)
        # a given plan with an exactly empty column on a frame that has mass: that frame stays live
        holed = plan.clone()
        holed[int(torch.nonzero(m.nonempty)[0]), :, 5] = 0.0
        trial_ref, _ = _reference_inband(m, case["robot_points"], None, *args, holed, margin, sdf)
        trial, _ = contact_transport_term_inband(m, case["robot_points"], None, *args, holed, margin, sdf)
        assert torch.equal(trial.residual, trial_ref.residual) and torch.equal(trial.weight, trial_ref.weight)
        # A5: the Jacobian given only on the non-empty frames gives the very same term.
        ne = m.nonempty
        sub, _ = contact_transport_term_inband(m, case["robot_points"], case["robot_jac"][ne], *args, None, margin,
                                               sdf, robot_jac_frames=ne)
        assert torch.equal(sub.jacobian, ref.jacobian) and torch.equal(sub.residual, ref.residual)


def test_inband_transport_refuses_a_jacobian_subset_missing_a_live_frame() -> None:
    case, sdf = _transport_case(1), _box_sdf()
    m = case["measure"]
    frames = m.nonempty.clone()
    frames[int(torch.nonzero(frames)[0])] = False
    with pytest.raises(ValueError, match="robot_jac_frames"):
        contact_transport_term_inband(m, case["robot_points"], case["robot_jac"][frames], case["robot_labels"], 0.055,
                                      1.1, 0.011, 5.0, 25, None, 0.0, sdf, robot_jac_frames=frames)


def test_inband_transport_scaling_sinkhorn_is_close() -> None:
    case, sdf = _transport_case(2), _box_sdf()
    m = case["measure"]
    args = (case["robot_labels"], 0.0055, 1.1, 0.011, 5.0, 25, None, 0.0167, sdf)
    ref, ref_plan = contact_transport_term_inband(m, case["robot_points"], case["robot_jac"], *args)
    got, plan = contact_transport_term_inband(m, case["robot_points"], case["robot_jac"], *args, sinkhorn="scaling")
    assert float((plan - ref_plan).abs().max()) <= 1e-12
    assert float((got.residual - ref.residual).abs().max()) <= 1e-9


# ---------------------------------------------------------------- A2, A5: canonical Jacobian


def _reference_canonical_jacobian(bk: BatchedKinematics, batch: Any, body_ids: np.ndarray, canon_pts: Tensor,
                                  obj_block: Tensor, mesh_scale: float) -> Tensor:
    """`BatchedKinematics.canonical_jacobian` as of origin/main 4a40535a, verbatim."""
    pu, pd = (torch.from_numpy(a) for a in bk.ancestor_pairs(body_ids))
    rot = quat_wxyz_to_matrix(obj_block[:, 3:7].to(torch.float64))
    pos = obj_block[:, :3].to(torch.float64)
    axis_c = torch.einsum("tji,tdj->tdi", rot, batch.xaxis)
    anchor_c = torch.einsum("tji,tdj->tdi", rot, batch.xanchor - pos[:, None, :]) / mesh_scale
    axis = axis_c[:, pd]
    cols = torch.where(bk._hinge[pd][None, :, None],
                       torch.linalg.cross(axis, canon_pts[:, pu] - anchor_c[:, pd], dim=-1), axis / mesh_scale)
    T, U = canon_pts.shape[0], canon_pts.shape[1]
    jac = torch.zeros(T, U, bk.n_dof, 3, dtype=torch.float64)
    jac[:, pu, pd] = cols
    return jac.permute(0, 1, 3, 2)


@pytest.fixture(scope="module")
def scene() -> RobotScene:
    return build_scene()


@scene_missing
def test_canonical_jacobian_is_bit_identical_and_frame_subsets_agree(scene: RobotScene) -> None:
    rng = np.random.default_rng(0)
    T = 11
    q = np.zeros((T, scene.model.nq))
    q[:, :28] = rng.uniform(-0.4, 0.4, (T, 28))
    q[:, 28:31] = rng.uniform(-0.2, 0.2, (T, 3))
    quat = rng.normal(size=(T, 4))
    q[:, 31:35] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    local, body_ids, _ = scene.surface_points(6, seed=0)
    obj_block = torch.from_numpy(q[:, 28:35])
    bk = BatchedKinematics.from_scene(scene)
    batch = bk.forward(q)
    canon = canonical_points(bk.world_points(batch, body_ids, local), obj_block, 0.18)
    ref = _reference_canonical_jacobian(bk, batch, body_ids, canon, obj_block, 0.18)
    got = bk.canonical_jacobian(batch, body_ids, canon, obj_block, 0.18)
    assert got.is_contiguous() and torch.equal(got, ref)
    frames = torch.zeros(T, dtype=torch.bool)
    frames[[1, 4, 5, 9]] = True
    sub = bk.canonical_jacobian(batch, body_ids, canon[frames], obj_block[frames], 0.18, frames=frames)
    assert torch.equal(sub, ref[frames])


# ---------------------------------------------------------------- A4: one assembly per accepted state


def _lm_problem() -> tuple[Tensor, Any]:
    gen = torch.Generator().manual_seed(0)
    T, n = 6, 3
    target = torch.randn(T, n, generator=gen, dtype=DT)

    def evaluate(x: Tensor, refresh: bool) -> tuple[list[BlockTerm], SmoothTerm | None]:
        r = torch.cat([torch.sin(3.0 * x) - target, 0.1 * x ** 3], dim=1)            # rejects happen
        jac = torch.cat([torch.diag_embed(3.0 * torch.cos(3.0 * x)), torch.diag_embed(0.3 * x ** 2)], dim=1)
        smooth = SmoothTerm(residual=x[1:] - x[:-1], jac_k=-torch.eye(n, dtype=DT).expand(T - 1, n, n),
                            jac_k1=torch.eye(n, dtype=DT).expand(T - 1, n, n), weight=torch.ones(T - 1, n, dtype=DT))
        return [BlockTerm(residual=r, jacobian=jac, weight=torch.ones_like(r))], smooth

    return 2.0 * torch.randn(T, n, generator=gen, dtype=DT), evaluate


def test_lm_assembles_once_per_accepted_state_and_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    x0, evaluate = _lm_problem()
    calls = {"assemble": 0, "refresh": 0}
    real = lm_blocks.assemble_block_system

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls["assemble"] += 1
        return real(*args, **kwargs)

    def counted_eval(x: Tensor, refresh: bool) -> Any:
        calls["refresh"] += int(refresh)
        return evaluate(x, refresh)

    monkeypatch.setattr(lm_blocks, "assemble_block_system", counting)
    cfg = BlockLMConfig(max_iter=40, rel_tol=0.0)
    bounds = (torch.full((3,), -1.2, dtype=DT), torch.full((3,), 1.2, dtype=DT))
    for b in (None, bounds):
        calls.update(assemble=0, refresh=0)
        got = lm_optimize_blocks(x0, counted_eval, cfg, bounds=b)
        assert got.iterations > calls["assemble"]                               # some trials were rejected
        assert calls["refresh"] - 1 <= calls["assemble"] <= calls["refresh"]  # one per accepted state (not the last)

        ref = _reference_lm(x0, evaluate, cfg, b)                               # assembles every iteration
        assert torch.equal(got.x, ref.x) and got.cost == ref.cost and got.iterations == ref.iterations


def _reference_lm(x0: Tensor, evaluate: Any, cfg: BlockLMConfig, bounds: Any) -> lm_blocks.BlockLMResult:
    """`lm_optimize_blocks` as of origin/main 4a40535a (assembly every iteration), verbatim but for names."""
    x = x0.clone()
    n = x.shape[1]
    box = None if bounds is None else lm_blocks._box(bounds, n, x)
    if box is not None:
        x = torch.maximum(torch.minimum(x, box[1]), box[0])
    lam = cfg.lam0
    terms, smooth = evaluate(x, True)
    cost = lm_blocks.total_block_cost(terms, smooth)
    it = 0
    for it in range(1, cfg.max_iter + 1):
        diag, offdiag, g = lm_blocks.assemble_block_system(terms, smooth, x.shape[0], n)
        if box is not None:
            active = ((x <= box[0]) & (g > 0.0)) | ((x >= box[1]) & (g < 0.0))
            diag, offdiag, g = lm_blocks.freeze_active_bounds(diag, offdiag, g, active)
        delta = lm_blocks.solve_block_tridiagonal(diag, offdiag, g, lam, blas_threads=cfg.blas_threads)
        trial = x + delta.to(x.dtype)
        if box is not None:
            trial = torch.maximum(torch.minimum(trial, box[1]), box[0])
        t_terms, t_smooth = evaluate(trial, False)
        trial_cost = lm_blocks.total_block_cost(t_terms, t_smooth)
        if trial_cost < cost:
            rel = (cost - trial_cost) / max(cost, 1e-12)
            x = trial
            lam = max(lam / cfg.lam_down, 1e-9)
            terms, smooth = evaluate(x, True)
            cost = lm_blocks.total_block_cost(terms, smooth)
            if rel < cfg.rel_tol:
                break
        else:
            lam *= cfg.lam_up
            if lam > 1e6:
                break
    return lm_blocks.BlockLMResult(x=x, cost=cost, iterations=it)


# ---------------------------------------------------------------- end to end on the whisking demo


@whisking_data_missing
def test_scaling_sinkhorn_whisking_reference_within_float_tolerance(scene: RobotScene) -> None:
    sdf, surface, normals = build_whisk_sdf_and_surface()
    inputs = load_whisking_inputs(np.arange(40))
    out, calls = {}, {}
    for mode in ("log", "scaling"):
        cfg = RetargetConfig(sinkhorn=mode)
        before = ot._SINKHORN_STATS["calls"]
        out[mode] = retarget_kinematic(inputs["obj_poses"], inputs["wrist_poses"], MESH_SCALE, inputs["hand_verts"],
                                       inputs["hand_labels"], scene, sdf, surface, normals, 50.0, cfg).qpos
        calls[mode] = ot._SINKHORN_STATS["calls"] - before
    assert calls["log"] == 0 and calls["scaling"] > 0                            # the option reaches the solver
    assert float(np.abs(out["scaling"] - out["log"]).max()) <= 1e-8


def test_scaling_sinkhorn_overflow_without_absorption_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """With re-absorption off and a tiny eps the scalings overflow to inf; those elements must be re-solved."""
    C, a, valid = _ot_problem(4, B=6)
    C = C.clone()
    C[:, :, 1:] += 50.0                                                            # column 0 is far cheaper
    ref = ot.sinkhorn_unbalanced_uniform_b(C, a, valid, 1e-3, 2.0, 0.02, 25)
    monkeypatch.setattr(ot, "_ABSORB_DRIFT", math.inf)
    before = ot._SINKHORN_STATS["fallbacks"]
    got = ot.sinkhorn_unbalanced_uniform_b_scaling(C, a, valid, 1e-3, 2.0, 0.02, 25)
    assert ot._SINKHORN_STATS["fallbacks"] > before
    assert torch.isfinite(got).all() and float((got - ref).abs().max()) <= 1e-12 * float(ref.abs().max())


def test_scaling_sinkhorn_checks_the_column_sums(monkeypatch: pytest.MonkeyPatch) -> None:
    """One iteration from the absorbed zero start: the row sums are exactly normalized, so only the column check can
    send the elements to the log domain."""
    C, a, valid = _ot_problem(5, B=4)
    monkeypatch.setattr(ot, "_SCALING_RANGE", 0.9)
    before = ot._SINKHORN_STATS["fallbacks"]
    got = ot.sinkhorn_unbalanced_uniform_b_scaling(C, a, valid, 0.01, 2.0, 0.02, 1)
    assert ot._SINKHORN_STATS["fallbacks"] - before == C.shape[0]
    assert torch.equal(got, ot.sinkhorn_unbalanced_uniform_b(C, a, valid, 0.01, 2.0, 0.02, 1))
