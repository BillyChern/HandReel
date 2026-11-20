"""Known-answer and mutation tests for fastdo.bench.metric_frame (frame-metric scorers).

Each scorer is tested on a construction whose answer is known in closed form, AND on a mutation it must detect
(a check that only accepts is not a check: feedback-verify-physically-not-by-proxy, question 2).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from fastdo.bench.alignment import umeyama_similarity
from fastdo.bench.metric_frame import (
    along_ray_components,
    apply_gauge,
    cd_h_hold,
    gauge_decomposition,
    hold_chamfer_cm2,
    pairwise_object_errors,
)

# A local copy of HOLD's evaluation module. It is not shipped: its licence is not verified in this repository.
# Only the equivalence test below reads it, and only when FASTDO_HOLD_EVAL names it.
HOLD_EVAL_ENV = "FASTDO_HOLD_EVAL"


def _traj(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([0.05, -0.03, 0.55])
    return base + 0.08 * rng.standard_normal((n, 3)) * np.array([1.0, 1.0, 0.5])


def _rot(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _poses(centres: np.ndarray, seed: int = 0) -> np.ndarray:
    out = np.tile(np.eye(4), (len(centres), 1, 1))
    for t in range(len(centres)):
        out[t, :3, :3] = _rot(seed + t)
    out[:, :3, 3] = centres
    return out


# ---------------------------------------------------------------- along-ray decomposition

def test_along_ray_gauge_is_purely_along_the_ray() -> None:
    gt = _traj()
    for lam in (0.8, 1.25):
        ar = along_ray_components(lam * gt, gt)
        np.testing.assert_allclose(ar.signed_along, (lam - 1.0) * np.linalg.norm(gt, axis=1), atol=1e-12)
        np.testing.assert_allclose(ar.perp, 0.0, atol=1e-12)
        np.testing.assert_allclose(ar.log_depth_ratio, np.log(lam), atol=1e-12)


def test_along_ray_perpendicular_shift_has_no_along_component() -> None:
    gt = _traj()
    u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    perp_dir = np.cross(u, np.array([0.0, 1.0, 0.0]))
    perp_dir /= np.linalg.norm(perp_dir, axis=1, keepdims=True)
    ar = along_ray_components(gt + 0.03 * perp_dir, gt)
    np.testing.assert_allclose(ar.signed_along, 0.0, atol=1e-12)
    np.testing.assert_allclose(ar.perp, 0.03, atol=1e-12)


def test_along_ray_signs_far_positive() -> None:
    gt = _traj()
    u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    assert np.all(along_ray_components(gt + 0.05 * u, gt).signed_along > 0.0499)
    assert np.all(along_ray_components(gt - 0.05 * u, gt).signed_along < -0.0499)


def test_along_ray_rejects_camera_centre() -> None:
    with pytest.raises(ValueError):
        along_ray_components(np.ones((1, 3)), np.zeros((1, 3)))


# ---------------------------------------------------------------- gauge-orbit decomposition

def test_gauge_decomposition_recovers_lambda_and_zero_off_orbit() -> None:
    gt = _traj()
    poses, s = apply_gauge(_poses(gt), 0.12, 1.17)
    dec = gauge_decomposition(poses[:, :3, 3], gt, pred_size=s, gt_size=0.12)
    assert dec.lam == pytest.approx(1.17, abs=1e-12)
    assert dec.off_orbit_rms == pytest.approx(0.0, abs=1e-12)
    assert dec.orbit_coherence == pytest.approx(0.0, abs=1e-12)
    assert dec.on_orbit_rms > 0.05


def test_gauge_decomposition_flags_right_size_wrong_distance() -> None:
    """Production's failure: object at the right size but 50 mm too far on every frame."""
    gt = _traj()
    u = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    dec = gauge_decomposition(gt + 0.05 * u, gt, pred_size=0.12, gt_size=0.12)
    assert dec.log_size_ratio == pytest.approx(0.0, abs=1e-12)
    assert dec.orbit_coherence < -0.05            # distance moved, size did not: off the orbit
    assert dec.off_orbit_rms > 0.0                # depths vary over the trajectory, so lam cannot absorb it all


def test_gauge_decomposition_ignores_nan_frames() -> None:
    gt = _traj()
    pred = 1.1 * gt
    pred[3] = np.nan
    dec = gauge_decomposition(pred, gt)
    assert dec.n_frames == len(gt) - 1
    assert dec.lam == pytest.approx(1.1, abs=1e-12)


def test_scaled_alignment_is_blind_to_the_gauge_and_along_ray_is_not() -> None:
    """The protocol-blindness known answer (G3): a similarity fit removes g_lambda exactly; our scorer does not."""
    rng = np.random.default_rng(3)
    obj = 0.05 * rng.standard_normal((500, 3)) + np.array([0.0, 0.0, 0.6])
    lam = 1.2
    s, r, t = umeyama_similarity(lam * obj, obj)
    aligned = s * (lam * obj) @ r.T + t
    assert np.max(np.linalg.norm(aligned - obj, axis=1)) < 1e-9          # ICP-style score: perfect
    ar = along_ray_components((lam * obj).mean(0, keepdims=True), obj.mean(0, keepdims=True))
    assert ar.signed_along[0] == pytest.approx(0.2 * np.linalg.norm(obj.mean(0)), rel=1e-12)  # ours: 120 mm


def test_apply_gauge_rejects_bad_lambda() -> None:
    with pytest.raises(ValueError):
        apply_gauge(np.tile(np.eye(4), (2, 1, 1)), 0.1, 0.0)


# ---------------------------------------------------------------- CD_h (HOLD-compatible)

def _hand_object(n_t: int = 6, n_v: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obj = 0.04 * rng.standard_normal((n_v, 3))
    centres = _traj(n_t, seed)
    verts = np.stack([obj + c for c in centres])
    wrist = centres + np.array([0.0, 0.08, 0.0])
    return verts, wrist


def test_cd_h_zero_for_identical_and_for_shared_translation() -> None:
    verts, wrist = _hand_object(n_v=3000)                 # <= 3000 vertices: randperm[:3000] keeps every vertex
    v, per = cd_h_hold(verts, wrist, verts, wrist)
    assert v == 0.0 and np.all(per == 0.0)
    shift = np.array([0.03, -0.02, 0.07])                 # hand AND object move together: HOLD cancels it
    v2, _ = cd_h_hold(verts + shift, wrist + shift, verts, wrist)
    assert v2 == pytest.approx(0.0, abs=1e-12)


def test_cd_h_has_a_subsample_floor_above_3000_vertices() -> None:
    """HOLD draws 3,000 vertices per side INDEPENDENTLY, so a perfect prediction of a denser mesh scores > 0.

    This floor is part of the published metric (4D-HOF's 5.98 cm^2 includes it); it is reported per mesh.
    """
    verts, wrist = _hand_object(n_v=4000)
    v, _ = cd_h_hold(verts, wrist, verts, wrist)
    assert 0.0 < v < 1.0


def test_cd_h_sees_a_hand_object_along_ray_gap() -> None:
    """Single-vertex object: an object-only shift of d metres gives exactly 2 (100 d)^2 cm^2."""
    gt_v = np.array([[[0.0, 0.0, 0.6]]])
    gt_w = np.array([[0.0, 0.05, 0.6]])
    v, _ = cd_h_hold(gt_v + np.array([0.0, 0.0, 0.03]), gt_w, gt_v, gt_w)
    assert v == pytest.approx(2 * 3.0 ** 2, rel=1e-12)


def test_cd_h_invalid_and_nan_frames() -> None:
    verts, wrist = _hand_object()
    pred = verts.copy()
    pred[2] = np.nan
    valid = np.ones(len(verts), dtype=bool)
    valid[4] = False
    v, per = cd_h_hold(pred + 0.001, wrist, verts, wrist, gt_valid=valid)
    assert np.isnan(per[2]) and np.isnan(per[4])
    assert np.isfinite(per[[0, 1, 3, 5]]).all()
    assert v == pytest.approx(np.nanmean(per))


def _hold_reference(hold_eval: Path) -> object:
    """HOLD's calculate_chamfer_f_scores, extracted from the cached MIT source (never vendored)."""
    tree = ast.parse(hold_eval.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "calculate_chamfer_f_scores")
    namespace: dict = {"np": np}
    from scipy.spatial import KDTree

    namespace["KDTree"] = KDTree
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(hold_eval), "exec"), namespace)  # noqa: S102
    return namespace["calculate_chamfer_f_scores"]


HOLD_EVAL_SHA256 = "6508454deeb2bb2c876bdc2628d48d2bb2c428a2c8ecc252513966eb65eb8b45"


def test_cd_h_matches_hold_bit_for_bit() -> None:
    """Re-run HOLD's eval_cd_f_right loop (seed once, randperm pred then gt, skip NaN frames) on the same data.

    The HOLD source is not shipped, because its licence is not verified in this repository. With `FASTDO_HOLD_EVAL`
    unset the test skips and says what to set. Once it is set, the test FAILS (never skips) when the file is absent or
    changed: a silently skipped equivalence test is how an unverified scorer reaches a paper table."""
    import hashlib

    import torch

    if not os.environ.get(HOLD_EVAL_ENV):
        pytest.skip(f"HOLD's evaluation source is not shipped (licence not verified here): set {HOLD_EVAL_ENV} to a "
                    f"local copy of it (sha256 {HOLD_EVAL_SHA256}) to run this equivalence test")
    hold_eval = Path(os.environ[HOLD_EVAL_ENV])
    if not hold_eval.is_file():
        pytest.fail(f"{HOLD_EVAL_ENV}={hold_eval} is not a file (copy HOLD's evaluation module there; sha256 "
                    f"{HOLD_EVAL_SHA256})")
    assert hashlib.sha256(hold_eval.read_bytes()).hexdigest() == HOLD_EVAL_SHA256, "HOLD reference source changed"
    ref_fn = _hold_reference(hold_eval)
    verts, wrist = _hand_object(n_t=5, n_v=3500, seed=7)
    gt_verts, gt_wrist = _hand_object(n_t=5, n_v=5200, seed=8)
    pred = verts.copy()
    pred[1] = np.nan
    rel_p = torch.from_numpy(pred - wrist[:, None, :])
    rel_g = torch.from_numpy(gt_verts - gt_wrist[:, None, :])
    torch.manual_seed(1)
    expected = []
    for t in range(len(rel_p)):
        if torch.isnan(rel_p[t].mean()):
            expected.append(float("nan"))
            continue
        ip = torch.randperm(rel_p[t].shape[0])[:3000]
        ig = torch.randperm(rel_g[t].shape[0])[:3000]
        expected.append(float(ref_fn(rel_p[t][ip].numpy(), rel_g[t][ig].numpy())[0]))
    _, per = cd_h_hold(pred, wrist, gt_verts, gt_wrist)
    np.testing.assert_array_equal(np.isnan(per), np.isnan(expected))
    np.testing.assert_array_equal(per[np.isfinite(per)], np.asarray(expected)[np.isfinite(expected)])


def test_hold_chamfer_units_are_cm2() -> None:
    a = np.zeros((1, 3))
    b = np.array([[0.01, 0.0, 0.0]])                     # 1 cm apart
    assert hold_chamfer_cm2(a, b) == pytest.approx(2.0)


# ---------------------------------------------------------------- frame-free pairwise errors (V2D)

def test_pairwise_errors_zero_and_rigid_invariant() -> None:
    a = _poses(_traj(10, 1), seed=1)
    b = _poses(_traj(10, 2), seed=50)
    out = pairwise_object_errors(a, b, a, b)
    assert np.all(out["dist_err"] == 0.0) and np.allclose(out["rel_pos_err"], 0.0, atol=1e-15)
    w = np.eye(4)
    w[:3, :3] = _rot(99)
    w[:3, 3] = [0.3, -1.0, 2.0]                          # an arbitrary prediction frame (the unknown camera)
    moved = pairwise_object_errors(w @ a, w @ b, a, b)
    np.testing.assert_allclose(moved["dist_err"], 0.0, atol=1e-12)
    np.testing.assert_allclose(moved["rel_pos_err"], 0.0, atol=1e-12)


def test_pairwise_errors_detect_the_gauge_and_one_object_along_ray() -> None:
    a = _poses(_traj(10, 1), seed=1)
    b = _poses(_traj(10, 2), seed=50)
    lam = 0.9
    ga, _ = apply_gauge(a, 0.1, lam)
    gb, _ = apply_gauge(b, 0.1, lam)
    out = pairwise_object_errors(ga, gb, a, b)
    np.testing.assert_allclose(out["dist_err"], (lam - 1.0) * out["dist_true"], atol=1e-12)
    u = a[:, :3, 3] / np.linalg.norm(a[:, :3, 3], axis=1, keepdims=True)
    shifted = a.copy()
    shifted[:, :3, 3] += 0.05 * u                        # A alone 50 mm too far
    assert np.all(pairwise_object_errors(shifted, b, a, b)["rel_pos_err"] > 0.049)
