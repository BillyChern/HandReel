"""Tracker-model CPU tests N1-N9: apparent-size track normalization, frame/track/point/cluster tokens, host reads."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from torch import Tensor

from fastdo.geometry.lie import se3_exp
from fastdo.solver.banded import ClipLayout
from fastdo.solver.terms import TermOutput, pointmap_term
from fastdo.tracker import features, normalize
from fastdo.tracker.clusters import mesh_cluster_ids
from fastdo.tracker.features import build_frame_tokens, plan_row_mass, track_tokens
from fastdo.tracker.normalize import (
    CLUSTER_TOKEN_DIM,
    POINT_TOKEN_DIM,
    cluster_geometry,
    cluster_tokens,
    normalize_frame_tokens,
    normalize_track_tokens,
    point_tokens,
    track_image_scale,
)
from fastdo.tracker.pack import (
    TrackerInputs,
    build_tracker_inputs,
    clips_to_frames,
    frames_to_clips,
)
from tests.tracker.synthetic_terms import CAUCHY_C, TermsFixture, make_terms_fixture
from tests.tracker.tracker_fixtures import (
    base_scene,
    fixture_tracker_inputs,
    max_rel,
    pixel_rescaled_scene,
    scaled_scene,
)

F64 = torch.float64
D_IMG_TABLE = [36.986483, 36.329275, 35.578917, 35.002113, 34.818632, 34.881486]
TR_DIAG_MAX = [1.0816793, 1.0853016, 0.010579794, 0.023968472, 0.023244275, 0.081493875, 0.087704176]
PM_DIAG_MAX = [0.92024661, 0.92024662, 0.92024660, 0.046094779, 0.084995570, 0.082132479, 0.10449325]


def _length(fx: TermsFixture) -> Tensor:
    return fx.log_scale.exp() * fx.template_diameter


def _d_ref(fx: TermsFixture, poses: Tensor | None = None) -> Tensor:
    """Independent Python-float D_img = sqrt(fx fy) L / max(z, 0.05 L) per frame."""
    poses = fx.poses if poses is None else poses
    f, L = math.sqrt(float(fx.K[0, 0]) * float(fx.K[1, 1])), float(_length(fx))
    return torch.tensor([f * L / max(float(poses[k, 2, 3]), 0.05 * L) for k in range(poses.shape[0])], dtype=F64)


def _elem_rel(a: Tensor, ref: Tensor) -> float:
    return float(((a - ref).abs() / ref.abs()).max())


def test_track_image_scale_definition() -> None:
    fx = make_terms_fixture()
    L = _length(fx)
    d = track_image_scale(fx.K, fx.poses, L)
    assert d.shape == (6,) and d.dtype == F64
    assert max_rel(d, _d_ref(fx)) <= 1e-12
    assert _elem_rel(d, torch.tensor(D_IMG_TABLE, dtype=F64)) <= 1e-6
    assert torch.equal(track_image_scale(fx.K.expand(6, 3, 3), fx.poses, L), d)
    assert torch.equal(track_image_scale(fx.K, fx.poses, float(L)), d)
    assert torch.equal(track_image_scale(fx.K, fx.poses, L.expand(6)), d)
    poses = fx.poses.clone()
    poses[1, 2, 3] = -0.2
    d2 = track_image_scale(fx.K, poses, L)
    assert bool(torch.isfinite(d2).all())
    assert math.isclose(float(d2[1]), 150.0 / 0.05, rel_tol=1e-12)
    assert max_rel(d2, _d_ref(fx, poses)) <= 1e-12


def test_normalized_track_block_commensurate_with_pointmap() -> None:
    fx = make_terms_fixture()
    tok = build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, mesh_cluster_ids(fx.Y, 64, 0), fx.template_diameter,
                             fx.poses, fx.log_scale)
    frame_n, ok = normalize_frame_tokens(tok, track_image_scale(fx.K, fx.poses, _length(fx)))
    assert bool(ok.all())
    d2 = _d_ref(fx)[:, None] ** 2
    torch.testing.assert_close(torch.expm1(frame_n[:, 21:28]) * 1e-3, tok[:, 21:28] / d2, rtol=1e-10, atol=1e-15)
    torch.testing.assert_close(torch.sinh(frame_n[:, 14:21]) * 1e-3, tok[:, 14:21] / d2, rtol=1e-10, atol=1e-15)
    assert _elem_rel((tok[:, 21:28] / d2).amax(0), torch.tensor(TR_DIAG_MAX, dtype=F64)) <= 1e-6
    assert _elem_rel(tok[:, 7:14].amax(0), torch.tensor(PM_DIAG_MAX, dtype=F64)) <= 1e-6
    pm_max = tok[:, 7:10].max()
    ratio = float((torch.expm1(frame_n[:, 21:23]) * 1e-3).max() / pm_max)
    assert 0.5 <= ratio <= 2.0 and math.isclose(ratio, 1.1793595, rel_tol=1e-6)
    for wrong in (tok[:, 21:23] / _d_ref(fx)[:, None], tok[:, 21:23] / CAUCHY_C**2):   # D_img unsquared, cauchy_c^2
        assert not 0.5 <= float(wrong.max() / pm_max) <= 2.0


@pytest.mark.parametrize("b", [0.5, 2.0])
def test_normalized_tokens_invariant_to_pixel_rescale(b: float) -> None:
    fx = make_terms_fixture()
    ids, L = mesh_cluster_ids(fx.Y, 64, 0), _length(fx)
    normalized = []
    for s in (base_scene(fx), pixel_rescaled_scene(fx, b)):
        tok = build_frame_tokens(s.pm, s.tr_raw, fx.plan, ids, fx.template_diameter, s.poses, s.log_scale)
        d = track_image_scale(s.K, s.poses, L)
        frame_n, ok = normalize_frame_tokens(tok, d)
        assert bool(ok.all())
        track_n = normalize_track_tokens(track_tokens(fx.anchors, s.tr_raw, fx.template_diameter), d)
        normalized.append((tok, frame_n, track_n))
    (tok0, n0, t0), (tok1, n1, t1) = normalized
    assert max_rel(tok1[:, 21:28], b**2 * tok0[:, 21:28]) <= 1e-12
    assert math.isclose(float(tok1[:, 21:28].max()), b**2 * float(tok0[:, 21:28].max()), rel_tol=1e-12)
    for lo, hi in ((0, 7), (7, 14), (14, 21), (21, 28), (28, 44), (44, 108), (108, 110)):
        assert max_rel(n1[:, lo:hi], n0[:, lo:hi]) <= 1e-12, (lo, hi)
    for c in range(6):
        assert max_rel(t1[..., c], t0[..., c]) <= 1e-12, c


@pytest.mark.parametrize("a", [0.5, 2.0])
@pytest.mark.parametrize("knn", [None, 16], ids=["dense", "sparse"])
def test_normalized_inputs_invariant_to_global_similarity(knn: int | None, a: float) -> None:
    fx = make_terms_fixture(knn=knn)
    s = scaled_scene(fx, a)
    ref, out = fixture_tracker_inputs(fx), fixture_tracker_inputs(fx, s)
    for field in dataclasses.fields(TrackerInputs):
        x, y = getattr(out, field.name), getattr(ref, field.name)
        if y.dtype == torch.bool:
            assert torch.equal(x, y), field.name
        else:
            assert max_rel(x, y) <= 1e-12, (field.name, max_rel(x, y))
    assert bool(ref.frame_mask.all())
    valid = fx.X_valid
    assert torch.equal(s.X[valid], a * fx.X[valid]) and not torch.equal(s.X[valid], fx.X[valid])
    assert torch.equal(s.poses[:, :3, 3], a * fx.poses[:, :3, 3])


def test_point_tokens_invariant_to_camera_move() -> None:
    fx = make_terms_fixture()
    G = se3_exp(torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6], dtype=F64))
    X2 = fx.X @ G[:3, :3].T + G[:3, 3]
    poses2 = G @ fx.poses
    pm2 = pointmap_term(X2, fx.X_valid, fx.Y, poses2, fx.log_scale, fx.eps, fx.rho_x, fx.rho_y, 0, plan=fx.plan)[0]
    rm, L = plan_row_mass(fx.plan), _length(fx)
    p1 = point_tokens(fx.X, fx.pm, rm, fx.poses, L)
    p2 = point_tokens(X2, pm2, rm, poses2, L)
    assert p1.shape == (6, 96, POINT_TOKEN_DIM) and p1.dtype == F64
    assert max_rel(p2, p1) <= 1e-9
    assert float((pm2.residual - fx.pm.residual).abs().max()) > 0.1 * float(fx.pm.residual.abs().max())


def test_point_tokens_definition() -> None:
    N, L = 4, 0.5
    poses = torch.eye(4, dtype=F64)[None].clone()
    poses[0, 2, 3] = 1.0
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(1, N, 3, generator=gen, dtype=F64)
    X[0, 2] = float("nan")                                  # invalid row: NaN allowed
    residual = 0.01 * torch.randn(1, N, 3, generator=gen, dtype=F64)
    residual[0, 2] = 0.0
    weight = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=F64)
    row_mass = torch.tensor([[1 / N, 0.0, 0.0, 0.5 / N]], dtype=F64)
    pm = TermOutput(residual, torch.zeros(1, N, 3, 7, dtype=F64), weight)
    out = point_tokens(X, pm, row_mass, poses, L)
    expected = torch.zeros(1, N, 7, dtype=F64)
    for i in range(N):
        if float(weight[0, i]) <= 0:
            continue
        m = N * float(row_mass[0, i])
        y = [(float(X[0, i, c]) - float(poses[0, c, 3])) / L for c in range(3)]
        rc = [float(residual[0, i, c]) / (math.sqrt(m) * L) if m > 1e-12 else 0.0 for c in range(3)]
        expected[0, i] = torch.tensor(y + [math.asinh(v / 1e-3) for v in rc] + [m], dtype=F64)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(out[0, 2], torch.zeros(7, dtype=F64)) and torch.equal(out[0, 1, 3:], torch.zeros(4, dtype=F64))
    bad_res = TermOutput(torch.zeros(1, N, 1, dtype=F64), torch.zeros(1, N, 1, 7, dtype=F64), weight)
    for args in ((X, bad_res, row_mass), (torch.zeros(1, N + 1, 3, dtype=F64), pm, row_mass),
                 (X, pm, torch.zeros(1, N + 1, dtype=F64))):
        with pytest.raises(ValueError):
            point_tokens(args[0], args[1], args[2], poses, L)


def test_normalize_frame_tokens_squash_and_finite_mask() -> None:
    x = torch.rand(4, 110, generator=torch.Generator().manual_seed(0), dtype=F64)
    x[0, 0:14] = 1e-3
    x[0, 14:28] = 4e-3
    x[1, 50] = float("nan")
    x[2, 3] = float("inf")
    d = torch.tensor([2.0, 2.0, 2.0, float("nan")], dtype=F64)
    out, finite = normalize_frame_tokens(x, d)
    assert finite.dtype == torch.bool and finite.tolist() == [True, False, False, False]
    assert abs(math.asinh(1.0) - 0.881373587) < 1e-9
    for lo, hi, value in ((0, 7, math.asinh(1.0)), (7, 14, math.log(2.0)), (14, 21, math.asinh(1.0)),
                          (21, 28, math.log(2.0))):
        torch.testing.assert_close(out[0, lo:hi], torch.full((7,), value, dtype=F64), rtol=1e-12, atol=0.0)
    assert torch.equal(out[0, 28:], x[0, 28:])
    assert torch.equal(out[1:], torch.zeros(3, 110, dtype=F64))
    assert normalize._INPUT_FLOOR == 10.0 ** features._LOG10_LO
    assert normalize._MASS_FLOOR == features._MASS_FLOOR
    with pytest.raises(ValueError):
        normalize_frame_tokens(x[:, :109], d)


def test_cluster_geometry_and_tokens() -> None:
    fx = make_terms_fixture(knn=16)
    ids = mesh_cluster_ids(fx.Y, 64, 0)
    centroids, fraction = cluster_geometry(fx.Y, ids, fx.template_diameter)
    assert centroids.shape == (64, 3) and fraction.shape == (64,) and centroids.dtype == fraction.dtype == F64
    for c in range(64):
        members = fx.Y[ids == c]
        ref = members.mean(0) / fx.template_diameter if members.shape[0] else torch.zeros(3, dtype=F64)
        torch.testing.assert_close(centroids[c], ref, rtol=1e-12, atol=1e-15)
        assert float(fraction[c]) == members.shape[0] / 256
    assert abs(float(fraction.sum()) - 1.0) <= 1e-12
    ids2 = torch.where(ids == 63, torch.full_like(ids, 62), ids)
    c2, f2 = cluster_geometry(fx.Y, ids2, fx.template_diameter)
    assert torch.equal(c2[63], torch.zeros(3, dtype=F64)) and float(f2[63]) == 0.0
    packed = fixture_tracker_inputs(fx, cluster_id=ids2)
    assert not bool(packed.cluster_mask[..., 63].any())
    assert torch.equal(packed.cluster_mask[0], (f2 > 0).expand(6, 64)) and bool(packed.cluster_mask.any())
    tok = build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, ids, fx.template_diameter, fx.poses, fx.log_scale)
    ct = cluster_tokens(tok, centroids, fraction)
    assert ct.shape == (6, 64, CLUSTER_TOKEN_DIM) and ct.dtype == F64
    assert torch.equal(ct[..., 0], tok[:, 44:108])
    assert torch.equal(ct[..., 1:4], centroids.expand(6, 64, 3)) and torch.equal(ct[..., 4], fraction.expand(6, 64))
    assert torch.equal(cluster_tokens(tok, centroids.expand(6, 64, 3), fraction.expand(6, 64)), ct)


def test_input_builders_read_nothing_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    fx = make_terms_fixture(knn=16)
    ids, L = mesh_cluster_ids(fx.Y, 64, 0), _length(fx)
    tok = build_frame_tokens(fx.pm, fx.tr_raw, fx.plan, ids, fx.template_diameter, fx.poses, fx.log_scale)
    track = track_tokens(fx.anchors, fx.tr_raw, fx.template_diameter)
    layout = ClipLayout.build((6,), "cpu")

    def run() -> list[Tensor]:
        d = track_image_scale(fx.K, fx.poses, L)
        frame_n, ok = normalize_frame_tokens(tok, d)
        track_n = normalize_track_tokens(track, d)
        pt = point_tokens(fx.X, fx.pm, plan_row_mass(fx.plan), fx.poses, L)
        cen, frac = cluster_geometry(fx.Y, ids, fx.template_diameter)
        ct = cluster_tokens(tok, cen, frac)
        clips = frames_to_clips(pt, layout)
        inputs = build_tracker_inputs(tok, track, fx.pm, fx.X, fx.plan, fx.poses, fx.log_scale,
                                      fx.template_diameter, fx.K, cen, frac)
        return [d, frame_n, ok, track_n, pt, cen, frac, ct, clips, clips_to_frames(clips, layout),
                *(getattr(inputs, f.name) for f in dataclasses.fields(TrackerInputs))]

    ref = run()

    def host_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("host read")

    with monkeypatch.context() as m:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__float__", "__int__"):
            m.setattr(torch.Tensor, name, host_read)
        out = run()
    assert len(out) == len(ref) == 18
    for a, b in zip(out, ref, strict=True):
        assert torch.equal(a, b)
