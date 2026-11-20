"""Window adapter, AR1D and config: known answers with mutations that fire."""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.smooth import (
    SmootherConfig,
    alongray_ranges,
    row_spacing,
    smooth_window,
    smoother_from_mapping,
)
from tests.smooth.synth import along, along_jitter, perp, rms, twist_trajectory


def _poses(R: np.ndarray, c: np.ndarray, s: float, p_bar: np.ndarray) -> np.ndarray:
    P = np.tile(np.eye(4), (len(c), 1, 1))
    P[:, :3, :3] = R
    P[:, :3, 3] = c - s * R @ p_bar
    return P


def test_ar1d_lemma1_weighted_range_mean_is_preserved_without_the_hold() -> None:
    """Design Lemma 1: diffuse CV prior => sum_k w_k (r_hat_k - r_k) = 0 for the weights of the final solve."""
    _, c = twist_trajectory(T=50)
    cm = along_jitter(c, 0.07, 0.0)
    for rw in (0, 3):
        out, gamma, _ = alongray_ranges(cm, np.ones(49), 0.0533, 0.25, reweights=rw, gauge_hold=False)
        assert abs(gamma) < 1e-8
        assert rms(along(out, c)) < 0.6 * rms(along(cm, c))
        assert np.allclose(perp(out, cm), 0.0, atol=1e-12)          # moves along its own ray only
    # mutation: an informative (non-diffuse) initial state breaks the identity
    _, g_bad, _ = alongray_ranges(cm + 0.0, np.ones(49), 0.0533, 0.25, reweights=0, gauge_hold=False,
                                  diffuse=0.05)
    assert abs(g_bad) > 1e-5


def test_ar1d_is_gauge_equivariant() -> None:
    _, c = twist_trajectory(T=30)
    cm = along_jitter(c, 0.07, 0.0)
    a, _, _ = alongray_ranges(cm, np.ones(29), 0.0533, 0.25)
    b, _, _ = alongray_ranges(1.1 * cm, np.ones(29), 0.0533, 0.25)
    assert np.abs(b - 1.1 * a).max() < 1e-10


def test_row_spacing() -> None:
    assert np.array_equal(row_spacing(np.array([0, 3, 6, 12, 15])), np.array([1.0, 1.0, 2.0, 1.0]))
    with pytest.raises(ValueError):
        row_spacing(np.array([0, 3, 3]))


@pytest.mark.parametrize("arm", ["se3", "alongray", "se3_contact", "alongray_contact"])
def test_window_keeps_scale_missing_rows_and_a_null_space_trajectory(arm: str) -> None:
    xi = (0.0, 0.0, 0.0, 0.002, 0.001, 0.003) if arm == "se3_contact" else (0.0, 0.04, 0.01, 0.002, 0.001, 0.0)
    R, c = twist_trajectory(xi=xi, wobble=(0, 0))           # the contact arm: both bodies at constant velocity
    s, p_bar = 0.13, np.array([0.01, -0.02, 0.03])
    P = _poses(R, c, s, p_bar)
    wrist = c + 0.05
    if arm.startswith("alongray"):      # AR1D's null space: constant-velocity RANGE profiles (AR1D-HO: equal velocity)
        d = np.array([0.1, -0.05, 1.0]) / np.linalg.norm([0.1, -0.05, 1])
        c = np.linspace(0.55, 0.65, len(c))[:, None] * d
        wrist = c + 0.05 * d
        P = _poses(R, c, s, p_bar)
    P[[4, 9]] = np.nan
    fr = np.arange(len(c)) * 3
    out = smooth_window(P, s, p_bar, fr, SmootherConfig(arm=arm), wrist=wrist, contact=np.ones(len(c), bool),
                        source_domain="third_person")
    assert np.isnan(out.poses[[4, 9]]).all()
    ok = np.isfinite(P).all(axis=(1, 2))
    assert np.abs(out.poses[ok] - P[ok]).max() < 1e-8
    assert (out.hand_shift is not None) == arm.endswith("_contact")
    if out.hand_shift is not None:
        assert np.abs(out.hand_shift).max() < 1e-8


def test_domain_gate_leaves_egocentric_and_contact_free_windows_at_base() -> None:
    """The registered stage acts only where the contact stage ran with a
    side, on a clip not declared egocentric; elsewhere the output is BASE, bit-for-bit."""
    R, c = twist_trajectory()
    s, p_bar = 0.13, np.array([0.01, -0.02, 0.03])
    P = _poses(R, along_jitter(c, 0.06, 0.002), s, p_bar)
    con = np.ones(len(c), bool)
    for arm in ("se3_contact", "alongray_contact"):
        cfg = SmootherConfig(arm=arm)
        for kw in ({"wrist": None, "contact": None, "source_domain": "third_person"},
                   {"wrist": c + 0.05, "contact": np.zeros(len(c), bool), "source_domain": "third_person"},
                   {"wrist": c + 0.05, "contact": con, "source_domain": "egocentric"}):
            out = smooth_window(P, s, p_bar, np.arange(len(c)), cfg, **kw)
            assert np.array_equal(out.poses, P, equal_nan=True) and out.hand_shift is None
            assert out.record["skipped"]
        acts = smooth_window(P, s, p_bar, np.arange(len(c)), cfg, wrist=c + 0.05, contact=con,
                             source_domain="third_person")
        assert "skipped" not in acts.record and np.abs(acts.poses - P).max() > 1e-4
    # mutation: the gate switched off (an object-only development arm) acts on the egocentric window
    off = smooth_window(P, s, p_bar, np.arange(len(c)), SmootherConfig(arm="se3", contact_only=False),
                        source_domain="egocentric")
    assert np.abs(off.poses - P).max() > 1e-4


def test_window_reduces_along_ray_jitter_of_the_posed_centroid() -> None:
    R, c = twist_trajectory()
    s, p_bar = 0.13, np.array([0.01, -0.02, 0.03])
    cm = along_jitter(c, 0.06, 0.002)
    P = _poses(R, cm, s, p_bar)
    obj = {"arm": "se3", "contact_only": False}
    out = smooth_window(P, s, p_bar, np.arange(len(c)), SmootherConfig(**obj))
    c_out = s * out.poses[:, :3, :3] @ p_bar + out.poses[:, :3, 3]
    assert rms(along(c_out, c)) < 0.6 * rms(along(cm, c))
    # mutation: a huge band makes the smoother pass the jitter through
    loose = smooth_window(P, s, p_bar, np.arange(len(c)), SmootherConfig(band=50.0, **obj))
    c_l = s * loose.poses[:, :3, :3] @ p_bar + loose.poses[:, :3, 3]
    assert rms(along(c_l, c)) > 0.9 * rms(along(cm, c))


def test_window_skips_short_windows() -> None:
    R, c = twist_trajectory(T=2)
    out = smooth_window(_poses(R, c, 0.1, np.zeros(3)), 0.1, np.zeros(3), np.arange(2), SmootherConfig(arm="se3", contact_only=False))
    assert out.record.get("skipped")


def test_config_null_is_off_and_keys_are_checked() -> None:
    assert smoother_from_mapping(None) is None
    assert smoother_from_mapping({}) == SmootherConfig()
    with pytest.raises(ValueError):
        smoother_from_mapping({"bogus": 1})
    with pytest.raises(ValueError):
        SmootherConfig(arm="kalman")
    with pytest.raises(ValueError):
        SmootherConfig(a_par=-1.0)
    assert SmootherConfig().arm == "se3_contact" and SmootherConfig().contact_only
    nz = SmootherConfig().noise(0.8)
    assert nz.a_perp == pytest.approx(0.0533 / 6.04) and nz.q_v == pytest.approx(0.25 * 0.0533 * 0.8)
    assert SmootherConfig(q_v_rel=0.0108).noise(0.8).q_v == pytest.approx(0.0108 * 0.8)
    with pytest.raises(ValueError):
        SmootherConfig(q_v_rel=-0.1)


def test_registered_config_writes_base_rotation_and_the_ablation_does_not() -> None:
    """Rev 1: the SE(3) candidate estimates rotation jointly but writes BASE's rotation (`smooth_rotation` False);
    the rotation-smoothing ablation changes it (mutation)."""
    from tests.smooth.synth import rot_noise

    R, c = twist_trajectory()
    s, p_bar = 0.13, np.array([0.01, -0.02, 0.03])
    P = _poses(rot_noise(R, 0.05), along_jitter(c, 0.06, 0.002), s, p_bar)
    kw = {"wrist": c + 0.05, "contact": np.ones(len(c), bool), "source_domain": "third_person"}
    held = smooth_window(P, s, p_bar, np.arange(len(c)), SmootherConfig(), **kw)
    assert np.array_equal(held.poses[:, :3, :3], P[:, :3, :3]) and np.abs(held.poses[:, :3, 3] - P[:, :3, 3]).max() > 1e-4
    rot = smooth_window(P, s, p_bar, np.arange(len(c)), SmootherConfig(smooth_rotation=True), **kw)
    assert np.abs(rot.poses[:, :3, :3] - P[:, :3, :3]).max() > 1e-3
    c_held = s * held.poses[:, :3, :3] @ p_bar + held.poses[:, :3, 3]
    c_rot = s * rot.poses[:, :3, :3] @ p_bar + rot.poses[:, :3, 3]
    assert np.abs(c_held - c_rot).max() < 1e-12            # the centroid trajectory is the same estimate
