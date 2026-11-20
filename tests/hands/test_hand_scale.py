"""Hand-size pooling: known answers for `fastdo.hands.hand_scale`."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.hands.hand_scale import (
    PoolConfig,
    compose_depth_scale,
    config_window_size,
    depth_factors,
    gauge_follow_residual,
    huber_location,
    identity_flags,
    mano_rest_skeleton_length,
    pooled_log_size,
    projected_skeleton_length,
    skeleton_length,
    window_size,
)

MANO_RIGHT = Path("/opt/fastdo/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl")


def _hand(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    j = rng.normal(0.0, 0.03, (21, 3))
    j[:, 2] += 0.5
    return j


def test_skeleton_length_scales_and_is_rigid_invariant() -> None:
    j = _hand()
    base = float(skeleton_length(j))
    assert float(skeleton_length(1.3 * j)) == pytest.approx(1.3 * base, rel=1e-12)
    R = Rotation.from_rotvec([0.3, -1.1, 0.7]).as_matrix()
    assert float(skeleton_length(j @ R.T + np.array([0.1, -0.2, 0.4]))) == pytest.approx(base, rel=1e-12)


def test_skeleton_length_ignores_fingertips() -> None:
    j = _hand()
    k = j.copy()
    k[[4, 8, 12, 16, 20]] += 0.05
    assert float(skeleton_length(k)) == float(skeleton_length(j))


def test_projection_is_gauge_invariant() -> None:
    """X -> f X about the camera centre keeps every pixel: the identity the depth factor rests on."""
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    j = np.stack([_hand(s) for s in range(4)])
    assert np.allclose(projected_skeleton_length(1.17 * j, K), projected_skeleton_length(j, K), rtol=1e-12)
    behind = j.copy()
    behind[1, 3, 2] = -0.1
    assert np.isnan(projected_skeleton_length(behind, K)[1])


def test_window_size_uses_only_usable_frames() -> None:
    j = np.stack([_hand(0)] * 8)
    j[0] *= 2.0                                    # an invalid frame
    j[1] *= 3.0                                    # an infilled frame
    valid = np.ones(8, bool)
    valid[0] = False
    infilled = np.zeros(8, bool)
    infilled[1] = True
    ws = window_size(j, valid, infilled, min_frames=5)
    assert ws is not None and ws.n_frames == 6
    assert ws.log_size == pytest.approx(math.log(float(skeleton_length(_hand(0)))), abs=1e-12)
    assert window_size(j, valid, infilled, min_frames=7) is None


def test_huber_is_robust_and_exact_on_constants() -> None:
    assert huber_location([0.2, 0.2, 0.2]) == pytest.approx(0.2, abs=1e-12)
    clean = [0.0, 0.01, -0.01, 0.005, -0.005]
    assert abs(huber_location([*clean, 1.0])) < 0.01         # one gross outlier barely moves it
    assert abs(np.mean([*clean, 1.0])) > 0.15                  # where the mean is dragged


def test_depth_factors_known_answers() -> None:
    assert depth_factors({"a": 0.3}) == {"a": 1.0}                         # one window: nothing to pool
    assert depth_factors({"a": 0.3, "b": 0.3}) == {"a": 1.0, "b": 1.0}     # identical sizes: identity
    f = depth_factors({"a": 0.0, "b": 0.1})
    assert math.log(f["a"]) == pytest.approx(0.05, abs=1e-9)               # small window grows toward the pool
    assert math.log(f["b"]) == pytest.approx(-0.05, abs=1e-9)
    for k, s in {"a": 0.0, "b": 0.1}.items():                               # after the move every window reads the pool
        assert s + math.log(f[k]) == pytest.approx(0.05, abs=1e-9)


def test_depth_factor_cap_and_leave_one_out() -> None:
    sizes = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.6}
    f = depth_factors(sizes)
    assert math.log(f["d"]) == pytest.approx(-0.15, abs=1e-12)              # capped
    loo = pooled_log_size({"a": 0.0, "b": 0.2}, PoolConfig(leave_one_out=True))
    assert loo == {"a": pytest.approx(0.2), "b": pytest.approx(0.0)}


def test_default_frame_set_is_the_anchors() -> None:
    """M-FRAMESET (rev 1): the registered size reads the frames the anchor reads (valid, infilled INCLUDED).

    `hand_anchor_pixels` samples every valid joint; `hands.npz` `{side}_valid` is True on infilled frames. A size
    measured without them corrects a size the anchor never saw; this test fails if the default reverts."""
    j = np.stack([_hand(0)] * 9)
    j[:4] *= 1.2                                   # four infilled frames, 20% larger
    valid = np.ones(9, bool)
    infilled = np.zeros(9, bool)
    infilled[:4] = True
    anchor = config_window_size(j, valid, infilled)                     # default PoolConfig()
    observed = config_window_size(j, valid, infilled, PoolConfig(frame_set="observed"))
    lengths = skeleton_length(j)
    assert anchor is not None and anchor.n_frames == 9
    assert anchor.log_size == pytest.approx(math.log(float(np.median(lengths))), abs=1e-12)
    assert observed is not None and observed.n_frames == 5
    assert observed.log_size == pytest.approx(math.log(float(skeleton_length(_hand(0)))), abs=1e-12)
    with pytest.raises(ValueError, match="frame_set"):
        PoolConfig(frame_set="all")


def test_identity_flags_mark_a_foreign_hand_only() -> None:
    sizes = {"a": 0.00, "b": 0.01, "c": -0.01, "d": 0.005, "e": 0.20}   # e: someone else's hand
    flags = identity_flags(sizes)
    assert flags == {"a": False, "b": False, "c": False, "d": False, "e": True}
    assert identity_flags({"a": 0.0}) == {"a": False}                     # below min_windows: nothing to compare


def test_recentre_makes_the_move_variance_only() -> None:
    sizes = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.6}                    # the cap makes the raw log f mean non-zero
    raw = [math.log(v) for v in depth_factors(sizes).values()]
    cen = [math.log(v) for v in depth_factors(sizes, PoolConfig(recentre=True)).values()]
    assert abs(np.mean(raw)) > 0.01
    assert np.mean(cen) == pytest.approx(0.0, abs=1e-12)
    assert np.std(cen) == pytest.approx(np.std(raw), abs=1e-12)          # only the mean moves
    assert depth_factors({"a": 0.3}, PoolConfig(recentre=True)) == {"a": 1.0}


def test_compose_is_a_product_never_a_replacement() -> None:
    assert compose_depth_scale(1.04, 0.98) == pytest.approx(1.04 * 0.98, rel=1e-15)
    assert compose_depth_scale(1.04) == 1.04
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            compose_depth_scale(bad, 1.0)


def test_gauge_follow_residual_catches_a_double_application() -> None:
    """KA-GAUGE with two different per-window factors: exact following passes at 1%, f^2 (M-DOUBLE) and a swapped
    factor file (each window given the other's factor) fail."""
    f = {"w1": 0.95, "w2": 1.06}
    base = {"w1": 1.30, "w2": 0.80}
    good = {k: base[k] * f[k] for k in f}
    double = {k: base[k] * f[k] ** 2 for k in f}
    swapped = {"w1": base["w1"] * f["w2"], "w2": base["w2"] * f["w1"]}
    assert max(gauge_follow_residual(good, base, f).values()) < 1e-12
    assert min(gauge_follow_residual(double, base, f).values()) > 0.01
    assert min(gauge_follow_residual(swapped, base, f).values()) > 0.01
    assert gauge_follow_residual({"w1": 1.0}, base, f).keys() == {"w1"}


@pytest.mark.skipif(not MANO_RIGHT.is_file(), reason="MANO model not on this machine")
def test_mano_rest_length_matches_the_models_own_joints() -> None:
    from fastdo.retarget.labels import _MANOUnpickler

    with open(MANO_RIGHT, "rb") as fh:
        m = _MANOUnpickler(fh, encoding="latin1").load()
    sd = m["shapedirs"]
    shapedirs = np.asarray(sd.a.x).ravel()[np.asarray(sd.idxs)].reshape(sd.preferred_shape)
    jr = m["J_regressor"].toarray()
    vt = np.asarray(m["v_template"], np.float64)
    zero = mano_rest_skeleton_length(np.zeros(10), vt, shapedirs, jr)
    j = np.asarray(m["J"], np.float64)                                      # the model's stored rest joints
    kin = np.asarray(m["kintree_table"])[0]
    ref = sum(np.linalg.norm(j[c] - j[kin[c]]) for c in range(1, 16))
    assert zero == pytest.approx(ref, rel=1e-6)
    lengths = mano_rest_skeleton_length(np.array([[2.0] + [0.0] * 9, [-2.0] + [0.0] * 9]), vt, shapedirs, jr)
    assert lengths.shape == (2,) and lengths[0] != pytest.approx(lengths[1], rel=1e-3)
