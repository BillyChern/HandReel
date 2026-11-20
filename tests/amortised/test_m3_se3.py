"""M3: the SE(3) object error (known answers and the 'wrong frame in xi' mutation)."""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.amortised import se3 as S


def _rand_q(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_log_exp_round_trip_within_1e12() -> None:
    rng = np.random.default_rng(0)
    th = np.concatenate([[0.0, 1e-9, 1e-6, 1e-4, 1e-3, 0.5, 2.0, 3.1, np.pi - 1e-6], rng.uniform(0, np.pi, 200)])
    ax = rng.normal(size=(th.size, 3))
    om = th[:, None] * ax / np.linalg.norm(ax, axis=1, keepdims=True)
    rho = rng.normal(size=(th.size, 3)) * 0.3
    p, q = S.se3_exp(rho, om)
    rho2, om2 = S.se3_log(p, q)
    assert np.abs(om2 - om).max() <= 1e-12
    assert np.abs(rho2 - rho).max() <= 1e-12
    # and the other way round from random poses
    q = _rand_q(rng, 300)
    p = rng.normal(size=(300, 3))
    p2, q2 = S.se3_exp(*S.se3_log(p, q))
    assert np.abs(p2 - p).max() <= 1e-12
    assert np.minimum(np.abs(q2 - q).max(1), np.abs(q2 + q).max(1)).max() <= 1e-12


def test_quat_mat_round_trip() -> None:
    rng = np.random.default_rng(1)
    q = _rand_q(rng, 500)
    q2 = S.quat_from_mat(S.quat_to_mat(q))
    assert np.minimum(np.abs(q2 - q).max(1), np.abs(q2 + q).max(1)).max() < 1e-14
    v = rng.normal(size=(500, 3))
    np.testing.assert_allclose(S.quat_rotate(q, v), np.einsum("nij,nj->ni", S.quat_to_mat(q), v), atol=1e-14)


@pytest.mark.parametrize("theta", [1e-5, 0.1, 0.7, 1.5, 2.5, 3.1])
def test_in_plane_translation_ratio_is_half_theta_over_sin(theta: float) -> None:
    """T differs from T^ref by a rotation theta about an axis and a translation PERPENDICULAR to it (in the reference's
    own frame): |rho| / |dp| = (theta/2) / sin(theta/2). The reference pose is NOT the identity."""
    rng = np.random.default_rng(2)
    q_ref = _rand_q(rng, 1)
    p_ref = rng.normal(size=(1, 3))
    axis = np.array([[0.3, -0.5, 0.81]])
    axis /= np.linalg.norm(axis)
    perp = np.cross(axis, [[1.0, 0.0, 0.0]])
    perp /= np.linalg.norm(perp)
    dp_local = 0.04 * perp
    q = S.quat_mul(q_ref, S.so3_exp_quat(theta * axis))
    p = p_ref + S.quat_rotate(q_ref, dp_local)
    rho, om = S.object_error(p, q, p_ref, q_ref)
    np.testing.assert_allclose(om, theta * axis, atol=1e-12)
    ratio = np.linalg.norm(rho) / np.linalg.norm(dp_local)
    assert ratio == pytest.approx((theta / 2) / np.sin(theta / 2), rel=1e-10)


def test_error_is_invariant_to_a_change_of_world_frame() -> None:
    rng = np.random.default_rng(3)
    n = 50
    q, q_ref, g = _rand_q(rng, n), _rand_q(rng, n), _rand_q(rng, n)
    p, p_ref, t = rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    rho, om = S.object_error(p, q, p_ref, q_ref)
    gp = lambda pp: S.quat_rotate(g, pp) + t
    rho2, om2 = S.object_error(gp(p), S.quat_mul(g, q), gp(p_ref), S.quat_mul(g, q_ref))
    np.testing.assert_allclose(rho2, rho, atol=1e-12)
    np.testing.assert_allclose(om2, om, atol=1e-12)
    # pure translation in the reference frame: rho is that translation, expressed in the reference's frame
    rho3, om3 = S.object_error(p_ref + S.quat_rotate(q_ref, np.tile([[0.01, 0.02, -0.03]], (n, 1))), q_ref, p_ref,
                               q_ref)
    np.testing.assert_allclose(rho3, np.tile([[0.01, 0.02, -0.03]], (n, 1)), atol=1e-14)
    np.testing.assert_allclose(om3, 0.0, atol=1e-15)


def test_batched_rows_are_bitwise_equal_to_single_rows() -> None:
    rng = np.random.default_rng(4)
    n = 97
    q, q_ref = _rand_q(rng, n), _rand_q(rng, n)
    p, p_ref = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    rho, om = S.object_error(p, q, p_ref, q_ref)
    for i in (0, 13, 96):
        r1, o1 = S.object_error(p[i:i + 1], q[i:i + 1], p_ref[i:i + 1], q_ref[i:i + 1])
        assert np.array_equal(r1[0], rho[i]) and np.array_equal(o1[0], om[i])
