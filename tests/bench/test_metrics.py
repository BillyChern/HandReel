import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from fastdo.bench.metrics import chamfer_distance, f_score, retarget_errors


def _cloud(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 3)).astype(np.float32)


def test_chamfer_identical_is_zero() -> None:
    a = _cloud()
    assert chamfer_distance(a, a) == pytest.approx(0.0, abs=1e-7)


def test_chamfer_shift_matches_analytic() -> None:
    a = _cloud()
    b = a + np.array([0.01, 0, 0], np.float32)   # nearest neighbour is the shifted twin
    assert chamfer_distance(a, b, squared=True) == pytest.approx(2 * 0.01**2, rel=1e-3)
    assert chamfer_distance(a, b, squared=False) == pytest.approx(2 * 0.01, rel=1e-3)


def test_f_score_bounds() -> None:
    a = _cloud()
    assert f_score(a, a, threshold=0.005) == pytest.approx(1.0)
    far = a + 10.0
    assert f_score(a, far, threshold=0.005) == pytest.approx(0.0)


def test_retarget_errors_rotation_only() -> None:
    t = 5
    ref = np.tile(np.eye(4), (t, 1, 1))
    pred = ref.copy()
    pred[:, :3, :3] = R.from_rotvec([0.3, 0, 0]).as_matrix()
    err = retarget_errors(pred, ref)
    assert err.e_pos == pytest.approx(0.0)
    assert err.e_rot == pytest.approx(0.3, abs=1e-6)
    assert err.success


def test_retarget_errors_failure_threshold() -> None:
    ref = np.tile(np.eye(4), (3, 1, 1))
    pred = ref.copy()
    pred[:, :3, 3] = [0.2, 0, 0]
    err = retarget_errors(pred, ref)
    assert err.e_pos == pytest.approx(0.2)
    assert not err.success


def test_geodesic_has_no_floor_on_identical_float32_rotations():
    """arccos of the trace read 0.028 deg for a float32 rotation against itself."""
    from fastdo.bench.metrics import _geodesic
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((5, 3, 3)))
    r = (q * np.sign(np.linalg.det(q))[:, None, None]).astype(np.float32)
    assert np.all(_geodesic(r, r) == 0.0)
    theta = np.array([1e-4, 0.3, 2.0, 3.1])
    rz = np.zeros((4, 3, 3))
    rz[:, 0, 0] = rz[:, 1, 1] = np.cos(theta)
    rz[:, 0, 1], rz[:, 1, 0], rz[:, 2, 2] = -np.sin(theta), np.sin(theta), 1.0
    np.testing.assert_allclose(_geodesic(rz, np.broadcast_to(np.eye(3), rz.shape)), theta, rtol=1e-9)
