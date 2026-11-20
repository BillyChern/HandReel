"""Smoother known answers, each with a mutation that fires.

KA-JITTER  injected along-ray jitter is recovered; perpendicular is not inflated.
KA-STATIC  a static object stays static (exactly).
KA-TWIST   a constant-twist trajectory is a fixed point (the process model's null space).
KA-GAUGE   a pure gauge offset is NOT changed: S(g_l X) = g_l S(X) (KA-G), and the hold makes the weighted range
           mean exact; the measured-noise (self-weighted) reference biases the window inward (mutation).
KA-MAP     the output is the minimiser of the stated MAP objective (scipy, independent residuals); a wrong
           adjoint transport is not (mutation).
KA-ANISO   anisotropy keeps real perpendicular motion; an isotropic smoother (kappa 1) blurs it (mutation).
KA-HUBER   a single 150 mm along-ray spike is down-weighted; without Huber it drags its neighbours (mutation).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from fastdo.smooth import rts as rts_mod
from fastdo.smooth.lie import se3_exp, se3_log, so3_log
from fastdo.smooth.rts import Noise, smooth_se3
from tests.smooth.synth import along, along_jitter, perp, rms, rot_noise, twist_trajectory

NZ = Noise(s_rot=0.03, a_par=0.04 / 0.6, a_perp=0.004 / 0.6, q_w=0.25 * 0.03, q_v=0.25 * 0.04)


def _noise(**kw: object) -> Noise:
    base = {f: getattr(NZ, f) for f in NZ.__dataclass_fields__}
    base.update(kw)
    return Noise(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("process", ["se3", "so3xr3"])
def test_ka_jitter_recovered_and_perp_not_inflated(process: str) -> None:
    R, c = twist_trajectory()
    cm = along_jitter(c, 0.04 / 0.6, 0.002)
    res = smooth_se3(rot_noise(R, 0.03), cm, np.ones(len(c) - 1), _noise(process=process))
    assert rms(along(res.c, c)) < 0.55 * rms(along(cm, c))
    assert rms(perp(res.c, c)) < 1.2 * rms(perp(cm, c))


def test_ka_static_stays_static() -> None:
    R, c = twist_trajectory(xi=(0, 0, 0, 0, 0, 0), wobble=(0, 0))
    res = smooth_se3(R, c, np.ones(len(c) - 1), NZ)
    assert np.abs(res.c - c).max() < 1e-12 and np.abs(res.R - R).max() < 1e-12


def test_ka_twist_fixed_point_and_irregular_rows() -> None:
    R, c = twist_trajectory(wobble=(0, 0))
    res = smooth_se3(R, c, np.ones(len(c) - 1), NZ)
    assert np.abs(res.c - c).max() < 1e-9 and np.abs(res.R - R).max() < 1e-9
    # rows 10..12 dropped: the window keeps the time axis through D_k = 2
    keep = np.r_[0:10, 11:len(c)]
    res2 = smooth_se3(R[keep], c[keep], np.diff(keep).astype(float), NZ)
    assert np.abs(res2.c - c[keep]).max() < 1e-9


@pytest.mark.parametrize("lam", [0.9, 1.1])
def test_ka_gauge_equivariance(lam: float) -> None:
    R, c = twist_trajectory()
    Rm, cm = rot_noise(R, 0.03), along_jitter(c, 0.06, 0.003)
    a = smooth_se3(Rm, cm, np.ones(len(c) - 1), NZ)
    b = smooth_se3(Rm, lam * cm, np.ones(len(c) - 1), _noise(q_v=NZ.q_v * lam))
    assert np.abs(b.c - lam * a.c).max() < 1e-8 and np.abs(b.R - a.R).max() < 1e-8


def test_ka_pure_gauge_offset_is_not_changed() -> None:
    R, c = twist_trajectory(wobble=(0, 0))
    for lam in (0.8, 1.25):                      # a window that is EXACTLY a gauge-scaled truth: untouched
        res = smooth_se3(R, lam * c, np.ones(len(c) - 1), _noise(q_v=NZ.q_v * lam))
        assert np.abs(res.c - lam * c).max() < 1e-9


def test_ka_hold_is_exact_and_its_mutation_fires() -> None:
    R, c = twist_trajectory()
    cm = along_jitter(c, 0.06, 0.003)
    w = lambda r: np.nan_to_num(r.w_c)
    res = smooth_se3(R, cm, np.ones(len(c) - 1), NZ)
    held = np.log(np.sum(w(res) * np.linalg.norm(res.c, axis=1)) / np.sum(w(res) * np.linalg.norm(cm, axis=1)))
    assert abs(held) < 1e-12
    bad = smooth_se3(R, cm, np.ones(len(c) - 1), NZ, gauge_hold=False)
    moved = np.log(np.sum(w(bad) * np.linalg.norm(bad.c, axis=1)) / np.sum(w(bad) * np.linalg.norm(cm, axis=1)))
    assert abs(moved) > 1e-5 and moved == pytest.approx(bad.gamma, abs=1e-12)


def test_ka_noise_reference_unbiased_and_self_weighting_mutation_fires() -> None:
    """gamma = the smoother's own gauge component (output vs its input, so the noise realisation cancels)."""
    R, c = twist_trajectory()
    gam: dict[str, list[float]] = {"estimate": [], "measured": []}
    for seed in range(8):
        cm = along_jitter(c, 0.04 / 0.6, 0.004, seed=10 + seed)
        Rm = rot_noise(R, 0.03, seed=40 + seed)
        for ref, vals in gam.items():
            vals.append(smooth_se3(Rm, cm, np.ones(len(c) - 1), _noise(noise_ref=ref)).gamma)
    assert abs(np.mean(gam["estimate"])) < 2e-3             # < 0.2% before the hold
    assert np.mean(gam["measured"]) < -5e-3                 # self-weighting pulls the window toward the camera


def _map_residuals(x: np.ndarray, Rm: np.ndarray, cm: np.ndarray, nz: Noise) -> np.ndarray:
    """The stated MAP objective (rts module docstring), written independently, fixed (measured) noise reference."""
    T = len(cm)
    X = x.reshape(T, 12)
    Ts = [se3_exp(X[k, :6]) for k in range(T)]
    out = []
    for k in range(T):
        R, c = Ts[k][:3, :3], Ts[k][:3, 3]
        out.append(so3_log(R.T @ Rm[k]) / nz.s_rot)
        z = np.linalg.norm(cm[k])
        u = cm[k] / z
        S = (nz.a_par * z) ** 2 * np.outer(u, u) + (nz.a_perp * z) ** 2 * (np.eye(3) - np.outer(u, u))
        out.append(np.linalg.solve(np.linalg.cholesky(S), cm[k] - c))
    q = np.array([nz.q_w] * 3 + [nz.q_v] * 3)
    Qc = np.diag(q ** 2)
    Q = np.block([[Qc / 3.0, Qc / 2.0], [Qc / 2.0, Qc]])
    L = np.linalg.cholesky(Q)
    for k in range(T - 1):
        f = np.r_[se3_log(np.linalg.inv(Ts[k]) @ Ts[k + 1]) - X[k, 6:], X[k + 1, 6:] - X[k, 6:]]
        out.append(np.linalg.solve(L, f))
    return np.concatenate(out)


def _as_params(res_R: np.ndarray, res_c: np.ndarray, xi: np.ndarray) -> np.ndarray:
    T = len(res_c)
    X = np.zeros((T, 12))
    for k in range(T):
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = res_R[k], res_c[k]
        X[k, :6] = se3_log(M)
        X[k, 6:] = xi[k]
    return X.ravel()


def _solve_and_check() -> float:
    R, c = twist_trajectory(T=14)
    Rm, cm = rot_noise(R, 0.03), along_jitter(c, 0.06, 0.003)
    nz = _noise(noise_ref="measured", reweights=0, gn_iters=30)
    res = rts_mod.smooth_se3(Rm, cm, np.ones(len(c) - 1), nz, gauge_hold=False)
    # scipy starts from the smoother's poses with finite-difference twists (the smoother does not return twists)
    # and minimises the independently written objective; at a true minimiser the poses do not move.
    xi = np.zeros((len(c), 6))
    for k in range(len(c) - 1):
        A, B = np.eye(4), np.eye(4)
        A[:3, :3], A[:3, 3] = res.R[k], res.c[k]
        B[:3, :3], B[:3, 3] = res.R[k + 1], res.c[k + 1]
        xi[k] = se3_log(np.linalg.inv(A) @ B)
    xi[-1] = xi[-2]
    x0 = _as_params(res.R, res.c, xi)
    f = lambda x: _map_residuals(x, Rm, cm, nz)
    sol = least_squares(f, x0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20000)
    Ts = [se3_exp(sol.x.reshape(len(c), 12)[k, :6]) for k in range(len(c))]
    c_opt = np.array([t[:3, 3] for t in Ts])
    return float(np.abs(c_opt - res.c).max())


def test_ka_map_output_is_the_minimiser() -> None:
    assert _solve_and_check() < 1e-6


def test_ka_map_mutation_wrong_adjoint_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rts_mod, "se3_adjoint", lambda T: np.eye(6))
    assert _solve_and_check() > 1e-4


def test_ka_aniso_keeps_real_perpendicular_motion() -> None:
    T = 40
    k = np.arange(T)
    c = np.stack([0.02 * np.sin(2 * np.pi * k / 6.0), np.full(T, -0.02), np.full(T, 0.6)], 1)   # real sideways shake
    R = np.repeat(np.eye(3)[None], T, 0)
    cm = along_jitter(c, 0.04 / 0.6, 0.0005)
    good = smooth_se3(R, cm, np.ones(T - 1), NZ)
    iso = smooth_se3(R, cm, np.ones(T - 1), _noise(a_perp=NZ.a_par))
    assert rms(perp(good.c, c)) < 3e-3
    assert rms(perp(iso.c, c)) > 2.0 * rms(perp(good.c, c))
    assert rms(along(good.c, c)) < 0.7 * rms(along(cm, c))


def test_ka_huber_downweights_a_spike_and_its_mutation_fires() -> None:
    R, c = twist_trajectory(wobble=(0, 0))
    cm = along_jitter(c, 0.01, 0.0)
    u = c[20] / np.linalg.norm(c[20])
    cm[20] = cm[20] + 0.25 * u                              # ~6 sd along the ray
    rob = smooth_se3(R, cm, np.ones(len(c) - 1), NZ)
    raw = smooth_se3(R, cm, np.ones(len(c) - 1), _noise(reweights=0))
    nb = np.r_[17:20, 21:24]
    assert rob.w_c[20] < 0.5
    assert rms(along(rob.c[nb], c[nb])) < 0.5 * rms(along(raw.c[nb], c[nb]))


def test_missing_rows_carry_no_measurement() -> None:
    R, c = twist_trajectory()
    cm = along_jitter(c, 0.05, 0.002)
    cm[[5, 6, 21]] = np.nan
    res = smooth_se3(R, cm, np.ones(len(c) - 1), NZ)
    assert np.isfinite(res.c).all() and np.isnan(res.w_c[[5, 6, 21]]).all()
    with pytest.raises(ValueError):
        smooth_se3(R[:2], c[:2], np.ones(1), NZ)


def test_rotation_noise_is_reduced() -> None:
    R, c = twist_trajectory()
    Rm = rot_noise(R, 0.03)
    res = smooth_se3(Rm, c, np.ones(len(c) - 1), NZ)
    ang = lambda A: np.mean([np.linalg.norm(so3_log(a.T @ b)) for a, b in zip(R, A, strict=True)])
    assert ang(res.R) < 0.6 * ang(Rm)
    assert np.allclose(np.einsum("kij,kil->kjl", res.R, res.R), np.eye(3), atol=1e-12)
