"""Contact-aware joint smoother (SE3-RTS+C): the MAP known answer with the grasp-rigidity factor, the
common-jitter known answer, and mutations that fire."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from fastdo.smooth import lie
from fastdo.smooth import rts as rts_mod
from fastdo.smooth.lie import se3_exp, se3_log, so3_exp, so3_log
from fastdo.smooth.rts import Noise, smooth_se3
from tests.smooth.synth import along_jitter, rms, rot_noise, twist_trajectory

NZ = Noise(s_rot=0.05, a_par=0.06, a_perp=0.01, q_w=0.0125, q_v=0.009, a_par_h=0.025, a_perp_h=0.007,
           q_h=0.004, q_c=0.004, reweights=0, gn_iters=40, noise_ref="measured")


def _aniso(x: np.ndarray, a_par: float, a_perp: float) -> np.ndarray:
    z = np.linalg.norm(x)
    u = x / z
    return (a_par * z) ** 2 * np.outer(u, u) + (a_perp * z) ** 2 * (np.eye(3) - np.outer(u, u))


def _residuals(x: np.ndarray, Rm, cm, hm, con, nz: Noise, process: str) -> np.ndarray:
    """The stated joint objective (rts docstring), written independently: 18 parameters per row."""
    T = len(cm)
    X = x.reshape(T, 18)
    out = []
    for k in range(T):
        if process == "se3":
            Tk = se3_exp(X[k, :6])
            R, c = Tk[:3, :3], Tk[:3, 3]
        else:
            R, c = so3_exp(X[k, :3]), X[k, 3:6]
        out.append(so3_log(R.T @ Rm[k]) / nz.s_rot)
        out.append(np.linalg.solve(np.linalg.cholesky(_aniso(cm[k], nz.a_par, nz.a_perp)), cm[k] - c))
        out.append(np.linalg.solve(np.linalg.cholesky(_aniso(hm[k], nz.a_par_h, nz.a_perp_h)), hm[k] - X[k, 12:15]))
        if con[k]:
            om, v = X[k, 6:9], X[k, 9:12]
            vc = R @ v if process == "se3" else v
            out.append((vc - X[k, 15:18] - np.cross(R @ om, c - X[k, 12:15])) / nz.q_c)
    q = np.array([nz.q_w] * 3 + [nz.q_v] * 3)
    L = np.linalg.cholesky(np.block([[np.diag(q ** 2) / 3, np.diag(q ** 2) / 2], [np.diag(q ** 2) / 2, np.diag(q ** 2)]]))
    Lh = np.linalg.cholesky(nz.q_h ** 2 * np.block([[np.eye(3) / 3, np.eye(3) / 2], [np.eye(3) / 2, np.eye(3)]]))
    for k in range(T - 1):
        if process == "se3":
            A, B = se3_exp(X[k, :6]), se3_exp(X[k + 1, :6])
            d = se3_log(np.linalg.inv(A) @ B)
        else:
            d = np.r_[so3_log(so3_exp(X[k, :3]).T @ so3_exp(X[k + 1, :3])), X[k + 1, 3:6] - X[k, 3:6]]
        out.append(np.linalg.solve(L, np.r_[d - X[k, 6:12], X[k + 1, 6:12] - X[k, 6:12]]))
        out.append(np.linalg.solve(Lh, np.r_[X[k + 1, 12:15] - X[k, 12:15] - X[k, 15:18], X[k + 1, 15:18] - X[k, 15:18]]))
    return np.concatenate(out)


def _window(T: int = 12):
    R, c = twist_trajectory(T=T, xi=(0.0, 0.03, 0.0, 0.003, 0.0, 0.002))
    d_body = np.array([0.0, 0.06, -0.03])
    h = c + np.einsum("kij,j->ki", R, d_body)
    Rm = rot_noise(R, 0.04)
    cm = along_jitter(c, 0.05, 0.003, seed=3)
    hm = along_jitter(h, 0.02, 0.003, seed=4)
    con = np.zeros(T, bool)
    con[3:10] = True
    return Rm, cm, hm, con


def _gap(process: str) -> float:
    Rm, cm, hm, con = _window()
    T = len(cm)
    res = rts_mod.smooth_se3(Rm, cm, np.ones(T - 1), NZ.__class__(**{**NZ.__dict__, "process": process}), hand_m=hm,
                             contact=con, gauge_hold=False)
    X = np.zeros((T, 18))
    for k in range(T):
        if process == "se3":
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = res.R[k], res.c[k]
            X[k, :6] = se3_log(M)
        else:
            X[k, :3], X[k, 3:6] = so3_log(res.R[k]), res.c[k]
        X[k, 12:15] = res.hand[k]
    for k in range(T - 1):
        if process == "se3":
            A, B = se3_exp(X[k, :6]), se3_exp(X[k + 1, :6])
            X[k, 6:12] = se3_log(np.linalg.inv(A) @ B)
        else:
            X[k, 6:9] = so3_log(res.R[k].T @ res.R[k + 1])
            X[k, 9:12] = res.c[k + 1] - res.c[k]
        X[k, 15:18] = res.hand[k + 1] - res.hand[k]
    X[-1, 6:] = X[-2, 6:]
    nz = NZ.__class__(**{**NZ.__dict__, "process": process})
    sol = least_squares(lambda x: _residuals(x, Rm, cm, hm, con, nz, process), X.ravel(), method="lm",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=50000)
    Y = sol.x.reshape(T, 18)
    c_opt = np.array([se3_exp(Y[k, :6])[:3, 3] for k in range(T)]) if process == "se3" else Y[:, 3:6]
    return float(max(np.abs(c_opt - res.c).max(), np.abs(Y[:, 12:15] - res.hand).max()))


@pytest.mark.parametrize("process", ["se3", "so3xr3"])
def test_ka_map_with_grasp_rigidity(process: str) -> None:
    assert _gap(process) < 1e-6


def test_ka_map_mutation_wrong_coupling_jacobian_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rts_mod, "so3_hat", lambda w: -lie.so3_hat(w))     # only the coupling block uses it
    assert _gap("se3") > 1e-4


def test_ka_common_jitter_keeps_the_relative_gap() -> None:
    """A rigid grasp whose object AND wrist carry the SAME along-ray jitter (what the per-frame contact stage
    produces; seen DexYCB corr of the first differences 0.63): smoothing both bodies removes it from both and keeps the
    hand-object gap; smoothing the object alone (the mutation) opens the gap by the jitter itself."""
    T = 40
    R, c = twist_trajectory(T=T, xi=(0.0, 0.0, 0.0, 0.003, 0.0, 0.002), wobble=(0.0, 0.0002))
    h = c + np.array([0.0, 0.06, -0.03])
    j = np.random.default_rng(7).normal(0, 0.05, T)               # common relative jitter
    cm, hm = c * (1 + j)[:, None], h * (1 + j)[:, None]
    nz = Noise(s_rot=0.05, a_par=0.06, a_perp=0.01, q_w=0.0125, q_v=0.009, a_par_h=0.025, a_perp_h=0.007,
               q_h=0.00375, q_c=0.0158)
    gap = lambda co, ho: rms(np.linalg.norm((co - ho) - (c - h), axis=1))
    joint = smooth_se3(R, cm, np.ones(T - 1), nz, hand_m=hm, contact=np.ones(T, bool))
    assert gap(joint.c, joint.hand) < 0.2 * rms(np.linalg.norm(cm - c, axis=1))  # well below the injected jitter
    assert rms(np.linalg.norm(joint.c - c, axis=1)) < 0.6 * rms(np.linalg.norm(cm - c, axis=1))
    assert rms(np.linalg.norm(joint.hand - h, axis=1)) < 0.6 * rms(np.linalg.norm(hm - h, axis=1))
    alone = smooth_se3(R, cm, np.ones(T - 1), nz)                               # mutation: the hand stays BASE's
    assert gap(alone.c, hm) > 5.0 * gap(joint.c, joint.hand)
