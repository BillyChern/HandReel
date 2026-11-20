"""Synthetic windows for the smoother tests (camera frame, metres, rows)."""

from __future__ import annotations

import numpy as np

from fastdo.smooth.lie import se3_exp, so3_exp


def twist_trajectory(T: int = 40, xi: tuple[float, ...] = (0.0, 0.04, 0.01, 0.002, 0.001, 0.0),
                     wobble: tuple[float, float] = (0.003, 0.0005), seed: int = 0,
                     start: tuple[float, float, float] = (0.05, -0.02, 0.6)) -> tuple[np.ndarray, np.ndarray]:
    """A rotating, translating centroid-frame trajectory (R, c); `wobble` = per-row twist noise (rad, m)."""
    rng = np.random.default_rng(seed)
    T0 = np.eye(4)
    T0[:3, 3] = start
    Ts = [T0]
    for _ in range(T - 1):
        e = np.r_[rng.normal(0, wobble[0], 3), rng.normal(0, wobble[1], 3)]
        Ts.append(Ts[-1] @ se3_exp(np.asarray(xi) + e))
    Ts = np.asarray(Ts)
    return Ts[:, :3, :3].copy(), Ts[:, :3, 3].copy()


def along_jitter(c: np.ndarray, sd_rel: float, perp_sd_m: float = 0.0, seed: int = 1) -> np.ndarray:
    """Measured centres: relative along-ray jitter sd_rel * |c| plus isotropic perp noise (m)."""
    rng = np.random.default_rng(seed)
    z = np.linalg.norm(c, axis=1, keepdims=True)
    u = c / z
    n = rng.normal(0, perp_sd_m, c.shape)
    n -= np.sum(n * u, 1, keepdims=True) * u
    return c + rng.normal(0, sd_rel, (len(c), 1)) * z * u + n


def rot_noise(R: np.ndarray, sd: float, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([r @ so3_exp(rng.normal(0, sd, 3)) for r in R])


def along(c: np.ndarray, ct: np.ndarray) -> np.ndarray:
    u = ct / np.linalg.norm(ct, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", c - ct, u)


def perp(c: np.ndarray, ct: np.ndarray) -> np.ndarray:
    u = ct / np.linalg.norm(ct, axis=1, keepdims=True)
    d = c - ct
    return np.linalg.norm(d - np.einsum("ij,ij->i", d, u)[:, None] * u, axis=1)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
