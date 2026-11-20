"""Per-video hand-size pooling: the estimator and its depth factor.

Production's metric anchor is HaWoR's hand: `hand_anchored_scale` multiplies the raw pointmap by
`k = mean_j z_hawor / mean_j z_pointmap` at the hand's own pixels. HaWoR places a MANO hand of its own estimated shape
`beta` at the depth `tz = 2 f / (b s)` of its weak-perspective camera. At fixed pixels a hand of size L sits at depth
proportional to L, so HaWoR's per-window size error enters the anchor, and with it the window gauge lambda, one to one
IF the network keeps its reprojection consistent with the image (an empirical question, not a property of the
code, whose `tz` does not read `beta`).

In egocentric video every window's hands belong to the camera wearer, so the true size of each hand is one constant per
(video, side). This module measures each window's size, pools it robustly per (video, side), and returns the factor
`f_w = s_pooled / s_w` that moves window w's hand (and the anchor read from it) about the camera centre to the pooled
size: `X -> f_w X` keeps every pixel and multiplies every depth by `f_w`. It is the per-side `depth_scale` hook with a
per-window factor, plus the same move applied to the hand itself, so that hand and object stay in one gauge.

Hand size: the summed length of the 15 bones between the 16 MANO skeleton joints (wrist, and three joints per finger),
read from the 21 OpenPose-order joints HaWoR writes (`fastdo.hands.hawor_npz`; the 5 fingertips are vertices and are
not used). MANO poses its skeleton joints by rigid transforms along the kinematic chain, so these bone lengths are
exactly invariant to pose and to the global rigid motion; they depend on `beta` only. Measuring them from the stored
joints, not from `{side}_betas`, uses the per-frame shapes the joints were built from (`hands.npz` stores only their
window mean).

Rev 1 (2026-10-12):
- the size is measured on the ANCHOR's frame set by default (`PoolConfig.frame_set = "anchor"`: every valid frame,
  infilled ones included, which is what `hand_anchor_pixels` reads); rev 0 excluded infilled frames ("observed"), so
  `f_w` corrected a size measured on other frames than the anchor's;
- `identity_flags` flags a window whose size lies more than `outlier_k` robust sds from its group's pool (a hand that
  may not be the wearer's: another person's hand, gloves); descriptive only, it never changes a factor;
- `PoolConfig.recentre` (off by default) re-centres log f to mean zero per group, making the move variance-only;
- `compose_depth_scale` composes the pooled factor with a per-side hand-depth factor as a PRODUCT, never a replacement;
- `gauge_follow_residual` is the KA-GAUGE check: per window, |log(scale_arm / scale_base) - log f_w|. Applying the
  factor twice (hand artefact rescaled AND the hook, giving f_w^2) leaves a residual of |log f_w|, so it is caught
  whenever |log f_w| exceeds the tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FRAME_SETS",
    "SKELETON_BONES",
    "PoolConfig",
    "WindowSize",
    "compose_depth_scale",
    "config_window_size",
    "depth_factors",
    "gauge_follow_residual",
    "huber_location",
    "identity_flags",
    "mano_rest_skeleton_length",
    "pooled_log_size",
    "projected_skeleton_length",
    "skeleton_length",
    "window_size",
]

# Parent-child pairs among the 16 MANO skeleton joints in the 21-joint OpenPose order (0 wrist; per finger, from the
# palm: 1-3 thumb, 5-7 index, 9-11 middle, 13-15 ring, 17-19 pinky; 4, 8, 12, 16, 20 are fingertip vertices).
SKELETON_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3),
    (0, 5), (5, 6), (6, 7),
    (0, 9), (9, 10), (10, 11),
    (0, 13), (13, 14), (14, 15),
    (0, 17), (17, 18), (18, 19),
)

# "anchor": valid frames, infilled included (the frames production's anchor reads); "observed": valid, not infilled.
FRAME_SETS: tuple[str, ...] = ("anchor", "observed")


def skeleton_length(joints: np.ndarray) -> np.ndarray:
    """Summed 3D length of `SKELETON_BONES` per frame.

    Args:
        joints: `(T, 21, 3)` (or `(21, 3)`) joints in metres, OpenPose order.

    Returns:
        `(T,)` float64 (a 0-d array for one frame); NaN where a joint is not finite.
    """
    j = np.asarray(joints, np.float64)
    a = np.array([p for p, _ in SKELETON_BONES])
    b = np.array([c for _, c in SKELETON_BONES])
    return np.linalg.norm(j[..., a, :] - j[..., b, :], axis=-1).sum(axis=-1)


def mano_rest_skeleton_length(betas: np.ndarray, v_template: np.ndarray, shapedirs: np.ndarray,
                              j_regressor: np.ndarray) -> np.ndarray:
    """Skeleton length of a MANO model at shape `betas` ((10,) or (N,10)): the 16 regressed rest joints
    `J_regressor (v_template + shapedirs beta)`, mapped to OpenPose order, summed over `SKELETON_BONES`.

    MANO joint order (16): 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb."""
    b = np.atleast_2d(np.asarray(betas, np.float64))
    v = v_template[None] + np.einsum("vck,nk->nvc", shapedirs[..., : b.shape[1]], b)
    j16 = np.einsum("jv,nvc->njc", np.asarray(j_regressor, np.float64), v)
    openpose = np.full((b.shape[0], 21, 3), np.nan)
    for op, mano in {0: 0, 1: 13, 2: 14, 3: 15, 5: 1, 6: 2, 7: 3, 9: 4, 10: 5, 11: 6, 13: 10, 14: 11, 15: 12,
                     17: 7, 18: 8, 19: 9}.items():
        openpose[:, op] = j16[:, mano]
    out = skeleton_length(openpose)
    return out if np.ndim(betas) > 1 else out[0]


def projected_skeleton_length(joints: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Summed image length (pixels) of `SKELETON_BONES` per frame, pinhole `K`; NaN where any joint has z <= 0."""
    j = np.asarray(joints, np.float64)
    z = j[..., 2:3]
    uv = np.concatenate([K[0, 0] * j[..., 0:1] / z + K[0, 2], K[1, 1] * j[..., 1:2] / z + K[1, 2]], axis=-1)
    a = np.array([p for p, _ in SKELETON_BONES])
    b = np.array([c for _, c in SKELETON_BONES])
    out = np.linalg.norm(uv[..., a, :] - uv[..., b, :], axis=-1).sum(axis=-1)
    bad = ~(j[..., 2] > 0).all(axis=-1)
    return np.where(bad, np.nan, out)


@dataclass(frozen=True)
class PoolConfig:
    """The pooling settings.

    min_frames: usable frames (valid, not infilled, finite) for a window's size to count.
    min_windows: usable windows of a (video, side) for pooling to act; fewer gives f = 1.
    huber_c: Huber tuning constant in robust-sd units.
    sd_floor: floor of the robust scale (log units) so that identical sizes do not divide by zero.
    max_abs_log_f: cap on |log f| (gate the action, not just the evidence).
    leave_one_out: pool over the OTHER windows of the video (evaluation sensitivity; production uses all).
    frame_set: which frames measure a window's size, one of `FRAME_SETS` (rev 1 default "anchor").
    recentre: re-centre log f to mean zero within each group (variance-only move; off by default).
    outlier_k: `identity_flags` threshold in robust-sd units (descriptive only).
    """

    min_frames: int = 5
    min_windows: int = 2
    huber_c: float = 1.345
    sd_floor: float = 0.005
    max_abs_log_f: float = 0.15
    leave_one_out: bool = False
    frame_set: str = "anchor"
    recentre: bool = False
    outlier_k: float = 3.0

    def __post_init__(self) -> None:
        if self.frame_set not in FRAME_SETS:
            raise ValueError(f"frame_set must be one of {FRAME_SETS}, got {self.frame_set!r}")


@dataclass(frozen=True)
class WindowSize:
    """One window-side's size: `log_size` = log of the median per-frame skeleton length (metres), from `n_frames`
    usable frames; `within_sd` = sd of the per-frame log length (per-frame beta jitter)."""

    log_size: float
    n_frames: int
    within_sd: float


def window_size(joints: np.ndarray, valid: np.ndarray, infilled: np.ndarray | None = None,
                min_frames: int = 5) -> WindowSize | None:
    """The size of one window-side from its stored joints, or None with fewer than `min_frames` usable frames."""
    length = skeleton_length(joints)
    use = np.asarray(valid, bool) & np.isfinite(length) & (length > 0)
    if infilled is not None:
        use &= ~np.asarray(infilled, bool)
    n = int(use.sum())
    if n < min_frames:
        return None
    logs = np.log(length[use])
    return WindowSize(float(np.log(np.median(length[use]))), n, float(np.std(logs)))


def config_window_size(joints: np.ndarray, valid: np.ndarray, infilled: np.ndarray | None,
                       cfg: PoolConfig | None = None) -> WindowSize | None:
    """`window_size` on the frame set `cfg.frame_set` selects ("anchor": `infilled` is ignored)."""
    cfg = cfg or PoolConfig()
    return window_size(joints, valid, infilled if cfg.frame_set == "observed" else None, cfg.min_frames)


def huber_location(x: Sequence[float], c: float = 1.345, sd_floor: float = 0.005, iters: int = 50) -> float:
    """Huber M-estimate of location, scale fixed at max(1.4826 MAD, `sd_floor`), started at the median (IRLS)."""
    a = np.asarray(x, np.float64)
    if a.size == 0:
        raise ValueError("huber_location of an empty sample")
    mu = float(np.median(a))
    scale = max(1.4826 * float(np.median(np.abs(a - mu))), sd_floor)
    for _ in range(iters):
        r = np.abs(a - mu) / scale
        w = np.where(r <= c, 1.0, c / np.maximum(r, 1e-300))
        new = float(np.sum(w * a) / np.sum(w))
        if abs(new - mu) < 1e-12:
            return new
        mu = new
    return mu


def pooled_log_size(sizes: Mapping[str, float], cfg: PoolConfig | None = None) -> dict[str, float | None]:
    """Per window: the pooled log size of its (video, side) group, or None when pooling does not act.

    Args:
        sizes: window id -> log size, all windows of ONE (video, side) group with a usable size.
        cfg: settings; `leave_one_out` pools each window over the others only.
    """
    cfg = cfg or PoolConfig()
    keys = list(sizes)
    if len(keys) < cfg.min_windows:
        return dict.fromkeys(keys)
    out: dict[str, float | None] = {}
    for k in keys:
        pool = [sizes[o] for o in keys if not (cfg.leave_one_out and o == k)]
        out[k] = huber_location(pool, cfg.huber_c, cfg.sd_floor) if pool else None
    return out


def depth_factors(sizes: Mapping[str, float], cfg: PoolConfig | None = None) -> dict[str, float]:
    """`f_w = exp(clip(log s_pooled - log s_w, +-max_abs_log_f))` per window of one (video, side) group; 1.0 where
    pooling does not act. Multiply window w's hand (joints, vertices, translation) and its anchor depth by `f_w`."""
    cfg = cfg or PoolConfig()
    pooled = pooled_log_size(sizes, cfg)
    logf: dict[str, float] = {}
    for k, s in sizes.items():
        p = pooled[k]
        if p is None or not math.isfinite(s):
            logf[k] = 0.0
            continue
        logf[k] = float(np.clip(p - s, -cfg.max_abs_log_f, cfg.max_abs_log_f))
    if cfg.recentre and any(v != 0.0 for v in logf.values()):
        mean = float(np.mean(list(logf.values())))
        logf = {k: v - mean for k, v in logf.items()}
    return {k: float(math.exp(v)) for k, v in logf.items()}


def identity_flags(sizes: Mapping[str, float], cfg: PoolConfig | None = None) -> dict[str, bool]:
    """Per window of one (video, side) group: True when |log s_w - pool| > `outlier_k` x max(1.4826 MAD, `sd_floor`)
    of the group (a hand that may not be the wearer's). Groups below `min_windows` flag nothing. Descriptive only."""
    cfg = cfg or PoolConfig()
    keys = list(sizes)
    if len(keys) < cfg.min_windows:
        return dict.fromkeys(keys, False)
    a = np.array([sizes[k] for k in keys], np.float64)
    pool = huber_location(a, cfg.huber_c, cfg.sd_floor)
    scale = max(1.4826 * float(np.median(np.abs(a - np.median(a)))), cfg.sd_floor)
    return {k: bool(abs(v - pool) > cfg.outlier_k * scale) for k, v in zip(keys, a, strict=True)}


def compose_depth_scale(f_pool: float, f_side: float = 1.0) -> float:
    """The factor the anchor of one window-side gets: the pooled factor TIMES the per-side hand-depth factor, never one
    in place of the other (a per-side calibration, if ever enabled, must survive pooling). Applied ONCE: either to the
    hook or to the hand artefact the hook reads, not both (that gives the square)."""
    for name, f in (("f_pool", f_pool), ("f_side", f_side)):
        if not (math.isfinite(f) and f > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {f!r}")
    return float(f_pool) * float(f_side)


def gauge_follow_residual(scale_arm: Mapping[str, float], scale_base: Mapping[str, float],
                          f_expected: Mapping[str, float]) -> dict[str, float]:
    """KA-GAUGE: per window, |log(scale_arm / scale_base) - log f_expected| (0 when the solve follows its own factor
    exactly). Windows missing from either run are absent from the result."""
    out: dict[str, float] = {}
    for k, f in f_expected.items():
        if k in scale_arm and k in scale_base:
            out[k] = abs(math.log(scale_arm[k] / scale_base[k]) - math.log(f))
    return out
