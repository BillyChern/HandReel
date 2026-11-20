"""A4 test: the IDENTIFIABLE-component motion test of a pre-contact lead, under MEASURED noise.

The noise constants (`A4_NOISE`) are measured robust sds (population: untouched rows of accepted ego-motion
windows).

What is tested. Monocular reconstruction cannot identify translation ALONG the camera ray, so the test uses only
the two perpendicular translation components and the rotation. Rows are the BINDING lead (the
structural run rows plus the skipped approach rows), in time order.

Statistics (row i, world frame):
- residuals r_i (6,) = [centre_i - centre medoid; Log(O_i S_g O_ref^T)] (`motion.pose_residuals`, m and rad);
- ray frame A_i (`rayframe.ray_frames`): the ray from row i's camera centre to the lead's ROBUST centre (geometric
  median of the centres), then the camera-x-derived perp-1 and perp-2 axes;
- per-row world covariance (reported, `a4_covariance`): blockdiag(A_i, A_i) diag(s^2) blockdiag(A_i, A_i)^T with s
  the measured robust sds (along-ray, perp-1, perp-2, rot about ray, rot perp-1, rot perp-2);
- whitening (`a4_whiteners`): the same transport with the ALONG-RAY precision set to 0, i.e. W_i = blockdiag(A_i,
  A_i) diag(0, 1/s_p1, 1/s_p2, 1/s_r0, 1/s_r1, 1/s_r2) blockdiag(A_i, A_i)^T (rank 5, kept in world coordinates so
  that sums over rows add like vectors even when the camera turns);
- z_i = W_i (r_i - a), a the Huber IRLS location, each z_i Huber-clipped to norm <= k = 3.55 (`ReferenceConfig`);
- TREND T = ||sum_i c_i z_i||^2 / sum_i c_i^2 (c_i = t_i - tbar) and SPLIT-HALF H = ||mean(first n//2) - mean(last
  n//2)||^2 (n >= 6), as `motion.py`;
- NULL: a BLOCK permutation. The rows, in lead order, are cut into consecutive blocks of `block_len` rows (the last
  block may be shorter); a null draw permutes the ORDER of the blocks and keeps each block's internal order; the
  permuted residuals take the time slots in order. With m blocks, all m! orders are enumerated when m! <= B + 1
  (exact p = #{stat_perm >= stat_obs} / m!, the identity included); otherwise B = 1999 random orders,
  p = (1 + #) / (B + 1) (`default_rng(seed)`). The smallest achievable combined p is reported (`min_p`): with
  m <= 3 blocks (m! <= 6, Bonferroni 2/m! >= 0.33) no lead can be significant;
- combined p = min(1, 2 min(p_T, p_H)) (p_T alone when n < 6); SIGNIFICANT iff combined p <= alpha = 0.05 (one-sided).
Magnitudes (physical units; P = I - ubar ubar^T removes the lead's mean ray direction ubar): a Huber IRLS line
r_i ~ a + b c_i under the same whitening gives the trend's end-to-end change, ||P b_pos|| (t_max - t_min) and
||b_rot|| (t_max - t_min); Huber IRLS locations of the halves give the half discrepancies. Per block the magnitude
is the larger of the two.
Decision: MOVED iff significant AND (perpendicular magnitude > 20 mm OR rotation magnitude > 15 deg). Fewer than 5
rows: undecided.

Block length (from (a)): block_len = ceil(max(tau_int(perp), tau_int(rotation))) = ceil(max(7.81, 5.73)) = 8 rows,
tau_int the robust integrated correlation time of the pooled components (perp's is TRUNCATED at lag 10, so 7.81 is
a lower bound; the video-cluster 95% intervals are [3.4, 14.1] and [4.3, 11.9] rows).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from fastdo.reference.config import ReferenceConfig
from fastdo.reference.motion import _clip, _irls, pose_residuals
from fastdo.reference.rayframe import ray_frames

__all__ = [
    "A4_NOISE",
    "A4Noise",
    "A4Test",
    "a4_covariance",
    "a4_motion_test",
    "a4_motion_test_residuals",
    "a4_whiteners",
    "block_orders",
    "geometric_median",
]


@dataclass(frozen=True)
class A4Noise:
    """Measured per-row noise in the ray frame (m, m, m, rad, rad, rad) and the block length (rows)."""
    sd: tuple[float, float, float, float, float, float]
    block_len: int
    source: str = ""
    floor_perp_m: float = 0.020
    floor_deg: float = 15.0
    alpha: float = 0.05
    permutations: int = 1999


# Measured robust sds (untouched rows of accepted ego-motion windows);
# tau_int: perp 7.81 (truncated), rotation 5.73 -> block_len 8.
A4_NOISE = A4Noise(
    # all six values carry the measurement's own digits (pinned by tests/reference/test_a4.py)
    sd=(0.02532439773426027, 0.0063767286870088995, 0.00520147366483646, math.radians(2.3812747814520385),
        math.radians(1.862999384392682), math.radians(2.2707974600174032)),
    block_len=8,
    source="robust sds of untouched rows of accepted ego-motion windows (primary/ego_motion)")


@dataclass(frozen=True)
class A4Test:
    n: int
    decided: bool
    moved: bool
    significant: bool
    exceeds_floor: bool
    p_value: float | None
    p_trend: float | None
    p_half: float | None
    n_blocks: int
    block_len: int
    exact: bool
    min_p: float | None              # the smallest combined p this lead length and block length can give
    perp_mm: float | None            # perpendicular magnitude (max of trend and half discrepancy)
    rot_deg: float | None
    trend_perp_mm: float | None
    trend_deg: float | None
    half_perp_mm: float | None
    half_deg: float | None
    floor_perp_mm: float
    floor_deg: float
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def geometric_median(X: np.ndarray, iters: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Weiszfeld's geometric median of the rows of X (n,d)."""
    X = np.asarray(X, dtype=np.float64)
    m = np.median(X, axis=0)
    for _ in range(iters):
        d = np.linalg.norm(X - m, axis=1)
        if np.any(d < 1e-15):
            w = (d < 1e-15).astype(float)
            return (w @ X) / w.sum()
        w = 1.0 / d
        nxt = (w @ X) / w.sum()
        if np.linalg.norm(nxt - m) < tol:
            return nxt
        m = nxt
    return m


def _block6(A: np.ndarray) -> np.ndarray:
    n = len(A)
    M = np.zeros((n, 6, 6))
    M[:, :3, :3] = A
    M[:, 3:, 3:] = A
    return M


def a4_covariance(A: np.ndarray, noise: A4Noise = A4_NOISE) -> np.ndarray:
    """(n,6,6) per-row WORLD covariance of [centre; world-left rotation] from the ray-frame sds (module docstring)."""
    M = _block6(np.asarray(A, dtype=np.float64))
    return M @ np.diag(np.square(noise.sd)) @ np.transpose(M, (0, 2, 1))


def a4_whiteners(A: np.ndarray, noise: A4Noise = A4_NOISE) -> np.ndarray:
    """(n,6,6) rank-5 world whiteners: along-ray precision 0, the rest 1/sd (module docstring)."""
    M = _block6(np.asarray(A, dtype=np.float64))
    d = np.array([0.0, *(1.0 / np.asarray(noise.sd[1:]))])
    return M @ np.diag(d) @ np.transpose(M, (0, 2, 1))


def block_orders(n: int, block_len: int, B: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """(orders (K,n) row-index arrays, exact): every block order when m! <= B + 1, else the identity plus B random
    orders. Blocks are consecutive rows in lead order; the last block may be shorter."""
    b = max(1, int(block_len))
    blocks = [np.arange(s, min(s + b, n)) for s in range(0, n, b)]
    m = len(blocks)
    if math.factorial(m) <= B + 1:
        return np.stack([np.concatenate([blocks[j] for j in o]) for o in itertools.permutations(range(m))]), True
    out = [np.arange(n)]
    for _ in range(B):
        out.append(np.concatenate([blocks[j] for j in rng.permutation(m)]))
    return np.stack(out), False


def _p(stat_perm: np.ndarray, obs: float) -> float:
    """#{draws >= obs} / #draws; the identity order is among the draws (exact: one of the m! orders; random: row 0,
    the "+1" of (1 + #) / (B + 1))."""
    return float(np.sum(stat_perm >= obs - 1e-12)) / len(stat_perm)


def a4_motion_test_residuals(times: np.ndarray, r: np.ndarray, A: np.ndarray, noise: A4Noise = A4_NOISE,
                             cfg: ReferenceConfig | None = None, seed: int = 0) -> A4Test:
    """The A4 test on precomputed world residuals `r (n,6)` (m, rad) with ray frames `A (n,3,3)` (module docstring)."""
    cfg = cfg or ReferenceConfig()
    tau = np.asarray(times, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    n = len(tau)
    fl_mm, fl_deg = noise.floor_perp_m * 1e3, noise.floor_deg
    m_blocks = math.ceil(n / max(1, noise.block_len)) if n else 0
    if n < cfg.motion_min_rows:
        return A4Test(n, False, False, False, False, None, None, None, m_blocks, noise.block_len, False, None, None,
                      None, None, None, None, None, fl_mm, fl_deg, "too_few_rows")
    W = a4_whiteners(A, noise)
    k = cfg.huber_k
    loc = _irls(np.broadcast_to(np.eye(6), (n, 6, 6)).copy(), r, W, k)
    z = _clip(np.einsum("nij,nj->ni", W, r - loc), k)
    c = tau - tau.mean()
    den = float(c @ c) if float(c @ c) > 0 else 1.0
    orders, exact = block_orders(n, noise.block_len, noise.permutations, np.random.default_rng(seed))
    zp = z[orders]
    t_obs = float(np.sum((c @ z) ** 2) / den)
    t_perm = np.sum(np.einsum("n,bnd->bd", c, zp) ** 2, axis=1) / den
    p_t = _p(t_perm, t_obs)
    use_half = n >= cfg.split_half_min_vote
    if use_half:
        h = n // 2
        h_obs = float(np.sum((z[:h].mean(0) - z[n - h:].mean(0)) ** 2))
        h_perm = np.sum((zp[:, :h].mean(1) - zp[:, n - h:].mean(1)) ** 2, axis=1)
        p_h = _p(h_perm, h_obs)
        p = min(1.0, 2.0 * min(p_t, p_h))
    else:
        p_h, p = None, p_t
    K = len(orders)
    min_p = min(1.0, (2.0 if use_half else 1.0) / K)
    X = np.zeros((n, 6, 12))
    X[:, :, :6] = np.eye(6)
    X[:, :, 6:] = c[:, None, None] * np.eye(6)
    b = _irls(X, r, W, k)[6:]
    ubar = A[:, :, 0].mean(axis=0)
    ubar = ubar / np.linalg.norm(ubar)
    P = np.eye(3) - np.outer(ubar, ubar)
    span = float(np.ptp(tau))
    tr_mm = float(np.linalg.norm(P @ b[:3]) * span * 1e3)
    tr_deg = float(np.degrees(np.linalg.norm(b[3:]) * span))
    hd_mm = hd_deg = None
    if use_half:
        h = n // 2
        I6 = np.broadcast_to(np.eye(6), (h, 6, 6))
        a1 = _irls(I6.copy(), r[:h], W[:h], k)
        a2 = _irls(I6.copy(), r[n - h:], W[n - h:], k)
        hd_mm = float(np.linalg.norm(P @ (a2 - a1)[:3]) * 1e3)
        hd_deg = float(np.degrees(np.linalg.norm((a2 - a1)[3:])))
    mag_mm = max(tr_mm, hd_mm or 0.0)
    mag_deg = max(tr_deg, hd_deg or 0.0)
    sig = p <= noise.alpha
    exceeds = mag_mm > fl_mm or mag_deg > fl_deg
    return A4Test(n, True, bool(sig and exceeds), bool(sig), bool(exceeds), p, p_t, p_h, m_blocks, noise.block_len,
                  exact, min_p, mag_mm, mag_deg, tr_mm, tr_deg, hd_mm, hd_deg, fl_mm, fl_deg,
                  "" if min_p <= noise.alpha else "too_few_blocks", {"trend_stat": t_obs, "null_draws": K})


def a4_motion_test(times: np.ndarray, p: np.ndarray, O: np.ndarray, cam_centres: np.ndarray, cam_rots: np.ndarray,
                   elements: np.ndarray | None = None, noise: A4Noise = A4_NOISE, cfg: ReferenceConfig | None = None,
                   seed: int = 0) -> A4Test:
    """A4 entry point: did the object MOVE over the binding lead, in the identifiable components?

    Args:
        times: (n,) row times (s), increasing (the lead's rows: run rows plus skipped approach rows).
        p: (n,3) world positions of the object's CENTRE c* (m) (`SymmetryGroup.c_star` posed).
        O: (n,3,3) world rotations.
        cam_centres: (n,3) world camera centres of the same rows (the screen's world frame).
        cam_rots: (n,3,3) rigid camera-to-world rotations of the same rows.
        elements: optional (K,3,3) symmetry group (index 0 identity).
        noise: the measured noise and block length (default: step (a)'s record).
        seed: the random block orders' seed (used only when the orders are not enumerated).
    """
    p = np.asarray(p, dtype=np.float64)
    A = ray_frames(cam_centres, cam_rots, geometric_median(p))
    return a4_motion_test_residuals(times, pose_residuals(p, O, elements), A, noise, cfg, seed)
