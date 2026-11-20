# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""The change between two predictions of one unit, split along and across a ray (the solve-change estimand).

`alongray.score` and `e_par_split` split the error of ONE prediction against the truth. This module answers a
different question: when a method changes its prediction (for example from an initialiser to a solve), how much of
that change lies along the camera ray, and does the change in error follow the change along the ray?

Conventions, all fixed (there is no option that changes them silently):

- **Frame and units.** Camera-frame positions in metres, camera centre at the origin, one reference point per frame
  (for example the object centre), as (T, 3) arrays.
- **Correspondence.** Each array comes with its own integer frame ids. The three id sets must be identical (no
  intersection, no trimming, no shift); rows are joined on the ids and reported in increasing id order. Duplicate or
  non-integer ids are refused.
- **Paired finite mask.** A frame is used only if `before`, `after` and `gt` are all finite there and the ray used is
  not degenerate (norm above `min_range`). Every quantity of a window, both errors included, is computed on that ONE
  set of frames. Each dropped frame is counted by reason (a frame may have several) and listed by id.
- **Ray.** The change `d_t = after_t - before_t` is split on the unit ray `u_t` of `gt` (`ray="gt"`, the default) or
  of `before` (`ray="before"`, which needs no ground truth). The error components (`delta_*`) always use the true
  ray, since they are errors against the truth.
- **Window share.** `share_along = sum_t (d_t . u_t)^2 / sum_t |d_t|^2`: a ratio of sums within the window (frames
  with a larger change weigh more), NOT a mean of per-frame squared cosines. A window whose change is zero on every
  used frame has no share (`status="zero_change"`).
- **Signed and absolute.** `signed_change_along_mean` is the mean of `d_t . u_t` (positive: moved away from the
  camera); `abs_change_along_mean` the mean of its absolute value. The error terms use the signed along-ray error
  `a_t = (c_t - gt_t) . u_t`: `delta_abs_along_mean = <|a_after|> - <|a_before|>`, and the signed means are reported
  apart.
- **Pooling.** `summarize_changes` reports the MEDIAN over windows of the per-window share, and apart from it the
  ratio of pooled sums; the median of the PAIRED per-window centre-error differences, and apart from it the
  difference of the two arms' medians; and Spearman correlations over windows with average ranks for ties.

These are descriptive statistics of one cohort. Nothing here is a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["CHANGE_CONVENTIONS", "RAYS", "WindowChange", "summarize_changes", "window_change"]

RAYS = ("gt", "before")
DROP_REASONS = ("n_nonfinite_gt", "n_nonfinite_before", "n_nonfinite_after", "n_ray_degenerate")
CHANGE_CONVENTIONS = {
    "frame": "camera frame, camera centre at the origin",
    "units": "metres",
    "correspondence": "rows joined on integer frame ids; the three id sets must be identical",
    "mask": "one paired mask per window: before, after and gt finite and the ray norm above min_range",
    "window_share": "sum (d . u)^2 / sum |d|^2 over the window's used frames (ratio of sums)",
    "error_ray": "gt",
}


@dataclass(frozen=True)
class WindowChange:
    """One window's change between two predictions, with its frame flow. Positions in metres.

    Attributes:
        unit_id: The window's id.
        ray: ``"gt"`` or ``"before"``, the ray the change is split on.
        status: ``"scored"``, ``"zero_change"`` (no change on any used frame) or ``"no_frames"``.
        frame_ids_used: Ids of the used frames, increasing.
        dropped_ids: Ids of the dropped frames, increasing.
        flow: ``n_input``, ``n_used``, ``n_dropped``, the per-reason counts ``n_nonfinite_gt``,
            ``n_nonfinite_before``, ``n_nonfinite_after``, ``n_ray_degenerate`` (not exclusive) and
            ``n_zero_change`` (used frames whose change is exactly zero; they add 0 to both sums).
        sum_change_sq, sum_along_sq, sum_across_sq: Sums over used frames of |d|^2, (d.u)^2 and the rest.
        share_along: ``sum_along_sq / sum_change_sq``; None unless ``status == "scored"``.
        signed_change_along_mean, abs_change_along_mean: Means of ``d.u`` and ``|d.u|``.
        e_before_mean, e_after_mean: Mean centre error |c - gt| of each prediction on the used frames.
        delta_centre_mean: ``e_after_mean - e_before_mean``.
        delta_abs_along_mean, delta_perp_mean: The same difference for |a| and for the across-ray error norm.
        signed_along_before_mean, signed_along_after_mean: Mean signed along-ray error of each prediction.
    """

    unit_id: str
    ray: str
    status: str
    frame_ids_used: tuple[int, ...]
    dropped_ids: tuple[int, ...]
    flow: dict[str, int] = field(default_factory=dict)
    sum_change_sq: float = 0.0
    sum_along_sq: float = 0.0
    sum_across_sq: float = 0.0
    share_along: float | None = None
    signed_change_along_mean: float = float("nan")
    abs_change_along_mean: float = float("nan")
    e_before_mean: float = float("nan")
    e_after_mean: float = float("nan")
    delta_centre_mean: float = float("nan")
    delta_abs_along_mean: float = float("nan")
    delta_perp_mean: float = float("nan")
    signed_along_before_mean: float = float("nan")
    signed_along_after_mean: float = float("nan")


def _points(a: Any, name: str) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} must be a (T, 3) array of camera-frame points, got shape {x.shape}")
    return x


def _ids(a: Any, n: int, name: str) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim != 1 or x.shape[0] != n:
        raise ValueError(f"{name} frame ids must be a 1-D array with one id per row ({n}), got shape {x.shape}")
    if x.dtype.kind not in "iu":
        raise ValueError(f"{name} frame ids must be an integer array, got dtype {x.dtype}")
    if x.dtype.kind == "u" and x.size and int(x.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} frame ids must fit in int64, got {int(x.max())}")
    xi = x.astype(np.int64)
    if np.unique(xi).size != xi.size:
        raise ValueError(f"{name} frame ids contain duplicate ids")
    return xi


def _ordered(x: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return x[np.argsort(ids, kind="stable")]


def window_change(before: Any, after: Any, gt: Any, ids_before: Any, ids_after: Any, ids_gt: Any, *,
                  unit_id: str, ray: str = "gt", min_range: float = 1e-6) -> WindowChange:
    """Split the change ``after - before`` of one window along and across a ray (see the module docstring).

    Args:
        before, after: (T, 3) predicted reference points of the two arms, camera frame, metres.
        gt: (T, 3) true reference points.
        ids_before, ids_after, ids_gt: integer frame ids of each array's rows; the three sets must be identical.
        unit_id: The window's id.
        ray: ``"gt"`` (the true ray) or ``"before"`` (the first prediction's ray).
        min_range: Rays with norm at or below this (metres) are degenerate and their frames are dropped.

    Returns:
        A WindowChange.

    Raises:
        ValueError: a shape other than (T, 3), ids that are not one integer per row, duplicate ids, id sets that
            differ, an unknown ``ray`` or a non-positive ``min_range``.
    """
    if ray not in RAYS:
        raise ValueError(f"ray must be one of {RAYS}, got {ray!r}")
    if not (np.isfinite(min_range) and min_range > 0):
        raise ValueError(f"min_range must be finite and positive, got {min_range!r}")
    b, a, g = _points(before, "before"), _points(after, "after"), _points(gt, "gt")
    ib, ia, ig = _ids(ids_before, len(b), "before"), _ids(ids_after, len(a), "after"), _ids(ids_gt, len(g), "gt")
    sb, sa, sg = np.sort(ib), np.sort(ia), np.sort(ig)
    if not (np.array_equal(sb, sg) and np.array_equal(sa, sg)):
        only = sorted(set(map(int, sb)) ^ set(map(int, sg)) | set(map(int, sa)) ^ set(map(int, sg)))
        raise ValueError(f"the frame ids of before, after and gt differ (first differing ids {only[:5]}); align "
                         "them explicitly: rows are never intersected, trimmed or shifted")
    b, a, g, ids = _ordered(b, ib), _ordered(a, ia), _ordered(g, ig), sg
    fin_g, fin_b, fin_a = np.isfinite(g).all(1), np.isfinite(b).all(1), np.isfinite(a).all(1)
    r = g if ray == "gt" else b
    with np.errstate(invalid="ignore"):
        rn = np.linalg.norm(r, axis=1)
        gn = np.linalg.norm(g, axis=1)
    ok_ray = np.isfinite(rn) & (rn > min_range) & np.isfinite(gn) & (gn > min_range)
    fin_ray = (fin_g if ray == "gt" else fin_b) & fin_g
    use = fin_g & fin_b & fin_a & ok_ray
    flow = {"n_input": int(len(ids)), "n_used": int(use.sum()), "n_dropped": int((~use).sum()),
            "n_nonfinite_gt": int((~fin_g).sum()), "n_nonfinite_before": int((~fin_b).sum()),
            "n_nonfinite_after": int((~fin_a).sum()), "n_ray_degenerate": int((fin_ray & ~ok_ray).sum())}
    used, dropped = tuple(int(i) for i in ids[use]), tuple(int(i) for i in ids[~use])
    if not use.any():
        flow["n_zero_change"] = 0
        return WindowChange(unit_id=str(unit_id), ray=ray, status="no_frames", frame_ids_used=used,
                            dropped_ids=dropped, flow=flow)
    b, a, g, r = b[use], a[use], g[use], r[use]
    u = r / np.linalg.norm(r, axis=1, keepdims=True)
    ut = g / np.linalg.norm(g, axis=1, keepdims=True)
    d = a - b
    par = np.sum(d * u, axis=1)
    d2 = np.sum(d * d, axis=1)
    flow["n_zero_change"] = int((d2 == 0.0).sum())
    s_tot, s_par = float(d2.sum()), float((par ** 2).sum())
    eb, ea = b - g, a - g
    ab, aa = np.sum(eb * ut, axis=1), np.sum(ea * ut, axis=1)
    pb = np.linalg.norm(eb - ab[:, None] * ut, axis=1)
    pa = np.linalg.norm(ea - aa[:, None] * ut, axis=1)
    e_b, e_a = float(np.linalg.norm(eb, axis=1).mean()), float(np.linalg.norm(ea, axis=1).mean())
    status = "scored" if s_tot > 0.0 else "zero_change"
    return WindowChange(
        unit_id=str(unit_id), ray=ray, status=status, frame_ids_used=used, dropped_ids=dropped, flow=flow,
        sum_change_sq=s_tot, sum_along_sq=s_par, sum_across_sq=float(max(s_tot - s_par, 0.0)),
        share_along=(s_par / s_tot) if status == "scored" else None,
        signed_change_along_mean=float(par.mean()), abs_change_along_mean=float(np.abs(par).mean()),
        e_before_mean=e_b, e_after_mean=e_a, delta_centre_mean=e_a - e_b,
        delta_abs_along_mean=float(np.abs(aa).mean() - np.abs(ab).mean()), delta_perp_mean=float(pa.mean() - pb.mean()),
        signed_along_before_mean=float(ab.mean()), signed_along_after_mean=float(aa.mean()))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho with average ranks for ties; NaN below 3 pairs or for a constant input."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_changes(windows: Iterable[WindowChange], assigned: Sequence[str] | None = None) -> dict[str, Any]:
    """Pool windows of one cohort, with every denominator (see the module docstring).

    Args:
        windows: WindowChange records, one per window, all split on the same ray.
        assigned: Optional ids of every assigned window; an assigned id without a record counts as missing.

    Returns:
        A dict with the counts ``n_assigned``, ``n_windows``, ``n_scored``, ``n_zero_change``, ``n_no_frames``,
        ``n_missing`` (and ``missing_ids``), the pooled frame flow ``frames``, and over scored windows:
        ``median_window_share``, ``pooled_share_ratio_of_sums``, ``n_windows_share_ge_0.9``, ``min_window_share``,
        ``max_window_share``; over windows with frames: ``median_paired_delta_centre``,
        ``difference_of_median_centre``, ``n_windows_worse``, ``n_windows_better``, the two Spearman correlations;
        plus ``estimand`` (what each number is) and ``conventions``.

    Raises:
        ValueError: duplicate window ids, duplicate assigned ids, a window that is not assigned, or windows split
            on different rays.
    """
    ws = list(windows)
    ids = [w.unit_id for w in ws]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate window ids: each window must appear once")
    rays = {w.ray for w in ws}
    if len(rays) > 1:
        raise ValueError(f"windows split on different rays {sorted(rays)} cannot be pooled")
    missing: list[str] = []
    if assigned is not None:
        asg = [str(x) for x in assigned]
        if len(set(asg)) != len(asg):
            raise ValueError("duplicate assigned ids")
        extra = sorted(set(ids) - set(asg))
        if extra:
            raise ValueError(f"windows not assigned: {extra[:5]}")
        missing = [x for x in asg if x not in set(ids)]
    scored = [w for w in ws if w.status == "scored"]
    framed = [w for w in ws if w.status != "no_frames"]
    sh = np.array([w.share_along for w in scored], np.float64)
    tot = float(sum(w.sum_change_sq for w in scored))
    de = np.array([w.delta_centre_mean for w in framed], np.float64)
    frames = {k: int(sum(w.flow.get(k, 0) for w in ws)) for k in ("n_input", "n_used", "n_dropped",
                                                                     *DROP_REASONS, "n_zero_change")}
    nan = float("nan")
    out: dict[str, Any] = {
        "n_assigned": len(assigned) if assigned is not None else None, "n_windows": len(ws), "n_scored": len(scored),
        "n_zero_change": sum(w.status == "zero_change" for w in ws),
        "n_no_frames": sum(w.status == "no_frames" for w in ws), "n_missing": len(missing), "missing_ids": missing,
        "frames": frames,
        "median_window_share": float(np.median(sh)) if sh.size else nan,
        "pooled_share_ratio_of_sums": (float(sum(w.sum_along_sq for w in scored)) / tot) if tot > 0 else nan,
        "n_windows_share_ge_0.9": int((sh >= 0.9).sum()),
        "min_window_share": float(sh.min()) if sh.size else nan,
        "max_window_share": float(sh.max()) if sh.size else nan,
        "median_paired_delta_centre": float(np.median(de)) if de.size else nan,
        "difference_of_median_centre": (float(np.median([w.e_after_mean for w in framed])
                                              - np.median([w.e_before_mean for w in framed])) if framed else nan),
        "n_windows_worse": int((de > 0).sum()), "n_windows_better": int((de < 0).sum()),
        "spearman_delta_centre_vs_delta_abs_along": _spearman(de, [w.delta_abs_along_mean for w in framed]),
        "spearman_delta_centre_vs_delta_perp": _spearman(de, [w.delta_perp_mean for w in framed]),
        "spearman_ties": "average ranks",
        "estimand": {
            "median_window_share": "median over windows of each window's along-ray share of its squared change "
                                   "(ratio of sums within the window); over scored windows",
            "pooled_share_ratio_of_sums": "sum of along-ray squared change over sum of squared change, pooled over "
                                          "scored windows; not the paper's statistic, which is the median above",
            "median_paired_delta_centre": "median over windows of (mean after error - mean before error), paired on "
                                          "each window's own used frames",
            "difference_of_median_centre": "median of after errors minus median of before errors; unpaired, "
                                           "reported only for contrast",
            "spearman": "over windows with frames, of the paired per-window deltas",
        },
        "conventions": {**CHANGE_CONVENTIONS, "change_ray": (rays.pop() if rays else None)},
    }
    return out
