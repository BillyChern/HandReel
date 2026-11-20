"""Apply a smoother arm to one window's poses (post-solve, post-contact; CPU).

Input: BASE's written trajectory (T, 4, 4) with object frame X = s R p + t, its scale s, the template centroid p_bar
(surface-sample mean of the predicted mesh, the scorer's `mesh_centroid`, label-free), the rows' frame_idx and,
for the hand arms, the contact side's HaWoR wrist (T, 3) and the production contact flags (T,).
Output: poses with the same s whose centroid frame (R_k, c_k = s R_k p_bar + t_k) is the smoothed one; rows BASE did
not deliver stay NaN (the smoother never creates a pose); for the hand arms a per-row translation (T, 3) to add to
every hand vertex and joint of that side (zero where no hand).

Arms (`SmootherConfig.arm`):
- `se3`: object-only SE(3) (or SO(3) x R^3, `process`) error-state RTS, anisotropic noise, gauge held (PRIMARY);
- `se3_contact`: the joint object + wrist smoother with the grasp-rigidity factor on contact rows;
- `se3_cotransport`: `se3`, then on contact rows the wrist is rescaled about the camera centre by the object's
  per-row range ratio |c_hat_k| / |c_k| (keeps the hand-object along-ray ratio the contact stage set);
- `alongray`: AR1D (`fastdo.smooth.alongray`);
- `alongray_contact`: AR1D-HO, the 1-D joint object + wrist range smoother with a range-velocity rigidity factor on
  contact rows (`fastdo.smooth.alongray.alongray_joint`; candidate C2).

Domain gate (`SmootherConfig.contact_only`, True in both candidates): the window is returned unchanged
(`record['skipped']`) unless the production contact stage ran on it with a contact side (a wrist and contact flags
with at least one flagged row are passed) and its declared source domain is not egocentric. The engine stage passes
the clip's `source_domain.json` domain; egocentric video therefore stays at BASE byte-for-byte (KA-EGO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fastdo.smooth.alongray import alongray_joint, alongray_ranges
from fastdo.smooth.config import SmootherConfig
from fastdo.smooth.rts import smooth_se3

__all__ = ["SmoothOutput", "row_spacing", "smooth_window"]


@dataclass(frozen=True)
class SmoothOutput:
    poses: np.ndarray
    hand_shift: np.ndarray | None
    record: dict[str, Any] = field(default_factory=dict)


def row_spacing(frame_idx: np.ndarray) -> np.ndarray:
    """D_k in rows: frame differences over the window's median difference (1 for a regular window)."""
    d = np.diff(np.asarray(frame_idx, np.float64))
    if d.size == 0 or np.any(d <= 0):
        raise ValueError("frame_idx must be strictly increasing with at least 2 rows")
    return d / float(np.median(d))


def gate(cfg: SmootherConfig, wrist: np.ndarray | None, contact: np.ndarray | None,
         source_domain: str | None) -> str | None:
    """Why the stage must leave this window at BASE (None = it acts)."""
    if not cfg.contact_only:
        return None
    if source_domain is not None and str(source_domain) == "egocentric":
        return "declared egocentric: the contact stage does not run, the smoother does not either"
    if wrist is None or contact is None or not np.any(np.asarray(contact, bool)):
        return "no contact-stage output with a contact side on this window"
    return None


def smooth_window(poses: np.ndarray, scale: float, centroid: np.ndarray, frame_idx: np.ndarray,
                  cfg: SmootherConfig, wrist: np.ndarray | None = None,
                  contact: np.ndarray | None = None, source_domain: str | None = None) -> SmoothOutput:
    """One window (module docstring). Gated off, or fewer than 3 delivered rows: returned unchanged
    (`record['skipped']`)."""
    P = np.asarray(poses, np.float64)
    why = gate(cfg, wrist, contact, source_domain)
    if why is not None:
        return SmoothOutput(P.copy(), None, {"skipped": why, "arm": cfg.arm})
    ok = np.isfinite(P).all(axis=(1, 2))
    if ok.sum() < 3:
        return SmoothOutput(P.copy(), None, {"skipped": "fewer than 3 delivered rows", "arm": cfg.arm})
    p_bar = np.asarray(centroid, np.float64)
    s = float(scale)
    R_m = np.where(ok[:, None, None], P[:, :3, :3], np.nan)
    c_m = np.full((len(P), 3), np.nan)
    c_m[ok] = s * P[ok, :3, :3] @ p_bar + P[ok, :3, 3]
    dt = row_spacing(frame_idx)
    zbar = float(np.median(np.linalg.norm(c_m[ok], axis=1)))
    rec: dict[str, Any] = {"arm": cfg.arm, "process": cfg.process, "zbar_m": zbar, "n_rows": len(P),
                           "n_delivered": int(ok.sum())}
    out = P.copy()
    shift = None
    if cfg.arm == "alongray":
        c_hat, gamma, w = alongray_ranges(c_m, dt, cfg.a_par * cfg.noise_mult, cfg.band * cfg.band_mult,
                                          cfg.huber_k, cfg.reweights, cfg.gauge_hold)
        R_hat = R_m
        rec.update(gamma=gamma, downweighted=int(np.sum(w[ok] < 1.0)))
    elif cfg.arm == "alongray_contact":
        jr = alongray_joint(c_m, None if wrist is None else np.asarray(wrist, np.float64),
                            contact if cfg.couple else None, dt, cfg.noise(zbar), gauge_hold=cfg.gauge_hold)
        c_hat, R_hat = jr.c, R_m
        rec.update(gamma=jr.gamma, gamma_hand=jr.gamma_hand, downweighted_c=int(np.nansum(jr.w_c < 1.0)),
                   downweighted_g=int(np.nansum(jr.w_g < 1.0)), coupled_rows=int(np.isfinite(jr.w_g).sum()))
        if jr.hand is not None:
            wm = np.asarray(wrist, np.float64)
            shift = np.where(np.isfinite(wm) & np.isfinite(jr.hand), jr.hand - wm, 0.0)
    else:
        nz = cfg.noise(zbar)
        use_hand = cfg.arm == "se3_contact" and wrist is not None
        res = smooth_se3(R_m, c_m, dt, nz, hand_m=np.asarray(wrist, np.float64) if use_hand else None,
                         contact=contact if (use_hand and cfg.couple) else None, gauge_hold=cfg.gauge_hold)
        c_hat = res.c
        R_hat = res.R if cfg.smooth_rotation else R_m
        rec.update(gamma=res.gamma, gamma_hand=res.gamma_hand, n_gn=res.n_gn,
                   downweighted_c=int(np.nansum(res.w_c < 1.0)), downweighted_R=int(np.nansum(res.w_R < 1.0)),
                   coupled_rows=int(np.isfinite(res.w_g).sum()) if res.w_g is not None else 0)
        if use_hand and res.hand is not None:
            wm = np.asarray(wrist, np.float64)
            shift = np.where(np.isfinite(wm), res.hand - wm, 0.0)
        if cfg.arm == "se3_cotransport" and wrist is not None and contact is not None:
            wm = np.asarray(wrist, np.float64)
            con = np.asarray(contact, bool) & ok & np.isfinite(wm).all(1)
            ratio = np.ones(len(P))
            ratio[con] = np.linalg.norm(c_hat[con], axis=1) / np.linalg.norm(c_m[con], axis=1)
            shift = np.where(con[:, None], (ratio[:, None] - 1.0) * np.nan_to_num(wm), 0.0)
            rec["cotransport_rows"] = int(con.sum())
    out[ok, :3, :3] = R_hat[ok]
    out[ok, :3, 3] = c_hat[ok] - s * R_hat[ok] @ p_bar
    rec["max_abs_range_change_mm"] = float(np.max(np.abs(np.linalg.norm(c_hat[ok], axis=1)
                                                          - np.linalg.norm(c_m[ok], axis=1))) * 1e3)
    return SmoothOutput(out, shift, rec)
