"""Branch-robust keyframe registration: solve from several registration hypotheses, keep one by a
label-free selector.

The along-ray instability of a re-solve is the KEYFRAME REGISTRATION's rotation-hypothesis choice. The 120
ICP hypotheses of `keyframe.register_keyframe_pose` converge into a few near-tied basins, and +-1-2% of initial scale or
seed + 1 flips the argmin (> 20 deg, often ~180 deg) in 44/48 perturbations, moving the track anchors up to 55 mm along
their rays; with the anchors fixed the LM is stable (0.05 mm). `solve_stage.branch_select` (`BranchSelectConfig`;
None = today's stage) turns that discrete choice into an explicit enumeration:

1. `prepare_solve` runs the registration at every `BranchSelectConfig.runs()` (seed + offset, init scale x mult; the
   production run first) with `register_keyframe_hypotheses(top_k)`. The production run's element 0 IS
   `register_keyframe_pose`'s result (identical operations), so the SolveCall `prepare_solve` returns is production's;
   every other hypothesis becomes a SolveCall of its own (`workers_solve.call_from_registration`: the production
   anchors, PnP and init gate, at the production seed), carried in the context under `BRANCH_KEY`;
2. after the production solve and its depth-anchor gate, `apply_branch_select` solves every clip's other candidates in
   ONE more `solve_many` call (production solver config) and gates them the same way;
3. `selector` scores every candidate (lower is better; the first minimum wins, so a tie keeps production's; with
   `margin` m another candidate wins only if its value is below (1 - m) x production's):
     reg_score  the registration's own score (production's criterion, now across runs)
     mf_score   the registration's criterion on EVERY frame at the solved poses: per frame with >= `MIN_POINTS`
                masked pointmap points, truncated (0.1 diameter) symmetric Chamfer between the frame's masked pointmap
                (a fixed-seed subsample of `N_OBS`) and the posed front-facing template, + 0.1 diameter x the fraction
                of front-facing posed samples outside the (2 px dilated) mask, divided by the diameter; mean over
                frames
     sil_iou    1 - IoU of the mask and the splat (3 px dilation) of the posed front-facing dense anchor samples, mean
                over frames with a mask (depth-free)
     sil_cov    the fraction of the mask the same splat does NOT cover, mean over frames with a mask (depth-free; the
                mask excludes the hand, so a hand-occluded part of the template costs nothing)
     medoid     the candidate whose solved template-centroid track has the least median (over the other candidates)
                mean per-frame distance to the others
4. the chosen candidate's call (its own registration and anchors in the context) and kept solution replace the
   production pair; `branch_record` (every candidate's run, rank, registration score, scale and selector value, and
   the chosen index) goes into the context (`BRANCH_KEY`) and `finish_solve` writes it to solve.json's
   `branch_select` block. G (the per-frame re-anchor) and contact then run on the chosen solution exactly as on
   production's.
A clip whose chosen candidate is the production one writes the production solve.npz byte for byte.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fastdo.engine.depth_anchor_gate import SolveManyFn, apply_depth_anchor_gate
from fastdo.engine.solve_stage_config import BranchSelectConfig, DepthAnchorGateConfig
from fastdo.solver.api import SolverConfig, TrajectorySolution
from fastdo.solver.batch import valid_pixels_of

__all__ = ["BRANCH_KEY", "MIN_POINTS", "N_OBS", "apply_branch_select", "branch_record", "candidate_values",
           "mf_score", "sil_cov", "sil_iou"]

logger = logging.getLogger(__name__)

BRANCH_KEY = "branch_select"       # the SolveCall context key and the solve.json block
TRUNC_FRAC, SIL_WEIGHT, MIN_POINTS, N_OBS, DILATE_PX, SPLAT_PX = 0.1, 0.1, 20, 512, 2, 3


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (F.max_pool2d(m, 2 * r + 1, stride=1, padding=r)[0, 0] > 0).numpy()


def _diameter(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(0) - points.min(0)))


def _splat_cover(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """(|splat & mask|, |splat | mask|), the splat = pixels (u, v) dilated by SPLAT_PX (on the joint bounding box)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    y0 = max(0, min(ys.min(), v.min()) - SPLAT_PX)
    y1 = min(H, max(ys.max(), v.max()) + SPLAT_PX + 1)
    x0 = max(0, min(xs.min(), u.min()) - SPLAT_PX)
    x1 = min(W, max(xs.max(), u.max()) + SPLAT_PX + 1)
    sp = np.zeros((y1 - y0, x1 - x0), bool)
    sp[v - y0, u - x0] = True
    sp = _dilate(sp, SPLAT_PX)
    m = mask[y0:y1, x0:x1]
    return int((sp & m).sum()), int((sp | m).sum()) + int(mask.sum() - m.sum())


def _front(points: np.ndarray, normals: np.ndarray, pose: np.ndarray, scale: float) -> np.ndarray:
    P = scale * points @ pose[:3, :3].T + pose[:3, 3]
    return P[((normals @ pose[:3, :3].T) * -P).sum(1) > 0]


def _project(P: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv = (P / np.maximum(P[:, 2:3], 1e-9)) @ K.T
    return np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)


def mf_score(points: np.ndarray, normals: np.ndarray, frames: Sequence[np.ndarray], masks: np.ndarray,
             dilated: Sequence[np.ndarray], K: np.ndarray, poses: np.ndarray, scale: float, seed: int = 0) -> float:
    """The multi-frame registration criterion (module docstring, step 3); NaN when no frame qualifies."""
    from scipy.spatial import cKDTree

    diam = scale * _diameter(points)
    tau = TRUNC_FRAC * diam
    rng = np.random.default_rng(seed)
    H, W = masks.shape[1:]
    rows = []
    for t, obs_t in enumerate(frames):
        if masks[t].sum() < MIN_POINTS or len(obs_t) < MIN_POINTS:
            continue
        o = obs_t[rng.permutation(len(obs_t))[:N_OBS]]
        Qf = _front(points, normals, poses[t], scale)
        if not len(Qf):
            continue
        d_obs = cKDTree(Qf).query(o)[0]
        d_mesh = cKDTree(o).query(Qf)[0]
        u, v = _project(Qf, K)
        ins = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Qf[:, 2] > 0)
        hit = np.zeros(len(u), bool)
        hit[ins] = dilated[t][v[ins], u[ins]]
        ch = float(np.minimum(d_obs, tau).mean() + np.minimum(d_mesh, tau).mean())
        rows.append((ch + SIL_WEIGHT * diam * float((~hit).mean())) / diam)
    return float(np.mean(rows)) if rows else math.nan


def _silhouette(dense_points: np.ndarray, dense_normals: np.ndarray, masks: np.ndarray, K: np.ndarray,
                poses: np.ndarray, scale: float) -> tuple[float, float]:
    """(sil_cov, sil_iou): means over frames with >= MIN_POINTS mask pixels of 1 - |splat & mask| / |mask| and of
    1 - IoU(mask, splat), the splat = the posed front-facing samples' pixels dilated by SPLAT_PX."""
    H, W = masks.shape[1:]
    cov, iou = [], []
    for t in range(masks.shape[0]):
        area = int(masks[t].sum())
        if area < MIN_POINTS:
            continue
        P = _front(dense_points, dense_normals, poses[t], scale)
        P = P[P[:, 2] > 1e-6]
        if not len(P):
            cov.append(1.0)
            iou.append(1.0)
            continue
        u, v = _project(P, K)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        inter, union = (_splat_cover(u[inside], v[inside], masks[t]) if inside.any() else (0, area))
        cov.append(1.0 - inter / area)
        iou.append(1.0 - inter / max(union, 1))
    return (float(np.mean(cov)), float(np.mean(iou))) if cov else (math.nan, math.nan)


def sil_iou(dense_points: np.ndarray, dense_normals: np.ndarray, masks: np.ndarray, K: np.ndarray,
            poses: np.ndarray, scale: float) -> float:
    """Mean over frames with >= MIN_POINTS mask pixels of 1 - IoU(mask, splat of the posed front-facing samples)."""
    return _silhouette(dense_points, dense_normals, masks, K, poses, scale)[1]


def sil_cov(dense_points: np.ndarray, dense_normals: np.ndarray, masks: np.ndarray, K: np.ndarray,
            poses: np.ndarray, scale: float) -> float:
    """Mean over frames with >= MIN_POINTS mask pixels of the fraction of mask pixels the splat of the posed
    front-facing samples does NOT cover (depth-free; the mask excludes the hand, so occlusion is not penalized)."""
    return _silhouette(dense_points, dense_normals, masks, K, poses, scale)[0]


def _centroid_track(solution: TrajectorySolution, points: np.ndarray) -> np.ndarray:
    poses = solution.poses.detach().cpu().double().numpy()
    return float(solution.scale) * poses[:, :3, :3] @ points.mean(0) + poses[:, :3, 3]


def candidate_values(selector: str, calls: Sequence[Any], solutions: Sequence[TrajectorySolution],
                     dense: tuple[Tensor, Tensor]) -> list[float]:
    """Every candidate's selector value (lower is better; NaN ranks last). `calls[i].context["registration"]` is
    candidate i's `KeyframeRegistration`; `dense` = the clip's (anchor-surface points, normals)."""
    if selector == "reg_score":
        return [float(c.context["registration"].score) for c in calls]
    points = calls[0].mesh.points.detach().cpu().double().numpy()
    if selector == "medoid":
        C = np.stack([_centroid_track(s, points) for s in solutions])
        D = np.linalg.norm(C[:, None] - C[None], axis=-1).mean(-1)
        return [float(np.median(np.delete(D[i], i))) if len(C) > 1 else 0.0 for i in range(len(C))]
    obs = calls[0].obs                  # every candidate shares the clip's masks and pointmaps
    masks = np.asarray(obs.masks.detach().cpu().numpy(), bool)
    K = np.asarray(calls[0].context["K"], np.float64) if not isinstance(calls[0].context["K"], Tensor) else \
        calls[0].context["K"].detach().cpu().double().numpy()
    if selector in ("sil_iou", "sil_cov"):
        dp = dense[0].detach().cpu().double().numpy()
        dn = F.normalize(dense[1].detach().cpu().double(), dim=-1).numpy()
        fn = sil_iou if selector == "sil_iou" else sil_cov
        return [fn(dp, dn, masks, K, s.poses.detach().cpu().double().numpy(), float(s.scale)) for s in solutions]
    if selector == "mf_score":
        pts, counts = valid_pixels_of(obs)
        pts_np = pts.detach().cpu().double().numpy()
        off = np.concatenate([[0], np.cumsum(counts.cpu().numpy())])
        frames = [pts_np[off[t]:off[t + 1]] for t in range(len(counts))]
        dilated = [_dilate(m, DILATE_PX) for m in masks]
        normals = calls[0].mesh.normals.detach().cpu().double().numpy()
        return [mf_score(points, normals, frames, masks, dilated, K, s.poses.detach().cpu().double().numpy(),
                         float(s.scale)) for s in solutions]
    raise ValueError(f"unknown branch selector {selector!r}")


def branch_record(cfg: BranchSelectConfig, meta: Sequence[Mapping[str, Any]], values: Sequence[float],
                  chosen: int) -> MappingProxyType[str, Any]:
    """solve.json's `branch_select` block."""
    return MappingProxyType({
        "top_k": cfg.top_k, "seed_offsets": list(cfg.seed_offsets), "scale_mults": list(cfg.scale_mults),
        "selector": cfg.selector, "min_rot_deg": cfg.min_rot_deg, "min_shift": cfg.min_shift, "margin": cfg.margin,
        "n_candidates": len(meta), "chosen": int(chosen), "production_kept": int(chosen) == 0,
        "candidates": [{**dict(m), "value": None if not math.isfinite(v) else float(v)}
                       for m, v in zip(meta, values, strict=True)]})


def apply_branch_select[CallT](clip_ids: Sequence[str], calls: Sequence[CallT], solutions: Sequence[TrajectorySolution],
                               cfg: BranchSelectConfig, base: SolverConfig, gate: DepthAnchorGateConfig | None,
                               solve_many: SolveManyFn) -> tuple[list[CallT], list[TrajectorySolution]]:
    """Steps 2-4 of the module docstring for a group of production (gated) solves.

    Raises:
        KeyError: If a call was prepared without `branch_select` (no `BRANCH_KEY` in its context).
        RuntimeError: If `solve_many` returns a different number of solutions than it was given clips.
    """
    extras: list[tuple[int, Any]] = []
    for i, call in enumerate(calls):
        for cand in call.context[BRANCH_KEY]["candidates"]:      # type: ignore[attr-defined]
            extras.append((i, cand))
    solved: list[TrajectorySolution] = []
    ex_calls: list[Any] = [c for _, c in extras]
    if extras:
        solved = solve_many([c.mesh for c in ex_calls], [c.obs for c in ex_calls], [c.init_poses for c in ex_calls],
                            [c.init_scale for c in ex_calls], base)
        if len(solved) != len(ex_calls):
            raise RuntimeError(f"the branch re-solve returned {len(solved)} solutions for {len(ex_calls)} candidates")
        if gate is not None:
            ids = [f"{clip_ids[i]}#b{j}" for j, (i, _) in enumerate(extras)]
            ex_calls, solved = apply_depth_anchor_gate(ids, ex_calls, solved, gate, base, solve_many)
    out_calls: list[CallT] = []
    kept: list[TrajectorySolution] = []
    for i, (clip_id, call, solution) in enumerate(zip(clip_ids, calls, solutions, strict=True)):
        info = call.context[BRANCH_KEY]                               # type: ignore[attr-defined]
        c_calls = [call] + [c for (j, _), c in zip(extras, ex_calls, strict=True) if j == i]
        c_sols = [solution] + [s for (j, _), s in zip(extras, solved, strict=True) if j == i]
        values = candidate_values(cfg.selector, c_calls, c_sols, info["dense"])
        v = np.asarray(values, np.float64)
        chosen = int(np.argmin(np.where(np.isfinite(v), v, np.inf))) if np.isfinite(v).any() else 0
        if chosen and math.isfinite(v[0]) and not v[chosen] < v[0] * (1.0 - cfg.margin):
            chosen = 0                                    # the production-anchored margin (BranchSelectConfig.margin)
        meta = [{"run": list(m["run"]), "rank": int(m["rank"]), "reg_score": float(m["reg_score"]),
                 "scale": float(s.scale)} for m, s in zip(info["meta"], c_sols, strict=True)]
        record = branch_record(cfg, meta, values, chosen)
        pick = c_calls[chosen]
        ctx = {key: val for key, val in pick.context.items() if key != BRANCH_KEY}   # type: ignore[attr-defined]
        out_calls.append(dataclasses.replace(pick, context=MappingProxyType({**ctx, BRANCH_KEY: record})))  # type: ignore[type-var]
        kept.append(c_sols[chosen])
        logger.info("branch select %s: %s over %d candidates -> %d (%s)", clip_id, cfg.selector, len(c_calls), chosen,
                    ", ".join("nan" if not math.isfinite(x) else f"{x:.5f}" for x in values))
    return out_calls, kept

