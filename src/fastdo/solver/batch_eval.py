"""The batched LM evaluate callback over a `batch.DeviceBatch`.

One call evaluates the frames of every clip, or of a named subset of clips: visibility-aware template measure
partial-OT pointmap term with a (kNN-truncated at the fine level) Sinkhorn solve
warm-started from cached potentials, and the Cauchy track term; the temporal priors always cover
every frame (they are cheap and the LM assembles every clip). Per-clip constants (diameter^2 normalization of
eps/rho, the track weight) become per-frame tensors built once per annealing level (`FrameInputs`).

Caching follows the single-clip solver exactly (it is the B = 1 case): `EvalCache` holds the visibility
mask, the plan and the Sinkhorn dual potentials of the last refresh of each frame, reset at each level
boundary. A trial evaluation (`refresh=False`) reuses them; `refresh=True` recomputes them. `frames` (a
device index of whole clips' frames, `ClipLayout.frames_of`) restricts both to those frames: a trial only
evaluates the clips that are still active and a refresh only the clips whose trial was accepted, and the
cached values of every other frame stay untouched, so a clip that stopped (or whose trial was rejected)
costs no Sinkhorn solve, no visibility pass and no residual evaluation. Terms come back for the evaluated
frames only, in `frames` order.

Memory (`fastdo.solver.memory` models it): a dense coarse/mid-level plan is (T,N,M) float64 (8 bytes per
entry), and a Sinkhorn refresh briefly holds 16 bytes per refreshed entry on top (24 for an eager float64
solve): the in-place squared float64 distances, the Sinkhorn-dtype cost, and the plan before and after its
cast. A refresh of every frame frees the cached plan first. A refresh or a subset trial that would exceed
`refresh_elements` (default `_REFRESH_ELEMENTS`) frame x point x sample entries runs over consecutive chunks of
its frames, writing each refreshed chunk into the cached plan in place, so a partial refresh never holds a
second full plan and the transient part is bounded by the chunk. Frames are independent in this step, so
chunking changes no result.

Plan moments (`SolverConfig.plan_moments`): every refresh reduces each chunk's plan (dense, kNN or
nearest-neighbour) to its canonical moments (`fastdo.solver.moments.PlanMoments`: barycenter B (T,N,3) and row mass w
(T,N) in canonical template coordinates) inside `terms.pointmap_term`, and the cache holds those in place of the plan,
written into the cached moments chunk by chunk exactly where the plan was. The dense (T,N,M) or kNN (T,N,k) plan then
never outlives its chunk, and a trial evaluates pbar = s R B + t in O(T N). The point-to-plane residual is not wired
into this callback, so the cached moments never carry normals.

Anisotropic scale (`SolverConfig.anisotropic_scale`): `log_scales` is
(B,3), each clip's log-stretch l in its template's principal frame, and every call builds the frames' `aniso.Stretch`.
Wherever the historical path transforms the template by (log_scale, pose), the stretched template A (y - c) with its
normals normalize(A^-1 n) enters at unit scale instead: the visibility z-buffer, the hand-occluder test, the
foreshortening weights, the nearest-neighbour plan and the Sinkhorn cost. The pointmap and track terms take the stretch
itself (their moments stay canonical, their Jacobians gain the three l columns), and a third term, the isotropy prior
(`aniso.prior_residuals`, weight `DeviceBatch.aniso_w`), joins them so LM's normal equations, merit and acceptance all
include it. The one scalar use, the z-buffer slack margin * s * diameter, takes the geometric-mean scale exp(mean l).

Depth anchor (`SolverConfig.lam_depth_anchor` > 0): a last term, the dead-zone absolute depth anchor
(`terms.depth_anchor_term`, one scalar residual per frame), joins the list after every other term. With the field at 0
(the default) it is never built. Inside its band it contributes exact zeros, and because `banded.assemble_from_blocks`
and `lm.clip_costs` sum terms in list order, adding zeros LAST leaves every sum of the other terms bit-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor

from fastdo.solver.aniso import Stretch, prior_residuals
from fastdo.solver.banded import PriorsLike
from fastdo.solver.batch import DeviceBatch, batch_priors
from fastdo.solver.lm import BatchEvaluate
from fastdo.solver.moments import PlanMoments, plan_moments, take_moments
from fastdo.solver.ot_sparse import SparsePlan
from fastdo.solver.terms import (
    TermOutput,
    depth_anchor_term,
    pointmap_term,
    track_term,
    transform_points,
)
from fastdo.solver.track_validity import frame_validity
from fastdo.solver.visibility import exclude_hand_occluded, foreshortening_weights, visible_samples

if TYPE_CHECKING:
    from fastdo.solver.api import SolverConfig

__all__ = ["EvalCache", "FrameInputs", "Level", "make_evaluate", "nn_plan"]

_REFRESH_ELEMENTS = 1 << 28   # frames x observed points x template samples per Sinkhorn chunk (~2 GB float64)


@dataclass(frozen=True)
class Level:
    """One coarse-to-fine annealing level: eps/rho in diameter^2 units, warm Sinkhorn iterations, the
    visibility slack/margin, foreshortening-weighted template mass, and the kNN truncation (fine only)."""

    eps: float
    rho_x: float
    rho_y: float
    n_warm: int
    vis_delta: float
    vis_margin: float
    vis_weighted: bool
    knn: int | None = None


class EvalCache:
    """Mutable per-solve cache shared by a level's evaluate callbacks (reset at each level boundary)."""

    def __init__(self) -> None:
        # The last refresh's correspondences: the dense or kNN plan, or its canonical moments when
        # `SolverConfig.plan_moments` is on.
        self.plan: Tensor | SparsePlan | PlanMoments | None = None
        self.potentials: tuple[Tensor, Tensor] | None = None
        self.mask: Tensor | None = None

    def reset(self) -> None:
        self.plan, self.potentials, self.mask = None, None, None


@dataclass(frozen=True)
class FrameInputs:
    """Per-frame (T, ...) solver inputs of a `DeviceBatch` for one annealing level: observations, template, intrinsics,
    hand masks, tracks, and the level's per-frame eps/rho and track weight (`track_w` (T,1)). The anisotropic scale adds
    each frame's clip principal frame (`axes` (T,3,3), `centroid` (T,3)) and isotropy-prior weight (`aniso_w` (T,)),
    None with the flag off. The depth anchor adds its `depth_centroid` (T,3), `depth_log_obs` (T,) and `depth_valid`
    (T,), None unless `SolverConfig.lam_depth_anchor` > 0."""

    X: Tensor
    X_valid: Tensor
    Y: Tensor
    Yn: Tensor
    Y_valid: Tensor
    vis_diameter: Tensor
    K: Tensor
    frame_hw: Tensor
    hand: Tensor | None
    anchors: Tensor
    tracks: Tensor
    vis: Tensor
    eps: Tensor
    rho_x: Tensor
    rho_y: Tensor
    track_w: Tensor | None
    axes: Tensor | None = None
    centroid: Tensor | None = None
    aniso_w: Tensor | None = None
    depth_centroid: Tensor | None = None
    depth_log_obs: Tensor | None = None
    depth_valid: Tensor | None = None

    def take(self, rows: Tensor | slice) -> FrameInputs:
        """The inputs of frames `rows` (a slice view, or an `index_select` of a device index)."""
        def pick(x: Tensor) -> Tensor:
            return x[rows] if isinstance(rows, slice) else x.index_select(0, rows)
        return FrameInputs(
            X=pick(self.X), X_valid=pick(self.X_valid), Y=pick(self.Y), Yn=pick(self.Yn), Y_valid=pick(self.Y_valid),
            vis_diameter=pick(self.vis_diameter), K=pick(self.K), frame_hw=pick(self.frame_hw),
            hand=None if self.hand is None else pick(self.hand), anchors=pick(self.anchors), tracks=pick(self.tracks),
            vis=pick(self.vis), eps=pick(self.eps), rho_x=pick(self.rho_x), rho_y=pick(self.rho_y),
            track_w=None if self.track_w is None else pick(self.track_w),
            axes=None if self.axes is None else pick(self.axes),
            centroid=None if self.centroid is None else pick(self.centroid),
            aniso_w=None if self.aniso_w is None else pick(self.aniso_w),
            depth_centroid=None if self.depth_centroid is None else pick(self.depth_centroid),
            depth_log_obs=None if self.depth_log_obs is None else pick(self.depth_log_obs),
            depth_valid=None if self.depth_valid is None else pick(self.depth_valid))

    def stretch(self, log_stretch: Tensor) -> Stretch:
        """These frames' `aniso.Stretch` at per-frame log-stretches `log_stretch` (T,3)."""
        if self.axes is None or self.centroid is None:
            raise ValueError("a log-stretch state needs SolverConfig.anisotropic_scale's principal frames")
        return Stretch(axes=self.axes, centroid=self.centroid, log_stretch=log_stretch)


def nn_plan(X: Tensor, X_valid: Tensor, Yk: Tensor, Y_valid: Tensor) -> Tensor:
    """One-hot nearest-neighbour plan with the same mass convention as the OT plan (a_i = 1/N_valid);
    padding template columns (`Y_valid` False) are never chosen. `X` is zero-padded, never NaN-padded."""
    T, N, _ = X.shape
    d = torch.where(Y_valid[:, None, :], torch.cdist(X, Yk), torch.full((), float("inf"), dtype=X.dtype,
                                                                         device=X.device))
    j = d.argmin(dim=2)
    plan = torch.zeros(T, N, Yk.shape[1], dtype=X.dtype, device=X.device)
    mass = X_valid.to(X.dtype) / X_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    plan.scatter_(2, j.unsqueeze(-1), mass.unsqueeze(-1))
    return plan


def _take_plan(plan: Tensor | SparsePlan | PlanMoments, frames: Tensor) -> Tensor | SparsePlan | PlanMoments:
    if isinstance(plan, PlanMoments):
        return take_moments(plan, frames)
    if isinstance(plan, SparsePlan):
        return SparsePlan(indices=plan.indices.index_select(0, frames), values=plan.values.index_select(0, frames))
    return plan.index_select(0, frames)


def _write(buf: Tensor | None, rows: Tensor | slice, part: Tensor, n_frames: int) -> Tensor:
    """Write `part` into rows `rows` (a slice or a device index) of the (T, ...) buffer `buf`, in place
    (allocated when None)."""
    if buf is None:
        buf = part.new_empty((n_frames, *part.shape[1:]))
    if isinstance(rows, slice):
        buf[rows].copy_(part)
    else:
        buf.index_copy_(0, rows, part)
    return buf


def _write_moments(buf: PlanMoments | None, rows: Tensor | slice, part: PlanMoments, n_frames: int) -> PlanMoments:
    """`_write` for every field of the (T, ...) moments `buf`, in place (allocated when None)."""
    normal = None
    if part.normal is not None:
        normal = _write(None if buf is None else buf.normal, rows, part.normal, n_frames)
    return PlanMoments(bary=_write(None if buf is None else buf.bary, rows, part.bary, n_frames),
                       mass=_write(None if buf is None else buf.mass, rows, part.mass, n_frames), normal=normal)


# "irls": LM accepts on Σ w·‖r‖² with w re-evaluated at the trial point -- for the Cauchy track term
# that is the Geman-McClure loss. "cauchy": it accepts on the Cauchy loss the IRLS step actually descends.
# "l2" (ablation only, opt-in): the track term is NOT robustified -- its IRLS weight is the visibility alone (the
# Cauchy weight 1/(1 + r²/c²) at c = inf is exactly 1) and LM accepts on Σ ω·v·‖r‖², the least-squares loss that step
# descends. Every other term is unchanged.
ROBUST_MERITS = ("irls", "cauchy", "l2")


def _cat_terms(parts: list[TermOutput]) -> TermOutput:
    if len(parts) == 1:
        return parts[0]
    extras = [p.extra for p in parts]
    merits = [p.merit for p in parts]
    return TermOutput(residual=torch.cat([p.residual for p in parts]), jacobian=torch.cat([p.jacobian for p in parts]),
                      weight=torch.cat([p.weight for p in parts]),
                      # (T,) per frame, so a chunked Sinkhorn refresh concatenates it along the frame
                      # axis exactly like the residual rows.
                      extra=(None if any(e is None for e in extras)
                             else torch.cat([e for e in extras if e is not None])),
                      merit=(None if any(m is None for m in merits)
                             else torch.cat([m for m in merits if m is not None])))


def _track_weight(db: DeviceBatch, cfg: SolverConfig) -> Tensor | None:
    """Per-frame `lam_track / scale2`, scaled by per-frame track validity when `cfg.track_validity`.

    The validity factor is 1.0 everywhere for exact tracks, so turning the flag on changes nothing
    on a clip whose tracks are clean; it decays only where the rank-4 fit says the tracks have
    stopped being consistent with any rigid motion (`track_validity`). Built once per level, outside any LM loop.
    """
    if cfg.lam_track <= 0:
        return None
    w = db.per_frame([cfg.lam_track / s2 for s2 in db.scale2])
    if cfg.track_validity:
        tracks = db.tracks.detach().cpu().numpy()
        vis = db.vis.detach().cpu().numpy()
        valid = np.ones(tracks.shape[0])
        for sl in db.layout.frame_slices():
            valid[sl] = frame_validity(tracks[sl], vis[sl])
        w = w * torch.as_tensor(valid, dtype=w.dtype, device=w.device)
    return w[:, None]


def make_evaluate(db: DeviceBatch, cfg: SolverConfig, level: Level, cache: EvalCache,
                  refresh_elements: int | None = None) -> BatchEvaluate:
    """The level's batched evaluate callback. `refresh_elements` bounds the frame x point x sample entries of
    one Sinkhorn chunk (None: `_REFRESH_ELEMENTS`, read at call time); `memory.plan_batches` picks it from the
    process's CUDA budget."""
    if cfg.robust_merit not in ROBUST_MERITS:
        raise ValueError(f"robust_merit must be one of {ROBUST_MERITS}, got {cfg.robust_merit!r}")
    cauchy_merit = cfg.robust_merit == "cauchy"      # accept on the loss the IRLS step descends
    track_c = math.inf if cfg.robust_merit == "l2" else cfg.cauchy_c   # no robust down-weighting under "l2"
    ci = db.layout.clip_index
    T = db.layout.n_frames
    full = FrameInputs(
        X=db.X, X_valid=db.X_valid, Y=db.Y, Yn=db.Yn, Y_valid=db.Y_valid, vis_diameter=db.vis_diameter, K=db.K,
        frame_hw=db.frame_hw, hand=db.hand, anchors=db.anchors, tracks=db.tracks, vis=db.vis,
        eps=db.per_frame([level.eps * s2 for s2 in db.scale2]),
        rho_x=db.per_frame([level.rho_x * s2 for s2 in db.scale2]),
        rho_y=db.per_frame([level.rho_y * s2 for s2 in db.scale2]),
        track_w=_track_weight(db, cfg), axes=db.shape_axes, centroid=db.shape_centroid, aniso_w=db.aniso_w,
        depth_centroid=db.depth_centroid, depth_log_obs=db.depth_log_obs, depth_valid=db.depth_valid)
    if cfg.lam_depth_anchor > 0 and (db.depth_centroid is None or db.depth_log_obs is None or db.depth_valid is None):
        raise ValueError("lam_depth_anchor > 0 needs a DeviceBatch prepared with it (prepare_batch under this config)")
    sinkhorn_dtype = torch.float32 if cfg.sinkhorn_float32 else None   # Sinkhorn-only cast
    memo: list[tuple[Tensor, FrameInputs]] = []                        # the last gathered subset

    def inputs_for(frames: Tensor | None) -> FrameInputs:
        if frames is None:
            return full
        if not memo or memo[0][0] is not frames:
            memo[:] = [(frames, full.take(frames))]
        return memo[0][1]

    def template_mask(v: FrameInputs, p: Tensor, ls: Tensor, st: Stretch | None = None) -> Tensor:
        Y, Yn, diameter = v.Y, v.Yn, v.vis_diameter
        if st is not None:                            # the stretched template at unit scale (`ls` is 0)
            Y, Yn, diameter = st.apply(v.Y), st.apply_normals(v.Yn), v.vis_diameter * st.mean_scale()
        geo = visible_samples(Y, Yn, p, ls, v.K, db.image_hw, cfg.vis_downsample, level.vis_margin,
                              level.vis_delta, frame_hw=v.frame_hw, sample_valid=v.Y_valid, diameter=diameter,
                              sparse=cfg.sparse_zbuffer)
        # Hand-occluder exclusion at every level: a real occluder at any pose.
        mask = exclude_hand_occluded(geo, Y, p, ls, v.K, v.hand) if v.hand is not None else geo
        if level.vis_weighted:                        # mid/fine: foreshortening-weighted template mass
            fs = foreshortening_weights(Y, Yn, p, ls)
            return fs * mask.to(fs.dtype)
        return mask.to(p.dtype)                       # coarse: uniform mass, capture range first

    def pointmap(v: FrameInputs, p: Tensor, ls: Tensor, plan: Tensor | SparsePlan | PlanMoments,
                 st: Stretch | None = None) -> TermOutput:
        pm, _plan = pointmap_term(v.X, v.X_valid, v.Y, p, ls, v.eps, v.rho_x, v.rho_y, cfg.n_sinkhorn, plan=plan,
                               with_deficiency=cfg.unbalanced_merit, stretch=st)
        return pm

    def chunk_frames(v: FrameInputs) -> int:
        """Frames per chunk: at most `refresh_elements` frame x point x sample entries, at least one frame."""
        limit = _REFRESH_ELEMENTS if refresh_elements is None else refresh_elements
        return max(1, limit // (v.X.shape[1] * v.Y.shape[1]))

    def cached(v: FrameInputs, p: Tensor, ls: Tensor, frames: Tensor | None, st: Stretch | None = None
               ) -> TermOutput:
        """Data term from the cached plan. A subset gathers its rows of a dense plan chunk by chunk, so the
        gathered copy never exceeds one chunk."""
        plan = cast("Tensor | SparsePlan | PlanMoments", cache.plan)
        if frames is None:
            return pointmap(v, p, ls, plan, st)
        n, per = frames.shape[0], chunk_frames(v)
        if isinstance(plan, (SparsePlan, PlanMoments)) or per >= n:
            return pointmap(v, p, ls, _take_plan(plan, frames), st)
        return _cat_terms([pointmap(v.take(rows), p[rows], ls[rows], _take_plan(plan, frames[rows]),
                                    None if st is None else st.take(rows))
                           for rows in (slice(s, min(s + per, n)) for s in range(0, n, per))])

    def refreshed(v: FrameInputs, p: Tensor, ls: Tensor, frames: Tensor | None, st: Stretch | None = None
                  ) -> TermOutput:
        """Recompute mask/plan/potentials of `frames` (None: every frame) and store them in the cache."""
        if frames is not None and (cache.plan is None or cache.potentials is None and cfg.correspondence != "nn"):
            raise ValueError("a refresh of a subset of frames needs a previous refresh of every frame")
        if frames is None:
            cache.plan = None                         # a refresh of every frame never needs the old plan
        mask = template_mask(v, p, ls, st) if cfg.visibility else None
        if mask is not None:
            cache.mask = mask if frames is None or cache.mask is None else cache.mask.index_copy(0, frames, mask)
        if cfg.correspondence == "nn":
            Yk = transform_points(v.Y if st is None else st.apply(v.Y), p, ls.exp())
            nn = nn_plan(v.X, v.X_valid, Yk, v.Y_valid)
            del Yk
            if cfg.plan_moments:                      # the one-hot plan's moments, like an OT plan's
                mom = plan_moments(nn, v.Y)
                del nn
                cache.plan = mom if frames is None else _write_moments(cast("PlanMoments", cache.plan), frames, mom, T)
                return pointmap(v, p, ls, mom, st)
            cache.plan = nn if frames is None else _write(cast("Tensor", cache.plan), frames, nn, T)
            return pointmap(v, p, ls, nn, st)
        n = v.X.shape[0]
        per = chunk_frames(v)
        old_pots = cache.potentials
        n_iter = cfg.n_sinkhorn if old_pots is None else level.n_warm
        if frames is None and per >= n:               # one Sinkhorn solve over every frame (e.g. one clip)
            pm, plan, pots = pointmap_term(v.X, v.X_valid, v.Y, p, ls, v.eps, v.rho_x, v.rho_y, n_iter, plan=None,
                                           mesh_mask=mask, potentials=old_pots, sinkhorn_dtype=sinkhorn_dtype,
                                           knn=level.knn, mesh_valid=v.Y_valid, fused_sinkhorn=cfg.fused_sinkhorn,
                with_deficiency=cfg.unbalanced_merit, moments=cfg.plan_moments, stretch=st)
            cache.plan, cache.potentials = plan, cast("tuple[Tensor, Tensor]", pots)
            return pm
        old_plan = cache.plan                         # None for a refresh of every frame (released above)
        cache.plan = None
        dense = old_plan if isinstance(old_plan, Tensor) else None
        mom_buf = old_plan if isinstance(old_plan, PlanMoments) else None
        idx_buf = old_plan.indices if isinstance(old_plan, SparsePlan) else None
        val_buf = old_plan.values if isinstance(old_plan, SparsePlan) else None
        f_buf, g_buf = old_pots if frames is not None and old_pots is not None else (None, None)
        parts: list[TermOutput] = []
        for start in range(0, n, per):
            rows = slice(start, min(start + per, n))
            target: Tensor | slice = rows if frames is None else frames[rows]   # rows of the whole batch
            vc = v if per >= n else v.take(rows)
            pots_in = None
            if old_pots is not None:
                pots_in = ((old_pots[0][target], old_pots[1][target]) if isinstance(target, slice) else
                           (old_pots[0].index_select(0, target), old_pots[1].index_select(0, target)))
            pm, part, part_pots = pointmap_term(
                vc.X, vc.X_valid, vc.Y, p[rows], ls[rows], vc.eps, vc.rho_x, vc.rho_y, n_iter, plan=None,
                mesh_mask=None if mask is None else mask[rows], potentials=pots_in, sinkhorn_dtype=sinkhorn_dtype,
                knn=level.knn, mesh_valid=vc.Y_valid, fused_sinkhorn=cfg.fused_sinkhorn,
                with_deficiency=cfg.unbalanced_merit, moments=cfg.plan_moments,
                stretch=None if st is None else st.take(rows))
            parts.append(pm)
            if isinstance(part, PlanMoments):
                mom_buf = _write_moments(mom_buf, target, part, T)
            elif isinstance(part, SparsePlan):
                idx_buf = _write(idx_buf, target, part.indices, T)
                val_buf = _write(val_buf, target, part.values, T)
            else:
                dense = _write(dense, target, part, T)
            f_part, g_part = cast("tuple[Tensor, Tensor]", part_pots)
            f_buf, g_buf = _write(f_buf, target, f_part, T), _write(g_buf, target, g_part, T)
            del part, part_pots, f_part, g_part       # the next chunk must not also hold this chunk's plan
        new_plan: Tensor | SparsePlan | PlanMoments | None = (
            mom_buf if mom_buf is not None
            else dense if idx_buf is None or val_buf is None else SparsePlan(indices=idx_buf, values=val_buf))
        if new_plan is None or f_buf is None or g_buf is None:
            raise RuntimeError(f"a chunked Sinkhorn refresh of {n} frames produced no plan or potentials")
        cache.plan, cache.potentials = new_plan, (f_buf, g_buf)
        return _cat_terms(parts)

    def evaluate(p: Tensor, log_scales: Tensor, refresh: bool, frames: Tensor | None = None
                 ) -> tuple[list[TermOutput], PriorsLike]:
        v = inputs_for(frames)
        ls_all = log_scales[ci]
        p_s = p if frames is None else p.index_select(0, frames)
        ls = ls_all if frames is None else ls_all.index_select(0, frames)
        st: Stretch | None = None
        if ls.dim() == 2:                             # (T,3) log-stretches; the stretch carries the scale
            st, ls = v.stretch(ls), ls.new_zeros(ls.shape[0])
        if not refresh and cache.plan is not None:
            pm = cached(v, p_s, ls, frames, st)
        else:
            pm = refreshed(v, p_s, ls, frames, st)
        terms: list[TermOutput] = [pm]
        if v.track_w is not None:
            tr = track_term(v.anchors, v.tracks, v.vis, p_s, ls, v.K, track_c, cauchy_merit=cauchy_merit,
                            stretch=st)
            # The merit is scaled by the SAME per-frame track weight as the IRLS weight, so on a clip whose residuals
            # are all small the two merits agree term-for-term.
            terms.append(TermOutput(tr.residual, tr.jacobian, tr.weight * v.track_w,
                                    merit=None if tr.merit is None else tr.merit * v.track_w[:, 0]))
        if st is not None:                            # the isotropy prior, lam_b / T_b on each of clip b's frames
            r, J = prior_residuals(st.log_stretch)
            terms.append(TermOutput(r, J, cast("Tensor", v.aniso_w)[:, None]))
        if cfg.lam_depth_anchor > 0:                  # anchor LAST, so in-band zeros leave every other sum unchanged
            terms.append(depth_anchor_term(cast("Tensor", v.depth_centroid), cast("Tensor", v.depth_log_obs),
                                           cast("Tensor", v.depth_valid), p_s, ls, cfg.lam_depth_anchor,
                                           cfg.depth_anchor_band, stretch=st))
        return terms, batch_priors(p, db.root_smooth, db.root_vel, db.layout)

    return evaluate
