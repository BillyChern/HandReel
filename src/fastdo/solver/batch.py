"""Batched multi-clip solve: B clips packed into one decoupled system.

`pack_clips` concatenates the frames of B clips in order (`clip_index`, per-clip frame slices) and pads
everything whose size differs between clips: template samples (with a validity mask), point tracks and
anchors (with `track_vis` False) and hand masks (False). A frame's pointmap enters the solver only through
its masked, finite pixels, so those are packed directly (`obs_points`, `obs_offsets`: row-major order,
the order the per-frame sampling draws from) instead of copying full padded images. `prepare_batch` applies a
`SolverConfig` exactly as the single-clip solver did per clip (same generator draws for the template and
observation subsampling, same per-clip diameter^2 normalization and data-support prior scaling) and moves
the result to the device, gathered per frame. `batch_eval.make_evaluate` is the batched LM callback: one
visibility pass, one Sinkhorn solve (per-frame eps/rho) and one track/prior evaluation over all T frames
of all clips, so the number of kernel launches per LM iteration does not grow with B.

Clip independence is exact, not approximate: data terms only touch their own frame, the constant-pose
and constant-velocity residuals that would straddle two clips carry weight 0 (`batch_priors`), and each
clip keeps its own scale unknown (see `banded.py`). A refresh evaluation may name a subset of clips; the
others keep their cached visibility mask, plan and Sinkhorn potentials, bit-identical to what a separate
solve of that clip would hold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from fastdo.solver.aniso import ANISO_DOF, principal_frame
from fastdo.solver.banded import ClipLayout, Priors
from fastdo.solver.terms import relative_motion_term, smoothness_term
from fastdo.solver.visibility import dilate_hand_masks

if TYPE_CHECKING:
    from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations

__all__ = ["DEPTH_MIN_POINTS", "MIN_VALID_OBS", "DeviceBatch", "ObjectPixels", "PackedBatch", "batch_priors",
           "frame_depths", "object_pixels_of", "pack_clips", "prepare_batch", "sample_points", "valid_pixels",
           "valid_pixels_of"]

MIN_VALID_OBS = 20   # a frame with fewer valid points is treated as unobserved (see api.py's module docstring)
DEPTH_MIN_POINTS = 10   # `frame_depths` is NaN below this many in-front points


@dataclass(frozen=True)
class ObjectPixels:
    """Every frame's pointmap under its object mask, NaNs kept: frame k's are `points[offsets[k]:offsets[k+1]]`
    in row-major order, the order `pointmaps[k][masks[k]]` yields. `points` (P0,3) float32, `offsets` (T+1,) int64.
    The masked `geometry.npz` stores exactly this, so the solve stage reads it without rebuilding dense pointmaps."""

    points: Tensor
    offsets: Tensor

    def __post_init__(self) -> None:
        if self.points.dim() != 2 or self.points.shape[1] != 3 or self.offsets.dim() != 1 or self.offsets.numel() < 1:
            raise ValueError(f"ObjectPixels needs points (P,3) and offsets (T+1,), got {tuple(self.points.shape)} "
                             f"and {tuple(self.offsets.shape)}")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != self.points.shape[0]:
            raise ValueError(f"offsets must run from 0 to {self.points.shape[0]} points")

    @property
    def n_frames(self) -> int:
        return self.offsets.numel() - 1

    def frame(self, k: int) -> Tensor:
        """Frame k's object pixels, `pointmaps[k][masks[k]]`."""
        return self.points[int(self.offsets[k]):int(self.offsets[k + 1])]


def object_pixels_of(masks: Tensor, pointmaps: Tensor) -> ObjectPixels:
    """`ObjectPixels` of dense `pointmaps` (T,H,W,3) under `masks` (T,H,W)."""
    T = masks.shape[0]
    flat = masks.reshape(T, -1)
    counts = flat.sum(1)
    return ObjectPixels(pointmaps.reshape(T, -1, 3)[flat], torch.cat([counts.new_zeros(1), counts.cumsum(0)]))


@dataclass(frozen=True)
class PackedBatch:
    """B clips, frames concatenated in clip order (T = sum of clip lengths), sizes padded to the batch max.

    Per clip: `mesh_points`/`mesh_normals` (B,M,3) with `mesh_valid` (B,M), `anchors` (B,J,3),
    `intrinsics` (B,3,3), `image_hw` (B,2) each clip's own (H,W), `init_scales`. Per frame: `obs_points`
    (P,3) every masked finite pointmap pixel, frame k's at `obs_points[obs_offsets[k]:obs_offsets[k+1]]` in
    row-major order; `hand_masks` (T,H,W) False-padded; `tracks` (T,J,2) and `track_vis` (T,J)
    False-padded; `init_poses` (T,4,4); `clip_index` (T,). `frame_slices[b]` selects clip b's frames.
    """

    mesh_points: Tensor
    mesh_normals: Tensor
    mesh_valid: Tensor
    anchors: Tensor
    intrinsics: Tensor
    image_hw: Tensor
    init_scales: tuple[float, ...]
    obs_points: Tensor
    obs_offsets: Tensor
    hand_masks: Tensor
    tracks: Tensor
    track_vis: Tensor
    init_poses: Tensor
    clip_index: Tensor
    frame_slices: tuple[slice, ...]

    @property
    def n_clips(self) -> int:
        return len(self.frame_slices)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(sl.stop - sl.start for sl in self.frame_slices)


def _pad_stack(items: Sequence[Tensor], fill: float | bool, cat: bool) -> Tensor:
    """Concatenate (`cat`, along dim 0, padding the later dims) or stack (padding every dim) `items` with
    `fill` up to the batch maximum; a single item is returned without a copy."""
    if len(items) == 1:
        return items[0] if cat else items[0].unsqueeze(0)
    first = 1 if cat else 0
    shape = [max(int(x.shape[d]) for x in items) for d in range(first, items[0].dim())]
    padded = []
    for x in items:
        if list(x.shape[first:]) != shape:
            out = torch.full((*x.shape[:first], *shape), fill, dtype=x.dtype)
            out[(*(slice(None),) * first, *(slice(0, n) for n in x.shape[first:]))] = x
            x = out
        padded.append(x)
    return torch.cat(padded) if cat else torch.stack(padded)


def pack_clips(meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
               init_scales: list[float]) -> PackedBatch:
    """Pack B clips (CPU tensors, as the caller passed them) into one `PackedBatch`."""
    B = len(meshes)
    if not (B == len(obs) == len(init_poses) == len(init_scales)) or B == 0:
        raise ValueError(f"need equally many (>0) meshes/obs/init_poses/init_scales, got "
                         f"{len(meshes)}/{len(obs)}/{len(init_poses)}/{len(init_scales)}")
    lengths = [int(p.shape[0]) for p in init_poses]
    for b, (o, n) in enumerate(zip(obs, lengths, strict=True)):
        if (o.masks.shape[0] != n or o.pointmaps.shape[0] != n or o.tracks.shape[0] != n
                or (o.object_pixels is not None and o.object_pixels.n_frames != n)):
            raise ValueError(f"clip {b}: observations do not have {n} frames like its init_poses")
    starts = [sum(lengths[:b]) for b in range(B)]
    pixels = [valid_pixels_of(o) for o in obs]
    counts = torch.cat([c for _, c in pixels])
    return PackedBatch(
        mesh_points=_pad_stack([m.points for m in meshes], 0.0, cat=False),
        mesh_normals=_pad_stack([m.normals for m in meshes], 0.0, cat=False),
        mesh_valid=_pad_stack([torch.ones(m.points.shape[0], dtype=torch.bool) for m in meshes], False, cat=False),
        anchors=_pad_stack([o.anchors for o in obs], 0.0, cat=False),
        intrinsics=torch.stack([o.intrinsics for o in obs]),
        image_hw=torch.tensor([[o.masks.shape[1], o.masks.shape[2]] for o in obs], dtype=torch.long),
        init_scales=tuple(float(s) for s in init_scales),
        obs_points=_pad_stack([pts for pts, _ in pixels], 0.0, cat=True),
        obs_offsets=torch.cat([counts.new_zeros(1), counts.cumsum(0)]),
        hand_masks=_pad_stack([o.hand_masks for o in obs], False, cat=True),
        tracks=_pad_stack([o.tracks for o in obs], 0.0, cat=True),
        track_vis=_pad_stack([o.track_vis for o in obs], False, cat=True),
        init_poses=_pad_stack(init_poses, 0.0, cat=True),
        clip_index=torch.repeat_interleave(torch.arange(B), torch.tensor(lengths)),
        frame_slices=tuple(slice(s, s + n) for s, n in zip(starts, lengths, strict=True)))


def valid_pixels(masks: Tensor, pointmaps: Tensor) -> tuple[Tensor, Tensor]:
    """Masked pixels with a finite pointmap, all frames at once: `(points (P,3), counts (T,))`, frame by frame
    and row-major within a frame (the order `pointmaps[k][masks[k] & finite]` yields)."""
    T = masks.shape[0]
    idx = masks.reshape(T, -1).nonzero()                                 # (P0,2) [frame, pixel], sorted
    pts = pointmaps.reshape(T, -1, 3)[idx[:, 0], idx[:, 1]]
    keep = torch.isfinite(pts).all(-1)
    return pts[keep], torch.bincount(idx[keep, 0], minlength=T)


def valid_pixels_of(obs: TrajectoryObservations) -> tuple[Tensor, Tensor]:
    """`valid_pixels` of one clip's observations, from `obs.object_pixels` when set (the same points in the
    same order, without touching the dense pointmaps), else from `masks` and `pointmaps`."""
    px = obs.object_pixels
    if px is None:
        return valid_pixels(obs.masks, obs.pointmaps)
    keep = torch.isfinite(px.points).all(-1)
    frame = torch.repeat_interleave(torch.arange(px.n_frames), px.offsets.diff())
    return px.points[keep], torch.bincount(frame[keep], minlength=px.n_frames)


def frame_depths(points: Tensor, offsets: Tensor, min_points: int = DEPTH_MIN_POINTS) -> Tensor:
    """Each frame's observed object depth (T,) float64, NaN where unobserved: the median camera-frame z of the
    frame's masked finite pointmap points with z > 0 (frame k's at `points[offsets[k]:offsets[k+1]]`, as
    `valid_pixels`), NaN below `min_points` of them. For an even count the median is the mean of the two middle values
    (numpy's convention).

    This is the per-frame quantity the depth anchor's d_t was measured against: all masked pixels, not the solver's
    `n_obs` subsample,
    so the anchor's target does not depend on the sampling seed."""
    T = offsets.shape[0] - 1
    out = torch.full((T,), float("nan"), dtype=torch.float64)
    for k, (start, stop) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)):
        z = points[start:stop, 2].double()
        z = z[torch.isfinite(points[start:stop]).all(-1) & (z > 0)]
        n = z.shape[0]
        if n < min_points:
            continue
        zs = z.sort().values
        out[k] = (zs[(n - 1) // 2] + zs[n // 2]) / 2
    return out


def sample_points(points: Tensor, offsets: Tensor, n_obs: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    """Per frame, up to n_obs of its valid pixels (`valid_pixels`, frame k at `points[offsets[k]:offsets[k+1]]`)
    -> X (T,n_obs,3) float64, valid (T,n_obs), CPU.

    A frame with fewer than `MIN_VALID_OBS` points is left entirely invalid rather than handing the
    pointmap term a handful of points: a near-fully-occluded frame's sparse, noisy pointmap sample can
    pull the fit toward a spurious partial match instead of leaning on the (data-support-scaled)
    temporal priors and the track term.
    """
    T = offsets.shape[0] - 1
    X = torch.zeros(T, n_obs, 3, dtype=torch.float64)
    valid = torch.zeros(T, n_obs, dtype=torch.bool)
    for k, (start, stop) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)):
        if stop - start < MIN_VALID_OBS:
            continue
        idx = torch.randperm(stop - start, generator=gen)[:n_obs]
        X[k, : idx.shape[0]] = points[start + idx].double()
        valid[k, : idx.shape[0]] = True
    return X, valid


@dataclass(frozen=True)
class DeviceBatch:
    """A `PackedBatch` with a `SolverConfig` applied, on the solver device, gathered per frame (T,...).

    `X`/`X_valid` sampled observations; `Y`/`Yn`/`Y_valid` (T,M,3)/(T,M,3)/(T,M) the clip's subsampled
    template; `vis_diameter` (T,) its extent (visibility margin units); `K` (T,3,3); `frame_hw` (T,2)
    with `image_hw` the batch max; `hand` (T,H,W) dilated hand masks or None; `anchors`/`tracks`/`vis`
    (T,J,3)/(T,J,2)/(T,J). Per clip (host floats): `diameters` (full template extent), `scale2`
    ((init_scale * diameter)^2). `root_smooth` (T-1,) and `root_vel` (T-2,) are the square roots of the
    support-scaled prior weights, zero wherever the residual would straddle two clips.

    Anisotropic scale (`SolverConfig.anisotropic_scale`, else all None): `shape_axes` (T,3,3) / `shape_centroid` (T,3)
    each frame's clip principal frame (`aniso.principal_frame` of the clip's whole template), `aniso_w` (T,) the
    isotropy prior's weight lam_b / T_b per frame (a clip's frames sum to lam_b, see `prepare_batch`), and `log_scale0`
    is (B,3). `poses0` are then CENTROID-referenced (t + s0 R c; `aniso`'s module docstring).

    Depth anchor (`SolverConfig.lam_depth_anchor` > 0, else all None): `depth_centroid` (T,3) each frame's clip template
    centroid (the mean of ALL the clip's `MeshSamples.points`, the contact stage's `centroid_obj`, not the `n_mesh`
    subsample), `depth_log_obs` (T,) log `frame_depths` (0.0 where unobserved) and `depth_valid` (T,) where it is
    observed.
    """

    layout: ClipLayout
    X: Tensor
    X_valid: Tensor
    Y: Tensor
    Yn: Tensor
    Y_valid: Tensor
    vis_diameter: Tensor
    K: Tensor
    image_hw: tuple[int, int]
    frame_hw: Tensor
    hand: Tensor | None
    anchors: Tensor
    tracks: Tensor
    vis: Tensor
    poses0: Tensor
    log_scale0: Tensor
    init_scales: tuple[float, ...]
    diameters: tuple[float, ...]
    scale2: tuple[float, ...]
    root_smooth: Tensor | None
    root_vel: Tensor | None
    shape_axes: Tensor | None = None
    shape_centroid: Tensor | None = None
    aniso_w: Tensor | None = None
    depth_centroid: Tensor | None = None
    depth_log_obs: Tensor | None = None
    depth_valid: Tensor | None = None

    def per_frame(self, values: Sequence[float]) -> Tensor:
        """One float per clip -> (T,) float64 per-frame tensor (built outside any LM loop)."""
        dev = self.layout.clip_index.device
        return torch.tensor(list(values), dtype=torch.float64, device=dev)[self.layout.clip_index]


def _prior_roots(lengths: Sequence[int], weights: Sequence[float], span: int, dev: str) -> Tensor | None:
    """sqrt(weight_b) for every residual over frames (k, ..., k+span) inside clip b, 0 across clips."""
    roots: list[float] = []
    for n, w in zip(lengths, weights, strict=True):
        roots.extend([w ** 0.5] * max(n - span, 0) + [0.0] * min(span, n))
    T = sum(lengths)
    return torch.tensor(roots[: T - span], dtype=torch.float64, device=dev) if T > span else None


PRIOR_SUPPORTS = ("all_frames", "observed_frames")


def prior_support(valid: Tensor, mode: str = "all_frames") -> float:
    """The factor that scales both temporal priors: mean valid pointmap points per frame, at least 1.

    `all_frames` (the shipped default) divides by every frame of the clip. `observed_frames` divides only
    by frames that have at least one valid point. They agree whenever every frame is observed; they part
    ways when pointmaps are absent on some frames, where `all_frames` silently weakens the priors by the
    fraction of frames left unobserved -- qc20 dropped pointmaps to 12% of rows and the priors with them,
    confounding "less depth information" with "less regularisation".

    Raises:
        ValueError: On an unknown `mode`.
    """
    if mode not in PRIOR_SUPPORTS:
        raise ValueError(f"prior_support must be one of {PRIOR_SUPPORTS}, got {mode!r}")
    n = valid.shape[0] if mode == "all_frames" else int(valid.any(dim=-1).sum())
    return max(1.0, float(valid.sum()) / max(1, n))


def _aniso_fields(packed: PackedBatch, cfg: SolverConfig, frames: Sequence[tuple[Tensor, Tensor]],
                  diameters: Sequence[float], supports: Sequence[float], layout: ClipLayout, poses0: Tensor
                  ) -> dict[str, Tensor]:
    """The `DeviceBatch` fields of `SolverConfig.anisotropic_scale` (its docstring), and the init converted
    exactly to the stretch model: l0 = log s0 on every axis and t = t0 + s0 R0 c.

    The prior E = lam_b sum_i (l_i - mean l)^2 is charged in the pointmap term's currency like the temporal priors:
    lam_b = lam_aniso * support_b * (s0_b D_b)^2 (support_b the clip's mean valid points per frame, D_b its template
    diameter), so lam_aniso is dimensionless (`SolverConfig.lam_aniso` has the calibration of its default)."""
    dev, ci = poses0.device, layout.clip_index
    axes = torch.stack([U for U, _ in frames]).to(dev)                  # (B,3,3)
    centroid = torch.stack([c for _, c in frames]).to(dev)              # (B,3)
    s0 = torch.tensor(packed.init_scales, dtype=torch.float64, device=dev)
    lam = [cfg.lam_aniso * sup * (s * d) ** 2 / n
           for sup, s, d, n in zip(supports, packed.init_scales, diameters, packed.lengths, strict=True)]
    poses = poses0.clone()
    poses[:, :3, 3] = poses0[:, :3, 3] + s0[ci, None] * (poses0[:, :3, :3] @ centroid[ci].unsqueeze(-1)).squeeze(-1)
    return {"shape_axes": axes[ci], "shape_centroid": centroid[ci],
            "aniso_w": torch.tensor(lam, dtype=torch.float64, device=dev)[ci], "poses0": poses,
            "log_scale0": s0.log()[:, None].expand(-1, ANISO_DOF).contiguous()}


def _depth_anchor_fields(packed: PackedBatch, layout: ClipLayout, dev: str) -> dict[str, Tensor]:
    """The `DeviceBatch` fields of `SolverConfig.lam_depth_anchor` (its docstring). Host work once per solve."""
    centroid = torch.stack([packed.mesh_points[b, packed.mesh_valid[b]].double().mean(0) for b in range(packed.n_clips)])
    z = frame_depths(packed.obs_points, packed.obs_offsets)
    valid = torch.isfinite(z)
    log_z = torch.where(valid, z, torch.ones_like(z)).log()
    return {"depth_centroid": centroid.to(dev)[layout.clip_index], "depth_log_obs": log_z.to(dev),
            "depth_valid": valid.to(dev)}


def prepare_batch(packed: PackedBatch, cfg: SolverConfig) -> DeviceBatch:
    dev = cfg.device
    lengths = packed.lengths
    Ys, Yns, Xs, Vs, diameters, supports = [], [], [], [], [], []
    frames: list[tuple[Tensor, Tensor]] = []                            # principal frames, anisotropic scale only
    for b, sl in enumerate(packed.frame_slices):
        gen = torch.Generator().manual_seed(cfg.seed)                   # per clip, as a separate solve
        pts = packed.mesh_points[b, packed.mesh_valid[b]]
        nrm = packed.mesh_normals[b, packed.mesh_valid[b]]
        diameters.append(float((pts.max(0).values - pts.min(0).values).norm()))
        if cfg.anisotropic_scale:                                       # the whole template, not the subsample
            frames.append(principal_frame(pts))
        perm = torch.randperm(pts.shape[0], generator=gen)[: cfg.n_mesh]
        Ys.append(pts.double()[perm])
        # Normals subsampled with the SAME permutation (visibility needs them index-aligned), normalized.
        Yns.append(torch.nn.functional.normalize(nrm.double()[perm], dim=-1))
        X, V = sample_points(packed.obs_points, packed.obs_offsets[sl.start:sl.stop + 1], cfg.n_obs, gen)
        Xs.append(X)
        Vs.append(V)
        supports.append(prior_support(V, cfg.prior_support))
    layout = ClipLayout.build(lengths, dev)
    ci = layout.clip_index
    Y = _pad_stack(Ys, 0.0, cat=False).to(dev)
    Y_valid = _pad_stack([torch.ones(y.shape[0], dtype=torch.bool) for y in Ys], False, cat=False).to(dev)
    vis_diameter = torch.stack([(Y[b, Y_valid[b]].max(0).values - Y[b, Y_valid[b]].min(0).values).norm()
                                for b in range(len(lengths))])
    hand = dilate_hand_masks(packed.hand_masks.to(dev), cfg.vis_hand_dilate) if cfg.visibility else None
    lam_vel = [cfg.lam_vel * s for s in supports]
    use_vel = cfg.lam_vel > 0 and max(lengths) > 2
    poses0 = packed.init_poses.double().to(dev)
    log_scale0 = torch.tensor([math.log(s) for s in packed.init_scales], dtype=torch.float64, device=dev)
    aniso = _aniso_fields(packed, cfg, frames, diameters, supports, layout, poses0) if cfg.anisotropic_scale else {}
    depth = _depth_anchor_fields(packed, layout, dev) if cfg.lam_depth_anchor > 0 else {}
    return DeviceBatch(
        layout=layout, X=torch.cat(Xs).to(dev), X_valid=torch.cat(Vs).to(dev), Y=Y[ci],
        Yn=_pad_stack(Yns, 0.0, cat=False).to(dev)[ci], Y_valid=Y_valid[ci], vis_diameter=vis_diameter[ci],
        K=packed.intrinsics.double().to(dev)[ci], image_hw=cast("tuple[int, int]", tuple(packed.image_hw.amax(0).tolist())),
        frame_hw=packed.image_hw.to(dev)[ci], hand=hand, anchors=packed.anchors.double().to(dev)[ci],
        tracks=packed.tracks.double().to(dev), vis=packed.track_vis.to(dev), poses0=aniso.pop("poses0", poses0),
        log_scale0=aniso.pop("log_scale0", log_scale0),
        init_scales=packed.init_scales, diameters=tuple(diameters),
        scale2=tuple((s * d) ** 2 for s, d in zip(packed.init_scales, diameters, strict=True)),
        root_smooth=_prior_roots(lengths, [cfg.lam_smooth * s for s in supports], 1, dev),
        root_vel=_prior_roots(lengths, lam_vel, 2, dev) if use_vel else None, **aniso, **depth)


def _at_rows(n_rows: int, rows: Tensor, values: Tensor) -> Tensor:
    """(n_rows, ...) exact zeros holding `values` at `rows`."""
    return values.new_zeros((n_rows, *values.shape[1:])).index_copy_(0, rows, values)


def batch_priors(poses: Tensor, root_smooth: Tensor | None, root_vel: Tensor | None, layout: ClipLayout) -> Priors:
    """Constant-pose and constant-velocity priors over concatenated clips: `smoothness_term`'s unit-weight
    relative motions a_k = Log(T_k^-1 T_{k+1}) and their Jacobians, weighted per residual. The velocity
    quadruple is `velocity_term`'s (dt = 1) built from the same unit-weight series.

    A residual whose frames straddle two clips is never evaluated: only the rows in
    `layout.pair_rows` / `layout.triple_rows` are computed and every other row is an exact zero, not a zero
    weight times a computed value. So a non-finite pose in one clip (a NaN trial step) cannot reach another
    clip's cost or normal equations through 0 * NaN. One clip has no such rows and is computed directly."""
    if root_smooth is None:
        return Priors()
    T = poses.shape[0]
    pairs, triples = layout.pair_rows, layout.triple_rows
    whole = layout.n_clips == 1
    if whole:
        a, j_own, j_next = smoothness_term(poses, 1.0)
    elif pairs.shape[0] == 0:                                  # every clip has one frame: no pair inside a clip
        a, j_own, j_next = poses.new_zeros(T - 1, 6), poses.new_zeros(T - 1, 6, 6), poses.new_zeros(T - 1, 6, 6)
    else:
        inside = relative_motion_term(poses.index_select(0, pairs), poses.index_select(0, pairs + 1), 1.0)
        a, j_own, j_next = (_at_rows(T - 1, pairs, x) for x in inside)
    smooth = (root_smooth[:, None] * a, root_smooth[:, None, None] * j_own, root_smooth[:, None, None] * j_next)
    if root_vel is None:
        return Priors(smooth=smooth)
    s1, s2 = root_vel[:, None], root_vel[:, None, None]
    if whole:
        velocity = (s1 * (a[1:] - a[:-1]), -s2 * j_own[:-1], s2 * (j_own[1:] - j_next[:-1]), s2 * j_next[1:])
        return Priors(smooth=smooth, velocity=velocity)
    later = triples + 1                                        # pairs k and k+1 of a triple lie in the same clip
    a0, a1 = a.index_select(0, triples), a.index_select(0, later)
    own0, own1, next0 = j_own.index_select(0, triples), j_own.index_select(0, later), j_next.index_select(0, triples)
    rows = (a1 - a0, -own0, own1 - next0, j_next.index_select(0, later))
    r, j_prev, j_mid, j_last = (_at_rows(T - 2, triples, x) for x in rows)
    return Priors(smooth=smooth, velocity=(s1 * r, s2 * j_prev, s2 * j_mid, s2 * j_last))
