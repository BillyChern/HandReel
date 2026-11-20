"""Public API: observations, config, solution, and solve_trajectory (coarse-to-fine LM).

Defaults below that differ from the original design, and why:

* `rho_x`/`rho_y`/`eps_fine` = 0.1/1e-3/1e-4, not the original 10.0/0.05/0.005: those values are a
  near-hard marginal constraint once compared to `eps` in the same (diameter^2) units, producing a
  ~1cm/1.5-2% translation/scale bias.
* rho anneals over 3 levels (coarse/mid/fine, `rho_x_coarse`.../`eps_mid`) because a single fixed rho
  has too little pointmap capture range for a 5cm/15deg perturbed init.
* `lam_track`'s balance constant is 1.0, not the original 1e-4: needed to break near-rotationally-
  symmetric shapes' unobservable axis.
* `lam_smooth` and `lam_vel` are rescaled inside `solve_trajectory` by the clip's mean
  valid-points/frame so they can bridge fully-occluded frames; `_MIN_VALID_OBS`=20 drops a
  frame's pointmap term entirely below that support instead of handing it a spurious partial match.
  The constant-velocity prior `lam_vel`=1.0 does the bridging (on smooth motion it interpolates instead
  of cutting the sagitta; a random walk that decorrelates inside the gap gains little) and the
  constant-pose `lam_smooth` drops from 1.0 to 0.1, kept as a damping prior for unobserved tails.
* confidence is 0.0 below `_MIN_VALID_OBS` pointmap support, not the vacuous `exp(-0)=1.0`. A
  support-invariant statistic normalized by the clip's noise floor (`fastdo.solver.confidence`) replaced
  the earlier `exp(-rms / (0.05 D))`, which drifted with the template measure.
* `n_sinkhorn`/`n_sinkhorn_warm(_fine)` = 50/10/10 ("cold 50 / warm 10"),
  `sinkhorn_float32=True`, and `lm_optimize`'s arrow-banded solver (`lm.py`) recover the runtime
  budget after the above; the BLAS thread cap this needed lives in `lm.py`, scoped via
  `threadpoolctl`.
* `fastdo.solver.synthetic`'s frustum-bounce guard keeps synthetic clips observable
  end-to-end; it is not part of this module.

`solve_trajectories` solves B clips as one decoupled system (`fastdo.solver.batch` packs and
evaluates them, `banded.ClipLayout` keeps them independent, `lm.lm_optimize_batch` gives each clip its own
damping and acceptance); every per-clip quantity above (sampling, diameter normalization, prior support
scaling, confidence noise floor) is computed per clip exactly as a separate solve would. `solve_trajectory`
is its B = 1 case, so there is one solve path.

Anisotropic scale (`SolverConfig.anisotropic_scale`, default off): the per-clip scale becomes a body-frame stretch in
the template's principal frame (`fastdo.solver.aniso`); `TrajectorySolution` then also carries the shape fields.

Depth anchor (`SolverConfig.lam_depth_anchor`, default 0 = absent): a dead-zone absolute depth anchor per frame
(`terms.depth_anchor_term`), exactly zero while the solved centroid depth stays within `depth_anchor_band` (log
units) of the observed masked-pointmap depth.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor

from fastdo.solver import batch_eval
from fastdo.solver._block_thomas import prefer_cusolver
from fastdo.solver.banded import marginal_covariances
from fastdo.solver.batch import (
    MIN_VALID_OBS,
    DeviceBatch,
    ObjectPixels,
    pack_clips,
    prepare_batch,
    sample_points,
    valid_pixels_of,
)
from fastdo.solver.batch_eval import EvalCache, Level, make_evaluate
from fastdo.solver.confidence import FrameConfidence, frame_confidence
from fastdo.solver.lm import LMConfig, lm_optimize_batch
from fastdo.solver.memory import (
    BatchPlan,
    ClipDims,
    allocator_headroom_bytes,
    card_headroom_bytes,
    plan_batches,
)
from fastdo.solver.moments import PlanMoments
from fastdo.solver.ot_sparse import SparsePlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeshSamples:
    points: Tensor    # (M,3) canonical surface samples
    normals: Tensor   # (M,3); required (not just when SolverConfig.visibility=True: this field has no
                      # default and was never optional) - used for the visibility-aware template measure
                      # when visibility=True, and for point_to_plane pointmap residuals.


@dataclass(frozen=True)
class TrajectoryObservations:
    intrinsics: Tensor
    masks: Tensor
    hand_masks: Tensor
    pointmaps: Tensor
    tracks: Tensor
    track_vis: Tensor
    anchors: Tensor
    # When set, the authority for every object-pixel read (`batch.valid_pixels_of`, `gate_init_poses`), and
    # `pointmaps` may be an all-NaN (T,H,W,3) placeholder: from a masked artifact the solve stage passes a zero-memory
    # NaN expand and reads the keyframe from `SolveInputs.keyframe_pointmap`.
    object_pixels: ObjectPixels | None = None


@dataclass(frozen=True)
class SolverConfig:
    n_obs: int = 1024
    n_mesh: int = 2048
    eps_coarse: float = 0.05
    eps_mid: float = 2e-3                 # 3-level rho/eps anneal, see module docstring
    eps_fine: float = 1e-4                # 0.005 (first design value) leaves a ~9mm bias at the default rho_y
    rho_x_coarse: float = 10.0            # rho is annealed together with eps, see module docstring
    rho_y_coarse: float = 1.0
    rho_x_mid: float = 1.0
    rho_y_mid: float = 0.1
    rho_x: float = 0.1                    # penalty for leaving observed mass unmatched, diameter^2 units (fine level)
    rho_y: float = 1e-3                   # penalty for leaving mesh mass unmatched, diameter^2 units (fine level)
    n_sinkhorn: int = 50                  # cold-start iterations; "cold 50 / warm 10"
    n_sinkhorn_warm: int = 10             # warm-started recomputes, coarse/mid levels
    n_sinkhorn_warm_fine: int = 10        # warm-started recomputes, fine level (see module docstring)
    # kNN-truncated Sinkhorn: the fine level's plan is local (measured mean
    # dropped row mass 2.4e-4 at k=32), so restricting each observed point's candidates to its 32
    # nearest VISIBLE mesh samples turns the dominant O(T N M) Sinkhorn cost into O(T N knn_fine) there
    # (a well-posed unbalanced OT problem in its own right, not an approximation).
    # Fine level only: the coarse/mid levels' near-hard mesh marginal makes their plan non-local by
    # design (Section 11.5's measured table), so they stay dense regardless of this field. `None` or
    # `0` disables truncation (a dense fine level, the behaviour before this field existed).
    knn_fine: int | None = 32

    iters_coarse: int = 10
    iters_mid: int = 5                    # mid annealing level's LM budget
    iters_fine: int = 20
    lam_track: float = 1.0
    # Per-frame track validity (default OFF, so the shipped path is bit-identical). When True the
    # track term's weight at frame k is scaled by the fraction of tracks still consistent with a
    # rigid motion there (`track_validity.frame_validity`, tracks only). Rationale: CoTracker's
    # error is bimodal and half of every clip's tracks are dead by frame ~14 of 24, so a
    # single weight is either too high late or too low early; a constant, a level schedule and a
    # per-clip gate were each rejected.
    track_validity: bool = False
    cauchy_c: float = 4.0
    # Temporal priors, per observed point (both rescaled by mean valid points/frame in solve_trajectory).
    # lam_vel: constant-velocity (second-difference) prior, the bridging prior;
    # 0 disables it. lam_smooth: constant-pose prior, a damping term (was 1.0).
    lam_vel: float = 1.0
    lam_smooth: float = 0.1
    # What the priors' support scale averages over: `all_frames` (shipped) or `observed_frames`, see
    # `batch.prior_support`. Only matters when some frames carry no valid pointmap point.
    prior_support: str = "all_frames"
    correspondence: str = "ot"
    # LM accepts a step when `sum w|r|^2` falls, but `r = sqrt(N*w)*(x - pbar)` with `w` the
    # unbalanced-OT row mass, so a model that transports no mass has NO residual however wrong its pose
    # is: measured over 15 clips, 14 detach somewhere on their gauge orbit and on 14/14 the accepted cost
    # there is 0.006-0.49x its value at the solve, while the unbalanced functional the clip's own Sinkhorn
    # minimises is 1.8-21x WORSE. With this True the merit adds the `rho_x * KL(P1 || a)` the Sinkhorn
    # already pays for, so abandoning the observations costs what it should. The Gauss-Newton DIRECTION is
    # unchanged (by the envelope theorem it is already the gradient of the full functional), and
    # `term_blocks` never sees the added field.
    #
    # DEFAULT OFF: it changes the accepted step sequence of every solve, so it would silently move every
    # frozen number in the project. Promote only on measured quality, not on the mechanism check alone.
    unbalanced_merit: bool = False
    # Plan moments (measured: ~98% of a batched solve's device peak sat in (T,N,M) tensors, 2.30 of 6.28 GiB the
    # cached dense plan). Between refreshes the plan P is fixed and the pointmap
    # term reads it only through pbar_i = sum_j P_ij (s R y_j + t) / w_i, w_i = sum_j P_ij. The Sim(3) action is affine,
    # so pbar = s R B + t with B = (P @ Y) / w the barycenter in CANONICAL coordinates, and the point-to-plane normal
    # is nbar = R Nbar with Nbar = (P @ normals) / w: (B, Nbar, w) are sufficient statistics of P at every trial pose.
    # With this True a refresh reduces each Sinkhorn chunk's plan (dense, kNN, or the "nn" one-hot) to (B, w) and the
    # cache (`batch_eval.EvalCache.plan`) holds `moments.PlanMoments` instead of the plan; trials cost O(T N), and the
    # dense plan never outlives its chunk (`memory.batch_peak_bytes` models this). `unbalanced_merit` needs only w, so
    # it is exact on moments; "nn" correspondences are reduced the same way. Nothing falls back to the plan.
    # DEFAULT ON: algebraically identical, the
    # reassociated sums move terms at float round-off only. 71 labelled clips: usable 40 -> 40, 0 harms, median |d post|
    # 0.000 mm, max pose diff 2.6e-6; Ego4D (13 frozen windows): upstream identical, gauge 1.0000000 (range +-2e-7).
    # With sparse_zbuffer the 5-clip Ego4D solve peak falls 26.2 -> 13.2 GB. False reproduces the earlier dense-plan
    # solves.
    plan_moments: bool = True
    # The track term's IRLS weight 1/(1+r²/c²) is exactly ρ′(s) for the Cauchy loss ρ(s) = c²·log1p(s/c²), so the
    # Gauss-Newton step descends Cauchy -- but LM's merit Σ w·s (w re-evaluated at the trial point) is Geman-McClure,
    # which discounts a dead track ~88x more (r²/c² at 37.5 px). "cauchy" accepts on the function the step descends.
    # DEFAULT "cauchy": on 71 labelled clips the pre-specified rule held (post-contact paired median -0.03 mm,
    # p = 0.040, 3 rescues vs 2 harms, known-answer 71/71),
    # and an LM trace showed why: under "irls" acceptance the shipped optimizer STALLS on outlier-heavy clips (1 of 10
    # coarse steps accepted on one clip, 40% on another), rejecting steps that descend the loss it actually minimises.
    # "irls" stays available to reproduce numbers frozen before the flip. "l2" (an ablation, opt-in): the track term
    # is plain least squares (IRLS weight = visibility, merit = sum of squares; `batch_eval.ROBUST_MERITS`).
    robust_merit: str = "cauchy"
    # Reject LM steps whose per-clip median object depth is at or below this (metres); None = no guard.
    # DEFAULT 0.05: bit-identical on all 71 labelled clips
    # (0/71 poses changed) and it keeps every observed behind-the-camera collapse in front of the camera.
    min_median_depth: float | None = 0.05
    sinkhorn_float32: bool = True          # Sinkhorn solve only; LM stays float64
    # Fused Triton kernels for the dense CUDA Sinkhorn (`ot_fused`). ON by default: both
    # opt-in conditions are met -- tests/solver/test_ot_fused.py passes on CUDA (11/11), and peak
    # memory is unchanged (7.48 GB at B=16 either way). The unfused solver is memory-bandwidth bound (~34k launches
    # per solve, elementwise log-domain ops each re-streaming the tensor), which is why batching plateaued at 1.24x;
    # fused measured 1.70x (B=1) to 2.32x (B=16) on the synthetic harness and 1.52x on real clips. qc26, 71
    # labelled clips vs GT: usable 39 -> 39, 0 harms, median |d post-contact| 0.000 mm (max raw pose diff 5.3e-4).
    # Portable: off CUDA, or where triton does not import, `ot_fused.fused_available` falls back to the reference
    # expressions, so CPU installs are unaffected.
    fused_sinkhorn: bool = True
    visibility: bool = True                # visibility-aware template measure, see visibility.py
    vis_downsample: int = 1                # z-buffer resolution divisor; full resolution:
                                            # a coarser buffer rejected genuinely-visible grazing-angle
                                            # samples whose depth varies a lot within one coarse cell
    # Build the visibility z-buffer over the pixels the samples land on (a sort + segmented amin) instead of
    # every pixel of every frame (8 bytes per pixel per frame: 2.2 GB per 100-frame clip at 1440x1920). `amin` is
    # exact, so the visibility mask is identical (`visibility._sparse_zbuffer_at`). DEFAULT ON: Ego4D
    # upstream identical, solve within T2 of the dense buffer (9/13 byte-identical, max 5.2e-6: sub-batch regrouping).
    # It changes memory (`memory.batch_peak_bytes`), not results.
    sparse_zbuffer: bool = True
    # Facing-test slack (n_j . v_j > -delta): a cosine slack (unitless - not an angle in degrees or
    # radians; see visible_samples's docstring), fields kept per-level. SolverConfig is the single
    # source of truth for the shipped values (not any prose elsewhere, including this comment).
    #
    # `vis_delta_coarse` trades DexYCB accuracy against synthetic large-init-error capture range - it
    # does not have one globally-better value, so pick per use case from the two configurations
    # measured (three-sequence DexYCB oracle gate, mean over pnp/gt/gt_perturbed; DO-AS-I-DO baseline
    # is F-5 0.71 / F-10 0.93 / Chamfer 0.66 cm^2):
    #
    #   vis_delta_coarse | DexYCB F-5 | DexYCB F-10 | DexYCB Chamfer | synthetic err_t (occ 0.0 / 0.5)
    #   0.0  (shipped)   |   0.770    |    0.927    |   0.52 cm^2    |     6.64 mm  /  6.73 mm
    #   0.25              |   0.539    |    0.960    |   0.65 cm^2    |     3.33 mm  /  6.68 mm
    #
    # 0.0 beats the baseline on all three DexYCB metrics (F-10 and Chamfer decisively, F-5 narrowly) and
    # is the shipped default; 0.25 trades a clear F-5 loss (0.539 vs baseline's 0.71) for a higher F-10
    # and roughly half the synthetic occlusion=0.0 translation error, i.e. more coarse-level capture
    # range from a badly-off initialization (the synthetic fixture's 5cm/15deg perturbed init) at the
    # cost of fine-level accuracy once a good pose is reached. Use `vis_delta_coarse=0.25` only if your
    # actual initialization errors are known to be large (comparable to or worse than 5cm/15deg) and a
    # coarser-but-more-robust fit is preferable to a more-accurate-but-less-forgiving one; DexYCB's own
    # pnp/gt_perturbed inits (recovered near-exactly by PnP, or perturbed by only 2cm/10deg) do not need
    # it, which is why 0.0 is the shipped default. `vis_delta_mid`/`vis_delta_fine` are not part of this
    # trade-off table - both stayed 0.0 throughout (a wider mid/fine slack regressed DexYCB on its own,
    # independent of the coarse-level trade-off above).
    vis_delta_coarse: float = 0.0
    vis_delta_mid: float = 0.0
    vis_delta_fine: float = 0.0
    vis_margin_coarse: float = 0.10
    vis_margin_mid: float = 0.05
    vis_margin_fine: float = 0.02
    # Hand-mask dilation radius, pixels (see visibility.py's dilate_hand_masks/exclude_hand_occluded).
    # 0 (no dilation): the GT hand label is exact, so a template sample projecting onto a hand pixel is
    # behind the hand by construction; a nonzero dilation was measured to remove visible template
    # samples right along the hand boundary, whose observations then pull toward other samples instead.
    vis_hand_dilate: int = 0
    # Per-frame confidence (`fastdo.solver.confidence`): exp(-statistic / confidence_scale), where the
    # statistic (the frame's mass-weighted RMS matched distance over the clip's noise floor) is ~1 at that floor, so a
    # frame at the floor scores ~exp(-1 / confidence_scale) (0.49 at 1.4). The confidence is a WITHIN-CLIP relative
    # signal: downstream QC must not accept or rank clips across clips on it; lateral translation and in-plane rotation
    # about the object axis are weakly visible in it (see TrajectorySolution).
    # `confidence_scale` is the one calibrated constant, bound by the DexYCB oracle runs alone
    # (run/pipeline/calibrate_confidence.py, outputs/eval/confidence_calibration.json): 1.4
    # is the smallest two-figure scale at which >= 90% of the frames with translation
    # error < 5 mm score > 0.5 over the three gate sequences x three init modes (0.920 = 138/150 at mass floor 0.7),
    # and the median within-clip Spearman(confidence, -error) over those sequences is > 0 (+0.109).
    # `confidence_mass_floor`: rows whose plan mass is below this fraction of the frame's mean valid-row mass are not
    # counted. At the fine level a row's mass falls roughly as exp(-r^2 / (rho_x (s D)^2)) with its matched distance
    # r, so 0.7 drops rows matched farther than ~0.19 s D (~28 mm on the synthetic test box) and keeps the 5-20 mm
    # rows of a 2 cm pose error. Measured on the synthetic renderer's see-through back-face rows (4-12% of a frame,
    # 44-50 mm off, 0.35-0.55x mass): at the solved poses good frames read z 1.72 (floors 0.1 and 0.5), 1.06 (0.6,
    # p90 1.28), 0.82 (0.7, p90 0.86), 0.79 (0.8); DexYCB oracle frames are insensitive to it.
    confidence_scale: float = 1.4
    confidence_mass_floor: float = 0.7
    # Anisotropic scale (motivation: a generated template's
    # proportions are seed samples, principal extents moving |dlog| 0.07/0.09/0.18 between two SAM 3D seeds while the
    # one solved scale moves 0.031). With this True the clip's scalar scale becomes a diagonal stretch in the template's
    # principal frame, p = R A (y - c) + t, A = U diag(exp l) U^T, l (3,) solved jointly with the poses (`aniso`). The
    # solution reports the geometric-mean `scale` exp(mean l) and the shape fields; poses keep today's convention.
    # DEFAULT OFF: opt-in like any quality lever until it is measured. Off, every op of the historical path is
    # unchanged (tests/solver/test_anisotropic_scale.py).
    anisotropic_scale: bool = False
    # Isotropy prior E = lam_b * sum_i (l_i - mean l)^2 with lam_b = lam_aniso * support * (s0 D)^2 (`batch`:
    # support the clip's mean valid points per frame, s0 D its initial metric diameter), i.e. charged per observed point
    # of ONE frame in the pointmap term's metres^2 currency, like the temporal priors. As a Gaussian prior of width
    # sigma_p against a per-point pointmap noise nu * s0 D, lam_aniso = (nu / sigma_p)^2: 0.01 is sigma_p = 0.15 (the
    # middle of the measured 0.1-0.2 log proportion noise) at nu = 0.015 (a 2 mm depth noise on a ~13 cm object, the
    # synthetic fixture's regime). A frame that sees axis i extended adds ~support * (s0 D)^2 * E[(u_i.(y-c))^2 / D^2]
    # (0.01-0.05 for a box) to its curvature, so the prior weighs as roughly 0.2-1 frame of POINTMAP evidence: any axis
    # the clip sees extended moves freely, and an axis it never sees stays at mean l. The track term is far stronger
    # still under the shipped lam_track: on test_anisotropic_scale's tumbling box (16 frames, at the truth) the l-block
    # diagonal is 1.3-4.9 from the pointmap, 3e5-2e6 from the tracks and 0.022 from this prior, so wherever tracks see
    # an axis the prior is inert and it acts on the axes neither term measures. Uncalibrated: to be calibrated on lab
    # clips (DexYCB subjects other than 07/08) against GT mesh extents.
    lam_aniso: float = 0.01
    # A per-frame ABSOLUTE depth anchor with a dead zone: with
    # d_t = log z_sol(t) - log z_obs(t), z_sol the camera-frame z of the posed template centroid (s R_t c + t_t)_z (c the
    # mean of the clip's `MeshSamples.points`, the contact stage's `centroid_obj`) and z_obs the median z of frame t's
    # masked finite in-front pointmap points (`batch.frame_depths`: `frame_depth`, NaN below 10 points,
    # and such a frame contributes nothing), frame t pays lam_depth_anchor * h(d_t)^2, h(d) = sign(d) max(0, |d| - band)
    # (`terms.depth_anchor_term`). Inside the band the residual and Jacobian are exact zeros; outside it the penalty is
    # convex and never redescends, so it restores a solve the redescending pointmap term has abandoned. It caps the
    # departure, it does not re-fit depth: z_sol is the CENTROID and z_obs the visible SURFACE, so a correct solve sits
    # at d > 0 by about the object's half-depth over its distance. Good lab clips sit at median d 0.001-0.065
    # (5th-95th percentile) and the catastrophic tail at |d| 0.36-0.74 for the pose ORIGIN z; for this centroid z the
    # same 71 clips read 0.001-0.063 and 0.36-0.73, hence the band 0.2.
    # Currency: lam multiplies a squared log-depth, not metres^2. The pointmap term of one frame costs ~support * dz^2
    # metres^2 near its optimum (support = valid points per frame, <= n_obs), and a depth excess e (log) is dz ~ z e,
    # so lam ~ support * z^2 (~500 at 1024 points and 0.7 m) weighs the excess like one frame of pointmap evidence.
    # DEFAULT 0.0: the term is never built (no computation). Uncalibrated; promote only on the operational endpoint.
    lam_depth_anchor: float = 0.0
    depth_anchor_band: float = 0.2
    # Hold the per-clip scale at the value it starts from (`init_scales`) on every level. It reaches
    # `LMConfig.fix_scale`, whose `_fix_scale` zeroes the scale row/column of the normal equations, so every step has
    # delta log s = 0 exactly and the returned `scale` is the initial one. The engine sets it only through
    # `solve_stage.fixed_scale` (a per-window scale written by the instance-size fusion post-pass,
    # `fastdo.engine.instance_scale`). DEFAULT False: `_levels` then builds the LMConfigs it always built.
    fix_scale: bool = False
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate the depth-anchor fields (the only fields validated here).

        Raises:
            ValueError: When `lam_depth_anchor` is not finite and >= 0, or `depth_anchor_band` is not finite and > 0.
        """
        lam, band = float(self.lam_depth_anchor), float(self.depth_anchor_band)
        if not (math.isfinite(lam) and lam >= 0.0):
            raise ValueError(f"lam_depth_anchor must be finite and >= 0, got {self.lam_depth_anchor!r}")
        if not (math.isfinite(band) and band > 0.0):
            raise ValueError(f"depth_anchor_band must be finite and > 0, got {self.depth_anchor_band!r}")


@dataclass(frozen=True)
class TrajectorySolution:
    """One clip's solve. `iterations` counts the fine level's LM iterations; `seconds` is the wall time of the
    (budget-planned) sub-batch it was solved in divided by its number of clips (the whole solve for B = 1);
    `accepted` holds the clip's LM accept/reject decisions per annealing level (coarse, mid, fine).

    `confidence` (T,) in [0, 1] (`fastdo.solver.confidence`): `exp(-z_k / SolverConfig.confidence_scale)`.
    `z_k` is frame k's mass-weighted RMS matched distance |x_i - pbar_i| at the final fine-level refresh (rows above
    `confidence_mass_floor`) divided by the clip's noise floor, 1.4826 x the median matched distance over the clip's
    supported frames. `z_k` is ~1 for a frame at the noise floor whatever its number of observations or visible
    template samples, so such a frame scores ~exp(-1 / confidence_scale) (0.49 at 1.4; well-converged synthetic frames
    read z 0.82-0.92, i.e. 0.52-0.56), and the confidence is exactly 0.0 for a frame with fewer than `_MIN_VALID_OBS`
    valid pointmap points. `per_frame_cost` (T,) is the RMS over valid rows of the pointmap
    term's residual (metres; it depends on support), which QC's `ot_residual_rms_frac` reads.

    The confidence is a WITHIN-CLIP relative signal. The floor is the clip's own, so it ranks frames
    within one clip and cannot see an error every frame of the clip shares (a clip-wide bias along the camera ray, a
    whole-clip shift). Downstream QC must not accept or rank clips across clips on it; a clip-level confidence needs a
    gauge uncertainty (a conformal gauge interval), out of scope here. Within a clip it detects errors
    along the camera ray well; lateral translation and in-plane rotation about the object axis are weakly visible
    (shipped defaults, synthetic box at GT poses, held frame 0.50-0.55: 2 cm along the ray 0.05-0.07, 2 cm across it
    0.34-0.37, 10 / 20 deg about the object z axis 0.46-0.52 / 0.40-0.46; `fastdo.solver.confidence`), so QC must
    not rely on it for those errors.

    Time base: the clock stops after the results are copied to the host. On CUDA, `seconds`
    therefore includes device work still queued at the end of the solve (the final correspondence refresh and
    the confidence). An earlier version stopped the clock before those copies and excluded that work, which is
    how its 0.87 s per DexYCB sequence was measured. Compare `seconds` across batch sizes on one tree, not against
    figures from that earlier version.

    `plan` is the planned sub-batch the clip was solved in (`memory.BatchPlan`): its clips, Sinkhorn
    chunk and peak estimate, and the CUDA budget and card readings it was planned against.

    Anisotropic scale (`SolverConfig.anisotropic_scale`; None when off): `shape_log_stretch` l (3,), `shape_axes`
    U (3,3) (columns, descending template extent) and `shape_centroid` c (3,), float64. `scale` is then exp(mean l)
    and `poses` keep the isotropic convention p = scale R y + t on the template's isotropic part; the solved object is
    exactly
    p = scale R Y' + t with Y' = `aniso.corrected_template_points(y, solution)`. `covariance` is then the marginal of
    the left pose perturbation of the solver's centroid-referenced pose (the reported pose times a fixed body-frame
    shift at the solution, which a left perturbation does not see), with the rank-3 stretch correction."""

    poses: Tensor
    scale: float
    covariance: Tensor
    per_frame_cost: Tensor
    confidence: Tensor
    iterations: int
    seconds: float
    accepted: tuple[tuple[bool, ...], ...] = ()
    plan: BatchPlan | None = None
    shape_log_stretch: Tensor | None = None
    shape_axes: Tensor | None = None
    shape_centroid: Tensor | None = None


SolveResult = TrajectorySolution


_MIN_VALID_OBS = MIN_VALID_OBS   # a frame with fewer valid points is treated as unobserved (module docstring)


def _sample_observations(obs: TrajectoryObservations, n_obs: int, gen: torch.Generator, device: str
                         ) -> tuple[Tensor, Tensor]:
    """Per frame, up to n_obs valid masked pointmap points -> X (T,n_obs,3), valid (T,n_obs) on `device`
    (see `batch.sample_points`, which the solver draws them with)."""
    points, counts = valid_pixels_of(obs)
    X, valid = sample_points(points, torch.cat([counts.new_zeros(1), counts.cumsum(0)]), n_obs, gen)
    return X.to(device), valid.to(device)


def _levels(cfg: SolverConfig) -> tuple[tuple[Level, LMConfig], ...]:
    """Coarse (uniform template mass, capture range), mid and fine (foreshortening-weighted mass; the fine
    level's kNN truncation is separate). n_obs/n_mesh stay full size on every level: reducing them at
    coarse/mid flipped test_ot_beats_nearest_neighbour_under_occlusion."""
    return ((Level(cfg.eps_coarse, cfg.rho_x_coarse, cfg.rho_y_coarse, cfg.n_sinkhorn_warm, cfg.vis_delta_coarse,
                   cfg.vis_margin_coarse, False), LMConfig(max_iter=cfg.iters_coarse, min_median_depth=cfg.min_median_depth, fix_scale=cfg.fix_scale)),
            (Level(cfg.eps_mid, cfg.rho_x_mid, cfg.rho_y_mid, cfg.n_sinkhorn_warm, cfg.vis_delta_mid,
                   cfg.vis_margin_mid, True), LMConfig(max_iter=cfg.iters_mid, lam0=3e-3, min_median_depth=cfg.min_median_depth, fix_scale=cfg.fix_scale)),
            # Fine level: n_sinkhorn_warm_fine sets final precision (test_recovers_trajectory's tolerance).
            (Level(cfg.eps_fine, cfg.rho_x, cfg.rho_y, cfg.n_sinkhorn_warm_fine, cfg.vis_delta_fine,
                   cfg.vis_margin_fine, True, cfg.knn_fine), LMConfig(max_iter=cfg.iters_fine, lam0=1e-3, min_median_depth=cfg.min_median_depth, fix_scale=cfg.fix_scale)))


def clip_dims(meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
              cfg: SolverConfig) -> list[ClipDims]:
    """Host-known sizes of every clip for `memory.plan_batches` (no packing, no device work)."""
    return [ClipDims(n_frames=int(p.shape[0]), n_mesh=min(cfg.n_mesh, int(m.points.shape[0])),
                     image_hw=(int(o.masks.shape[1]), int(o.masks.shape[2])), n_tracks=int(o.anchors.shape[0]))
            for m, o, p in zip(meshes, obs, init_poses, strict=True)]


def plan_solve(dims: list[ClipDims], config: SolverConfig) -> list[BatchPlan]:
    """The planned sub-batches `solve_trajectories` solves clips of sizes `dims` in (`memory.plan_batches`).

    Every clip is planned against `allocator_headroom_bytes`, the `apply_gpu_budget` / FASTDO_GPU_MEM_GB cap, alone.
    The card's free memory (`card_headroom_bytes`) is read only for more than one clip and bounds only sub-batches of
    more than one clip, so whether a one-clip solve chunks or raises does not depend on other processes on a shared
    card. Each plan records both readings and whether the card narrowed it."""
    card = card_headroom_bytes(config.device) if len(dims) > 1 else None
    return plan_batches(dims, config, allocator_headroom_bytes(config.device), batch_eval._REFRESH_ELEMENTS, card)


def solve_trajectories(meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
                       init_scales: list[float], config: SolverConfig) -> list[SolveResult]:
    """Solve B clips at once (coarse-to-fine partial-OT LM), returning one `SolveResult` per clip in order.

    The clips form one decoupled system (see `fastdo.solver.batch`), so each result equals a separate
    solve of that clip up to round-off, while the per-iteration kernel launches and host synchronizations
    are shared by all B clips. Before any device allocation, `plan_solve` splits the clips into consecutive
    sub-batches that fit the process's CUDA budget (`apply_gpu_budget`; for more than one clip also the card's free
    memory, never for one clip) and bounds their Sinkhorn chunks; a clip that cannot fit alone raises
    `memory.GpuBudgetError`. Each result records its plan (`TrajectorySolution.plan`). Off CUDA there is no budget
    and every clip is in one batch. cuSOLVER is pinned once for the whole solve and the caller's backend preference
    is restored on exit, also on an error (`_block_thomas.prefer_cusolver`). Clips of one sub-batch
    share padded sizes (template samples, image, tracks), so every frame pays for the largest: for throughput, batch
    clips of similar template size (a CPU run of four clips with templates of 256-1024 samples took 47.6 s batched
    against 27.7 s one by one)."""
    if not (len(meshes) == len(obs) == len(init_poses) == len(init_scales)) or not meshes:
        raise ValueError(f"need equally many (>0) meshes/obs/init_poses/init_scales, got "
                         f"{len(meshes)}/{len(obs)}/{len(init_poses)}/{len(init_scales)}")
    plans = plan_solve(clip_dims(meshes, obs, init_poses, config), config)
    if len(plans) > 1 or plans[0].refresh_elements < batch_eval._REFRESH_ELEMENTS:
        logger.info("solve_trajectories: CUDA budget plan (clips, chunk, GiB, card-limited) %s",
                    [(len(p.clips), p.refresh_elements, round(p.peak_bytes / 2**30, 2), p.card_limited) for p in plans])
    out: list[SolveResult] = []
    with prefer_cusolver(config.device):              # once per solve, never inside the LM loop
        for plan in plans:
            pick = list(plan.clips)
            sols = _solve_batch([meshes[i] for i in pick], [obs[i] for i in pick], [init_poses[i] for i in pick],
                                [init_scales[i] for i in pick], config, plan.refresh_elements)
            out.extend(replace(sol, plan=plan) for sol in sols)
    return out


def _final_scores(db: DeviceBatch, cfg: SolverConfig, cache: EvalCache, refresh_elements: int, poses: Tensor,
                  log_scale: Tensor) -> tuple[Tensor, FrameConfidence]:
    """One fine-level refresh at `poses`: the per-frame cost (RMS of the pointmap residual over valid rows, the
    definition) and the per-frame confidence of every clip (`fastdo.solver.confidence`, from the refreshed plan's row
    masses)."""
    terms, _ = make_evaluate(db, cfg, _levels(cfg)[-1][0], cache, refresh_elements)(poses, log_scale, True)
    pm = terms[0]
    pm_res = pm.residual.norm(dim=-1)                                        # (T,N)
    pm_support = pm.weight.sum(1)                                            # (T,) valid pointmap point count
    rms = ((pm_res**2 * pm.weight).sum(1) / pm_support.clamp_min(1.0)).sqrt()
    plan = cache.plan
    if isinstance(plan, PlanMoments):                                        # the moments carry w = P.sum(-1)
        row_mass = plan.mass
    else:
        row_mass = plan.values.sum(2) if isinstance(plan, SparsePlan) else cast("Tensor", plan).sum(2)
    conf = frame_confidence(pm.residual, row_mass, db.X_valid, db.layout.frame_slices(), scale=cfg.confidence_scale,
                            mass_floor=cfg.confidence_mass_floor, min_support=_MIN_VALID_OBS)
    return rms, conf


def _solve_batch(meshes: list[MeshSamples], obs: list[TrajectoryObservations], init_poses: list[Tensor],
                 init_scales: list[float], cfg: SolverConfig, refresh_elements: int) -> list[SolveResult]:
    """One planned sub-batch of `solve_trajectories`, with Sinkhorn chunks of at most `refresh_elements` entries."""
    t_start = time.perf_counter()
    packed = pack_clips(meshes, obs, init_poses, init_scales)
    db = prepare_batch(packed, cfg)
    layout = db.layout
    cache = EvalCache()
    poses, log_scale = db.poses0, db.log_scale0
    accepted: list[tuple[tuple[bool, ...], ...]] = []
    for level, lm_cfg in _levels(cfg):
        cache.reset()          # eps/rho change discontinuously between levels: no plan/potential/mask reuse
        res = lm_optimize_batch(poses, log_scale, make_evaluate(db, cfg, level, cache, refresh_elements), lm_cfg,
                                layout)
        poses, log_scale = res.poses, res.log_scale
        accepted.append(res.accepted)
    # O(T) marginal covariances via the block-tridiagonal backward (RTS) recursion. `lam=0.0`
    # matches the old convention of only adding the numerical floor, not LM's step damping.
    cov = marginal_covariances(res.system)
    # The final refresh warm-starts from the fine level's cache; below _MIN_VALID_OBS points a frame's confidence is
    # 0.0, not the vacuous exp(-0) = 1.0.
    rms, conf = _final_scores(db, cfg, cache, refresh_elements, poses, log_scale)
    shape: list[tuple[Tensor | None, Tensor | None, Tensor | None]] = [(None, None, None)] * layout.n_clips
    if log_scale.dim() == 2:                                             # back to the isotropic pose convention
        poses, log_scale, shape = _isotropic_outputs(db, poses, log_scale)
    poses_cpu, cov_cpu = poses.float().cpu(), cov.cpu()
    cost_cpu, conf_cpu = rms.float().cpu(), conf.confidence.float().cpu()
    floors_mm = (conf.noise_floor * 1e3).tolist()
    scales = log_scale.exp().cpu().tolist()
    costs = res.cost.cpu().tolist()
    seconds = (time.perf_counter() - t_start) / layout.n_clips
    out = []
    for b, sl in enumerate(layout.frame_slices()):
        logger.info("solve_trajectories: clip %d/%d T=%d iters=%d cost=%.4g scale=%.4f noise floor %.2f mm %.2fs/clip",
                    b + 1, layout.n_clips, sl.stop - sl.start, res.iterations[b], costs[b], scales[b], floors_mm[b],
                    seconds)
        out.append(TrajectorySolution(poses=poses_cpu[sl], scale=scales[b], covariance=cov_cpu[sl],
                                      per_frame_cost=cost_cpu[sl], confidence=conf_cpu[sl],
                                      iterations=res.iterations[b], seconds=seconds,
                                      accepted=tuple(level_acc[b] for level_acc in accepted),
                                      shape_log_stretch=shape[b][0], shape_axes=shape[b][1], shape_centroid=shape[b][2]))
    return out


def _isotropic_outputs(db: DeviceBatch, poses: Tensor, log_stretch: Tensor
                       ) -> tuple[Tensor, Tensor, list[tuple[Tensor | None, Tensor | None, Tensor | None]]]:
    """The solver's centroid-referenced poses and (B,3) log-stretches -> (poses in today's convention, (B,)
    geometric-mean log-scales, per-clip shape fields (log-stretch, axes, centroid)). With s = exp(mean l),
    p = R A (y - c) + t = s R Y' + (t - s R c) where Y' = c + (A / s)(y - c) (`aniso.corrected_template_points`), so
    t_out = t - s R c (`aniso`'s docstring)."""
    ci = db.layout.clip_index
    axes, centroid = cast("Tensor", db.shape_axes), cast("Tensor", db.shape_centroid)
    log_scale = log_stretch.mean(-1)
    out = poses.clone()
    shift = (poses[:, :3, :3] @ centroid.unsqueeze(-1)).squeeze(-1)      # R c, per frame
    out[:, :3, 3] = poses[:, :3, 3] - log_scale.exp()[ci, None] * shift
    first = [sl.start for sl in db.layout.frame_slices()]
    shape: list[tuple[Tensor | None, Tensor | None, Tensor | None]] = [
        (log_stretch[b].detach().cpu(), axes[k].detach().cpu(), centroid[k].detach().cpu()) for b, k in enumerate(first)]
    return out, log_scale, shape


def solve_trajectory(mesh: MeshSamples, obs: TrajectoryObservations, init_poses: Tensor, init_scale: float,
                     cfg: SolverConfig) -> TrajectorySolution:
    """One clip: the B = 1 case of `solve_trajectories`."""
    return solve_trajectories([mesh], [obs], [init_poses], [init_scale], cfg)[0]


def score_trajectory(mesh: MeshSamples, obs: TrajectoryObservations, poses: Tensor, scale: float,
                     cfg: SolverConfig) -> tuple[Tensor, Tensor]:
    """`(per_frame_cost, confidence)`, (T,) float32 on the host, of one clip held at `poses` / `scale` without solving.

    The statistics `solve_trajectories` reports, from one fine-level refresh at the given poses (visibility mask and a
    cold Sinkhorn of `n_sinkhorn` iterations; the solve's own final refresh warm-starts from its fine level), planned
    against the CUDA budget like a one-clip solve. Use it to see how the confidence ranks given poses, e.g. a solved
    trajectory with one frame deliberately displaced. With `anisotropic_scale` the clip is held at the
    isotropic stretch log(scale) on every axis."""
    refresh = plan_solve(clip_dims([mesh], [obs], [poses], cfg), cfg)[0].refresh_elements
    db = prepare_batch(pack_clips([mesh], [obs], [poses], [scale]), cfg)
    rms, conf = _final_scores(db, cfg, EvalCache(), refresh, db.poses0, db.log_scale0)
    return rms.float().cpu(), conf.confidence.float().cpu()
