# alongray

The unaligned along-ray error `e_par` of hand-object reconstructions, split exactly into a **gauge part** and an
**off-orbit term**, plus the usual pose-aligned metrics for contrast. Dependencies: numpy and scipy only.

Under a pinhole camera, a monocular reconstruction is image-exact under one scalar per body: scaling a body's
position and size by `lam` about the camera centre leaves every projected point of that body where it was. The
statement is about isolated projected points. When bodies occlude or touch each other, moving one in depth relative to
another can change which surface is visible, or whether they intersect, and so change pixels. Protocols that align
the prediction first (root-aligned MPJPE, PA-MPJPE, Chamfer after scaled ICP) remove a shared translation or scale
before scoring. So they can hide this error, or confound it with shape. Some relative hand-object errors stay visible
to them. `e_par` scores the reconstruction where it sits in the camera frame, and its split says how much of the
error that single scalar explains.

## Terms used below

- **body**: the object or the hand, each scored at one reference point per frame (object centre, wrist).
- **window** (or **unit**): one scored stretch of frames of one clip; every per-body scale below is fitted per window.
- **along-ray error**: the component of the error along the line from the camera centre through the true point.
- **gauge**: scaling a body's position and size by one factor `lam` about the camera centre. Under a pinhole
  camera this leaves every projected point of that body in place, so images alone cannot fix it. The **orbit** of a
  trajectory is the set of all such scalings of it.
- **gauge part / off-orbit part**: the share of the along-ray error that one fitted `lam` explains, and the rest.
  JSON keys: `e_par`, `gauge_part`, `off_orbit_part`, `lam`, `z_rms` (the RMS true range).

## Quickstart (about one minute on CPU; needs network for the install)

Needs Python 3.10 or newer; the commands below ask uv for 3.12, which it downloads if it is missing. Run them in
bash or zsh (on Windows, activate with `.venv\Scripts\activate`), from this directory (the one that holds this README
and `pyproject.toml`). With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install .
python examples/make_example.py out                 # writes out/pred.npz and out/gt.npz (synthetic, seeded)
alongray out/pred.npz out/gt.npz --intrinsics 600,600,320,240 --out out/scores.json
```

Without uv: `python3 -m venv .venv && source .venv/bin/activate && python -m pip install .`, then the last two lines.
The example's prediction is the truth moved by a pure gauge: the object scaled by 1.1 about the camera centre (10%
too far and 10% too big), and the hand moved 10% farther along its wrist ray at its true size. The command prints the
scores as JSON and writes them to `out/scores.json`. Abbreviated (the hand block has the same fields):

```
"object": {"e_par": 0.0628..., "gauge_part": 0.0628..., "off_orbit_part": 1.3e-16, "lam": 1.1000000000000003,
           "reproj_rms_px": 4.2e-14, "shift_part": 0.0623..., "collinearity": 0.9835..., ...}
"hand":   {"e_par": 0.0626..., "gauge_part": 0.0626..., "off_orbit_part": 2.2e-16, "lam": 1.0999999999999996, ...
           "aligned": {..., "mpjpe_root_aligned": 2.0e-17, "pa_mpjpe": 1.1e-16}, ...}
```

How to read it:
- Both bodies have `lam` 1.1 and about 63 mm of `e_par`, all of it in `gauge_part`. The hand gets 1.1 too although it
  kept its size, because `lam` is fitted to the reference point (the wrist) only: it is a position scale, not a size
  ratio.
- `off_orbit_part` (metres) and `reproj_rms_px` (pixels) are zero to float rounding, and so are root-aligned MPJPE
  and PA-MPJPE: the aligned metrics and the reprojection do not see this error.
- The takeaway: `collinearity` is 0.98 and `shift_part` is 0.062 m, so on this window one constant depth shift would
  explain almost as much as the scale. The split locates the error along the ray; it does not by itself show that a
  scale caused it (section "Scale or constant shift?").

`tests/test_quickstart.py` checks these numbers (it runs the example script and the command's entry point
in-process). Then see "Install" and "Tests" below.

## Install

From this directory, inside the environment you will use (activate it first, e.g. `source .venv/bin/activate`):

```
uv pip install .                     # numpy and scipy; without uv: python -m pip install .
uv pip install ".[test]"             # adds pytest, for the tests
```

A venv made by `uv venv` has no pip of its own, so `python -m pip` fails there; use `uv pip`. Outside a uv venv,
`python -m pip install` takes the same arguments.

From the root of a checkout of this repository, into its uv env (after `uv sync`, which does not install alongray):
`uv pip install ./packages/alongray`. A later `uv sync` removes it again.

## Prior work

The split of a 3D error into a component along the camera ray and a component across it is not new. EgoRayFormer (arXiv 2610.26641, Sec. 4.5 and Fig. 3) decomposes joint errors along and across the ray from the device centre, and monocular 3D detection and depth evaluation report depth and lateral errors separately. This package uses the same radial/perpendicular decomposition for object centres and wrists; it reports the full perpendicular RMS over both perpendicular axes, not a per-axis value. It adds the exact gauge / off-orbit split, failure-aware pooling and input checks.

## Definitions

Inputs are camera-frame positions in metres, with the camera centre at the origin. Each body is scored at one
reference point per frame: the object centre and the hand root (wrist). Write `c_t` for the predicted point in frame
`t`, `c_gt,t` for the true one, `<.>` for the mean over the frames where both prediction and truth are finite, and

| quantity | definition |
|---|---|
| true range | `Z_t = \|c_gt,t\|`, true ray `u_t = c_gt,t / Z_t` |
| signed along-ray error | `a_t = (c_t - c_gt,t) . u_t` (positive: predicted too far) |
| **e_par** | `<a^2>^(1/2)`, no alignment of any kind |
| per-body least-squares scale | `lam = sum_t c_t . c_gt,t / sum_t \|c_gt,t\|^2`, the point of the gauge orbit `{mu c_gt}` closest to the prediction |
| off-orbit residual (along the ray) | `o_t = (c_t - lam c_gt,t) . u_t` |
| **gauge part** | `\|lam - 1\| <Z^2>^(1/2)` |
| **off-orbit term** | `<o^2>^(1/2)` |
| oracle floor | `<o^2>^(1/2) / lam`: `e_par` after the oracle correction `g_{1/lam}: c -> c / lam`. It is **not** the minimum of `e_par` over all gauge moves, which is lower whenever the off-orbit term is non-zero (that minimum is `e_par_min`, below) |

## The exact split

Since `a_t = (lam - 1) Z_t + o_t`, expanding the square gives a cross term `2 (lam - 1) <Z o>`. The condition
`sum_t Z_t o_t = 0` is the normal equation that defines `lam`, so that cross term is zero and

```
e_par^2 = (lam - 1)^2 <Z^2> + <o^2>          (exact; holds to float rounding)
```

A pure scale error `c_t = k c_gt,t` gives `lam = k`, `o = 0`, and puts all of `e_par` in the gauge part. A constant
along-ray offset `d` puts `collinearity * d^2` of its mean square in the gauge part, where
`collinearity = <Z>^2 / <Z^2>`. Over one window of real hand-object video the true range barely varies, so
`collinearity` is close to 1: it was 0.995 (DexYCB) and 0.999 (HOI4D) at the median on our development units (from
development data that does not ship with the package; no shipped fixture reproduces these medians), and a constant offset then lands almost entirely in the gauge part. For a nonzero constant offset the off-orbit
fraction of its mean square is `1 - collinearity`, so more than half is off-orbit exactly when `collinearity` is
below 0.5. After a gauge move `c -> kappa c`,
`e_par(kappa)^2 = (kappa lam - 1)^2 <Z^2> + kappa^2 <o^2>`.

The two parts split the uncorrected error in quadrature into components parallel and orthogonal to the truth's scale
direction. In mean square they are exactly what moving the truth to its nearest orbit point `lam c_gt` takes away
and leaves. They are generally not the gain and the remainder of the best gauge move of the prediction, because that move rescales `o`
too. That move is `kappa_star = lam <Z^2> / (lam^2 <Z^2> + <o^2>)` and leaves
`e_par_min = (<Z^2> <o^2> / (lam^2 <Z^2> + <o^2>))^(1/2)`; `alongray.kappa_star(split)` and `alongray.e_par_min(split)`
return them for an `EParSplit` or a `BodyScore`. `e_par_min` lies below `oracle_floor` whenever `<o^2> > 0`, and
strictly above the off-orbit term exactly when `<o^2> > 0` and `(1 - lam^2) <Z^2> > <o^2>`. With zero residual
(`<o^2> = 0`) both are zero, whatever `lam` is: a prediction at half the true range on every frame has `e_par_min`
and off-orbit term 0. Both read the ground truth, so they are references, not corrections a method can
run. Because the off-orbit term scales with the prediction, shrinking an object toward the camera lowers it without
fitting the images any better. When two methods' scales differ, compare `oracle_floor` or `e_par_min` as well, since
a gauge move of the prediction leaves both unchanged.

## Scale or constant shift?

The split locates the error along the ray. It does not by itself show that a scale caused it. `e_par_split` and
`score` therefore also report:

| field | meaning |
|---|---|
| `collinearity` | `<Z>^2 / <Z^2>` in (0, 1]; 1 means a scale and a constant depth shift are indistinguishable |
| `shift_part` | `\|<a>\|`: what ONE constant along-ray shift removes |
| `shift_scale_part` | what a joint least-squares fit of `a` on `span{1, Z}` removes; it bounds both explanations together |
| `low_n` | fewer than `MIN_ROWS = 3` frames (see Limits) |

Read `gauge_part` as evidence of a scale error only when `gauge_part^2` clearly exceeds `shift_part^2`, that is,
when the scale's unique share `shift_scale_part^2 - shift_part^2` is large against the shift's unique share
`shift_scale_part^2 - gauge_part^2`. `collinearity < 1` is necessary, not sufficient: a pure constant shift at
`collinearity = 0.85` still puts 85% of its mean square into the gauge part. Report `collinearity` and `shift_part`
beside the split.

The intrinsics are not needed for `e_par` or its split, since every ray passes through the origin of the camera
frame. If you pass them, `reproj_rms_px` reports the reference point's reprojection error. That error is exactly
zero under a pure gauge, which makes it a useful sanity check.

## Input conventions

- **Units: metres.** If `z_rms` is above 10 (m), the scorer warns that the inputs look like millimetres. It also warns
  when `lam` falls outside [0.1, 10]. It raises `ValueError` when `lam` is not finite and positive, which usually
  means mixed frames, mixed units, or missing frames filled with zeros instead of NaN. Mark missing frames with NaN;
  any frame that is not finite in both prediction and truth is dropped.
- **Camera frame, camera at the origin.** `e_par` and its split depend only on rays through the origin, so they are
  invariant to the choice of camera axes (OpenCV +z forward or OpenGL -z forward), provided prediction and truth use
  the same axes. The reprojection contrast needs **+z forward** (OpenCV). With `k` given and no frame in front of the
  camera, it warns and reports `reproj_rms_px = NaN` with `n_reproj_frames = 0`.
- **Object poses are object-to-camera** (T, 4, 4) rigid transforms, with a bottom row of `[0, 0, 0, 1]` (checked).
  Their translation is used as the reference point. It must be the **same physical point** in prediction and truth,
  e.g. both mesh centroids. A template origin in one and a centroid in the other shows up as error. The object accepts
  only (T, 3) or (T, 4, 4).
- **Hands.** Pass (T, 3) wrists, or (T, J, 3) joints in the **same joint order** in prediction and truth.
  `hand_root` is the reference joint, and the root used by root-aligned MPJPE. It defaults to 0, the wrist in MANO's
  21-joint order. Check your own ordering, since some datasets put the wrist elsewhere.
- **One scale per body and window.** `lam` is a least-squares fit over the frames, so far frames weigh more (by
  `Z^2`). It is a position scale, not a size ratio. With **T = 1**, the normal equation forces `o = 0`: all
  along-ray error is reported as gauge, and the split says nothing. Use windows of several frames with varying
  range.
- **Frame correspondence.** Without frame ids, row `t` of the prediction and row `t` of the truth are **assumed to
  be the same frame** (synchronized row by row). Nothing checks this: a prediction shifted by one frame, or missing a
  frame and so shifted after it, is scored as given whenever the lengths agree. Pass `pred_ids` / `gt_ids` to
  `score` (or `frame_ids` in both `.npz` files, with `--require-frame-ids` to make them mandatory) and the rows are
  joined on the ids by the rules of `alongray.change`: one integer id per row, no duplicates, the **same id set** on
  both sides (no intersection, no trimming, no shift), rows taken in increasing id order. Anything else is refused.
  Without ids the outputs match 0.3.1 in the four recorded cases of `tests/test_frame_ids.py`, under the rule stated
  in "What the tests cover"; the test does not cover every possible input.
- Inputs must be (T, 3) arrays (or the shapes above for `score`). Other shapes are refused, not reshaped. To split
  corresponding points (joints, same-topology vertices) over frames x points, pass the flattened (T*N, 3) arrays.

## Limits

1. **Scale versus constant shift** (section above): with `collinearity` near 1 the gauge part cannot be attributed
   to the scale.
2. **Few frames.** With one valid frame the off-orbit error is zero. The gauge share is one for nonzero along-ray
   error and undefined at zero error. With two frames one degree of freedom is left. A split over fewer than `MIN_ROWS = 3`
   frames carries `low_n = True` and warns (`UserWarning`, on stderr from the command line); leave such windows out
   or report them apart, and count them (`alongray-pool` reports `n_low_n`).
3. **One `lam` per body and window, least-squares weighted** (Input conventions). It is not the scale that
   minimises `e_par`, and `oracle_floor` is not that minimum; they are `kappa_star(split)` and `e_par_min(split)`.
4. **Camera frame and intrinsics.** The gauge is a scaling about the camera centre. A world-frame input, or a
   prediction expressed with a different camera (wrong focal length or principal point), adds error that is not a
   pure gauge; part of it lands in `e_perp` and in the off-orbit term.
5. **A fixed-size hand.** A method that keeps MANO at its template size cannot move along its own gauge; its
   along-ray error appears as translation, which root-aligned MPJPE ignores.
6. **Objects at the camera.** The camera-frame refusal uses one reference point per body. An object very close to
   the camera whose model origin (the reference point) is offset more than `tol` (at least 5 cm) behind the camera
   in some frames is refused although the frame is right; pass the mesh centroid, not an offset origin, as the
   reference point.
7. **Descriptive only.** Nothing here is a hypothesis test; `cluster_bootstrap` gives percentile intervals.

## Alignments side by side

`alongray.aligned_errors(pred, gt, valid=None, root=None)` reports the mean and RMS error after each alignment, and
`alongray.gauge_sensitivity(gt, lams)` what each one reports when the prediction is the truth moved by a PURE gauge
(a move that leaves every projected point in place under a pinhole camera, so any non-zero entry is error that the
projected points alone cannot reveal):

| alignment | fitted | error it reports for a pure gauge `p = lam g` |
|---|---|---|
| `none` | nothing | `\|lam - 1\| <\|g\|>` |
| `translation` | one offset per window | `\|lam - 1\|` x the mean distance of g from its window mean (the RMS spread for the `rms` entry; 0 for a static point) |
| `scale` | one scale about the camera centre per window, per body (`kappa p ~ g`) | 0 |
| `sim3` | one similarity per window | 0 |
| `root_frame` | translation per frame (root-aligned MPJPE) | `\|lam - 1\|` x the mean root-relative size |
| `sim3_frame` | similarity per frame (PA-MPJPE) | 0 |

Alignments that remove the gauge also remove real error along the gauge direction; alignments that keep it report
the gauge as error. Neither separates the two; the split does. `valid` must be a (T,) frame mask. Frame-scope
alignments need 2 (`root_frame`) or 3 (`sim3_frame`) points per frame and are `None` for a single reference point;
`sim3` is `None` when the true points are collinear to a relative tolerance of 1e-6.

## Several bodies: the relative gauge

`alongray.scene_gauge({"object": (pred, gt), "hand": (pred, gt)}, hand_root=0)` returns each body's `lam`, ONE
least-squares scale `shared_lam` over every body, `relative_log_gauge` (`{"object/hand": log(lam_o / lam_h)}`), and per
body `residual_after_shared_scale`: the body's `e_par` after the prediction is divided by `shared_lam`. The relative
gauge moves one body along its rays relative to the other while, under a pinhole camera, every projected point of
both stays put: the hand-object placement error that one scene similarity cannot fix. Occlusion and contact between
the two bodies are not modelled, and they can change pixels when one body moves in depth relative to the other. The command line adds it with `--scene`.

## The change between two predictions: `alongray.change`

`score` splits the error of ONE prediction against the truth. `alongray.window_change(before, after, gt, ids_before,
ids_after, ids_gt, unit_id=..., ray="gt")` answers a different question: when a method changes its prediction of a
unit (for example from its initialiser to its solve), how much of that change lies along the camera ray, and does the
change in error follow the change along the ray? `alongray.summarize_changes(windows, assigned=None)` pools windows.
The conventions are fixed in code; none has a silent default that changes the number:

| convention | what the code does |
|---|---|
| frame and units | camera frame, camera centre at the origin, metres; (T, 3) reference points only (other shapes refused) |
| ray | the change `d = after - before` is split on the unit ray of `gt` (`ray="gt"`) or of `before` (`ray="before"`: only the direction comes from the prediction; `gt` and `ids_gt` are still required, for the common frame set and the error terms); the error terms always use the true ray. Windows split on different rays are not pooled |
| correspondence and time | every array carries integer frame ids; the three id sets must be identical, and rows are joined on the ids. Duplicate, non-integer, missing or shifted ids are refused; rows are never intersected, trimmed or shifted |
| paired finite mask | a frame is used only where `before`, `after` and `gt` are all finite and both the chosen ray's norm and the true ray's norm exceed `min_range` (1e-6 m); both errors of a window use that one set of frames |
| dropped denominators | per window: `n_input`, `n_used`, `n_dropped` and the reasons `n_nonfinite_gt`, `n_nonfinite_before`, `n_nonfinite_after`, `n_ray_degenerate` (not exclusive), the dropped ids, and `n_zero_change`; per cohort: `n_assigned`, `n_windows`, `n_scored`, `n_zero_change` (windows with no change have no share), `n_no_frames`, `n_missing` |
| window share | `sum (d . u)^2 / sum \|d\|^2` over the window: a ratio of sums, not a mean of per-frame squared cosines |
| signed vs absolute | the signed mean `d . u` (positive: away from the camera) and the mean of `\|d . u\|`; for the errors, `<\|a_after\|> - <\|a_before\|>` and both signed means |
| pooling | the median over windows of the window share (`median_window_share`), with the ratio of pooled sums apart and labelled; the median of the PAIRED per-window centre-error differences (`median_paired_delta_centre`), with the unpaired difference of medians apart and labelled; Spearman over windows with average ranks for ties |

A difference of medians taken over two arms is not a paired difference: with errors (1, 2, 3) cm before and (3, 1,
2) cm after, the medians are equal while the median paired change is -1 cm (`tests/test_change.py`).

## The scale gauge on synthetic points: `alongray.gauge_demo`

Observability education, not error estimation. For a calibrated pinhole camera, `pi(K x) = pi(K lam x)` for every
`lam > 0`. `gauge_demo.scaled_scene_pair(points, lam, K)` builds a synthetic scene and its copy scaled by `lam` about
the camera centre and reports their largest pixel difference (at most 1e-8 px in float64 for `lam = 0.5` and `2` in
`tests/test_gauge_demo.py`); points at or below 1 mm depth, or behind the camera, are refused. Two negative controls
show metric information breaking the gauge: `fixed_anchor_control` (a hand anchor of fixed metric size can follow the
scaling only by changing size, so its pixels move) and `second_camera_control` (a second camera at a fixed non-zero
baseline sees different pixels; a zero baseline is the positive control). The module estimates no error of any
method, and it does not show that a complete rendered objective, with occlusion, contact or priors, is invariant. The
gauge freedom is classical: Triggs, McLauchlan, Hartley and Fitzgibbon, "Bundle Adjustment: A Modern Synthesis"
(2000), https://perception.inrialpes.fr/Publications/2000/TMHF00/Triggs-va99.pdf .

## Pooling many units

`alongray.pool_splits(splits)` reports medians, means and RMS over units, the pooled share
`sum gauge_part^2 / sum e_par^2` (units weighted by their mean square), the median per-unit share, the median
`collinearity` and `n_low_n`. `alongray.cluster_bootstrap(stat, clusters, n_boot, rng)` resamples clusters with
replacement: the clusters are the sorted distinct labels, each draw is ONE `rng.integers(0, K, K)` call on a numpy
Generator, the drawn units are those of clusters `pick[0], pick[1], ...` in their original order, and the interval
is `numpy.percentile` at `(100 - 100 level) / 2` and `(100 + 100 level) / 2`. With fewer than two distinct
clusters there is no between-cluster information, and `cluster_bootstrap` returns the string
`alongray.NOT_ESTIMABLE` (`"NOT_ESTIMABLE"`) instead of an interval. With two or more clusters it returns an
interval. A draw whose statistic does not exist (for the pooled share: every drawn cluster has zero error) is an
undefined draw: it is counted, left out of the percentiles, and never removes the point estimate; `alongray-pool`
then reports `pooled_share_ci95_undefined_draws` and a reason, and the interval is conditional on a defined share.
The shares `shift`, `shift_scale` and `pure_shift_null` need each unit's first moments (`z1`, `a1`). If any unit
lacks them, those three have no full-cohort value: `pooled_share` gives `null` for each, `pooled_share_undefined`
names each with its reason, and `pooled_share_ci95` carries no interval for them (the reason is in
`pooled_share_ci95_component_reason`); `gauge` and `off_orbit` keep their values and intervals. In
`cluster_bootstrap` a non-finite value of one key is undefined for that key on that draw: it is counted in
`info["undefined_by_key"]`, left out of that key's percentiles, and a key with no finite draw has no interval.
Two cases look alike and differ. A statistic that is defined and equal on every draw gives a zero-width interval.
A statistic that is undefined on every draw gives `NOT_ESTIMABLE`; if every unit has zero error, the point estimate
`pooled_share` is itself `null` with a `pooled_share_reason`, the interval's reason says that no point estimate exists
either, `pooled_share_ci95_undefined_draws` gives the count, and the zero lengths and the counts are still reported. In the
JSON of `alongray-pool` (and `pool_rows`) a group whose share exists keeps its point estimate `pooled_share` and reports
`"pooled_share_ci95": "NOT_ESTIMABLE"` with a `pooled_share_ci95_reason`; in `alongray-coverage` the row's
`ci95_cluster_bootstrap` is `"NOT_ESTIMABLE"`, with a `ci95_cluster_bootstrap_reason`. Read the interval field as a string or a mapping of `[lo, hi]`
pairs. A one-cluster group still makes its draws, so later groups sharing the generator get the same intervals as
before. `alongray.split_from_moments` and `alongray-pool`
pool rows in three forms. In them `a_r` is the prediction's **range** along the true ray, `<c_pred, c_gt> / |c_gt|`,
**not** the along-ray error `a` of the definitions above (`a = a_r - Z`):

1. `"moments"`: `pp = <a_r^2>`, `pz = <a_r Z>`, `zz = <Z^2>`, `n`, optionally `z1 = <Z>` and `a1 = <a_r>` (these add
   `shift_part`, `shift_scale_part` and `collinearity`);
2. per-frame `"z"` (true ranges, all > 0) and `"a_r"` lists (`"a"` is a deprecated alias and warns);
3. `"split"`: an `EParSplit` as a dict in metres, with `e_par, gauge_part, off_orbit_part, lam, z_rms, oracle_floor,
   n_frames` present and finite.

Inconsistent rows are refused, never clamped. Moments must obey Cauchy-Schwarz (`pz^2 <= pp zz`, `z1^2 <= zz`,
`a1^2 <= pp`) to a relative `MOMENT_RTOL = 1e-9`; a violation would make a squared part negative. A stored split
must have a positive integer `n_frames`, positive `lam` and `z_rms`, and satisfy `e_par^2 = gauge_part^2 +
off_orbit_part^2`, `gauge_part = |lam - 1| z_rms` and `oracle_floor = off_orbit_part / lam` to a relative
`SPLIT_RTOL = 1e-6`.

```
alongray-pool rows.jsonl --group-key corpus --cluster-key video --boot 10000 --seed 0 [--out pooled.json] \
    [--assigned units.txt] [--failure-policy report|refuse]
```

**Denominators.** A pooled number over the units a method delivered says nothing about the units it failed on. A row
may carry `"status"`: `"scored"` (the default; the row has one of the three forms), `"failed"` (assigned, but the
method delivered nothing to score; no form) or `"excluded"` (left out by a stated rule, with a `"reason"`). With
`--assigned` (one `unit_id`, or `unit_id group` with `--group-key`, per line) an assigned unit without a row counts as
missing and a row outside the manifest is refused. Every group reports `n_assigned`, `n_scored`, `n_failed`,
`n_excluded`, `n_missing`, `excluded_reasons`, `assignment` and a `cohort` sentence. Without `--assigned`,
`assignment` is `"observed rows only"`: the assigned count is then the number of rows and `n_missing` is 0 by
construction, which says nothing about units that never produced a row. An assigned unit may be listed once. Every statistic, the bootstrap interval
included, is over the scored units only. `--failure-policy refuse` (`failure_policy="refuse"`) refuses a group
with any failed or missing unit, so that what is pooled is every assigned unit not excluded for a stated reason; the
default `report` pools the scored units and prints the denominators beside them. Quote the denominators with the
pooled numbers.

## Changes

See `CHANGELOG.md`.

## Minimal example

```python
import numpy as np
import alongray

rng = np.random.default_rng(0)
T = 30
gt_obj = np.column_stack([rng.normal(0, 0.05, T), rng.normal(0, 0.05, T), rng.uniform(0.45, 0.75, T)])
gt_hand = gt_obj[:, None, :] + 0.03 * rng.standard_normal((T, 21, 3))      # (T, J, 3), joint 0 = wrist

pred_obj = 1.1 * gt_obj                                                    # object centre 10% farther from the camera
pred_hand = gt_hand + 0.1 * gt_hand[:, :1]                                 # wrist 10% farther; joint distances unchanged

res = alongray.score(pred_obj, gt_obj, pred_hand, gt_hand, k=(600.0, 600.0, 320.0, 240.0))
for body in ("object", "hand"):
    b = res[body]
    print(body, f"e_par {b.e_par * 1e3:.1f} mm = gauge {b.gauge_part * 1e3:.1f} (+) off-orbit "
                f"{b.off_orbit_part * 1e3:.1f}; lam {b.lam:.3f}; reproj {b.reproj_rms_px:.1e} px")
print("root-aligned MPJPE", res["hand"].aligned["mpjpe_root_aligned"], "PA-MPJPE", res["hand"].aligned["pa_mpjpe"])
```

Both bodies report about 63 mm of `e_par`, all of it in the gauge part, with `lam = 1.100`. The reprojection error,
root-aligned MPJPE and PA-MPJPE are all zero to float rounding. `tests/test_readme.py` runs this example.

The lower-level functions are `alongray.e_par_split(c_pred, c_gt, kappa=1.0)`, `along_ray_components`,
`gauge_decomposition`, `apply_gauge`, `kappa_star` and `e_par_min`. The aligned protocols are `mpjpe_root_aligned`, `pa_mpjpe`,
`object_cd_after_scaled_icp`, `scaled_icp` and `umeyama_similarity`.

## Adapting a dataset: `examples/ho3d_shaped_adapter.py`

`python examples/ho3d_shaped_adapter.py OUT_DIR` turns a synthetic sequence laid out like a per-frame hand-object
annotation (an object rotation and translation, 21 hand joints, a camera matrix, frame numbers in file names) into
alongray inputs, scores it with `score(..., pred_ids=..., gt_ids=...)`, and writes `pred.npz` and `gt.npz` with
`frame_ids` for `alongray OUT_DIR/pred.npz OUT_DIR/gt.npz --require-frame-ids`. Every array is generated from a
seeded generator; the example reads and ships no dataset, model or mesh file. It makes five conventions explicit:
object-to-camera (4, 4) transforms; the same physical object point on both sides (a template origin against a
mesh centroid is scored as error, about 10 mm of off-orbit error here); metres (a millimetre prediction is
converted); the wrist as hand joint 0 in the same joint order on both sides; and integer frame ids, with the
prediction's rows shuffled and joined on them. It also flips OpenGL-style camera axes to OpenCV. Check each
convention against the documentation of the dataset copy you use. `tests/test_ho3d_example.py` runs it.

## API

`alongray.score(pred_object, gt_object, pred_hand=None, gt_hand=None, k=None, *, hand_root=0,
pred_object_points=None, gt_object_points=None, pred_ids=None, gt_ids=None) -> Score`

- `pred_object`, `gt_object`: (T, 3) centres, or (T, 4, 4) object-to-camera poses whose translation is the centre.
- `pred_hand`, `gt_hand`: (T, 3) wrists, or (T, J, 3) joints with root `hand_root`.
- `k`: (3, 3) intrinsics or `(fx, fy, cx, cy)`. Optional. It must be a finite pinhole matrix: positive focal
  lengths, `K[1, 0] = 0` and bottom row `[0, 0, 1]` (a skew `K[0, 1]` is allowed). Anything else is refused,
  because the reprojection divides by the camera-frame depth, which is the homogeneous coordinate only for that row.
- `pred_object_points`, `gt_object_points`: optional per-frame (V, 3) object clouds, used for the aligned Chamfer.
- `pred_ids`, `gt_ids`: optional integer frame ids, one per row of every prediction / truth array (both or
  neither; the hand and the clouds share the object's ids). The two id sets must be identical and free of
  duplicates; non-integer ids (floats, booleans) are refused. The rows are joined on the ids in increasing id order,
  so shuffled rows with their ids give exactly the result of the sorted rows. Without ids the rows are assumed
  synchronized row by row (Input conventions). `score_body` takes the same two arguments.

`score(...)["object"]` and `["hand"]` are `BodyScore`s with these fields:

- `e_par`, `gauge_part`, `off_orbit_part`, `lam`, `z_rms`, `oracle_floor`: as defined above;
- `e_perp`, `e_3d`: unaligned RMS of the perpendicular and of the full 3-D error;
- `n_frames`, `reproj_rms_px`, `n_reproj_frames`: frames used, the reprojection error, and the frames it uses;
- `shift_part`, `shift_scale_part`, `collinearity`, `low_n`: the constant-shift reading and the few-frames flag;
- `aligned`: a dict of pose-aligned metrics, reported for contrast:
  - `centre_rms_after_lam`: the 3-D residual after the per-body scale;
  - `centre_ate_sim3`: the error after one similarity over the trajectory;
  - `mpjpe_root_aligned` and `pa_mpjpe`: for (T, J, 3) hands only;
  - `chamfer_cm2_after_scaled_icp`: with point clouds only (NaN when no frame has finite clouds in both);
- `aligned_n_frames`: for each key of `aligned`, the frames its value was computed on (0 when it is NaN). The
  metrics do not share one denominator: the centre metrics use the frames whose reference point is finite in both,
  root-aligned MPJPE and PA-MPJPE the frames whose every joint is finite in both, and the Chamfer the frames whose
  clouds are finite in both. Quote each metric with its own count;
- `n_frames_input`: the frames given, before any is dropped.

Error magnitudes (`e_par`, `gauge_part`, `off_orbit_part`, `z_rms`, `oracle_floor`, `e_perp`, `e_3d`, `shift_part`,
`shift_scale_part` and the aligned centre and joint metrics) are in metres, `reproj_rms_px` is in pixels and the
Chamfer in cm^2. `lam` and `collinearity` are unitless, the frame fields are counts and `low_n` is a flag. Each body
gets its own `lam`. Frames
that are not finite in both prediction and truth are dropped.

## Command line

```
alongray PRED.npz GT.npz [--intrinsics fx,fy,cx,cy] [--hand-root 0] [--scene] [--require-frame-ids]
         [--out scores.json]
alongray --version
```

Both files use the same keys, in camera-frame metres:

- `object` (required): (T, 3) or (T, 4, 4);
- `hand` (optional, in both files or neither): (T, 3) or (T, J, 3);
- `object_points` (optional, in both files or neither): (T, V, 3);
- `K` (optional, in either file): (3, 3); if both files hold one they must be equal, and `--intrinsics` overrides
  both;
- `frame_ids` (optional, in both files or neither): (T,) integer ids, one per row; the rows are joined on them with
  the rules of `score`'s `pred_ids` / `gt_ids` (identical id sets, no duplicates, integers only, increasing id
  order). `--require-frame-ids` refuses (exit 2) files without them;
- `frame_idx` (optional, in both files or neither; not together with `frame_ids`): (T,) integer ids; the files are
  matched on the **shared** ids (the intersection, kept from 0.2 for compatibility), and non-integer, duplicate,
  disjoint or one-sided `frame_idx` are refused. Prefer `frame_ids`, which never drops a frame silently;
- `valid` (optional, in either file): a (T,) frame mask, boolean or 0/1 integers; any other shape or value is
  refused.
- `image_size` (optional, in either file): `(W, H)` in pixels; used only by the camera-frame warnings.

Without `frame_ids`, `frame_idx` and `valid`, every body is scored on its own, exactly as in 0.1 (bodies may have
different frame counts), and row `t` of each array is assumed to be frame `t` in both files (synchronized row by
row; nothing checks it). With any of the three keys, every body in a file must have the object's frame count; a
`valid` mask applies to its own file's rows before the join.

The command line also runs heuristic camera-frame checks on the ground-truth object centre and hand root. They
**warn by default and refuse only on strong evidence**. A refusal is not proof of a convention error: a genuinely
bad prediction can also put at least 90% of its centres behind the camera, and the command then refuses it too.
Read the refusal as "check the declared axes first":

- **refused:** ground-truth points on the minority side of the camera (sign of z: + for OpenCV, - for OpenGL axes)
  beyond `tol = max(0.05 m, 0.1 median|z|)` ("world-frame, mixed-frame, or an object origin behind the camera");
  and a mirrored prediction, with at least 90% of its points on the opposite side from the truth beyond `tol`
  (the usual cause is different axis conventions in the two files);
- **warned:** ground-truth points of the minority sign within `tol`; any other predicted points on the wrong side
  (a bad prediction is still scored); and, per body, either fewer than half of the points projecting inside
  `[-W/2, 1.5 W] x [-H/2, 1.5 H]` (only with intrinsics, +z forward and an explicit `image_size` key `(W, H)`), or,
  without those, a median `max(|x|, |y|) / |z|` above 3, wider than any ordinary lens. The image bounds are never
  derived from the principal point.

A world frame that happens to look like a camera frame (for example a z-up table frame with every z near 0.8 m and
points inside a plausible cone) passes. `--scene` adds the relative gauge; it needs a hand in both files.

The command prints the scores as JSON, with NaN printed as `null`. Warnings go to stderr. The exit status is 2 on
any usage error, or on degenerate or malformed input, such as wrong shapes, a non-positive `lam` or an unreadable
file.

## Before comparing two output folders: `alongray-coverage`

`alongray-coverage` reads one frozen protocol and two methods' existing output folders, and reports coverage and
conventions before any number is compared. It scores nothing and changes no scorer. The file formats are in the
docstring of `alongray.coverage_io`.

```
alongray-coverage PROTOCOL.json A_DIR B_DIR [--reference GT_DIR] [--summary] [--ids] [--boot N] [--out R.json]
alongray-coverage --write-known-answers DIR
```

The report gives:

- the flow for each method, stage by stage, with the reason for every dropped unit: assigned; annotated in the
  protocol; declared in the manifest (a unit with no status is counted, so an inner join cannot drop a failure
  silently); delivered; declared source identity (the manifest declares the protocol's sequence, camera and object;
  an ID match as text is not enough, and the physical content of a file is not checked); on the frame grid;
  delivered on annotated frames;
- the common (unit, frame) IDs of the two methods as counts, and with `--ids` as the exact list, with sequences,
  objects, cameras and clusters. Delivery over
  all assigned units is reported apart from the common IDs. Nothing is inferred about the quality of missing
  outputs; the direction of any bias they cause is unknown;
- each convention side by side (length unit, frame, camera axes, frame-index base, clock, object origin, scale,
  symmetry, registration, alignment-fit support, aggregation): MATCH, CONVERTED (declared and converted), MISMATCH
  or UNKNOWN. Undeclared conventions are never guessed. Use `frame` (mean over frames) or `sequence` (per-sequence
  frame mean, then equal weight over sequences) for `aggregation`; other strings are compared verbatim;
- screening checks on the common frames (`alongray.conventions`): unit or scale ratio, frame or transform
  discrepancy, a constant object-origin offset (flagged only when it fits better in the object frame than a constant
  offset fits in the camera frame), rotation disagreement (only when the protocol lists `object_rotation` under
  `modality.compare`), symmetry sensitivity, camera axes, 0- vs 1-based frame IDs (also on stride-1 grids), and the
  clock. Their numbers between the two methods are labelled "screening statistic, not a score";
- with `--reference`, each method is screened against the reference on the common IDs only. The report gives PASS
  or FLAGGED and the flag codes, never a value, so it holds no per-method accuracy. With a reference, a 0- vs 1-based
  shift is also chosen as the shift that minimises the discrepancy to the reference. That search compares the
  method's common frames f with whatever reference samples exist at f + k; the shifted samples need not be common
  IDs or frames the protocol scores, unlike the unshifted comparison above. The reference's own flow and
  flags (non-finite rows, camera axes) are reported;
- informational lines (`info`, which never change the verdict): `DELIVERY_DIFFERS` for any non-delivered status or
  any difference in delivery share, and `NONFINITE_ROWS`;
- a matching checklist: source frame, ground truth, clock, units, camera, object, symmetry and alignment-fit support,
  each VERIFIED, MISMATCH or UNVERIFIED;
- the methods' own per-unit scores, if supplied, re-aggregated on the common units only. For each metric both
  methods are aggregated on the same units, those for which both supplied a finite value; the others are counted,
  listed and flagged (`SCORES_NOT_FINITE_FOR_BOTH`). Equal declared frame counts do not show that two methods scored
  the same frame IDs. Each aggregate is labelled
  frame-weighted or sequence-weighted and carries the number of clusters; `--boot` adds a percentile interval that
  resamples the protocol's cluster (sequence or object). In a draw each sequence counts by the drawn share of its
  frames, so a cluster drawn k times counts k times in both weightings;
- never a speed ratio. Timing fields in the manifests are listed and ignored.

Overlapping windows that repeat a (sequence, camera, object, frame) key are refused unless the protocol declares
`overlap_policy: keep_with_multiplicity`. Then the keyed and the distinct counts are both reported. Duplicate frame
IDs inside one unit file are always refused. Unit files must lie inside their folder, and pickled arrays are refused.

Verdicts and exit status:

| verdict | exit | when |
|---|---|---|
| COMPARABLE | 0 | common IDs exist and nothing is flagged |
| COMPARABLE_WITH_FLAGS | 1 | common IDs exist; read every flag before comparing |
| (input error) | 2 | malformed, mistyped or inconsistent input, a usage error, or any internal error |
| NOT_PAIRABLE | 3 | a method has reported numbers only, or the intersection is empty |
| COUNTS_ONLY | 3 | published counts without IDs; checked for consistency, not recomputed |
| REFUSED | 4 | incompatible input modality, outputs or provenance; undeclared provenance; refused overlap |

`--write-known-answers DIR` writes 25 synthetic cases, each with `protocol.json`, folders `a/`, `b/` and `ref/`, and
`expected.json` (the verdict and the EXACT flag and info codes). Folder `a` is the truth, and `b` is the truth under
one mutation of its files in most cases; a few cases change `protocol.json` or `a`'s manifest instead, and the
counts-only case has no trajectories at all. The cases: identical, shifted object origin, mm/m swap (undeclared and declared), missing frames, frame- vs
sequence-mean aggregation, a symmetric object, wrong modality, wrong split, reported-only, counts-only, timing
fields, a world frame, OpenGL axes (undeclared and declared), 0- vs 1-based IDs (undeclared and declared, and
undeclared on a stride-1 grid), a clock at the wrong rate, overlapping windows (refused and kept), camera and object
ID collisions, asymmetric release coverage, a failed output missing from the manifest, and quaternions read in the
wrong component order. Point your own adapter at the data and check that the validator reads your folder as it
reads `a`. The thresholds of the screening checks are fixed constants in `alongray.conventions`, not measurement
confidences.

Few-percent global scale errors are NOT caught: a 3% scale gives a median ratio of 1.03 and no flag, and even a 3.0x
scale escapes `UNIT_OR_SCALE_SUSPECT` (it is flagged as `FRAME_OR_TRANSFORM_SUSPECT`). On the AGILE HO3D release,
reading the object poses without dividing by `mano_transforms.scales` (the reading in AGILE's README) against the evaluator
reading gives a back-transformed median log norm ratio of 1.0127. The numeric screening tests did not flag this
comparison. The declared scale conventions differed, so the overall verdict was COMPARABLE_WITH_FLAGS. A
mirror or a constant camera-frame offset below the discrepancy threshold is not caught either. The thresholds are
`DISCREPANCY_TOL` = 0.5 (median |p_b - p_a| over median |p_a|), so a constant 20 cm offset at 0.5 m reads COMPARABLE
(relative discrepancy 0.39), and `ROTATION_TOL_DEG` = 60 deg. For small rotations, transposing doubles the angular
discrepancy, so a transpose of rotations of about 0.33 rad gives about 38 deg and reads COMPARABLE (about 53 deg on
the known-answer rotations, also COMPARABLE). A transpose can also escape the rotation screen near a half-turn: a
170 deg rotation and its transpose are 20 deg apart. Without `--reference`, a 0- vs 1-based shift whose ids all lie on the grid (a stride-1
method that also drops its last frame) is caught only by the between-method shift screen, and only when the motion
between neighbouring frames is large against the two methods' disagreement (the shift must halve the discrepancy).
A reference check that covers no common ID, or only non-finite reference rows, reads NOT_CHECKED with
`REFERENCE_NOT_CHECKED`, never PASS. A delivered unit whose file lacks an array the protocol compares raises
`COMPARED_ARRAY_MISSING`.

## Tests

From this directory, in the activated venv:

```
uv pip install ".[test]"             # without uv: python -m pip install ".[test]"
python -m pytest tests               # tests the source tree in src/ (pyproject sets pythonpath = ["src"])
```

To test the installed copy instead, run the tests from a directory that has no `src/`:

```
mkdir ../alongray-tests && cp -r tests README.md KNOWN_ANSWERS.md examples ../alongray-tests/
printf '[pytest]\n' > ../alongray-tests/pytest.ini   # stops pytest from using a config file of an enclosing project
(cd ../alongray-tests && python -m pytest -p no:cacheprovider tests)
```

Use a new directory each time (`mkdir` refuses an existing one, so an old copy is never merged into), and delete it
afterwards. `uv pip install .` leaves a setuptools `build/` directory and `src/alongray.egg-info` here; both are
git-ignored. One `uv pip install ".[test]"` covers the quickstart's install too.

Four warnings are expected (UserWarnings about few frames and a point behind the camera, raised on purpose). In a
clean uv venv on 2026-11-09 (Python 3.12.3, numpy 2.5.4, scipy 1.18.1), both commands above printed `402 passed, 4
warnings`, in the source tree and in the installed copy; with numpy 2.1.2 the source tree gave the same. The tests
cover:

- a closed-form known answer for the split;
- the shift reading, `low_n`, the alignments and pure-gauge sensitivity, the scene gauge and the pooling, each with
  known answers;
- exactness on random trajectories;
- a pure-scale error, which lands entirely in the gauge part;
- the CLI and the quickstart's numbers (`tests/test_quickstart.py`);
- frame ids in `score` and the command line, and the unchanged no-id outputs (`tests/test_frame_ids.py`: bit for
  bit against the id-free call in the same environment. In the four recorded 0.3.1 cases, zeros must match exactly;
  a nonzero value whose recorded magnitude is below 1e-9 must keep a magnitude below 1e-9; other finite values must
  agree within eight units in the last place, since numpy builds differ by a unit in the last place on a few values.
  These are checks on fixture outputs, not bounds on every possible input); the synthetic adapter example (`tests/test_ho3d_example.py`);
- the bootstrap with zero, one and two clusters (`tests/test_pool_one_cluster.py`);
- `tests/test_coverage.py`, the coverage validator on every known-answer case.

`tools/mutate_split.py` repeats the mutant check. It mutates the modules (`core.py`, `score.py`, `cli.py`,
`pool.py`, `change.py` and the others) one way at a time, and checks that a FAILED test of the package's own suite
catches each non-equivalent mutant (129 of 129 in 0.4.0, 2026-11-09). Of the two equivalent mutants, the one that
skips bootstrap draws consuming no random bits survives, as expected; the one that changes only float rounding is now
also caught: on the known-answer case whose wrist is exact, it turns the recorded exact zero `e_par` into a rounding
residual (about 1e-16 m), and `tests/test_frame_ids.py` compares recorded zeros exactly. Run it with
`python tools/mutate_split.py` (it needs pytest; each mutant reruns the whole suite, so it takes several minutes).
`ALONGRAY_EXTRA_TESTS=<file>` (with `ALONGRAY_EXTRA_PYTHONPATH`) also runs one more test file against each mutant.

The tests use synthetic data only. The package ships no dataset content, model weights or MANO files.
`tests/test_fastdo_parity.py` ships: it checks bit-for-bit equality with the `fastdo.bench` metric modules on
synthetic fixtures, runs when `fastdo` is importable (it is inside the HandReel repository), and otherwise skips
unless `ALONGRAY_REQUIRE_FASTDO=1` makes a missing `fastdo` a failure. One further check is kept in the development
repository, not in this package: reproduction of stored development results. It needs development records, which do
not ship.

**What the fixtures do not show.** The shipped fixtures are synthetic known answers. **They reproduce no figure of
the accompanying papers, and the package makes no such claim.** The error split of `score` and `pool_splits` is
not the per-window radial share of a solve change; that statistic is `alongray.change`. The paper statistics it computes
(the median over windows of the along-ray share of a solve's change, and the Spearman correlation of paired
per-window changes in centre error and in along-ray error) were measured on development windows of DexYCB and HOI4D.
Those per-frame records are derived from licensed annotations and do not ship, so a user cannot rerun those numbers
from this package.

`KNOWN_ANSWERS.md` lists what each test fixture checks, so that the tests can serve as a known-answer pack for other
evaluators.

## Unknown outcomes: `alongray.missing_bounds`

`alongray-coverage` reports which units each method delivered and says nothing about the quality of the missing
ones. When a binary endpoint is scored on a frozen common set of N assigned, ground-truth-scorable units and some
outcomes are genuinely unknown, `alongray.missing_bounds.sensitivity_card(protocol, a, b)` (command line
`python -m alongray.missing_bounds PROTOCOL.json A.json B.json`) reports, for each method, its known passes `s` and
unknowns `u`, its pass fraction in `[s/N, (s + u)/N]`, and the difference of the two methods' pass fractions in
`[(s_A - s_B - u_B)/N, (s_A - s_B + u_A)/N]`. Both intervals are sharp over every binary completion of the unknowns;
`tests/test_missing_bounds.py` checks the endpoints exactly against every completion for N <= 4. Known delivery
failures count as the protocol says (`delivery_failure_counts_as`, which must be declared, as `"fail"`) and are never
turned into unknowns; non-scorable ground truth is listed and kept out of N; duplicate ids, a unit definition or
endpoint that differs from the protocol's, statuses that do not cover exactly the scorable units and undefined
statuses are refused. These are finite-cohort sensitivity bounds, **not confidence intervals and not rankings**, and
they can be uninformatively wide. `application_gate` reads NOT_ESTIMABLE below 20 common scorable units or without
any genuine unknown. Partial-identification precedent: Horowitz and Manski (2006),
https://www.sciencedirect.com/science/article/pii/S030440760500062X .

## Licence of the data you score

alongray ships no data. The outputs and annotations it is run on keep their own terms; for the datasets used in its
development:

- DexYCB: CC BY-NC 4.0.
- HOI4D: CC BY-NC 4.0.
- HO3D-v3: academic, non-commercial research only; redistribution of any part of the dataset requires express
  permission from its authors.
- Released reconstructions of other methods carry their authors' licences (AGILE's, for example, is a
  non-commercial research licence). A method's code licence does not cover outputs that depend on model weights
  under other terms.

Before you share anything, raw outputs, derived numbers, per-frame records or figures, check the terms that apply
to it: the dataset's licence and terms of use, the licences of the models and weights that produced the
predictions, and any terms on derived outputs. Some of these restrict redistribution or commercial use of derived
material as well. Nothing in this package, its documentation or its licence grants any right to a dataset, a
model, or outputs derived from them; the Apache-2.0 licence covers this package's own code only.

## How to cite

Use `CITATION.cff`, and cite the accompanying paper as the repository's `CITATION.cff` gives it.

## Licence

Apache-2.0 (see `LICENSE` and `NOTICE`). Copyright 2026 The alongray authors. Each source file carries an SPDX
licence identifier.
