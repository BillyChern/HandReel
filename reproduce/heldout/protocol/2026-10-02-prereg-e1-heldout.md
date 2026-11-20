# E1 pre-registration — held-out paired quality comparison, ours vs DO AS I DO, on HOI4D FINAL_TEST and DexYCB subject 08

**Registered:** 2026-10-01T18:27:28Z (`date -u`; the commit that adds this file to main is the registration record)

Status: **REGISTERED**. Drafted 2026-10-01T16:38Z as a draft (internal record, not released),
and finalised to the authors' decisions of 2026-10-01 ~17:05Z. Registration precedes the draw of the
subject-08 units, every DO AS I DO run on a held-out unit (from 2026-10-08) and every run of ours on a held-out unit
(after the freeze). After registration nothing below changes except by a dated, labelled amendment committed before the
change takes effect; no amendment may depend on any E1 output of either method (section 12).

Authority:
- Rulings of 2026-10-01T16:15Z. **R-A** DO AS I DO runs on the final sets SEALED from 2026-10-08; nobody scores or
  reads the outputs until after our code/result freeze (~2026-10-26). **R-B** non-inferiority margin **1.15x**, n larger
  than 36 HOI4D + 24 DexYCB, honest power calculation. **GPU priority:** if E1 + arm D + the freeze headline do not fit,
  arm D shrinks first; E1 keeps full n.
- Rulings of 2026-10-01 ~17:05Z (binding). **U1** proceed at 1.15x although joint power at parity is ~0.35; the paper
  reports per-corpus paired ratios with 95% CIs whatever the verdict. **U2** PRIMARY arm for us = ESTIMATED intrinsics
  (same condition for both methods); calibrated-K arm SECONDARY. **U3** a THIRD DexYCB view per subject-08 sequence
  (n_D = 3R); recompute power, budget and cap. **U4** defaults: (a) a QC refusal or undelivered output is OUR failure in
  ITT (the convention of an internal record); (b) during sealed runs automated code may read held-out object NAMES/LABELS only, to prompt
  BOTH pipelines identically — never poses, GT trajectories or scores — and every read is logged; (c) the
  simulation-calibrated thresholds, with an exact cluster sign test as a reported sensitivity analysis; (d) the
  untracked head-to-head harness is ported into committed run/study/e1_*.py before registration.

## 0. What of the held-out sets has been seen before registration

- HOI4D FINAL_TEST: only the committed HOI4D split manifest (internal record, not released), and of it only the fields
  `pool`, `clip_id`, `sequence`, `category`, `instance`, `rank`, `start`, `stop`, `shift_px`. No HOI4D file of a
  FINAL_TEST sequence was opened.
- DexYCB subject 08: only directory and file NAMES under data/dexycb/20201002-subject-08 (100 sequence ids, 8 serial
  ids, `color_*.jpg` counts per serial directory; a names record (internal record, not released), written by `e1_select.py --names`,
  which opens no file content — tested), plus this project's own 2026-09-13 motion-prior split record (names
  only; internal record, not released). No meta.yml, pose.npz, labels_*.npz, image or calibration file of subject 08 was opened.
- Every number in sections 7 and 8 comes from development data.

## 1. Question and claim

Is our production pipeline at the freeze tree **non-inferior** to DO AS I DO, as released, on held-out labelled video,
in per-clip object-centre error, with margin 1.15x (our error may be at most 15% larger), when both methods run at the
internet-video deployment condition (estimated intrinsics)? Two corpora are tested separately: **H** = HOI4D FINAL_TEST
(egocentric, moving camera) and **D** = DexYCB subject 08 (third-person lab). The paper may say "non-inferior within 15%
on held-out <corpus>" only for a corpus that passes section 6.4. Whatever the verdicts, the paper reports each corpus'
paired median ratio with its 95% CI (U1). Until E1 reports, the parity statement stays scoped to the 6 h2h development
clips (freeze plan s. 3).

## 2. Units (deterministic; from manifests and file names; run/study/e1_select.py)

### 2.1 H — HOI4D FINAL_TEST: all 72 windows

- Universe: every entry of the split manifest's `windows` (section 0) with `pool == "final_test"` (72 windows,
  12 per category ToyCar/Mug/Bottle/Bowl/Kettle/Knife, 55 object instances: 38 with one window, 17 with two).
- No exclusion. Every window passed the split's GT gate when the split was committed (M332), before any pipeline output.
- n_H = 72 is the ceiling of the committed split. Growing it would need GT-gate reads of new HOI4D sequences, which this
  study does not do.
- Unit = window `[start, stop)`, 150 source frames at 15 fps. Cluster = object instance (category, instance); stratum =
  category.

### 2.2 D — DexYCB subject 08, untouched sequences, right hand, THREE serials per sequence (U3)

- **TOUCHED (excluded entirely, 28 of 100 sequences, M341).** Sort the 100 sequence directory names of
  `20201002-subject-08`. TOUCHED = the names at sorted index % 4 == 0 (25: the 2026-09-13 motion-prior calibration split,
  whose GT object poses were read) ∪ {20201002_105803, 20201002_110227, 20201002_110940} (the 2026-09-13 realdiag
  `newsubj` runs). 28 names, sha256 of the newline-joined sorted list
  `0f0f875725acd12ed4eb268e5457286e492eebb35a22cc1e98fae87389fae09d` (the names record, section 0). 72 remain.
- **Right hand.** Of the 72, keep those with meta.yml `mano_sides == ['right']` (a label field, read by
  `e1_select.py --draw` only, which refuses to run until this file is committed with its Registered stamp; each read is
  logged). Production anchors on the right hand (an internal record); DO AS I DO runs with `anchor_hand right`. Call the kept count R
  (subject 07 had 50/100 right; R ≈ 36 is the planning value).
- **Serials.** S = the sorted serial directory names minus 836212060125 (7: 839512060362, 840412060917, 841412060263,
  932122060857, 932122060861, 932122061900, 932122062010). The i-th kept sequence in sorted order (i = 0..R-1) takes
  S[(3i) mod 7], S[(3i+1) mod 7], S[(3i+2) mod 7]: three views per sequence, balanced across cameras.
- **Input eligibility (before any run, inputs only).** A chosen view whose decode-row-0 keyframe prompt the bench cannot
  build (`bench_clips.dexycb_prompt`'s refusal on the GT object label of frame 0, the s255 study's "prompt-unbuildable") is
  replaced by S[(3i+3) mod 7], then S[(3i+4) mod 7], ..., until three eligible views or none remain. Replacements are
  listed. This is a property of the shared input, not of either method.
- n_D = 3R (planning: 108; 4.5x the 24 first proposed). Unit = the whole clip (68–74 frames at 30 fps; mean 72.61 over the
  504 UNTOUCHED non-836212060125 serial directories, from file names). Cluster = sequence. No strata.
- The draw writes the units file (released, reduced, as ../units_*.json) (units, clusters, replacements, the run order, input hashes, the code
  SHA, the Registered stamp) and it is committed before the first run.

### 2.3 Run order

Seeded permutation (numpy default_rng(2027)) within each corpus; the two queues are merged by fractional position
(rank + 0.5)/n_c, ties H first, so at any stop point both corpora have completed the same random fraction
(`e1_select.run_order`, tested).

## 3. Pipelines

### 3.1 Ours (O) — current production at the freeze tree (run/study/e1_ours.py)

- Code: the commit named by the freeze manifest (freeze plan item 5, Gate A, ~2026-10-26), run from a detached worktree
  pinned by `git rev-parse HEAD` of that commit (never by the name main); `e1_ours.py --mode final` refuses unless
  `--pin` equals HEAD and src/, run/conf, run/pipeline are clean. The SHA is entered by amendment A-F before ours runs;
  A-F may not change anything else.
- Config: run/conf/config.yaml at that commit, `qc_stage.rule_path` = the production AcceptanceRule
  (outputs/qc/acceptance_rule.json, sha256 e564b52e92d3…), with the source-domain profile enforced in code (an internal record, M346):
  H → egocentric (contact OFF, V2 dead-frame branch OFF); D → third-person lab (contact ON as M88, full V2). Deterministic
  hands (M350); defaults as merged (M352). Shape = our own SAM 3D mesh (`bench.dexycb_shape: model`), never the GT mesh.
- Entry point: upstream (decode, mask, shape) through the registered bench entry point (run/pipeline/bench_throughput.py),
  then, per arm, the registered stage workers (`bench_stages.build_stage_fns`) on a fresh copy of each unit's upstream:
  intrinsics → hands → geometry → tracks → solve → contact (D) → qc. Decode at the operating stride: H 2 (75 rows),
  D 3 (23–25 rows). Keyframe prompt = the GT object LABEL of decode row 0 (the bench default for both corpora).
- **PRIMARY arm O-EST (U2):** camera.json removed from the arm's copy of the upstream; intrinsics from the production
  intrinsics stage (MoGe-2), exactly the EST construction of an internal study. Each method uses its own native intrinsic estimator;
  neither is given calibration. **SECONDARY arm O-CAL:** camera.json kept (the shipped calibration).
- **Delivered (U4a)** = the pipeline wrote solve.npz AND QC accepted the clip (the internal convention above: production delivers
  only on acceptance). Undelivered for any reason (prompt refusal, the no-right-hand drop, a stage's ValueError /
  FloatingPointError refusal, no solve, QC rejection, missing output) is OUR failure in ITT. A CUDA / OOM error is an
  infrastructure fault: one in-process retry, then the unit is re-run before scoring; it is never a method failure.

### 3.2 Theirs (T) — DO AS I DO as released, on every source frame (run/study/e1_theirs.py)

- Code: third_party/do-as-i-do at **824591b808c342b20079c3b4198a8c2bdf88c74e** (MIT), never modified: every stage is
  reconstruction/run_pipeline.sh's command with its cwd and flags, run with each env's absolute interpreter (envs as built
  by run/setup/build_baseline_envs.sh; pip-freeze hashes recorded per unit; `git status` of third_party recorded before
  and after every unit and a change fails the unit). Infrastructure reused from the 2026-09 h2h runs: the flash_attn and
  diff_gaussian_rasterization builds (the pylib and pylib_dgr directories), the Fast-SAM3D checkpoint staging
  (fastsam3d_hf), the ffmpeg PATH shim for TAPIR; HF_HUB_OFFLINE=1.
- Stages, in run_pipeline.sh's order: 0 frames (ffmpeg) + config.json + reference frame; 1 SAM 3 object (points, see
  substitution 1) and anchor hand (`--text "right hand"`); 2 SAM 3D mesh of the reference frame, MoGe pointmap of the
  reference frame, **HaWoR** (`demo.py --vis_mode cam --img_focal <reference MoGe focal> --static_camera`), MoGe
  pointmaps of all frames, GeoCalib gravity; 2.5 TAPIR; 3 track_object (identical flags) and
  convert_layout_to_camera_frame; 4 optimize_translation_scale. Not run: run_project_mesh_combined (a visualization whose
  outputs nothing downstream reads) and the commented-out smoothing and viser steps.
- Input: H = the HOI4D source video (align_rgb/image.mp4) trimmed frame-exactly to `[start, stop)` and re-encoded libx264
  crf 12 (decoded count must be 150); D = the clip's `color_*.jpg` encoded libx264 crf 12, 30 fps, every frame (decoded
  count must equal the jpg count). Every source frame is tracked.
- Harness substitutions (the only ones):
  1. **Object prompt.** The released `--click` is interactive. It is replaced by 3 positive points sampled from the GT
     object LABEL at the reference frame (distance-transform maximum, then farthest-point sampling over pixels at least
     half as deep; the h2h sampler, ported). obj_id = `object` (no class name is passed). Ours gets the same kind of input
     (a GT object label at one frame).
  2. **Reference frame N′ (automated rule for missed hand masks).** Their run fails at Stage 4 if the anchor-hand mask is
     empty or gives < 10 ray hits on the HaWoR mesh at the reference frame (optimize_translation_scale.py:333-357).
     - N0 = local frame 0 (= our decode row 0). Candidates = our decode rows ≥ N0 in order (H stride 2, D stride 3).
     - **Scout:** their Stage 1 hand pass as released at N0 (`run_sam3_video.py --text "right hand" --frame_idx N0`).
     - N′₁ = the first candidate where (i) the GT object label has ≥ 1000 px (D) or ≥ 500 px (H, the bench GATE
       `min_prompt_px`), and (ii) their own scout right-hand mask is non-empty. If N′₁ = N0 the scout's run continues.
     - Run Stages 0–2 at n = N′, then a **preflight** that executes their own reference-frame check (their
       `load_pointmap_and_intrinsics`, `load_mask`, `raycast_first_hits` imported read-only from their scripts, rng =
       default_rng(0), ≥ 10 hits, nonzero pointmap depth) before Stage 2.5. If it fails, take the next candidate meeting
       (i)–(ii) and rerun Stages 0–2. At most 3 candidates; exhaustion = failure `NPRIME_EXHAUSTED`.
     - Per-frame hand-mask misses after N′ are handled by their own code; nothing is changed there.
     - The rule reads their masks and GT labels only through code, as pixel COUNTS (section 9).
  3. **Headless.** No viser or GUI; HaWoR with the released flags (`--static_camera` on moving-camera HOI4D is an
     asymmetry against them, reported, not changed).
  4. **Memory.** No artificial GPU-memory cap (the 2026-09-20 ruling lifted per-job size caps); one retry after an OOM.
- Output: `layout_camera_frame_optimized.json` + the reference-frame mesh, converted in the runner (the h2h
  convert_layout formula, ported: poses, mesh_scale, frame_idx) to `doasido.npz`, then packed and sealed (section 9).
- Delivered = a converted trajectory covering ≥ 1 evaluation frame. Failure classes: NPRIME_EXHAUSTED, a stage's
  non-zero exit after the infrastructure retry, timeout (3x the unit's expected wall: H 3.6 h, D 2.0 h), missing
  output. Infrastructure faults (OOM, host crash, disk guard) get one retry and are not method failures unless the retry
  fails the same way.

### 3.3 Asymmetries (stated in the paper whatever the result)

- In their favour: N′ moves their reference frame to where their own hand detector fires (a user would do this);
  every source frame is tracked (2x our frames on H, 3x on D).
- Neutral: both methods use HaWoR hands (unlike the h2h, where they had GT hands) and estimated intrinsics in the primary.
- Against them: re-encoded input (crf 12) while ours reads the JPEGs / decodes the source; `--static_camera` on
  egocentric H; their MoGe v1 vs our MoGe-2.
- Against us: QC rejection counts as our failure; our contact stage is ON for D (production third-person profile).

## 4. Evaluation frames and the primary metric (run/study/e1_score.py)

- Evaluation frames E(u) = our decode rows of unit u (H: start, start+2, … < stop; D: 0, 3, 6, …) ∩ GT-valid frames
  (H: objpose valid; D: pose valid). The same E(u) for both methods and both of our arms.
- **Object-centre error** e_m(u) = mean over f ∈ E(u) of ‖c_m(f) − c_GT(f)‖ in mm, where c_GT(f) = R_GT(f)·c̄_model +
  t_GT(f), c̄_model = the centroid of 10,000 area-uniform samples (seed 0) of the GT CAD/YCB model, and
  c_m(f) = s_m·R_m(f)·c̄_mesh,m + t_m(f), c̄_mesh,m = the same centroid of method m's OWN mesh (frame-free; qc23's
  `post_centroid_mm`, the h2h `centroid_mm`). Ours is scored on its FINAL trajectory (after contact where the profile
  runs it).
- A method's missing frame in E(u) (no pose, non-finite, or behind the camera) takes the pose of the nearest delivered
  frame in time (earlier on ties). The imputed fraction is reported per method.
- **Pre-declared bad value (ITT):** if only ours is undelivered, d(u) = +∞; if only theirs, d(u) = −∞; if both,
  d(u) = 0 (implemented as ±1e6). Because the endpoint is a median, only this ordering matters.

## 5. Endpoints

- **Primary (per corpus, ITT, O-EST vs T):** the paired median log-ratio M_c = median over units u of corpus c of
  d(u) = log(e_O-EST(u) / e_T(u)), with the bad values of section 4. NOT the ratio of medians.
- Secondary (reported, not gated; no α is spent on them):
  - S1 complete-case M_c (both delivered); and ITT with QC refusals scored on our written trajectory (yield vs accuracy).
  - **S2 O-CAL vs T**, the same statistic, ITT and complete-case (U2: secondary).
  - S3 F-5 metrics (strict, icp_frame, icp_clip; the h2h protocols), paired median differences, same bootstrap.
  - S4 metric-scale error |log(s_m / s_a,m)|; paired.
  - S5 rotation error: per-frame geodesic angle of the RELATIVE rotation R_m(f)R_m(f0)ᵀ vs R_GT(f)R_GT(f0)ᵀ, clip median;
    paired. Rotationally symmetric objects are flagged.
  - S6 counts: good (e ≤ 50 mm), tail (e > 150 mm), wins (e_O < e_T), with cluster-bootstrap CIs.
  - S7 strata: H by category and by camera/person; D by serial.
  - S8 delivery: failure rates and classes per method; N′ − N0 distribution; imputed-frame fractions.
  - S9 their dense-frame score: e_T on ALL GT-valid source frames vs on E(u).
  - S10 cost: per-unit wall of each method on this card (descriptive; not the speed headline).
  - **S11 exact cluster sign test (U4c sensitivity):** per cluster, the median of its units' d; k = the number of clusters
    with that median < log 1.15, n = clusters whose median ≠ log 1.15; one-sided exact binomial p = P(Bin(n, ½) ≥ k).
    Reported next to the primary for every corpus; it never changes a verdict.

## 6. Analysis (run/study/e1_stats.py)

1. Per corpus: d(u) for all RUN units (section 10 defines NOT RUN); M_c as above.
2. **Clustered bootstrap:** B = 10,000, numpy default_rng(1); resample clusters with replacement (H: instances within
   category; D: sequences); exact median of the resampled multiset. One-sided p_c = (1 + #{M*_c ≥ log 1.15}) / (B + 1).
3. **Holm over the two corpora only (the primary family is {H, D} for O-EST; S2 is outside it), on size-calibrated
   thresholds** (family α = 0.05). The percentile cluster bootstrap of a median is anti-conservative with few clusters:
   at a true ratio of exactly 1.15, on development-like data with the registered design, it rejected 5.95–6.33% of D
   replicates and 4.5–4.6% of H replicates at a nominal 5% (`e1_power.py --size-only`, 4,000 replicates × 2,000 draws
   per scale, seed 20261003). The registered thresholds are the null p quantiles at each level, minimised over the three
   scales of section 8.2: **a_H(0.025) = 0.0250, a_H(0.05) = 0.0500; a_D(0.025) = 0.0180, a_D(0.05) = 0.0380**
   (released as ../e1-power-size.json). Corpus c passes if p_c ≤ a_c(0.025), or if p_c ≤ a_c(0.05) and the other corpus
   has p ≤ its a(0.025). Reported for each corpus: the plain percentile 95% UB and the calibrated UB (the 1 − a_c(0.05)
   percentile), both against log 1.15 = 0.13976. If the realised R differs from 36 by more than 6, `--size-only` is
   rerun with `--n-dex-seq R --dex-views 3` before the first run and its thresholds replace these (a mechanical,
   outcome-free amendment).
4. **Verdict per corpus:** NON-INFERIOR (passes 3); INFERIOR (the one-sided 95% LOWER bound of M_c > log 1.15);
   otherwise INCONCLUSIVE. INCOMPLETE replaces the verdict if fewer than 80% of the corpus' units RAN (section 10).
5. Hierarchical, only for a NON-INFERIOR corpus: LOWER ERROR is stated if its calibrated UB at the level it passed at is
   < 0 (fixed sequence after non-inferiority; no α is spent).
6. Reported with every verdict, ITT and complete-case side by side (never complete-case alone): n assigned / run /
   delivered per method, M_c with its two-sided 95% CI, exp(M_c) as a ratio with its CI (U1), the per-cluster count, the
   sign test (S11), and the same rows for O-CAL (S2).

## 7. Known answers (development data only; run before registration unless marked)

Run 2026-10-01 on the registration tree, development data only (a smoke-run directory; the 6 h2h DexYCB clips and one HOI4D
DEV window; no subject 08, no HOI4D FINAL_TEST). Tolerance for KA-S / KA-O: 0.1 mm per clip.

- **KA-S (scorer, theirs) — PASS.** `e1_score.py --ka` on the stored s219 DO AS I DO h2h outputs reproduces every
  per-clip `centroid_mm` to ≤ 5.3e-13 mm (6/6); 6-clip mean **50.479 mm** (M341: 50.5). On all 6 clips the E1 frame rule
  (decode rows ∩ GT-valid) gives exactly the h2h frames; no frame is missing, so imputation changes nothing.
- **KA-O (scorer, ours) — PASS.** On the stored s255 V2 h2h outputs (18 = 6 clips × 3 replicates) every value is
  reproduced to ≤ 6.0e-13 mm; mean of replicate means **50.154 mm** (M341: 50.2). Paired d through e1_stats
  (descriptive): median −0.293, sign test 4/2.
- **KA-N (N′ rule) — PASS.** Started at the h2h N_run frames on the stored W1 masks, the E1 scan re-derives ruling 7's N′
  (105903 18→33, 135250 0→21, 135545 3→30) and leaves 104655, 112215, 135841 unmoved (6/6).
- **KA-R (runner + seal round trip), DexYCB — PASS.** `e1_theirs.py --mode smoke` on one DexYCB development unit
  (HaWoR hands, their own): rc 0, N′ 0 → 9 on the first attempt (scout right-hand mask empty on rows 0–6; preflight 1,879
  hits), wall 2,614 s (s3_track 2,170 s), peak 26.0 GiB, 0.73 GPU-h; third_party unchanged before/after; pack → SEAL →
  verify → unseal OK (archive 460 KB, work dir deleted); `e1_score.py --mode smoke --score` gives **e_T = 51.05 mm**
  (25 rows, none imputed) vs 52.01 mm in the h2h (GT hands then; not a pass criterion).
- **KA-P (preflight equivalence) — PASS on the one unit run** (preflight pass, Stage 4 logged no reference-frame error);
  only the positive case has been exercised.
- **Ours runner (`e1_ours.py --mode smoke`), 6 h2h clips + 1 HOI4D DEV window — ran, 14/14 arm-units rc 0.** This is
  production with OUR OWN SAM 3D mesh and no track jitter, so it is NOT a re-run of the h2h ours rows (those used DO AS I
  DO's template mesh, M306); it is the freeze-plan item-1 measurement and a harness check. DexYCB, written trajectories:
  O-CAL mean 48.3 mm (median 50.0; delivered 5/6, one QC-rejected), O-EST mean 81.5 mm (median 90.2; delivered 4/6,
  two QC-rejected) — vs M306's 113.6 / 138.1 mm on the template-shape harness. On the clip of the
  KA-R run: O-EST 50.0 vs theirs 51.05 mm. HOI4D DEV window: O-EST 95.8, O-CAL 93.5 mm. These are SEEN development
  numbers: descriptive only, never a planning δ.
- **Not yet run (must pass before 2026-10-08; a failure blocks launch):** KA-R on one HOI4D DEV window (theirs, ~1.35
  GPU-h; the H path has passed its dry run and a frame-exact trim check only).

## 8. Power (development data only; run/study/e1_power.py)

### 8.1 Data (sha256 recorded in the power record, internal record, not released)

`python run/study/e1_power.py --size-only && python run/study/e1_power.py` (defaults `--n-dex-seq 36 --dex-views 3`)
writes a scratch directory; the two JSON results are committed as the size record (released as ../e1-power-size.json)
and the power record (internal, not released; they replace the draft's two-view files).

| source | what | role |
|---|---|---|
| the s255 confirmation scores (internal) | DexYCB subject 07 (s255, spent one-shot), V2 `post_centroid_mm`, 332 clips / 50 sequences | shape and clustering of D-like errors |
| the s253 DEV-72 and DEV-2 scores (internal) | HOI4D DEV-72 + DEV-2 (s253), V2 (contact off), 136 windows / 95 instances | shape and clustering of H-like errors |
| the s255 h2h centroids + the s219 summary (internal) | the 6 h2h dev clips: ours V2 vs DO AS I DO (GT hands), centroid | the ONLY paired dispersion |
| the HOI4D split manifest (section 0) | FINAL_TEST pool: category/instance/rank only | H cluster design (38 singletons + 17 pairs) |

Development statistics (log of per-clip error; robust sd = 1.4826 MAD):

| data | units / clusters | median error | robust sd | sd | ICC(1) within cluster |
|---|---|---|---|---|---|
| D-like: subject 07 V2 (CAL, contact on) | 332 / 50 sequences | 71.5 mm | 0.573 | 0.553 | 0.156 |
| H-like: HOI4D DEV-72 + DEV-2 V2 (contact off) | 136 / 95 instances | 58.6 mm | 0.764 | 0.705 | 0.221 |
| h2h paired d = log(ours V2 / DO AS I DO), 6 clips | 6 / 6 | d median −0.293 | 0.262 | 0.451 | — |

h2h d per clip: 104655 +0.235, 112215 −0.497, 135841 +0.639, 105903 −0.365, 135250 −0.221, 135545 −0.442. These are SEEN
data (V2 was designed on 135841, M325), DO AS I DO had GT hands there and ours ran CAL, so the h2h median is NOT used as
the planning δ; only its dispersion is. The low within-sequence ICC on D (0.16) means a third view of a grasp adds nearly
an independent unit (design effect 1.31 at 3 views vs 1.16 at 2).

### 8.2 Model

d(u) = δ + s·z(u). The z are OUR dev log errors, standardised (minus the median, divided by 1.4826·MAD), drawn by
resampling whole dev clusters (D: 3 views of one subject-07 sequence; H: instance clusters within category, matching the
FINAL_TEST sizes), so the dev tails and within-cluster correlation are kept. s is the robust scale of d:
- **s = h2h MAD 0.262** (optimistic), **s = h2h SD 0.451 (planning value)**,
- **s = ρ0 bound** = √2 × ours' dev robust sd (0.810 for D, 1.080 for H): theirs as dispersed as ours and uncorrelated
  with ours (pessimistic: both methods take metric scale from HaWoR hand depth and share the along-ray ambiguity).
- True ratio exp(δ) ∈ {1.00, 1.05, 1.10}. Failures: none, or 3% per method independently (d = ±10, or 0 if both).
- Test exactly as section 6; 2,000 replicates × 2,000 bootstrap draws (MC SE ≤ 1.1 points); trade-off rows 1,000 ×
  1,000 (≤ 1.6 points). Seed 20261002; the size check uses 20261003.
- The dev data are CAL-condition errors of ours; the EST primary's dispersion is not separately modelled (s already spans
  0.26–1.08).

### 8.3 Power at the chosen n (n_H = 72, n_D = 108 = 36 sequences × 3), Holm-adjusted

Power = P(verdict NON-INFERIOR) per corpus under Holm on the calibrated thresholds; "both" = both corpora pass.
2,000 replicates (MC SE ≤ 1.1 points). The power record (internal, not released).

| scale s of d | failures | true ratio | H | D | both | median 95% UB H / D |
|---|---|---|---|---|---|---|
| h2h MAD 0.262 (optimistic) | none | 1.00 | 0.94 | 0.99 | 0.93 | 0.070 / 0.050 |
| | | 1.05 | 0.60 | 0.82 | 0.54 | 0.119 / 0.099 |
| | | 1.10 | 0.22 | 0.30 | 0.10 | 0.165 / 0.145 |
| **h2h SD 0.451 (planning)** | none | **1.00** | **0.53** | **0.68** | **0.41** | 0.120 / 0.090 |
| | | **1.05** | **0.22** | **0.39** | **0.13** | 0.171 / 0.134 |
| | | **1.10** | **0.09** | **0.12** | **0.02** | 0.216 / 0.185 |
| ρ0 bound (H 1.080, D 0.810) | none | 1.00 | 0.18 | 0.31 | 0.09 | 0.292 / 0.152 |
| | | 1.05 | 0.09 | 0.15 | 0.03 | 0.337 / 0.207 |
| | | 1.10 | 0.04 | 0.07 | 0.01 | 0.384 / 0.250 |
| h2h SD 0.451 | 3% each | 1.00 | 0.50 | 0.65 | 0.38 | 0.125 / 0.091 |
| | | 1.05 | 0.22 | 0.36 | 0.12 | 0.175 / 0.135 |
| | | 1.10 | 0.09 | 0.12 | 0.02 | 0.221 / 0.187 |

The margin is log 1.15 = 0.140. At a true ratio of exactly 1.15 the calibrated test rejects ≤ 5% of simulated replicates
at each of the three scales, by construction of the thresholds (section 6.3); its size under an error structure unlike
the development data is not guaranteed.

### 8.4 Trade-off (single corpus, no failures; power at a_c(0.05) / at a_c(0.025) = Holm's first step)

1,000 replicates × 1,000 draws (MC SE ≤ 1.6 points). Thresholds are the chosen design's; for other n they are
approximate (the percentile bootstrap is probably more anti-conservative at smaller n, so read non-chosen rows as upper
bounds).

| design | n | GPU-h theirs (instructed / fixed-cost) | s | 1.00 | 1.05 | 1.10 |
|---|---|---|---|---|---|---|
| H, 6 per category | 36 | 40.2 / 42.7 | h2h SD | 0.41 / 0.32 | 0.27 / 0.19 | 0.17 / 0.11 |
| | | | ρ0 | 0.25 / 0.18 | 0.18 / 0.12 | 0.12 / 0.08 |
| H, 8 per category | 48 | 53.6 / 57.0 | h2h SD | 0.45 / 0.35 | 0.27 / 0.17 | 0.14 / 0.10 |
| | | | ρ0 | 0.20 / 0.15 | 0.12 / 0.07 | 0.08 / 0.05 |
| **H, all (chosen)** | **72** | **80.4 / 85.5** | h2h SD | 0.54 / 0.40 | 0.26 / 0.18 | 0.16 / 0.09 |
| | | | ρ0 | 0.25 / 0.16 | 0.14 / 0.08 | 0.07 / 0.03 |
| D, 1 view per sequence | 36 | 19.5 / 25.7 | h2h SD | 0.40 / 0.29 | 0.26 / 0.17 | 0.14 / 0.08 |
| | | | ρ0 | 0.20 / 0.11 | 0.14 / 0.08 | 0.07 / 0.04 |
| D, 2 views (the draft) | 72 | 38.9 / 51.4 | h2h SD | 0.60 / 0.49 | 0.41 / 0.31 | 0.16 / 0.09 |
| | | | ρ0 | 0.32 / 0.22 | 0.20 / 0.13 | 0.10 / 0.06 |
| **D, 3 views (chosen, U3)** | **108** | **58.4 / 77.1** | h2h SD | 0.78 / 0.69 | 0.47 / 0.34 | 0.20 / 0.13 |
| | | | ρ0 | 0.41 / 0.29 | 0.23 / 0.16 | 0.12 / 0.07 |

### 8.5 Reading

- **E1 at the chosen n is powered for parity only if the two methods' errors are at least as correlated as on the h2h
  clips.** At the planning scale the chance that BOTH corpora pass at a true ratio of 1.00 is 0.41 (H 0.53, D 0.68; the
  draft's two-view design gave 0.35 with D 0.55); at 1.05 and 1.10, inside the margin, most outcomes will be
  INCONCLUSIVE. Under the pessimistic ρ0 bound E1 can rarely pass. U1 accepts this; the paper states it and reports the
  per-corpus ratio with its CI whatever the outcome.
- The planning scale rests on 6 seen clips with GT hands for DO AS I DO; its own 90% upper confidence limit (0.79) is near
  the D ρ0 bound. The power figures are therefore a range, not a number.
- The third view (U3) raises D's power at parity from 0.55 to 0.68 under Holm (planning scale) for +19.5–25.7 GPU-h.
  H cannot grow inside the split; 80% power at parity on H would need about 2x its information.
- A 3% failure rate on each side costs 2–3 points at parity.

## 9. Sealing (R-A) and the sealed-phase guards (run/study/e1_seal.py, e1_common.py)

- **Directory:** <host path> (gitignored). The E1 runner is its only writer. Each
  unit is packed after it finishes into `<unit>.tar.zst` (the outputs of 9.3, the N′ record, the stage logs), and
  `SEAL.sha256` gets one appended line: file name, size, sha256, UTC time.
- **Status, the only thing visible before the freeze:** the status log status.jsonl, one line per unit: unit id, corpus, rc,
  failure class (fixed regexes on exit codes and error strings), N′ attempt count, stage reached, wall s, peak GPU memory.
  No pose, scale, error, frame index or image-derived number (tested).
- **Tamper evidence:** the day's SEAL.sha256 lines (names, sizes, hashes; no content) and the day's label-read log lines
  are committed to git each day (a seal ledger and a label-read log; internal record, not released).
- **Who may read what, when:** before the unseal, nobody (person or automated process) opens a sealed file. Exception: for a unit
  whose failure class is `infra-unknown`, the operator may read the last 50 lines of its stderr to decide the
  infrastructure retry, through `e1_seal.read_stderr_tail`, which logs the access (time, reason). Their stdout (per-frame
  scales and residuals) is never displayed.
- **Label reads (U4b), enforced in code:** automated code may read held-out object NAMES and LABELS (DexYCB meta.yml label
  fields and the `seg` member of labels_*.npz; HOI4D 2Dseg object labels) to build the SAME kind of prompt for both
  pipelines (our decode-row-0 prompt; their reference-frame points and N′ condition (i)) and for the draw's right-hand and
  eligibility rules. Every read is appended to the label-read log label_reads.jsonl (UTC, unit, file, member, purpose);
  `e1_common.check_label_path` refuses pose files (pose.npz, objpose/, Hand_pose/). GT poses, GT trajectories and scores
  of a held-out unit are read only through `e1_common.require_gt_pose_access`, which raises `SealedPhaseError` until the
  unseal record exists and `HeldOutRefused` in the development layout (tested: the scorer refuses before opening any GT
  file).
- **Unseal:** only after (a) the freeze manifest is committed (Gate A), (b) every ours-side E1 run is complete, and (c) the
  scorer and analysis are committed and the known answers pass. `e1_seal.unseal` verifies every hash against the
  committed ledger (a mismatch voids that unit, listed), extracts, and only then writes UNSEALED.json, the record
  the GT guard reads. Scoring is ONE pass of the committed scorer over both methods; the summary prints section 6.

### 9.3 Hygiene (no pixels kept)

- Their run directory holds extracted PNG frames, per-frame pointmaps, SAM 3 masks, HaWoR renders and overlays. After the
  unit finishes the runner keeps only: config.json, the N′ record (frame ids and pixel COUNTS), layout_camera_frame.json,
  layout_camera_frame_optimized.json, the object mesh (.obj), gravity.json, the logs, run_meta.json and doasido.npz;
  everything else, including the exported input video, is deleted before the next unit starts.
- Ours: per-unit arm directories keep the npz/json artifacts the scorer reads; the decode.npz pixel arrays (upstream and
  arm copies) are deleted after the scoring pass. No HOI4D/DexYCB pixel is copied into the project's records or the paper.
- Disk: a unit starts only if free disk minus a 15 GB reservation stays ≥ 175 GB (`e1_common.disk_ok`).

## 10. GPU budget, cap, schedule and stop rules

### 10.1 Arithmetic (with the third view)

Theirs, as instructed (measured wall 26.81 s per source frame on this card, shared H100 with a ~19 GB tenant, M152):
- H: 72 windows × 150 frames = 10,800 frames × 26.81 s = 289,548 s = **80.4 GPU-h**.
- D: 108 clips × 72.61 frames = 7,842 frames × 26.81 s = 210,241 s = **58.4 GPU-h** (two views: 38.9).
- N′: the scout pass is their own Stage 1 hand pass, reused when N′ = N0; a moved N′ reruns Stages 0–2 (~525 s). Assume
  every unit pays one extra hand pass (180 × ~55 s = 2.8 GPU-h) and 15% of units one rerun (27 × 525 s = 3.9 GPU-h).
- **Theirs total ≈ 145.5 GPU-h.**
- Fixed-cost sensitivity: 26.81 s/frame is the gated-stage wall of a 138-frame clip; it amortises ~684 s of per-clip
  fixed stages and excludes HaWoR (~185 s) and the MoGe/GeoCalib/Stage-4 tail (~105 s, assumed). As F + r·N with
  F ≈ 974 s and r ≈ 22.0 s/frame: H 72 × 4,274 s = 85.5 GPU-h, D 108 × 2,571 s = 77.1 GPU-h, **total ≈ 169.3 GPU-h**
  with the N′ items. The short D clips carry the fixed cost.
- Ours: measured on the smoke (7 units): upstream ≈ 21 s and each arm 13–37 s per unit (model loads amortised); 180 units × (21 + 2 × 20) s ≈ **3.1 GPU-h** (cap 5).
- Known answers (section 7), before registration: ≈ 1.0 GPU-h spent on 2026-10-01 (theirs 0.73, ours 0.25); the outstanding HOI4D KA-R ≈ 1.35 GPU-h.
- **E1 total ≈ 147–173 GPU-h** (the authors' estimate, ~150–170).

### 10.2 Registered cap and capacity

- **Registered cap: 185 GPU-h of lock-held wall** = theirs ≤ 178 (the fixed-cost estimate + 5%), ours ≤ 5, reruns of
  infrastructure faults ≤ 2. It replaces the draft's 155 (two views).
- Card-hours 2026-10-08T00:00Z → 2026-11-05T23:59Z: 29 days × 24 h = 696 h, minus the freeze headline window
  2026-10-22 → 10-26 (no DO AS I DO or other study jobs during admitted passes) = **576 h**. Other commitments in that span:
  arm D ~40 GPU-h, ablation/stride/DexYCB-160/Amdahl/CALIB rehearsal ~25, S2 TEST one-shot ~10. E1 at its cap + these
  ≈ 260 h ≈ 45% of the span.
- Theirs at ~12 lock-hours/day from 10-08 needs ~14 days (to ~10-21); whatever is unfinished resumes 10-27 in the
  registered order. Under the GPU priority ruling, arm D shrinks before E1 does.
- Marginal prices: one H window 1.12–1.19 GPU-h; one D clip 0.54–0.71 GPU-h; the third view costs +19.5–25.7 GPU-h.

### 10.3 Schedule

| date (UTC) | step |
|---|---|
| 2026-10-01 | known answers on development data (section 7); **register** (this file on main) |
| by 10-07 | operator: `e1_select.py --draw` (subject-08 label reads, logged) and commit the units file (section 2.2); if \|R − 36\| > 6, rerun `e1_power.py --size-only --n-dex-seq R --dex-views 3` and commit the thresholds (6.3) |
| 10-08 → ~10-21 | theirs, SEALED, in the registered order: `e1_theirs.py --mode final --queue` (each unit under gpu.lock with FASTDO_STAGE=e1_theirs, resumable); daily commit of the seal ledger and label-read log |
| 10-22 → 10-26 | no E1 GPU job (headline passes); freeze manifest; amendment A-F (freeze SHA) |
| 10-27 → 10-28 | ours: `e1_ours.py --mode final --pin <A-F SHA>` upstream, then O-EST, then O-CAL, on every unit theirs RAN; theirs resumes only if unfinished |
| ≥ 10-29 | unseal (section 9), one scoring pass (`e1_score.py --mode final --score`, CPU), `e1_stats.summarise`, verdicts; ledger entry; paper tables |
| 11-05 23:59Z | hard stop for any E1 GPU run |

### 10.4 Stop rules

- **Budget/calendar stop:** at the registered cap for theirs, or at 2026-11-05T23:59Z, no new unit starts; a running unit
  finishes. Units never started are **NOT RUN**: listed by id, excluded from ITT and complete-case alike (non-delivery for
  budget reasons is not a method failure), never silently dropped. Because the order is a seeded random interleave, the
  RUN set is a random subset of each corpus. Under the GPU priority ruling, arm D shrinks before E1 does.
- **INCOMPLETE:** a corpus with < 80% of its units RUN gets no verdict; its statistics are reported labelled INCOMPLETE.
- **Ours always runs every unit that theirs RAN** (cheap); if ours cannot run (infrastructure), E1 is not scored until it
  has.
- **Void:** a failed known answer voids scoring until fixed (no outcome is read in between); a seal-hash mismatch voids
  that unit (listed). A deviation from section 3 found after the fact is reported with its affected units and the verdict
  is given with and without them.
- No interim look at any outcome; there is no early stop for success or futility.

## 11. Reporting

One ledger entry and one paper table per corpus, whatever the verdict: n assigned / RUN / NOT RUN / delivered per method;
M_c, exp(M_c) with its 95% CI, the 95% UB (and the Holm-level calibrated bound), the verdict; complete-case next to ITT;
the sign test; S1–S11; the asymmetries of 3.3; the power is not recomputed post hoc. A failed or INCONCLUSIVE corpus is
reported in the main text, not only the supplement.

## 12. Amendments

- After registration, before 2026-10-08: any change not informed by a held-out output (none exist yet), dated and
  labelled.
- After 2026-10-08: only changes that cannot depend on E1 outputs (A-F, the realised-R threshold rerun of 6.3,
  infrastructure fixes, a scorer bug found on development data); each one states why it is outcome-independent. Their
  sealed status lines may inform only infrastructure amendments.

Amendment index (each file is the record; this list was added 2026-10-14 and changes no text above):
- 2026-10-06-e1-amendment-disk-floor.md (this folder): infrastructure, disk floor 175 → 150 GB (superseded).
- 2026-10-07-e1-amendment-no-disk-guards.md (this folder): infrastructure, no disk guards.
- 2026-10-14-e1-amendment-A-F-freeze.md (this folder): **A-F**, freeze SHA fb6961213faf5600ba6c8a97acc31a6cc82eb038.
- 2026-10-14-e1-amendment-A-U-unseal.md (this folder): **A-U**, unseal at the 80% RUN bar once ours has every RUN unit;
  theirs stops under the stop record; no interim look.
- 2026-10-14-e1-amendment-A-N.md (this folder): **A-N**, the NMS-timeout sensitivity analysis S-N.
- 2026-10-14-e1-amendment-A-T.md (this folder): **A-T**, a TIMEOUT under host contention is an infrastructure fault (one retry, scored on the retry).
- 2026-10-15-e1-amendment-A-T2.md (this folder): **A-T2**, A-T retries only in a clean GPU window; an unresolvable TIMEOUT is excluded from both arms, with the sensitivity both ways.
- 2026-10-19-e1-amendment-A-V.md (this folder): **A-V**, the 2026-10-16 foreign edit to DO AS I DO's tree; an INFRA_* failure at stage start is NOT RUN (35 units); the 5 A-T retries bounded by the 2026-10-21T23:59Z calendar stop; the spanning-unit sensitivity and the disclosure.
- 2026-10-22-e1-amendment-A-W.md (this folder): **A-W**, the authors rule the two A-V TIMEOUT events with a clean poll and no retry start (D:20201002_111644:841412060263, D:20201002_113031:932122062010) INFRA-UNRESOLVED (reason USER_RULED_NO_RETRY_START); the A-T2 paired exclusion and sensitivity apply unchanged.

## 13. Committed harness (U4d) and tests

All in run/study/ (U4d; the untracked head-to-head harness is not used by any E1 run):

| module | role |
|---|---|
| e1_common.py | paths, the two layouts (final / smoke), disk floor, `is_heldout`, `refuse_heldout`, `require_gt_pose_access` (SealedPhaseError / HeldOutRefused), `log_label_read` + `check_label_path`, `prereg_registered` |
| e1_select.py | section 2: `--names` (file names only), `--dev` (smoke units), `--draw` (registered draw; refuses before registration), `run_order` |
| e1_theirs.py, e1_theirs_inputs.py, e1_theirs_preflight.py | section 3.2: input export, GT-label prompt, the N′ rule with their own reference-frame preflight, every released stage, conversion to doasido.npz, hygiene, packing |
| e1_seal.py | section 9: pack, SEAL.sha256, ledger lines, verify, unseal (writes UNSEALED.json only after verification), logged stderr access |
| e1_ours.py | section 3.1: bench upstream, EST / CAL arms on fresh copies, delivered rule, pin check |
| e1_score.py | section 4: centre error (impute / exclude), ours (final trajectory, QC rule) and theirs; KA-S / KA-O |
| e1_stats.py | sections 5–6: ITT bad values, stratified cluster bootstrap, Holm on calibrated thresholds, verdicts, sign test, summary |
| e1_power.py | section 8 |

Tests (tests/run/test_e1_*.py, CPU): selection determinism and the names-only guarantee; the seal manifest round trip and
tamper detection; the status-line whitelist; the N′ candidate rule; the scorer on synthetic known-answer data
(a 30 mm offset scores 30.000 mm; imputation earlier on ties); the scorer REFUSES GT poses of a held-out unit before the
unseal without opening any GT file, and in the smoke layout; the statistics (bad values, verdicts, Holm, INCOMPLETE,
sign test vs scipy).
