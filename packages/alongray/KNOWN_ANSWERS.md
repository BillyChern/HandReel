# Known-answer pack

An evaluator is only as trustworthy as the cases on which its answer is known in advance. The scoring fixtures of
`alongray` construct predictions with known errors and check the reported values. Other tests check input validity
or compare the scorer with a second implementation. Taken together they form a known-answer pack that anyone can
run against `alongray`, or adapt to check another evaluator of hand-object reconstructions.

Run the pack, from the package directory, with

    python -m pytest tests

It needs numpy, scipy and pytest, and uses no dataset. The pack's fixtures are synthetic: they reproduce no figure of
the accompanying papers. The parity test against the `fastdo.bench` metric modules (`tests/test_fastdo_parity.py`) ships
with the package and skips where `fastdo` is not importable; the check that reproduces stored development results
needs development records, is kept outside the package and is not part of the pack.

## The case that motivates the pack

`test_scale_known_answer.py` holds a synthetic 21-joint hand over 24 frames. The fixture samples the wrist depth
uniformly between 0.45 and 0.75 m. The baseline prediction has the wrist exact and, on every coordinate of the other
joints, Gaussian noise with a standard deviation of 4 mm. The second prediction translates that hand
rigidly, in each frame, by 0.1 times its wrist position. The wrist lands 10% too far along its own ray; the hand keeps
its size and shape, so every other joint moves by the same vector, not along its own ray (46-75 mm along it and, at the maximum,
9.3 mm across it after rounding). This is the error of a monocular method that puts a fixed-size hand model at the wrong depth.

| prediction | wrist e_par | wrist e_perp | root-aligned MPJPE | PA-MPJPE |
|---|---|---|---|---|
| baseline | 0.0 mm | 0.0 mm | 6.09 mm | 5.64 mm |
| wrist 10% too far, hand translated rigidly | 60.1 mm | 0.0 mm | 6.09 mm | 5.64 mm |
| every joint times 1.1 about the camera | 60.1 mm | 0.0 mm | 8.24 mm | 5.64 mm |

In this construction, the translation moves the unaligned along-ray error by 10% of the hand's range and leaves
both aligned metrics where they were. The statement covers this rigid 10% wrist-depth error only; in the third row,
scaling the hand changes root-aligned MPJPE while PA-MPJPE stays fixed. Two tests assert the identities:
root-aligned MPJPE equal to the baseline's to 1e-12 m (the test asks for a relative 1e-12, and at this 6.09 mm
value pytest's absolute floor of 1e-12 is the larger bound), PA-MPJPE to a relative 1e-9 or the same absolute floor,
`lam = 1.1`, `e_par = 0.1 z_rms`,
all of it in the gauge part and none in the perpendicular error. A third test, `test_printed_table_values`, reads
the three rows of the table from this file and compares each cell with the fixture's rounded output; it also checks
the 46-75 mm and 9.3 mm figures and the sampling range of the wrist depth against numbers written in the test. The third row is the true 10% scale about the camera centre: every joint moves
along its own ray and the hand grows by 10%. That is the case that changes root-aligned MPJPE (6.09 to 8.24 mm),
because root alignment keeps the size; PA-MPJPE still does not see it.

## What each fixture checks

### Raw against aligned error

| test | construction | known answer |
|---|---|---|
| `test_scale_known_answer.py::test_scaled_placement_moves_e_par_and_neither_aligned_metric` | the case above | aligned metrics unchanged to 1e-12 m (absolute floor) / relative 1e-9; `e_par = 0.1 z_rms` |
| `test_scale_known_answer.py::test_printed_table_values` | the case above, all three rows | the fixture's outputs, rounded, equal the cells of the table above, which the test reads from this file; 46-75 mm along the ray and a maximum of 9.3 mm across it after rounding; wrist depth inside 0.45 to 0.75 m |
| `test_scale_known_answer.py::test_full_gauge_contrast_root_alignment_sees_the_size_change` | every joint times 1.1 about the camera | PA-MPJPE unchanged; root-aligned MPJPE changes |
| `test_score.py::test_pure_gauge_moves_e_par_and_no_aligned_protocol` | object pose translations and object-local points times 1.1, rotations kept; hand moved by 0.1 of its wrist at fixed size | `e_par = 0.1 z_rms` for both bodies; reprojection error of the object centre and of the wrist zero; scale-aligned and Sim(3)-aligned centre errors, root-aligned MPJPE and PA-MPJPE all zero; Chamfer after scaled ICP unchanged |
| `test_sensitivity.py::test_pure_gauge_known_values` | a pure gauge of 1.25 | each alignment reports its closed form: `0.25 <\|g\|>` unaligned, `0.25` times the mean root-relative size after root alignment, zero after scale, Sim(3) and per-frame similarity alignment |
| `test_sensitivity.py::test_scale_and_sim3_residuals_are_gauge_free` | a noisy prediction and the same prediction times 2 | scale- and Sim(3)-aligned residuals equal |
| `test_readme.py::test_readme_example` | the README's example | the README's stated numbers |

### Along-ray against perpendicular error, and the exact split

| test | construction | known answer |
|---|---|---|
| `test_split.py::test_split_known_answer_in_closed_form` | ranges 1, 2, 3 on the optical axis with off-orbit residuals 1, 1, -1, plus a perpendicular offset | `lam = 1.5`, gauge part squared 7/6, off-orbit part squared 1 |
| `test_split.py::test_split_sums_exactly_on_random_trajectories` | random trajectories under random gauge moves | `e_par^2 = gauge^2 + off-orbit^2` to 1e-12, and the closed form after a gauge move |
| `test_split.py::test_normal_equation_is_what_makes_the_split_exact` | the split taken at three scales other than `lam` and other than 1 | the two terms no longer sum to `e_par^2` (a mutation guard). At scale 1 the cross term is zero as well, so that scale is not tested |
| `test_split.py::test_pure_scale_error_lands_entirely_in_the_gauge_part` | prediction = `lam` times truth | off-orbit part zero; the oracle correction removes all of `e_par` |
| `test_kappa_star.py::test_known_answer` | `lam = 2`, `<Z^2> = 1`, `<o^2> = 1` | `kappa_star = 0.4`, `e_par_min^2 = 0.2`, oracle floor 0.5, `e_par^2 = 2` |
| `test_kappa_star.py::test_kappa_star_and_e_par_min_match_a_grid_search` | random `lam`, off-orbit size and frame count; a grid of 3,001 gauge moves | no grid point below `e_par_min`; the grid's best within one step of `kappa_star` |
| `test_kappa_star.py::test_the_parts_are_not_the_gain_and_remainder_of_the_best_move` | `lam = 2`, `<o^2> = <Z^2>`; `lam = 0.5`, `<o^2> = <Z^2> / 4` | `e_par_min` below, then above, the off-orbit term; the gauge part is not `e_par - e_par_min` |
| `test_split.py::test_constant_along_ray_offset_puts_collinearity_of_its_square_in_the_gauge_part` | +50 mm along every ray | `e_par = 50 mm`; `gauge_part^2 = collinearity * (50 mm)^2` exactly, the rest off-orbit |
| `test_split.py::test_perpendicular_error_is_invisible_to_e_par` | 30 mm perpendicular to every ray | `e_par = 0`; perpendicular component 30 mm |
| `test_split.py::test_signed_along_has_the_sign_of_far_minus_near` | a prediction too far | positive signed error (a sign-convention guard) |
| `test_score.py::test_score_matches_the_low_level_split_and_off_orbit_error_is_seen` | scale 1.05 plus noise | the one-call API equals the low-level split; `e_3d^2 = e_par^2 + e_perp^2` |
| `test_shift.py` (all tests) | constant shifts and pure scales over flat and wide ranges | `collinearity` times a shift's mean square lands in the gauge part; shift and scale are indistinguishable over a flat range and separable over a wide one |
| `test_scene.py` (all tests) | two bodies on gauges 1.2 and 0.8 | the shared scale, the relative gauge and each residual in closed form |

### Missing frames

| test | construction | known answer |
|---|---|---|
| `test_split.py::test_nan_frames_are_dropped_consistently` | a NaN in the prediction at one frame, an infinity in the truth at another | the split equals the split on the remaining frames, with the frame count reduced by two |
| `test_sensitivity.py::test_mask_nan_and_refusals` | a NaN joint, a masked frame, a wrong-shape mask, an empty mask | the two frames dropped; the bad masks refused |
| `test_cli_io.py::test_frame_idx_matching_and_masks` | a prediction holding 7 of 10 frames, shuffled, with one frame masked in each file | 5 frames scored, equal to the split on exactly those frames |
| `test_cli_io.py` refusal cases | one-sided, duplicate, disjoint or non-integer frame ids; bad masks | refused with exit status 2 and a stated reason |
| `test_score_denominators.py` (all tests) | a NaN finger joint in one frame, a NaN wrist in another, a NaN object cloud | each metric's own frame count: hand centre 9 of 10, hand joints 8 of 10, object centre 10 of 10, Chamfer 9 of 10; 0 and NaN when no frame is usable |
| `test_score.py::test_zero_filled_prediction_warns` | missing frames filled with zeros instead of NaN | a warning |
| `test_split.py::test_single_frame_puts_everything_in_the_gauge_part`, `test_shift.py::test_low_n_flag_and_warning` | one or two usable frames | one frame: the off-orbit along-ray residual is zero; one or two frames: `low_n` set and a warning |

### Unequal frame counts

| test | construction | known answer |
|---|---|---|
| `test_cli_io.py::test_01_style_unequal_body_lengths_still_score` | an object over 10 frames and a hand over 8, scored without frame ids | each body scored on its own; every field present in the stored output of version 0.1 (`tests/data/v01_unequal_T.json`) matches it to a relative and absolute tolerance of 1e-12; fields added after version 0.1 are outside this check |
| `test_cli_io.py` case `count_mismatch_with_valid` | unequal body lengths together with a frame mask | refused: with frame ids or masks, every body must have the object's frame count |
| `test_split.py::test_errors`, `test_sensitivity.py::test_mask_nan_and_refusals` | prediction and truth of different lengths | refused, never truncated |

### Inputs the scorer must refuse or flag

`test_score.py` and `test_split.py` also check that a joint array is never broadcast against a wrist trajectory,
that mismatched joint sets, non-trajectory shapes, a non-positive scale and malformed poses are refused, and that
millimetre inputs and implausible scales are flagged. `test_intrinsics.py` checks that a non-canonical bottom row, a non-zero `K[1, 0]`, non-finite entries and a
non-positive focal length are refused, and that a canonical pinhole with skew gives the homogeneous projection.
`test_cli_io.py` checks the camera-frame heuristics: a
mirrored prediction and a ground truth crossing the camera plane are refused, while a z-up world frame and a wrist
outside the image are only warned. A principal point at the image corner is scored without a warning.

### Pooling

`test_pool.py` checks that the three row forms (moments, per-frame ranges, stored splits) pool to the same numbers,
that the pooled gauge and off-orbit shares sum to one when the pooled total is positive, and that the
cluster-bootstrap interval sits exactly at the stated percentiles.
`test_pool_moments_validity.py` checks that moments violating Cauchy-Schwarz beyond the tolerance, non-integer or
non-positive counts, and stored splits that break any of the three identities are refused, while the rounding of a
pure scale's exact zero is accepted. `test_pool_cohort.py` checks the cohort denominators (assigned, scored,
failed, excluded with reasons, missing from the manifest), that the statistics are those of the scored units alone,
and that the `refuse` policy refuses a group with a failed or missing unit.

### The change between two predictions (`alongray.change`)

`test_change.py` builds every case by hand: a change along the true ray (share 1), across it (share 0), at known
angles to it (share = the energy-weighted mean of the squared cosines, not their plain mean), and along a first
prediction that sits a known angle off the true ray (share 1 on its own ray, cos^2 of the angle on the true ray). It
checks that rows are joined on frame ids, that differing, duplicate or non-integer ids are refused, that a non-finite
row in any of the three inputs drops that frame from every quantity and is counted by reason, that a window with no
change has no share, that signed and absolute along-ray terms are kept apart, and, over windows, the median of window
shares against the ratio of pooled sums, the median of paired differences against the difference of medians (errors
(1, 2, 3) against (3, 1, 2) cm), Spearman known answers, and the cohort denominators. `test_cloud_length.py` checks
that object clouds must number one per trajectory frame.

### The scale-gauge illustration and the missing-outcome card

`test_gauge_demo.py`: a synthetic scene and its copies scaled by 0.5 and 2 project to the same pixels to 1e-8 px
under an independent float64 projection; near-zero and negative depths are refused; a fixed-size anchor and a second
camera at a 10 cm baseline move pixels by more than 1 px (a zero baseline does not). `test_missing_bounds.py`: the
card's bounds equal the minimum and maximum over every binary completion of the unknowns for every status pattern of
two methods on N = 1 to 4 units; delivery failures count as declared; refusals.

## Adapting the pack to another evaluator

Feed the same constructions to your evaluator and compare. The most informative is the first: in this fixture the two aligned metrics do not distinguish the
rigidly translated hand from the baseline. If your evaluator likewise reports every number unchanged for that pair, it
does not see this along-ray error. Run the
true-scale row as well: a benchmark that reports root-aligned MPJPE sees that one, so the pair separates a depth error
from a size error.
