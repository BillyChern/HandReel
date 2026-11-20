# Changelog

## Unreleased

- Fixed: the `--boot` interval of the sequence-weighted aggregate in `alongray-coverage` grouped each bootstrap draw
  by sequence label, so a sequence drawn several times counted once. Each sequence now counts by the drawn share of
  its frames, which is its multiplicity when it lies in one cluster and keeps the point estimate under the identity
  draw when several clusters share it. Point estimates, frame-weighted intervals and the pooled-share intervals
  are unchanged; sequence-weighted intervals from earlier versions can be too narrow.

Fixes. The first entry changes a computed number, the sequence-weighted `--boot` interval. The others change no
computed number for inputs that were accepted and unambiguous before.
- `alongray-coverage` refuses ambiguous method names. The report keys its tables by method name, so two inputs with
  one name (two folders with the same basename, or equal `method` fields) overwrote each other's inputs, counts and
  conventions, and a method named `protocol` or `status` overwrote a column of the convention table. Both now give
  the verdict `REFUSED` with the reason (`tests/test_coverage.py`).
- The legacy `frame_idx` key of the command line needs one id per row of the file's `object`, representable as int64.
  A shorter list left rows without an id and dropped them silently; a longer one addressed a row that does not
  exist. Both exit with status 2. Files with different frame sets still match on their shared ids
  (`tests/test_cli_io.py`).

- `cluster_bootstrap` keeps the point estimate when a draw has no defined statistic. If cluster A has zero error and
  cluster B has error, the pooled share exists, but a draw of A alone has none; that draw raised and the whole
  estimate was lost. Such a draw is now counted (`pooled_share_ci95_undefined_draws`), left out of the
  percentiles, and reported with a reason; with no defined draw the interval is `NOT_ESTIMABLE`
  (`tests/test_pool.py`).
- A cohort in which every unit has zero error keeps its zero lengths and counts; `pooled_share` is `null` with a
  reason instead of an exception. `pool_rows` refuses an assigned unit listed more than once, as the command line
  does, and every group reports `assignment` (`manifest` or `observed rows only`).
- The `alongray-pool` manifest needs one line per assigned unit: a repeated unit id is refused with both line
  numbers, and without `--group-key` a second column is refused. Without a manifest, `n_missing` is 0 by
  construction and describes the observed rows only.

## 0.4.1 (2026-11-10)

Fix: repeated JSON keys are refused. Every JSON input alongray reads (`alongray-pool` rows and `--rows-key`
documents, the `alongray-coverage` protocol and reports, the `alongray.missing_bounds` inputs) is parsed by the new
`alongray.strict_json`, which refuses a repeated key in any object, nested objects included, and names the key and
the file (and the line, for JSON-lines rows). Plain JSON parsing keeps the last value of a repeated key, so a row such
as `{"status": "ok", ..., "status": "failed"}` was read silently as one of the two. Such inputs now exit with status 2
(`alongray-pool`, `missing_bounds`) or are refused as unreadable (`alongray-coverage`). No computed number changes for
inputs without repeated keys (`tests/test_strict_json.py`: a valid control parses identically to plain JSON).

## 0.4.0 (2026-11-09)

New:
- Optional frame ids. `score` and `score_body` take `pred_ids` / `gt_ids` (one integer id per row, both or neither);
  the command line reads a `frame_ids` array from both files, and `--require-frame-ids` refuses files without one.
  The rows are joined on the ids by the rules of `alongray.change`: identical id sets (no intersection, no
  trimming, no shift), no duplicates, integers only, increasing id order. A prediction shifted by one frame or
  missing a frame is refused instead of scored. `frame_idx` (shared-id matching, from 0.2) is unchanged and may not
  be combined with `frame_ids`.
- `examples/ho3d_shaped_adapter.py`: an adapter from per-frame hand-object annotations to alongray inputs, on
  generated synthetic data only (object-to-camera transforms, one physical object point, metres, wrist root, frame
  ids, camera axes).
- `alongray.NOT_ESTIMABLE`.

Behaviour changes that can break existing callers:
- `cluster_bootstrap` returns the string `NOT_ESTIMABLE` instead of an interval when there are fewer than two
  distinct clusters (0.3.1 returned a zero-width interval for one cluster). `pool_rows` / `alongray-pool` then report
  `"pooled_share_ci95": "NOT_ESTIMABLE"` and a `pooled_share_ci95_reason`, and keep the point estimate;
  `alongray-coverage` reports `"ci95_cluster_bootstrap": "NOT_ESTIMABLE"`. With two or more clusters every output is
  bit-identical to 0.3.1 (`tests/test_pool_one_cluster.py`).

Unchanged: without ids every output of `score` and of the command line is what 0.3.1 gives (`tests/test_frame_ids.py`:
bit for bit against the id-free call in the same environment, and against outputs recorded with 0.3.1 on numpy 2.1.2
in four cases: zeros must match exactly, a nonzero value whose recorded magnitude is below 1e-9 must keep a magnitude
below 1e-9, and other finite values must agree within eight units in the last place. These are checks on the outputs
of those fixtures, not bounds on every possible input); the README now states that rows are then assumed synchronized row by row.

Also: uint64 frame ids above the int64 range are refused (in `alongray.change` as well) instead of wrapping to
negative ids.

README: the licence section no longer says "Publish the derived numbers, not the raw outputs."; it asks users to check
the dataset, model and derived-output terms before sharing anything, and states that nothing in the package grants
rights to them.

## 0.3.1 (2026-11-09)

No change to any computed number, output key or command-line behaviour.

- README: a quickstart that runs from this directory (`examples/make_example.py`, then the `alongray` command), install
  and test commands that run from this directory, and this changelog in place of the README's change notes.
- `tests/test_quickstart.py` runs the quickstart.
- Packaging: copyright holder "The alongray authors"; `CITATION.cff` (author fields to be filled at release);
  `.venv/` ignored. The two development-only test files (parity with the authors' development implementation, and
  reproduction of stored development results) moved out of the package; outside the development repository they always skipped.
- `tools/mutate_split.py`: the optional extra suite is now named by `ALONGRAY_EXTRA_TESTS`.

## 0.3.0 (2026-11-08)

New: `alongray.change` (`window_change`, `summarize_changes`), the change between two predictions of one unit split
along and across a ray, with the conventions in the README; `alongray.gauge_demo`, the scale-gauge illustration;
`alongray.missing_bounds`, the missing-outcome sensitivity card.

Behaviour changes that can break existing callers:
- Intrinsics must be finite canonical pinhole matrices (positive focal lengths, `K[1,0]=0`, bottom row `[0,0,1]`; skew allowed); the command line exits 2 on a bad `K`.
- Pooling refuses moments that break Cauchy-Schwarz and stored splits that fail the decomposition identities (relative tolerance 1e-6) instead of clamping them.
- Pooling reports assigned, scored, failed, excluded and missing units, with a `failure_policy` (`report` or `refuse`); output has new keys.
- `BodyScore` gains `aligned_n_frames` and `n_frames_input`; the Chamfer key is present, as NaN, when no frame is usable.
- Object point clouds must number one per trajectory frame (`len(points) == T`), not merely match each other.

## 0.2.0 (2026-11-04)

- New fields `shift_part`, `shift_scale_part`, `collinearity` and `low_n` on `EParSplit` and `BodyScore` (with
  defaults, after the 0.1 fields), and `MIN_ROWS = 3`.
- `e_par_split` (and so `score`) now warns (`UserWarning`) when fewer than 3 frames are used.
- New modules `alongray.sensitivity`, `alongray.scene` and `alongray.pool`, and the `alongray-pool` command.
- Command line: the keys `frame_idx` and `valid` are now interpreted (0.1 ignored them); `--scene`; and the
  heuristic camera-frame checks, which warn on suspicious input and refuse (exit 2) only a ground truth that
  crosses the camera plane beyond the tolerance or a mirrored prediction; 0.1 scored such inputs. Without `frame_idx`
  and `valid`, bodies are scored on their own exactly as in 0.1, and the default JSON output has the same keys plus
  the four new fields per body.

## 0.1.0 (2026-10-30)

First release: `e_par`, its gauge / off-orbit split, `score`, the aligned metrics and the `alongray` command.
