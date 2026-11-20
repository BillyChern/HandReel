# The held-out test: protocol, unit IDs and the verdict check

**Note on `protocol/`.** The files in `protocol/` are the registered pre-registration and amendment records, edited for
public release. Internal references were replaced and the rule text is unchanged: references to internal record files,
internal commits and internal record numbers now point to a file of this folder or say that the record is internal and
not released; internal paths were removed; role and workflow wording was made neutral (the authors; the operator, who
ran the study harness); and host paths, a host volume name and another tenant's job name were redacted. Each rule, its
date and time, and the statement that it was fixed before any scored outcome are as registered (amendments A-U,
A-V and A-W say which run-status and refusal information they used); where a rule names a role or
a file, only that name changed. `protocol/redactions.json` gives the starting line and a description of each edit and
the sha256 of both copies of each document. An edit can extend past the line on which it starts. No edit adds or
removes a line, so line numbers are preserved. Kept as
registered: the freeze commit and its three tree ids that amendment A-F and the freeze manifest record (commits of the
authors' internal repository, not of this release's history), the file hashes, and the labels and branch names of
internal records the rules cite (such as M341, s255, U-1 to U-8 or e1/unseal-prep), which are not released.

This folder holds what the paper's held-out test against DO AS I DO (Table 1; Secs. 5.1 and 5.2 of the submitted
main paper) proposes for release without dataset media or per-unit results: the registered protocol, the unit-ID
lists, the thresholds that the analysis used and the QC rule's constants. These are materials proposed for release;
the authors' decisions on them remain open. **This release contains no per-unit error derived
from HOI4D or DexYCB ground truth.** Regenerate the errors after acquiring the datasets
(`regenerate_unit_errors.py`, below); with that file, `reproduce/e1_verdict_from_scores.py` recomputes the error
ratios, intervals and median errors that Table 1 prints. Nothing here is a dataset frame, mesh, mask, ground-truth
pose, per-unit error or model output.

| file | what it is | source |
|---|---|---|
| `protocol/` | the pre-registration (2026-10-01) and every amendment its index lists (disk floor, no disk guards, A-F freeze, A-U, A-N, A-T, A-T2, A-V, A-W), and the freeze manifest that A-F records; edited for public release (above) | the authors' registered documents |
| `units_hoi4d_72.json` | the 72 HOI4D test windows: unit and clip IDs, HOI4D sequence path, category, instance, start and stop frame, frame count, stride, cluster and stratum | the registered draw |
| `units_dexycb_s08_108.json` | the 108 DexYCB subject-08 views (36 sequences x 3 cameras): unit and clip IDs, subject, sequence, camera serial, frame count, stride, cluster; and the six views the draw replaced | the registered draw |
| `regenerate_unit_errors.py` | writes `unit_errors_est.json` (not shipped; regenerate after acquiring the datasets): one row per assigned unit (180) with its corpus, cluster and stratum from the unit lists, the two units amendments A-T2 and A-W remove, and, from your own results, whether it ran (DO AS I DO began an attempt on it; amendment A-V) and, for HandReel (estimated intrinsics) and DO AS I DO, whether it delivered and its mean unaligned object-centre error in mm. It also gives `processed_frames`, `complete_centres` (the registered fill of a frame without a usable pose) and `unit_centre_error_mm`, an averaging helper for completed rows that refuses a missing or repeated frame; it is not the registered scorer, which is not released | this release |
| `e1-power-size.json` | the size-calibrated one-sided thresholds per corpus (and the simulation rows they came from), read by `run/study/e1_stats.load_thresholds` | the registered power script's size-only run, byte for byte |
| `qc_rule_constants.json` | the constants of the QC acceptance rule that HandReel ran with: five feature names, six logistic weights, the feature means and standard deviations, tau and the conformal threshold, unrounded, with the fit's counts and settings; a version-1 `fastdo.qc` rule file | the registered rule file (sha256 prefix `e564b52e92d3`), reduced to its constants by `derive_qc_rule_constants.py` (below) |

## Recompute the held-out values of Table 1

First acquire HOI4D and DexYCB under their terms, run both pipelines on the units of the two lists, score each
unit's mean unaligned object-centre error over its processed frames with ground truth (Sec. 5.1) after completing
each method's poses by the registered rule (the script's docstring states it; averaging only the frames a method
delivers gives a different number), and write one
JSONL line per unit (format in the script's docstring). Then, from the repository root, in the main env (numpy and
scipy; no GPU, about a second):

```bash
uv run --frozen python reproduce/heldout/regenerate_unit_errors.py --results my_results.jsonl
uv run --frozen python reproduce/e1_verdict_from_scores.py
```

The regenerated file is derived from the datasets' ground truth: keep it under their terms and do not redistribute
it (`.gitignore` lists it). Without it, the second command prints `skipped: ...` and exits 0.

With the file, the script prints six aggregate values per corpus, each beside `matches the reference` or
`DIFFERS from the reference`, and exits 0 only if every value matches. On a file in which no unit of a corpus was
delivered by both pipelines, three of the six (the complete-case ratio, the within-50 mm counts and the medians) do
not exist and print as `unavailable`; an interval end past the range of a float prints as `above the float range`. The reference values are aggregate values of
the authors' registered analysis. The submitted main paper prints four of the six:

| script key | what it is | where the submitted paper prints it |
|---|---|---|
| `analysed` | analysed units over assigned units: 58/72 on HOI4D, 85/108 on DexYCB | Sec. 5.1 |
| `delivered` | units that HandReel, DO AS I DO and both delivered: 58/49/49 on HOI4D, 58/69/48 on DexYCB | Sec. 5.2; Table 1, caption |
| `mitt` | the exponentiated median of the failure-coded paired log error ratios, with its two-sided 95% bootstrap interval, where a failure to deliver counts as an error and a joint failure as a tie, and the verdict: 0.50 [0.33, 0.63] on HOI4D, 0.85 [0.58, 1.00] on DexYCB, both non-inferior | Table 1 |
| `median_mm` | the median error in mm over the units that both pipelines delivered, HandReel/DO AS I DO: 56.8/101.3 on HOI4D, 76.6/112.5 on DexYCB | Table 1 |
| `complete_case` | the error ratio and its interval on the units that both pipelines delivered, with estimated camera intrinsics, on 49 HOI4D windows and 48 DexYCB views | not in the main paper; the supplement's detailed table of the held-out test reports it |
| `within_50mm` | among units both pipelines delivered, the count with mean centre error at most 50 mm, separately for each pipeline | not in the submitted paper; the script keeps the registered analysis's counts as its reference |

Table 1 marks the DexYCB rows: that verdict is not robust if joint failures count against HandReel, and the authors
partly used this subject in development. The script prints `NI` for DexYCB and does not compute that mark.

The authors ran the script for this release on their private copy of the registered per-unit errors: exit 0, every
value matches its reference. Errors from your own runs match the references only as far as your runs reproduce the
registered ones.

The analysis is `run/study/e1_stats.summarise`, unchanged (median paired log-ratio with the pre-declared failure
values, stratified cluster bootstrap with B = 10,000 and seed 1, Holm over the two corpora on the calibrated
thresholds), and `run/paper/e1_table.py` formats the numbers. It runs on the analysed units: the units that ran,
without the two DexYCB units that amendments A-T2 and A-W remove from both pipelines (INFRA-UNRESOLVED; `removed`
in the score file). On HOI4D 58 of 72 assigned units ran and 58 are analysed; on DexYCB 87 of 108 ran (80.56%) and
85 are analysed after the two removals. Before reporting the primary results for 58 HOI4D windows and 85 DexYCB
views, the script checks that DO AS I DO started at least 80% of the assigned units in each corpus, counting both
excluded DexYCB timeouts as starts. If you call summarise directly, check completeness against the full assignment
and pass only runs with no recorded removal to the function. `--json <file>` also writes the values. In this study a unit ran when DO AS I DO
began an attempt on it (amendment A-V): the 35 units on which the baseline refused to start count as not run,
although HandReel processed them. The reference values that the submitted
paper prints are those of its Table 1 and Secs. 5.1 and 5.2 (recheck them against the published version).
`tests/reproduce/test_heldout_release.py` checks the skip, the regeneration script's refusals and, on synthetic
errors, that the script computes and compares every value.

**Seeds.** The bootstrap is discrete, so an interval end can move at the second decimal from one seed to another.
The printed intervals are those of seed 1, the registered seed. Do not replace them with another seed's.

This recomputes the verdict from per-unit errors. It does not rerun either pipeline, the scorer, the QC decisions or
the exclusions, and it does not show that the errors are correct. It does not recompute the sensitivity analyses
that the supplement reports, such as the ratio when joint failures count against HandReel: their per-unit inputs
are not here.

## Provenance

| file included in this candidate | made from | sha256 of the source |
|---|---|---|
| `units_hoi4d_72.json`, `units_dexycb_s08_108.json` | the registered draw of held-out units, also pinned by the secondaries record | `72465b68c75d87b593bc4a67332963d0e38906bd62c89565b43c3b5d007524fb` |
| `e1-power-size.json` | unchanged | `def45c58916f0e8bdebe2d9e49440beb4d4487614693443334b412e79d861a65` (this file) |

The registered per-unit score file (sha256 prefix `efcb767640df`, pinned by the secondaries record) is not released.
The unit lists keep the identifiers and window
bounds of the draw and leave out `shift_px`, a per-window pixel offset that registers HOI4D's annotation to the
frames: it is derived from HOI4D's annotations, not an identifier, so a scorer that needs it must recompute it.

The protocol copies differ from the registered documents only where `protocol/redactions.json` says (the note at the
top of this page): absolute host paths became `<host path>`, two mentions of the host volume lost the volume's name,
another tenant's job name on the shared GPU became `<another tenant's job>`, and internal references and role wording
were replaced. `redactions.json` gives the starting line and a description of each edit and the sha256 of each
original and of each released copy. Paths in the protocol
under `run/` and `outputs/` name code and outputs in the authors' repository at the time; most of them are not in this
release.

### Editorial notes on three protocol statements

The protocol copies keep these statements as registered. The notes below say how far each one reaches.

- **`run/conf/config.yaml`.** The sha256 that amendment A-F and the freeze manifest give for this path (`0c4585e8...`)
  is of the file at the freeze commit in the authors' repository. The file of the same name in this release is a later
  file with another sha256 (`5638959d...`). The freeze hash does not describe the shipped file.
- **Amendment A-U, "No sealed file and no ours-side held-out output exists or was read."** The line before it records
  what the authors read at that amendment: status counts only, 75 of 108 DexYCB units and 50 of 72 HOI4D units marked
  RUN. Those runs existed. The authors recorded only status counts at this amendment. They reported opening no sealed
  archive and had not run HandReel on these units.
- **Order of the freeze events.** The header of the freeze documents carries 2026-10-14T07:14:19Z, and the manifest
  cites a fetch of the main branch at 07:15:57Z, which is later than the header. The header time alone
  does not prove the order of events. The order follows from the freeze commit and from the
  receipt of the first run after it, which are internal records and are not in this release.

### Editorial note of 2026-11-19: what was validated and reported after registration

The protocol copies keep their registered wording. This note records what happened to three of its items
afterwards and which of two printings of one threshold decides. It separates validation or reporting that stayed
incomplete from inputs that this release merely does not ship. The historical validation and outcomes records it cites are
internal and are not in this release; the protocol and `e1-power-size.json`, cited below, are included in this candidate.

- **Known answer KA-R on a HOI4D development window.** Section 7 of the protocol lists it as not yet run and as a
  condition of launch. It ran and passed on 2026-10-02, before the launch.
- **Known answers KA-S and KA-O at the unseal.** The runbook asked for both scorer checks to be repeated at the
  unseal. Their development inputs had been deleted by then. KA-S was re-established on 2026-10-26 from the
  ground-truth frames and returned the same six errors as on 2026-10-01. KA-O was not rerun: it keeps its pass of
  2026-10-01, and the authors checked that the code it exercises had not changed since. That is an incomplete
  validation, not an unshipped input.
- **Registered secondaries S3, S4 and S9.** Section 5 of the protocol registers eleven secondary analyses. The
  authors' outcomes record lists S3 (the F-5 metrics), S4 (metric-scale error) and S9 (the baseline's dense-frame
  score) as not reported, a deviation from the protocol. Omitting S3, S4 and S9 leaves the primary verdict
  unchanged. Amendment A-N separately conditions a headline claim on S-N when either pipeline logs an NMS time-limit
  warning for any unit of that corpus. That is incomplete reporting, not an unshipped input.
- **Printed and exact thresholds.** Section 6 of the protocol prints the size-calibrated DexYCB levels as 0.0180 and
  0.0380. The decisions used the full-precision values of `e1-power-size.json`, 0.017991... and 0.037981.... With
  10,000 bootstrap draws the possible p-values 180/10,001 and 380/10,001 lie between each printed level and its
  exact one, so the two printings can disagree; the full-precision file is the authoritative one. Both recorded
  primary p-values (1/10,001 and 2/10,001) lie far below either printing.

## Not here

- **The original registered QC rule file** (sha256 `e564b52e92d3...`, registered by the protocol). It is not in
  this release; a copy reduced to its constants is, with a hash that differs. Besides the fitted constants, the
  original holds its provenance: the DexYCB development clip and sequence IDs that the authors fitted it on, two
  absolute host paths and an internal commit. Those are not released. `qc_rule_constants.json` is the copy reduced to
  the constants (five feature names, six logistic weights, the feature means and standard deviations, tau 0.05, the
  conformal threshold 0.6907751134070998 at alpha 0.1 on 19 calibration clips) and the fit's counts and settings
  (71 clips, 39 good and 32 bad, fit fraction 0.5, seed 42). Every constant is the original's value, unrounded; the
  supplement's settings table prints the threshold rounded. Because the copy leaves the provenance out, its sha256 is
  not the registered `e564b52e92d3...`: the registered hash refers to the original file, which the authors keep. The
  copy's sha256 is `eddd0032993fe4256887c3bd882933ad5de92b22e92e0f167eec5c89d3699dda`, and
  `derive_qc_rule_constants.py` rebuilds it byte for byte from the registered file (`--check` confirms a copy; by
  default it refuses any input whose sha256 is not the registered one, and with `--allow-unregistered` it marks
  what it writes or prints as unregistered). The copy is a record of the test, not a rule for new
  video: the supplement's limitations report that the authors fitted the QC on 71 DexYCB development clips with
  ground-truth template meshes, and that its five-fold cross-validated area under the curve on those clips was
  0.4904, where 0.5 is chance. Without the per-unit QC features, which are not released, a user cannot re-derive
  its held-out decisions from it.
- The registered per-unit errors, run records and queue records, DO AS I DO's sealed outputs and the scoring,
  sealing and study scripts (`reproduce/README.md`, "Unshipped inputs of the held-out verdict").
- Any frame, mask, mesh, ground-truth pose or reconstruction of a held-out unit.

## Datasets and licences

The unit IDs are names from two datasets' public releases. No file of either dataset, and no error computed against
their ground truth for a single unit, is included; the reference values above are aggregates.

- **HOI4D**: Liu et al., "HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction", CVPR 2022
  (<https://hoi4d.github.io/>), licensed CC BY-NC 4.0 (<https://creativecommons.org/licenses/by-nc/4.0/>).
- **DexYCB**: Chao et al., "DexYCB: A Benchmark for Capturing Hand Grasping of Objects", CVPR 2021
  (<https://dex-ycb.github.io/>), licensed CC BY-NC 4.0 (<https://creativecommons.org/licenses/by-nc/4.0/>).

Changes: the two unit-list files list identifiers of the datasets' sequences and windows. A per-unit error file you regenerate is
derived from the datasets' annotations: non-commercial use only, as both licences require, and not for
redistribution with this code.
