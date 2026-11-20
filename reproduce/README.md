# Reproducing the paper's numbers: what this release holds

**Most of the paper's records are not in this release, and no command here reruns an experiment.** This folder
holds three things:

- `regenerate_tables.py`, which prints tables with the paper source's table generator from their records and
  compares every input and output with a pinned sha256 (`manifest.json`);
- `records/`, with 14 of the paper source's 31 records, all aggregate only and **proposed for release**: they await
  the authors' release decision;
- `heldout/`, with the held-out test's protocol, unit-ID lists and thresholds, and a script that rebuilds the
  per-unit centre errors from your own runs after you acquire the datasets. Per-unit errors never ship. From them,
  `e1_verdict_from_scores.py` recomputes the error ratios and median errors of the submitted Table 1
  (`heldout/README.md`).

`regenerate_tables.py` checks all 14 records. It uses ten of them to write four of the generator's tables,
`solve.tex`, `stride.tex`, `optim.tex` and `weak_rows.tex`, and writes a fifth, `hyper.tex`, from settings in the
generator. **None of the generator's tables is an exhibit of the
submitted main paper.** The paper source writes the submitted Tables 1 to 5 by hand, in files that the generator
does not write. The supplement is not final, and this release does not state which generator tables it prints. The
other generator tables require `tools/make_tables.py` and records from the authors' internal tree. The inventory
below lists what this release includes.

`docs/paper_exhibits.md` maps every exhibit of the submitted main paper (Tables 1 to 5, Figs. 1 to 5, Algorithm 1)
to what this release contains toward reproducing it. Every table, figure and section number in this folder's
documents is a number of the submitted main paper; recheck them against the published version. These documents cite
the supplement without numbers, because its numbering is not final. `scripts/check_paper_quotes.py` checks that
every cited main-paper number exists in the paper's PDF with the declared caption or heading. It does not check
what a sentence claims (`docs/paper_exhibits.md`, "What the checker checks, and the manual audit").

The authors edited the protocol records in `heldout/protocol/` for public release: they replaced internal references
(record files, commits, record numbers and paths), made role wording neutral and left the rule text unchanged.
`heldout/protocol/redactions.json` lists each edit (`heldout/README.md`, note on `protocol/`).

`regenerate_tables.py` is a printing step: it re-reads the stored records, regenerates the tables, and compares
every output with a pinned sha256. Without `--paper-src` it re-reads only the 14 aggregate records in `records/`.
Only with `--paper-src` does it also re-read the per-unit score records, which are not in this release. A match
shows that the generated tables follow from the records, as the paper source's generator printed them at the pinned
revision. It does not show that the records are correct, it does not cover any table of the submitted main paper,
and it does not reproduce any scorer, timing, unit exclusion or quality-check decision.

## Contents

| file | what it is |
|---|---|
| `regenerate_tables.py` | without `--paper-src`: checks the records in `records/`, runs `make_released_tables.py` and compares the five tables it writes with the generator's output at the pinned revision. With `--paper-src`: copies the paper source's table generator and records into a new directory, runs them there, regenerates the held-out test macros with `run/paper/e1_table.py`, and checks every input and output against `manifest.json`; `--repin` rebuilds `manifest.json` from another paper revision |
| `make_released_tables.py` | (needs Python 3.12 or later, as the rest of the release does: it nests quotes inside f-strings) the paper generator's functions for the five tables that the records in `records/` support, with comments, docstrings and LaTeX comment lines removed (its docstring says what else changed) |
| `records/` | **proposed for release**: the 14 aggregate-only records, byte for byte as the paper source holds them (sha256 in `manifest.json`) |
| `manifest.json` | the paper source revision, the generator's sha256, every record's sha256 and release status, and each expected output's sha256, the records it reads and, for the five released tables, the sha256 of the table without its `%` comment lines |
| `e1_verdict_from_scores.py` | recomputes the held-out test's aggregate values per corpus (analysed and assigned units, deliveries, the error ratio with its interval and verdict, the ratio on the units that both pipelines delivered, the within-50 mm counts and the median errors) from a per-unit error file that you regenerate (`heldout/regenerate_unit_errors.py`), with `run/study/e1_stats.py`, and compares each with its reference value; without that file it prints that the check is skipped and exits 0. `heldout/README.md` says which of these values the submitted paper prints, and where |
| `heldout/` | the held-out test's registered protocol and amendments (edited for public release, above), the 72 HOI4D and 108 DexYCB subject-08 unit-ID lists, the calibrated thresholds, the QC rule's constants and `regenerate_unit_errors.py` (per-unit errors are not shipped); its `README.md` gives sources, sha256, what was left out and the datasets' CC BY-NC 4.0 terms |
| `alongray_shares.py`, `strict_json.py` | the per-method median share of squared centre error along the camera ray, with its sequence-cluster bootstrap interval; needs `uv pip install ./packages/alongray`; `--demo DIR` runs it on synthetic windows of known share (1, 0 and 1/2). The submitted paper reports such a share on 32 DexYCB development units (Sec. 5.2); the inputs for the paper's values are not released |

Related code elsewhere in this repository:

- `run/paper/e1_table.py`: writes the held-out test macros (`e1_results.tex`) from the verdict record. Its default
  `--out` names a `paper/` directory that this release does not have, so pass `--out`.
- `run/study/e1_stats.py`: the estimator behind the held-out test results (paired median log-ratio, clustered
  bootstrap, Holm over the two corpora, sign-test sensitivity). numpy and scipy only, no data access; its tests,
  `tests/run/test_e1_stats.py`, run without torch. The `heldout/` directory holds the unit lists, thresholds and QC
  constants. After you acquire the datasets and regenerate the per-unit errors, `e1_verdict_from_scores.py` runs it
  on that file.

## Commands

```bash
python3 reproduce/regenerate_tables.py --help
python3 reproduce/regenerate_tables.py --out <new or empty dir>                # the released check: this folder only
python3 reproduce/regenerate_tables.py --paper-src <paper source dir> --out <new or empty dir>
# re-pin to another paper revision, then check again without --repin:
python3 reproduce/regenerate_tables.py --paper-src <paper source at REV> --out <new or empty dir> --repin REV
uv run --frozen python reproduce/heldout/regenerate_unit_errors.py --results <your results.jsonl>   # needs the datasets
uv run --frozen python reproduce/e1_verdict_from_scores.py              # the held-out values; skipped without that file
uv pip install ./packages/alongray && uv run --frozen python reproduce/alongray_shares.py --demo <new dir>
python3 scripts/check_paper_quotes.py --main <main paper PDF>          # numbers and wording patterns; not a claim check
python3 scripts/check_paper_quotes.py --patterns-only                # without the PDF: wording patterns only
```

- `--help`: checked.
- The released check: run for this release (CPU only, seconds; no third-party package needed). The 14 records in
  `records/` matched their pins, and the five tables that `make_released_tables.py` writes (`solve.tex`,
  `stride.tex`, `optim.tex`, `weak_rows.tex` and `hyper.tex`) equal the paper generator's output at the pinned
  revision, without its `%` comment lines, which name files that are not released. The check compares every other
  line. The shipped records regenerate no other output: each other output reads at least one record that is not
  released.
- The check with `--paper-src`: run for this release on a copy of the paper source at the pinned revision
  `2b6e3f934`, and again on the authors' full internal source tree for the submitted revision, which includes the
  generators and records (CPU only, seconds). The manuscript-only source bundle of the submitted paper does not hold
  those inputs, so the check cannot run on it. Both times all 32
  pinned inputs matched, and the run regenerated all 23 expected outputs byte for byte (22 tables, plus the body of
  `e1_results.tex`, whose first line names the record path and is not compared).
- `e1_verdict_from_scores.py`: run for this release without a regenerated file: exit 0, `skipped: ...`. The authors
  ran it on their private copy of the registered per-unit errors, rebuilt through `regenerate_unit_errors.py`: exit
  0, every value matches its reference (`heldout/README.md`). numpy and scipy; about a second.
- `alongray_shares.py --demo`: run for this release: along 1.000, lateral 0.000, mixed 0.500 (`tests/reproduce/
  test_alongray_shares.py`).
- `scripts/check_paper_quotes.py`: run for this release against the submitted main paper's PDF (pdftotext): exit 0.
  A clean exit means four things. Every cited table, figure, section and algorithm number of the main paper exists
  in the PDF with the caption or heading that the script's `IDENTITY` table declares. No document in its `DOCS`
  cites a supplement number. Every quotation is a section name of this release, a passage that its `NOT_PAPER`
  table declares, or verbatim in the main paper, and `CITATION.cff` and `README.md` give the paper's title. No
  document contains a wording that matches the script's `FORBIDDEN` patterns, a finite hand-written list. A clean
  exit does not mean that the claims are right: the script binds a number to a caption, not a sentence to its
  evidence, and a paraphrase, a changed unit or a claim attached to the wrong table passes
  (`tests/reproduce/test_check_paper_quotes.py` asserts these three limits). A manual audit, made on 2026-11-17
  for this candidate, compared each results sentence of `README.md` and `docs/paper_exhibits.md` with the words of
  the submitted main paper. The PDF is not in this release; the tests run the script on synthetic text and run
  the pattern check on the shipped documents.
- `--repin REV`: writes `paper_revision` REV and the sha256 of the generator, every record and every regenerated output
  into `manifest.json` (or `--manifest`); each record keeps its release status, each output its label and the records
  it reads. Checked on a synthetic paper source (`tests/reproduce/test_regenerate_tables.py`).

`--paper-src` is only read. The report goes to `<out>/report.json`; the exit code is 0 only when every pinned input
and every expected output matches. `--allow-unpinned` runs on changed inputs and reports which outputs then differ.

## What the check covers, and what it does not

- **It compares the generator's output, not the compiled paper.** The check compares the tables that
  `tools/make_tables.py` writes with the pinned sha256. The submitted main paper prints none of them: hand-written
  files of the paper source hold its Tables 1 to 5. So the check covers no table of the submitted main PDF.
- **Table 1.** The generator's `tables/main.tex` is not the submitted Table 1, and no table or section of the
  submitted main paper prints the `\Eone...` macros of `e1_results.tex`. The check covers both files as generator
  outputs only. `e1_verdict_from_scores.py` is the route to the submitted Table 1's values (`heldout/README.md`).
- **Table 2, second row.** `records/timing_t1_analysis_subset.json` ships. It holds the aggregate analysis of the
  first Ego4D timing study: the baseline's mean seconds (`primary.mean_D_meas`), the ratio with its interval
  (`ci.X_meas`) and the attempted and jointly completed windows (`primary.n_A`, `primary.n_J`). Table 2 prints
  these values rounded. No command prints that row, and the study's per-window records are not released.
- **Table 2, held-out log rows.** `records/2026-10-31-412-cost-gap.json` ships. Its `ladder.analysed_H` and
  `ladder.analysed_D` give both corpora's logged seconds per unit (`theirs` for DO AS I DO, `ours_two` for HandReel
  over two intrinsics settings) and their ratio (`r_two`). Table 2 prints these values rounded, and its caption
  gives their qualifiers: the rows are a log, not a timing study. No command prints those rows.
- **The held-out verdict's per-unit inputs are in no record of this list.** The verdict and secondaries records hold the
  analysis results and the sha256 of the per-unit inputs, not the per-unit centre errors that `run/study/e1_stats.py`
  reads. Those errors are not released: regenerate them after acquiring the datasets (`heldout/README.md`); see
  "Unshipped inputs of the held-out verdict" for what is missing.
- **Paper revision.** `manifest.json` pins the paper source revision `2b6e3f934` (full SHA in `manifest.json`), the
  paper-source revision that generated the pinned tables. The authors report that the run on their full
  internal tree for the submitted revision matched the pinned generator and 31 records and regenerated all 23
  outputs with the expected hashes. If the published paper's records or generator differ from the pinned revision, re-pin with
  `--repin` before the check means anything for it.

## The generator's tables

The generator writes 22 tables, and `run/paper/e1_table.py` writes `e1_results.tex`. `manifest.json` lists each
with its sha256 and the records it reads. From this release alone, `regenerate_tables.py` regenerates these five:

| table | records it reads | what it holds |
|---|---|---|
| `solve.tex` | `solve_arms_contrasts.json`, `solve_arms_levels.json`, `lm_solve_paired.json`, `solve_perframe_contrast.json`, `solve_window_weighted.json` | five pipeline variants on 44 development units (42 DexYCB clips and 2 HOI4D windows, from 20 videos), with and without the windowed solve. The supplement's table on the windowed solve's effect reports centre-error differences: the windowed-solve-only comparison (the paper's variant C against A) is `paired_vs_init.core_lm.e_mm` in `lm_solve_paired.json`, and the other paired comparisons are in `solve_arms_contrasts.json`, whose record arm C is the paper's variant E (the supplement, Further Ablations) |
| `stride.tex` | `stride_safe_carry.json` | for each scored ground-truth channel, the largest stride, in study-grid frames, for which the upper 95% confidence bound on the 95th-percentile carry error meets the registered tolerance in every evaluable contact phase at that stride and at every smaller one. The table prints the stride selected by that rule (`ub` in the record), not the point estimate. The study grid is the rate the record gives as `rate_hz` (30 Hz for DexYCB and OakInk2, 15 Hz for HOI4D); it need not be a dataset's native capture rate, so a stride of one keeps every study-grid frame and says nothing about native samples. A historical table: it does not describe the input cadence the pipeline runs at |
| `optim.tex` | `summary_first_order.json`, `summary_multistart.json`, `summary_second_order.json` | other step rules applied to the solve's objective, each compared with LM within its own development cohort: second-order rules on 44 units (undamped Gauss-Newton succeeds on 42), first-order rules on 8 and nine-start LM on 11; the time is a median paired solve-time ratio, not pipeline wall time |
| `weak_rows.tex` | `weak_rows_summary.json` | per-row centre and off-orbit changes, paired within each weak stratum: the historical C-B labels compare the solve with depth layers against the initialiser with depth layers. Medians are formed within windows and then across windows. The table comes from the legacy full supplement |
| `hyper.tex` | none: the generator prints settings from the held-out study as text. See `run/conf/config.yaml` for the release defaults and `reproduce/heldout/` for the historical QC constants | the pipeline's settings: the defaults at the tree pinned for the held-out test. Its description of the automatic keyframe abbreviates the ranking: the implementation (`src/fastdo/prompt/choose.py`) multiplies the interaction score by the held-like probability and takes the highest product |

The other 17 tables and `e1_results.tex` each read at least one record that is not released.

## Records

Each record is an input of the command; sha256 prefixes below, full values in `manifest.json`. The "read by" column
names the generator tables that read each record (the `reads` lists of `manifest.json`). Records that ship are
listed by their release names; the others by a descriptive label. `docs/record_names.md` maps every name and label
to the paper source's file name, which `manifest.json` keeps as each record's `path`. The 14 records marked
**proposed for release** are in `reproduce/records/`, byte for byte under their release names, and await the
authors' release decision. The authors scanned each before adding it: no list or table of per-unit, per-window or
per-sequence rows, no dataset unit, video or sequence identifier (DexYCB, HOI4D, HO3D-v3 and Ego4D patterns), no
frame-derived array and no file path. The other 17 are not released, for the reason given.

| record | read by | sha256 | contents | release |
|---|---|---|---|---|
| `records/2026-10-26-e1-secondaries.json` | `bench.tex`, `heldout.tex`, `main.tex`, `main_time.tex` | `65f91846ec96` | the held-out test's secondary analyses: HOI4D/DexYCB unit identifiers with the sha256 of each unit's files, and per-camera and per-corpus summaries; no per-unit centre errors | not released: per-unit or dataset-derived rows |
| `records/2026-10-26-e1-verdict.json` | `bench.tex`, `coverage.tex`, `e1_results.tex`, `evidence.tex`, `heldout.tex`, `main.tex`, `main_time.tex`, `t2_evidence.tex` | `8cc081ede8f8` | the held-out test's verdict; DexYCB unit identifiers in its exclusion bookkeeping | not released: per-unit or dataset-derived rows |
| `records/2026-10-29-shrm-verdict.json` | `evidence.tex`, `speed.tex`, `speed_model.tex`, `speed_sens.tex`, `t2_evidence.tex` | `0c89bf78e0c7` | Ego4D video identifiers | not released: per-unit or dataset-derived rows |
| `records/2026-10-29-stage-seconds-r0.json` | no generated table | `eca664f80896` | aggregate only | not released: aggregate only, but names internal file paths |
| `records/2026-10-31-408-speed-attribution.json` | `decomp.tex`, `evidence.tex`, `t2_evidence.tex` | `4fb7ec8271b5` | aggregate only | not released: aggregate only, but names internal file paths |
| `records/2026-10-31-412-cost-gap.json` | `bench.tex`, `evidence.tex`, `heldout.tex`, `main.tex`, `main_time.tex`, `speed.tex`, `speed_model.tex`, `t2_evidence.tex` | `1a432df542e7` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `timing_t1_cost_records` | no generated table | `38fcb121de59` | per-window Ego4D video identifiers with timings | not released: per-unit or dataset-derived rows |
| `records/SOURCES.json` | no generated table | `984d62aab2bc` | provenance index of the records (names internal source paths) | not released: provenance index of the records; names internal file paths |
| `records/medians.json` | `speed.tex`, `speed_model.tex` | `c925dbcc24a2` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/solve_perframe_contrast.json` | `ablate.tex`, `evidence.tex`, `solve.tex`, `t2_evidence.tex` | `53db0fa5f59f` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `perframe_contrast_units` | `ablate.tex` | `53e7e5ffee2b` | per-unit DexYCB errors and Ego4D per-window values | not released: per-unit or dataset-derived rows |
| `comparators_reported` | `bench.tex`, `dexycb.tex`, `dexycb_ci.tex`, `ho3d_full.tex`, `main.tex`, `main_time.tex`, `supp_reported.tex` | `dd9eaf467513` | comparators' published values parsed from a study script | not released: comparators' published values, with a note naming an internal script and commit |
| `dexycb_benchmark_scores` | `bench.tex`, `coverage.tex`, `dexycb.tex`, `dexycb_ci.tex`, `evidence.tex`, `ho3d_full.tex`, `main.tex`, `main_time.tex`, `supp_reported.tex`, `t2_evidence.tex` | `85cee8de94d2` | per-unit DexYCB scores | not released: per-unit or dataset-derived rows |
| `ho3d_benchmark_scores` | `bench.tex`, `coverage.tex`, `evidence.tex`, `ho3d_full.tex`, `main.tex`, `main_time.tex`, `supp_reported.tex`, `t2_evidence.tex` | `fd03703bac57` | per-window HO3D-v3 scores | not released: per-unit or dataset-derived rows |
| `records/stride_safe_carry.json` | `ablate.tex`, `stride.tex` | `fcca22904d2e` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/timing_t1_analysis_subset.json` | `bench.tex`, `coverage.tex`, `evidence.tex`, `main.tex`, `main_time.tex`, `speed.tex`, `speed_model.tex`, `speed_sens.tex`, `t2_evidence.tex` | `28a0b48956ee` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `hoi4d_rerun_analysis` | `bench.tex`, `evidence.tex`, `main.tex`, `main_time.tex`, `t2_evidence.tex` | `58bef342a837` | per-unit HOI4D values | not released: per-unit or dataset-derived rows |
| `records/hoi4d_rerun_descriptive.json` | `bench.tex`, `main.tex`, `main_time.tex` | `71364890104f` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/lm_solve_paired.json` | `ablate.tex`, `solve.tex` | `0d9422742c37` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/solve_arms_contrasts.json` | `ablate.tex`, `solve.tex` | `1848e64ea50a` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/solve_arms_levels.json` | `solve.tex` | `0467b663e2a7` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/solve_window_weighted.json` | `solve.tex` | `797204dc985a` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/weak_rows_summary.json` | `weak_rows.tex` | `5e9ec2f5f8dd` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/summary_first_order.json` | `ablate.tex`, `optim.tex` | `721ecc43a9f4` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/summary_multistart.json` | `ablate.tex`, `optim.tex` | `463f1651aa3a` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/summary_second_order.json` | `ablate.tex`, `optim.tex` | `927aeb62cd31` | aggregate only | **proposed for release**, in `reproduce/records/` |
| `records/transcribed.json` | `ablate.tex`, `bench.tex`, `dexycb.tex`, `dexycb_ci.tex`, `evidence.tex`, `heldout.tex`, `ho3d_full.tex`, `main.tex`, `main_time.tex`, `runtimes.tex`, `supp_reported.tex`, `t2_evidence.tex` | `e0c22d678bc4` | values transcribed from published papers and internal records | not released: values transcribed from published papers and internal records; names internal file paths |
| `records/wp1/wp1_delivery_cover.md` | no generated table | `14f505f96fd8` | transcription of published comparator results with quotations | not released: transcription of published comparator results, with internal delivery notes |
| `records/wp1/wp1_extra_refs.bib.txt` | no generated table | `87412f2931e3` | draft bibliography entry | not released: no table reads it |
| `records/wp1/wp1_tables.tex.txt` | `runtimes.tex`, `supp_reported.tex` | `a696c9ad52fd` | transcription of published comparator results with quotations | not released: transcription of published comparator results, with internal delivery notes |
| `records/wp1/wp1_verification.md` | no generated table | `7f407a1fdb7d` | transcription of published comparator results with quotations | not released: transcription of published comparator results, with internal delivery notes |

## Unshipped inputs of the held-out verdict

`run/study/e1_stats.py` and the study scripts computed the verdict record (`2026-10-26-e1-verdict.json`) and the
held-out values of Table 1 from the inputs below. Three of them are included in this candidate in `heldout/`, in four files proposed for release: the unit identifiers in two files, the registered size-calibrated
thresholds, and a reduced copy of the QC constants (`heldout/README.md`: sources, sha256 and what was left out). The
full registered QC file and the per-unit centre errors stay private, as does every other input below; none of those
is in this release or in the paper source. The secondaries record (`2026-10-26-e1-secondaries.json`) pins the sha256 of the score file, the unit list
and each per-unit run record. For this release the authors checked every kept copy against those values, and all of them match.
The verdict record itself is the exception: the paper's copy differs from the one the secondaries record hashed, because
a later paper commit (2759d3853) rewrote seven `doc` fields from absolute paths to repo-relative ones (five under
`at_bookkeeping`, two under `infra_unresolved_listing`) and changed nothing else; `reproduce/verdict_doc_fields.txt`
lists the seven key paths and both sha256 prefixes. Its sha256 in `manifest.json` is therefore not the value pinned
inside the secondaries record. Other records may carry the same path normalisation.

| input | what it is | kept privately as |
|---|---|---|
| per-unit score file | one row per assigned unit (180 rows): unit, corpus, cluster, stratum, and the delivery and centre error of DO AS I DO and of HandReel under each of its two intrinsics settings. `summarise` does not take all 180: the wrapper passes only the 143 analysed rows (58 and 85), those that ran and were not removed (`e1_verdict_from_scores.py`, `cells`), so a direct caller must make the same selection or the two removed DexYCB timeouts enter | an uncommitted file; sha256 `efcb767640df` in the secondaries record. **Not released:** per-unit errors derived from HOI4D and DexYCB ground truth are never shipped. For the submitted estimated-intrinsics comparison, use `heldout/regenerate_unit_errors.py` (`heldout/README.md`) to write a score file from your own runs on the 180 assigned units, after acquiring the datasets; it does not reproduce this file's second intrinsics setting |
| per-unit run records | the unit status log, the result logs of HandReel's two intrinsics settings, HandReel's queue log and per-turn records, each unit's N' file and the sensitivity record | uncommitted files; sha256 values in the secondaries record |
| unit list | the registered draw of held-out units | a committed file in the authors' repository; sha256 in the secondaries record. **Proposed for release:** `heldout/units_hoi4d_72.json` and `heldout/units_dexycb_s08_108.json`, without the per-window `shift_px` (derived from HOI4D's annotations) |
| QC rule file | the acceptance rule that HandReel ran with, registered by sha256 (`e564b52e92d3`) before the run | an uncommitted file. **Proposed for release in reduced form:** `heldout/qc_rule_constants.json` holds its constants, unrounded, without the development clip IDs, host paths and internal commit of its provenance, so its sha256 is not the registered one (`heldout/README.md`) |
| threshold JSON | the size-calibrated one-sided thresholds per corpus that `load_thresholds` reads; the output of the power script's size-only run | a committed file in the authors' repository. **Proposed for release:** `heldout/e1-power-size.json`, byte for byte |
| power script | the size and power simulation that wrote the threshold JSON | a committed script in the authors' repository |
| DO AS I DO's sealed outputs | DO AS I DO's packed per-unit outputs, sealed by sha256 as each unit finished. The registered order verifies every hash and unseals them before the single scoring pass (`heldout/protocol/`, the registration's unseal rule and amendment A-U); this release holds that registered order, not a receipt of the execution | uncommitted files with their seal manifest |
| study and scoring scripts | unit selection, both pipelines' runners, sealing, scoring, the secondary analyses and the verdict | committed scripts in the authors' repository |

Never preserved: two of HandReel's queue turns left no result record. Every other per-unit input above exists.

## Reproducibility gaps

1. **Records.** 14 aggregate-only records are proposed for release in `records/` (see "Records"). Eight records
   hold per-unit or per-window rows, or unit, video or sequence identifiers, on DexYCB, HOI4D, HO3D-v3 or Ego4D, and
   are not released. Withholding them is the authors' choice while each dataset's terms are reviewed. It is not a
   finding that the terms forbid it: DexYCB, for example, is CC BY-NC 4.0. Nine more hold no per-unit data and are
   not released. Eight name internal file paths or carry internal notes: two aggregates, the provenance index and
   five files that record or document comparator or transcribed values. The ninth is an unused draft bibliography
   entry. The released check writes five generator tables, and none
   of them is a table of the submitted main paper.
2. **The table generator.** `tools/make_tables.py` lives in the paper source (pinned by sha256) and is not released:
   its comments and the provenance strings it prints into the generated tables name internal files.
   `make_released_tables.py` holds its functions for the five tables above, without those comments; it matches the
   generator's tables on every line except the `%` comment lines.
3. **The paper source holds the submitted Tables 1 to 5 as hand-written files.** No command here prints them.
   `docs/paper_exhibits.md` says what this release holds toward each.
4. **Figures.** Figs. 1, 3 and 4 show frames of DexYCB or HOI4D clips with per-unit overlays; a renderer needs those
   frames and the per-unit reconstructions, and this release ships neither. Fig. 2 combines a method diagram with
   example frames and reconstruction overlays, and Fig. 5 is a drawing in the paper source; no script here draws
   them. `docs/paper_exhibits.md` lists the assets that each figure needs.
5. **No experiment can be rerun from this release.** The scripts that ran the held-out test, the fresh HOI4D
   windows, the Ego4D timing studies, the HO3D-v3 and DexYCB scoring and the ablations are not in this release, and
   their inputs (ground truth, sealed outputs, most unit lists) are licensed or not released.
   `run/study/e1_stats.py` is included in this candidate with the unit lists, thresholds and `heldout/regenerate_unit_errors.py`, so a user
   can recompute the held-out verdict from per-unit errors that the user regenerates with the datasets, but cannot
   rerun the test. `packages/alongray` (with `alongray-coverage`) and `alongray_shares.py` ship as code, but not
   with the inputs of any exhibit.
6. **Paper revision.** See "What the check covers, and what it does not": the pin is `2b6e3f934`, the paper-source
   revision that generated the pinned tables; the authors report that the run on their full internal tree for the
   submitted revision matched the pinned inputs and regenerated outputs. Re-pin with one `--repin` command.
7. **Held-out verdict inputs.** See "Unshipped inputs of the held-out verdict": the unit lists and thresholds are included
   in this candidate in `heldout/` (the per-unit errors are not; regenerate them with the datasets), with the QC rule's constants (not the
   hash-registered file itself); the run records and the sealed outputs do not.

## Rights

This folder holds code, file names, sizes and sha256 values, the 14 aggregate-only records proposed for release, and
`heldout/`: the held-out test's protocol and the HOI4D and DexYCB unit IDs (identifiers from HOI4D and DexYCB, both
CC BY-NC 4.0; attribution in `heldout/README.md`); no per-unit error. No dataset frames,
reconstructions or model weights.

The check with `--paper-src` needs the paper source and its other records, which the authors provide only under the
datasets' terms.

Elsewhere in this release, code and tests name some public DexYCB development sequences (from subjects 01, 03, 04, 06
and 09, for example `20200820-subject-03/20200820_135250`): the data-gated tests in `tests/policy/dexycb_data.py`,
`tests/io/` and `tests/run/` load them when DexYCB is on the host. Some of these tests also record values derived
from those sequences: `tests/run/test_build_dexycb_hands.py` holds, for four named sequences, mean joint errors,
label fractions and frame lists, in its docstring and in the constants its assertions compare against. They are
development diagnostics, not raw media and not held-out inputs. These are directory names of DexYCB's public release, not held-out units. Outside `heldout/`,
no sequence of the held-out subjects 07 and 08 is named (the subject-07 names in `tests/io/test_dexycb_reader.py` and
`tests/run/test_build_dexycb_hands.py` are synthetic, for the refusal tests). `heldout/units_dexycb_s08_108.json`
names the 108 subject-08 views of the held-out test, and `heldout/units_hoi4d_72.json` the 72 HOI4D windows. No
DexYCB or HOI4D data ships.
