# The submitted paper's exhibits

This page maps each exhibit and each headline claim of the submitted main paper to what this release contains toward
reproducing it. [`README.md`](../README.md) is the user documentation, and [`reproduce/README.md`](../reproduce/README.md) covers the commands.

**Which paper.** Every table, figure, section and algorithm number on this page is a number of the submitted main
paper (11 pages: Tables 1 to 5, Figs. 1 to 5, Algorithm 1, Secs. 1 to 7). Recheck them against the published version.
The supplement's numbering is not final, so this page names its parts without numbers. This page quotes no sentence
of the paper: it restates each condition and gives the place.

**What the checker checks, and the manual audit.** `scripts/check_paper_quotes.py --main <main paper PDF>` checks
five things: every cited exhibit number exists in the PDF with the caption or heading that this release declares for
it; no document cites a supplement number; every passage in double quotes in the checked Markdown documents is the
name of a section of this release's own documents, a passage that the script lists with its own non-paper source, or
verbatim in the main paper; the title matches; and no document contains a wording from a finite, hand-written pattern
list. Each check runs over the document lists that the script names (`DOCS`, `NUMBERS_ONLY`, `pattern_docs`), not
over every file. It is not a semantic check, and a pass verifies neither a claim's full scope nor its meaning. It does not test whether a sentence says what its cited
exhibit shows, and a paraphrase or a changed unit passes its patterns. A manual audit, made on 2026-11-17 for this
candidate, therefore compared each results sentence of `docs/results.md`, and of this page with
the words of the submitted main paper. Repeat that audit whenever a results sentence changes.

**What this release reproduces.** No script in this release reruns an experiment of the paper. The release ships no
benchmark video or image, no model weights and no per-unit benchmark score. This candidate does include the unit lists of the
held-out test (`reproduce/heldout/`, proposed for release) and one runnable example that downloads its public input separately, with reference outputs under `release/reference/`. Two
exhibits have a partial route:

- Table 1: from per-unit errors that you regenerate with the datasets, `reproduce/e1_verdict_from_scores.py`
  recomputes the error ratios, their intervals and the median errors, and compares them with the printed values.
- Table 2: two aggregate records hold the values behind its second row and its held-out log rows
  (`reproduce/records/timing_t1_analysis_subset.json`, `reproduce/records/2026-10-31-412-cost-gap.json`). No command
  prints those rows.

Every other exhibit is not reproducible from this release, for the reason its row gives.

## Exhibits

| Exhibit | What it shows | Paper location | In this release | Obtain separately |
|---|---|---|---|---|
| Fig. 1 | Visual abstract: HandReel carries one mesh with point tracks, beside an illustration of per-frame pose sampling, and the Ego4D reconstruction-time comparison of Table 2, rows 1 and 2 | Sec. 1 | Not reproducible from this release: the images show a DexYCB development clip, and the time comparison repeats Table 2 | Selected DexYCB frames and matching HandReel outputs for the overlays and 3D views |
| Fig. 2 | The method: one mesh per window, poses that start from point tracks, and a joint refinement of all poses and one scale | Secs. 1 and 4 | The diagram itself is not in this release; the stage implementations and the public example are. It contains DexYCB-derived frames and HandReel reconstruction overlays, so an exact copy needs those frames and outputs, which this repository does not ship. `release/example/run_example.py` runs the stages that the diagram names on one public clip | For the exact figure: the source DexYCB frames and the matching HandReel overlays. For the public example: the model weights and MANO (README, "Weights") |
| Fig. 3 | One DexYCB development clip from start to end: ground truth, DO AS I DO and HandReel in six frames, and each pipeline's object-centre error per source frame | Sec. 5.2 | Not reproducible from this release: the figure needs DexYCB frames and both pipelines' per-frame outputs, and the release ships neither | DexYCB; DO AS I DO's code |
| Fig. 4 | Gallery: HandReel's posed mesh and the estimated hand on one frame of each of six development clips | Sec. 5.2 | Not reproducible from this release: the figure needs DexYCB and HOI4D frames and HandReel's outputs on them | DexYCB; HOI4D |
| Fig. 5 | The two applications: (a) sums of separately measured stage times for one video; (b) the reconstruction times of Table 2, row 2, for many videos | Secs. 5.3 and 5.4 | Not reproducible from this release: the bars repeat Table 5 and Table 2, and no script here draws them | none |
| Algorithm 1 | The windowed solve: Levenberg-Marquardt over three blur levels | Sec. 4.2 | The solver code is in `src/fastdo/solver/` (the Levenberg-Marquardt loop in `lm.py`), and the example runs it as its solve stage. This release is a later commit than the tree frozen for the held-out test and the tree timed in the Ego4D timing studies (README, "Known gaps") | none |
| Table 1 | Held-out object-centre error on 58 HOI4D windows and 85 DexYCB views: median error on the units that both pipelines delivered, and the error ratio with its 95% confidence interval (the exponential of the median paired log-error ratio, under the registered rules for units that a pipeline failed to deliver) | Secs. 5.1 and 5.2 | The unit lists, the registered protocol and the thresholds (`reproduce/heldout/`). `reproduce/heldout/regenerate_unit_errors.py` then `reproduce/e1_verdict_from_scores.py` recompute the ratios, intervals and median errors from per-unit errors that you produce, and print each beside its reference value (stdout; `--json`). No script reruns the pipelines, the scorer or the quality check | HOI4D; DexYCB; the model weights and MANO; DO AS I DO's code |
| Table 2 | Reconstruction time per window: two Ego4D timing studies against DO AS I DO's reconstruction stage, the held-out log, and GraG on HO3D-v3 | Sec. 5.2 | Timing studies and GraG row: not reproducible from this release, which holds neither the studies' window lists nor their timer and analysis scripts. Row 2: `reproduce/records/timing_t1_analysis_subset.json` holds the first study's aggregate analysis (`primary.mean_D_meas`, the baseline's mean seconds; `ci.X_meas`, the ratio and its interval; `primary.n_A` and `primary.n_J`, the attempted and jointly completed windows), which Table 2 prints rounded. Held-out log rows: `reproduce/records/2026-10-31-412-cost-gap.json` holds the logged seconds per unit and their ratio (`ladder.analysed_H`, `ladder.analysed_D`: `theirs`, `ours_two`, `r_two`), which Table 2 prints rounded. No command prints these rows | Ego4D; HO3D-v3; DO AS I DO's and GraG's code |
| Table 3 | Published reconstruction timings for six methods (AGILE reports the times for HOLD and MagicHOI), beside the measured Ego4D times of Table 2, row 1 | Sec. 5.2 | Not reproducible from this release: the reported rows are values from the cited papers, and the measured rows repeat Table 2 | none to run. To check the reported values: the supplement's Reconstruction Specialists and Per-Frame FoundationPose subsection points to the published timing sources; AGILE's sixth table gives the HOLD and MagicHOI timings and its Table 1 gives accuracy on successful sequences with the success rates; 4D-HOF's Table 1 gives the reported Chamfer values |
| Table 4 | Error ratio on 31 fresh HOI4D windows under three configurations of DO AS I DO that vary its depth model and focal scale | Secs. 5.1 and 5.2 | Not reproducible from this release: it ships neither the window list nor the per-window errors | HOI4D; DO AS I DO's code |
| Table 5 | Stage times for one video, measured separately: reconstruction, scene build with retargeting, and physics, with their sums | Sec. 5.3 | Not reproducible from this release: the Table 2 record supports the reconstruction times, but the release provides no full set of stage records, no command to recreate Table 5 and no path from a reconstruction to a physics run (README, "Known gaps") | Ego4D; DO AS I DO's code |

## Headline claims

Each row restates one claim with the conditions that the paper attaches to it. The numbers are the paper's.

| Claim | Paper location | Conditions that the paper states | In this release |
|---|---|---|---|
| Across 58 analysed held-out HOI4D windows, the paired object-centre error ratio of HandReel to DO AS I DO was 0.50 under the registered failure rules, 95% interval 0.33 to 0.63 | Table 1; Sec. 5.2 | 58 analysed windows of 72, with ground-truth object prompts for both pipelines. A failure to deliver counts as an error, and a joint failure as a tie. HandReel delivered all 58 windows and DO AS I DO 49. On the 49 that both delivered, the median error was 56.8 mm against 101.3 mm | As Table 1 above |
| On held-out DexYCB, HandReel's ratio was 0.85, 95% interval 0.58 to 1.00, which is non-inferior at the 1.15 margin | Table 1; Secs. 5.1 and 5.2 | 85 analysed views of 108. The paper marks this verdict: it is not robust if joint failures count against HandReel, and the authors partly used this subject in development. DO AS I DO delivered 69 views. HandReel delivered 58 after its quality check rejected 27 outputs. Among the 48 views both delivered, median error was 76.6 mm for HandReel and 112.5 mm for DO AS I DO | As Table 1 above |
| On 31 fresh HOI4D windows, HandReel's median error ratio was below 1 under each of three configurations of DO AS I DO that vary its depth model and focal scale, from 0.46 to 0.65. The interval for the setting with HandReel's focal factor includes one | Table 4; Sec. 5.2 | Both pipelines used prompts derived from ground truth: the object mask for HandReel and three points for DO AS I DO. Both pipelines delivered every analysed window. A pre-registered rule excluded 2 of 33 windows before scoring | Not reproducible from this release |
| One window at a time, DO AS I DO's reconstruction stage took about 30x as long as HandReel: about 2,000 s against 66 s per window | Table 2, row 1; Sec. 5.2 | Second Ego4D timing study: the paper timed both pipelines one window at a time. The study drew 74 windows, both pipelines completed 25, and the paper analyses 20 of them by a rule that the authors set after the study ended. The bracket 25 to 37 is an envelope | Not reproducible from this release |
| When DO AS I DO restarted for every window, its reconstruction stage took 130x as long per window as HandReel's per-window share of batched runs: 2,100 s against a 17 s per-window share | Table 2, row 2; Sec. 5.2 | First Ego4D timing study, on the nine of 19 attempted windows that both pipelines completed, which is below the registered minimum of 10. 95% confidence interval 97 to 170. The paper timed the baseline's side. HandReel's 17 s is a share of batched runs, which the paper splits among windows in proportion to their stage seconds; it is not the time of one window alone (see "Timing boundaries") | Not reproducible from this release |
| With both in batched runs, where the paper models the baseline, DO AS I DO's reconstruction stage would take about 120x as long as HandReel's per-window share: a modelled value of about 1,600 s against a 14 s per-window share | Table 2, row 3; Sec. 5.2 | The baseline's seconds are a modelled value: two instances that run at once, with the throughput gain of 1.2 that two instances showed on three window pairs. HandReel's side is its per-window share of batched runs. The model uses the same 20 analysed windows as row 1. The bracket 87 to 150 is an envelope | Not reproducible from this release |
| Application 1, one video to a robot trajectory: DO AS I DO's full pipeline would take about 3x as long per window with its own reconstruction as with HandReel's, 3,069 s against 991 s | Table 5; Fig. 5; Sec. 5.3 | A sum of separately measured stage times; the paper times no full run. Both sums share one 894 s physics time and neither holds a gravity term; HandReel's sum also holds 2.63 CPU core-seconds for the certificate check. The 894 s is the median elapsed time of one planner arm, which the authors measured after HandReel's reconstruction at 2 perturbation samples (the released default is 4), while two planner arms ran at once on the same GPU. With HandReel reconstruction, physics is the largest term in the stage sum | Not reproducible from this release |
| Application 2, many videos with one retargeting policy: every video still needs its own reconstruction, 2,100 s per window for DO AS I DO when it restarts for every window against HandReel's 17 s per-window share of batched runs | Sec. 5.4; Fig. 5(b); Table 2, row 2 | Reconstruction time only. The paper makes no end-to-end timing of this pairing and gives no retargeting quality for it | Not reproducible from this release |

Ego4D has no ground-truth object poses, so the Ego4D studies measure speed only (Sec. 1); the error results in these
rows come from the held-out test and the fresh HOI4D windows. The paper reports further error analyses on
development data that these rows do not list. The paper also states limits that these rows do not repeat. Centre
error ignores orientation. Both pipelines delivered 13 of the 25 evaluated DexYCB views of objects without symmetry.
On those 13 views, HandReel had the higher median rotation error (Sec. 5.2). The experiments do not establish
rotation tracking (Sec. 6).

## Timing boundaries

The submitted paper states these boundaries for its timings. Each line restates one and gives its place.

- **Hardware.** Both pipelines ran on one H100 GPU with a 17-core CPU quota, on a shared host (Sec. 5.2).
- **What restarts.** DO AS I DO ran its released code. It restarted and reloaded its models for every window
  (Sec. 5.2).
- **Batched runs.** In the first study, HandReel made six batched runs. Each run covered 26 windows and loaded its
  models once, and the timing includes that load (Sec. 5.2). The second study also made batched runs of HandReel
  over its 74 windows (Sec. 5.2).
- **Per-window share.** HandReel's per-window share is the mean wall time of such batched runs, which the authors
  split among the windows in proportion to their stage seconds (Sec. 5.2). The 17 s of Table 2, row 2, and the 14 s
  of row 3 are such shares.
- **One window at a time.** In the second study, both pipelines ran one window at a time (Sec. 5.2; Table 2, row 1).
- **Completed windows.** A window counts as completed when DO AS I DO finishes and HandReel's solve runs, even if
  HandReel's quality check then rejects it (Sec. 5.2).
- **The modelled row.** Table 2 marks a modelled value with an approximation sign. In row 3 the paper models the
  baseline: two instances that run at once (Sec. 5.2).
- **Ratios and brackets.** Each ratio divides the baseline's seconds by HandReel's, from unrounded seconds, so
  dividing the printed seconds need not give the printed ratio. A bracket is a 95% confidence interval, and an
  envelope in rows 1 and 3 (Table 2, caption).
- **The held-out log.** The held-out rows of Table 2 are not a timing study. They give ratios of mean logged
  seconds, and HandReel's seconds there sum two intrinsics settings, at its frame stride and a higher scheduling
  priority (Table 2, caption).
- **GraG.** On seven of eight HO3D-v3 windows, the median per-window ratio of GraG time to HandReel time was 8.9. The
  study's GPU-time cap ended it before the eighth window ran. Both methods processed the same 24 frames per window
  and reloaded their models each time. GraG used its offline contact fallback. HandReel started from a ground-truth
  mask and skipped prompt detection (Sec. 5.2; Table 2).
- **Stage sums.** Table 5 sums stage times that the authors measured separately, and the sums time no full run
  (Table 5, caption; Sec. 5.3).
- **Reported times.** The published rows of Table 3 use each source's own input, hardware and timing scope (AGILE
  is the source for HOLD and MagicHOI), and the paper does not match them to its measured rows (Table 3, caption).

The supplement describes the cohorts behind the rows of Table 2, selected run settings and the stage terms behind
Table 5 (the supplement, Experiment Details and Additional Experiments). It does not list per-window identifiers
or every setting of every run.

**This release's own timing.** `release/example/run_example.py` logs the reconstruction seconds of one window on
your machine: the sum of its eight stage seconds, model loading included, with viewer time apart. That boundary
differs from every row of Table 2, and the example computes no ratio to another pipeline (`release/QUICKSTART.md`,
step 4).

## Tables that the paper-source generator writes

`reproduce/regenerate_tables.py` prints tables with the paper source's table generator and compares them with
pinned hashes (`reproduce/README.md`). From this release alone it checks 14 aggregate records, uses ten of them
to write `solve.tex`, `stride.tex`, `optim.tex` and `weak_rows.tex`, and writes `hyper.tex` from settings in the
generator. None of the generator's tables is
an exhibit of the submitted main paper: the paper source holds its Tables 1 to 5 as hand-written files. The
supplement is not final, and this release does not state which of these tables it prints. For the windowed-solve table in the
supplement's Further Ablations, the `solve.tex` row in `reproduce/README.md` identifies the aggregate behind each
paired comparison.
