# Record and table names: this release and the paper source

The paper source names some records and tables after internal study numbers. This release ships its records and
the weak-rows table under descriptive names instead, and `reproduce/README.md` and `docs/paper_exhibits.md` use those
names, or a descriptive label for a record that is not released. This table maps each name back to the paper source's
file name, which `reproduce/manifest.json` keeps as the record's `path` (the name `--paper-src` opens) next to its
`release_path`. A renamed record's bytes, and so its sha256, are unchanged.

| name in this release | released? | paper source's file name | sha256 prefix |
|---|---|---|---|
| `reproduce/records/solve_perframe_contrast.json` | yes | `records/s281_analysis.json` | `53db0fa5f59f` |
| `reproduce/records/stride_safe_carry.json` | yes | `records/s396_partA_safe.json` | `fcca22904d2e` |
| `reproduce/records/timing_t1_analysis_subset.json` | yes | `records/s405_analysis_subset.json` | `28a0b48956ee` |
| `reproduce/records/hoi4d_rerun_descriptive.json` | yes | `records/s432-descriptive.json` | `71364890104f` |
| `reproduce/records/lm_solve_paired.json` | yes | `records/s435_paired.json` | `0d9422742c37` |
| `reproduce/records/solve_arms_contrasts.json` | yes | `records/s524_contrasts.json` | `1848e64ea50a` |
| `reproduce/records/solve_arms_levels.json` | yes | `records/s524_levels.json` | `0467b663e2a7` |
| `reproduce/records/solve_window_weighted.json` | yes | `records/s548_summary.json` | `797204dc985a` |
| `reproduce/records/weak_rows_summary.json` | yes | `records/s566_summary.json` | `5e9ec2f5f8dd` |
| `timing_t1_cost_records` | no | `records/2026-10-31-s405-cost-records.json` | `38fcb121de59` |
| `perframe_contrast_units` | no | `records/s282_analysis.json` | `53e7e5ffee2b` |
| `comparators_reported` | no | `records/s395_reported.json` | `dd9eaf467513` |
| `dexycb_benchmark_scores` | no | `records/s395_score_x1.json` | `85cee8de94d2` |
| `ho3d_benchmark_scores` | no | `records/s395_score_x2.json` | `fd03703bac57` |
| `hoi4d_rerun_analysis` | no | `records/s432-analysis.json` | `58bef342a837` |
| `tables/weak_rows.tex` (written by `reproduce/make_released_tables.py`) | printed from the record | `tables/s566.tex` | content compared without `%` lines |

Every other record and table keeps the paper source's name.

`reproduce/README.md`, "The generator's tables", lists the five outputs of `reproduce/regenerate_tables.py`. Four
tables use the records listed there; `hyper.tex` uses settings in the generator and reads no record. The command
prints the five with the paper generator's functions and compares them with that generator's output at the pinned
paper-source revision, ignoring comment lines. None of the five is a table of the submitted main paper, whose
tables the paper source writes by hand.

Study numbers that remain in this release, and why:

- inside the two renamed records `solve_arms_levels.json` and `hoi4d_rerun_descriptive.json`, a few keys and one note
  name study numbers. Renaming them would change the records' pinned sha256, so they stay, and
  `reproduce/make_released_tables.py` reads those keys as they are;
- the weak-rows table keeps the paper's LaTeX label `tab:s566`, because the released check compares the table with
  the paper generator's output;
- `reproduce/manifest.json` keeps the paper source's file names as each record's `path`;
- the registered protocol documents in `reproduce/heldout/protocol/` keep the labels of internal records their rules
  cite; they are edited for public release only as `reproduce/heldout/protocol/redactions.json` lists
  (`reproduce/heldout/README.md`).
