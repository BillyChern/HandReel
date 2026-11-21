# Development notes: the engine, the layout, the evaluation checks and the benchmark

Material moved from the README on 2026-11-21 so that the README stays short. Paths are relative to the repository
root.

## The engine

HandReel (Python package `fastdo`) is a staged engine (decode, mask, shape, intrinsics, hands, geometry, tracks, solve,
then the optional qc and retarget stages), configured with Hydra (`run/conf/config.yaml`). The mask, shape, intrinsics,
hands, geometry and tracks stages use pinned pretrained models (SAM 2.1, SAM 3D Objects, MoGe-2, HaWoR, CoTracker3).
Each stage writes its artifacts per window. The solve stage fits the object's poses and scale over the window. [`release/MODELS.md`](../release/MODELS.md) lists the external
files for the released reconstruction configurations, their sources and licensing information, and the SHA-256 hashes
recorded on the authors' system. The release also contains research code for retargeting to a robot hand
and for physics-based tracking; these do not yet run from this release (see ["Known gaps"](../README.md#known-gaps)).

## Evaluation

Most of the paper's records are not in this release, and no script here reruns a paper experiment. Two checks run
from this release:

- `python3 reproduce/regenerate_tables.py --out <new or empty dir>` checks all 14 aggregate-only records that this
  release includes in `reproduce/records/`. It uses ten of them to
  write four tables of the paper source's table generator and writes a fifth from settings in the generator. It
  compares the five outputs with that generator's output at a pinned paper-source revision, ignoring comment lines. None of
  the five is a table of the submitted main paper, whose Tables 1 to 5 the paper source writes by hand.
- With per-unit errors that you regenerate after acquiring HOI4D and DexYCB,
  `reproduce/e1_verdict_from_scores.py` recomputes the error ratios, intervals and median errors of Table 1
  (`reproduce/heldout/README.md`).

[`reproduce/README.md`](../reproduce/README.md) says what each check covers, and [`docs/paper_exhibits.md`](paper_exhibits.md) maps every exhibit of the submitted
main paper to what this release holds toward it.

## The benchmark entry point

The benchmark entry point runs the reconstruction on DO AS I DO's whisking demo clip (from the baseline submodule). As
configured, it reruns 3 identical windows of that one clip:

```bash
FASTDO_GPU_MEM_GB=32 uv run --frozen --extra dev --extra frontend \
  python run/pipeline/bench_throughput.py paths.root=$PWD \
  'bench.stages=[decode,gate,mask,shape,intrinsics,hands,geometry,tracks,solve]'
```

The demo supplies the baseline clip, its SAM 3D object mesh and the pre-optimization keyframe layout. Before timing,
the runner projects the mesh to make one silhouette prompt for all three copies.

It logs a clips-per-GPU-hour figure. The logged rate divides the attempted copies by the summed stage hours, CPU stages
included; a copy that stops early contributes only its partial time. That figure is smoke throughput on duplicated windows, not an evaluation: the
three windows are the same clip, nothing is scored, and it is not comparable with the paper's reconstruction times. On this
clip the depth-anchor gate flags all three solves (the log reports FLAGGED; in one independent run it moved 31 of 46
frames, by up to 0.51 m), so do not read the benchmark's output as a reconstruction of good quality either.

The `qc` stage needs a fitted acceptance rule (`run/pipeline/calibrate_acceptance.py`, then
`qc_stage.rule_path=<rule>`); retarget is opt-in and runs only on qc-accepted windows (`retarget` in `bench.stages`,
`retarget_stage.enabled=true`). Physics tracking runs in DO AS I DO's `retargeting` env.

## Repository layout

```
run/conf/          Hydra configs
run/pipeline/      entry scripts (benchmark, data builders, calibration, evaluation)
run/paper/         e1_table.py: the paper's held-out-test macros from the verdict record
run/study/         e1_stats.py: the held-out-test estimator (no other study script is in this release)
run/robot/         Unitree G1 package export
run/setup/         baseline environment setup
run/data/          dataset manifests and downloaders
src/fastdo/        the package
packages/          alongray and nullaudit, standalone packages with their own LICENSE and NOTICE (uv sync: nullaudit only)
release/           pinned worker-environment locks, the weights manifest, a one-clip example and its reference output
reproduce/         the table regeneration check and its records; the held-out test's protocol, unit IDs and verdict check
scripts/           setup_all.sh, fetch_baseline.sh, fetch_weights.py, view_clip.py, render_reference_png.py, dependency_licences.py
docs/              the audit tool, the timing protocol, the paper exhibits and the record names
licenses/          licence texts of the third-party code adapted here
third_party/       DO AS I DO official code, unmodified; scripts/fetch_baseline.sh clones it here at the pinned commit
tests/             pytest suite
```

`VERSION` holds the release version and its date. It names no commit: for that, run `git rev-parse HEAD` in your
clone, or see the commit on GitHub that your copy came from.
