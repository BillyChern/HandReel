# Data and third-party notices

This file names, by topic, every file of this release that carries something derived from a dataset or from a
third-party model or project, and says what that something is. It complements `NOTICE`, `THIRD_PARTY_LICENSES.md`,
`release/MODELS.md` and the `ATTRIBUTION.txt` of the reference example. Apart from the README figure described in the next section, the release contains no dataset frame, no
dataset annotation, no model weight and no hand-model geometry.

The datasets and their terms are listed in `THIRD_PARTY_LICENSES.md`. Users obtain each dataset from its publisher
under its own terms; Ego4D identifiers are not redistributed here.

## Dataset frames in the README figure

`docs/assets/gallery.jpg` is the only file of this release that shows dataset frames. It is not under the Apache-2.0
licence of the code: its frames come from HOI4D and DexYCB, both CC BY-NC 4.0 (non-commercial use, attribution).
It shows six development clips, one frame per tile.

- HOI4D (Liu et al., 2022, https://hoi4d.github.io/): tile "knife", sequence ZY20210800001/H1/C13/N48/S238/s03/T4,
  frame 88.
- DexYCB (Chao et al., 2021, https://dex-ycb.github.io/): tile "sugar box", sequence 20200709_142553 (subject 01),
  camera 841412060263, frame 36; tile "cracker box", sequence 20200813_145653 (subject 02), camera 932122061900,
  frame 36; tile "potted meat can", sequence 20200709_144814 (subject 01), camera 840412060917, frame 36; tile
  "tomato soup can", sequence 20200709_142926 (subject 01), camera 932122062010, frame 36; tile "pitcher", sequence
  20200813_153152 (subject 02), camera 932122062010, frame 36.

Changes made to the frames: each frame cropped and scaled, with the outlines of HandReel's posed object mesh and of
the estimated hand drawn over it; composed with its label into one figure and saved as JPEG.

## Outputs of third-party models on the example clip

The object mesh, the solve result, the camera intrinsics and the run metadata under `release/reference/` are outputs of this code and of the third-party models it calls (`THIRD_PARTY_LICENSES.md`, `release/MODELS.md`) on one clip that its publisher lists under CC BY 4.0 (`ATTRIBUTION.txt` in that folder). The licences of those models do not say whether outputs are derivative works; the files are provided for checking an installation and carry the models' terms as far as those apply.

Files (9):

- `release/reference/v2d_t3_ep011_cam_a/example_report.json`
- `release/reference/v2d_t3_ep011_cam_a/intrinsics.npz`
- `release/reference/v2d_t3_ep011_cam_a/mesh.obj`
- `release/reference/v2d_t3_ep011_cam_a/profile.json`
- `release/reference/v2d_t3_ep011_cam_a/shape.json`
- `release/reference/v2d_t3_ep011_cam_a/solve.json`
- `release/reference/v2d_t3_ep011_cam_a/solve.npz`
- `release/reference/v2d_t3_ep011_cam_a/source_domain.json`
- `release/reference/v2d_t3_ep011_cam_a/timing_ledger.jsonl`

## Aggregate result records

The records under `reproduce/records/` hold pooled statistics only: no unit identifier and no per-unit row. Each record names the data it was computed on.

Files (15):

- `reproduce/heldout/e1-power-size.json`
- `reproduce/records/2026-10-31-412-cost-gap.json`
- `reproduce/records/hoi4d_rerun_descriptive.json`
- `reproduce/records/lm_solve_paired.json`
- `reproduce/records/medians.json`
- `reproduce/records/solve_arms_contrasts.json`
- `reproduce/records/solve_arms_levels.json`
- `reproduce/records/solve_perframe_contrast.json`
- `reproduce/records/solve_window_weighted.json`
- `reproduce/records/stride_safe_carry.json`
- `reproduce/records/summary_first_order.json`
- `reproduce/records/summary_multistart.json`
- `reproduce/records/summary_second_order.json`
- `reproduce/records/timing_t1_analysis_subset.json`
- `reproduce/records/weak_rows_summary.json`

## Held-out unit lists

`reproduce/heldout/` lists the identifiers and frame windows of the held-out units (HOI4D and DexYCB, both CC BY-NC 4.0; attribution in `reproduce/heldout/README.md`). The lists are split definitions: they contain no frame, no annotation and no error. The datasets themselves must be obtained from their publishers under their terms.

Files (7):

- `reproduce/heldout/protocol/2026-10-02-prereg-e1-heldout.md`
- `reproduce/heldout/protocol/2026-10-14-e1-amendment-A-T.md`
- `reproduce/heldout/protocol/2026-10-19-e1-amendment-A-V.md`
- `reproduce/heldout/protocol/2026-10-22-e1-amendment-A-W.md`
- `reproduce/heldout/regenerate_unit_errors.py`
- `reproduce/heldout/units_dexycb_s08_108.json`
- `reproduce/heldout/units_hoi4d_72.json`

## Constants fitted, measured or chosen on datasets

These files hold settings (constants, thresholds, defaults) that were fitted, measured or chosen on DexYCB, HOI4D or Ego4D runs, or on the demonstration that the DO AS I DO authors release with their code. Each file states its own source beside the value. No dataset frame or annotation is included.

Files (14):

- `reproduce/heldout/qc_rule_constants.json`
- `src/fastdo/amortised/fallback.py`
- `src/fastdo/bench/yield_model.py`
- `src/fastdo/contact/__init__.py`
- `src/fastdo/contact/constants_production.json`
- `src/fastdo/engine/contact_ray_smoother.py`
- `src/fastdo/engine/workers_contact.py`
- `src/fastdo/geometry/ego_motion.py`
- `src/fastdo/qc/interaction.py`
- `src/fastdo/shape/wall_qc.py`
- `src/fastdo/smooth/config.py`
- `src/fastdo/solver/api.py`
- `src/fastdo/tba/comparators.py`
- `tests/retarget/whisking_data.py`

## Pooled focal multiplier

The default correction to the estimated focal length is one number fitted with equal weight on three datasets; the fixture that records the fit names them and holds summaries from a fourth. One of the three carries a share-alike tag at its host. The number is a pooled statistic; no dataset content is included.

Files (8):

- `docs/output.md`
- `release/reference/v2d_t3_ep011_cam_a/README.md`
- `release/reference/v2d_t3_ep011_cam_a/intrinsics.npz`
- `run/conf/config.yaml`
- `src/fastdo/geometry/intrinsics.py`
- `tests/fixtures/records/focal-scale-constants.json`
- `tests/fixtures/release/config_defaults_resolved.json`
- `tests/geometry/test_intrinsics_focal_scale.py`

## Hand-model indices

These files hold fingertip vertex indices and a joint-to-finger grouping of the MANO hand model. They contain none of the model's vertices, faces or skinning weights. MANO must be obtained from its authors under the MANO licence; this release never ships the model.

Files (3):

- `src/fastdo/hands/hawor_npz.py`
- `src/fastdo/retarget/labels.py`
- `src/fastdo/robot/dex3_retarget.py`

## Dataset names in code, tests and configuration

These files name sequences, subjects, camera serial numbers or clips of the datasets each file states, so that data-gated tests and loaders can find a user's own copy. A name is a locator, not data. No Ego4D video identifier appears in this release.

Files (34):

- `README.md`
- `reproduce/README.md`
- `run/conf/config.yaml`
- `run/data/sources.yaml`
- `run/pipeline/_dexycb_hands_driver.py`
- `src/fastdo/bench/bench_clips.py`
- `src/fastdo/engine/per_frame_reanchor.py`
- `src/fastdo/gate/dataset_dexycb.py`
- `src/fastdo/hands/hand_camera.py`
- `src/fastdo/policy/dexycb_inputs.py`
- `src/fastdo/robot/grasp_closure.py`
- `src/fastdo/shape/backend.py`
- `tests/bench/dexycb_tree.py`
- `tests/bench/test_bench_clips.py`
- `tests/bench/test_v2d_track3.py`
- `tests/fixtures/release/config_defaults_resolved.json`
- `tests/gate/fakes.py`
- `tests/gate/test_dataset.py`
- `tests/gate/test_labels.py`
- `tests/hands/test_hand_camera.py`
- `tests/io/test_dexycb_frames.py`
- `tests/io/test_dexycb_reader.py`
- `tests/io/test_dexycb_real.py`
- `tests/io/test_dexycb_world.py`
- `tests/io/test_hoi4d.py`
- `tests/policy/dexycb_data.py`
- `tests/policy/test_dexycb_refs_left.py`
- `tests/policy/test_stage.py`
- `tests/reproduce/test_heldout_release.py`
- `tests/run/test_bench_throughput.py`
- `tests/run/test_build_dexycb_hands.py`
- `tests/run/test_gate_pipelines.py`
- `tests/shape/sam3d_kernel_pin_gate.py`
- `tests/solver/test_depth_anchor.py`

## Values tied to named development sequences

A few tests and one driver record values measured on named DexYCB development sequences (conversion checks against the dataset's own annotations), as `reproduce/README.md` (Rights) discloses. No held-out unit appears with a value anywhere in this release.

Files (6):

- `reproduce/heldout/protocol/2026-10-02-prereg-e1-heldout.md`
- `run/pipeline/_dexycb_hands_driver.py`
- `src/fastdo/hands/hand_camera.py`
- `tests/policy/dexycb_data.py`
- `tests/run/test_build_dexycb_hands.py`
- `tests/shape/sam3d_kernel_pin_gate.py`

## Quoted aggregate measurements

Comments, docstrings, test messages and documents in these files quote aggregate figures (errors, rates, timings) measured on the datasets named beside them. They are derived statistics, quoted for context; the program does not read them.

Files (64):

- `README.md`
- `docs/audit.md`
- `docs/paper_exhibits.md`
- `packages/alongray/README.md`
- `reproduce/README.md`
- `reproduce/heldout/README.md`
- `reproduce/make_released_tables.py`
- `run/conf/config.yaml`
- `run/data/sources.yaml`
- `run/pipeline/bench_sam2_variants.py`
- `run/pipeline/eval_solver_dexycb.py`
- `scripts/check_paper_quotes.py`
- `src/fastdo/audit/census.py`
- `src/fastdo/bench/bench_clips.py`
- `src/fastdo/engine/batched_runner.py`
- `src/fastdo/engine/geometry_io.py`
- `src/fastdo/engine/orchestrator.py`
- `src/fastdo/engine/shape_reuse.py`
- `src/fastdo/engine/source_profile.py`
- `src/fastdo/engine/workers_frames.py`
- `src/fastdo/engine/workers_geometry.py`
- `src/fastdo/engine/workers_qc.py`
- `src/fastdo/engine/workers_retarget.py`
- `src/fastdo/geometry/moge_model.py`
- `src/fastdo/io/datasets/dexycb_world.py`
- `src/fastdo/io/datasets/hoi4d.py`
- `src/fastdo/io/video.py`
- `src/fastdo/policy/bc.py`
- `src/fastdo/prompt/hands23_detect.py`
- `src/fastdo/qc/reference_rate.py`
- `src/fastdo/retarget/baseline_io.py`
- `src/fastdo/retarget/kinematic.py`
- `src/fastdo/retarget/kinematic_terms.py`
- `src/fastdo/retarget/rate.py`
- `src/fastdo/shape/backend.py`
- `src/fastdo/shape/fast_postprocess.py`
- `src/fastdo/shape/sam2_variants.py`
- `src/fastdo/shape/sam3d_backend.py`
- `src/fastdo/shape/sam3d_worker.py`
- `src/fastdo/solver/confidence.py`
- `src/fastdo/solver/ot_sparse.py`
- `src/fastdo/solver/terms.py`
- `src/fastdo/teacher/layout_reader.py`
- `tests/engine/test_orchestrator.py`
- `tests/engine/test_source_profile.py`
- `tests/engine/test_workers.py`
- `tests/engine/test_workers_qc.py`
- `tests/io/test_dexycb_world.py`
- `tests/io/test_hoi4d.py`
- `tests/io/test_video.py`
- `tests/io/test_video_window.py`
- `tests/policy/test_stage.py`
- `tests/prompt/test_hands23_client.py`
- `tests/reproduce/test_check_paper_quotes.py`
- `tests/retarget/test_contact.py`
- `tests/retarget/test_kinematic.py`
- `tests/retarget/test_kinematic_fast.py`
- `tests/retarget/test_wrist_target.py`
- `tests/run/test_bench_throughput.py`
- `tests/run/test_gate_pipelines.py`
- `tests/shape/test_shape_fast_options.py`
- `tests/solver/test_memory.py`
- `tests/solver/test_solve_synthetic.py`
- `tests/tba/test_carve.py`

## Robot and retargeting constants from upstream projects

These files reproduce constants from the robot maker's description files and from an upstream retargeting implementation, under the BSD 3-Clause, Apache-2.0 and MIT licences whose texts are in `licenses/`.

Files (2):

- `src/fastdo/robot/dex3_retarget.py`
- `src/fastdo/robot/g1_constants.py`

## One test fixture without a kept generator

`packages/alongray/tests/data/pool_bootstrap_golden_0.3.1.json` stores small input rows and the outputs that
version 0.3.1 of that package gave on them. The package's tests use synthetic inputs only; the generator of these rows
was not kept, and no dataset value is known to be in the file. It stays as recorded because regenerating it would
remove the comparison with 0.3.1 that the test exists for.
