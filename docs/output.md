# Output files: layout, keys, units and frames

What a run of the reconstruction writes (`README.md`, "Run on your own video", gives the short version). Every path, key and
convention below is the code's; the module is named beside each.

## Layout

Per window, a directory `<bench.artifact_root>/<clip_id>/` (default `outputs/engine/<clip_id>/`) with:

- the stage artifacts (`decode.npz`, `mask.npz`, `mesh.obj`, `intrinsics.npz`, `hands.npz`, `geometry.npz`,
  `tracks.npz`, `solve.npz`) and `profile.json` (seconds per stage; each stage's `peak_gpu_mem_gb` is the larger of two
  numbers when both exist: the peak CUDA allocation of the engine's own process, in decimal GB, and a peak that the
  stage's worker reports. Only the hands worker (HaWoR) reports one, and it reports GiB, so the field mixes two
  units. Allocations of workers whose measurements do not reach this profile, such as SAM 3D's, are
  missing, and several stages read 0.0:
  in the example decode, mask, shape and geometry. It is not the card's peak);
- `mask_prompt.npz`, the keyframe object mask the mask stage starts from: `keyframe_index` (a decode row, 0-d int64)
  and `mask` (H, W) bool (`fastdo.engine.workers_video.write_mask_prompt`), written before the stages run;
- `source_domain.json`, the clip's `clip_id`, `source` and `domain` (`fastdo.engine.source_domain.write_domain_record`),
  written before the stages run;
- side files that some stages write next to their artifact, for example `shape.json`, `sdf.npz` and `samples.npz`
  (shape), `hands_hawor.npz` (the HaWoR driver's raw output, which `hands.npz` is written from), and `solve.json` and
  `source_profile.json` (solve: the source-domain profile applied to the clip);
- `clip_record.npz` (the `ClipRecord` schema, `src/fastdo/io/clip_record.py`) when QC accepts the window; `qc.json` records the decision and rejected windows have no
  `clip_record.npz`. A resumed benchmark run (`bench.resume`) must be the same run: it is refused when the clip
  set differs, when the stage list or any config key other than `bench.resume` and `bench.overwrite` differs from
  the record that the first run left in the output directory (`resume_settings.json`), and when that record is
  absent. The record also covers the bytes of the rule files that the config names (`qc_stage.rule_path` and the
  two drift-screen constants files), so a rule replaced at the same path is a changed setting; other files that
  the config names are covered by path only. With `bench.resume` and no manifest in the output directory the run
  is a fresh one and refuses clip directories that already exist. After any change of settings, use a fresh
  artifact root AND a fresh output directory. A value that the config reads from the environment is not part of
  the record. The yield report does not count a window whose
  QC stage reads done while a stage before QC reads failed;
- with the opt-in retarget stage, robot-hand joint trajectories for qc-accepted windows.

The artifact root also holds `.gpu.lock`, an empty lock file that the GPU stages hold while they run
(`fastdo.engine.orchestrator.gpu_lock`); the pipelined runner holds it for its whole run. It stays after the run.
Each run writes `manifest.parquet` (stage state and seconds per window). The batched and pipelined runners also
append to `batches.jsonl` next to it, one line per committed stage-batch attempt with its clips, wall time and
errors (`fastdo.engine.batched_runner`); the serial runner (`fastdo.engine.run_manifest`) writes no such file. The
benchmark entry point writes `yield_report.json`.

**The shipped example stops at `solve.npz`.** No acceptance rule fitted for new video ships, so the qc stage does not run and no
`clip_record.npz` is written (`README.md`, "Known gaps"). The files below are what the example gives you on a GPU. Under
`--stub` on CPU, every stage artifact is an empty placeholder file (0 bytes). `release/reference/v2d_t3_ep011_cam_a/`
holds `solve.npz`, `hands.npz` (without its four MANO mesh arrays, two vertex and two face arrays, and without the MANO pose, shape and translation arrays), `intrinsics.npz`,
`mesh.obj` and `source_domain.json` from one GPU run of the example, so you can read these files without a GPU.

## Units and frames

Depths and lengths are in metres. The solver projects a camera-frame point (X, Y, Z) with the intrinsics in
`intrinsics.npz` (`fastdo.solver.terms.project`). For Z of at least 1e-6 m its pixel in the decoded frames is
(fx X / Z + cx, fy Y / Z + cy); for smaller depths it uses 1e-6 m as the denominator. T counts
the rows of the file being read. When the optional drift QC truncates a window, `solve.npz` and an accepted
`clip_record.npz` keep the source frame indices of the kept prefix, while the decode, mask and hand files can still
cover the full window. Match rows by `frame_idx` and consult `n_rows` and `valid_range` in `solve.json`.

`solve.npz` (`fastdo.engine.workers_qc.write_solve_artifact`):

| key | shape, dtype | meaning |
|---|---|---|
| `poses` | (T, 4, 4) float32 | object-to-camera transform per row: rotation R = `poses[:, :3, :3]`, translation t = `poses[:, :3, 3]` in metres. The scale is not in this matrix |
| `scale` | () float64 | metres per unit of `mesh.obj`. A mesh vertex y is at `scale * R @ y + t` in the camera frame (default solve; with `anisotropic_scale` see the note on the shape fields below) |
| `covariance` | (T, 6, 6) float64 | per-row marginal covariance of the pose perturbation; note below the table |
| `per_frame_cost` | (T,) float32 | RMS pointmap residual per row, metres |
| `confidence` | (T,) float32 | per-row score in [0, 1], relative within the clip only; 0.0 for a row with too few pointmap points (`fastdo.solver.api.TrajectorySolution`) |
| `iterations`, `seconds` | () int64, () float64 | fine-level solver iterations; wall time of the solve batch divided by its number of clips |
| `frame_idx` | (T,) int64 | source frame index of each row |
| `track_weight` | (T, Q) float32 | final IRLS weight of each of the Q point tracks |

If an optional post-solve step moves the printed poses, `per_frame_cost` and `confidence` keep their solver values.

`covariance`: the perturbation is xi = [rho, theta] in `Exp(xi) @ pose` (`fastdo.geometry.lie.apply_left`; rho in
metres, theta in radians); the covariance comes from the solver's Gauss-Newton Hessian
(`fastdo.solver.banded.marginal_covariances`).

With the solver's `anisotropic_scale` option on (off by default), `solve.npz` also holds `shape_log_stretch` (3,),
`shape_axes` (3, 3) and `shape_centroid` (3,), float64 (`fastdo.solver.aniso`). `mesh.obj` and `scale` do not
include that stretch. For such a solve, first correct the template with
`fastdo.solver.aniso.corrected_template_points(points, solve)`, then apply scale and pose: `scale * R @ y' + t`.
The function reads the three shape fields as attributes. An `NpzFile` from `np.load` has keys, not attributes, so
passing it directly leaves the points uncorrected without an error. Pass its attribute view `z.f` and a
floating-point vertex tensor:

```python
import numpy as np
import torch
from fastdo.solver.aniso import corrected_template_points

with np.load("solve.npz", allow_pickle=False) as z:
    y_corrected = corrected_template_points(torch.as_tensor(vertices, dtype=torch.float64), z.f)
    scale = float(z["scale"])
```

Here `vertices` is the (N, 3) vertex array of `mesh.obj`. The call returns the corrected template only; scale and
pose are applied afterwards as above.

`hands.npz` (`fastdo.engine.workers_hands.write_hands_artifact`), for each side `left` and `right`:

| key | shape, dtype | meaning |
|---|---|---|
| `{side}_params` | (T, 48) float32 | MANO pose, axis-angle: global orientation (3), then 15 joints (45) |
| `{side}_betas` | (10,) float32 | MANO shape, the mean over the valid rows |
| `{side}_trans` | (T, 3) float32 | MANO translation, camera frame, metres |
| `{side}_joints` | (T, 21, 3) float32 | joint positions, camera frame, metres |
| `{side}_vertices` | (T, 778, 3) float32 | MANO mesh vertices, camera frame, metres |
| `{side}_faces` | (1552, 3) int32 | triangles of the MANO mesh, indices into the vertices |
| `{side}_valid` | (T,) bool | rows with a hand estimate. On the batched HaWoR path (the example), the other rows hold copies of the nearest valid row: mask each side by `valid`. A wholly missing hand has an all-false mask and zero pose, joints and vertices; the face topology can still be present. Valid rows include filled interior gaps |
| `{side}_infilled` | (T,) bool | rows HaWoR filled in across a detection gap (all False when its output has no such key, so an all-false mask cannot establish that the hand was detected) |

plus `frame_idx` (T,) int64, `camera_K` (3, 3) float64 (the camera the hand outputs are in: the `K` of
`intrinsics.npz`) and `hawor_camera_K` (3, 3) float64 (HaWoR's own camera: focal length `K[0, 0]`, principal point at
the image centre).

`intrinsics.npz` (`fastdo.geometry.intrinsics.write_intrinsics`): `K` (3, 3) float32, the pixel intrinsics of the
decoded frames; `source`, a 0-d string, `"moge2"` when MoGe-2 estimated `K` (the example) or `"given"` when the clip
directory has a `camera.json`; and `focal_scale` (0-d float64) when the focal correction is on. The shipped
`run/conf/config.yaml` turns it on by default: `geometry_stage.intrinsics_focal_scale: 1.1040399273866555` multiplies
the MoGe-2-estimated `fx` and `fy` by that pooled lab-camera constant (the example's `focal_scale` is 1.10404), and a
`camera.json` clip is not affected. `--set geometry_stage.intrinsics_focal_scale=null` turns it off.

`mesh.obj` (`fastdo.engine.workers_shape`, `_write_mesh`): the object mesh in canonical units, with its bounding-box
centre at the origin and a bounding-box diagonal of 1. This is the frame of `y` above. `shape.json` records the map from
the shape model's raw output: canonical = (raw - `normalization_center`) * `normalization_scale`.

`geometry.npz` holds the metric MoGe-2 pointmaps in metres. By default (`geometry_stage.pointmap_storage: masked`) it
keeps only the object pixels of each row and the keyframe's full pointmap; `fastdo.engine.geometry_io.unpack_pointmaps`
rebuilds the array from it and the matching `mask.npz`: the full keyframe and the saved object pixels of the other
rows, with off-mask entries NaN. A reader that needs full background measurements must select dense storage upstream.

## Conventions of `clip_record.npz`

From `src/fastdo/io/clip_record.py` (written only when QC accepts the window):

| field | shape | meaning |
|---|---|---|
| `frame_idx` | (T,) int64 | source frame indices of the window |
| `timestamps` | (T,) float64 | seconds |
| `intrinsics` | (3, 3) float32 | pinhole camera matrix |
| `cam_to_world` | (T, 4, 4) float32 | camera-to-world transform per frame. It need not be measured camera motion: without a camera transform from the geometry stage, every row repeats one up-to-world rotation with zero translation, and a missing up vector defaults to [0, -1, 0]. `qc.json` records `camera_source` and `gravity_source` |
| `object_pose` | (T, 4, 4) float32 | object-to-camera transform per frame |
| `object_scale` | float | metres per unit of the canonical mesh `mesh.obj` (`object_mesh_path`): `solve.npz` `scale` |
| `gravity_cam` | (3,) float32 | up vector in the camera frame of frame 0 |
| `confidence` | (T,) float32 | per-frame solve confidence copied from `solve.npz`; relative within this clip |
| `hand_left`, `hand_right` | `HandTrack` or none | MANO `params` (T, 48: global orientation + 15 joints, axis-angle), `betas` (10,), `trans` (T, 3) and `joints` (T, 21, 3) in the camera frame, `valid` (T,) |

Lengths are in metres: the translations of `cam_to_world` and `object_pose`, and the hand `trans` and `joints`
(`src/fastdo/engine/workers_geometry.py`, `src/fastdo/engine/workers_shape.py`). `object_scale` is `solve.npz` `scale`; `qc.json` `poses_source` names the pose used for export, `solve.npz` or
the matching contact-stage correction (`fastdo.qc.export`). A vertex y of `mesh.obj` is at
`object_scale * R @ y + t` in the camera frame, with R and t the rotation and translation of `object_pose`. The
exported mesh and scale omit the stretch of an `anisotropic_scale` solve; apply the shape fields of `solve.npz`
first, as described under `solve.npz`.
