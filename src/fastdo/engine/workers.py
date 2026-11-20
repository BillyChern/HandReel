"""The engine's stage workers, re-exported from their stage-family modules.

Every worker matches `fastdo.engine.orchestrator.StageFn`, reads the previous stages' artifacts from the clip's
artifact directory and writes `<stage>.npz` there (`shape` writes `mesh.obj`, `sdf.npz`, `samples.npz`, `shape.json`;
`gate` writes `gate.json`; `solve` writes `solve.npz` and `solve.json`; `qc` writes `qc.json` and `clip_record.npz`; `retarget` writes
`retarget.json` and `trajectory_kinematic.npz`; the opt-in `driftqc` stage writes `driftqc.json` and, unless it rejects the
clip, `valid_range.json`):

| stage | module | batching |
|---|---|---|
| `decode` | `workers_frames.make_decode_worker(sources)` | none (per clip) |
| `encoder` | `workers_frames.make_encode_worker` | frame-batched: crops of several clips share forward passes |
| `gate` | `workers_gate.make_gate_worker(cfg)` (needs a gate bundle) | none (per clip, CPU) |
| `mask` | `workers_video.make_mask_worker` | clip-batched: `batching.run_clips_concurrently` |
| `shape` | `workers_shape.make_shape_worker` | none (one geometry-only reconstruction on the keyframe) |
| `intrinsics` | `workers_geometry.make_intrinsics_worker` | none (MoGe-2 on 3 keyframes) |
| `hands` | `workers_hands.make_hands_worker(cfg)` | clip-batched in the HaWoR driver: `hands_batch.run_hands_stage` |
| `geometry` | `workers_geometry.make_geometry_worker` | frame-batched: clips sharing a camera share `run_moge` |
| `tracks` | `workers_video.make_tracks_worker` | clip-batched: `batching.run_clips_concurrently` |
| `driftqc` | `workers_driftqc.make_driftqc_worker(cfg)` | none, CPU |
| `solve` | `workers_solve.make_solve_worker(cfg)` | none (per clip; GPU-serialized) |
| `qc` | `workers_qc.make_qc_worker_from_config(cfg)` (or `make_qc_worker(rule)`) | none (per clip, CPU) |
| `retarget` | `workers_retarget.make_retarget_worker(cfg)` | none (per clip, CPU MuJoCo) |

The module-level instances below are ready-made `StageFn`s with default settings. Building one allocates and loads
nothing: each worker applies `fastdo.bench.gpu_budget.apply_gpu_budget()` on its first call, before any allocation,
and loads its model then, once, for every later clip. `solve_worker` runs the solve stage with
`SolveStageConfig()` (its defaults, on `cuda`). `decode`, `gate`, `qc` and `retarget` have no default
instance: they need the clip-to-video table, a gate bundle, a saved acceptance rule and the retarget stage config. `make_template_shape_worker` gives
calibration clips their GT template mesh as shape (never a default). `driftqc` is opt-in (`driftqc_stage.enabled`, default
false): it is not in `STAGE_NAMES`, has no default instance, and `with_driftqc` inserts it after `tracks`.
"""

from __future__ import annotations

from fastdo.engine.workers_driftqc import (
    DRIFT_REJECTED_PREFIX,
    DRIFTQC_FILE,
    DriftQcStageConfig,
    DriftRejected,
    make_driftqc_worker,
    with_driftqc,
)
from fastdo.engine.workers_frames import (
    CROP_MODES,
    ENCODER_FILE,
    EncodeWorker,
    center_crops,
    letterbox_crops,
    make_decode_worker,
    make_encode_worker,
)
from fastdo.engine.workers_gate import (
    GATE_REJECTED_PREFIX,
    GateRejected,
    GateStageConfig,
    gate_outcome_counts,
    make_gate_worker,
    read_gate_json,
)
from fastdo.engine.workers_geometry import (
    DEFAULT_HAND_SIDES,
    GEOMETRY_FILE,
    GeometryWorker,
    ScaleFn,
    keyframe_rows,
    make_geometry_worker,
    make_intrinsics_worker,
)
from fastdo.engine.workers_hands import (
    HANDS_FILE,
    HANDS_OUTPUT,
    HAWOR_FILE,
    make_hands_worker,
    write_hands_artifact,
)
from fastdo.engine.workers_qc import (
    CLIP_RECORD_FILE,
    GATE_FILE,
    QC_FILE,
    SOLVE_FILE,
    QcStageConfig,
    make_qc_worker,
    make_qc_worker_from_config,
    read_solve_artifact,
    write_solve_artifact,
)
from fastdo.engine.workers_retarget import (
    RETARGET_FILE,
    TRAJECTORY_FILE,
    RetargetStageConfig,
    make_retarget_worker,
)
from fastdo.engine.workers_shape import (
    MESH_FILE,
    SAMPLES_FILE,
    SDF_FILE,
    SHAPE_META_FILE,
    make_shape_worker,
    make_template_shape_worker,
    read_mesh_samples,
    read_sdf_grid,
    read_shape_meta,
)
from fastdo.engine.workers_solve import (
    SOLVE_META_FILE,
    SolveCall,
    SolveStageConfig,
    finish_solve,
    make_solve_worker,
    prepare_solve,
)
from fastdo.engine.workers_video import (
    MASK_FILE,
    PROMPT_FILE,
    TRACKS_FILE,
    make_mask_worker,
    make_tracks_worker,
    read_mask_prompt,
    write_mask_prompt,
)

__all__ = [
    "CLIP_RECORD_FILE",
    "CROP_MODES",
    "DEFAULT_HAND_SIDES",
    "DRIFTQC_FILE",
    "DRIFT_REJECTED_PREFIX",
    "ENCODER_FILE",
    "GATE_FILE",
    "GATE_REJECTED_PREFIX",
    "GEOMETRY_FILE",
    "HANDS_FILE",
    "HANDS_OUTPUT",
    "HAWOR_FILE",
    "MASK_FILE",
    "MESH_FILE",
    "PROMPT_FILE",
    "QC_FILE",
    "RETARGET_FILE",
    "SAMPLES_FILE",
    "SDF_FILE",
    "SHAPE_META_FILE",
    "SOLVE_FILE",
    "SOLVE_META_FILE",
    "TRACKS_FILE",
    "TRAJECTORY_FILE",
    "DriftQcStageConfig",
    "DriftRejected",
    "EncodeWorker",
    "GateRejected",
    "GateStageConfig",
    "GeometryWorker",
    "QcStageConfig",
    "RetargetStageConfig",
    "ScaleFn",
    "SolveCall",
    "SolveStageConfig",
    "center_crops",
    "encode_worker",
    "finish_solve",
    "gate_outcome_counts",
    "geometry_worker",
    "hands_worker",
    "intrinsics_worker",
    "keyframe_rows",
    "letterbox_crops",
    "make_decode_worker",
    "make_driftqc_worker",
    "make_encode_worker",
    "make_gate_worker",
    "make_geometry_worker",
    "make_hands_worker",
    "make_intrinsics_worker",
    "make_mask_worker",
    "make_qc_worker",
    "make_qc_worker_from_config",
    "make_retarget_worker",
    "make_shape_worker",
    "make_solve_worker",
    "make_template_shape_worker",
    "make_tracks_worker",
    "mask_worker",
    "prepare_solve",
    "read_gate_json",
    "read_mask_prompt",
    "read_mesh_samples",
    "read_sdf_grid",
    "read_shape_meta",
    "read_solve_artifact",
    "shape_worker",
    "solve_worker",
    "tracks_worker",
    "with_driftqc",
    "write_hands_artifact",
    "write_mask_prompt",
    "write_solve_artifact",
]

encode_worker = make_encode_worker()
mask_worker = make_mask_worker()
shape_worker = make_shape_worker()
intrinsics_worker = make_intrinsics_worker()
hands_worker = make_hands_worker(None)
geometry_worker = make_geometry_worker()
tracks_worker = make_tracks_worker()
solve_worker = make_solve_worker()
