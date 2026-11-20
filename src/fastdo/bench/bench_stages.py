"""Real stage workers of the throughput benchmark (`run/pipeline/bench_throughput.py`).

`build_stage_fns` builds every registered worker (nothing loads until a worker's first clip) and a status label for
every stage of `STAGE_NAMES`:

- decode (ffmpeg, or JPEG sequences for DexYCB), encoder, mask, intrinsics, geometry and tracks: "computed". The
  encoder uses the gate bundle's crop when the gate is registered, else "center". A `geometry_stage.hand_sides` other
  than the default `right` labels geometry "computed (hand anchor <value>)"; a `geometry_stage.hand_depth_scale` that
  corrects anything appends "hand depth x{left: .., right: ..}" to that label. `bench.mask_compile` other than `none`
  builds the mask stage's compiled SAM 2.1 VOS predictor, labelled "computed (SAM 2.1 VOS predictor, ...)".
- shape: "computed (<bench.shape.backend>)". `bench.shape_cache_dir` (null: off) enables the per-clip shape cache of
  the model worker; the template worker is never cached. With `bench.dexycb_shape: template` the DexYCB clips get
  their GT YCB mesh (`textured_simple.obj`, the calibration protocol) and the label starts with "borrowed", so such a
  run is never reported as a complete pipeline.
- hands: batched HaWoR, or the demo's precomputed npz for whisking clips ("borrowed").
- solve: `make_profiled_solve_worker` of `SolveStageConfig.from_mapping(cfg.solve_stage)` (each clip's
  source-domain profile, `fastdo.engine.source_profile`, applied per clip), "computed (M1 solver on <device>)".
- source profile: `check_run_profile` over the clips' domains before anything is built, so a config that
  explicitly contradicts the egocentric profile (`contact_stage.enabled=true`, `dead_frame_branch=true` on an
  egocentric clip) is refused up front unless `source_profile.study_override`; the contact stage, when registered, is
  `make_profiled_contact_worker`. Study harnesses that pass `clips=[]` get no up-front check; their clips are resolved
  per clip (`source_domain.json`, else `source_profile.domain`, else refused).
- qc: only when `qc_stage.rule_path` names a saved AcceptanceRule, "computed (rule <path>, sha256 <12 hex>)"; otherwise
  `QC_UNREGISTERED`. The qc stage has no accept-all default.
- gate: registered iff `gate_stage.model_path` names a gate bundle, "computed (gate bundle <path>, sha256 <12 hex>,
  threshold <tau>, guaranteed recall <g> on <n> contact-centred 100DOH train positives)"; otherwise `GATE_UNBUILT`.
  The gate reads the encoder's tokens, so a `bench.stages` listing gate without encoder is refused.
- retarget (opt-in): registered only with `retarget_stage.enabled` (default false, so a default run's stages and
  headline are unchanged): `make_retarget_worker(RetargetStageConfig.from_mapping(cfg.retarget_stage))` with the
  driftqc flag, "computed (kinematic reference, CPU; side ..., joint_limits ..., lam_tip ...)". It reads the qc stage's
  `qc.json`/`clip_record.npz`, so enabling it without a registered qc stage is refused. Otherwise `NOT_REGISTERED`.
- driftqc (opt-in): registered only with `driftqc_stage.enabled`, "computed (driftqc v1 TRUNC, CPU;
  constants sha256 <12 hex>)", its label then following `tracks`; otherwise absent from both mappings. The same flag
  (`driftqc_enabled`) is passed to the solve and qc workers here and to the batched solve's `prepare_solve` partial
  (`bench_batched.build_batched_stages`), so with driftqc disabled a clip dir holding driftqc artifacts is a ValueError,
  never a slice.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from fastdo.bench.bench_clips import (
    BenchClip,
    dexycb_template_mesh,
    make_image_sequence_decode_worker,
)
from fastdo.bench.oakink2_bench import make_image_window_decode_worker
from fastdo.bench.yield_report import GATE_UNBUILT
from fastdo.geometry.ego_motion import EgoMotionConfig
from fastdo.engine import (
    STAGE_NAMES,
    DriftQcStageConfig,
    GateStageConfig,
    QcStageConfig,
    SolveStageConfig,
    StageFn,
    make_decode_worker,
    make_driftqc_worker,
    make_encode_worker,
    make_gate_worker,
    make_geometry_worker,
    make_hands_worker,
    make_intrinsics_worker,
    make_mask_worker,
    make_qc_worker_from_config,
    make_shape_worker,
    make_template_shape_worker,
    make_tracks_worker,
    with_driftqc,
)
from fastdo.engine.shape_reuse import load_donor_map, make_reuse_shape_worker
from fastdo.engine.source_profile import (
    SourceProfileConfig,
    check_run_profile,
    make_profiled_contact_worker,
    make_profiled_solve_worker,
)
from fastdo.engine.workers_contact import ContactStageConfig, with_contact
from fastdo.engine.workers_driftqc import FROZEN_CONSTANTS_SHA256
from fastdo.engine.workers_geometry import (
    DEFAULT_HAND_SIDES,
    HAND_SIDES_AUTO,
    POINTMAP_STORAGES,
    check_hand_depth_scale,
)
from fastdo.engine.workers_prompt import PromptStageConfig, make_prompt_worker, with_prompt
from fastdo.engine.workers_retarget import RetargetStageConfig, make_retarget_worker
from fastdo.engine.workers_tba import make_tba_worker, with_tba
from fastdo.gate.bundle import load_gate_bundle
from fastdo.geometry import moge_model
from fastdo.geometry.intrinsics import check_focal_scale
from fastdo.hands.hawor_batched import hands_config_from_cfg
from fastdo.qc.rule_io import acceptance_rule_sha256
from fastdo.shape.backend import ShapeConfig
from fastdo.shape.sam2_compile import check_mask_compile
from fastdo.tba.config import tba_from_mapping

__all__ = ["DEXYCB_SHAPES", "GEOMETRY_HAND_SIDES", "NOT_REGISTERED", "QC_UNREGISTERED", "build_stage_fns",
           "driftqc_enabled", "geometry_hand_depth_scale", "geometry_hand_sides", "mask_compile", "resolve_path",
           "retarget_enabled", "shape_config", "source_profile_config", "whisking_paths"]

DEXYCB_SHAPES: tuple[str, ...] = ("model", "template")
# `geometry_stage.hand_sides` -> the geometry worker's `hand_sides`. `right` is the measured protocol and the
# production default; `auto` anchors on the right hand whenever it has a valid decode row and on the left only when it
# has none (bit-identical on every clip `right` already handles); `left` forces the left hand (diagnostics).
GEOMETRY_HAND_SIDES: dict[str, tuple[str, ...] | str] = {"right": DEFAULT_HAND_SIDES, "auto": HAND_SIDES_AUTO,
                                                         "left": ("left",)}
QC_UNREGISTERED = "not registered: qc_stage.rule_path is null (run run/pipeline/calibrate_acceptance.py)"
NOT_REGISTERED: dict[str, str] = {
    "gate": GATE_UNBUILT,
    "qc": QC_UNREGISTERED,
    "retarget": "not registered: opt-in (retarget_stage.enabled), runs only on qc-accepted clips",
}


def retarget_enabled(cfg: DictConfig) -> bool:
    """`retarget_stage.enabled` (False without the key or the block): whether the bench registers the retarget worker.

    Raises:
        ValueError: On a value that is not a bool.
    """
    enabled = (cfg.get("retarget_stage") or {}).get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"retarget_stage.enabled must be a bool, got {enabled!r}")  # noqa: TRY004  # as siblings
    return enabled


def _retarget_stage(cfg: DictConfig, enabled: bool) -> tuple[StageFn | None, str]:
    """`(worker, label)` with `retarget_stage.enabled`, else `(None, NOT_REGISTERED["retarget"])`."""
    if not retarget_enabled(cfg):
        return None, NOT_REGISTERED["retarget"]
    mapping = cast("dict[str, Any]", OmegaConf.to_container(cfg.retarget_stage, resolve=True))
    stage_cfg = RetargetStageConfig.from_mapping(mapping)
    side = ("from prompt" if stage_cfg.side_from_prompt else "right when present") if stage_cfg.side is None \
        else stage_cfg.side
    label = (f"computed (kinematic reference, CPU; side {side}, joint_limits {stage_cfg.retarget.joint_limits}, "
             f"lam_tip {stage_cfg.retarget.lam_tip:g})")
    return make_retarget_worker(stage_cfg, driftqc_enabled=enabled), label


def resolve_path(base: Path, path: Any) -> Path:
    """`path` if absolute, else `base / path`, resolved."""
    p = Path(str(path))
    return p if p.is_absolute() else (base / p).resolve()


def whisking_paths(cfg: DictConfig, root: Path) -> dict[str, Path]:
    """The whisking demo's `video`, `mesh_path`, `hand_npz` and `layout_json` (relative entries under `demo_root`)."""
    demo = resolve_path(root, cfg.whisking.demo_root)
    return {key: resolve_path(demo, cfg.whisking[key]) for key in ("video", "mesh_path", "hand_npz", "layout_json")}


def geometry_row_stride(cfg: DictConfig) -> int:
    """`geometry_stage.row_stride` (1 without the key): MoGe-2 on every k-th decode row plus the keyframe.

    Raises:
        ValueError: On a value that is not an int >= 1.
    """
    stride = (cfg.get("geometry_stage") or {}).get("row_stride", 1)
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError(f"geometry_stage.row_stride must be an int >= 1, got {stride!r}")
    return stride


def geometry_hand_sides(cfg: DictConfig) -> tuple[str, ...] | str:
    """`geometry_stage.hand_sides` (`right` without the key): which hand's joints set the clip's metric scale.

    `right` returns `DEFAULT_HAND_SIDES`, today's worker exactly. `auto` returns `HAND_SIDES_AUTO` (the right hand
    whenever it has a valid decode row, else the left): it recovers the clips `right` drops at geometry for having only
    a left hand and leaves every other clip byte-identical. `left` forces the left hand. `auto` stays opt-in: a left
    hand's anchor bias is unmeasured, so switching it on changes the POPULATION behind every quality number.

    Raises:
        ValueError: On anything but the strings `right`, `auto` or `left` (a list, null, or another spelling included).
    """
    value = (cfg.get("geometry_stage") or {}).get("hand_sides", "right")
    if not isinstance(value, str) or value not in GEOMETRY_HAND_SIDES:
        raise ValueError(f"geometry_stage.hand_sides must be one of {sorted(GEOMETRY_HAND_SIDES)}, got {value!r}")
    return GEOMETRY_HAND_SIDES[value]


def geometry_hand_depth_scale(cfg: DictConfig) -> dict[str, float] | None:
    """`geometry_stage.hand_depth_scale` (null or absent: off): a per-side relative correction of the hand depth the
    metric scale is anchored on, `{left: f_L, right: f_R}`. The geometry stage's clip scale is multiplied by the factor
    of the side that anchors it (`workers_geometry` module docstring); f = 1 / (1 + r) cancels a relative depth bias r.
    Returns None when off or when both factors are exactly 1.0 (byte-identical to off). Opt-in; the
    contact stage is not corrected.

    Raises:
        ValueError: On anything but null or a mapping with exactly `left` and `right`, each a number in
            `HAND_DEPTH_SCALE_RANGE`.
    """
    value = (cfg.get("geometry_stage") or {}).get("hand_depth_scale")
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value)
    try:
        return check_hand_depth_scale(value)
    except ValueError as exc:
        raise ValueError(f"geometry_stage.hand_depth_scale: {exc}") from None


def geometry_moge_batch_size(cfg: DictConfig) -> int:
    """`geometry_stage.moge_batch_size` (`moge_model._BATCH_SIZE` = 2 without the key): MoGe-2 frames per forward, for
    both the `geometry` and the `intrinsics` worker. A batch of 8 is faster but not bit-identical; gated by qc22.

    Raises:
        ValueError: On a value that is not an int >= 1.
    """
    batch = (cfg.get("geometry_stage") or {}).get("moge_batch_size", moge_model._BATCH_SIZE)
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError(f"geometry_stage.moge_batch_size must be an int >= 1, got {batch!r}")
    return batch


def mask_compile(cfg: DictConfig) -> str:
    """`bench.mask_compile` (`none` without the key): how the `mask` stage's SAM 2.1 video predictor is built
    (`fastdo.shape.sam2_compile`). `none` = the eager predictor (byte-identical to before); `vos_nocg`/`vos_cg_clone` =
    upstream's torch.compiled VOS predictor (not bit-exact), opt-in until its pose gate (qc24) rules.

    Raises:
        ValueError: On a value outside `MASK_COMPILE_MODES`.
    """
    return check_mask_compile((cfg.get("bench") or {}).get("mask_compile", "none"))


def geometry_moge_cuda_graph(cfg: DictConfig) -> bool:
    """`geometry_stage.moge_cuda_graph` (False without the key): CUDA-graph replay of the MoGe-2 forward in both the
    `geometry` and the `intrinsics` worker (`fastdo.geometry.moge_cuda_graph`). Opt-in until its GPU gate rules.

    Raises:
        ValueError: On a value that is not a bool.
    """
    enabled = (cfg.get("geometry_stage") or {}).get("moge_cuda_graph", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"geometry_stage.moge_cuda_graph must be a bool, got {enabled!r}")  # noqa: TRY004  # as siblings
    return enabled


def geometry_moge_overlap(cfg: DictConfig) -> bool:
    """`geometry_stage.moge_overlap` (False without the key): the `geometry` worker's MoGe-2 batches as a 2-stage
    pipeline (`fastdo.geometry.moge_overlap`); the `intrinsics` worker is unaffected. Opt-in until its gate rules.

    Raises:
        ValueError: On a value that is not a bool.
    """
    enabled = (cfg.get("geometry_stage") or {}).get("moge_overlap", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"geometry_stage.moge_overlap must be a bool, got {enabled!r}")  # noqa: TRY004  # as siblings
    return enabled


def geometry_moge_num_tokens(cfg: DictConfig) -> int | None:
    """`geometry_stage.moge_num_tokens` (None without the key, or null): MoGe-2's base ViT tokens in the `geometry`
    worker only (the intrinsics are split out: on internet video K comes from MoGe, and fewer tokens there moved
    fx by +1.8% median and with it the whole gauge). None = MoGe's default (3600), byte-identical; an int in
    [1200, 3600] is faster but moves the depth, opt-in until its gates rule.

    Raises:
        ValueError: On a value that is not null or an int in `moge_model.NUM_TOKENS_RANGE`.
    """
    tokens = (cfg.get("geometry_stage") or {}).get("moge_num_tokens", None)
    return moge_model.check_num_tokens(tokens, "geometry_stage.moge_num_tokens")


def intrinsics_moge_num_tokens(cfg: DictConfig) -> int | None:
    """`geometry_stage.intrinsics_moge_num_tokens` (None without the key, or null): MoGe-2's base ViT tokens in the
    `intrinsics` worker (3 keyframes, so its cost is negligible). None = MoGe's default (3600), byte-identical.

    Raises:
        ValueError: On a value that is not null or an int in `moge_model.NUM_TOKENS_RANGE`.
    """
    tokens = (cfg.get("geometry_stage") or {}).get("intrinsics_moge_num_tokens", None)
    return moge_model.check_num_tokens(tokens, "geometry_stage.intrinsics_moge_num_tokens")


def intrinsics_focal_scale(cfg: DictConfig) -> float | None:
    """`geometry_stage.intrinsics_focal_scale` (None without the key, null, or 1.0): a multiplier of the
    MoGe-2-ESTIMATED fx and fy (a clip with `camera.json` is unaffected). None = off, byte-identical.

    Raises:
        ValueError: On a value `fastdo.geometry.intrinsics.check_focal_scale` refuses.
    """
    value = (cfg.get("geometry_stage") or {}).get("intrinsics_focal_scale", None)
    return check_focal_scale(value, "geometry_stage.intrinsics_focal_scale")


def geometry_pointmap_storage(cfg: DictConfig) -> str:
    """`geometry_stage.pointmap_storage` (`dense` without the block): how `geometry` writes its pointmaps.

    Raises:
        ValueError: On an unknown value, or on `masked` while driftqc is enabled -- driftqc samples pointmaps at
            TRACKED pixels, which can leave the object mask, and a masked artifact has no values there.
    """
    storage = str((cfg.get("geometry_stage") or {}).get("pointmap_storage", "dense"))
    if storage not in POINTMAP_STORAGES:
        raise ValueError(f"geometry_stage.pointmap_storage must be one of {POINTMAP_STORAGES}, got {storage!r}")
    if storage == "masked" and driftqc_enabled(cfg):
        raise ValueError("geometry_stage.pointmap_storage: masked is incompatible with driftqc_stage.enabled: "
                         "driftqc reads tracked, possibly off-object, pixels")
    return storage


def geometry_ego_motion(cfg: DictConfig) -> EgoMotionConfig:
    """`geometry_stage.ego_motion` (absent or `enabled: false` = off, geometry.npz byte-identical)."""
    block = (cfg.get("geometry_stage") or {}).get("ego_motion")
    return EgoMotionConfig.from_mapping(None if block is None else {str(k): block[k] for k in block})


def driftqc_enabled(cfg: DictConfig) -> bool:
    """`DriftQcStageConfig.from_mapping(cfg.driftqc_stage).enabled`; a config without the block is disabled.

    Raises:
        ValueError: On an invalid `driftqc_stage` block (unknown key, non-bool `enabled`, enabled with a path unset).
    """
    return DriftQcStageConfig.from_mapping(cfg.get("driftqc_stage") or {}).enabled


def _driftqc_stage(cfg: DictConfig, root: Path) -> tuple[StageFn | None, str]:
    """`(worker, label)` with `driftqc_stage.enabled` (paths resolved against `root`; the worker asserts the frozen
    files' sha256 at construction), else `(None, "")`."""
    stage_cfg = DriftQcStageConfig.from_mapping(cfg.get("driftqc_stage") or {})
    if not stage_cfg.enabled:
        return None, ""
    paths = [str(resolve_path(root, path)) for path in stage_cfg.required_paths()]
    worker = make_driftqc_worker(dataclasses.replace(stage_cfg, constants_path=paths[0],
                                                     trackweight_constants_path=paths[1], detector_path=paths[2]))
    return worker, f"computed (driftqc v1 TRUNC, CPU; constants sha256 {FROZEN_CONSTANTS_SHA256[:12]})"


def shape_config(cfg: DictConfig) -> ShapeConfig:
    """`bench.shape` as the `ShapeConfig` the `shape` stage's model worker is built with (studies read it from here, so
    a knob they set is the one the engine would build).

    Raises:
        ValueError: On an invalid `bench.shape` block (`ShapeConfig`'s own checks).
    """
    return ShapeConfig(**cast("dict[str, Any]", OmegaConf.to_container(cfg.bench.shape, resolve=True)))


def _shape_stage(cfg: DictConfig, root: Path, clips: list[BenchClip]) -> tuple[StageFn, str]:
    shape_cfg = shape_config(cfg)
    cache_dir = cfg.bench.get("shape_cache_dir")
    model_worker = make_shape_worker(shape_cfg, cache_dir=None if cache_dir is None else resolve_path(root, cache_dir))
    reuse_note = ""
    donors_path = (cfg.bench.get("shape_reuse") or {}).get("donors")
    if donors_path is not None:           # opt-in cross-window template reuse; null = the model worker unchanged
        donors = load_donor_map(resolve_path(root, donors_path))
        model_worker = make_reuse_shape_worker(model_worker, donors)
        reuse_note = f"; {len(donors)} windows reuse a donor's template (bench.shape_reuse)"
    mode = str(cfg.bench.get("dexycb_shape", "model"))
    if mode not in DEXYCB_SHAPES:
        raise ValueError(f"bench.dexycb_shape must be one of {DEXYCB_SHAPES}, got {mode!r}")
    if mode == "model":
        return model_worker, f"computed ({shape_cfg.backend}){reuse_note}"
    dexycb_root = resolve_path(root, cfg.dexycb.root)
    templates = {clip.clip_id: dexycb_template_mesh(dexycb_root, clip.path.relative_to(dexycb_root).as_posix())
                 for clip in clips if clip.source == "dexycb"}
    template_worker = make_template_shape_worker(templates)

    def shape_worker(clip_id: str, artifact_dir: Path) -> None:
        (template_worker if clip_id in templates else model_worker)(clip_id, artifact_dir)

    return shape_worker, (f"borrowed (GT YCB template for dexycb clips, the calibration protocol; {shape_cfg.backend} "
                          "for the others)")


def _qc_stage(cfg: DictConfig, root: Path, enabled: bool) -> tuple[StageFn | None, str]:
    qc_cfg = QcStageConfig.from_mapping(cfg.qc_stage)
    if not qc_cfg.rule_path:
        return None, QC_UNREGISTERED
    path = resolve_path(root, qc_cfg.rule_path)
    worker = make_qc_worker_from_config(dataclasses.replace(qc_cfg, rule_path=str(path)), driftqc_enabled=enabled)
    return worker, f"computed (rule {path}, sha256 {acceptance_rule_sha256(path)[:12]})"


def _gate_stage(cfg: DictConfig, root: Path) -> tuple[StageFn | None, str, str]:
    """`(worker, status label, encoder crop)`: `(None, GATE_UNBUILT, "center")` without `gate_stage.model_path`."""
    gate_cfg = GateStageConfig.from_mapping(cfg.get("gate_stage") or {})
    if not gate_cfg.model_path:
        return None, NOT_REGISTERED["gate"], "center"
    path = resolve_path(root, gate_cfg.model_path)
    bundle = load_gate_bundle(path)
    cal = bundle.calibration
    label = (f"computed (gate bundle {path}, sha256 {bundle.sha256[:12]}, threshold {cal.threshold:.4f}, guaranteed "
             f"recall {cal.guaranteed_recall:.3f} on {cal.n_pos} contact-centred 100DOH train positives)")
    return make_gate_worker(dataclasses.replace(gate_cfg, model_path=str(path)), bundle=bundle), label, bundle.crop


def decode_worker_options(bench: Any) -> dict[str, Any]:
    """`make_decode_worker`'s optional settings, read from the bench config.

    The S2 windowing rule's downscale (`decode_scale_height`, its 720) and the window-local frame probe
    (`decode_local_probe`) were both implemented with NO caller -- a capability whose unit tests pass and
    which nothing can switch on. Reading them here is what makes them reachable.

    Both defaults leave every existing run byte-identical, so the S2 run opts in explicitly rather than
    other experiments silently changing resolution underneath themselves.
    """
    scale_height = bench.get("decode_scale_height")
    return {"local_probe": bool(bench.get("decode_local_probe", False)),
            "packet_table": bool(bench.get("decode_packet_table", False)),
            "scale_height": None if scale_height is None else int(scale_height)}


def build_video_decoders(bench: Any, videos: Sequence[BenchClip], stride: int
                         ) -> tuple[dict[int, StageFn], dict[str, int], StageFn]:
    """`(decoder per stride, clip_id -> stride, fallback)` for the video clips.

    One worker PER DISTINCT STRIDE: S2 fixes the operating rate, so s_v is 5 at 50 fps and 6 at 59.94 fps
    and a single run-wide stride cannot express an EK-100 run.

    `decode_source`, not `path`: a windowed clip decodes only its own range (S2's P4). Passing `path`
    here is what made the worker's window path unreachable.
    """
    by_stride: dict[int, dict[str, Path | tuple[Path, int, int]]] = {}
    for clip in videos:
        by_stride.setdefault(clip.decode_stride or stride, {})[clip.clip_id] = clip.decode_source
    options = decode_worker_options(bench)
    decoders = {s: make_decode_worker(sources, stride=s, **options) for s, sources in by_stride.items()}
    stride_of = {clip.clip_id: (clip.decode_stride or stride) for clip in videos}
    fallback = decoders.get(stride) or make_decode_worker({}, stride=stride, **options)
    return decoders, stride_of, fallback


def source_profile_config(cfg: DictConfig) -> SourceProfileConfig:
    """`SourceProfileConfig.from_mapping(cfg.source_profile)`; a config without the block gets the defaults."""
    return SourceProfileConfig.from_mapping(cfg.get("source_profile"))


def build_stage_fns(cfg: DictConfig, root: Path, clips: list[BenchClip]) -> tuple[dict[str, StageFn], dict[str, str]]:
    """The real stage workers, and a status label for every stage of `STAGE_NAMES` (module docstring). Nothing loads
    here but the gate bundle: every other worker loads its model on its first clip.

    Raises:
        ValueError: On an unknown `bench.hands_source` or `bench.dexycb_shape`, an invalid `solve_stage`,
            `gate_stage`, `driftqc_stage` or (enabled) `retarget_stage` block, an invalid rule or gate bundle file, a
            registered gate that `bench.stages` lists without the encoder, or `retarget_stage.enabled` without qc.
        FileNotFoundError: If `qc_stage.rule_path` or `gate_stage.model_path` is set but absent, or, with
            `bench.dexycb_shape: template`, a DexYCB clip's `textured_simple.obj` is missing.
        SourceProfileConflict: If the config explicitly contradicts the egocentric profile for an egocentric clip of
            `clips` without `source_profile.study_override`, or `source_profile.domain` relabels a clip.
    """
    solve_cfg = SolveStageConfig.from_mapping(cfg.solve_stage)
    contact_cfg = ContactStageConfig.from_mapping(cfg.get("contact_stage"))
    profile = source_profile_config(cfg)
    check_run_profile({clip.clip_id: cast("Any", clip.domain) for clip in clips}, solve_cfg, contact_cfg.enabled,
                      profile)
    bench = cfg.bench
    stride = int(bench.stride)
    enabled = driftqc_enabled(cfg)
    gate_worker, gate_status, crop = _gate_stage(cfg, root)
    requested = None if bench.get("stages") is None else {str(stage) for stage in bench.stages}
    if gate_worker is not None and requested is not None and "gate" in requested and "encoder" not in requested:
        raise ValueError("bench.stages lists gate without encoder, but the gate stage (gate_stage.model_path) reads the "
                         "encoder's tokens: add encoder to bench.stages, or drop gate")
    sequences = {clip.clip_id: clip.path for clip in clips if clip.kind == "image_sequence"}
    videos = [c for c in clips if c.kind == "video"]
    video_decoders, stride_of, unknown_clip = build_video_decoders(bench, videos, stride)
    image_decode = make_image_sequence_decode_worker(sequences, stride=stride)
    # an image_window clip (OakInk2) decodes a stride of one PNG directory's [start, stop) image range.
    image_windows = {c.clip_id: (c.path, c.window[0], c.window[1], c.decode_stride or stride)
                     for c in clips if c.kind == "image_window" and c.window is not None}
    window_decode = make_image_window_decode_worker(image_windows) if image_windows else None

    def decode_worker(clip_id: str, artifact_dir: Path) -> None:
        if clip_id in sequences:
            image_decode(clip_id, artifact_dir)
            return
        if window_decode is not None and clip_id in image_windows:
            window_decode(clip_id, artifact_dir)
            return
        # an unregistered clip falls through to a worker that raises the decode worker's own KeyError
        video_decoders.get(stride_of.get(clip_id, -1), unknown_clip)(clip_id, artifact_dir)

    hands_source = str(bench.hands_source)
    if hands_source == "batched":
        hands_worker = make_hands_worker(hands_config_from_cfg(cfg))
        hands_status = "computed (batched HaWoR, one driver call per clip)"
    elif hands_source == "precomputed":
        npz = whisking_paths(cfg, root)["hand_npz"]
        hands_worker = make_hands_worker(hands_config_from_cfg(cfg),
                                         sources={c.clip_id: npz for c in clips if c.source == "whisking"})
        hands_status = f"borrowed (precomputed {npz.name} for whisking clips; batched HaWoR for the others)"
    else:
        raise ValueError(f"bench.hands_source must be 'batched' or 'precomputed', got {hands_source!r}")
    shape_worker, shape_status = _shape_stage(cfg, root, clips)
    fns: dict[str, StageFn] = {
        "decode": decode_worker, "encoder": make_encode_worker(crop=crop),
        "mask": make_mask_worker(str(bench.mask_variant), mask_compile=mask_compile(cfg)), "shape": shape_worker,
        "intrinsics": make_intrinsics_worker(moge_batch_size=geometry_moge_batch_size(cfg),
                                             moge_cuda_graph=geometry_moge_cuda_graph(cfg),
                                             moge_num_tokens=intrinsics_moge_num_tokens(cfg),
                                             focal_scale=intrinsics_focal_scale(cfg)), "hands": hands_worker,
        "geometry": make_geometry_worker(hand_sides=geometry_hand_sides(cfg),
                                         hand_depth_scale=geometry_hand_depth_scale(cfg),
                                         pointmap_storage=geometry_pointmap_storage(cfg),
                                         row_stride=geometry_row_stride(cfg),
                                         moge_batch_size=geometry_moge_batch_size(cfg),
                                         moge_cuda_graph=geometry_moge_cuda_graph(cfg),
                                         moge_overlap=geometry_moge_overlap(cfg),
                                         moge_num_tokens=geometry_moge_num_tokens(cfg),
                                         ego_motion=geometry_ego_motion(cfg)),
        "tracks": make_tracks_worker(n_queries=int(bench.n_track_queries)),
        "solve": make_profiled_solve_worker(solve_cfg, contact_cfg.enabled, profile, driftqc_enabled=enabled),
    }
    compile_mode = mask_compile(cfg)
    labels = dict.fromkeys(fns, "computed") | {"shape": shape_status, "hands": hands_status,
                                               "mask": "computed" if compile_mode == "none" else
                                               f"computed (SAM 2.1 VOS predictor, mask_compile {compile_mode})",
                                               "solve": f"computed (M1 solver on {solve_cfg.device})"}
    hand_anchor = (cfg.get("geometry_stage") or {}).get("hand_sides", "right")
    if hand_anchor != "right":
        labels["geometry"] = f"computed (hand anchor {hand_anchor})"
    depth_scale = geometry_hand_depth_scale(cfg)
    if depth_scale is not None:
        labels["geometry"] += f" (hand depth x{depth_scale})"
    if gate_worker is not None:
        fns["gate"], labels["gate"] = gate_worker, gate_status
    qc_worker, labels["qc"] = _qc_stage(cfg, root, enabled)
    if qc_worker is not None:
        fns["qc"] = qc_worker
    retarget_worker, labels["retarget"] = _retarget_stage(cfg, enabled)
    if retarget_worker is not None:
        if qc_worker is None:
            raise ValueError("retarget_stage.enabled is true, but the qc stage is not registered (qc_stage.rule_path is "
                             "null): the retarget stage reads qc.json and clip_record.npz. Set qc_stage.rule_path, or "
                             "set retarget_stage.enabled=false")
        fns["retarget"] = retarget_worker
    driftqc_worker, driftqc_label = _driftqc_stage(cfg, root)
    names = STAGE_NAMES
    if driftqc_worker is not None:
        fns["driftqc"], labels["driftqc"], names = driftqc_worker, driftqc_label, with_driftqc(STAGE_NAMES)
    prompt_cfg = PromptStageConfig.from_mapping(cfg.get("prompt_stage"))
    if prompt_cfg.enabled:    # S2's P3: the keyframe prompt raw video has no label for
        fns["prompt"] = make_prompt_worker(prompt_cfg)
        labels["prompt"] = f"computed (hands23 + sam2 {prompt_cfg.sam2_variant}, min_score {prompt_cfg.min_score})"
        names = with_prompt(names)
    if contact_cfg.enabled:   # the promoted along-ray correction; qc then scores the corrected trajectory
        fns["contact"] = make_profiled_contact_worker(contact_cfg, solve_cfg, profile)   # third-person only
        labels["contact"] = (f"computed (contact correction, {contact_cfg.cell}/{contact_cfg.hands_source}, "
                             f"hands {contact_cfg.hands})")
        names = with_contact(names)
    tba_cfg = tba_from_mapping(cfg.get("tba_stage"))
    if tba_cfg is not None:   # opt-in; null registers nothing (KA-ID: the stage list and artefacts are BASE's)
        fns["tba"] = make_tba_worker(tba_cfg)
        labels["tba"] = (f"computed ({'TBA-V' if tba_cfg.carve else 'TBA'}, gauge {tba_cfg.gauge}, CPU; "
                         "not read by qc)")
        names = with_tba(names)
    return fns, {stage: labels.get(stage, NOT_REGISTERED.get(stage, "not registered")) for stage in names}
