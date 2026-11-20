"""SAM 2.1 variant x resolution sweep on the whisking demo: seconds and mask IoU against native large.

Runs SAM 2.1 hiera large, base_plus, small and tiny (open on Hugging Face) at the demo's native 720p and at a 512 px
short side, under bf16 autocast, on the whisking demo's 46 stride-3 frames (source frames 0..135). Every row uses the
keyframe prompt of `whisking_headtohead.py`: the shipped SAM 3D keyframe mesh at its pre-optimization layout
(reference frame 125, `read_keyframe_layout`), rendered on the nearest stride-3 row (126).

Per row: `seconds` is the CUDA-synchronized wall time of `propagate_masks_at` (JPEG staging, the forward and reverse
propagation passes, mask readback and upsampling). The model load, and an untimed warm-up propagation on
`warmup_frames` frames around the keyframe, are excluded. `mean_iou` and `min_frame_iou` compare each frame's mask with
the first row, `large` at native 720p (frames where both masks are empty are skipped). Each size loads once for both
short sides. Row labels are `make_mask_worker` variant strings (`variant_label`): a row at the native short side is the
bare size (`small`), a reduced one `<size>@<px>` (`small@512`), so `chosen` can become the mask default as is.

Outputs, under `sam2_bench.out_dir`: `sam2_variants.json` (rows, reference, `min_iou`, `chosen =
choose_sam2_variant(rows, min_iou)`, run metadata) and `sam2_variants.md` (the table). The script does not change
the `make_mask_worker` default.

Resolution caveat: SAM 2's video predictor resizes every frame to its 1024x1024 `image_size` before the image encoder,
so the 512 px rows save frame staging and loading, not backbone compute (`fastdo.shape.sam2_variants`).

IoU caveat: a reduced row's masks are resampled down to SAM 2's input and back to native with nearest-exact sampling,
which alone costs IoU on a small object. On the whisking prompt (4484 px) a predictor that echoes the prompt perfectly
scores 0.9054 at a 512 px short side (0.8765 at 360, 0.9356 at 576, 0.9613 at 640). A `@512` row thus has under 0.6
IoU points above the 0.90 bound, and its IoU drop is mostly resampling, not model size.

This is a GPU job: use at most 2 CPU threads and, on a host shared with other GPU jobs, hold the shared lock, e.g.
  flock "$FASTDO_GPU_TOOLS_DIR/gpu.lock" env FASTDO_CPU_THREADS=2 OMP_NUM_THREADS=2 \\
      uv run python run/pipeline/bench_sam2_variants.py paths.outputs=${FASTDO_ROOT}/outputs
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from torch import Tensor

from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.bench.headtohead import mask_iou, render_silhouette
from fastdo.io.video import decode_frames, sample_frame_indices
from fastdo.shape import masks_sam2
from fastdo.shape.sam2_variants import (
    REFERENCE_LABEL,
    SAM2_VARIANTS,
    choose_sam2_variant,
    propagate_masks_at,
    resize_short_side,
    variant_label,
)
from fastdo.teacher.layout_reader import read_baseline_layout, read_keyframe_layout

logger = logging.getLogger(__name__)

JSON_NAME = "sam2_variants.json"
MD_NAME = "sam2_variants.md"


def sweep_labels(sizes: Sequence[str], short_sides: Sequence[int], native_short_side: int) -> list[str]:
    """`variant_label` of every size, then every short side, on frames with `native_short_side`: the run order."""
    return [variant_label(size, side, native_short_side) for size in sizes for side in short_sides]


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def run_sweep(frames: Tensor, keyframe_index: int, prompt: Tensor, sizes: Sequence[str], short_sides: Sequence[int],
              load_predictor: Callable[[str], Any], device: str = "cuda", bf16: bool = True, warmup_frames: int = 0,
              clock: Callable[[], float] = time.perf_counter) -> list[dict[str, Any]]:
    """Times `propagate_masks_at` for every size x short side and scores its masks against the first row.

    Args:
        frames: `(N,H,W,3)` uint8 RGB frames.
        keyframe_index: Row of `frames` carrying the prompt.
        prompt: `(H,W)` bool prompt mask at native resolution.
        sizes: SAM 2.1 sizes (`SAM2_VARIANTS` keys). The first size at the first short side is the IoU reference.
        short_sides: Short sides SAM 2 sees, in pixels.
        load_predictor: `model_id -> SAM2VideoPredictor`; called once per size, outside the timing.
        device: Torch device string.
        bf16: Run under bf16 autocast (CUDA only).
        warmup_frames: Frames around the keyframe propagated once, untimed, before every timed row (0: none).
        clock: Monotonic clock in seconds.

    Returns:
        One row per label (`variant_label`): `label`, `size`, `model_id`, `short_side` (as requested),
        `sam2_input_hw`, `n_frames`, `seconds`, `mean_iou`, `min_frame_iou`.

    Raises:
        ValueError: If `sizes` or `short_sides` is empty or a size is unknown.
    """
    if not sizes or not short_sides:
        raise ValueError("sizes and short_sides must be non-empty")
    unknown = [size for size in sizes if size not in SAM2_VARIANTS]
    if unknown:
        raise ValueError(f"unknown SAM 2.1 variant(s) {unknown}; expected {tuple(SAM2_VARIANTS)}")
    n = int(frames.shape[0])
    native_short_side = min(int(frames.shape[1]), int(frames.shape[2]))
    lo = max(0, keyframe_index - warmup_frames // 2)
    hi = min(n, lo + warmup_frames)
    reference: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    for size in sizes:
        model_id = SAM2_VARIANTS[size]
        predictor = load_predictor(model_id)
        for side in short_sides:
            if warmup_frames > 0:
                propagate_masks_at(frames[lo:hi], keyframe_index - lo, prompt, side, device, predictor, bf16)
            _synchronize(device)
            start = clock()
            masks = propagate_masks_at(frames, keyframe_index, prompt, side, device, predictor, bf16).numpy()
            _synchronize(device)
            seconds = clock() - start
            reference = masks if reference is None else reference
            ious = np.array([mask_iou(masks[i], reference[i]) for i in range(n)], dtype=np.float64)
            finite = ious[np.isfinite(ious)]
            rows.append({
                "label": variant_label(size, int(side), native_short_side), "size": size, "model_id": model_id, "short_side": int(side),
                "sam2_input_hw": [int(v) for v in resize_short_side(frames[:1], side).shape[1:3]], "n_frames": n,
                "seconds": float(seconds), "mean_iou": float(finite.mean()) if finite.size else math.nan,
                "min_frame_iou": float(finite.min()) if finite.size else math.nan,
            })
            logger.info("%s: %.3f s, mean IoU %.4f", rows[-1]["label"], seconds, rows[-1]["mean_iou"])
        del predictor
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return rows


def sweep_report(rows: Sequence[Mapping[str, Any]], min_iou: float = 0.90,
                 meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The sweep's JSON document: rows, the IoU reference, `min_iou`, the chosen default and run metadata."""
    return {"reference": str(rows[0]["label"]) if rows else REFERENCE_LABEL, "min_iou": float(min_iou),
            "chosen": choose_sam2_variant(rows, min_iou), "rows": [dict(row) for row in rows],
            "meta": dict(meta or {})}


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """The sweep rows as a Markdown table."""
    reference = rows[0]["label"] if rows else REFERENCE_LABEL
    lines = [f"| label | seconds | mean IoU vs {reference} | min frame IoU | SAM 2 input (H x W) |",
             "|---|---:|---:|---:|---|"]
    for row in rows:
        h, w = row["sam2_input_hw"]
        lines.append(f"| {row['label']} | {row['seconds']:.2f} | {row['mean_iou']:.4f} | {row['min_frame_iou']:.4f} "
                     f"| {h} x {w} |")
    return "\n".join(lines) + "\n"


def write_sweep_outputs(out_dir: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    """Writes `sam2_variants.json` and `sam2_variants.md` under `out_dir`; returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / JSON_NAME, out_dir / MD_NAME
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(f"{markdown_table(report['rows'])}\nChosen default (min IoU {report['min_iou']:.2f}): "
                       f"`{report['chosen']}`\n")
    return json_path, md_path


def _resolve(root: Path, base: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path) if (base / path).exists() else (root / path).resolve()


def whisking_inputs(cfg: DictConfig, root: Path) -> tuple[Tensor, list[int], int, Tensor]:
    """`(frames, frame_idx, keyframe_index, prompt)`: the head-to-head's stride-3 frames and SAM 3D keyframe prompt."""
    w = cfg.whisking
    demo = _resolve(root, root, str(w.demo_root))
    video, mesh_path = _resolve(root, demo, str(w.video)), _resolve(root, demo, str(w.mesh_path))
    layout_json = _resolve(root, demo, str(w.layout_json))
    _, _, K = read_baseline_layout(layout_json)
    shape_pose, shape_scale = read_keyframe_layout(layout_json, int(w.keyframe_frame_idx))
    n_full, stride = int(w.n_frames_full), int(w.stride)
    frames, idx = decode_frames(video, stride=stride, max_frames=len(sample_frame_indices(n_full, stride)))
    k = int(np.argmin(np.abs(np.asarray(idx) - int(w.keyframe_frame_idx))))
    mesh = cast("trimesh.Trimesh", trimesh.load(str(mesh_path), process=False, force="mesh"))
    pose = shape_pose.double().numpy()
    posed = shape_scale * np.asarray(mesh.vertices, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
    hw = (int(frames.shape[1]), int(frames.shape[2]))
    prompt = render_silhouette(posed, np.asarray(mesh.faces, dtype=np.int64), K, hw)
    return frames, idx, k, torch.from_numpy(prompt)


def run_bench(cfg: DictConfig, root: Path) -> dict[str, Any]:
    """Runs the sweep on the whisking demo and writes its outputs; returns the report."""
    b = cfg.sam2_bench
    sizes, short_sides = [str(s) for s in b.sizes], [int(s) for s in b.short_sides]
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available: these timings are not comparable with the GPU budget")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, idx, k, prompt = whisking_inputs(cfg, root)
    native_short_side = min(int(frames.shape[1]), int(frames.shape[2]))
    if sweep_labels(sizes, short_sides, native_short_side)[0] != REFERENCE_LABEL:
        raise ValueError(f"the first sweep row must be the reference {REFERENCE_LABEL!r} (large at the native "
                         f"{native_short_side} px short side)")
    rows = run_sweep(frames, k, prompt, sizes, short_sides,
                     load_predictor=lambda model_id: masks_sam2.load_sam2_predictor(device=device, model_id=model_id),
                     device=device, bf16=bool(b.bf16), warmup_frames=int(b.warmup_frames))
    meta = {"video": str(_resolve(root, _resolve(root, root, str(cfg.whisking.demo_root)), str(cfg.whisking.video))),
            "frame_idx": idx, "keyframe_index": k, "keyframe_frame_idx": idx[k], "prompt_pixels": int(prompt.sum()),
            "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "",
            "torch": torch.__version__, "bf16": bool(b.bf16), "warmup_frames": int(b.warmup_frames),
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "timing": "propagate_masks_at wall time, CUDA-synchronized; model load and warm-up excluded"}
    report = sweep_report(rows, float(b.min_iou), meta)
    json_path, _ = write_sweep_outputs(_resolve(root, root, str(b.out_dir)), report)
    logger.info("wrote %s; chosen default %s", json_path, report["chosen"])
    return report


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    run_bench(cfg, Path(hydra.utils.get_original_cwd()))


if __name__ == "__main__":
    main()
