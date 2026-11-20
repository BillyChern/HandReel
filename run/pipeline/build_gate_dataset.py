"""Build the gate token dataset: DINOv2 tokens of the 100DOH train clips and DexYCB sequences, and the gate index.

The GPU step. The DexYCB id file is checked first, by pure string checks, against
`gate_data.dexycb_allowed_subjects` and the sequence names in `gate_data.dexycb_exclude_files`; a refused id writes
nothing and no DexYCB path is built before it. Then, for each 100DOH train clip (`doh_train_entries`: test-split entries
are dropped before any path is built) and each DexYCB id, the deployed stage workers run in `gate_data.work_dir/<clip>`:
decode (ffmpeg at `gate_data.stride` for 100DOH, every JPEG for DexYCB so a token row is a source frame), then the
encoder with `gate_data.crop`. `encoder.npz` moves to `gate_data.token_dir/<clip>.npz` and `decode.npz` is deleted in a
`finally`, so only DINOv2 tokens persist. A clip whose token file already holds the expected rows, crop and frame
indices (`0, s, 2s, ...` at its decode stride `s`) is skipped.
DexYCB frame labels are read before a sequence is encoded; a sequence without a usable window is skipped with its
reason. The index (`fastdo.gate.dataset.write_gate_index`) records the 100DOH held-out split counts, the id and
exclusion files' sha256, every `gate_data` threshold, the per-source and per-partition counts and the skipped sequences.

  cd ${FASTDO_ROOT} && env FASTDO_GPU_MEM_GB=4 FASTDO_CPU_THREADS=2 \
    OMP_NUM_THREADS=2 nice -n 19 uv run --extra dev --extra frontend python run/pipeline/build_gate_dataset.py \
    paths.root=${FASTDO_ROOT} gate_data.doh_root=/path/to/authorized/100DOH \
    gate_data.dexycb_ids_file=/path/to/dexycb_ids.json 'gate_data.dexycb_exclude_files=[/path/to/eval_sequences.json]'
The exclusion list is empty by default: pass the JSON files of your DexYCB evaluation and calibration sequences, as
above, so that no evaluation sequence enters the gate's training data.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from fastdo.bench.bench_clips import (
    doh_clip_path,
    doh_train_entries,
    make_image_sequence_decode_worker,
)
from fastdo.bench.bench_stages import resolve_path
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.engine import StageFn, make_decode_worker, make_encode_worker
from fastdo.engine.worker_support import DECODE_FILE
from fastdo.engine.workers_frames import ENCODER_FILE
from fastdo.gate.dataset import (
    GateItem,
    check_dexycb_ids,
    dexycb_gate_items,
    dexycb_token_name,
    doh_gate_items,
    excluded_dexycb_sequences,
    item_counts,
    read_dexycb_frame_labels,
    read_dexycb_id_file,
    write_gate_index,
)
from fastdo.gate.features import DEFAULT_CROP
from fastdo.gate.labels import dexycb_partition
from fastdo.io.video import sample_frame_indices

logger = logging.getLogger(__name__)

DecodeFactory = Callable[[str, Mapping[str, Path], int], StageFn]     # (kind, clip_id -> source, stride) -> worker
EncodeFactory = Callable[[str], StageFn]                               # crop -> encoder worker


def default_decode_factory(kind: str, sources: Mapping[str, Path], stride: int) -> StageFn:
    """The deployed decode worker: ffmpeg for `"video"`, JPEG sequences for `"image_sequence"`."""
    if kind == "video":
        return make_decode_worker(sources, stride=stride)
    if kind == "image_sequence":
        return make_image_sequence_decode_worker(sources, stride=stride)
    raise ValueError(f"unknown decode kind {kind!r}")


def default_encode_factory(crop: str) -> StageFn:
    """The deployed DINOv2-large encoder worker with `crop`."""
    return make_encode_worker(crop=crop)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _token_rows(path: Path, crop: str, step: int) -> int | None:
    """Row count of an existing token file whose `frame_idx` is the decode stage's `0, step, 2*step, ...` (one entry per
    token row) and whose crop is `crop`, else None: tokens from another stride or crop are never reused."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        rows, idx = int(data["tokens"].shape[0]), np.asarray(data["frame_idx"], dtype=np.int64)
        found = str(data["crop"]) if "crop" in data.files else DEFAULT_CROP
    same_stride = np.array_equal(idx, np.arange(rows, dtype=np.int64) * step)
    return rows if same_stride and found == crop else None


def _encode_clip(clip_id: str, decode: StageFn, encode: StageFn, work_dir: Path, token_path: Path) -> None:
    work = work_dir / clip_id
    work.mkdir(parents=True, exist_ok=True)
    try:
        decode(clip_id, work)
        encode(clip_id, work)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work / ENCODER_FILE, token_path)
    finally:
        (work / DECODE_FILE).unlink(missing_ok=True)
        if not any(work.iterdir()):
            work.rmdir()


def build_gate_dataset(cfg: DictConfig, root: Path, decode_factory: DecodeFactory = default_decode_factory,
                       encode_factory: EncodeFactory = default_encode_factory) -> tuple[str, dict[str, Any]]:
    """Builds the token files and writes the index (module docstring); returns `(index sha256, provenance)`.

    Raises:
        ValueError: If `gate_data.doh_root` or `gate_data.dexycb_ids_file` is null, a DexYCB id is refused, the
            partition is degenerate, or a 100DOH train entry lacks its contact annotation.
        FileExistsError: If the index exists and `gate_data.overwrite` is false.
        FileNotFoundError: If an exclusion file, or a DexYCB sequence's mesh or labels, is missing.
    """
    gd = cfg.gate_data
    if gd.doh_root is None or gd.dexycb_ids_file is None:
        raise ValueError("gate_data.doh_root and gate_data.dexycb_ids_file must both be set")
    index_path = resolve_path(root, gd.index_path)
    if index_path.exists() and not bool(gd.overwrite):
        raise FileExistsError(f"{index_path} exists; set gate_data.overwrite=true to rebuild the gate index")
    ids_file = resolve_path(root, gd.dexycb_ids_file)
    exclude_files = [resolve_path(root, path) for path in (gd.get("dexycb_exclude_files") or [])]
    excluded = excluded_dexycb_sequences(exclude_files)
    ids = check_dexycb_ids(read_dexycb_id_file(ids_file), [str(s) for s in gd.dexycb_allowed_subjects], excluded)
    partitions = dexycb_partition(ids, str(gd.val_subject), int(gd.val_every))

    doh_root, dexycb_root = resolve_path(root, gd.doh_root), resolve_path(root, cfg.dexycb.root)
    token_dir, work_dir = resolve_path(root, gd.token_dir), resolve_path(root, gd.work_dir)
    stride, crop = int(gd.stride), str(gd.crop)
    thresholds: dict[str, Any] = {"contact_m": float(gd.contact_m), "free_m": float(gd.free_m),
                                  "still_m": float(gd.still_m), "min_negative_frames": int(gd.min_negative_frames)}
    entries, held_out = doh_train_entries(doh_root)
    doh_items = doh_gate_items(doh_root, token_dir)             # every contact label is checked before any decode
    encode = encode_factory(crop)
    video_decode = decode_factory("video", {item.item_id: doh_clip_path(doh_root, entry)
                                            for item, entry in zip(doh_items, entries, strict=True)}, stride)
    for item, entry in zip(doh_items, entries, strict=True):
        token_path = Path(item.token_file)
        rows = _token_rows(token_path, crop, stride)
        expected = (len(sample_frame_indices(int(entry["decoded_frames"]), stride)) if "decoded_frames" in entry
                    else rows)
        if rows is None or rows != expected:
            _encode_clip(item.item_id, video_decode, encode, work_dir, token_path)
        logger.info("100DOH %s: tokens in %s (%s)", item.clip_key, token_path, item.partition)

    sequence_decode = decode_factory("image_sequence", {dexycb_token_name(s): dexycb_root / s for s in ids}, 1)
    dexycb_items: list[GateItem] = []
    skipped: list[dict[str, str]] = []
    for seq_id in ids:
        clip_id = dexycb_token_name(seq_id)
        token_path = token_dir / f"{clip_id}.npz"
        try:
            labels = read_dexycb_frame_labels(dexycb_root, seq_id, int(gd.n_object_points))
        except ValueError as exc:
            skipped.append({"seq_id": seq_id, "reason": f"frame labels: {exc}"})
            continue
        items = dexycb_gate_items(seq_id, labels, stride, partitions[seq_id], str(token_path), **thresholds)
        if not items:
            skipped.append({"seq_id": seq_id, "reason": "no contact window and no no-contact prefix window"})
            continue
        n_frames = int(labels.distance_m.shape[0])
        if _token_rows(token_path, crop, 1) != n_frames:
            _encode_clip(clip_id, sequence_decode, encode, work_dir, token_path)
        rows = _token_rows(token_path, crop, 1)
        if rows != n_frames:
            skipped.append({"seq_id": seq_id, "reason": f"{rows} token rows for {n_frames} label frames"})
            continue
        dexycb_items.extend(items)

    all_items = doh_items + dexycb_items
    provenance: dict[str, Any] = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "doh_root": str(doh_root),
        "doh_split": "train", "doh_held_out": held_out, "dexycb_root": str(dexycb_root),
        "dexycb_ids_file": str(ids_file), "dexycb_ids_sha256": _sha256(ids_file), "n_dexycb_ids": len(ids),
        "dexycb_ids_per_subject": dict(sorted(Counter(seq_id.split("/")[0] for seq_id in ids).items())),
        "dexycb_allowed_subjects": [str(s) for s in gd.dexycb_allowed_subjects],
        "dexycb_exclude_files": {str(path): _sha256(path) for path in exclude_files},
        "n_excluded_sequence_names": len(excluded), "dexycb_partitions": dict(sorted(Counter(partitions.values()).items())),
        "val_subject": str(gd.val_subject), "val_every": int(gd.val_every), "crop": crop, "stride": stride,
        **thresholds, "n_object_points": int(gd.n_object_points), "counts": item_counts(all_items),
        "skipped_dexycb": skipped, "token_dir": str(token_dir), "work_dir": str(work_dir),
    }
    sha = write_gate_index(index_path, all_items, provenance)
    logger.info("gate index %s (sha256 %s): %s; %d DexYCB sequences skipped", index_path, sha[:12],
                provenance["counts"]["partition_source"], len(skipped))
    return sha, provenance


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    try:
        root = Path(hydra.utils.get_original_cwd())
    except ValueError:                    # called with a composed config, outside a Hydra run (tests)
        root = Path.cwd()
    build_gate_dataset(cfg, root)


if __name__ == "__main__":
    main()
