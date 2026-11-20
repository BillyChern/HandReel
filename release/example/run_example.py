#!/usr/bin/env python3
"""Run the reconstruction on ONE window of ONE public video, then write its outputs and one timing-ledger row.

    uv run --frozen --extra dev --extra frontend python release/example/run_example.py --out outputs/example
    uv run --frozen python release/example/run_example.py --stub --video <any file> --out outputs/example_stub   # CPU, no model
    uv run --frozen --extra dev --extra frontend python release/example/run_example.py --spec my_clip.json \
      --video my_clip.mp4 --out outputs/my_clip                                     # your own video

The input is `example_input.json`: a 6.85 s head-mounted clip from the V2D Challenge Track 3 public split (CC BY 4.0;
see that file for the attribution). It is not shipped. On first use it is downloaded from the pinned dataset revision
into `--data-dir`: the download goes to `<file>.part`, is hashed, and is renamed to the file only when its sha256 is
the pinned one (otherwise to `<file>.sha256-mismatch`, kept for inspection). `--video` points at a copy you already
have.

Camera. The clip is `ego_cam_a`, a 108-degree HFOV lens with a 14-term rational distortion model, on a device mounted
upside down (`rotation_deg` 180 in the dataset's calibration). The stored frames are upright (frame 0 checked by eye:
floor, table and hands at the bottom), so nothing is rotated. They are not undistorted either: the pipeline estimates
a pinhole K per clip with MoGe-2 (the intrinsics stage), as it does for internet video, which has no calibration.
For this example, the input specification (`example_input.json`, `camera.distortion_at_box_px`) reports that applying
the factory undistortion would move pixels near the cup by 1-44 px and pixels at the left and right image edges by
about 205-259 px. The pipeline uses the original frames. The release does not independently recompute these ranges.
The run is a
demonstration of the pipeline, not a calibrated measurement (release/QUICKSTART.md, step 4).

What runs: the production engine (`fastdo.engine`, the same stage workers as `run/pipeline/bench_throughput.py`) on a
single clip, through the batched runner of the default config, with the default stages decode, mask, shape,
intrinsics, hands, geometry, tracks and solve. The keyframe object mask is made before the run (untimed, as the
benchmark does): SAM 2.1's image predictor on the box of `example_input.json`, freed before the engine starts. qc runs
only with `--qc-rule <AcceptanceRule json>`, because the qc stage has no accept-all default. Hydra overrides of
`run/conf/config.yaml` go through `--set key=value` (repeatable). CoTracker3's torch.hub code is pinned: unless
`FASTDO_COTRACKER_HUB_REF` is already set, it is set to `fastdo.model_pins.COTRACKER_HUB_COMMIT`, and the report
records the value used. Preflight: `scripts/fetch_weights.py --yes --only cotracker3` must have run before the GPU
run. Without --stub, a run with the tracks stage stops before anything else when that ref's torch.hub code directory
(`<torch hub dir>/facebookresearch_co-tracker_<ref>`, with its `hubconf.py`) or the checkpoint
`<torch hub dir>/checkpoints/scaled_online.pth` is missing, instead of letting torch.hub fetch unverified code and
weights at run time (`require_cotracker_hub`).

Outputs, under `--out` (generated, never shipped):
- `engine/<clip_id>/`: the stage artifacts (`decode.npz`, `mask.npz`, `mesh.obj`, `intrinsics.npz`, `hands.npz`,
  `geometry.npz`, `tracks.npz`, `solve.npz`, and `clip_record.npz` when qc ran) and `profile.json`;
- `manifest.parquet`: the engine manifest (state and seconds per stage);
- `example_report.json` (`REPORT_SCHEMA`): input, pins, stage states and seconds, artifact list, host;
- `timing_ledger.jsonl` (`LEDGER_SCHEMA`): one appended row per run, in the cost ledger's unit-row form;
- `ATTRIBUTION.txt`: the dataset credit and licence the outputs carry (CC BY 4.0 requires it; written only when the
  input is the pinned clip, any other file gets no credit line), and the model licences that also bind them.

`delivered` is None when qc did not run: a window is delivered only when the qc stage accepts it. `frames` counts the
window's source frames; `timing_basis.frames_decoded` is the number of sampled rows planned for the requested
window and stride.

Your own video: `--spec <json>` replaces `example_input.json` with a description you write (start from
`release/example/spec_template.json`): `clip_id`, `domain`, `video` (`fps`, `n_frames`, `width`, `height`), `window`,
`keyframe` and `box_xyxy` on the keyframe, in source pixels. `--video` is then required, nothing is downloaded and no
sha256 is checked (there is no pin for your file); the report records the file's sha256, and `ATTRIBUTION.txt` names
no dataset. The video block must match the file: its width and height are checked with `ffprobe` before the run.

`--stub` replaces every stage worker and the box-to-mask step with placeholders that write empty artifacts and never
imports a model, but keeps the batched runner and its config validation, so argument parsing, config loading,
orchestration and output schema can be checked on a CPU-only machine. Its seconds are not timings. `--stub` never
downloads the clip: give `--video <file>` (any file; the stub does not decode it), or put the clip in `--data-dir`
first. Without either, it exits with a one-line message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPEC_FILE = Path(__file__).with_name("example_input.json")
OWN_SPEC_KEYS = ("clip_id", "domain", "video", "window", "keyframe", "box_xyxy")   # what a --spec file must hold
OWN_VIDEO_KEYS = ("fps", "n_frames", "width", "height")
CLIP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")                       # a directory name on any filesystem
DEFAULT_STAGES: tuple[str, ...] = ("decode", "mask", "shape", "intrinsics", "hands", "geometry", "tracks", "solve")
REPORT_SCHEMA = "fastdo.example.report/v2"
LEDGER_SCHEMA = "fastdo.example.timing_row/v2"
LEDGER_FILE, REPORT_FILE, MANIFEST_FILE = "timing_ledger.jsonl", "example_report.json", "manifest.parquet"
ATTRIBUTION_FILE = "ATTRIBUTION.txt"
VERSION_FILE = "VERSION"            # read when the tree is not a git checkout (e.g. a source archive)
URL_TIMEOUT_S = 120.0               # per blocking socket operation of the clip download
COTRACKER_REF_ENV = "FASTDO_COTRACKER_HUB_REF"
COTRACKER_HUB_REPO_DIR = "facebookresearch_co-tracker"   # torch.hub's `<owner>_<repo>` prefix of its code directory
COTRACKER_CHECKPOINT = "checkpoints/scaled_online.pth"   # hubconf's load_state_dict_from_url target, under the hub dir
COTRACKER_FETCH = "uv run --frozen python scripts/fetch_weights.py --yes --only cotracker3"
GPU_MEM_ENV, STUB_GPU_MEM_GB = "FASTDO_GPU_MEM_GB", "32"     # fastdo.bench.gpu_budget.ENV_VAR; QUICKSTART's value
# Artifact each stage leaves in the clip directory (the engine's own file names; tests/release pins them to the
# engine's constants).
STAGE_ARTIFACTS: dict[str, str] = {
    "decode": "decode.npz", "mask": "mask.npz", "shape": "mesh.obj", "intrinsics": "intrinsics.npz",
    "hands": "hands.npz", "geometry": "geometry.npz", "tracks": "tracks.npz", "solve": "solve.npz",
    "qc": "clip_record.npz",
}
REPORT_KEYS = ("schema", "clip_id", "stub", "input", "camera", "window", "stride", "stages", "stage_states",
               "stage_seconds", "total_seconds", "delivered", "errors", "artifacts", "execution", "pins", "host",
               "commit", "created_utc", "ledger_row", "attribution")
LEDGER_KEYS = ("schema", "unit_id", "method", "corpus", "assigned", "run", "analysed", "delivered", "exclusion_reason",
               "seconds", "frames", "cluster", "metrics", "timing_basis", "stage_seconds", "clip_id", "input_sha256",
               "commit", "created_utc")
logger = logging.getLogger("run_example")


def load_spec(path: Path = SPEC_FILE, own: bool = False) -> dict[str, Any]:
    """The input description: `example_input.json`, or with `own` a user's `--spec` file (`OWN_SPEC_KEYS` required;
    `source` defaults to "own_video", `object` to "object", and it carries no dataset, file or sha256 pin).

    Raises:
        ValueError: On a missing key, an unusable clip id, or a window, keyframe or box outside the video.
    """
    try:
        spec = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON ({e})") from None
    if own:
        missing = [k for k in OWN_SPEC_KEYS if k not in spec]
        missing += [f"video.{k}" for k in OWN_VIDEO_KEYS if k not in spec.get("video", {})]
        if missing:
            raise ValueError(f"{path}: missing {', '.join(missing)} (see release/example/spec_template.json)")
        if not CLIP_ID.fullmatch(str(spec["clip_id"])):
            raise ValueError(f"{path}: clip_id {spec['clip_id']!r} must be letters, digits, '_', '.' or '-'")
        for k in ("dataset", "file", "sha256"):           # a pin of the example clip never applies to your file
            spec.pop(k, None)
        spec.setdefault("source", "own_video")
        spec.setdefault("object", "object")
        spec.setdefault("camera", {"intrinsics": "estimated per clip by MoGe-2 (the intrinsics stage)"})
    # Frame numbers and sizes are whole numbers as written: 3.5 would be cut to frame 3 further on while the report
    # kept 3.5, and true would count as frame 1. Each is refused before any conversion.
    window = spec["window"]
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError(f"{path}: window must be [first frame, frame after the last]")
    counts = {"window[0]": window[0], "window[1]": window[1], "keyframe": spec["keyframe"],
              **{f"video.{k}": spec["video"].get(k) for k in ("n_frames", "width", "height")}}
    wrong = [k for k, v in counts.items() if not isinstance(v, int) or isinstance(v, bool)]
    if wrong:
        raise ValueError(f"{path}: {', '.join(wrong)} must be whole numbers, written without a decimal point")
    start, stop = window
    if not 0 <= start <= spec["keyframe"] < stop <= spec["video"]["n_frames"]:
        raise ValueError(f"{path}: need 0 <= window start <= keyframe < window stop <= n_frames")
    x0, y0, x1, y1 = spec["box_xyxy"]
    if not (0 <= x0 < x1 <= spec["video"]["width"] and 0 <= y0 < y1 <= spec["video"]["height"]):
        raise ValueError(f"{path}: box_xyxy {spec['box_xyxy']} is not inside the {spec['video']['width']}x"
                         f"{spec['video']['height']} frame (x0 < x1, y0 < y1, source pixels)")
    if not float(spec["video"]["fps"]) > 0:
        raise ValueError(f"{path}: video.fps must be positive")
    return spec


def require_decoded_keyframe(start: int, rows: Sequence[int], stride: int, keyframe: int) -> None:
    """Refuse a keyframe that the run does not decode.

    The object mask is made on the source frame `keyframe` and attached to a decoded row. Off the decoded grid the
    nearest row is another frame, so the mask and the box would describe a frame the pipeline never pairs them with.

    Args:
        start: first source frame of the window.
        rows: the decoded frames, as offsets from `start` (`sample_frame_indices`).
        stride: the decode stride, for the message.
        keyframe: the spec's keyframe, in source frames.

    Raises:
        ValueError: If `keyframe` is not one of the decoded frames; the message names the nearest ones.
    """
    frames = [start + int(r) for r in rows]
    if keyframe in frames:
        return
    near = [max((f for f in frames if f < keyframe), default=None),
            min((f for f in frames if f > keyframe), default=None)]
    names = " or ".join(str(f) for f in near if f is not None)
    raise ValueError(f"keyframe {keyframe} is not a decoded frame: this window is decoded every {stride} source frames "
                     f"from frame {start}, so a mask made on frame {keyframe} would be attached to another frame; use "
                     f"keyframe {names} and give box_xyxy on that frame")


def probe_size(video: Path) -> tuple[int, int]:
    """`(width, height)` of the first video stream (ffprobe).

    Raises:
        ValueError: If ffprobe cannot read a video stream from the file.
    """
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=False)
    try:
        w, h = (int(v) for v in r.stdout.strip().split(",")[:2])
    except ValueError:
        raise ValueError(f"{video}: ffprobe found no video stream ({r.stderr.strip() or 'no output'})") from None
    return w, h


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="output directory (created; its clip directory is replaced)")
    ap.add_argument("--video", type=Path, help="a local copy of the input clip (default: <data-dir>/<file>)")
    ap.add_argument("--spec", type=Path,
                    help="your own input description instead of example_input.json (start from "
                         "release/example/spec_template.json); needs --video; nothing is downloaded or sha256-checked")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "example", help="where the clip is downloaded")
    ap.add_argument("--no-fetch", action="store_true", help="fail instead of downloading a missing clip")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES), help="comma-separated engine stages, in order")
    ap.add_argument("--qc-rule", type=Path, help="a saved AcceptanceRule JSON; adds the qc stage")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="Hydra override of run/conf/config.yaml (repeatable)")
    ap.add_argument("--card", choices=("dedicated", "shared", "unrecorded"), default="unrecorded",
                    help="was the GPU otherwise idle? recorded in the ledger row's timing basis")
    ap.add_argument("--stub", action="store_true",
                    help="placeholder workers, no model, CPU only (smoke test); never downloads the clip, so it needs "
                         "--video <file> (any file) unless the clip is already in --data-dir")
    a = ap.parse_args(argv)
    if a.spec is not None and a.video is None:
        ap.error("--spec needs --video <your clip>: nothing is downloaded for your own spec")
    a.stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
    if a.qc_rule is not None and "qc" not in a.stages:
        a.stages = (*a.stages, "qc")
    if "qc" in a.stages and a.qc_rule is None and not a.stub:
        ap.error("the qc stage needs --qc-rule (it has no accept-all default)")
    if not a.stages or len(set(a.stages)) != len(a.stages):
        ap.error(f"--stages must name distinct stages, got {a.stages}")
    return a


def compose_config(overrides: Sequence[str]) -> Any:
    """`run/conf/config.yaml` composed with `overrides` (no Hydra run directory is created)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "run" / "conf"), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def run_overrides(a: argparse.Namespace, spec: Mapping[str, Any], stride: int) -> list[str]:
    out = a.out.resolve()
    base = [f"paths.root={REPO}", f"bench.out_dir={out}", f"bench.artifact_root={out / 'engine'}",
            f"bench.stride={stride}", "bench.overwrite=true", "bench.stages=[" + ",".join(a.stages) + "]"]
    if a.qc_rule is not None:
        base.append(f"qc_stage.rule_path={a.qc_rule.resolve()}")
    return base + list(a.overrides)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_cotracker_hub(hub_dir: Path, ref: str) -> None:
    """Preflight of the tracks stage: CoTracker3's torch.hub code at `ref` and its checkpoint must already be in
    `hub_dir` (`torch.hub.get_dir()`), where `scripts/fetch_weights.py --yes --only cotracker3` puts them. Otherwise
    torch.hub would download the code and the weights, unverified, at run time.

    Raises:
        SystemExit: If the code directory's `hubconf.py` or the checkpoint is missing.
    """
    code = hub_dir / f"{COTRACKER_HUB_REPO_DIR}_{ref.replace('/', '_')}"       # torch.hub's own directory name
    missing = [str(p) for p in (code / "hubconf.py", hub_dir / COTRACKER_CHECKPOINT) if not p.is_file()]
    if missing:
        raise SystemExit(f"CoTracker3 is not fetched: missing {', '.join(missing)}. Run `{COTRACKER_FETCH}` before "
                         f"the GPU example (release/QUICKSTART.md, step 3); it verifies the checkpoint's sha256 and "
                         f"unpacks the code at the pinned commit. ({COTRACKER_REF_ENV}={ref}; unset it to use the pin.)")


SAM3D_REPO = "facebook/sam-3d-objects"
MANO_FILES = ("_DATA/data/mano/MANO_RIGHT.pkl", "_DATA/data_left/mano_left/MANO_LEFT.pkl")   # where HaWoR reads them


def missing_gated_inputs(stages: Sequence[str], hawor_root: Path,
                         cached: Callable[[str, str, str], Any] | None = None) -> list[str]:
    """The SAM 3D checkpoint files (shape stage) and MANO files (hands stage) that are not in place, one line each.

    SAM 3D files are looked up in the Hugging Face cache at the pinned revision (`huggingface_hub`'s own lookup, so
    `HF_HOME` / `HF_HUB_CACHE` are honoured); the list of files is `release/models.json`. MANO is looked up in the
    HaWoR checkout, where the hands driver reads it. Nothing is downloaded.
    """
    from fastdo.model_pins import HF_REVISIONS

    if cached is None:
        from huggingface_hub import try_to_load_from_cache as cached
    out: list[str] = []
    if "shape" in stages:
        rev = HF_REVISIONS[SAM3D_REPO]
        entry = next(m for m in json.loads((REPO / "release" / "models.json").read_text())["models"]
                     if m["key"] == "sam3d")
        gone = [f["file"] for f in entry["files"] if not isinstance(cached(SAM3D_REPO, f["file"], revision=rev), str)]
        if gone:
            out.append(f"SAM 3D Objects ({SAM3D_REPO} at {rev}): {len(gone)} of {len(entry['files'])} files are not in "
                       f"the Hugging Face cache (first: {gone[0]}). It is gated: accept its terms at "
                       f"https://huggingface.co/{SAM3D_REPO}, run `uv run --frozen hf auth login`, then "
                       "`uv run --frozen python scripts/fetch_weights.py --yes --only sam3d`")
    if "hands" in stages:
        for rel in MANO_FILES:
            if not (hawor_root / rel).is_file():
                out.append(f"MANO: {hawor_root / rel} is missing. Register at https://mano.is.tue.mpg.de, download "
                           "MANO_RIGHT.pkl and MANO_LEFT.pkl and copy them there (release/QUICKSTART.md, step 3)")
    return out


def resolve_input(a: argparse.Namespace, spec: Mapping[str, Any],
                  fetch: Callable[[str, Path], None] | None = None) -> tuple[Path, str, bool]:
    """`(video path, its sha256, whether it matches the spec)`; downloads the clip when it is missing.

    Raises:
        FileNotFoundError: If the clip is missing and fetching is off (--no-fetch or --stub); `main` turns it into a
            one-line exit message.
        ValueError: If the sha256 differs from the spec's (outside --stub), or a fresh download does.
        RuntimeError: If a leftover `.part` or `.sha256-mismatch` of an earlier download is in the way.
    """
    if a.spec is not None:                          # your own video: no download, no pin
        if not a.video.is_file():
            raise FileNotFoundError(f"{a.video} not found (--video of your --spec)")
        if not a.stub:
            size = probe_size(a.video)
            want = (int(spec["video"]["width"]), int(spec["video"]["height"]))
            if size != want:
                raise ValueError(f"{a.video} is {size[0]}x{size[1]}, but {a.spec} says {want[0]}x{want[1]}")
        return a.video, sha256_file(a.video), False
    video = a.video if a.video is not None else a.data_dir / spec["file"]
    if not video.is_file():
        if a.no_fetch or a.stub:
            why = "--stub never downloads it" if a.stub else "--no-fetch is set"
            raise FileNotFoundError(f"{video} not found and fetching is off ({why}): pass --video <file>"
                                    + (" (any file works with --stub)" if a.stub else ", or drop --no-fetch"))
        ds = spec["dataset"]
        url = f"https://huggingface.co/datasets/{ds['hf_repo']}/resolve/{ds['revision']}/{spec['file']}"
        logger.info("downloading %s (%s; %s)", url, ds["licence"], ds["attribution"])
        download_verified(url, video, spec["sha256"], fetch or _url_fetch)
    digest = sha256_file(video)
    ok = digest == spec["sha256"]
    if not ok and not a.stub:
        raise ValueError(f"{video}: sha256 {digest} differs from the pinned {spec['sha256']}")
    return video, digest, ok


def download_verified(url: str, dest: Path, sha256: str, fetch: Callable[[str, Path], None]) -> None:
    """Download `url` to `<dest>.part`, then rename it to `dest` only when its sha256 is `sha256`; otherwise to
    `<dest>.sha256-mismatch` (kept for inspection, never read) and raise. A leftover of either name is refused, never
    reused or deleted.

    Raises:
        ValueError: On a digest mismatch.
        RuntimeError: On a leftover temporary.
    """
    part, bad = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".sha256-mismatch")
    for leftover in (part, bad):
        if leftover.exists() or leftover.is_symlink():
            raise RuntimeError(f"{leftover} already exists (an earlier download?); inspect and remove it, then re-run")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch(url, part)
    digest = sha256_file(part)
    if digest != sha256:
        part.replace(bad)
        raise ValueError(f"{url}: sha256 {digest} differs from the pinned {sha256}; kept as {bad}")
    part.replace(dest)


def _url_fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=URL_TIMEOUT_S) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def read_frame(video: Path, index: int, width: int, height: int) -> np.ndarray:
    """Source frame `index` as `(H,W,3)` uint8 RGB (ffmpeg; every source frame kept, none duplicated or dropped)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{index})",
                          "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    if len(raw) != width * height * 3:
        raise ValueError(f"{video}: frame {index} decoded to {len(raw)} bytes, expected {width}x{height}x3")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def box_mask(frame: np.ndarray, box: Sequence[float], device: str,
             loader: Callable[[str], Any] | None = None) -> np.ndarray:
    """SAM 2.1 Hiera-L image predictor's single mask for `box` (x0, y0, x1, y1) on `frame`. The predictor is freed
    (and the CUDA cache emptied) before returning, so the engine starts with the card as the benchmark does."""
    import gc

    import torch

    if loader is None:
        from fastdo.shape.masks_sam2 import load_sam2_image_predictor

        def loader(dev: str) -> Any:
            return load_sam2_image_predictor(device=dev)
    predictor = loader(device)
    try:
        with torch.inference_mode():
            predictor.set_image(frame)
            masks, _, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False)
        return np.asarray(masks[0] > 0.5)
    finally:
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stub_box_mask(frame_hw: tuple[int, int], box: Sequence[float]) -> np.ndarray:
    mask = np.zeros(frame_hw, dtype=bool)
    x0, y0, x1, y1 = (round(v) for v in box)
    mask[max(y0, 0):y1, max(x0, 0):x1] = True
    return mask


def stub_stage_fns(stages: Sequence[str]) -> dict[str, Callable[[str, Path], None]]:
    """Placeholder workers: each writes its stage's artifact name with no content."""
    def make(stage: str) -> Callable[[str, Path], None]:
        def worker(_clip_id: str, artifact_dir: Path) -> None:
            (artifact_dir / STAGE_ARTIFACTS.get(stage, f"{stage}.stub")).write_bytes(b"")
        return worker
    return {stage: make(stage) for stage in stages}


NVIDIA_PROC = Path("/proc/driver/nvidia")
_VERSION_TOKEN = re.compile(r"\d+\.\d+(?:\.\d+)*")


def parse_driver_version(text: str) -> str:
    """The driver version from the NVRM line of /proc/driver/nvidia/version: its first token shaped like
    digits.digits[.digits]. Both the proprietary line ("... x86_64 Kernel Module  550.54.15 ...") and the open one
    ("... Open Kernel Module for x86_64  595.91.07 ...") parse; 'unrecorded' when no token matches."""
    for line in text.splitlines():
        if line.startswith("NVRM version:"):
            for token in line.split():
                if _VERSION_TOKEN.fullmatch(token):
                    return token
    return "unrecorded"


def _proc_cards(proc: Path) -> list[tuple[str, str]]:
    """(model, UUID) per card, in PCI bus order."""
    cards = []
    for info in sorted((proc / "gpus").glob("*/information")):
        fields = dict(line.split(":", 1) for line in info.read_text().splitlines() if ":" in line)
        cards.append((fields.get("Model", "").strip(), fields.get("GPU UUID", "").strip()))
    return cards


def visible_cards(cards: Sequence[tuple[str, str]], visible: str | None) -> list[str]:
    """The models of the cards CUDA_VISIBLE_DEVICES leaves visible. Unset: every card. Empty: none. Otherwise its
    comma-separated indices or GPU UUIDs (a unique prefix is enough), stopping at the first entry
    that names no card, as CUDA does. A MIG entry is recorded as such: /proc does not map it to a card.
    Indices are read in PCI bus order. CUDA numbers cards fastest-first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, so on
    a host with mixed card models set CUDA_DEVICE_ORDER=PCI_BUS_ID for this record to name the cards CUDA uses."""
    if visible is None:
        return [model for model, _ in cards]
    models = []
    for entry in (e.strip() for e in visible.split(",") if e.strip()):
        if entry.startswith("MIG-"):
            models.append("MIG instance")
            continue
        if entry.isdigit():
            hit = [cards[int(entry)]] if int(entry) < len(cards) else []
        else:
            hit = [c for c in cards if c[1] and c[1].startswith(entry)] if entry.startswith("GPU-") else []
        if len(hit) != 1:
            break
        models.append(hit[0][0])
    return models


def gpu_info(proc: Path = NVIDIA_PROC) -> dict[str, str]:
    """GPU model and driver from /proc (no CUDA call); 'none' on a host without the NVIDIA driver or when
    CUDA_VISIBLE_DEVICES hides every card. Only the cards CUDA_VISIBLE_DEVICES leaves visible are listed."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    models = visible_cards(_proc_cards(proc), visible)
    ver = proc / "version"
    driver = parse_driver_version(ver.read_text()) if ver.is_file() else "none"
    return {"gpus": ", ".join(m for m in models if m) or "none", "driver": driver, "cuda_visible_devices": str(visible),
            "cpu": platform.processor() or platform.machine(), "python": platform.python_version()}


def git_commit(repo: Path = REPO) -> str:
    """The checkout's commit; in a release archive, zip or tar (no `.git` of its own, maybe unpacked inside someone else's
    repository), the `VERSION` file's content instead. `git archive` writes the commit and its date into `VERSION`
    (`export-subst` in `.gitattributes`); an unsubstituted `$Format:...$` placeholder counts as no VERSION file."""
    if (repo / ".git").exists():                  # a directory, or a file in a linked worktree
        try:
            return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                                  text=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    version = repo / VERSION_FILE
    if version.is_file() and not version.read_text().startswith("$Format:"):
        return "release archive: " + " ".join(version.read_text().split())
    return "unknown (no git checkout, no VERSION file)"


def unit_id(clip_id: str) -> str:
    """The cost ledger's shipped unit-id form (`u-` + 16 hex); the salt is public because the clip is."""
    return "u-" + hashlib.sha256(("example\x00" + clip_id).encode()).hexdigest()[:16]


def pins() -> dict[str, Any]:
    """The model pins this run used: CoTracker3's hub ref (as set in this process) and `fastdo.model_pins`."""
    from fastdo import model_pins

    return {COTRACKER_REF_ENV: os.environ.get(COTRACKER_REF_ENV), "hf_revisions": dict(model_pins.HF_REVISIONS),
            "dinov2_hub_commit": model_pins.DINOV2_HUB_COMMIT}


def run_window(cfg: Any, a: argparse.Namespace, clip: Any, out: Path, mask: np.ndarray, keyframe_row: int
               ) -> tuple[Any, dict[str, Any]]:
    """Runs the engine on the one clip through the batched runner; returns `(manifest, execution provenance)`. Under
    --stub the placeholder StageFns go straight to the runner (`build_batched_stages` would build real workers)."""
    from fastdo.bench.bench_batched import (
        batched_engine_config,
        build_batched_stages,
        execution_provenance,
    )
    from fastdo.engine import ClipManifest, EngineConfig, run_manifest, write_mask_prompt
    from fastdo.engine.batched_runner import run_manifest_batched
    from fastdo.engine.batched_stages import HandsBatchStage
    from fastdo.engine.source_domain import write_domain_record

    artifact_root = out / "engine"
    clip_dir = artifact_root / clip.clip_id
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    if a.stub:
        fns: dict[str, Any] = stub_stage_fns(a.stages)
    else:
        from fastdo.bench.bench_stages import build_stage_fns

        fns = build_stage_fns(cfg, REPO, [clip])[0]
    missing = [s for s in a.stages if s not in fns]
    if missing:
        raise ValueError(f"no worker is registered for {missing}")
    write_domain_record(clip_dir, clip.clip_id, clip.source, clip.domain)
    write_mask_prompt(clip_dir, keyframe_row, mask)
    manifest_path = out / MANIFEST_FILE
    manifest = ClipManifest.create([clip.clip_id], [clip.manifest_source], [clip.n_frames])
    manifest.save(manifest_path)
    engine_cfg = EngineConfig(artifact_root=artifact_root, stages=a.stages, system="fastdo_example",
                              manifest_path=manifest_path)
    selected = {s: fns[s] for s in a.stages}
    if not bool(cfg.bench.batched.enabled):          # only through --set bench.batched.enabled=false
        run_manifest(manifest, engine_cfg, selected)
        return manifest, execution_provenance(manifest, None)
    batched_cfg = batched_engine_config(cfg.bench, engine_cfg)
    mapping = selected if a.stub else build_batched_stages(cfg, selected, artifact_root / ".hands_jobs",
                                                           artifact_root / ".gpu.lock")
    try:
        run = run_manifest_batched(manifest, batched_cfg, mapping)
    finally:
        hands = mapping.get("hands")
        if isinstance(hands, HandsBatchStage):
            hands.close()
    return manifest, execution_provenance(manifest, run)


def attribution_text(spec: Mapping[str, Any], matches_pin: bool = True, input_name: str | None = None) -> str:
    """The `ATTRIBUTION.txt` text. The dataset credit is written only when the input is the pinned clip
    (`matches_pin`); any other file (`input_name`, e.g. a `--stub --video` placeholder, or the video of a `--spec`,
    which names no dataset) gets no credit line."""
    ds = spec.get("dataset")
    if ds is None:
        credit = (f"Input: {input_name or 'a file you supplied'}, described by your own --spec file. No dataset credit "
                  "is recorded\nhere: its licence and attribution are yours to state.\n"
                  "The stage artifacts in this directory are derived from that file by this pipeline.\n")
    elif matches_pin:
        credit = (f"Input: {spec['file']} ({ds['name']}), by NVIDIA.\n"
                  f"Source: https://huggingface.co/datasets/{ds['hf_repo']} at revision {ds['revision']}.\n"
                  f"Licence: {ds['licence']} ({ds['licence_url']}).\n"
                  "Modified: reconstructed by this pipeline (the object mesh, the object pose per frame, the hand "
                  "meshes\nand the other stage artifacts in this directory are derived from the clip).\n")
    else:
        credit = (f"Input: {input_name or 'a file you supplied'}. It is not the pinned example clip (its sha256 "
                  "differs), so no\ndataset credit is recorded here: its licence and attribution are yours to state.\n"
                  "The stage artifacts in this directory are derived from that file by this pipeline.\n")
    return (credit
            + "The outputs are also produced by third-party models whose licences bind them (several are\n"
            "non-commercial, HaWoR's is NoDerivatives, MANO's forbids redistribution): read release/MODELS.md and\n"
            "THIRD_PARTY_LICENSES.md before you use or share them.\n")


def relative_to_out(value: Any, out: Path) -> Any:
    """`value` with every string that is an absolute path under `out` made relative to it (the batch log paths of the
    execution provenance), so the report names no host path, like `ledger_row`. Other values are returned as they are."""
    if isinstance(value, Mapping):
        return {k: relative_to_out(v, out) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [relative_to_out(v, out) for v in value]
    if isinstance(value, str) and os.path.isabs(value):
        path = Path(value)
        if path == out or out in path.parents:
            return path.relative_to(out).as_posix()
    return value


def build_records(a: argparse.Namespace, spec: Mapping[str, Any], manifest: Any, execution: Mapping[str, Any],
                  stride: int, video: Path, digest: str, verified: bool, out: Path, ledger_line: int
                  ) -> tuple[dict, dict]:
    """`(report, ledger row)` of a finished run; `ledger_line` is the row's 1-based line in the ledger file."""
    from fastdo.bench.bench_clips import sample_frame_indices

    clip_id = spec["clip_id"]
    states = {s: manifest.status(clip_id, s).state for s in a.stages}
    seconds = {s: float(manifest.status(clip_id, s).seconds) for s in a.stages}
    errors = {s: manifest.status(clip_id, s).error for s in a.stages if manifest.status(clip_id, s).error}
    solved = states.get("solve") == "done"
    delivered: bool | None = None                    # qc did not run: not decided
    if "qc" in states:
        delivered = solved and states["qc"] == "done"
        if delivered and not a.stub:
            delivered = bool(json.loads((out / "engine" / clip_id / "qc.json").read_text())["accepted"])
    failed = next((s for s in a.stages if states[s] != "done"), None)
    clip_dir = out / "engine" / clip_id
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    execution_kind = "pipelined" if execution.get("execution") == "batched" else "serial"
    start, stop = spec["window"]
    decoded = len(sample_frame_indices(stop - start, stride))
    row = {
        "schema": LEDGER_SCHEMA, "unit_id": unit_id(clip_id), "method": "fastdo", "corpus": spec["source"],
        "assigned": True, "run": True, "analysed": solved, "delivered": delivered,
        "exclusion_reason": None if solved else f"stage {failed}: {states.get(failed)}",
        "seconds": sum(seconds.values()), "frames": float(stop - start), "cluster": None,
        "metrics": {},
        "timing_basis": {"measurement": "modelled" if a.stub else "measured", "card": a.card,
                         "attribution": "stage_sum", "cache": "cold", "execution": execution_kind,
                         "cpu_priority": f"nice {os.nice(0)}", "stages": list(a.stages), "source_rate_carry": None,
                         "hardware": gpu_info()["gpus"], "frames_basis": "source", "frames_decoded": decoded,
                         "stride": stride,
                         "delivered_basis": "qc accepted" if "qc" in states else "qc not run: delivered is None",
                         "note": "stub run: placeholder workers, not a timing" if a.stub else
                                 "one window, cold start (model loads included), keyframe prompt untimed"},
        "stage_seconds": seconds, "clip_id": clip_id, "input_sha256": digest, "commit": git_commit(),
        "created_utc": now,
    }
    report = {
        "schema": REPORT_SCHEMA, "clip_id": clip_id, "stub": a.stub,
        "input": {"file": spec.get("file", video.name), "path": video.name, "sha256": digest,
                  "sha256_matches_pin": verified, "own_spec": a.spec.name if a.spec is not None else None,
                  "dataset": spec.get("dataset"), "object": spec["object"], "box_xyxy": spec["box_xyxy"],
                  "keyframe": spec["keyframe"]},
        "camera": dict(spec.get("camera", {})),
        "window": list(spec["window"]), "stride": stride, "stages": list(a.stages), "stage_states": states,
        "stage_seconds": seconds, "total_seconds": row["seconds"], "delivered": delivered, "errors": errors,
        "artifacts": sorted(p.name for p in clip_dir.iterdir() if p.is_file()),
        "execution": relative_to_out(execution, out),                    # batch log paths relative to --out
        "pins": pins(), "host": gpu_info(), "commit": row["commit"], "created_utc": now,
        "ledger_row": {"file": LEDGER_FILE, "line": ledger_line},        # relative to --out
        "attribution": ATTRIBUTION_FILE,
    }
    return report, row


def main(argv: Sequence[str] | None = None, fetch: Callable[[str, Path], None] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    try:
        spec = load_spec(a.spec, own=True) if a.spec is not None else load_spec()
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from None
    sys.path.insert(0, str(REPO / "src"))
    from fastdo.bench.bench_clips import BenchClip, keyframe_row, operating_stride
    from fastdo.engine.source_domain import parse_domain
    from fastdo.io.video import sample_frame_indices
    from fastdo.model_pins import COTRACKER_HUB_COMMIT

    os.environ.setdefault(COTRACKER_REF_ENV, COTRACKER_HUB_COMMIT)     # torch.hub reads the pinned co-tracker code
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # The default batched config declares stage budgets up to 13.4 GiB and SAM 3D needs 19, so the per-process cap must
    # be set (QUICKSTART: 32). The stub validates the same config under the same cap; it allocates nothing.
    if a.stub:
        os.environ.setdefault(GPU_MEM_ENV, STUB_GPU_MEM_GB)
        logger.info("--stub: every stage is a placeholder, so the engine's WARNING lines about prewarm hooks, the "
                    "memory ledger and CUDA that follow are expected here and do not mean the run failed")
    elif not os.environ.get(GPU_MEM_ENV):
        raise SystemExit(f"set {GPU_MEM_ENV} (release/QUICKSTART.md step 4 uses 32): the default 12 GiB cap is below "
                         "the default config's stage budgets and SAM 3D's 19 GiB")
    if not a.stub and "tracks" in a.stages:             # before any download, model load or budget change
        import torch.hub

        require_cotracker_hub(Path(torch.hub.get_dir()), os.environ[COTRACKER_REF_ENV])
    if not a.stub:                       # the benchmark's process budgets (FASTDO_CPU_THREADS, FASTDO_GPU_MEM_GB)
        from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget

        apply_cpu_budget()
        apply_gpu_budget()
    stride = operating_stride(float(spec["video"]["fps"]))
    try:                                 # before any download or model load: the mask frame must be a decoded frame
        w0, w1 = (int(x) for x in spec["window"])
        require_decoded_keyframe(w0, sample_frame_indices(w1 - w0, stride), stride, int(spec["keyframe"]))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    cfg = compose_config(run_overrides(a, spec, stride))
    if not a.stub:                       # before any download: the real stages' host paths (FASTDO_* variables)
        from fastdo.hostpaths import HostPathNotSet, check_stage_paths

        try:
            check_stage_paths(cfg, a.stages)
        except HostPathNotSet as e:
            raise SystemExit(str(e)) from None
        gated = missing_gated_inputs(a.stages, Path(str(cfg.hands.hawor_root)))   # before any GPU work or download
        if gated:
            raise SystemExit("the GPU example cannot run yet:\n  " + "\n  ".join(gated))
    try:
        video, digest, verified = resolve_input(a, spec, fetch)
    except (FileNotFoundError, ValueError, RuntimeError) as e:      # missing clip, sha256 or size mismatch, leftover
        raise SystemExit(str(e)) from None
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start, stop = spec["window"]
    clip = BenchClip(spec["clip_id"], spec["source"], "video", video.resolve(), stop - start, window=(start, stop),
                     decode_stride=stride, domain=parse_domain(spec["domain"]))
    row_k = keyframe_row(stop - start, stride, spec["keyframe"] - start)
    hw = (int(spec["video"]["height"]), int(spec["video"]["width"]))
    if a.stub:
        mask = stub_box_mask(hw, spec["box_xyxy"])
    else:
        frame = read_frame(video, int(spec["keyframe"]), hw[1], hw[0])
        mask = box_mask(frame, spec["box_xyxy"], device=str(cfg.device))
    manifest, execution = run_window(cfg, a, clip, out, mask, row_k)
    ledger = out / LEDGER_FILE
    line = (len(ledger.read_text().splitlines()) if ledger.is_file() else 0) + 1
    report, row = build_records(a, spec, manifest, execution, stride, video, digest, verified, out, line)
    (out / ATTRIBUTION_FILE).write_text(attribution_text(spec, verified, video.name))
    (out / REPORT_FILE).write_text(json.dumps(report, indent=1) + "\n")
    with open(ledger, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    timing = ("--stub: placeholder workers, %.1f s is not a timing" if a.stub else
              "reconstruction time on this machine %.1f s (sum of the stage seconds, one cold window, model loading "
              "included; viewer and render time are not part of it, and no ratio to another method is computed)")
    logger.info("%s: stages %s; " + timing + "; delivered %s; report %s", spec["clip_id"], report["stage_states"],
                report["total_seconds"], report["delivered"], out / REPORT_FILE)
    return 0 if all(s == "done" for s in report["stage_states"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
