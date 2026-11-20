"""Detector side of the prompt stage: runs Hands23 in ITS OWN interpreter and returns boxes and scores.

This file is executed by the Hands23 env's interpreter (`$FASTDO_HANDS23_ROOT/env_hands23/bin/python`), which has
detectron2 and the Hands23 repository; it must therefore import NOTHING from `fastdo`. The one exception is `fastdo/engine/npz_rows.py`, which imports only
the standard library and numpy and is loaded BY PATH (the `fastdo` package is never imported): it reads the candidate
rows of `decode.npz` without materialising the whole 0.6-1.5 GB `frames` array.
`fastdo.prompt.hands23_client` drives it in one of two modes:

    python hands23_detect.py --serve --weights model_hands23.pth --device cuda   # WARM: the model is built ONCE
    python hands23_detect.py --request request.json --reply reply.json           # one-shot, for debugging

**Warm mode is what the engine uses, and it is the difference between 7.5 s and 2 s per window.** Building the
model and loading its weights costs ~5.5 s and a one-shot process pays that for EVERY window; served, it is paid
once per run. The loop reads one request-file path per line of stdin, writes the reply to the path named inside
that request, and acknowledges with one JSON line on stdout. A bad request is answered with an error line, not by
dying, so one unusable window does not throw away the warm model.

Request: `{"decode_npz": path, "rows": [int, ...], "weights": path, "device": "cuda"}`. In warm mode `weights` must
match the weights the server was started with; a mismatch is refused rather than silently answered by the wrong
model. Two optional keys change HOW the rows are computed, never which rows or at what resolution; absent (the
client sends neither at its defaults) is the original one-row loop, byte for byte:

* `"prefetch": P` (`prompt_stage.detect_prefetch`) prepares rows on P CPU threads ahead of the GPU. The BGR flip and
  the resize are ~50-100 ms of every ~150-200 ms row, and PIL's resize and numpy's copies release the GIL, so they
  overlap the previous row's forward. Data movement only: the model sees the same tensors in the same calls.
* `"batch": B` (`prompt_stage.detect_batch`) runs the normalise/pad, backbone and RPN on B rows per call.

**Batching cannot go through `model([...])`.** Hands23's `hoRCNNROIHeads._inference_z` reads only
`pred_instances[0]` and pools every box from image 0 of the feature maps, so a list of B images through the
model's own `forward` silently returns image 0's hands and drops the rest. `_forward_batch` therefore replays
detectron2's `GeneralizedRCNN.inference` with the batch-safe parts batched (normalise/pad, backbone, RPN) and
the ROI heads and postprocess called once per image on that image's slice of the features. Every frame of a
window has one size, so the padded batch tensor is the single-image tensor stacked; what moves is the float-level
result of a batched convolution. Measured (72 Ego4D windows, batch 8): every decision and chosen row held,
chosen boxes moved <= 0.12 px, but 31/56 SAM 2 masks changed (IoU >= 0.9989) and 10 windows flipped a NON-chosen
record's object link (a threshold or argmax crossing), so batching is not an exact lever; prefetch is.

Reply: `{"rows": {"<row>": [record, ...]}, "meta": {...}}` with, per hand record,
`hand_score`, `side`, `contact_idx`, `interaction_score`, and, when the head links one, `obj_box`, `obj_score`,
`p_held_like` (the mass on the held-like touch classes, which is what separates a carried object from a surface).

**No pixels leave this process**: it opens `decode.npz` itself and returns numbers only.

Determinism comes from eval mode, a fixed seed and cuDNN's deterministic kernels, NEVER from
`torch.use_deterministic_algorithms`, which the gate lane measured at 3.8 s against 66 ms (58x) for this model.

Licence: Hands23 code is MIT. Its training dataset is CC BY-NC 4.0. The public checkpoint instructions do not give a
separate licence; the project treats the optional detector as non-commercial. detectron2 is Apache-2.0.
Cite Cheng, Shan, Hassen, Higgins, Fouhey, NeurIPS 2023.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# This file runs in the Hands23 env without `fastdo` importable, so it repeats `fastdo.hostpaths.hands23_root` inline,
# including its refusal of a set-but-empty variable (tests/test_hostpaths.py checks that the two agree).


def _env_dir(var: str, default: str) -> str:
    """`$var` when set, else `default`; a set-but-empty variable raises (as `fastdo.hostpaths.env_path`)."""
    v = os.environ.get(var)
    if v is None:
        return default
    if not v:
        raise ValueError(f"{var} is set but empty; unset it to use the default {default}")
    return v


GATE_REFIT = Path(_env_dir("FASTDO_HANDS23_ROOT", os.path.join(_env_dir("FASTDO_SCRATCH_ROOT", os.path.join(
    _env_dir("FASTDO_PRIMARY_ROOT", os.path.join(_env_dir("FASTDO_EXT_ROOT", "/opt/fastdo"), "handreel")),
    "temp")), "hands23")))
HANDS23_REPO = GATE_REFIT / "src/hands23_detector"
CONFIG_YAML = HANDS23_REPO / "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml"
TOUCH_HELD_LIKE = (1, 2, 4, 6)          # tool-held, tool-used, container-held, neither-held (probe_h23.py's HELD_LIKE)
THRESHOLDS = {"HAND": 0.7, "FIRSTOBJ": 0.5, "SECONDOBJ": 0.3, "HAND_RELA": 0.3, "OBJ_RELA": 0.7}   # repo demo values
CONTACT_OBJECT = 3                       # "object_contact" in the training-order mapping
NPZ_ROWS = Path(__file__).resolve().parents[1] / "engine/npz_rows.py"   # stdlib + numpy only (module docstring)


def _load_npz_rows():
    """`fastdo/engine/npz_rows.py` as a standalone module, without importing the `fastdo` package."""
    spec = importlib.util.spec_from_file_location("_fastdo_npz_rows", NPZ_ROWS)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {NPZ_ROWS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


npz_rows = _load_npz_rows()


def build(weights: Path, device: str):
    sys.path.insert(0, str(HANDS23_REPO))
    from detectron2.config import get_cfg
    from detectron2.data import transforms as T
    from detectron2.modeling import build_model
    import hodetector.modeling.roi_heads  # noqa: F401  (registers the Hands23 ROI heads)

    torch.backends.cudnn.deterministic = True     # determinism from eval mode, a seed and cuDNN's kernels only:
    torch.backends.cudnn.benchmark = False        # torch.use_deterministic_algorithms costs 58x on this model
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG_YAML))
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.DEVICE = device
    for key, value in THRESHOLDS.items():
        setattr(cfg, key, value)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(THRESHOLDS["HAND"], THRESHOLDS["FIRSTOBJ"], THRESHOLDS["SECONDOBJ"])
    model = build_model(cfg)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    result = model.load_state_dict(state["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"weights do not match the config: missing={result.missing_keys[:3]} "
                         f"unexpected={result.unexpected_keys[:3]}")
    model.eval()
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    return model, aug


def records(instances) -> list[dict]:
    """The fields the prompt stage scores on, from one frame's detections."""
    boxes = instances.pred_boxes.tensor.detach().float().cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.detach().float().cpu().numpy()
    dz = instances.pred_dz.detach().float().cpu().numpy()
    out = []
    for i in range(len(classes)):
        if classes[i] != 0:                       # hands only; objects arrive through the link below
            continue
        rec = {"hand_score": round(float(scores[i]), 5),
               "side": "right_hand" if dz[i, 5] == 1 else "left_hand",
               "contact_idx": int(dz[i, 8]),
               "interaction_score": round(float(dz[i, 9]), 5)}
        rec["side"] = "right" if rec["side"] == "right_hand" else "left"
        link = int(dz[i, 4])
        if link >= 0:
            touch = dz[link, 18:25]
            rec.update(obj_box=[round(float(v), 2) for v in boxes[link]],
                       obj_score=round(float(scores[link]), 5),
                       p_held_like=round(float(sum(touch[j] for j in TOUCH_HELD_LIKE)), 5))
        out.append(rec)
    return out


def sha256(path: Path) -> str:
    """The weights' digest, for provenance. Warm mode computes it once, not per window (0.43 s for 446 MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def handle(request: dict, model, aug, weights: Path, device: str, weights_sha: str) -> dict:
    """One window's detections. `records()` per candidate row, plus the provenance the client records.

    Raises:
        ValueError: If the request names weights other than the ones this process loaded.
    """
    asked = request.get("weights")
    if asked is not None and Path(str(asked)).resolve() != weights.resolve():
        raise ValueError(f"this detector serves {weights}, but the request asks for {asked}")
    rows = [int(r) for r in request["rows"]]
    batch, prefetch = int(request.get("batch", 1)), int(request.get("prefetch", 0))
    if batch < 1 or prefetch < 0:
        raise ValueError(f"batch must be >= 1 and prefetch >= 0, got batch {batch}, prefetch {prefetch}")
    frames = npz_rows.read_npz_rows(Path(request["decode_npz"]), "frames", rows)   # only these rows are read

    torch.manual_seed(0)                                   # per request: a served window matches a one-shot window
    started = time.time()
    per_row: dict[str, list[dict]] = {}
    peak = 0.0
    if batch > 1 or prefetch > 0:
        per_row, peak = _detect_fast(rows, frames, model, aug, device, batch, prefetch)
        return {"rows": per_row,
                "meta": {"weights_sha256": weights_sha, "config": str(CONFIG_YAML), "thresholds": THRESHOLDS,
                         "device": device, "torch": torch.__version__, "seconds": round(time.time() - started, 3),
                         "peak_gpu_gib_reserved": round(peak, 3), "n_rows": len(rows), "batch": batch,
                         "prefetch": prefetch}}
    for row, frame in zip(rows, frames):
        image = np.ascontiguousarray(frame[:, :, ::-1])     # detectron2's models take BGR
        h, w = image.shape[:2]
        x = aug.get_transform(image).apply_image(image)
        x = torch.as_tensor(x.astype("float32").transpose(2, 0, 1))
        if device.startswith("cuda"):
            x = x.cuda(non_blocking=False)
        with torch.no_grad():
            instances = model([{"image": x, "height": h, "width": w}])[0]["instances"]
        per_row[str(row)] = records(instances)
        if device.startswith("cuda"):
            peak = max(peak, torch.cuda.max_memory_reserved() / 2**30)

    return {"rows": per_row,
            "meta": {"weights_sha256": weights_sha, "config": str(CONFIG_YAML), "thresholds": THRESHOLDS,
                     "device": device, "torch": torch.__version__, "seconds": round(time.time() - started, 3),
                     "peak_gpu_gib_reserved": round(peak, 3), "n_rows": len(rows)}}


def _prepare(frame: np.ndarray, aug) -> dict:
    """One row as detectron2 input: exactly the one-row loop's BGR flip, resize and float tensor (on the CPU)."""
    image = np.ascontiguousarray(frame[:, :, ::-1])         # detectron2's models take BGR
    h, w = image.shape[:2]
    x = aug.get_transform(image).apply_image(image)
    return {"image": torch.as_tensor(x.astype("float32").transpose(2, 0, 1)), "height": h, "width": w}


def _forward_batch(model, inputs: list[dict]) -> list:
    """`GeneralizedRCNN.inference` for several images: backbone and RPN batched, ROI heads per image.

    Hands23's ROI heads handle one image only (module docstring), so each image's heads see a one-image
    `ImageList` and batch-1 feature maps, exactly the shapes the one-row loop gives them.
    """
    from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
    from detectron2.structures import ImageList

    images = model.preprocess_image(inputs)
    features = model.backbone(images.tensor)
    proposals, _ = model.proposal_generator(images, features, None)
    out = []
    for i, one_input in enumerate(inputs):
        one = ImageList(images.tensor[i:i + 1], [images.image_sizes[i]])
        feats = {name: value[i:i + 1] for name, value in features.items()}
        results, _ = model.roi_heads(one, feats, [proposals[i]], None)
        out.append(GeneralizedRCNN._postprocess(results, [one_input], one.image_sizes)[0]["instances"])
    return out


def _prepared(frames: np.ndarray, aug, prefetch: int, lookahead: int) -> Iterator[dict]:
    """`_prepare` of every frame, in order. With `prefetch` threads, up to `lookahead` rows are prepared ahead."""
    if prefetch <= 0:
        for frame in frames:
            yield _prepare(frame, aug)
        return
    with ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="h23_prep") as pool:
        pending: deque = deque()
        todo = iter(frames)
        for frame in todo:
            pending.append(pool.submit(_prepare, frame, aug))
            if len(pending) >= lookahead:
                break
        while pending:
            ready = pending.popleft().result()
            for frame in todo:                       # refill one slot per row consumed
                pending.append(pool.submit(_prepare, frame, aug))
                break
            yield ready


def _detect_fast(rows: list[int], frames: np.ndarray, model, aug, device: str, batch: int,
                 prefetch: int) -> tuple[dict[str, list[dict]], float]:
    """`records()` per row, `batch` rows per forward, rows prepared on `prefetch` threads. Same keys and order as
    the one-row loop; at `batch` 1 the forward is the loop's own `model([...])` call."""
    per_row: dict[str, list[dict]] = {}
    peak = 0.0
    source = _prepared(frames, aug, prefetch, lookahead=batch + 2 * prefetch)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        inputs = [next(source) for _ in chunk]
        if device.startswith("cuda"):
            for one in inputs:
                one["image"] = one["image"].cuda(non_blocking=False)
        with torch.no_grad():
            if batch == 1:
                instances = [model(inputs)[0]["instances"]]
            else:
                instances = _forward_batch(model, inputs)
        if len(instances) != len(chunk):
            raise RuntimeError(f"the batched forward returned {len(instances)} results for {len(chunk)} rows")
        for row, inst in zip(chunk, instances):
            per_row[str(row)] = records(inst)
        if device.startswith("cuda"):
            peak = max(peak, torch.cuda.max_memory_reserved() / 2**30)
    return per_row, peak


def serve(weights: Path, device: str) -> None:
    """Warm loop: build once, then one request-file path per line of stdin, one JSON ack per line of stdout."""
    build_started = time.time()
    model, aug = build(weights, device)
    weights_sha = sha256(weights)
    ready = {"ready": True, "build_seconds": round(time.time() - build_started, 3), "weights_sha256": weights_sha}
    print(json.dumps(ready), flush=True)

    served = 0
    for line in sys.stdin:
        path = line.strip()
        if not path or path == "quit":
            return
        try:
            request = json.loads(Path(path).read_text())
            reply = handle(request, model, aug, weights, device, weights_sha)
            reply["meta"]["warm"] = True
            reply["meta"]["served"] = served = served + 1
            Path(request["reply"]).write_text(json.dumps(reply))
            print(json.dumps({"ok": True, "reply": request["reply"]}), flush=True)
        except Exception as exc:  # noqa: BLE001 -- one bad window must not cost the warm model
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="warm mode: build once, serve request paths from stdin")
    ap.add_argument("--weights", help="warm mode: the weights to serve")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--request", help="one-shot mode: the request JSON")
    ap.add_argument("--reply", help="one-shot mode: where to write the reply")
    args = ap.parse_args()

    if args.serve:
        if not args.weights:
            raise SystemExit("--serve needs --weights")
        serve(Path(args.weights), str(args.device))
        return

    if not args.request or not args.reply:
        raise SystemExit("one-shot mode needs --request and --reply")
    request = json.loads(Path(args.request).read_text())
    weights = Path(request["weights"])
    device = str(request.get("device", args.device))
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, aug = build(weights, device)
    reply = handle(request, model, aug, weights, device, sha256(weights))
    reply["meta"]["warm"] = False
    Path(args.reply).write_text(json.dumps(reply))


if __name__ == "__main__":
    main()
