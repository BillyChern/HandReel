"""SAM 2.1 video mask propagation from a single keyframe mask.

Uses the SAM 2.1 hiera-large video predictor (``facebook/sam2.1-hiera-large`` on Hugging Face,
ungated; weights cache under ``$HF_HOME``) prompted with a mask on one keyframe, then propagated to every other frame.
Needs the ``frontend`` extra (``uv sync --extra dev --extra frontend``): ``sam2`` and its
transitive ``Pillow`` dependency are not part of the base install, so both are imported lazily
(inside ``_load_predictor``/``_write_jpeg_frames``) rather than at module scope -- this lets
``fastdo.shape`` (and anything that imports it) still import cleanly with only ``--extra dev``;
only actually *calling* ``propagate_masks_sam2`` with the real ``sam2`` model needs the extra.

``sam2``'s video predictor only accepts a video *path* (an mp4 file or a directory of
``<index>.jpg`` frames), not an in-memory tensor, so ``propagate_masks_sam2`` writes ``frames`` to
a temporary directory of JPEGs and points ``init_state`` at it. ``propagate_in_video`` only walks
forward from its ``start_frame_idx``, so both a forward and a ``reverse=True`` pass are run to
cover the whole clip from a keyframe that may sit in the middle (verified end to end against a
tiny synthetic clip and mask: both passes' frame indices union to the full frame range; a
regression test for this union/ordering against a duck-typed fake predictor, injected
via the new ``predictor`` parameter -- no GPU/weights/network needed for that test).

Verified 2026-09-12: importing ``sam2`` prints one ``UserWarning`` ("cannot import name '_C' from
'sam2'") because the optional compiled CUDA post-processing extension was not built from this
pip/git install; sam2's own message says results are still usable in most cases with only some
post-processing (hole filling in mask scores) limited -- this adapter does not attempt to build
that extension.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fastdo.model_pins import hf_revision
from fastdo.shape.sam2_compile import VOS_COMPILED, check_mask_compile, compile_mode_patch

logger = logging.getLogger(__name__)

__all__ = ["MODEL_ID", "load_sam2_image_predictor", "load_sam2_predictor", "propagate_masks_sam2"]

MODEL_ID = "facebook/sam2.1-hiera-large"
_MODEL_ID = MODEL_ID
# Guards the swap of Hydra's process-wide GlobalHydra in `_under_sam2_hydra`: without it, concurrent loads
# (one per `fastdo.engine.batching.run_clips_concurrently` slot, each in its own pool thread) clear and
# re-initialize the singleton under each other ("GlobalHydra is already initialized").
_HYDRA_LOCK = threading.RLock()   # re-entrant: `_import_sam2` may run inside `_under_sam2_hydra`
# Threads for the JPEG round-trip SAM 2's video predictor requires; PIL releases the GIL in both directions.
_JPEG_WORKERS = 8
_CUDA = torch.device("cuda")      # sam2's own `load_video_frames` default for `compute_device`


def propagate_masks_sam2(frames: Tensor, keyframe_index: int, keyframe_mask: Tensor, device: str = "cuda",
                         predictor: Any = None) -> Tensor:
    """Propagates a keyframe mask to every frame with the SAM 2.1 video predictor.

    Args:
        frames: ``(N,H,W,3)`` uint8 RGB frames.
        keyframe_index: Index into ``frames`` carrying the prompt mask.
        keyframe_mask: ``(H,W)`` boolean (or 0/1) mask at ``keyframe_index``.
        device: torch device string for the predictor (unused when ``predictor`` is given).
        predictor: A ``sam2`` ``SAM2VideoPredictor``-compatible object (``init_state``,
            ``add_new_mask``, ``propagate_in_video``). ``None`` (the default) loads the real
            SAM 2.1 hiera-large predictor via ``_load_predictor``; tests inject a fake here to
            exercise the forward/reverse union without a GPU, weights, or network access.

    Returns:
        ``(N,H,W)`` bool mask per frame.

    Raises:
        ValueError: If ``frames``/``keyframe_mask`` have the wrong shape, or ``keyframe_index``
            is out of range.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {tuple(frames.shape)}")
    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    if not 0 <= keyframe_index < n:
        raise ValueError(f"keyframe_index {keyframe_index} out of range [0,{n})")
    if tuple(keyframe_mask.shape) != (h, w):
        raise ValueError(f"keyframe_mask must be ({h},{w}), got {tuple(keyframe_mask.shape)}")

    if predictor is None:
        predictor = _load_predictor(device)
    frames_np = frames.detach().cpu().numpy()
    mask_np = keyframe_mask.detach().cpu().numpy().astype(bool)

    out = np.zeros((n, h, w), dtype=bool)
    out[keyframe_index] = mask_np
    with tempfile.TemporaryDirectory(prefix="fastdo_sam2_frames_") as tmp_dir:
        _write_jpeg_frames(frames_np, Path(tmp_dir))
        _install_parallel_jpeg_loader()
        state = predictor.init_state(video_path=tmp_dir)
        predictor.add_new_mask(state, frame_idx=keyframe_index, obj_id=1, mask=mask_np)

        for fidx, _obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=keyframe_index):
            out[fidx] = (logits[0, 0] > 0.0).detach().cpu().numpy()
        for fidx, _obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=keyframe_index,
                                                                    reverse=True):
            out[fidx] = (logits[0, 0] > 0.0).detach().cpu().numpy()

    return torch.from_numpy(out)


def _load_predictor(device: str) -> Any:
    """The default SAM 2.1 hiera-large predictor: ``load_sam2_predictor(device)``."""
    return load_sam2_predictor(device)


def load_sam2_predictor(device: str = "cuda", model_id: str = MODEL_ID, mask_compile: str = "none") -> Any:
    """Returns a ``sam2`` ``SAM2VideoPredictor`` -- untyped since ``sam2`` ships no type stubs.

    Thread-safe: concurrent calls are serialized, since each swaps Hydra's process-wide global state (and, in a
    vos mode, ``torch.compile`` for the duration of the build).

    Args:
        device: torch device string for the predictor.
        model_id: Hugging Face checkpoint, e.g. one of ``fastdo.shape.sam2_variants.SAM2_VARIANTS``.
        mask_compile: ``bench.mask_compile`` (``fastdo.shape.sam2_compile``). ``none`` is the eager predictor, built
            exactly as before; ``vos_nocg``/``vos_cg_clone`` build upstream's ``SAM2VideoPredictorVOS``
            (``vos_optimized=True``) with its ``torch.compile`` calls rewritten by ``compile_mode_patch``.

    Raises:
        RuntimeError: If ``sam2`` is not importable, with an actionable install hint (rather than
            letting a bare ``ModuleNotFoundError`` surface); or if a vos build compiled other forwards than
            ``VOS_COMPILED`` (the knob did not reach what it is for).
        ValueError: On an unknown ``mask_compile``.
    """
    mask_compile = check_mask_compile(mask_compile)
    build_hf = _hf_builder(model_id, "build_sam2_video_predictor")
    if mask_compile == "none":
        return _under_sam2_hydra(lambda: build_hf(device=device))

    def build_vos() -> Any:
        with compile_mode_patch(mask_compile) as rewritten:
            predictor = build_hf(device=device, vos_optimized=True)
        if set(rewritten) != VOS_COMPILED or len(rewritten) != len(VOS_COMPILED):
            raise RuntimeError(f"mask_compile {mask_compile!r} rewrote {sorted(rewritten)}, not the "
                               f"{len(VOS_COMPILED)} forwards {sorted(VOS_COMPILED)}: sam2's VOS build changed")
        return predictor

    return _under_sam2_hydra(build_vos)


def load_sam2_image_predictor(device: str = "cuda", model_id: str = MODEL_ID) -> Any:
    """Returns a ``sam2`` ``SAM2ImagePredictor``, under the same Hydra guard as the video predictor.

    `SAM2ImagePredictor.from_pretrained` composes its config the same way `build_sam2_video_predictor_hf`
    does, so calling it directly inside a ``@hydra.main`` app fails identically ("Cannot find primary
    config 'configs/sam2.1/sam2.1_hiera_l.yaml'"). The `prompt` stage hit exactly that end to end.

    Args:
        device: torch device string for the predictor.
        model_id: Hugging Face checkpoint, e.g. one of ``fastdo.shape.sam2_variants.SAM2_VARIANTS``.

    Raises:
        RuntimeError: If ``sam2`` is not importable, with the same install hint.
    """
    predictor_cls = _sam2_symbol("sam2.sam2_image_predictor", "SAM2ImagePredictor")
    if hf_revision(model_id) is None:
        return _under_sam2_hydra(lambda: predictor_cls.from_pretrained(model_id, device=device))
    build = _hf_builder(model_id, "build_sam2")       # from_pretrained's own steps, at the pinned revision
    return _under_sam2_hydra(lambda: predictor_cls(build(device=device), device=device))


def _hf_builder(model_id: str, builder: str) -> Callable[..., Any]:
    """`sam2.build_sam.<builder>_hf` bound to `model_id`, with the checkpoint read at the commit `fastdo.model_pins`
    pins for it: sam2's own `_hf_download` reads `main`, which a download at a commit (scripts/fetch_weights.py) never
    writes, and which may have moved. An unpinned `model_id` keeps sam2's `<builder>_hf` call exactly."""
    revision = hf_revision(model_id)
    if revision is None:
        build_hf = _sam2_symbol("sam2.build_sam", f"{builder}_hf")
        return lambda **kwargs: build_hf(model_id, **kwargs)
    build = _sam2_symbol("sam2.build_sam", builder)

    def build_pinned(**kwargs: Any) -> Any:
        config_name, ckpt_path = _pinned_hf_download(model_id, revision)
        return build(config_file=config_name, ckpt_path=ckpt_path, **kwargs)
    return build_pinned


def _pinned_hf_download(model_id: str, revision: str) -> tuple[str, str]:
    """sam2's `_hf_download(model_id)` (model config name, local checkpoint path) at `revision`."""
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = _sam2_symbol("sam2.build_sam", "HF_MODEL_ID_TO_FILENAMES")[model_id]
    return config_name, hf_hub_download(repo_id=model_id, filename=checkpoint_name, revision=revision)


def _import_sam2(module: str) -> Any:
    """``importlib.import_module(module)`` for a ``sam2`` module, leaving Hydra's global state as it found it.

    ``sam2/__init__`` initialises Hydra's process-wide config module when nothing owns it yet, so a bare first
    import leaves ``GlobalHydra`` initialised for everything after it ("GlobalHydra is already initialized" in every
    later ``hydra.initialize``; 69 tests broke when a test imported sam2 directly). Pipeline scripts are Hydra apps,
    so there sam2 skips the call; either way the caller's state is restored here.
    """
    if module in sys.modules:
        return sys.modules[module]
    from hydra.initialize import get_gh_backup

    with _HYDRA_LOCK:
        backup = get_gh_backup()
        try:
            return importlib.import_module(module)
        finally:
            _restore_global_hydra(backup)


def _restore_global_hydra(backup: Any) -> None:
    """``hydra.initialize.restore_gh_from_backup(backup)``, except that restoring "no GlobalHydra" never deletes a
    singleton that is not there.

    `backup` is what ``get_gh_backup()`` returned before the guarded step. The three cases:

    - GlobalHydra absent before (backup None) and absent after: nothing to undo, so nothing is deleted. Hydra's
      helper deletes the singleton unconditionally and raises ``KeyError`` here. That is the second
      ``_import_sam2`` of a caller that is not a Hydra app (``release/example/run_example.py``): the first import ran
      ``sam2/__init__``, which created GlobalHydra, and removed it again; a later sam2 submodule import never runs
      ``sam2/__init__``.
    - absent before, created by the guarded step (``sam2/__init__``, or ``GlobalHydra.instance()``): it is removed.
    - present before (backup a copy of it): that copy is put back, exactly as Hydra's helper does it.
    """
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.singleton import Singleton
    from hydra.initialize import restore_gh_from_backup

    if backup is None:
        Singleton._instances.pop(GlobalHydra, None)
    else:
        restore_gh_from_backup(backup)


def _sam2_symbol(module: str, name: str) -> Any:
    """Imports one ``sam2`` symbol, turning a bare ``ModuleNotFoundError`` into an actionable hint."""
    try:
        return getattr(_import_sam2(module), name)
    except ImportError as exc:
        raise RuntimeError(
            "sam2 is not installed. It ships in this project's optional 'frontend' extra: run "
            "`uv sync --extra dev --extra frontend`."
        ) from exc


def _under_sam2_hydra(build: Callable[[], Any]) -> Any:
    """Runs ``build()`` with sam2's Hydra config search path swapped in, then restores the caller's.

    sam2 composes its model config through Hydra's *process-wide* config search path, which it points at
    its own package on import -- unless a Hydra app (every ``run/pipeline`` script is ``@hydra.main``)
    already owns that global state, in which case the lookup fails with "Cannot find primary config
    'configs/sam2.1/...'". The whole swap holds ``_HYDRA_LOCK``, so loads from several threads run one at
    a time, and the caller's state is restored even when ``build`` raises.
    """
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import get_gh_backup

    with _HYDRA_LOCK:
        backup = get_gh_backup()
        GlobalHydra.instance().clear()
        try:
            with initialize_config_module("sam2", version_base="1.2"):
                return build()
        finally:
            _restore_global_hydra(backup)


def _write_jpeg_frames(frames: np.ndarray, out_dir: Path) -> None:
    from PIL import Image

    def write(i: int) -> None:
        Image.fromarray(frames[i]).save(out_dir / f"{i:05d}.jpg", quality=95)

    with ThreadPoolExecutor(_JPEG_WORKERS) as pool:     # PIL's encoder releases the GIL
        list(pool.map(write, range(frames.shape[0])))


def _load_jpeg_folder_parallel(video_path: str, image_size: int, offload_video_to_cpu: bool, img_mean: Any,
                               img_std: Any, compute_device: Any) -> tuple[Tensor, int, int]:
    """`sam2.utils.misc.load_video_frames_from_jpg_images` (non-async), each frame on a thread pool.

    The upstream loop decodes and resizes one frame at a time -- ~5 s of a 100-frame 1080x1920 window. This
    is the same listing, the same per-frame `_load_img_as_tensor` and the same normalisation, so the tensor is
    bit-identical (`test_parallel_jpeg_loader_is_bit_identical`); only the frames are loaded concurrently.
    """
    load_img = _sam2_symbol("sam2.utils.misc", "_load_img_as_tensor")
    names = [p for p in os.listdir(video_path) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    if not names:
        raise RuntimeError(f"no images found in {video_path}")
    mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    images = torch.zeros(len(names), 3, image_size, image_size, dtype=torch.float32)
    with ThreadPoolExecutor(_JPEG_WORKERS) as pool:
        loaded = list(pool.map(lambda name: load_img(os.path.join(video_path, name), image_size), names))
    for n, (img, _, _) in enumerate(loaded):
        images[n] = img
    video_height, video_width = loaded[-1][1], loaded[-1][2]
    if not offload_video_to_cpu:
        images, mean, std = images.to(compute_device), mean.to(compute_device), std.to(compute_device)
    images -= mean
    images /= std
    return images, video_height, video_width


def _install_parallel_jpeg_loader() -> None:
    """Routes the video predictor's synchronous JPEG-folder loads through `_load_jpeg_folder_parallel`; idempotent.

    Every other call (mp4 or bytes input, `async_loading_frames=True`) goes to sam2's own `load_video_frames`.
    """
    module = _import_sam2("sam2.sam2_video_predictor")
    original = module.load_video_frames
    if getattr(original, "_fastdo_parallel", False):
        return

    def load_video_frames(video_path: Any, image_size: int, offload_video_to_cpu: bool,
                          img_mean: Any = (0.485, 0.456, 0.406), img_std: Any = (0.229, 0.224, 0.225),
                          async_loading_frames: bool = False, compute_device: Any = _CUDA) -> Any:
        if isinstance(video_path, str) and os.path.isdir(video_path) and not async_loading_frames:
            return _load_jpeg_folder_parallel(video_path, image_size, offload_video_to_cpu, img_mean, img_std,
                                              compute_device)
        return original(video_path, image_size, offload_video_to_cpu, img_mean, img_std, async_loading_frames,
                        compute_device)

    load_video_frames._fastdo_parallel = True  # type: ignore[attr-defined]
    module.load_video_frames = load_video_frames
