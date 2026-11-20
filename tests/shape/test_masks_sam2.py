"""Tests for fastdo.shape.masks_sam2.

``propagate_masks_sam2`` itself needs the real SAM 2.1 video predictor (Hugging Face download)
and a GPU to be meaningful, so most of it is exercised by the real-demo smoke run; the input
validation here needs neither (checked before the model is ever loaded), and the forward/reverse
union logic is checked with a duck-typed fake predictor injected via the ``predictor`` parameter
-- still real frame-to-JPEG writing (cheap, no GPU), so that one test
needs Pillow (the ``frontend`` extra) and is skipped without it, matching the module's own
lazy-import contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fastdo.shape.masks_sam2 import propagate_masks_sam2


def test_wrong_frames_ndim_raises() -> None:
    frames = torch.zeros(4, 8, 8, dtype=torch.uint8)  # missing channel dim
    mask = torch.zeros(8, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="N,H,W,3"):
        propagate_masks_sam2(frames, keyframe_index=0, keyframe_mask=mask)


def test_keyframe_index_out_of_range_raises() -> None:
    frames = torch.zeros(4, 8, 8, 3, dtype=torch.uint8)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match="keyframe_index"):
        propagate_masks_sam2(frames, keyframe_index=4, keyframe_mask=mask)


def test_keyframe_mask_shape_mismatch_raises() -> None:
    frames = torch.zeros(4, 8, 8, 3, dtype=torch.uint8)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="keyframe_mask"):
        propagate_masks_sam2(frames, keyframe_index=0, keyframe_mask=mask)


# --- forward/reverse propagate_in_video union -----------------------------------------------


class _FakeSam2Predictor:
    """Duck-typed ``SAM2VideoPredictor`` stand-in: no GPU, no weights, no network.

    ``forward_masks``/``backward_masks`` map frame index -> ``(H,W)`` bool mask; each is yielded
    from ``propagate_in_video`` (respectively ``reverse=False``/``True``) as a synthetic logits
    tensor (``+10``/``-10`` so ``logits > 0`` reproduces the mask exactly).
    """

    def __init__(self, forward_masks: dict[int, np.ndarray], backward_masks: dict[int, np.ndarray]) -> None:
        self.forward_masks = forward_masks
        self.backward_masks = backward_masks
        self.init_state_calls: list[str] = []
        self.add_new_mask_calls: list[tuple[int, int]] = []

    def init_state(self, video_path: str) -> dict[str, str]:
        self.init_state_calls.append(video_path)
        return {"video_path": video_path}

    def add_new_mask(self, state: dict[str, str], frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        self.add_new_mask_calls.append((frame_idx, obj_id))

    def propagate_in_video(self, state: dict[str, str], start_frame_idx: int, reverse: bool = False):
        masks = self.backward_masks if reverse else self.forward_masks
        for fidx in sorted(masks, reverse=reverse):
            logits = torch.where(torch.from_numpy(masks[fidx]), torch.tensor(10.0), torch.tensor(-10.0))
            yield fidx, [1], logits[None, None]


def test_propagate_masks_sam2_unions_forward_and_reverse_passes() -> None:
    pytest.importorskip("PIL")  # _write_jpeg_frames still runs for real; needs the frontend extra

    n, h, w, keyframe_index = 6, 4, 4, 3
    rng = np.random.default_rng(0)

    def _mask() -> np.ndarray:
        return rng.random((h, w)) > 0.5

    keyframe_mask = _mask()
    # Deliberately exclude keyframe_index from both directions: `out[keyframe_index]` must come
    # from the pre-seed (the input keyframe_mask), not from either propagate_in_video pass.
    forward = {4: _mask(), 5: _mask()}
    backward = {2: _mask(), 1: _mask(), 0: _mask()}
    predictor = _FakeSam2Predictor(forward, backward)

    frames = torch.zeros(n, h, w, 3, dtype=torch.uint8)
    out = propagate_masks_sam2(frames, keyframe_index, torch.from_numpy(keyframe_mask), predictor=predictor)

    assert out.shape == (n, h, w)
    np.testing.assert_array_equal(out[0].numpy(), backward[0])
    np.testing.assert_array_equal(out[1].numpy(), backward[1])
    np.testing.assert_array_equal(out[2].numpy(), backward[2])
    np.testing.assert_array_equal(out[keyframe_index].numpy(), keyframe_mask)
    np.testing.assert_array_equal(out[4].numpy(), forward[4])
    np.testing.assert_array_equal(out[5].numpy(), forward[5])

    assert predictor.add_new_mask_calls == [(keyframe_index, 1)]
    assert len(predictor.init_state_calls) == 1


def test_load_predictor_composes_sam2_config_inside_a_running_hydra_app(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`run/pipeline` scripts call SAM 2 from inside `@hydra.main`, whose global config search path
    points at `run/conf`; `_load_predictor` must still compose sam2's own model config there, and
    hand the app its search path back afterwards."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import get_gh_backup, restore_gh_from_backup

    # Back up before sam2's first import: `import sam2` initializes GlobalHydra, and restoring that state would
    # leave it initialized for later tests that start their own Hydra app.
    session_backup = get_gh_backup()
    pytest.importorskip("sam2")
    from sam2 import build_sam

    from fastdo.shape import masks_sam2

    seen: dict[str, object] = {}

    # The default model is pinned (fastdo.model_pins), so the build reads the checkpoint at the pinned commit and then
    # calls sam2's non-hf builder: that is the seam faked here.
    def _fake_build(config_file: str, ckpt_path: str, device: str = "cuda") -> str:
        seen["model"] = compose(config_name="configs/sam2.1/sam2.1_hiera_l.yaml").model
        seen["config_file"], seen["ckpt_path"] = config_file, ckpt_path
        return "predictor"

    def _fake_download(model_id: str, revision: str) -> tuple[str, str]:
        seen["revision"] = revision
        return build_sam.HF_MODEL_ID_TO_FILENAMES[model_id][0], "ckpt.pt"

    monkeypatch.setattr(build_sam, "build_sam2_video_predictor", _fake_build)
    monkeypatch.setattr(masks_sam2, "_pinned_hf_download", _fake_download)
    (tmp_path / "app.yaml").write_text("app_key: 1\n")
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
            assert masks_sam2._load_predictor("cpu") == "predictor"
            assert seen["model"] is not None
            assert seen["revision"] == masks_sam2.hf_revision(masks_sam2.MODEL_ID) and seen["ckpt_path"] == "ckpt.pt"
            assert compose(config_name="app").app_key == 1
    finally:
        restore_gh_from_backup(session_backup)


def test_concurrent_predictor_loads_serialize_the_hydra_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loads from several threads (one per `run_clips_concurrently` slot) must not clear GlobalHydra under each
    other: builds run one at a time, each composes sam2's config, and the app's Hydra state comes back."""
    import threading
    import time

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import get_gh_backup, restore_gh_from_backup

    session_backup = get_gh_backup()
    pytest.importorskip("sam2")
    from sam2 import build_sam

    from fastdo.shape import masks_sam2

    n_threads = 3
    inside, max_inside, composed = [0], [0], list[str]()
    errors: list[str] = []
    guard = threading.Lock()
    barrier = threading.Barrier(n_threads, timeout=30)

    def _fake_build(model_id: str, device: str = "cuda") -> str:
        with guard:
            inside[0] += 1
            max_inside[0] = max(max_inside[0], inside[0])
        time.sleep(0.05)                                    # a real build downloads and instantiates here
        name = str(compose(config_name="configs/sam2.1/sam2.1_hiera_t.yaml").model._target_)
        time.sleep(0.05)
        with guard:
            inside[0] -= 1
            composed.append(name)
        return model_id

    def _load() -> None:
        barrier.wait()
        try:
            masks_sam2.load_sam2_predictor(device="cpu", model_id="facebook/sam2.1-hiera-tiny")
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(f"{type(exc).__name__}: {exc}")

    monkeypatch.setattr(build_sam, "build_sam2_video_predictor_hf", _fake_build)
    (tmp_path / "app.yaml").write_text("app_key: 1\n")
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
            threads = [threading.Thread(target=_load) for _ in range(n_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
            assert errors == []
            assert max_inside[0] == 1                        # one Hydra swap at a time
            assert len(composed) == n_threads and all(name.endswith("SAM2Base") for name in composed)
            assert compose(config_name="app").app_key == 1   # the app's search path is back
    finally:
        restore_gh_from_backup(session_backup)


def test_the_image_predictor_also_composes_sam2_config_inside_a_running_hydra_app(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The `prompt` stage's box masker loads SAM 2's IMAGE predictor from inside `bench_throughput`, a
    `@hydra.main` app. `SAM2ImagePredictor.from_pretrained` composes its config exactly as the video builder
    does, so it needs the same guard: calling it directly failed all 3 clips of the first end-to-end run with
    "Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'"."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import get_gh_backup, restore_gh_from_backup

    session_backup = get_gh_backup()
    pytest.importorskip("sam2")
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    from fastdo.shape import masks_sam2

    seen: dict[str, object] = {}

    def _fake_from_pretrained(model_id: str, device: str = "cuda", **kwargs: object) -> str:
        seen["model"] = compose(config_name="configs/sam2.1/sam2.1_hiera_l.yaml").model
        seen["device"], seen["model_id"] = device, model_id
        return "image_predictor"

    monkeypatch.setattr(SAM2ImagePredictor, "from_pretrained", staticmethod(_fake_from_pretrained))
    monkeypatch.setattr(masks_sam2, "hf_revision", lambda _model_id: None)   # from_pretrained: the unpinned path
    (tmp_path / "app.yaml").write_text("app_key: 1\n")
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
            got = masks_sam2.load_sam2_image_predictor(device="cpu", model_id="facebook/sam2.1-hiera-large")
            assert got == "image_predictor"
            assert seen["model"] is not None and seen["device"] == "cpu"
            assert compose(config_name="app").app_key == 1   # the app's search path is back
    finally:
        restore_gh_from_backup(session_backup)



def test_the_pinned_image_predictor_also_composes_sam2_config_inside_a_running_hydra_app(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pinned branch of `load_sam2_image_predictor` (the default model has a pin) builds through `build_sam2`
    with the checkpoint `hf_hub_download` returns at the pinned commit, not through `from_pretrained`. It must compose
    sam2's config inside a `@hydra.main` app as well, and give the app its search path back. No network, no GPU."""
    import huggingface_hub
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.initialize import get_gh_backup, restore_gh_from_backup

    session_backup = get_gh_backup()
    pytest.importorskip("sam2")
    from sam2 import build_sam, sam2_image_predictor

    from fastdo import model_pins
    from fastdo.shape import masks_sam2

    model_id = masks_sam2.MODEL_ID
    assert masks_sam2.hf_revision(model_id) == model_pins.HF_REVISIONS[model_id]      # the pinned branch
    downloads: list[dict[str, object]] = []
    seen: dict[str, object] = {}

    def _fake_download(**kwargs: object) -> str:
        downloads.append(kwargs)
        return str(tmp_path / "ckpt.pt")

    def _fake_build(config_file: str, ckpt_path: str, device: str = "cuda", **kwargs: object) -> str:
        seen["model"] = compose(config_name=config_file).model
        seen["ckpt_path"], seen["device"] = ckpt_path, device
        return "sam2_model"

    class _FakeImagePredictor:
        def __init__(self, model: object, **kwargs: object) -> None:
            self.model, self.kwargs = model, kwargs

        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("the pinned branch must not call from_pretrained (it reads main)")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(build_sam, "build_sam2", _fake_build)
    monkeypatch.setattr(sam2_image_predictor, "SAM2ImagePredictor", _FakeImagePredictor)
    (tmp_path / "app.yaml").write_text("app_key: 1\n")
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
            got = masks_sam2.load_sam2_image_predictor(device="cpu")
            assert isinstance(got, _FakeImagePredictor) and got.model == "sam2_model"
            assert got.kwargs == {"device": "cpu"}
            assert str(seen["model"]._target_).endswith("SAM2Base") and seen["device"] == "cpu"
            assert seen["ckpt_path"] == str(tmp_path / "ckpt.pt")
            _config, checkpoint = build_sam.HF_MODEL_ID_TO_FILENAMES[model_id]
            assert downloads == [{"repo_id": model_id, "filename": checkpoint,
                                  "revision": model_pins.HF_REVISIONS[model_id]}]
            assert compose(config_name="app").app_key == 1   # the app's search path is back
    finally:
        restore_gh_from_backup(session_backup)

def test_the_prompt_stages_box_masker_goes_through_the_guarded_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug was that `workers_prompt` bypassed the guard by calling `from_pretrained` itself; pin that it
    now loads through `load_sam2_image_predictor`, with the configured variant and device."""
    import numpy as np

    from fastdo.engine import workers_prompt
    from fastdo.shape.sam2_variants import SAM2_VARIANTS

    calls: list[tuple[str, str]] = []

    class _Predictor:
        def set_image(self, image: np.ndarray) -> None:
            self.image = image

        def predict(self, box: np.ndarray, multimask_output: bool = True):
            return np.ones((1, 4, 5), dtype=np.float32), None, None

    def _fake_loader(device: str = "cuda", model_id: str = "") -> _Predictor:
        calls.append((device, model_id))
        return _Predictor()

    monkeypatch.setattr("fastdo.shape.masks_sam2.load_sam2_image_predictor", _fake_loader)
    masker = workers_prompt._sam2_box_masker("tiny", "cpu")
    mask = masker(np.zeros((4, 5, 3), np.uint8), [0.0, 0.0, 3.0, 3.0])

    assert calls == [("cpu", SAM2_VARIANTS["tiny"])]
    assert mask.dtype == bool and mask.shape == (4, 5) and mask.all()


def test_parallel_jpeg_loader_is_bit_identical(tmp_path) -> None:
    """The thread-pooled JPEG-folder load must equal sam2's own loader bit for bit."""
    from fastdo.shape import masks_sam2

    if importlib.util.find_spec("sam2") is None:
        pytest.skip("sam2 not installed")
    misc = masks_sam2._import_sam2("sam2.utils.misc")

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:90, 0:160]
    frames = np.stack([np.stack([(xx * 1.3 + t * 7) % 256, (yy * 2.1 + t) % 256, rng.integers(0, 255, (90, 160))], -1)
                       for t in range(12)]).astype(np.uint8)
    masks_sam2._write_jpeg_frames(frames, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{i:05d}.jpg" for i in range(12)]
    want = misc.load_video_frames_from_jpg_images(str(tmp_path), image_size=64, offload_video_to_cpu=True)
    got = masks_sam2._load_jpeg_folder_parallel(str(tmp_path), 64, True, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                                                torch.device("cpu"))
    assert torch.equal(got[0], want[0]) and got[1:] == want[1:] == (90, 160)


def test_the_parallel_loader_is_installed_once_and_only_for_jpeg_folders(tmp_path, monkeypatch) -> None:
    from fastdo.shape import masks_sam2

    if importlib.util.find_spec("sam2") is None:
        pytest.skip("sam2 not installed")
    module = masks_sam2._import_sam2("sam2.sam2_video_predictor")

    calls: list[tuple] = []
    monkeypatch.setattr(module, "load_video_frames", lambda *a, **k: calls.append(a) or "original")
    masks_sam2._install_parallel_jpeg_loader()
    masks_sam2._install_parallel_jpeg_loader()
    wrapped = module.load_video_frames
    assert wrapped._fastdo_parallel
    assert wrapped("clip.mp4", 64, True) == "original"                      # not a folder: sam2's own path
    assert wrapped(str(tmp_path), 64, True, async_loading_frames=True) == "original"
    assert len(calls) == 2


def test_importing_sam2_leaves_hydra_as_it_found_it() -> None:
    """sam2/__init__ initialises Hydra's global config module; our import must not leak that to later tests."""
    from hydra.core.global_hydra import GlobalHydra

    from fastdo.shape import masks_sam2

    if importlib.util.find_spec("sam2") is None:
        pytest.skip("sam2 not installed")
    for name in [m for m in sys.modules if m == "sam2" or m.startswith("sam2.")]:
        sys.modules.pop(name)                     # force a genuinely first import
    GlobalHydra.instance().clear()
    masks_sam2._import_sam2("sam2.sam2_video_predictor")
    assert not GlobalHydra.instance().is_initialized()


def _without_global_hydra() -> object:
    """Removes the GlobalHydra singleton (as in a process that is not a Hydra app) and returns what to restore."""
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.singleton import Singleton

    return Singleton._instances.pop(GlobalHydra, None)


def _restore_singleton(saved: object) -> None:
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.singleton import Singleton

    Singleton._instances.pop(GlobalHydra, None)
    if saved is not None:
        Singleton._instances[GlobalHydra] = saved


def test_two_sam2_imports_outside_a_hydra_app_do_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside a Hydra app (release/example/run_example.py), the first `_import_sam2` runs the package `__init__`,
    which creates GlobalHydra, and removes it again; the second imports a submodule without running `__init__`.
    Restoring "no GlobalHydra" then must not raise KeyError. A stub package does what sam2/__init__ does, so this
    runs without sam2."""
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.singleton import Singleton

    from fastdo.shape import masks_sam2

    pkg = tmp_path / "stub_sam2_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from hydra.core.global_hydra import GlobalHydra\nGlobalHydra.instance()\n")
    (pkg / "first.py").write_text("")
    (pkg / "second.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    saved = _without_global_hydra()
    try:
        masks_sam2._import_sam2("stub_sam2_pkg.first")
        assert GlobalHydra not in Singleton._instances           # the import left Hydra as it found it
        masks_sam2._import_sam2("stub_sam2_pkg.second")         # KeyError: GlobalHydra before the fix
        assert GlobalHydra not in Singleton._instances
    finally:
        _restore_singleton(saved)
        for name in [m for m in sys.modules if m == "stub_sam2_pkg" or m.startswith("stub_sam2_pkg.")]:
            sys.modules.pop(name)


def test_the_image_predictors_two_sam2_imports_outside_a_hydra_app_do_not_raise() -> None:
    """The same with the real sam2 and the two modules `load_sam2_image_predictor` imports in turn (the predictor
    class, then `sam2.build_sam` for the pinned checkpoint). No model is built."""
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.singleton import Singleton

    from fastdo.shape import masks_sam2

    if importlib.util.find_spec("sam2") is None:
        pytest.skip("sam2 not installed")
    for name in [m for m in sys.modules if m == "sam2" or m.startswith("sam2.")]:
        sys.modules.pop(name)                     # force a genuinely first import
    saved = _without_global_hydra()
    try:
        assert masks_sam2._sam2_symbol("sam2.sam2_image_predictor", "SAM2ImagePredictor") is not None
        assert "sam2.build_sam" not in sys.modules    # else the second import below would not reach Hydra
        assert masks_sam2._sam2_symbol("sam2.build_sam", "build_sam2") is not None
        assert GlobalHydra not in Singleton._instances
    finally:
        _restore_singleton(saved)
