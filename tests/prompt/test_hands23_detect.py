"""The detector side's decode read: only the candidate rows, the same bytes, and still no `fastdo` import.

`hands23_detect.py` runs in the Hands23 interpreter, which has no `fastdo`; it loads `fastdo/engine/npz_rows.py` BY
PATH. Detectron2 is not here, so the model and resize are faked and `records` is replaced by a digest of the exact
image the model received: equal digests mean the detector saw the same pixels as the full read gave it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from tests.engine.fakes import forbid_full_frames, random_frames, write_decode

DETECT = Path(__file__).resolve().parents[2] / "src/fastdo/prompt/hands23_detect.py"

FAKES = textwrap.dedent('''
    import hashlib
    from types import SimpleNamespace

    def install(module):
        """Identity resize, a model that returns its input image, and records that digest that image."""
        module.records = lambda image: [{"sha": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                                         "shape": list(image.shape)}]
        aug = SimpleNamespace(get_transform=lambda image: SimpleNamespace(apply_image=lambda x: x))
        model = lambda batch: [{"instances": batch[0]["image"]}]
        return model, aug
''')


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, DETECT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fakes() -> Any:
    namespace: dict[str, Any] = {}
    exec(FAKES, namespace)          # noqa: S102 -- the same fakes source runs in the subprocess below
    return namespace["install"]


def _request(decode: Path, rows: list[int]) -> dict:
    return {"decode_npz": str(decode), "rows": rows}


def _handle(module: ModuleType, request: dict) -> dict:
    model, aug = _fakes()(module)
    reply = module.handle(request, model, aug, Path("w.pth"), "cpu", "d" * 64)
    return reply["rows"]


def _full_read(path: Path, name: str, rows: list[int]) -> np.ndarray:
    with np.load(path) as data:                       # the detector's read before row-only reads
        return np.asarray(data[name])[rows]


@pytest.fixture
def decode(tmp_path: Path) -> Path:
    write_decode(tmp_path, random_frames(23, h=18, w=26, seed=9), stride=3)
    return tmp_path / "decode.npz"


def test_the_detector_reads_only_its_rows_and_sees_the_same_pixels(decode: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("_hands23_detect_under_test")
    rows = [0, 5, 10, 15, 20, 22]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module.npz_rows, "read_npz_rows", _full_read)
        before = _handle(module, _request(decode, rows))
    forbid_full_frames(monkeypatch)
    after = _handle(module, _request(decode, rows))

    assert after == before and sorted(int(r) for r in after) == rows
    frames = random_frames(23, h=18, w=26, seed=9)
    for row in rows:                                     # the detector's BGR image, computed independently
        bgr = np.ascontiguousarray(frames[row][:, :, ::-1]).astype(np.float32).transpose(2, 0, 1)
        assert after[str(row)][0]["sha"] == hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()


def test_the_detector_still_imports_nothing_from_fastdo(decode: Path) -> None:
    """Run as the Hands23 interpreter would: any `fastdo` import fails, yet the rows are read and answered."""
    script = textwrap.dedent(f'''
        import importlib.abc, importlib.util, json, sys
        from pathlib import Path

        class NoFastdo(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "fastdo" or name.startswith("fastdo."):
                    raise ImportError(f"the detector imported {{name}}")
                return None

        sys.meta_path.insert(0, NoFastdo())
        spec = importlib.util.spec_from_file_location("hands23_detect", {str(DETECT)!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = {{}}
        exec({FAKES!r}, namespace)
        model, aug = namespace["install"](module)
        reply = module.handle({json.dumps(_request(decode, [22, 3, 3]))}, model, aug, Path("w.pth"), "cpu", "d" * 64)
        print(json.dumps(reply["rows"]))
    ''')
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False,
                          cwd=str(decode.parent))
    assert done.returncode == 0, done.stderr[-2000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got == _handle(_load("_hands23_detect_reference"), _request(decode, [22, 3, 3]))


# -- the opt-in batched forward ------------------------------------------------------------------------------

def _batched_fakes(module: ModuleType, calls: list[int]) -> tuple[Any, Any]:
    """The same fakes, plus a `_forward_batch` that hands each input image back as that row's 'instances'."""
    model, aug = _fakes()(module)

    def forward_batch(_model: Any, inputs: list[dict]) -> list[Any]:
        calls.append(len(inputs))
        return [one["image"] for one in inputs]

    module._forward_batch = forward_batch
    return model, aug


ROWS = [0, 5, 10, 15, 20, 22]


def test_an_absent_batch_and_batch_one_are_the_one_row_loop(decode: Path) -> None:
    """The default path: no `batch` key and `batch: 1` give the same reply, and neither touches `_forward_batch`."""
    module = _load("_hands23_detect_default")
    calls: list[int] = []
    model, aug = _batched_fakes(module, calls)
    absent = module.handle(_request(decode, ROWS), model, aug, Path("w.pth"), "cpu", "d" * 64)
    one = module.handle({**_request(decode, ROWS), "batch": 1}, model, aug, Path("w.pth"), "cpu", "d" * 64)
    assert calls == []
    assert absent["rows"] == one["rows"]
    assert "batch" not in absent["meta"] and "batch" not in one["meta"]     # the reply's keys are unchanged too
    zero = module.handle({**_request(decode, ROWS), "prefetch": 0}, model, aug, Path("w.pth"), "cpu", "d" * 64)
    assert zero["rows"] == absent["rows"] and set(zero["meta"]) == set(absent["meta"])


@pytest.mark.parametrize("batch", [2, 4, 5, 6, 64])
def test_the_batched_path_answers_every_row_with_the_same_key_and_pixels(decode: Path, batch: int) -> None:
    """Per-row records keyed exactly as the one-row loop keys them, in the same order, from the same pixels."""
    module = _load(f"_hands23_detect_batch_{batch}")
    calls: list[int] = []
    model, aug = _batched_fakes(module, calls)
    reference = module.handle(_request(decode, ROWS), model, aug, Path("w.pth"), "cpu", "d" * 64)
    batched = module.handle({**_request(decode, ROWS), "batch": batch}, model, aug, Path("w.pth"), "cpu", "d" * 64)
    assert list(batched["rows"]) == list(reference["rows"]) == [str(r) for r in ROWS]
    assert batched["rows"] == reference["rows"]
    assert calls == [min(batch, len(ROWS) - i) for i in range(0, len(ROWS), batch)]
    assert batched["meta"]["batch"] == batch and batched["meta"]["n_rows"] == len(ROWS)


def test_repeated_rows_keep_the_one_row_loops_keys(decode: Path) -> None:
    module = _load("_hands23_detect_repeat")
    model, aug = _batched_fakes(module, [])
    rows = [22, 3, 3]
    reference = module.handle(_request(decode, rows), model, aug, Path("w.pth"), "cpu", "d" * 64)
    batched = module.handle({**_request(decode, rows), "batch": 2}, model, aug, Path("w.pth"), "cpu", "d" * 64)
    assert batched["rows"] == reference["rows"] and list(batched["rows"]) == list(reference["rows"])


def test_a_batch_below_one_is_an_error(decode: Path) -> None:
    module = _load("_hands23_detect_bad_batch")
    model, aug = _batched_fakes(module, [])
    with pytest.raises(ValueError, match="batch"):
        module.handle({**_request(decode, ROWS), "batch": 0}, model, aug, Path("w.pth"), "cpu", "d" * 64)
    with pytest.raises(ValueError, match="prefetch"):
        module.handle({**_request(decode, ROWS), "prefetch": -1}, model, aug, Path("w.pth"), "cpu", "d" * 64)


def test_a_forward_that_drops_rows_is_an_error(decode: Path) -> None:
    """The failure `model([...])` would give (Hands23's heads answer image 0 only) must raise, not shorten the reply."""
    module = _load("_hands23_detect_drop")
    model, aug = _fakes()(module)
    module._forward_batch = lambda _model, inputs: [inputs[0]["image"]]
    with pytest.raises(RuntimeError, match="results for"):
        module.handle({**_request(decode, ROWS), "batch": 3}, model, aug, Path("w.pth"), "cpu", "d" * 64)


def _install_fake_detectron2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Just enough of detectron2 for `_forward_batch`: an ImageList and GeneralizedRCNN._postprocess."""
    from types import SimpleNamespace

    class ImageList:
        def __init__(self, tensor: Any, image_sizes: list) -> None:
            self.tensor, self.image_sizes = tensor, image_sizes

    class GeneralizedRCNN:
        @staticmethod
        def _postprocess(results: list, inputs: list, image_sizes: list) -> list:
            assert len(results) == len(inputs) == len(image_sizes) == 1
            return [{"instances": (results[0], inputs[0]["tag"], image_sizes[0])}]

    for name, attrs in {"detectron2": {}, "detectron2.modeling": {}, "detectron2.modeling.meta_arch": {},
                        "detectron2.modeling.meta_arch.rcnn": {"GeneralizedRCNN": GeneralizedRCNN},
                        "detectron2.structures": {"ImageList": ImageList}}.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))


def test_forward_batch_batches_the_backbone_and_runs_the_heads_once_per_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backbone and RPN see all B images once; the ROI heads see ONE image's features and proposals per call."""
    _install_fake_detectron2(monkeypatch)
    module = _load("_hands23_detect_forward_batch")
    log: list[tuple] = []

    class Model:
        def __call__(self, inputs: list) -> None:
            raise AssertionError("model([...]) drops every image but the first (Hands23's heads); never call it")

        def preprocess_image(self, inputs: list) -> Any:
            from types import SimpleNamespace
            log.append(("pre", len(inputs)))
            return SimpleNamespace(tensor=np.stack([x["image"] for x in inputs]),
                                   image_sizes=[(x["height"], x["width"]) for x in inputs])

        def backbone(self, tensor: np.ndarray) -> dict:
            log.append(("backbone", tensor.shape[0]))
            return {"p2": tensor * 2, "p3": tensor * 3}

        def proposal_generator(self, images: Any, features: dict, targets: None) -> tuple[list, dict]:
            log.append(("rpn", features["p2"].shape[0]))
            return [f"proposals{i}" for i in range(features["p2"].shape[0])], {}

        def roi_heads(self, images: Any, features: dict, proposals: list, targets: None) -> tuple[list, dict]:
            assert images.tensor.shape[0] == 1 and len(images.image_sizes) == 1
            assert {k: v.shape[0] for k, v in features.items()} == {"p2": 1, "p3": 1}
            assert len(proposals) == 1
            log.append(("heads", proposals[0], float(features["p3"][0, 0, 0, 0])))
            return [f"instances<{proposals[0]}>"], {}

    inputs = [{"image": np.full((3, 4, 5), float(i)), "height": 8, "width": 10 + i, "tag": i} for i in range(3)]
    out = module._forward_batch(Model(), inputs)

    assert log[:3] == [("pre", 3), ("backbone", 3), ("rpn", 3)]
    assert log[3:] == [("heads", f"proposals{i}", 3.0 * i) for i in range(3)]   # image i's own features
    assert out == [(f"instances<proposals{i}>", i, (8, 10 + i)) for i in range(3)]


@pytest.mark.parametrize(("batch", "prefetch"), [(1, 1), (1, 2), (1, 5), (1, 64), (3, 1), (4, 2), (64, 3)])
def test_prefetch_gives_the_one_row_loops_records_prepared_off_the_main_thread(decode: Path, batch: int,
                                                                               prefetch: int) -> None:
    """Data movement only: the same pixels reach the model, keyed and ordered as before, prepared on the pool."""
    import threading

    module = _load(f"_hands23_detect_prefetch_{batch}_{prefetch}")
    calls: list[int] = []
    model, aug = _batched_fakes(module, calls)
    reference = module.handle(_request(decode, ROWS), model, aug, Path("w.pth"), "cpu", "d" * 64)
    threads: list[str] = []
    prepare = module._prepare

    def spy(frame: np.ndarray, aug_: Any) -> dict:
        threads.append(threading.current_thread().name)
        return prepare(frame, aug_)

    module._prepare = spy
    got = module.handle({**_request(decode, ROWS), "batch": batch, "prefetch": prefetch}, model, aug,
                        Path("w.pth"), "cpu", "d" * 64)
    assert list(got["rows"]) == list(reference["rows"]) and got["rows"] == reference["rows"]
    assert len(threads) == len(ROWS) and all(name.startswith("h23_prep") for name in threads)
    assert calls == ([] if batch == 1 else [min(batch, len(ROWS) - i) for i in range(0, len(ROWS), batch)])
    assert got["meta"]["prefetch"] == prefetch and got["meta"]["batch"] == batch


def test_prefetch_one_uses_the_models_own_call_not_the_batched_replay(decode: Path) -> None:
    """At batch 1 the prefetch path must make the loop's exact `model([one input])` call (byte-identity)."""
    module = _load("_hands23_detect_prefetch_call")
    model, aug = _fakes()(module)
    seen: list[int] = []

    def counting(batch_: list) -> Any:
        seen.append(len(batch_))
        return model(batch_)

    def no_replay(_model: Any, _inputs: list) -> list:
        raise AssertionError("batch 1 must not use _forward_batch")

    module._forward_batch = no_replay
    module.handle({**_request(decode, ROWS), "prefetch": 2}, counting, aug, Path("w.pth"), "cpu", "d" * 64)
    assert seen == [1] * len(ROWS)
