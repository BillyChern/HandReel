"""CPU tests of the batched HaWoR driver's request loop (`run/pipeline/_hawor_batched_driver.py`) with fake models.

The driver file is loaded by path, as in the hawor env. `FakeHawor` from `test_hawor_core` stands in for the network, and
a fake model API stands in for `_hawor_driver_models` (detector, MANO). No HaWoR code or GPU is involved.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fastdo.hands.hand_npz import validate_hand_npz
from fastdo.hands.hawor_batched import SECONDS_KEYS
from tests.hands.test_hawor_core import OPS, FakeHawor, _Clip, _clips

REPO = Path(__file__).resolve().parents[2]
DRIVER_FILE = REPO / "run" / "pipeline" / "_hawor_batched_driver.py"


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fastdo_test_hawor_driver", DRIVER_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shared(driver: ModuleType) -> dict[str, ModuleType]:
    return driver.load_shared_modules(REPO)


class _FakeApi:
    """`_hawor_driver_models`-shaped: raw boxes looked up by frame shape; MANO outputs are simple functions of inputs."""

    def __init__(self, clips: list[_Clip], fail_detect_shape: tuple[int, ...] | None = None,
                 fail_load: bool = False) -> None:
        self.boxes = {c.frames.shape: c.boxes for c in clips}
        self.fail_detect_shape, self.fail_load = fail_detect_shape, fail_load
        self.warm_ups = 0

    def load_models(self, hawor_root: Path, shared: Any) -> SimpleNamespace:
        if self.fail_load:
            raise FileNotFoundError("weights missing: /fake/hawor.ckpt")
        faces = {"left": np.zeros((1552, 3), np.int32), "right": np.zeros((1552, 3), np.int32)}
        return SimpleNamespace(hawor=FakeHawor(), yolo=None, mano={}, ops=OPS, faces=faces)

    def warm_up(self, models: Any, shared: Any, det_frames_per_batch: int) -> None:
        self.warm_ups += 1

    def detect_clip(self, models: Any, shared: Any, frames: np.ndarray, det_conf: float,
                    det_frames_per_batch: int) -> dict[str, np.ndarray]:
        if frames.shape == self.fail_detect_shape:
            raise RuntimeError("fake detector exploded")
        return {side: boxes.copy() for side, boxes in self.boxes[frames.shape].items()}

    def run_mano(self, models: Any, side: str, rot: np.ndarray, hand_pose: np.ndarray, betas: np.ndarray,
                 trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        offset = (trans + 0.01 * rot + 0.001 * betas[:, :3]).astype(np.float32)
        return (np.repeat(offset[:, None], 778, axis=1), np.repeat(offset[:, None], 21, axis=1))


def _request(tmp_path: Path, clips: list[_Clip], **options: Any) -> dict[str, Any]:
    opts = {"det_conf": 0.2, "det_frames_per_batch": 16, "bf16": False, "temporal": "source_grid",
            "crops_per_backbone_batch": 8, "windows_per_heads_batch": 4, "clips_per_group": 8,
            "max_crops_per_group": 4096, "upload_frames_per_chunk": 5, "max_fill_gap": 120} | options
    specs = []
    for i, clip in enumerate(clips):
        path = tmp_path / f"clip{i}_frames.npz"
        np.savez(path, frames=clip.frames, frame_idx=clip.frame_idx, fps=np.float64(10.0))
        specs.append({"clip_id": f"clip{i}", "frames_path": str(path), "focal_px": clip.geom.focal_px,
                      "out_npz": str(tmp_path / "out" / f"clip{i}.npz")})
    return {"version": 1, "hawor_root": str(tmp_path), "options": opts, "clips": specs}


def _run(driver: ModuleType, shared: Any, tmp_path: Path, api: _FakeApi, request: dict[str, Any]) -> tuple[int, dict]:
    response = tmp_path / "response.json"
    code = driver.run_request(request, response, shared, api, device="cpu")
    return code, json.loads(response.read_text())


def _arrays(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def test_driver_writes_valid_npz_and_response(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    clips = _clips()
    api = _FakeApi(clips)
    request = _request(tmp_path, clips)

    code, body = _run(driver, shared, tmp_path, api, request)

    assert code == 0 and body["version"] == 1 and body["error"] == "" and api.warm_ups == 1
    assert [c["status"] for c in body["clips"]] == ["done", "done"]
    for spec, result, clip in zip(request["clips"], body["clips"], clips):
        assert set(result["seconds"]) == set(SECONDS_KEYS)
        assert result["seconds"]["total"] == pytest.approx(sum(v for k, v in result["seconds"].items() if k != "total"))
        assert validate_hand_npz(Path(spec["out_npz"])) == result["n_frames"] == len(clip.frame_idx)
    second = _arrays(request["clips"][1]["out_npz"])
    assert np.flatnonzero(second["left_infilled"]).tolist() == [3]   # clip B's missing left row, filled
    assert second["left_valid"].all() and second["right_valid"].all()
    np.testing.assert_array_equal(second["frame_idx"], clips[1].frame_idx)


def test_driver_grouping_does_not_change_outputs(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    clips = _clips()
    outputs = []
    for name, options in (("one", {}), ("per_clip", {"clips_per_group": 1}), ("crop_limit", {"max_crops_per_group": 30})):
        run_dir = tmp_path / name
        run_dir.mkdir()
        request = _request(run_dir, clips, **options)
        code, _ = _run(driver, shared, run_dir, _FakeApi(clips), request)
        assert code == 0
        outputs.append([_arrays(spec["out_npz"]) for spec in request["clips"]])
    for other in outputs[1:]:
        for ours, theirs in zip(outputs[0], other):
            assert ours.keys() == theirs.keys()
            for key, value in ours.items():
                if value.dtype.kind == "f" and not key.endswith("_box"):
                    np.testing.assert_allclose(theirs[key], value, rtol=0, atol=1e-5, err_msg=key)
                else:
                    np.testing.assert_array_equal(theirs[key], value, err_msg=key)


def test_driver_clip_failure_is_isolated(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    clips = _clips()
    code, body = _run(driver, shared, tmp_path, _FakeApi(clips, fail_detect_shape=clips[0].frames.shape),
                      _request(tmp_path, clips))

    assert code == 1
    assert [c["status"] for c in body["clips"]] == ["failed", "done"]
    assert "detect" in body["clips"][0]["error"] and "fake detector exploded" in body["clips"][0]["error"]


def test_driver_load_failure_exits_2(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    clips = _clips()
    code, body = _run(driver, shared, tmp_path, _FakeApi(clips, fail_load=True), _request(tmp_path, clips))

    assert code == 2 and "weights missing" in body["error"] and body["clips"] == []


def test_driver_group_failure_fails_every_clip_of_the_group(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("heads out of memory")

    core = SimpleNamespace(**vars(shared["hawor_core"]))
    core.run_heads = _boom
    broken = dict(shared) | {"hawor_core": core}
    clips = _clips()

    code, body = _run(driver, broken, tmp_path, _FakeApi(clips), _request(tmp_path, clips))

    assert code == 1
    assert [c["status"] for c in body["clips"]] == ["failed", "failed"]
    assert all("heads" in c["error"] and "out of memory" in c["error"] for c in body["clips"])


def test_driver_bad_frames_npz_fails_that_clip(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    clips = _clips()
    request = _request(tmp_path, clips)
    np.savez(request["clips"][0]["frames_path"], frames=clips[0].frames.astype(np.float32),
             frame_idx=clips[0].frame_idx, fps=np.float64(10.0))

    code, body = _run(driver, shared, tmp_path, _FakeApi(clips), request)

    assert code == 1
    assert [c["status"] for c in body["clips"]] == ["failed", "done"]
    assert body["clips"][0]["error"].startswith("read: ValueError")


class _RngApi(_FakeApi):
    """Detections jittered by torch's global RNG, so outputs depend on the RNG state each request starts from."""

    def detect_clip(self, models: Any, shared: Any, frames: np.ndarray, det_conf: float,
                    det_frames_per_batch: int) -> dict[str, np.ndarray]:
        import torch

        raw = super().detect_clip(models, shared, frames, det_conf, det_frames_per_batch)
        return {side: boxes + np.float64(torch.rand(()).item()) for side, boxes in raw.items()}


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_serve_matches_fresh_calls_byte_for_byte(driver: ModuleType, shared: Any, tmp_path: Path,
                                                 order: tuple[int, ...]) -> None:
    """Every request served by one warm process writes the bytes a fresh process writes for it."""
    import io

    import torch

    clips = _clips()
    fresh, warm = tmp_path / "fresh", tmp_path / "warm"
    for i in range(3):
        (fresh / f"r{i}").mkdir(parents=True)
        request = _request(fresh / f"r{i}", clips)
        torch.manual_seed(0)                                   # a fresh process starts from the same RNG state
        assert driver.run_request(request, fresh / f"r{i}" / "response.json", shared, _RngApi(clips),
                                  device="cpu") == 0
    paths = []
    for i in range(3):
        (warm / f"r{i}").mkdir(parents=True)
        body = _request(warm / f"r{i}", clips) | {"hawor_root": str(tmp_path)}   # one server, one HaWoR root
        (warm / f"r{i}" / "request.json").write_text(json.dumps(body))
        paths.append(str(warm / f"r{i}" / "request.json"))
    api, protocol = _RngApi(clips), io.StringIO()
    torch.manual_seed(0)

    code = driver.serve(io.StringIO("".join(f"{paths[i]}\n" for i in order)), protocol, shared, api, device="cpu")

    assert code == 0 and api.warm_ups == 1
    assert [json.loads(line) for line in protocol.getvalue().splitlines()] == [
        {"request": paths[i], "code": 0, "serving": True} for i in order]
    for n, i in enumerate(order):
        body = json.loads((warm / f"r{i}" / "response.json").read_text())
        assert (body["load_seconds"] == 0.0) == (n > 0) and len(body["clips"]) == 2
        for c in range(2):
            ours = (warm / f"r{i}" / "out" / f"clip{c}.npz").read_bytes()
            assert ours == (fresh / f"r{i}" / "out" / f"clip{c}.npz").read_bytes(), (i, c)


def test_serve_load_failure_answers_2_and_stops(driver: ModuleType, shared: Any, tmp_path: Path) -> None:
    import io

    clips = _clips()
    request = tmp_path / "request.json"
    request.write_text(json.dumps(_request(tmp_path, clips)))
    protocol = io.StringIO()

    code = driver.serve(io.StringIO(f"{request}\n{request}\n"), protocol, shared, _FakeApi(clips, fail_load=True),
                        device="cpu")

    assert code == 2
    assert [json.loads(line) for line in protocol.getvalue().splitlines()] == [
        {"request": str(request), "code": 2, "serving": False}]
    assert "weights missing" in json.loads((tmp_path / "response.json").read_text())["error"]
