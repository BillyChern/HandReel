"""Tests for the opt-in `hands.deterministic` (CPU): the request, the driver environment, the response flag, and
the driver lifting YOLO's NMS wall-clock limit only when asked. Also that the determinism trace module stays unloaded
unless its environment variable is set."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from fastdo.hands.hawor_batched import (
    BATCH_INVARIANT_KEYS,
    DETERMINISTIC_KEYS,
    OPTION_KEYS,
    HandsClipInput,
    HandsConfig,
    driver_env,
    read_response,
    write_request,
)
from tests.hands.test_hawor_core import _clips
from tests.hands.test_hawor_driver_cpu import (  # noqa: F401 - fixtures
    _FakeApi,
    _request,
    _run,
    driver,
    shared,
)


class _Api(_FakeApi):
    """The CPU fake plus the determinism hook: records each `disable_nms_time_limit` call."""

    def __init__(self, clips: Any) -> None:
        super().__init__(clips)
        self.nms_calls = 0

    def disable_nms_time_limit(self) -> bool:
        self.nms_calls += 1
        return self.nms_calls == 1


@pytest.fixture
def cfg() -> HandsConfig:
    return HandsConfig(hawor_python=Path("/py"), hawor_root=Path("/root"))


def test_deterministic_is_off_by_default_and_the_request_bytes_are_unchanged(cfg: HandsConfig, tmp_path: Path) -> None:
    clips = [HandsClipInput("a", tmp_path / "a/decode.npz", 812.5, tmp_path / "a/hands_hawor.npz")]
    before = {"version": 1, "hawor_root": str(Path(cfg.hawor_root).absolute()),
              "options": {key: getattr(cfg, key) for key in OPTION_KEYS},
              "clips": [{"clip_id": "a", "frames_path": str(tmp_path / "a/decode.npz"), "focal_px": 812.5,
                         "out_npz": str(tmp_path / "a/hands_hawor.npz")}]}

    assert cfg.deterministic is False
    assert write_request(tmp_path / "off", clips, cfg).read_text() == json.dumps(before, indent=1)


def test_deterministic_on_adds_its_option_after_batch_invariant(cfg: HandsConfig, tmp_path: Path) -> None:
    clips = [HandsClipInput("a", tmp_path / "a/decode.npz", 812.5, tmp_path / "a/hands_hawor.npz")]
    both = replace(cfg, batch_invariant=True, deterministic=True)

    options = json.loads(write_request(tmp_path / "on", clips, both).read_text())["options"]

    assert DETERMINISTIC_KEYS == ("deterministic",)
    assert list(options) == [*OPTION_KEYS, *BATCH_INVARIANT_KEYS, "deterministic"] and options["deterministic"] is True


def test_driver_env_is_unchanged_by_deterministic() -> None:
    assert "CUBLAS_WORKSPACE_CONFIG" not in driver_env({"PATH": "/usr/bin"})


def test_read_response_records_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "response.json"
    body = {"version": 1, "error": "", "load_seconds": 0.0, "warmup_seconds": 0.0, "compute_seconds": 0.0,
            "peak_cuda_gb": 0.0, "clips": []}
    path.write_text(json.dumps(body))
    assert read_response(path, 0).deterministic is False
    path.write_text(json.dumps(body | {"deterministic": True}))
    assert read_response(path, 0).deterministic is True


def test_driver_lifts_the_nms_time_limit_only_when_asked(driver: ModuleType, shared: Any,  # noqa: F811
                                                         tmp_path: Path) -> None:
    clips = _clips()
    for sub in ("off", "on"):
        (tmp_path / sub).mkdir()
    before = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
              torch.are_deterministic_algorithms_enabled())

    api = _Api(clips)
    code, body = _run(driver, shared, tmp_path / "off", api, _request(tmp_path / "off", clips))
    assert code == 0 and "deterministic" not in body and api.nms_calls == 0

    api = _Api(clips)
    code, on = _run(driver, shared, tmp_path / "on", api, _request(tmp_path / "on", clips, deterministic=True))
    assert code == 0 and on["deterministic"] is True and api.nms_calls == 1
    assert (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled()) == before              # numerics flags untouched


def test_driver_outputs_are_unchanged_by_the_option_on_cpu(driver: ModuleType, shared: Any,  # noqa: F811
                                                           tmp_path: Path) -> None:
    clips = _clips()
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    _, off = _run(driver, shared, tmp_path / "a", _Api(clips), _request(tmp_path / "a", clips))
    _, on = _run(driver, shared, tmp_path / "b", _Api(clips), _request(tmp_path / "b", clips, deterministic=True))
    for a, b in zip(off["clips"], on["clips"], strict=True):
        assert a["status"] == b["status"] == "done"
    for i in range(len(clips)):
        x = (tmp_path / "a" / "out" / f"clip{i}.npz").read_bytes()
        y = (tmp_path / "b" / "out" / f"clip{i}.npz").read_bytes()
        assert x == y


def test_trace_is_not_loaded_without_its_env_var(driver: ModuleType) -> None:  # noqa: F811
    assert driver._TRACE is None


def test_disable_nms_time_limit_wraps_the_module_attribute_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import math
    import sys
    import types

    seen: list[dict[str, Any]] = []

    def fake_nms(prediction: Any, conf_thres: float = 0.25, iou_thres: float = 0.45, *args: Any,
                 max_time_img: float = 0.05, **kwargs: Any) -> list[Any]:
        seen.append({"conf": conf_thres, "max_time_img": max_time_img})
        return [prediction]

    nms = types.ModuleType("ultralytics.utils.nms")
    nms.non_max_suppression = fake_nms                                        # type: ignore[attr-defined]
    utils = types.ModuleType("ultralytics.utils")
    utils.nms = nms                                                           # type: ignore[attr-defined]
    root = types.ModuleType("ultralytics")
    root.utils = utils                                                        # type: ignore[attr-defined]
    for name, module in (("ultralytics", root), ("ultralytics.utils", utils), ("ultralytics.utils.nms", nms)):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "run/pipeline/_hawor_driver_models.py"
    spec = importlib.util.spec_from_file_location("fastdo_test_driver_models_260", path)
    assert spec is not None and spec.loader is not None
    models = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, models)                     # dataclasses resolve their module
    spec.loader.exec_module(models)

    fake_nms("p", 0.2)
    assert seen[-1]["max_time_img"] == 0.05                                   # the limit ultralytics applies today
    assert models.disable_nms_time_limit() is True
    assert models.disable_nms_time_limit() is False                          # idempotent: one wrapper only
    assert nms.non_max_suppression("p", 0.2, 0.45) == ["p"]
    assert seen[-1] == {"conf": 0.2, "max_time_img": math.inf}
    assert nms.non_max_suppression.__wrapped__ is fake_nms                    # type: ignore[attr-defined]
