"""Gate bundles: the fitted gate, its calibration, encoder crop and model card in one weights-only file.

`save_gate_bundle` writes a `torch.save` of a plain dict (version, config, state_dict, calibration, crop,
hand_conf_source, card) through a tmp sibling and `os.replace`, and returns the file's sha256.
`load_gate_bundle` reads the bytes once, hashes those bytes and parses them with `torch.load(weights_only=True)`, so the
recorded sha256 is of exactly what was loaded and no pickled object can execute.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fastdo.engine.workers_frames import CROP_MODES
from fastdo.gate.calibrate import GateCalibration
from fastdo.gate.model import GateConfig, InteractionGate

__all__ = ["GATE_BUNDLE_VERSION", "GateBundle", "load_gate_bundle", "save_gate_bundle"]

GATE_BUNDLE_VERSION = 1


@dataclass(frozen=True)
class GateBundle:
    """A loaded gate bundle; `model` is in eval mode, `path` is resolved and `sha256` is of the bytes loaded."""

    model: InteractionGate
    config: GateConfig
    calibration: GateCalibration
    crop: str
    hand_conf_source: str
    card: dict[str, Any]
    path: str
    sha256: str


def save_gate_bundle(path: Path, model: InteractionGate, calibration: GateCalibration, crop: str,
                     card: Mapping[str, Any], hand_conf_source: str = "none") -> str:
    """Writes the bundle (module docstring) and returns its sha256.

    Raises:
        ValueError: If `crop` is not in `CROP_MODES`, or the card is not JSON-serializable.
    """
    if crop not in CROP_MODES:
        raise ValueError(f"gate bundle crop must be one of {CROP_MODES}, got {crop!r}")
    payload: dict[str, Any] = {
        "version": GATE_BUNDLE_VERSION,
        "config": dataclasses.asdict(model.cfg),
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "calibration": dataclasses.asdict(calibration),
        "crop": crop,
        "hand_conf_source": str(hand_conf_source),
        "card": json.loads(json.dumps(dict(card))),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gate_bundle(path: Path) -> GateBundle:
    """Loads a bundle written by `save_gate_bundle`. Building the model does not advance the caller's global RNG.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If the file is not a weights-only bundle, its version is not `GATE_BUNDLE_VERSION`, or its crop is
            unknown.
    """
    path = Path(path)
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        raise ValueError(f"{path} is not a weights-only gate bundle: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != GATE_BUNDLE_VERSION:
        version = payload.get("version") if isinstance(payload, dict) else None
        raise ValueError(f"{path}: gate bundle version {version!r} is not {GATE_BUNDLE_VERSION}")
    crop = str(payload["crop"])
    if crop not in CROP_MODES:
        raise ValueError(f"{path}: gate bundle crop {crop!r} is not one of {CROP_MODES}")
    config = GateConfig(**payload["config"])
    with torch.random.fork_rng(devices=[]):
        model = InteractionGate(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return GateBundle(model=model, config=config, calibration=GateCalibration(**payload["calibration"]), crop=crop,
                      hand_conf_source=str(payload.get("hand_conf_source", "none")),
                      card=dict(payload.get("card") or {}), path=str(path.resolve()), sha256=sha256)
