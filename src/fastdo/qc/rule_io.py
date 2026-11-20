"""Versioned JSON form of a fitted `AcceptanceRule`: the rule the engine's `qc` stage loads.

Version 1 has exactly these top-level keys: `format` (`ACCEPTANCE_RULE_FORMAT`), `version` (1), `feature_names`
(`FEATURE_NAMES`, the `QcFeatures` field order the weights and statistics refer to), `logistic_weights` (6 floats: the
bias, then one weight per feature), `feature_mean` (5), `feature_std` (5), `tau`, `calibration`
(`{threshold, alpha, n_calibration}`) and `provenance` (an object, possibly empty, recording how the rule was fitted).

Floats are written with Python's shortest round-trip repr, so a save/load round trip is bit-exact. Saving writes a tmp
sibling and moves it into place with `os.replace`; on any failure the tmp file is removed and the previous file stays
intact. Loading reads the file once and fails loudly on another format, an unsupported version, another feature order,
a missing or unknown key, or a non-numeric, non-finite or wrongly sized entry; `AcceptanceRule`'s own checks (shapes,
finiteness, positive std) still apply. A later format (e.g. stored calibration scores) gets a new version.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.qc.acceptance import AcceptanceRule
from fastdo.qc.conformal import ConformalCalibration
from fastdo.qc.features import QcFeatures

__all__ = ["ACCEPTANCE_RULE_FORMAT", "ACCEPTANCE_RULE_VERSION", "FEATURE_NAMES", "acceptance_rule_from_dict",
           "acceptance_rule_sha256", "acceptance_rule_to_dict", "load_acceptance_rule",
           "load_acceptance_rule_and_sha256", "read_acceptance_rule_provenance", "save_acceptance_rule"]

ACCEPTANCE_RULE_FORMAT: str = "fastdo.qc.AcceptanceRule"
ACCEPTANCE_RULE_VERSION: int = 1
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(QcFeatures))
_SUPPORTED_VERSIONS: tuple[int, ...] = (1,)
_KEYS = frozenset({"format", "version", "feature_names", "logistic_weights", "feature_mean", "feature_std", "tau",
                   "calibration", "provenance"})
_CALIBRATION_KEYS = frozenset({"threshold", "alpha", "n_calibration"})


def acceptance_rule_to_dict(rule: AcceptanceRule, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The version-1 JSON object of `rule` (module docstring), with `provenance` (default `{}`)."""
    calibration = rule.calibration
    return {"format": ACCEPTANCE_RULE_FORMAT, "version": ACCEPTANCE_RULE_VERSION, "feature_names": list(FEATURE_NAMES),
            "logistic_weights": [float(x) for x in rule.logistic_weights],
            "feature_mean": [float(x) for x in rule.feature_mean], "feature_std": [float(x) for x in rule.feature_std],
            "tau": float(rule.tau),
            "calibration": {"threshold": float(calibration.threshold), "alpha": float(calibration.alpha),
                            "n_calibration": int(calibration.n_calibration)},
            "provenance": dict(provenance or {})}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _numbers(value: Any, n: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{name} must be a list of {n} numbers, got {value!r}")
    return np.array([_number(v, f"{name}[{i}]") for i, v in enumerate(value)], dtype=np.float64)


def acceptance_rule_from_dict(payload: Mapping[str, Any]) -> AcceptanceRule:
    """The `AcceptanceRule` of a version-1 JSON object.

    Raises:
        ValueError: On another `format`, an unsupported `version` (the message names the supported ones),
            `feature_names` other than `FEATURE_NAMES` in order, a missing or unknown top-level key, or a
            non-numeric, non-finite or wrongly sized entry (and on anything `AcceptanceRule` itself rejects).
    """
    if not isinstance(payload, Mapping):         # a malformed rule file is a ValueError (the loader's contract)
        raise ValueError(f"an acceptance rule must be a JSON object, got {type(payload).__name__}")  # noqa: TRY004
    if payload.get("format") != ACCEPTANCE_RULE_FORMAT:
        raise ValueError(f"not an acceptance rule: format {payload.get('format')!r}, expected {ACCEPTANCE_RULE_FORMAT!r}")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in _SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported acceptance rule version {version!r}; supported versions: {list(_SUPPORTED_VERSIONS)}")
    if set(payload) != _KEYS:
        raise ValueError(f"acceptance rule keys must be exactly {sorted(_KEYS)}: missing {sorted(_KEYS - set(payload))}, "
                         f"unknown {sorted(set(payload) - _KEYS)}")
    if payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError(f"feature_names {payload['feature_names']!r} are not the QcFeatures fields in order "
                         f"{list(FEATURE_NAMES)}")
    calibration = payload["calibration"]
    if not isinstance(calibration, Mapping) or set(calibration) != _CALIBRATION_KEYS:
        raise ValueError(f"calibration must be an object with exactly the keys {sorted(_CALIBRATION_KEYS)}")
    alpha, tau = _number(calibration["alpha"], "calibration.alpha"), _number(payload["tau"], "tau")
    if not (0.0 < alpha < 1.0 and tau > 0.0):
        raise ValueError(f"need 0 < calibration.alpha < 1 and tau > 0, got {alpha} and {tau}")
    if not isinstance(payload["provenance"], Mapping):
        raise ValueError("provenance must be a JSON object")  # noqa: TRY004 (a malformed rule file is a ValueError)
    n, n_calibration = len(FEATURE_NAMES), calibration["n_calibration"]
    if isinstance(n_calibration, bool) or not isinstance(n_calibration, int) or n_calibration < 1:
        raise ValueError(f"calibration.n_calibration must be an integer >= 1, got {n_calibration!r}")
    return AcceptanceRule(
        calibration=ConformalCalibration(threshold=_number(calibration["threshold"], "calibration.threshold"),
                                         alpha=alpha, n_calibration=n_calibration),
        logistic_weights=_numbers(payload["logistic_weights"], n + 1, "logistic_weights"),
        feature_mean=_numbers(payload["feature_mean"], n, "feature_mean"),
        feature_std=_numbers(payload["feature_std"], n, "feature_std"), tau=tau)


def save_acceptance_rule(path: Path, rule: AcceptanceRule, provenance: Mapping[str, Any] | None = None) -> None:
    """Writes `rule` as version-1 standard JSON to `path` via a tmp sibling and `os.replace` (ValueError on NaN)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        text = json.dumps(acceptance_rule_to_dict(rule, provenance), indent=2, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _load(path: Path) -> tuple[AcceptanceRule, dict[str, Any], str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"acceptance rule {path} not found (fit one with run/pipeline/calibrate_acceptance.py)")
    data = path.read_bytes()             # one read: the rule, its provenance and the sha256 all come from these bytes
    try:
        payload = json.loads(data)
        return acceptance_rule_from_dict(payload), dict(payload["provenance"]), hashlib.sha256(data).hexdigest()
    except ValueError as exc:            # json.JSONDecodeError and UnicodeDecodeError are ValueErrors too
        raise ValueError(f"{path}: {exc}") from exc


def load_acceptance_rule(path: Path) -> AcceptanceRule:
    """The rule saved at `path`.

    Raises:
        FileNotFoundError: Naming the path, when the file is absent.
        ValueError: Naming the path, when the file is not valid JSON or not a valid version-1 rule.
    """
    return _load(path)[0]


def load_acceptance_rule_and_sha256(path: Path) -> tuple[AcceptanceRule, str]:
    """The rule saved at `path` and the hex sha256 of exactly the bytes it was parsed from (one read of the file)."""
    rule, _, sha256 = _load(path)
    return rule, sha256


def read_acceptance_rule_provenance(path: Path) -> dict[str, Any]:
    """The `provenance` object of the (validated) rule saved at `path`."""
    return _load(path)[1]


def acceptance_rule_sha256(path: Path) -> str:
    """Hex sha256 of the rule file's bytes (the qc stage records it next to each verdict)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
