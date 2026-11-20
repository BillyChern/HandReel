# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""0.4.1: every JSON alongray reads is parsed strictly; a repeated key (nested ones included) is refused by name."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from alongray import coverage_io, missing_bounds, strict_json
from alongray.pool import main as pool_main

PKG = Path(__file__).resolve().parents[1]


def test_repeated_key_refused_and_named() -> None:
    with pytest.raises(strict_json.DuplicateKeyError, match=r"rows line 3: repeated key 'status'"):
        strict_json.loads('{"status": "ok", "u": 1, "status": "failed"}', "rows line 3")


def test_nested_repeated_key_refused() -> None:
    with pytest.raises(strict_json.DuplicateKeyError, match=r"repeated key 'b'"):
        strict_json.loads('{"a": [{"b": 1, "b": 2}]}')


def test_valid_control_equals_plain_json() -> None:
    text = '{"a": [1, {"b": 2, "c": [3.5, null]}], "d": "x"}'
    assert strict_json.loads(text) == json.loads(text)  # strict-json-reference
    assert strict_json.loads("[1, 2]") == [1, 2]


def _pool_rows(tmp_path: Path, extra: str = "") -> Path:
    rng = np.random.default_rng(0)
    lines = []
    for i in range(4):
        g = np.column_stack([rng.normal(0, .05, 8), rng.normal(0, .05, 8), rng.uniform(.5, .8, 8)])
        p = g * 1.05 + rng.normal(0, .002, g.shape)
        z = np.linalg.norm(g, axis=1)
        row = json.dumps({"g": "x", "cl": str(i), "z": z.tolist(), "a_r": (np.einsum("ij,ij->i", p, g) / z).tolist()})
        lines.append(row[:-1] + extra + "}" if i == 2 else row)
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join(lines) + "\n")
    return f


def test_pool_control_reads(tmp_path: Path) -> None:
    assert pool_main([str(_pool_rows(tmp_path)), "--group-key", "g"]) == 0


def test_pool_refuses_repeated_key(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    f = _pool_rows(tmp_path, ', "g": "y"')
    assert pool_main([str(f), "--group-key", "g"]) == 2
    assert "line 3: repeated key 'g'" in capsys.readouterr().err


def test_pool_rows_key_refuses_nested_repeated_key(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    f = tmp_path / "doc.json"
    f.write_text('{"rows": [{"g": "x", "cl": "0", "cl": "1", "z": [1, 1, 1], "a_r": [1, 1, 1]}]}')
    assert pool_main([str(f), "--rows-key", "rows"]) == 2
    assert "repeated key 'cl'" in capsys.readouterr().err


def test_coverage_reader_refuses_repeated_key(tmp_path: Path) -> None:
    f = tmp_path / "protocol.json"
    f.write_text('{"units": {"u0": {"seq": "a", "seq": "b"}}}')
    with pytest.raises(ValueError, match=r"repeated key 'seq'"):
        coverage_io._read(f)


def test_missing_bounds_refuses_repeated_key(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    f = tmp_path / "p.json"
    f.write_text('{"units": ["u0"], "units": ["u0", "u1"]}')
    assert missing_bounds.main([str(f), str(f), str(f)]) == 2
    assert "repeated key 'units'" in capsys.readouterr().err


def test_no_bare_json_parsing_left() -> None:
    """Every JSON alongray reads goes through alongray.strict_json (src, examples and tools)."""
    bare = [f"{p.relative_to(PKG)}:{i}" for d in ("src", "examples", "tools") for p in sorted((PKG / d).rglob("*.py"))
            if p.name != "strict_json.py"
            for i, line in enumerate(p.read_text().splitlines(), 1) if re.search(r"\bjson\.loads?\(", line)]
    assert bare == []


def test_plain_json_mutant_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture) -> None:
    """Mutant: plain json (the last value of a repeated key wins). The refusal tests fail against it."""
    plain = lambda text, where="", **kw: json.loads(text, **kw)  # noqa: E731  strict-json-reference
    monkeypatch.setattr(strict_json, "loads", plain)
    f = _pool_rows(tmp_path, ', "g": "y"')
    assert pool_main([str(f), "--group-key", "g"]) == 0                     # accepted: g silently becomes "y"
    with pytest.raises(AssertionError):
        test_pool_refuses_repeated_key(tmp_path / "again" if (tmp_path / "again").mkdir() is None else tmp_path,
                                       capsys)
    with pytest.raises(AssertionError):
        test_missing_bounds_refuses_repeated_key(tmp_path, capsys)
