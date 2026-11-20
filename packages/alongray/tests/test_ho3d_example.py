# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""examples/ho3d_shaped_adapter.py: synthetic data only; its pure-gauge prediction gives lam = 1.05 with no off-orbit
term on both bodies, the origin mismatch shows up as off-orbit error, and its files score from the command line with
--require-frame-ids."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from alongray.cli import main as cli_main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ho3d_shaped_adapter.py"


def test_ho3d_shaped_adapter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = runpy.run_path(str(EXAMPLE), run_name="example")
    summary = mod["main"](tmp_path)
    for body in ("object", "hand"):
        b = summary[body]
        assert b["lam"] == pytest.approx(1.05, rel=1e-12)
        assert b["gauge_part"] == pytest.approx(b["e_par"], rel=1e-12) and b["off_orbit_part"] < 1e-12
        assert b["n_frames"] == 16 and b["reproj_rms_px"] < 1e-9
    assert summary["origin_mismatch"]["off_orbit_part"] > 1e-3
    with np.load(tmp_path / "pred.npz") as p, np.load(tmp_path / "gt.npz") as g:
        assert p["frame_ids"].dtype.kind == "i" and not np.array_equal(p["frame_ids"], g["frame_ids"])
        assert sorted(p["frame_ids"].tolist()) == g["frame_ids"].tolist()
        assert g["object"].shape == (16, 4, 4) and g["hand"].shape == (16, 21, 3)
        assert np.allclose(g["object"][:, 3], [0.0, 0.0, 0.0, 1.0])
        assert (g["object"][:, 2, 3] > 0).all()                          # OpenCV axes, metres
        assert np.allclose([np.linalg.det(r) for r in g["object"][:, :3, :3]], 1.0)
    capsys.readouterr()
    out = tmp_path / "s.json"
    assert cli_main([str(tmp_path / "pred.npz"), str(tmp_path / "gt.npz"), "--require-frame-ids",
                     "--out", str(out)]) == 0
    cli = json.loads(out.read_text())
    assert cli["object"]["e_par"] == summary["object"]["e_par"] and cli["hand"]["lam"] == summary["hand"]["lam"]
