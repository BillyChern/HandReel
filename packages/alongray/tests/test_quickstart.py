# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""The README's quickstart: examples/make_example.py, then the `alongray` command, gives the numbers the README
states."""

from __future__ import annotations

import json
import re
import runpy
from pathlib import Path

import pytest

from alongray.cli import main as cli_main

PKG = Path(__file__).resolve().parents[1]
EXAMPLE = PKG / "examples" / "make_example.py"


def test_quickstart_commands_are_the_ones_the_readme_shows():
    text = (PKG / "README.md").read_text()
    block = re.search(r"## Quickstart.*?```bash\n(.*?)```", text, re.DOTALL).group(1)
    assert "python examples/make_example.py out" in block
    assert "alongray out/pred.npz out/gt.npz --intrinsics 600,600,320,240 --out out/scores.json" in block


def test_quickstart_gives_the_stated_numbers(tmp_path, capsys):
    runpy.run_path(str(EXAMPLE), run_name="example").get("main")(tmp_path)
    out = tmp_path / "scores.json"
    rc = cli_main([str(tmp_path / "pred.npz"), str(tmp_path / "gt.npz"), "--intrinsics", "600,600,320,240",
                   "--out", str(out)])
    assert rc in (0, None)
    res = json.loads(out.read_text())
    for body, e_par in (("object", 0.0628), ("hand", 0.0626)):
        b = res[body]
        assert b["lam"] == pytest.approx(1.1, abs=1e-12)
        assert b["e_par"] == pytest.approx(e_par, abs=5e-5)
        assert b["gauge_part"] == pytest.approx(b["e_par"], abs=1e-12)
        assert abs(b["off_orbit_part"]) < 1e-12 and abs(b["reproj_rms_px"]) < 1e-12
        assert b["collinearity"] == pytest.approx(0.98, abs=0.005)
        assert b["shift_part"] == pytest.approx(0.062, abs=5e-4)
    assert abs(res["hand"]["aligned"]["mpjpe_root_aligned"]) < 1e-12
    assert abs(res["hand"]["aligned"]["pa_mpjpe"]) < 1e-12
