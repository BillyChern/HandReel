# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""The README's minimal example runs and says what the README says it says."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_example(capsys: pytest.CaptureFixture[str]) -> None:
    code = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)[0]
    namespace: dict = {}
    exec(compile(code, str(README), "exec"), namespace)  # noqa: S102
    res = namespace["res"]
    for body in ("object", "hand"):
        b = res[body]
        assert b.lam == pytest.approx(1.1, rel=1e-12)
        assert b.off_orbit_part == pytest.approx(0.0, abs=1e-12)
        assert b.e_par == pytest.approx(b.gauge_part, rel=1e-12)
        assert 0.055 < b.e_par < 0.07
        assert b.reproj_rms_px < 1e-9
    assert res["hand"].aligned["mpjpe_root_aligned"] < 1e-12 and res["hand"].aligned["pa_mpjpe"] < 1e-12
    assert "lam 1.100" in capsys.readouterr().out


PKG = Path(__file__).resolve().parents[1]
KNOWN = PKG / "KNOWN_ANSWERS.md"


def _flat(p: Path) -> str:
    return " ".join(p.read_text().split())


def test_the_pixel_claim_is_scoped() -> None:
    """A4: the image-exactness of a per-body gauge holds for a pinhole camera and isolated projected points;
    occlusion and contact between bodies can change pixels."""
    for p in [README, KNOWN, *sorted((PKG / "src" / "alongray").glob("*.py"))]:
        t = _flat(p)
        assert "moves no pixel" not in t and "every pixel of both stays put" not in t, p.name
        assert "every pixel of the body stays where it is" not in t, p.name
    t = _flat(README).lower()
    assert "pinhole" in t and "isolated" in t and "occlusion" in t and "contact" in t


def test_no_fixture_claims_to_reproduce_a_paper_figure() -> None:
    """The shipped fixtures are synthetic known answers; they reproduce no figure of the accompanying papers,
    and the pooled error split is not the per-window radial share of a solve change."""
    for p in (README, KNOWN):
        t = _flat(p)
        assert "97%" not in t and "97 %" not in t and "0.99 " not in t, p.name
        assert "reproduce no figure of the accompanying papers" in t, p.name
    t = _flat(README)
    assert "not the per-window radial share of a solve change" in t
    assert "do not ship, so a user cannot rerun those numbers from this package" in t     # the limitation
    assert "development data that does not ship" in t        # the 0.995 / 0.999 collinearity medians


def test_the_readme_grants_no_rights_and_states_the_row_assumption() -> None:
    """The licence section sends users to the applicable terms and says nothing here grants rights; the frame-id
    options and the row-by-row synchronization assumption are documented."""
    t = _flat(README)
    assert "Publish the derived numbers" not in t
    assert "check the terms that apply to it" in t
    assert "Nothing in this package, its documentation or its licence grants any right" in t
    assert "assumed to be the same frame** (synchronized row by row)" in t
    assert "pred_ids=None, gt_ids=None) -> Score" in t and "--require-frame-ids" in t and "`frame_ids`" in t
    assert '"pooled_share_ci95": "NOT_ESTIMABLE"' in t
