"""Regression tests of fastdo.audit for inputs that were silently ignored:

- an unknown npz field (a misspelt `sim_dtt`) was ignored, silently dropping the check it meant to ask for;
- a `hand_away` (npz field or `--hand-away`) given with a removal that does not use it was ignored.
Both are now input errors (exit status 2, as every other refused input).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastdo.audit.__main__ import main
from tests.audit.test_input_validation_regressions import _carry_npz

N, W = 120, 20


def _kinematic(tmp: Path, **extra: object) -> Path:
    ref = np.tile([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], (N, 1))
    p = tmp / "k.npz"
    np.savez(p, reference=ref, warmup_rows=np.int64(W), required_rows=np.int64(N), **extra)
    return p


def test_an_unknown_npz_field_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _carry_npz(tmp_path, sim_dtt=np.float64(0.002))
    assert main([str(p)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "sim_dtt" in err and "Traceback" not in err, err
    assert main([str(_carry_npz(tmp_path))]) == 0                                       # green control


def test_an_unknown_field_on_a_kinematic_run_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(_kinematic(tmp_path, refrence=np.zeros(3)))]) == 2
    assert "refrence" in capsys.readouterr().err
    assert main([str(_kinematic(tmp_path))]) == 0                                       # green control


@pytest.mark.parametrize("removal", ["displace", "xml"])
def test_an_npz_hand_away_unused_by_the_removal_is_exit_2(tmp_path: Path, removal: str,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    p = _carry_npz(tmp_path, hand_away=np.array([0.0, 2.0]))
    assert main([str(p), "--removal", removal]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "hand_away" in err and "Traceback" not in err, err
    assert main([str(p), "--removal", "pairs"]) == 0                                    # green control: used


@pytest.mark.parametrize("removal", ["displace", "xml"])
def test_a_hand_away_argument_unused_by_the_removal_is_exit_2(tmp_path: Path, removal: str,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    p = _carry_npz(tmp_path)
    assert main([str(p), "--removal", removal, "--hand-away", "0", "2"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "--hand-away" in err and "Traceback" not in err, err
    assert main([str(p), "--removal", removal]) == 0                                    # green control
