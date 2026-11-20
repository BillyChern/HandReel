# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Inconsistent moments and stored splits are refused, not clamped (alongray.pool, A3).

Moments of real numbers obey Cauchy-Schwarz (``pz^2 <= pp zz``, ``z1^2 <= zz``, ``a1^2 <= pp``); a negative squared
term in the split means the moments are not moments of any trajectory. Rounding may push an exact zero slightly
negative, so the checks carry a relative tolerance; beyond it the row is refused.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alongray import e_par_split, split_from_moments, split_of_row
from alongray.pool import MOMENT_RTOL


def _traj(n: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.45, 0.75, n)])


def _moments(p: np.ndarray, g: np.ndarray) -> dict[str, float]:
    z = np.linalg.norm(g, axis=1)
    a = np.einsum("ij,ij->i", p, g) / z
    return {"pp": float(np.mean(a * a)), "pz": float(np.mean(a * z)), "zz": float(np.mean(z * z)), "n": len(z),
            "z1": float(np.mean(z)), "a1": float(np.mean(a))}


@pytest.mark.parametrize("bad, match", [
    ({"pz": 2.0}, "Cauchy-Schwarz"),                         # pz^2 = 4 > pp zz = 1: off-orbit^2 would be -3
    ({"pp": -1.0}, "pp"),
    ({"zz": 0.0}, "zz"),
    ({"pp": math.nan}, "finite"),
    ({"pz": math.inf}, "finite"),
    ({"n": 0}, "n"),
    ({"n": 2.5}, "n"),
    ({"n": True}, "n"),
    ({"z1": 1.5}, "z1"),                                     # z1^2 > zz
    ({"z1": -0.5}, "z1"),                                    # ranges are positive
    ({"a1": 1.5}, "a1"),                                     # a1^2 > pp
])
def test_inconsistent_moments_are_refused(bad: dict, match: str) -> None:
    m = {"pp": 1.0, "pz": 0.9, "zz": 1.0, "n": 5, "z1": 0.99, "a1": 0.9, **bad}
    with pytest.raises(ValueError, match=match):
        split_from_moments(**m)
    with pytest.raises(ValueError, match=match):
        split_of_row({"unit_id": "u", "moments": m})


def test_rounding_at_an_exact_zero_is_within_tolerance() -> None:
    """A pure scale has off-orbit^2 = 0 exactly; computed moments may round it to a tiny negative number."""
    for seed in range(50):
        g = _traj(seed=seed)
        m = _moments(1.1 * g, g)
        assert m["pp"] - m["pz"] ** 2 / m["zz"] <= MOMENT_RTOL * m["pp"]
        sp = split_from_moments(**m)
        assert sp.off_orbit_part <= 1e-6 * sp.e_par and sp.lam == pytest.approx(1.1, rel=1e-12)


def test_just_beyond_the_tolerance_is_refused() -> None:
    pp = zz = 1.0
    pz = math.sqrt(pp * zz * (1 + 10 * MOMENT_RTOL))
    with pytest.raises(ValueError, match="Cauchy-Schwarz"):
        split_from_moments(pp, pz, zz, 5)


def test_the_identity_holds_on_every_accepted_split() -> None:
    for seed in range(20):
        rng = np.random.default_rng(seed)
        g = _traj(seed=seed)
        sp = split_from_moments(**_moments(rng.uniform(0.8, 1.2) * g + rng.normal(scale=0.01, size=g.shape), g))
        assert sp.e_par ** 2 == pytest.approx(sp.gauge_part ** 2 + sp.off_orbit_part ** 2, rel=1e-9, abs=1e-15)


def _stored(seed: int = 0) -> dict:
    g = _traj(seed=seed)
    return dataclasses.asdict(e_par_split(1.05 * g + 0.01, g))


def test_a_consistent_stored_split_is_accepted() -> None:
    sp = _stored()
    assert split_of_row({"unit_id": "u", "split": sp}).e_par == sp["e_par"]


@pytest.mark.parametrize("edit, match", [
    ({"n_frames": 0}, "n_frames"),
    ({"n_frames": 3.5}, "n_frames"),
    ({"lam": 0.0}, "lam"),
    ({"lam": -1.0}, "lam"),
    ({"e_par": -0.1}, "non-negative"),
    ({"z_rms": 0.0}, "z_rms"),
    ({"e_par": 1.0}, "identity"),                              # e_par^2 != gauge^2 + off^2
    ({"gauge_part": 0.5, "e_par": None}, "gauge_part"),        # gauge != |lam - 1| z_rms (e_par kept consistent)
    ({"oracle_floor": 1.0}, "oracle_floor"),                   # oracle floor != off / lam
])
def test_an_inconsistent_stored_split_is_refused(edit: dict, match: str) -> None:
    sp = {**_stored(), **edit}
    if edit.get("e_par", 0) is None:                          # keep the e_par identity true, break only the gauge
        sp["e_par"] = math.sqrt(sp["gauge_part"] ** 2 + sp["off_orbit_part"] ** 2)
    with pytest.raises(ValueError, match=match):
        split_of_row({"unit_id": "u", "split": sp})
