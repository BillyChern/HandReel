# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Known answer: a rigid translation that puts the wrist 10% too far along its ray leaves root-aligned MPJPE and
PA-MPJPE unchanged but moves the unaligned along-ray error by 10% of the hand's range.

The fixture is synthetic. A 21-joint hand moves through 24 frames; the wrist depth is sampled uniformly in
0.45-0.75 m. The baseline prediction has the wrist exact and Gaussian noise with a 4 mm standard deviation on each
coordinate of the other joints, so its aligned errors are not zero. The shifted prediction translates that same hand rigidly by 0.1 times its wrist position in each frame: the
wrist lands 10% too far along its own ray, the hand keeps its size, and every other joint moves by the same vector,
not along its own ray. That is the error a monocular method makes when it puts a fixed-size hand at the wrong depth.
For contrast, the full gauge multiplies every joint by 1.1 about the camera centre (each joint along its own ray, the
hand 10% larger); root-aligned MPJPE then sees the size change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alongray import score

K_SCALE = 1.1
T, J = 24, 21


def _rot(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _fixture() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    wrist = np.column_stack([rng.uniform(-0.08, 0.08, T), rng.uniform(-0.06, 0.06, T), rng.uniform(0.45, 0.75, T)])
    local = 0.035 * rng.standard_normal((J, 3))
    local[0] = 0.0
    gt = np.stack([local @ _rot(rng).T + wrist[t] for t in range(T)])
    noise = 0.004 * rng.standard_normal((T, J, 3))
    noise[:, 0] = 0.0                                           # wrist exact: only the articulation is wrong
    base = gt + noise
    obj = wrist + np.array([0.0, 0.05, 0.03])                   # an object scored exactly, to satisfy the API
    return {"gt": gt, "base": base, "obj": obj, "wrist": wrist}


def _hand(pred: np.ndarray, f: dict[str, np.ndarray]):
    return score(f["obj"], f["obj"], pred, f["gt"])["hand"]


def test_scaled_placement_moves_e_par_and_neither_aligned_metric() -> None:
    f = _fixture()
    scaled = f["base"] + (K_SCALE - 1.0) * f["base"][:, :1]    # the whole hand shifted by 0.1 x its wrist position
    b0, b1 = _hand(f["base"], f), _hand(scaled, f)

    assert b0.aligned["mpjpe_root_aligned"] > 0.003            # the articulation error is visible ...
    assert b1.aligned["mpjpe_root_aligned"] == pytest.approx(b0.aligned["mpjpe_root_aligned"], rel=1e-12)
    assert b1.aligned["pa_mpjpe"] == pytest.approx(b0.aligned["pa_mpjpe"], rel=1e-9)   # ... and unchanged

    assert b0.e_par == pytest.approx(0.0, abs=1e-15)           # the wrist (reference point) was exact
    assert b1.lam == pytest.approx(K_SCALE, rel=1e-13)
    assert b1.e_par == pytest.approx((K_SCALE - 1.0) * b1.z_rms, rel=1e-12)
    assert b1.gauge_part == pytest.approx(b1.e_par, rel=1e-12) and b1.off_orbit_part == pytest.approx(0, abs=1e-13)
    assert b1.e_perp == pytest.approx(b0.e_perp, abs=1e-15)    # all of the change is along the ray
    assert 0.05 < b1.e_par < 0.07                              # about 60 mm at 0.6 m


def test_full_gauge_contrast_root_alignment_sees_the_size_change() -> None:
    f = _fixture()
    gauge = K_SCALE * f["base"]                                 # position and size both times 1.1
    b0, b2 = _hand(f["base"], f), _hand(gauge, f)
    assert b2.e_par == pytest.approx((K_SCALE - 1.0) * b2.z_rms, rel=1e-12)
    assert b2.aligned["pa_mpjpe"] == pytest.approx(b0.aligned["pa_mpjpe"], rel=1e-9)   # similarity absorbs size
    assert abs(b2.aligned["mpjpe_root_aligned"] - b0.aligned["mpjpe_root_aligned"]) > 0.001


TABLE_ROWS = {"baseline": "base", "wrist 10% too far, hand translated rigidly": "shift",
              "every joint times 1.1 about the camera": "gauge"}


def _table_cells() -> dict[str, tuple[float, ...]]:
    """The four millimetre cells of each row of the table in KNOWN_ANSWERS.md, read from the file."""
    text = (Path(__file__).resolve().parents[1] / "KNOWN_ANSWERS.md").read_text()
    out: dict[str, tuple[float, ...]] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] in TABLE_ROWS:
            out[TABLE_ROWS[cells[0]]] = tuple(float(c.removesuffix(" mm")) for c in cells[1:])
    return out


def test_printed_table_values() -> None:
    """Reads the table of KNOWN_ANSWERS.md and compares each cell with this fixture's rounded output, so a change of
    the note or of the code that separates the two fails here."""
    f = _fixture()
    base = f["base"]
    shift = base + (K_SCALE - 1.0) * base[:, :1]
    rows = {name: _hand(pred, f) for name, pred in (("base", base), ("shift", shift), ("gauge", K_SCALE * base))}
    printed = _table_cells()
    assert printed == {"base": (0.0, 0.0, 6.09, 5.64), "shift": (60.1, 0.0, 6.09, 5.64),
                       "gauge": (60.1, 0.0, 8.24, 5.64)}             # the cells as the note prints them today
    for name, (e_par_mm, e_perp_mm, root_mm, pa_mm) in printed.items():
        b = rows[name]
        assert round(b.e_par * 1e3, 1) == e_par_mm
        assert round(b.e_perp * 1e3, 1) == e_perp_mm
        assert round(b.aligned["mpjpe_root_aligned"] * 1e3, 2) == root_mm
        assert round(b.aligned["pa_mpjpe"] * 1e3, 2) == pa_mm
        assert b.e_perp == pytest.approx(0.0, abs=1e-15)

    wrist_z = f["wrist"][:, 2]
    assert 0.45 <= wrist_z.min() and wrist_z.max() <= 0.75     # the sampling range of the wrist depth

    step = (K_SCALE - 1.0) * base[:, :1] * np.ones_like(base)   # the one vector every joint of a frame moves by
    ray = base / np.linalg.norm(base, axis=2, keepdims=True)
    along = (step * ray).sum(axis=2)
    across = np.linalg.norm(step - along[..., None] * ray, axis=2)
    assert 46.0 <= along[:, 1:].min() * 1e3 and along[:, 1:].max() * 1e3 <= 75.0
    assert round(across[:, 1:].max() * 1e3, 1) == 9.3
