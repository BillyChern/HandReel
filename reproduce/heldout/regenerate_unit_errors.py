"""Rebuild the held-out test's per-unit error file from your own results on the datasets.

    python reproduce/heldout/regenerate_unit_errors.py --results my_results.jsonl \\
        [--out reproduce/heldout/unit_errors_est.json]

This release ships no per-unit error derived from HOI4D or DexYCB ground truth. To run
`reproduce/e1_verdict_from_scores.py`, regenerate the file after acquiring both datasets under their own terms
(HOI4D: <https://hoi4d.github.io/>; DexYCB: <https://dex-ycb.github.io/>), running both methods on the held-out units
(`units_hoi4d_72.json`, `units_dexycb_s08_108.json`) and scoring each unit against the dataset's ground truth.

`--results` is a JSONL file with one JSON object per line, one per assigned unit (all 180, each once):

    {"unit_id": "<as in the unit lists>", "run": true,
     "ours": {"delivered": true, "e_mm": <float>}, "theirs": {"delivered": false, "e_mm": null}}

- `run`: for this study, set `run` to true when DO AS I DO began an attempt on the unit (`protocol/`, amendment
  A-V). Thirty-five baseline start refusals have `run` false even though HandReel processed them; the registered
  analysis drops HandReel's results for those units, so enter both methods as not delivered there. That leaves 58
  of 72 HOI4D units and 87 of 108 DexYCB units with `run` true. The two DexYCB timeouts of amendment A-W have `run`
  true and are then removed from both methods (`REMOVED` below), which leaves 85 analysed DexYCB units;
- `delivered`: whether the method returned a result that is scored (ours: only after passing its QC);
- `e_mm`: the unit's mean unaligned object-centre error in mm over its scored frames (Sec. 5.1 of the paper: the processed
  frames with ground truth), `null` when not delivered. `processed_frames` lists a unit's processed frames.
  `unit_centre_error_mm` below is an AVERAGING HELPER FOR COMPLETED ROWS, not the registered scorer: it needs one
  centre for every scored frame and refuses a missing or repeated frame. Complete each method's centres first by the
  registered rule (`protocol/`, pre-registration Sec. 4, with the convention the registered scorer executed):
  both methods are scored on the same grid, the unit's processed frames that the ground truth marks valid; a pose
  is usable when it is a finite full pose (for DO AS I DO also with positive translation z); a frame without a
  usable pose takes that method's nearest usable pose, the earlier one on a tie; then the centre distance is taken
  at every grid frame and averaged. `complete_centres` below applies that fill to per-frame centres once you have
  decided which poses are usable. Averaging only the frames a method happens to deliver gives a different number
  (frames 0, 3, 6 with 10 mm at 0, 40 mm at 6 and no pose at 3: 25 mm over the two, 20 mm by the rule).

Any other key is refused, except `corpus`, `cluster` and `stratum`, which must then equal the unit lists' values.
The script fills each unit's corpus, cluster and stratum from the unit lists, marks the two DexYCB units that the
registered amendments A-T2 and A-W remove from both methods (`REMOVED`, named in `protocol/`), checks every row and
writes the file that `e1_verdict_from_scores.py` reads. Errors you compute from your own runs are your own: they
reproduce the paper's printed values only as far as your runs reproduce the registered ones. The registered scorer,
DO AS I DO's sealed outputs and HandReel's registered per-unit outputs are not released (`reproduce/README.md`). The file
you write is derived from the datasets' ground truth: keep it under their terms (both CC BY-NC 4.0) and do not
redistribute it with this code.

Exit 1 with a one-line message on any input error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HELDOUT = Path(__file__).resolve().parent
DEFAULT_OUT = HELDOUT / "unit_errors_est.json"
FORMAT = "handreel.heldout_unit_errors"
# Removed from both methods by amendments A-T2 and A-W (protocol/2026-10-22-e1-amendment-A-W.md): INFRA-UNRESOLVED.
REMOVED = ("D:20201002_111644:841412060263", "D:20201002_113031:932122062010")
REQUIRED = {"unit_id", "run", "ours", "theirs"}
FROM_LISTS = ("corpus", "cluster", "stratum")             # optional in a row; must then equal the unit lists' value


def load_units() -> dict[str, dict[str, Any]]:
    """unit_id -> {corpus, cluster, stratum, unit} from the two shipped unit lists."""
    out: dict[str, dict[str, Any]] = {}
    for name in ("units_hoi4d_72.json", "units_dexycb_s08_108.json"):
        doc = json.loads((HELDOUT / name).read_text())
        for u in doc["units"]:
            out[u["unit_id"]] = {"corpus": doc["corpus"], "cluster": u["cluster"], "stratum": u.get("stratum"),
                                 "unit": u}
    return out


def processed_frames(unit: dict[str, Any]) -> list[int]:
    """Source frame indices of a unit's processed frames: HOI4D windows `range(start, stop, stride)`; DexYCB views
    every `stride`-th frame of the whole clip (`n_frames` source frames)."""
    if "start" in unit:
        return list(range(int(unit["start"]), int(unit["stop"]), int(unit["stride"])))
    return list(range(0, int(unit["n_frames"]), int(unit["stride"])))


def _unique_frames(frames: np.ndarray, what: str) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(f"{what} frame IDs must be a 1-D integer array")
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"{what} frame IDs repeat")
    return frames.astype(np.int64)


def _flags(values: object, name: str) -> np.ndarray:
    """Per-frame flags as booleans. Only Boolean values or the integers 0 and 1 are accepted: a plain cast would
    read NaN and the string 'False' as true and change the fill or the scored frames."""
    v = np.asarray(values)
    if v.dtype != np.bool_ and not (np.issubdtype(v.dtype, np.integer) and np.isin(v, (0, 1)).all()):
        raise ValueError(f"'{name}' must be Boolean or 0/1 integers, got {v.dtype}")
    return v.astype(bool)


def complete_centres(grid: list[int], pred_frames: np.ndarray, pred_xyz_m: np.ndarray,
                     usable: np.ndarray) -> np.ndarray:
    """One centre per frame of `grid`: the method's own centre where its pose is usable, else the centre of its
    nearest usable frame (by frame ID), the earlier one on a tie. `usable` is yours to decide per frame (see the
    module docstring); a non-finite centre is never usable.

    Raises:
        ValueError: If no frame is usable, a frame ID repeats, or a shape does not match.
    """
    pred_frames = _unique_frames(pred_frames, "prediction")
    pred_xyz_m, usable = np.asarray(pred_xyz_m, np.float64), _flags(usable, "usable")
    if pred_xyz_m.shape != (len(pred_frames), 3) or usable.shape != (len(pred_frames),):
        raise ValueError("frames, (N, 3) centres and the usable flags must have matching lengths")
    ok = usable & np.isfinite(pred_xyz_m).all(axis=1)
    if not ok.any():
        raise ValueError("no usable pose: the unit is not delivered, do not score it")
    order = np.argsort(pred_frames[ok], kind="stable")
    f_ok, x_ok = pred_frames[ok][order], pred_xyz_m[ok][order]
    grid_arr = _unique_frames(np.asarray(list(grid)), "grid")              # no cast first: 3.5 is refused, not 3
    # argmin returns the first minimum, and f_ok is ascending, so a tie goes to the earlier frame
    pick = np.abs(grid_arr[:, None] - f_ok[None, :]).argmin(axis=1)
    return x_ok[pick]


def unit_centre_error_mm(pred_frames: np.ndarray, pred_xyz_m: np.ndarray, gt_frames: np.ndarray,
                         gt_xyz_m: np.ndarray, gt_valid: np.ndarray, scored: list[int]) -> float:
    """Mean distance in mm between predicted and ground-truth object centres (camera frame, metres, no alignment)
    over the frames of `scored` that the ground truth marks valid. An averaging helper for completed rows: every
    such frame needs exactly one finite prediction (`complete_centres`); it does not apply the registered fill.

    Raises:
        ValueError: If a scored valid frame has no finite prediction, a frame ID repeats or is not an integer,
            `scored` names a frame the ground truth lacks, a frame marked valid has a non-finite ground-truth
            centre, no frame is scorable, or a shape does not match.
    """
    pred_frames, gt_frames = _unique_frames(pred_frames, "prediction"), _unique_frames(gt_frames, "ground-truth")
    pred_xyz_m, gt_xyz_m = np.asarray(pred_xyz_m, np.float64), np.asarray(gt_xyz_m, np.float64)
    gt_valid = _flags(gt_valid, "gt_valid")
    if pred_xyz_m.shape != (len(pred_frames), 3) or gt_xyz_m.shape != (len(gt_frames), 3) or \
            gt_valid.shape != (len(gt_frames),):
        raise ValueError("frames and (N, 3) centres must have matching lengths")
    keep = _unique_frames(np.asarray(list(scored)), "scored")              # no cast first: 3.5 is refused, not 3
    gt_all = {int(f): (x, bool(v)) for f, x, v in zip(gt_frames, gt_xyz_m, gt_valid)}
    absent = [int(f) for f in keep if int(f) not in gt_all]
    if absent:
        raise ValueError(f"scored frames {absent[:5]} are not in the ground truth")
    broken = [int(f) for f in keep if gt_all[int(f)][1] and not np.isfinite(gt_all[int(f)][0]).all()]
    if broken:                                                             # never a silent change of denominator
        raise ValueError(f"scored frames {broken[:5]} are marked valid but their ground-truth centre is not finite")
    grid = [int(f) for f in keep if gt_all[int(f)][1]]
    if not grid:
        raise ValueError("no scored frame has valid ground truth")
    pred = {int(f): x for f, x in zip(pred_frames, pred_xyz_m)}
    missing = [f for f in grid if f not in pred or not np.isfinite(pred[f]).all()]
    if missing:
        raise ValueError(f"scored frames {missing[:5]} have no finite prediction: complete the rows by the "
                         "registered rule first (complete_centres)")
    return float(np.mean([np.linalg.norm(pred[f] - gt_all[f][0]) for f in grid]) * 1000.0)


def _side(row: dict[str, Any], key: str, uid: str) -> dict[str, Any]:
    s = row.get(key)
    if not isinstance(s, dict) or set(s) != {"delivered", "e_mm"} or not isinstance(s["delivered"], bool):
        raise ValueError(f"{uid}: '{key}' must be {{\"delivered\": bool, \"e_mm\": float or null}}")
    e = s["e_mm"]
    if s["delivered"] != (e is not None):
        raise ValueError(f"{uid}: '{key}' gives e_mm exactly when delivered")
    if e is not None and (not isinstance(e, (int, float)) or isinstance(e, bool) or not math.isfinite(e) or e <= 0):
        raise ValueError(f"{uid}: '{key}' e_mm must be a positive finite number")
    return {"delivered": s["delivered"], "e_mm": None if e is None else float(e)}


def build(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The per-unit error file from one result row per assigned unit."""
    units = load_units()
    seen: dict[str, dict[str, Any]] = {}
    for n, row in enumerate(results, 1):
        if not isinstance(row, dict):
            raise TypeError(f"row {n}: not a JSON object")
        uid = row.get("unit_id")
        if uid not in units:
            raise ValueError(f"row {n}: {uid!r} is not a unit of the shipped unit lists")
        unknown = sorted(set(row) - REQUIRED - set(FROM_LISTS))
        if unknown:
            raise ValueError(f"{uid}: unknown key(s) {', '.join(unknown)}")
        for key in sorted(set(row) & set(FROM_LISTS)):
            if row[key] != units[uid][key]:
                raise ValueError(f"{uid}: '{key}' is {row[key]!r}, but the shipped unit lists give {units[uid][key]!r}")
        if uid in seen:
            raise ValueError(f"{uid}: given twice")
        if row.get("run") is not True and row.get("run") is not False:
            raise ValueError(f"{uid}: 'run' must be true or false")
        ours, theirs = _side(row, "ours", uid), _side(row, "theirs", uid)
        if not row["run"] and (ours["delivered"] or theirs["delivered"]):
            raise ValueError(f"{uid}: a unit with 'run' false carries no delivered result: the registered analysis "
                             "drops both methods' rows for it (amendment A-V)")
        seen[uid] = {"run": row["run"], "ours": ours, "theirs": theirs}
    missing = sorted(set(units) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} assigned units have no row, e.g. {missing[0]}")
    rows = [{"unit_id": uid, "corpus": units[uid]["corpus"], "cluster": units[uid]["cluster"],
             "stratum": units[uid]["stratum"], "run": seen[uid]["run"],
             "removed": "INFRA-UNRESOLVED" if uid in REMOVED else None,
             "ours": seen[uid]["ours"], "theirs": seen[uid]["theirs"]} for uid in units]
    return {"format": FORMAT, "version": 1, "arm": "EST",
            "source": "regenerated by the user from their own runs and the datasets' ground truth; not distributable",
            "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, required=True, help="JSONL, one result row per assigned unit")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output file (default {DEFAULT_OUT.name})")
    a = ap.parse_args(argv)
    try:
        lines = [ln for ln in a.results.read_text().splitlines() if ln.strip()]
        doc = build([json.loads(ln) for ln in lines])
    except (OSError, ValueError, TypeError) as e:
        print(f"regenerate_unit_errors: {e}", file=sys.stderr)
        return 1
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    n = {c: sum(1 for r in doc["rows"] if r["corpus"] == c) for c in ("H", "D")}
    print(f"regenerate_unit_errors: wrote {a.out} ({n['H']} HOI4D and {n['D']} DexYCB units); keep it to yourself: "
          "it is derived from the datasets' ground truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
