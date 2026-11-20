# Copyright 2026 The alongray authors. SPDX-License-Identifier: Apache-2.0
"""reproduce/alongray_shares.py: known answers on synthetic windows (a pure depth error has share 1, a pure lateral
error share 0, equal along-ray and lateral parts share 1/2), the bootstrap's determinism and the reading rule, the
input contract (unique (group, window) pairs; identical ordered frame ids, or equal row counts without them) and the
provenance block."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
try:                                     # uv sync does not install alongray; the package source is pure Python
    import alongray  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "packages" / "alongray" / "src"))
sys.path.insert(0, str(ROOT / "reproduce"))
import strict_json as SJ

# ALONGRAY_SHARES_PATH lets the NaN mutant test run one test of this file against a mutated copy of the script
SCRIPT = Path(os.environ.get("ALONGRAY_SHARES_PATH", str(ROOT / "reproduce" / "alongray_shares.py")))
spec = importlib.util.spec_from_file_location("alongray_shares", SCRIPT)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def test_demo_known_answers(tmp_path, capsys):
    assert T.main(["--demo", str(tmp_path / "d"), "--boot", "200", "--json", str(tmp_path / "out.json")]) == 0
    doc = SJ.loads((tmp_path / "out.json").read_text())
    res = doc["groups"]
    assert abs(res["along"]["median_share"] - 1.0) < 1e-12 and res["along"]["reading"] == "along-ray dominated"
    assert {r["interval"] for r in res.values()} == {"estimated"} and doc["provenance"]["origin_tolerance_m"] == 1e-6
    assert res["lateral"]["median_share"] < 1e-12 and res["lateral"]["reading"] == "not along-ray dominated"
    assert abs(res["mixed"]["median_share"] - 0.5) < 1e-12
    assert {r["n_windows"] for r in res.values()} == {6} and {r["n_clusters"] for r in res.values()} == {3}
    prov = doc["provenance"]
    assert prov["numpy"] == np.__version__ and prov["bootstrap"]["B"] == 200 and prov["bootstrap"]["seed"] == 0
    assert len(prov["input_sha256"]) == 24 and len(prov["manifest_sha256"]) == 64     # 6 gt + 18 pred files
    assert res["along"]["windows_used"] == [f"w{i}" for i in range(6)] and res["along"]["windows_left_out"] == []
    out = capsys.readouterr().out
    assert "along: median share 1.000 [1.000, 1.000] (6 windows, 3 clusters; along-ray dominated)" in out


def test_window_with_too_few_frames_is_left_out(tmp_path):
    gt = np.array([[0.0, 0.0, 0.6], [0.01, 0.0, 0.6], [np.nan, 0.0, 0.6]])
    np.savez(tmp_path / "gt.npz", object=gt)
    np.savez(tmp_path / "p.npz", object=1.1 * gt)
    (tmp_path / "m.jsonl").write_text(json.dumps({"group": "g", "window": "w", "cluster": "c", "pred": "p.npz",
                                                  "gt": "gt.npz"}) + "\n")
    r = T.pool(tmp_path / "m.jsonl", 10, 0)["groups"]["g"]
    assert r["n_windows"] == 0 and r["left_out"] == 1 and r["n_assigned"] == 1
    assert r["windows_left_out"] == [{"window": "w", "reason": "fewer than 3 frames finite in both"}]
    assert r["window_flags"][0]["n_nonfinite"] == 1


def _pair(tmp_path, pred_ids=None, gt_ids=None, t_pred=5, t_gt=5):
    rng = np.random.default_rng(1)
    gt = np.column_stack([rng.normal(0, 0.05, t_gt), rng.normal(0, 0.05, t_gt), rng.uniform(0.5, 0.8, t_gt)])
    pred = 1.1 * gt[:t_pred] if t_pred <= t_gt else np.vstack([1.1 * gt, 1.1 * gt[: t_pred - t_gt]])
    np.savez(tmp_path / "gt.npz", object=gt, **({} if gt_ids is None else {"frame_idx": np.asarray(gt_ids)}))
    np.savez(tmp_path / "p.npz", object=pred, **({} if pred_ids is None else {"frame_idx": np.asarray(pred_ids)}))


def _manifest(tmp_path, rows):
    (tmp_path / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path / "m.jsonl"


ROW = {"group": "g", "window": "w", "cluster": "c", "pred": "p.npz", "gt": "gt.npz"}


def test_duplicate_group_window_is_refused(tmp_path, capsys):
    _pair(tmp_path)
    m = _manifest(tmp_path, [ROW, {**ROW, "cluster": "d"}])
    with pytest.raises(T.ManifestError, match="duplicate \\(group, window\\)"):
        T.pool(m, 10, 0)
    assert T.main([str(m)]) == 2 and "duplicate (group, window)" in capsys.readouterr().err
    # the same window id in another group is not a duplicate
    assert T.pool(_manifest(tmp_path, [ROW, {**ROW, "group": "h"}]), 10, 0)["groups"]["h"]["n_windows"] == 1


def test_identical_ordered_frame_ids_are_accepted(tmp_path):
    _pair(tmp_path, [3, 4, 5, 6, 7], [3, 4, 5, 6, 7])
    r = T.pool(_manifest(tmp_path, [ROW]), 10, 0)["groups"]["g"]
    assert r["n_windows"] == 1 and abs(r["median_share"] - 1.0) < 1e-12


@pytest.mark.parametrize("pred_ids,gt_ids,msg", [
    ([3, 4, 5, 7, 6], [3, 4, 5, 6, 7], "frame_idx differ"),          # same ids, misordered
    ([3, 4, 5, 6, 8], [3, 4, 5, 6, 7], "frame_idx differ"),          # one id differs
    ([3, 4, 5, 6, 7], None, "frame_idx is in pred only"),
    (None, [3, 4, 5, 6, 7], "frame_idx is in gt only"),
    ([3, 4, 5, 6], [3, 4, 5, 6], "one per row"),                    # ids do not cover the rows
])
def test_frame_ids_must_match_in_order(tmp_path, pred_ids, gt_ids, msg):
    _pair(tmp_path, pred_ids, gt_ids)
    with pytest.raises(T.ManifestError, match=msg):
        T.pool(_manifest(tmp_path, [ROW]), 10, 0)


def test_without_frame_ids_the_row_counts_must_agree(tmp_path):
    _pair(tmp_path, t_pred=6, t_gt=5)
    with pytest.raises(T.ManifestError, match="pred has 6 rows and gt 5"):
        T.pool(_manifest(tmp_path, [ROW]), 10, 0)


def test_missing_manifest_key_is_refused(tmp_path):
    _pair(tmp_path)
    with pytest.raises(T.ManifestError, match="missing cluster"):
        T.pool(_manifest(tmp_path, [{k: v for k, v in ROW.items() if k != "cluster"}]), 10, 0)


def test_bootstrap_is_seeded_by_cell_and_resamples_clusters():
    v, cl = [0.2, 0.9, 0.5, 0.7, 0.95, 0.4], ["a", "a", "b", "b", "c", "c"]
    a, b = T.median_ci(v, cl, 46, 500, 0), T.median_ci(v, cl, 46, 500, 0)
    assert a == b and a[0] == float(np.median(v)) and a[1] <= a[0] <= a[2]
    assert T.median_ci(v, cl, 47, 500, 0) != a
    # one cluster: the median is kept and the interval is not estimable (it would be zero-width)
    med, lo, hi = T.median_ci(v, ["x"] * 6, None, 50, 0)
    assert med == float(np.median(v)) and np.isnan(lo) and np.isnan(hi)


def _windows(tmp_path, n, clusters, scale=1.1):
    rng = np.random.default_rng(2)
    rows = []
    for i in range(n):
        gt = np.column_stack([rng.normal(0, 0.05, 8), rng.normal(0, 0.05, 8), rng.uniform(0.5, 0.8, 8)])
        np.savez(tmp_path / f"g{i}.npz", object=gt)
        np.savez(tmp_path / f"p{i}.npz", object=scale * gt + rng.normal(0, 0.01, gt.shape))
        rows.append({"group": "g", "window": f"w{i}", "cluster": clusters[i], "pred": f"p{i}.npz", "gt": f"g{i}.npz"})
    return _manifest(tmp_path, rows)


def _interval_is_honest(r):
    """The property: an interval is reported only with >= 2 clusters, and never as a zero-width stand-in."""
    if r["n_clusters"] < 2:
        return r["ci95"] is None and r["interval"].startswith("NOT ESTIMABLE") and r["reading"].startswith(
            "not determined")
    return r["ci95"] is not None and r["interval"] == "estimated"


def test_one_cluster_gives_no_interval_and_no_reading(tmp_path, capsys):
    m = _windows(tmp_path, 4, ["seq"] * 4)
    r = T.pool(m, 200, 0)["groups"]["g"]
    assert _interval_is_honest(r) and r["n_windows"] == 4 and np.isfinite(r["median_share"])
    assert "1 distinct cluster(s)" in r["interval"]
    assert T.main([str(m), "--boot", "50"]) == 0 and "interval NOT ESTIMABLE" in capsys.readouterr().out
    r2 = T.pool(_windows(tmp_path, 4, ["a", "a", "b", "b"]), 200, 0)["groups"]["g"]
    assert _interval_is_honest(r2) and r2["n_clusters"] == 2


def test_mutant_without_the_cluster_guard_is_caught(tmp_path, monkeypatch):
    """Mutant: the pre-guard estimator (an interval from one cluster). The property check must reject it."""
    m = _windows(tmp_path, 4, ["seq"] * 4)

    def unguarded(values, clusters, cell, n_boot, seed):
        v = np.asarray(values, np.float64)
        return float(np.median(v)), float(np.median(v)), float(np.median(v))     # every draw = the whole group

    monkeypatch.setattr(T, "median_ci", unguarded)
    monkeypatch.setattr(T, "interval_status", lambda clusters: "estimated")
    assert not _interval_is_honest(T.pool(m, 200, 0)["groups"]["g"])


def _write(tmp_path, pred, gt):
    np.savez(tmp_path / "gt.npz", object=np.asarray(gt, float))
    np.savez(tmp_path / "p.npz", object=np.asarray(pred, float))
    return _manifest(tmp_path, [ROW])


def _gt(n=6):
    rng = np.random.default_rng(3)
    return np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), rng.uniform(0.5, 0.8, n)])


def test_share_equals_alongray_score_on_ordinary_windows():
    import alongray

    rng = np.random.default_rng(4)
    for _ in range(20):
        g = _gt(12)
        p = 1.05 * g + rng.normal(0, 0.02, g.shape)
        b = alongray.score(p, g)["object"]
        assert abs(T.window_share(p, g)["share"] - (b.e_par / b.e_3d) ** 2) < 1e-12


def test_negative_fitted_scale_still_gets_its_share(tmp_path):
    """A prediction through the camera centre (c = -g): alongray's gauge fit refuses lam = -1; the share is defined."""
    import alongray

    g = _gt()
    with pytest.raises(ValueError):
        alongray.score(-g, g)
    r = T.pool(_write(tmp_path, -g, g), 50, 0)["groups"]["g"]
    assert r["n_windows"] == 1 and abs(r["median_share"] - 1.0) < 1e-12


def test_true_point_at_the_camera_centre_is_refused_with_its_reason(tmp_path):
    g = _gt()
    g[2] = 0.0
    r = T.pool(_write(tmp_path, 1.1 * g, g), 50, 0)["groups"]["g"]
    assert r["refused"] == 1 and r["n_windows"] == 0 and r["n_assigned"] == 1
    assert "true point at the camera centre on 1 row(s)" in r["windows_refused"][0]["reason"]


def _isotropic(n=60):
    rng = np.random.default_rng(1)
    g = np.column_stack([rng.normal(0, 0.05, n), rng.normal(0, 0.05, n), 0.5 + rng.normal(0, 0.02, n)])
    return g + rng.normal(0, 0.02, g.shape), g


def test_zero_filled_prediction_rows_refuse_the_window(tmp_path):
    """Zero-filled predicted rows push an isotropic window's share towards 1; any such row refuses the window."""
    p, g = _isotropic()
    clean = T.window_share(p, g)
    assert clean["status"] == "scored" and clean["share"] < 0.5
    for k in (1, 3, 30):
        z = p.copy()
        z[:k] = 0.0
        w = T.window_share(z, g)
        assert w["status"] == "refused" and w["share"] is None and w["n_zero_pred"] == k
        assert w["reason"].startswith(f"{k} of 60 predicted row(s) at the camera centre")
    z = p.copy()
    z[:3] = 1e-9                                                           # within the tolerance of the origin
    assert T.window_share(z, g)["status"] == "refused"
    z = p.copy()
    z[:3] = np.nan                                                         # missing frames written as NaN: dropped
    w = T.window_share(z, g)
    assert w["status"] == "scored" and w["n_nonfinite"] == 3 and w["share"] < 0.5
    z = p.copy()
    z[:3] = 0.0
    r = T.pool(_write(tmp_path, z, g), 50, 0)["groups"]["g"]
    assert r["refused"] == 1 and r["n_windows"] == 0 and r["window_flags"][0]["n_zero_pred"] == 3


def test_mutant_that_scores_zero_filled_rows_is_caught():
    """Mutant: the earlier behaviour (zero rows scored as given, alongray's warning silenced). It turns the isotropic
    window along-ray (share > 0.9 with 3 of 60 rows zero-filled), which the refusal above prevents."""
    import warnings

    import alongray

    p, g = _isotropic()
    p[:3] = 0.0

    def mutant(pred, gt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = alongray.score(pred, gt)["object"]
        return (b.e_par / b.e_3d) ** 2

    assert mutant(p, g) > 0.9                                              # the defect the guard exists for
    assert T.window_share(p, g)["status"] == "refused"


def test_reader_warnings_are_recorded_and_re_emitted(monkeypatch):
    import warnings

    import alongray

    real = alongray.reference_points

    def noisy(x, *a, **k):
        warnings.warn("core warning from the reader", UserWarning, stacklevel=2)
        return real(x, *a, **k)

    monkeypatch.setattr(T.alongray, "reference_points", noisy)
    p, g = _isotropic()
    with pytest.warns(UserWarning, match="core warning from the reader"):
        w = T.window_share(p, g)
    assert w["warnings"] == ["core warning from the reader"] and w["status"] == "scored"


def test_reader_refusal_is_recorded_not_raised(tmp_path):
    g = np.tile(np.eye(4), (5, 1, 1))
    g[:, 2, 3] = 0.6
    bad = g.copy()
    bad[:, 3, 0] = 1.0                                                     # bottom row is not 0 0 0 1
    r = T.pool(_write(tmp_path, bad, g), 50, 0)["groups"]["g"]
    assert r["refused"] == 1 and r["windows_refused"][0]["reason"].startswith("input refused by alongray's reader")


def test_reading_rule():
    assert T.reading(0.9, 0.78, 0.93) == "along-ray dominated"
    assert T.reading(0.6, 0.30, 0.70) == "not determined"
    assert T.reading(0.2, 0.10, 0.30) == "not along-ray dominated"


def _strict_json(text):
    def refuse(c):
        raise ValueError(f"non-standard JSON constant {c}")
    return SJ.loads(text, "output", parse_constant=refuse)


def test_all_refused_group_writes_strict_json(tmp_path, capsys):
    """A group whose every window is refused: null median and interval, the reason, and standard JSON (no NaN)."""
    p, g = _isotropic(20)
    p[:2] = 0.0                                                            # zero-filled rows: the window is refused
    m = _write(tmp_path, p, g)
    out = tmp_path / "out.json"
    assert T.main([str(m), "--boot", "50", "--json", str(out)]) == 0
    r = _strict_json(out.read_text())["groups"]["g"]
    assert r["median_share"] is None and r["ci95"] is None and r["n_windows"] == 0 and r["refused"] == 1
    assert r["median"] == r["interval"] == "NOT ESTIMABLE: 0 scored windows (1 refused)"
    assert "g: median share NOT ESTIMABLE interval NOT ESTIMABLE" in capsys.readouterr().out


def test_mutant_restoring_nan_is_killed_by_a_failed_test(tmp_path):
    """Mutant: the r3 behaviour (median NaN for an empty group, NaN allowed in the JSON). Running
    test_all_refused_group_writes_strict_json against it must FAIL."""
    src = SCRIPT.read_text()
    mutated = src.replace("            med = None\n", "").replace(", allow_nan=False)", ")")
    assert mutated.count("med = None") == 0 and "allow_nan=False" not in mutated
    mpath = tmp_path / "alongray_shares_mutant.py"
    mpath.write_text(mutated)
    r = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        f"{Path(__file__).resolve()}::test_all_refused_group_writes_strict_json"],
                       env={**os.environ, "ALONGRAY_SHARES_PATH": str(mpath)}, cwd=HERE, capture_output=True, text=True,
                       check=False)
    assert r.returncode == 1, r.stdout[-2000:]
    assert "FAILED" in r.stdout and "test_all_refused_group_writes_strict_json" in r.stdout


def test_repeated_manifest_key_refused(tmp_path, capsys):
    p, g = _windows_for_dupkey()
    m = _write(tmp_path, p, g)
    line = m.read_text().splitlines()[0]
    m.write_text(line[:-1] + ', "group": "other"}\n')
    assert T.main([str(m), "--boot", "20"]) == 2
    assert "manifest line 1: repeated key 'group'" in capsys.readouterr().err


def _windows_for_dupkey():
    t = np.linspace(0, 1, 10)[:, None]
    g = np.hstack([0.1 * t, 0.05 * np.ones_like(t), 0.6 + 0.1 * t])
    return g * 1.05, g
