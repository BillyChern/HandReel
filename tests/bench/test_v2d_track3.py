"""Tests for fastdo.bench.v2d_track3: quaternion convention, NaN for invisible rows, name-based object matching.

The synthetic tests always run. The real-data test runs only where the public split was downloaded
(/opt/fastdo/data/v2d_track3_public, HF revision 5f68335f) and checks the inventory facts the design relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.bench import v2d_track3 as v2d

REAL = v2d.DEFAULT_ROOT


def test_pose7_identity_and_convention() -> None:
    m = v2d.pose7_to_matrix(np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(m[:3, :3], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(m[:3, 3], [0.1, 0.2, 0.3])
    rng = np.random.default_rng(0)
    q_xyzw = Rotation.random(5, random_state=1).as_quat()
    pose = np.concatenate([rng.standard_normal((5, 3)), q_xyzw[:, [3, 0, 1, 2]]], axis=1)   # file order w-first
    np.testing.assert_allclose(v2d.pose7_to_matrix(pose)[:, :3, :3], Rotation.from_quat(q_xyzw).as_matrix(),
                               atol=1e-12)


def test_pose7_mutation_xyzw_order_is_detected() -> None:
    """Reading the file's w-first quaternion as x-first must NOT give the same rotation (the test can fire)."""
    q_xyzw = Rotation.from_euler("xyz", [30, 40, 50], degrees=True).as_quat()
    right = v2d.pose7_to_matrix(np.r_[0, 0, 0, q_xyzw[[3, 0, 1, 2]]])[:3, :3]
    wrong = v2d.pose7_to_matrix(np.r_[0, 0, 0, q_xyzw])[:3, :3]
    assert np.abs(right - wrong).max() > 0.1


def test_pose7_zero_quaternion_is_nan() -> None:
    m = v2d.pose7_to_matrix(np.zeros(7))
    assert np.isnan(m).all()


def _write_fake_split(root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"episode_indices": [7], "vicon_subjects": ["a", "b"]}))
    (root / "meta" / "episodes_metadata.jsonl").write_text(json.dumps({"episode_index": 7, "task": "T"}) + "\n")
    q = [1.0, 0.0, 0.0, 0.0]
    rows = [
        [{"name": "a", "pose": [0.1, 0, 0.6] + q, "visible": True}, {"name": "b", "pose": [0.2, 0, 0.6] + q,
                                                                     "visible": True}],
        # row order swapped (as in the real episode 11), and b invisible (all-zero pose)
        [{"name": "b", "pose": [0.0] * 7, "visible": False}, {"name": "a", "pose": [0.3, 0, 0.6] + q,
                                                               "visible": True}],
    ]
    table = pa.table({"observation.objects": rows, "timestamp": [0.0, 0.05], "frame_index": [0, 1]})
    table = table.replace_schema_metadata({"pose_convention": "world_T_object", "objects": json.dumps(["a", "b"])})
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000007.parquet")


def test_load_episode_matches_objects_by_name_and_nans_invisible(tmp_path: Path) -> None:
    _write_fake_split(tmp_path)
    ep = v2d.load_episode(7, tmp_path)
    assert ep.objects == ("a", "b") and ep.task == "T" and ep.n_frames == 2
    np.testing.assert_allclose(ep.world_T_object[1, 0, :3, 3], [0.3, 0, 0.6])   # 'a' found in slot 1 of row 1
    assert np.isnan(ep.world_T_object[1, 1]).all() and not ep.visible[1, 1]
    assert ep.meta["row_order_differs_from_metadata"] == "1"


@pytest.mark.skipif(not REAL.exists(), reason="V2D Track 3 public split not downloaded")
def test_real_public_split_inventory_facts() -> None:
    idx = v2d.episode_indices(REAL)
    assert len(idx) == 20
    total, two_obj, reordered = 0, 0, 0
    for i in idx:
        ep = v2d.load_episode(i, REAL)
        total += ep.n_frames
        two_obj += len(ep.objects) >= 2
        reordered += int(ep.meta["row_order_differs_from_metadata"]) > 0
        assert np.isfinite(ep.world_T_object[ep.visible]).all()
    assert total == 5246 and two_obj == 13
    assert reordered >= 1                                   # the README's "sorted by name" is not true everywhere
    cams, pairs, _ = v2d.load_calibration(REAL)
    ego = next(p for p in pairs if p.rig == "ego")
    assert (ego.left, ego.right) == ("ego_cam_c", "ego_cam_b") and abs(ego.baseline_m - 0.075) < 1e-3
    assert cams["ego_cam_a"].rotation_deg == 180
    verts, _ = v2d.load_mesh("white_pot", REAL)
    assert 0.3 < float(np.linalg.norm(verts.max(0) - verts.min(0))) < 0.6     # metres, not mm
