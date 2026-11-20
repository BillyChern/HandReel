"""Tests for `fastdo.robot.rest`: resting faces as gravity candidates and the support check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from fastdo.robot.rest import stable_ups, support_inside
from tests.robot import synthetic

_C2W = np.eye(4)
_C2W[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]       # camera looks along world +x, level


def test_stable_ups_keep_only_faces_near_a_looking_down_camera() -> None:
    """Known answer: a 10 x 6 x 4 cm box at identity, a level camera looking along +x. Head-pitch ups run from +z
    (level) to 75 deg toward -x (looking down). Faces whose up is +z (the 10 x 6 face, most probable) or -x (15 deg
    from the 75 deg prior) qualify; +x, +-y and -z are 90+ deg away from every prior."""
    box = trimesh.creation.box(extents=[0.10, 0.06, 0.04])
    ups = stable_ups(box, np.eye(3), _C2W)
    dirs = [np.round(u.up_world, 6).tolist() for u in ups]
    assert dirs == [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
    assert ups[0].probability > ups[1].probability and ups[0].prior_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert ups[1].prior_angle_deg == pytest.approx(15.0, abs=1e-6)
    narrow = stable_ups(box, np.eye(3), _C2W, max_prior_deg=10.0)
    assert [np.round(u.up_world, 6).tolist() for u in narrow] == [[0.0, 0.0, 1.0]]
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
    turned = stable_ups(box, rot, _C2W)                                  # the object frame turned: up follows it
    assert np.allclose(turned[0].up_world, [0.0, 0.0, 1.0]) and np.allclose(turned[0].up_object, rot.T @ [0, 0, 1])


def test_support_inside_on_a_resting_face() -> None:
    hull = trimesh.creation.box(extents=[0.10, 0.06, 0.04])
    pose = np.eye(4)
    assert support_inside(hull.vertices, np.zeros(3), pose, 0.003) == pytest.approx(0.03)
    assert support_inside(hull.vertices, np.array([0.04, 0.0, 0.0]), pose, 0.003) == pytest.approx(0.01)
    assert support_inside(hull.vertices, np.array([0.07, 0.0, 0.0]), pose, 0.003) == pytest.approx(-0.02)


def test_explicit_up_face_is_used_and_reported(g1, tmp_path: Path) -> None:
    """On the synthetic clip (a 6 cm cube at identity, camera along +x), resting face 0 is the cube's top face
    (up +z, the record's own up): the explicit choice reproduces the record-up placement and names the face in the
    adaptations; an index past the candidates is refused."""
    from fastdo.robot.kinematic_path import (
        KinematicPathConfig,
        load_reconstruction,
        run_kinematic_path,
    )
    recon = load_reconstruction(synthetic.write_clip(g1, tmp_path / "clip"))
    base = run_kinematic_path(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1)
    assert len(base.stable_ups) >= 1 and np.allclose(base.stable_ups[0].up_world, [0, 0, 1], atol=1e-6)
    face = run_kinematic_path(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY, up_face=0), g1)
    assert face.choice.up_face == 0 and face.up_source.startswith("up_face=0")
    assert np.allclose(face.placement.T_rv, base.placement.T_rv, atol=1e-9)
    assert any("resting hull face 0" in a for a in face.adaptations)
    side = run_kinematic_path(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY, up_face=1), g1)
    assert np.allclose(side.stable_ups[1].up_world, [-1, 0, 0], atol=1e-6)
    assert np.allclose(side.placement.T_rv[:3, :3] @ [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-9)   # -x is up now
    with pytest.raises(ValueError, match="resting-face candidates"):
        run_kinematic_path(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY, up_face=99), g1)
