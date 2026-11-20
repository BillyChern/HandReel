"""Shared fixtures of the robot-path tests: the verified official G1 + Dex3-1 model (skipped when the pinned
unitree_ros clone is not on this machine)."""

from __future__ import annotations

import pytest

from fastdo.robot.g1_model import G1Upper, load_g1_upper, resolve_unitree_ros


@pytest.fixture(scope="session")
def unitree_root():
    try:
        return resolve_unitree_ros()
    except FileNotFoundError as exc:
        pytest.skip(f"official unitree_ros clone not available: {exc}")


@pytest.fixture(scope="session")
def g1(unitree_root) -> G1Upper:
    return load_g1_upper(unitree_root)


@pytest.fixture(scope="session")
def built(g1, tmp_path_factory):
    """`(clip dir, Reconstruction, KinematicPathResult, package dir)` of the synthetic clip (`tests/robot/synthetic.py`)."""
    from fastdo.robot.kinematic_path import (
        KinematicPathConfig,
        load_reconstruction,
        run_kinematic_path,
    )
    from fastdo.robot.package import write_package
    from tests.robot import synthetic

    clip = synthetic.write_clip(g1, tmp_path_factory.mktemp("clip") / "synthetic")
    recon = load_reconstruction(clip)
    result = run_kinematic_path(recon, KinematicPathConfig(table_xy=synthetic.OBJECT_XY), g1)
    pkg = write_package(tmp_path_factory.mktemp("pkg") / "package", result, recon)
    return clip, recon, result, pkg
