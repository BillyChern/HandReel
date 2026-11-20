"""The `contact` stage worker (the promoted correction): artifact contract, side choice, refusal to be stale."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastdo.engine.workers_contact import (
    CONTACT_FILE,
    ContactStageConfig,
    make_contact_worker,
    read_contact_artifact,
)


def _clip(tmp_path: Path, *, left_far: bool = True, no_hand: bool = False, t: int = 4) -> Path:
    """A clip directory with the three artifacts the stage reads: an object at 0.6 m, a right hand on it."""
    d = tmp_path / "clip"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    sphere = rng.normal(size=(600, 3))
    points = (0.04 * sphere / np.linalg.norm(sphere, axis=1, keepdims=True)).astype(np.float32)
    np.savez(d / "samples.npz", points=points, normals=points)

    poses = np.tile(np.eye(4), (t, 1, 1))
    poses[:, :3, 3] = np.array([0.0, 0.0, 0.60])
    np.savez(d / "solve.npz", poses=poses, scale=np.array(1.0), frame_idx=np.arange(t, dtype=np.int64))

    right = np.tile(np.array([0.0, 0.0, 0.665]), (t, 21, 1))     # 25 mm beyond the sphere's near surface
    left = np.tile(np.array([0.3, 0.2, 0.60]), (t, 21, 1)) if left_far else right.copy()
    np.savez(d / "hands.npz", right_joints=right, right_valid=np.ones(t, bool) & (not no_hand),
             left_joints=left, left_valid=np.ones(t, bool) & (not no_hand), frame_idx=np.arange(t, dtype=np.int64))
    return d


def test_the_artifact_has_its_documented_contract(tmp_path: Path) -> None:
    d = _clip(tmp_path)
    make_contact_worker(ContactStageConfig(enabled=True))("clip", d)

    z = np.load(d / CONTACT_FILE)
    assert set(z) == {"poses", "shift_mm", "frame_idx", "side", "n_corrected", "solve_sha256", "constants",
                      "centroid_obj"}
    assert z["poses"].shape == (4, 4, 4) and z["shift_mm"].shape == (4,)
    assert str(z["side"]) == "right" and int(z["n_corrected"]) == 4
    assert np.all(z["shift_mm"] != 0)
    assert json.loads(str(z["constants"]))["tau_mm"] == 20.0
    solved = np.load(d / "solve.npz")["poses"]
    moved = z["poses"][:, :3, 3] - solved[:, :3, 3]
    # the motion is along the ray to the object's CENTROID (not the optical axis: a sampled surface's centroid
    # sits a millimetre or so off the pose origin, and the correction follows the centroid ray by construction)
    ray = solved[0, :3, :3] @ z["centroid_obj"] + solved[0, :3, 3]
    ray = ray / np.linalg.norm(ray)
    for row in moved:
        assert np.linalg.norm(row - (row @ ray) * ray) < 1e-12
    assert np.all(np.abs(moved @ ray) > 0)


def test_auto_picks_the_hand_that_is_touching(tmp_path: Path) -> None:
    d = _clip(tmp_path, left_far=True)
    make_contact_worker(ContactStageConfig(enabled=True, hands="auto"))("clip", d)
    assert str(np.load(d / CONTACT_FILE)["side"]) == "right"


def test_a_forced_side_is_used_even_when_it_is_the_far_hand(tmp_path: Path) -> None:
    d = _clip(tmp_path, left_far=True)
    make_contact_worker(ContactStageConfig(enabled=True, hands="left"))("clip", d)
    z = np.load(d / CONTACT_FILE)
    assert str(z["side"]) == "left"
    # a hand 0.36 m from the object is uninformative: the shift must fall back toward the prior, not track it
    assert np.all(np.abs(z["shift_mm"]) < 60.0)


def test_no_valid_hand_is_not_a_failure(tmp_path: Path) -> None:
    d = _clip(tmp_path, no_hand=True)
    make_contact_worker(ContactStageConfig(enabled=True))("clip", d)
    z = np.load(d / CONTACT_FILE)
    assert str(z["side"]) == "" and int(z["n_corrected"]) == 0
    assert np.array_equal(z["poses"], np.load(d / "solve.npz")["poses"])


def test_a_row_mismatch_raises(tmp_path: Path) -> None:
    d = _clip(tmp_path)
    h = dict(np.load(d / "hands.npz"))
    h["right_joints"] = h["right_joints"][:3]
    h["right_valid"] = h["right_valid"][:3]
    np.savez(d / "hands.npz", **h)
    with pytest.raises(ValueError, match="rows"):
        make_contact_worker(ContactStageConfig(enabled=True))("clip", d)


def test_the_worker_is_deterministic(tmp_path: Path) -> None:
    d = _clip(tmp_path)
    w = make_contact_worker(ContactStageConfig(enabled=True))
    w("clip", d)
    first = (d / CONTACT_FILE).read_bytes()
    w("clip", d)
    assert (d / CONTACT_FILE).read_bytes() == first


def test_a_stale_artifact_is_refused_not_consumed(tmp_path: Path) -> None:
    """A re-solved clip must not have the old correction applied to its new poses."""
    d = _clip(tmp_path)
    make_contact_worker(ContactStageConfig(enabled=True))("clip", d)
    assert read_contact_artifact(d) is not None

    s = dict(np.load(d / "solve.npz"))
    s["poses"] = s["poses"] + 0.01                                  # the solve changed; the correction did not
    np.savez(d / "solve.npz", **s)
    assert read_contact_artifact(d) is None


def test_the_config_rejects_nonsense(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cell"):
        ContactStageConfig(cell="made_up")
    with pytest.raises(ValueError, match="hands"):
        ContactStageConfig(hands="both")
    with pytest.raises(ValueError, match="grid_mm"):
        ContactStageConfig(grid_mm=(300.0, -300.0, 2.0))
    with pytest.raises(ValueError, match="unknown keys"):
        ContactStageConfig.from_mapping({"enabled": True, "typo": 1})
    assert ContactStageConfig.from_mapping(None) == ContactStageConfig()
    assert ContactStageConfig.from_mapping({"grid_mm": [-120, 120, 2]}).grid()[0] == -120.0


def test_a_null_constants_path_means_the_packaged_file(tmp_path: Path) -> None:
    """The Hydra config writes `constants: null` for "use the shipped production fit"; a real run caught this."""
    from fastdo.engine.workers_contact import DEFAULT_CONSTANTS

    for cfg in (ContactStageConfig.from_mapping({"enabled": True, "constants": None}),
                ContactStageConfig.from_mapping({"enabled": True, "constants": ""}),
                ContactStageConfig(enabled=True)):
        assert cfg.constants_path() == DEFAULT_CONSTANTS
        assert cfg.load().tau_mm == 20.0

    d = _clip(tmp_path)
    make_contact_worker(ContactStageConfig.from_mapping({"enabled": True, "constants": None}))("clip", d)
    assert str(np.load(d / CONTACT_FILE)["side"]) == "right"


def test_a_left_only_clip_gets_the_left_hand_and_exactly_the_right_handed_correction(tmp_path: Path) -> None:
    """`hands: auto` on a clip whose only valid hand is the LEFT (what `geometry_stage.hand_sides: auto` now lets
    through) corrects with the left hand, and the correction has no side-specific term: the production constants are
    one side-POOLED fit (pooled LOCO beat per-side; the `prod` cell's 75 fit clips are 40 left / 35 right), so the
    same joints give the same shifts and the same constants whichever side label they carry."""
    (tmp_path / "r").mkdir()
    (tmp_path / "l").mkdir()
    right_dir = _clip(tmp_path / "r", left_far=True)
    make_contact_worker(ContactStageConfig(enabled=True, hands="auto"))("clip", right_dir)
    left_dir = _clip(tmp_path / "l", left_far=True)
    h = dict(np.load(left_dir / "hands.npz"))
    h["left_joints"], h["left_valid"] = h["right_joints"].copy(), h["right_valid"].copy()
    h["right_valid"] = np.zeros_like(h["right_valid"])
    np.savez(left_dir / "hands.npz", **h)
    make_contact_worker(ContactStageConfig(enabled=True, hands="auto"))("clip", left_dir)

    r, l = np.load(right_dir / CONTACT_FILE), np.load(left_dir / CONTACT_FILE)
    assert str(r["side"]) == "right" and str(l["side"]) == "left"
    assert int(l["n_corrected"]) == int(r["n_corrected"]) == 4
    for key in ("poses", "shift_mm", "frame_idx", "centroid_obj", "constants"):
        assert l[key].tobytes() == r[key].tobytes(), key


def test_the_248_options_are_off_by_default_and_in_the_shipped_config() -> None:
    """`bias_free` and `touch_gate_mm` stay off, in the dataclass AND in run/conf/config.yaml (the shipped ruler
    is the geometric one, so it is no longer the dataclass default; see the test below)."""
    from omegaconf import OmegaConf

    assert ContactStageConfig().is_m88_default_ruler()
    conf = OmegaConf.load(Path(__file__).resolve().parents[2] / "run/conf/config.yaml")
    shipped = ContactStageConfig.from_mapping(OmegaConf.to_container(conf.contact_stage))
    assert shipped.bias_free is False and shipped.touch_gate_mm is None


def test_bias_free_writes_zero_bias_constants_and_the_extra_keys(tmp_path: Path) -> None:
    d = _clip(tmp_path)
    make_contact_worker(ContactStageConfig(enabled=True))("clip", d)
    m88 = dict(np.load(d / CONTACT_FILE))
    make_contact_worker(ContactStageConfig(enabled=True, bias_free=True))("clip", d)
    z = np.load(d / CONTACT_FILE)
    c = json.loads(str(z["constants"]))
    assert c["b_mm"] == 0.0 and c["mu_mm"] == 0.0
    assert c["a"] == json.loads(str(m88["constants"]))["a"]
    assert set(z) == set(m88) | {"min_cost_mm", "gated", "bias_free", "touch_gate_mm"}
    assert bool(z["bias_free"]) and np.isnan(float(z["touch_gate_mm"]))
    assert not np.array_equal(z["shift_mm"], m88["shift_mm"])
    assert not z["gated"].any() and np.all(np.isfinite(z["min_cost_mm"]))


def test_the_touch_gate_leaves_a_non_touching_hand_uncorrected(tmp_path: Path) -> None:
    """The left hand sits ~0.3 m to the side of the object: at a 50 mm gate no frame is corrected."""
    d = _clip(tmp_path)
    make_contact_worker(ContactStageConfig(enabled=True, hands="left", touch_gate_mm=50))("clip", d)
    z = np.load(d / CONTACT_FILE)
    solved = np.load(d / "solve.npz")["poses"]
    assert int(z["n_corrected"]) == 0 and z["gated"].all()
    assert np.array_equal(z["poses"], solved) and float(z["touch_gate_mm"]) == 50.0
    assert np.all(z["min_cost_mm"] > 50.0)
    make_contact_worker(ContactStageConfig(enabled=True, hands="right", touch_gate_mm=50))("clip", d)
    z = np.load(d / CONTACT_FILE)
    assert int(z["n_corrected"]) == 4 and not z["gated"].any()


def test_the_248_options_reject_nonsense() -> None:
    with pytest.raises(ValueError, match="touch_gate_mm"):
        ContactStageConfig(touch_gate_mm=-1.0)
    with pytest.raises(ValueError, match="bias_free"):
        ContactStageConfig(bias_free="yes")  # type: ignore[arg-type]
    assert ContactStageConfig.from_mapping({"touch_gate_mm": 30}).touch_gate_mm == 30.0


def test_the_267b_geometric_ruler_is_the_shipped_config_and_contact_stays_off_by_default() -> None:
    """The shipped config uses the geometric ruler; the dataclass default stays the posterior-mean ruler; contact itself
    stays disabled by default (third-person runs enable it; the egocentric profile forces it off)."""
    from omegaconf import OmegaConf

    assert ContactStageConfig().geometric is False
    conf = OmegaConf.load(Path(__file__).resolve().parents[2] / "run/conf/config.yaml")
    shipped = ContactStageConfig.from_mapping(OmegaConf.to_container(conf.contact_stage))
    assert shipped.geometric is True and shipped.enabled is False
    assert not shipped.is_m88_default_ruler()
    c = shipped.load()
    assert (c.a, c.b_mm, c.mu_mm) == (1.0, 0.0, 0.0)


def test_geometric_is_267a_offline_definition_and_writes_its_flag(tmp_path: Path) -> None:
    """a = 1, b = mu = 0, sig and tau from the file: exactly the offline `replace(base.bias_free(), a=1.0)`."""
    import dataclasses

    base = ContactStageConfig().load()
    geo = ContactStageConfig(geometric=True).load()
    assert geo == dataclasses.replace(base.bias_free(), a=1.0)
    assert (geo.a, geo.b_mm, geo.mu_mm, geo.sig_mm, geo.tau_mm) == (1.0, 0.0, 0.0, base.sig_mm, base.tau_mm)
    assert ContactStageConfig(geometric=True, bias_free=True).load() == geo
    d = _clip(tmp_path)
    make_contact_worker(ContactStageConfig(enabled=True))("clip", d)
    m88 = dict(np.load(d / CONTACT_FILE))
    make_contact_worker(ContactStageConfig(enabled=True, geometric=True))("clip", d)
    z = np.load(d / CONTACT_FILE)
    c = json.loads(str(z["constants"]))
    assert (c["a"], c["b_mm"], c["mu_mm"]) == (1.0, 0.0, 0.0)
    assert set(z) == set(m88) | {"min_cost_mm", "gated", "bias_free", "touch_gate_mm", "geometric"}
    assert bool(z["geometric"]) and not bool(z["bias_free"])
    with pytest.raises(ValueError, match="geometric"):
        ContactStageConfig(geometric="yes")  # type: ignore[arg-type]
    assert ContactStageConfig.from_mapping({"geometric": True}).geometric is True
