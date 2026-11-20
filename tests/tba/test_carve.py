"""R2 known answers on synthetic scenes: M-CARVE+, M-CARVE-FALSE, M-HAND and the abstain cap, each
with a mutation of the estimator under which the check must FAIL.

Scene (`tests.tba.synth.carve_scene`): an ellipsoid turning by 70 deg over 16 rows; a 3 cm sphere "hand" touches its
camera-facing side; each row's object region is the object silhouette MINUS the hand silhouette, as production masks
exclude occluding hand pixels. Evidence goes through `FrameEvidence` (object region from lifted pixels, hand
silhouettes from the stored hand vertices), as on real windows.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastdo.tba.carve import carve_template, remesh
from fastdo.tba.config import TbaConfig
from fastdo.tba.geometry import rotvec_to_matrix
from fastdo.tba.window import FrameEvidence
from tests.tba.synth import as_window, carve_scene, sdf_on_grid


@pytest.fixture(scope="module")
def scene():
    return carve_scene()                         # 70 deg: DexYCB's GT theta_eff is 47 deg (median, live frames)


@pytest.fixture(scope="module")
def wide():
    return carve_scene(theta_deg=135.0)


def _K() -> np.ndarray:
    from tests.tba.synth import K

    return K


def _grids(sc, extend: float):
    """(template grid, GT inside mask on the same lattice): the template swept `extend` x the object's depth extent
    along the keyframe view (away from the camera: the hidden side)."""
    v, f = sc.sw.mesh_v, sc.sw.mesh_f
    depth_ext = float(np.ptp(v @ sc.view_dir))
    shifts = np.outer(np.linspace(0.0, extend * depth_ext, 7), sc.view_dir)
    lo = np.minimum(v.min(0), (v + shifts[-1]).min(0)) - 0.15
    hi = np.maximum(v.max(0), (v + shifts[-1]).max(0)) + 0.15
    tmpl = sdf_on_grid(v, f, lo, hi, n=56, shifts=shifts)
    gt = sdf_on_grid(v, f, lo, hi, n=56)
    assert np.allclose(tmpl.origin, gt.origin) and tmpl.step == gt.step
    return tmpl, gt.values.reshape(-1) < 0.0


def _evidence(sc, cfg: TbaConfig) -> FrameEvidence:
    win = as_window(sc.sw, sc.sw.poses, sc.sw.scale, obj_points=sc.obj_points, hands=(sc.hand_v, sc.hand_f))
    return FrameEvidence(win, cfg)


def _removed(res, mask_flat: np.ndarray) -> float:
    """Fraction of the voxels in `mask_flat` (flat lattice) the carving ACTION removed (0 when it abstained)."""
    carved = np.zeros_like(mask_flat)
    carved[res.inside_idx[res.carved]] = True
    return float((carved & mask_flat).sum()) / max(int(mask_flat.sum()), 1)


def _perturb(poses: np.ndarray, scale: float, cm: np.ndarray, along_m: float, rot_deg: float, jitter: bool,
             seed: int) -> np.ndarray:
    """Poses moved by a constant along-ray offset, a constant rotation about the object centre (random axis), and
    optionally per-row jitter (sd 10 mm along the ray, 2 deg)."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    out = poses.copy()
    for k in range(len(poses)):
        R, t = poses[k, :3, :3], poses[k, :3, 3]
        c = scale * R @ cm + t
        u = c / np.linalg.norm(c)
        d = along_m + (rng.normal(0.0, 0.010) if jitter else 0.0)
        w = np.radians(rot_deg) * axis / np.linalg.norm(axis)
        if jitter:
            w = w + np.radians(2.0) * rng.normal(size=3) / np.sqrt(3.0)
        R2 = rotvec_to_matrix(w) @ R
        out[k, :3, :3] = R2
        out[k, :3, 3] = (c + d * u) - scale * R2 @ cm
    return out


def _carve_plus(sc, cfg: TbaConfig, rows: np.ndarray | None = None) -> tuple[float, float]:
    tmpl, gt_in = _grids(sc, 0.3)
    added = (tmpl.values.reshape(-1) < 0.0) & ~gt_in
    assert added.sum() > 0.15 * gt_in.sum()                          # the extension is real
    res = carve_template(tmpl, sc.sw.poses, sc.sw.scale, _K(), _evidence(sc, cfg), cfg, rows=rows)
    return _removed(res, added), _removed(res, gt_in)


@pytest.mark.xfail(strict=True, reason=(
    "M-CARVE+ FAILS AS REGISTERED (vote quantile q = 0.6): a hidden-side voxel projects outside the silhouette only "
    "in views turned far from the keyframe view (the far tip of a 30% extension of this ellipsoid needs ~58 deg), so a "
    "60% majority of the seeing rows never votes it out on a 70 deg sweep: 5% of the added volume is removed (15% at "
    "q 0.4, 33% at q 0.25; 2% / 41% / 80% on a 120 deg sweep, cap off). The 25% cap also abstains whenever 80% of "
    "the extension exceeds 25% of the template (here it does). Geometric, not a bug; whether to change the "
    "registered rule is undecided. strict: this flips to a failure the day the registered rule passes."))
def test_m_carve_plus_registered_rule_removes_the_hidden_side_extension(scene) -> None:
    """M-CARVE+ as registered: a template extended by 30% of its depth along the keyframe view (hidden side); carving
    at the true poses removes >= 80% of the added volume (70 deg sweep, the registered TbaConfig)."""
    removed_added, removed_true = _carve_plus(scene, TbaConfig())
    assert removed_true <= 0.02
    assert removed_added >= 0.80


def test_m_carve_plus_mechanism_with_a_wide_sweep_and_a_low_quantile(wide) -> None:
    """The carving mechanism itself: on a 135 deg sweep at vote quantile 0.25, cap off, the hidden-side extension is
    removed (>= 80%) with <= 2% of the true volume. Mutation: only the keyframe votes -> nothing of the extension goes.
    (With the registered 25% cap this template ABSTAINS: the 30%-of-depth sweep adds 47% of the ellipsoid's volume,
    so removing 80% of it is 26% of the template. The registered M-CARVE+ is incompatible with the cap whenever the
    extension exceeds ~31% of the template volume.)"""
    cfg = TbaConfig(vote_q=0.25, cap=1.0)
    removed_added, removed_true = _carve_plus(wide, cfg)
    assert removed_added >= 0.80 and removed_true <= 0.02
    assert _carve_plus(wide, TbaConfig(vote_q=0.25))[0] == 0.0          # the registered cap abstains here
    one = np.zeros(len(wide.sw.poses), bool)
    one[wide.sw.key] = True
    assert _carve_plus(wide, TbaConfig(vote_q=0.25, min_views=1, cap=1.0), rows=one)[0] < 0.80


@pytest.mark.parametrize("along_m", [-0.04, 0.04])
def test_m_carve_false_little_true_volume_removed_at_the_expected_pose_error(scene, along_m: float) -> None:
    """M-CARVE-FALSE: the GT mesh as template, poses perturbed by TBA's expected residual (constant +-40 mm along the
    ray, 5 deg rotation, per-row jitter 10 mm / 2 deg), hand-UNKNOWN active: removed TRUE volume <= 5% on >= 90% of
    perturbation draws (10 seeds)."""
    cfg = TbaConfig()
    gt_grid, gt_in = _grids(scene, 0.0)
    ev = _evidence(scene, cfg)
    cm = scene.sw.mesh_v.mean(0)
    fr = [_removed(carve_template(gt_grid, _perturb(scene.sw.poses, scene.sw.scale, cm, along_m, 5.0, True, s),
                                  scene.sw.scale, _K(), ev, cfg), gt_in) for s in range(10)]
    assert np.mean(np.asarray(fr) <= 0.05) >= 0.9, fr
    if along_m > 0:
        return            # moved AWAY the template projects smaller than the region: no rule can carve false volume
    # mutation: a single 'outside' vote carves (vote quantile 0.05, cap off) -> true volume goes and the check fires
    bad = TbaConfig(vote_q=0.05, min_views=1, cap=1.0)
    fb = [_removed(carve_template(gt_grid, _perturb(scene.sw.poses, scene.sw.scale, cm, along_m, 5.0, True, s),
                                  scene.sw.scale, _K(), _evidence(scene, bad), bad), gt_in) for s in range(3)]
    assert np.mean(np.asarray(fb) <= 0.05) < 0.9, fb


def test_m_hand_the_unknown_rule_protects_the_grasp_region(scene) -> None:
    """M-HAND: GT template at the true poses. With the hand-UNKNOWN rule the true volume within 2 cm of the hand is
    kept; with it disabled, the removed true volume near the grasp rises measurably (the rule is doing work)."""
    gt_grid, gt_in = _grids(scene, 0.0)
    pts = gt_grid.centres().reshape(-1, 3)
    near = gt_in & (np.linalg.norm(pts - scene.hand_canon, axis=1) <= scene.hand_radius + 0.02 / scene.sw.scale)
    assert near.sum() > 50
    on, off = TbaConfig(), TbaConfig(hand_unknown=False)
    r_on = carve_template(gt_grid, scene.sw.poses, scene.sw.scale, _K(), _evidence(scene, on), on)
    r_off = carve_template(gt_grid, scene.sw.poses, scene.sw.scale, _K(), _evidence(scene, off), off)
    assert _removed(r_on, near) <= 0.01
    assert _removed(r_off, near) >= 0.05 + _removed(r_on, near)


def test_the_cap_abstains_and_a_carved_template_remeshes_watertight(wide) -> None:
    """A template inflated 1.6x (most of it over-extended) would lose more than the cap: carving ABSTAINS (template
    unchanged); mutation: cap 1.0 applies it. A carved template (135 deg sweep, q 0.25, cap off) re-meshes
    watertight with the kept voxel volume, close to the true volume."""
    import trimesh

    v, f = wide.sw.mesh_v, wide.sw.mesh_f
    lo, hi = 1.6 * v.min(0) - 0.15, 1.6 * v.max(0) + 0.15
    big = sdf_on_grid(1.6 * v, f, lo, hi, n=56)
    cfg = TbaConfig(vote_q=0.25)
    res = carve_template(big, wide.sw.poses, wide.sw.scale, _K(), _evidence(wide, cfg), cfg)
    assert res.removed_frac > cfg.cap and res.abstained and not res.applied and not res.carved.any()
    loose = TbaConfig(vote_q=0.25, cap=1.0)
    assert carve_template(big, wide.sw.poses, wide.sw.scale, _K(), _evidence(wide, loose), loose).applied
    tmpl, gt_in = _grids(wide, 0.3)
    res = carve_template(tmpl, wide.sw.poses, wide.sw.scale, _K(), _evidence(wide, loose), loose)
    assert res.applied
    V, F = remesh(tmpl, res)
    m = trimesh.Trimesh(V, F, process=True)
    assert m.is_watertight and m.volume > 0
    kept = int((tmpl.values.reshape(-1) < 0).sum() - res.carved.sum())
    assert m.volume == pytest.approx(kept * tmpl.step ** 3, rel=0.1)
    gt_vol = gt_in.sum() * tmpl.step ** 3
    assert abs(m.volume - gt_vol) < 0.1 * gt_vol
