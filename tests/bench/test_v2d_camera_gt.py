"""Known-answer and mutation tests for fastdo.bench.v2d_camera_gt (V2D camera-frame GT construction).

Every check is paired with a mutation it must detect (feedback-verify-physically-not-by-proxy, question 2): swapped
stereo roles must be refused, a wrong ICP basin must show as low coverage, and a 5% metric error in the stereo cloud
(a baseline/focal mistake) must raise the registration residual above the noise-only fit.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastdo.bench import v2d_track3 as v2d
from fastdo.bench.metric_frame import along_ray_components
from fastdo.bench.v2d_camera_gt import (
    camera_frame_poses,
    device_ruler,
    ego_rig,
    latency_by_agreement,
    points_from_disparity,
    rectify_points,
    register_scene,
    stereo_rectification,
)

REAL = v2d.DEFAULT_ROOT


def _cam(name: str, f: float, dist: np.ndarray | None = None) -> v2d.CameraCalib:
    k = np.array([[f, 0.0, 640.0], [0.0, f, 400.0], [0.0, 0.0, 1.0]])
    return v2d.CameraCalib(name=name, resolution=(1280, 800), K=k, distortion_model="Perspective",
                           distortion=np.zeros(14) if dist is None else dist, hfov_deg=127.0, rotation_deg=0,
                           extrinsics_to_stereo_left=None)


def _right_T_left(baseline: float = 0.075, seed: int = 0) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", np.random.default_rng(seed).normal(0, 0.8, 3), degrees=True).as_matrix()
    t[:3, 3] = [-baseline, 0.0005, 0.0002]
    return t


def _project(x_cam: np.ndarray, cam: v2d.CameraCalib) -> np.ndarray:
    import cv2

    uv, _ = cv2.projectPoints(x_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.distortion)
    return uv.reshape(-1, 2)


def _scene_points(n: int = 300, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.35, 1.4, n)
    xy = rng.uniform(-0.45, 0.45, (n, 2)) * z[:, None]          # within ~+-24 deg of the axis
    return np.column_stack([xy, z])


def _triangulate_known_answer(left: v2d.CameraCalib, right: v2d.CameraCalib, r_t_l: np.ndarray) -> float:
    rect = stereo_rectification(left, right, r_t_l)
    x_l = _scene_points()
    x_r = x_l @ r_t_l[:3, :3].T + r_t_l[:3, 3]
    ul = rectify_points(_project(x_l, left), left, rect.R1, rect.P1)
    ur = rectify_points(_project(x_r, right), right, rect.R2, rect.P2)
    assert np.max(np.abs(ul[:, 1] - ur[:, 1])) < 1e-3                  # rectified rows agree
    rec = points_from_disparity(ul, ul[:, 0] - ur[:, 0], rect)
    return float(np.max(np.linalg.norm(rec - x_l, axis=1)))


def test_rectified_triangulation_recovers_points_pinhole() -> None:
    err = _triangulate_known_answer(_cam("L", 565.0), _cam("R", 568.0), _right_T_left())
    assert err < 1e-6


def test_rectified_triangulation_with_rational_distortion() -> None:
    dist = np.zeros(14)
    dist[[0, 1, 4, 5, 6, 7]] = [4.69, 1.31, 0.015, 5.06, 2.73, 0.159]      # ego_cam_c's rational terms (file order)
    err = _triangulate_known_answer(_cam("L", 563.0, dist), _cam("R", 567.0, dist), _right_T_left(seed=2))
    assert err < 1e-5


def test_swapped_left_right_is_refused() -> None:
    """Mutation: hand the pair over as (right, left) with the same transform -> disparities change sign."""
    left, right, r_t_l = _cam("L", 565.0), _cam("R", 568.0), _right_T_left()
    rect = stereo_rectification(left, right, r_t_l)
    x_l = _scene_points()
    x_r = x_l @ r_t_l[:3, :3].T + r_t_l[:3, 3]
    ul = rectify_points(_project(x_r, right), left, rect.R1, rect.P1)     # the right image read as the left one
    ur = rectify_points(_project(x_l, left), right, rect.R2, rect.P2)
    with pytest.raises(ValueError):
        points_from_disparity(ul, ul[:, 0] - ur[:, 0], rect)


@pytest.mark.skipif(not REAL.exists(), reason="V2D Track 3 public split not downloaded")
def test_real_ego_rig_conventions_and_triangulation() -> None:
    cams, pairs, _ = v2d.load_calibration(REAL)
    rig = ego_rig(cams, pairs)
    assert rig.left.name == "ego_cam_c" and rig.right.name == "ego_cam_b"
    assert rig.right_T_left[0, 3] < -0.07                                # OpenCV convention: right at x = -B
    assert abs(rig.a_T_left[0, 3]) == pytest.approx(0.038, abs=0.003)   # RGB camera midway between the pair
    rect = stereo_rectification(rig.left, rig.right, rig.right_T_left)
    assert rect.baseline_m == pytest.approx(rig.baseline_m, rel=0.01)
    assert _triangulate_known_answer(rig.left, rig.right, rig.right_T_left) < 1e-4


# ---------------------------------------------------------------- scene registration

def _box_samples(extent: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    axis = rng.integers(0, 3, n)
    sign = rng.choice([-1.0, 1.0], n)
    p = rng.uniform(-0.5, 0.5, (n, 3)) * extent
    p[np.arange(n), axis] = sign * 0.5 * extent[axis]
    nrm = np.zeros((n, 3))
    nrm[np.arange(n), axis] = sign
    return p, nrm


def _scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two 'scans' posed in a mocap world (a pot-sized box on a table and a thin piece), z up, ~0.6 m high."""
    p1, n1 = _box_samples(np.array([0.19, 0.13, 0.12]), 4000, seed)
    p2, n2 = _box_samples(np.array([0.14, 0.03, 0.22]), 3000, seed + 1)
    r2 = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    p1 = p1 + [0.05, -0.02, 0.66]
    p2 = p2 @ r2.T + [-0.16, 0.10, 0.72]
    return np.vstack([p1, p2]), np.vstack([n1, n2 @ r2.T])


def _cam_T_world() -> np.ndarray:
    """A head camera ~0.55 m above and behind the table looking down ~50 deg (camera z forward, y down)."""
    look = Rotation.from_euler("x", -140, degrees=True).as_matrix()     # world -> camera rotation
    t = np.eye(4)
    t[:3, :3] = look
    t[:3, 3] = -look @ np.array([0.0, -0.45, 1.25])
    return t


def _observed_cloud(t: np.ndarray, noise: float, clutter: int, scale: float = 1.0, seed: int = 3) -> np.ndarray:
    mp, mn = _scene()
    pc = mp @ t[:3, :3].T + t[:3, 3]
    nc = mn @ t[:3, :3].T
    vis = np.einsum("ij,ij->i", nc, -pc) > 0.0                         # back-face culling (convex parts)
    rng = np.random.default_rng(seed)
    x = pc[vis] + rng.normal(0.0, noise, (int(vis.sum()), 3))
    lo, hi = x.min(0) - 0.15, x.max(0) + 0.15
    junk = rng.uniform(lo, hi, (clutter, 3))                            # hands, table, background
    return np.vstack([x, junk]) * scale


def _perturb(t: np.ndarray, deg: float, metres: float, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = np.eye(4)
    d[:3, :3] = Rotation.from_rotvec(np.radians(deg) * rng.normal(size=3) / np.sqrt(3)).as_matrix()
    d[:3, 3] = metres * rng.normal(size=3) / np.sqrt(3)
    return d @ t


def _pose_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rel = a @ np.linalg.inv(b)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1))))
    return ang, float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def test_register_scene_recovers_cam_T_world_with_noise_and_clutter() -> None:
    truth = _cam_T_world()
    mp, mn = _scene()
    cloud = _observed_cloud(truth, noise=0.002, clutter=3000)
    fit = register_scene(mp, mn, cloud, _perturb(truth, 6.0, 0.03))
    ang, dt = _pose_err(fit.cam_T_world, truth)
    assert fit.converged and ang < 0.1 and dt < 0.002                 # 2 mm noise: ~1 mm, ~0.04 deg measured
    assert fit.cloud_scale == pytest.approx(1.0, abs=0.003)            # NN bias ~0.2% (~1.3 mm at 0.7 m)
    assert fit.model_coverage > 0.8 and fit.inlier_rms_m < 0.003


def test_register_scene_is_exact_on_noiseless_boxes() -> None:
    """Point-to-plane alone leaves box corners scale-invariant; the point-to-point term must make this exact."""
    truth = _cam_T_world()
    mp, mn = _scene()
    fit = register_scene(mp, mn, _observed_cloud(truth, 0.0, 0), _perturb(truth, 6.0, 0.03))
    ang, dt = _pose_err(fit.cam_T_world, truth)
    assert ang < 1e-4 and dt < 1e-6 and fit.cloud_scale == pytest.approx(1.0, abs=1e-6)


def test_point_to_plane_alone_is_not_enough_with_free_scale() -> None:
    """Mutation: point_weight = 0 (pure point-to-plane) must fail the noiseless known answer (the term is needed)."""
    truth = _cam_T_world()
    mp, mn = _scene()
    fit = register_scene(mp, mn, _observed_cloud(truth, 0.0, 0), _perturb(truth, 6.0, 0.03), point_weight=0.0)
    assert abs(fit.cloud_scale - 1.0) > 0.01 or _pose_err(fit.cam_T_world, truth)[1] > 0.005


def test_register_scene_wrong_basin_shows_low_coverage() -> None:
    """Mutation: an initialisation 90 deg off must not look like a good fit (the coverage check fires)."""
    truth = _cam_T_world()
    mp, mn = _scene()
    cloud = _observed_cloud(truth, noise=0.002, clutter=3000)
    good = register_scene(mp, mn, cloud, _perturb(truth, 6.0, 0.03))
    bad = register_scene(mp, mn, cloud, _perturb(truth, 90.0, 0.15, seed=11))
    assert _pose_err(bad.cam_T_world, truth)[1] > 0.01
    assert bad.model_coverage < good.model_coverage - 0.2


def test_register_scene_calibrates_a_stereo_scale_error_to_the_scans() -> None:
    """A 5% stereo ruler error (wrong baseline or focal) is recovered as cloud_scale ~ 1/1.05: the GT inherits the
    Vicon scans' metric, not the stereo baseline's."""
    truth = _cam_T_world()
    mp, mn = _scene()
    fit = register_scene(mp, mn, _observed_cloud(truth, 0.002, 3000, scale=1.05), _perturb(truth, 6.0, 0.03))
    assert fit.cloud_scale == pytest.approx(1.0 / 1.05, rel=0.003)
    ang, dt = _pose_err(fit.cam_T_world, truth)
    assert ang < 0.15 and dt < 0.002


def test_rigid_only_fit_turns_a_stereo_scale_error_into_depth_error() -> None:
    """Mutation: holding the scale fixed cannot absorb a 5% ruler error; it surfaces as > 1 cm of translation (the
    reason the GT construction keeps the scale free)."""
    truth = _cam_T_world()
    mp, mn = _scene()
    fit = register_scene(mp, mn, _observed_cloud(truth, 0.002, 3000, scale=1.05), _perturb(truth, 6.0, 0.03),
                         estimate_scale=False)
    assert fit.cloud_scale == 1.0
    assert _pose_err(fit.cam_T_world, truth)[1] > 0.01


def test_camera_frame_gt_scores_a_gauge_error_exactly() -> None:
    """End to end: GT camera poses from the construction; a prediction off by g_lambda scores (lambda-1)|c|."""
    truth = _cam_T_world()
    w = np.tile(np.eye(4), (4, 1, 1, 1))
    w[:, 0, :3, 3] = [[0.05, -0.02, 0.66], [0.06, -0.01, 0.70], [0.08, 0.0, 0.74], [0.09, 0.02, 0.78]]
    cam = camera_frame_poses(np.tile(truth, (4, 1, 1)), w)[:, 0]
    pred = cam.copy()
    pred[:, :3, 3] *= 1.1
    ar = along_ray_components(pred[:, :3, 3], cam[:, :3, 3])
    np.testing.assert_allclose(ar.signed_along, 0.1 * np.linalg.norm(cam[:, :3, 3], axis=1), atol=1e-12)
    np.testing.assert_allclose(ar.perp, 0.0, atol=1e-12)


# ---------------------------------------------------------------- one ruler per device, latency ---------------

_POT_AXIS_BASE = np.array([0.05, -0.02, 0.61])
_POT_H = 0.13


def _pot(n: int = 6000, seed: int = 0, r: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """A closed-top cylinder (white_pot-like: curved, rotationally symmetric) standing on a 0.61 m table."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    side = rng.uniform(0.0, 1.0, n) < 0.8
    p, nrm = np.zeros((n, 3)), np.zeros((n, 3))
    p[side] = np.column_stack([r * np.cos(th[side]), r * np.sin(th[side]), rng.uniform(0.0, _POT_H, side.sum())])
    nrm[side] = np.column_stack([np.cos(th[side]), np.sin(th[side]), np.zeros(side.sum())])
    rr = r * np.sqrt(rng.uniform(0.0, 1.0, (~side).sum()))
    p[~side] = np.column_stack([rr * np.cos(th[~side]), rr * np.sin(th[~side]), np.full((~side).sum(), _POT_H)])
    nrm[~side] = [0.0, 0.0, 1.0]
    return p + _POT_AXIS_BASE, nrm


def _occluded_cloud(t: np.ndarray, mp: np.ndarray, mn: np.ndarray, noise: float, clutter: int, scale: float,
                    seed: int, occlude: bool = True) -> np.ndarray:
    """Back-face culled, the image-left 40% of the visible object removed (a grasping hand), noise and clutter."""
    pc = mp @ t[:3, :3].T + t[:3, 3]
    vis = np.einsum("ij,ij->i", mn @ t[:3, :3].T, -pc) > 0.0
    u = pc[:, 0] / pc[:, 2]
    if occlude:
        vis &= u >= np.quantile(u[vis], 0.4)
    rng = np.random.default_rng(seed)
    x = pc[vis] + rng.normal(0.0, noise, (int(vis.sum()), 3))
    lo, hi = x.min(0) - 0.15, x.max(0) + 0.15
    return np.vstack([x, rng.uniform(lo, hi, (clutter, 3))]) * scale


def _centre_along_mm(fit_t: np.ndarray, truth: np.ndarray, centre_world: np.ndarray) -> float:
    c = fit_t[:3, :3] @ centre_world + fit_t[:3, 3]
    g = truth[:3, :3] @ centre_world + truth[:3, 3]
    return float(along_ray_components(c[None], g[None]).signed_along[0] * 1e3)


def test_single_symmetric_object_free_scale_is_refused_and_fixed_ruler_is_exact() -> None:
    """Known answer for the 7 one-object episodes: on one occluded pot the free scale collapses (flagged), while a
    rigid refit at the device ruler recovers the camera-frame centre to < 1 mm; a 1% ruler error costs ~(s-1) z."""
    truth = _cam_T_world()
    mp, mn = _pot()
    centre = _POT_AXIS_BASE + [0.0, 0.0, _POT_H / 2]
    z = float(np.linalg.norm(truth[:3, :3] @ centre + truth[:3, 3]))
    for seed in range(3):
        cloud = _occluded_cloud(truth, mp, mn, 0.002, 1500, 1.03, seed)       # stereo ruler 3% long
        init = _perturb(truth, 4.0, 0.02, seed=20 + seed)
        free = register_scene(mp, mn, cloud, init)
        assert not free.converged                                            # the collapse is flagged, not used
        fixed = register_scene(mp, mn, cloud, init, estimate_scale=False, fixed_scale=1.0 / 1.03)
        assert abs(_centre_along_mm(fixed.cam_T_world, truth, centre)) < 1.0
        off = register_scene(mp, mn, cloud, init, estimate_scale=False, fixed_scale=1.01 / 1.03)
        assert _centre_along_mm(off.cam_T_world, truth, centre) == pytest.approx(10.0 * z, rel=0.2)  # 1% of z, mm


def test_device_ruler_pools_static_scan_frames() -> None:
    """One ruler per device from free-scale fits on the two-box (dish-rack-like) scene, scatter reported."""
    truth = _cam_T_world()
    mp, mn = _scene()
    fits = [register_scene(mp, mn, _observed_cloud(truth, 0.002, 1500, scale=1.02, seed=s), _perturb(truth, 5.0, 0.02,
                                                                                                    seed=30 + s))
            for s in range(5)]
    ruler = device_ruler(np.array([f.cloud_scale for f in fits]), ok=np.array([f.converged for f in fits]))
    assert ruler.n_frames == 5 and ruler.scale == pytest.approx(1.0 / 1.02, abs=0.004)
    assert ruler.log_sd < 0.01                                               # the registered scatter gate
    with pytest.raises(ValueError):
        device_ruler(np.array([1.0, np.nan, 0.0]))


def test_latency_by_agreement_recovers_a_vicon_lag() -> None:
    """A static pot is lag-blind, a box moving at 0.5 m/s is not: per-object fits agree only at the true lag."""
    truth = _cam_T_world()
    pot_p, pot_n = _pot(n=3000)
    box_p, box_n = _box_samples(np.array([0.14, 0.05, 0.10]), 3000, 9)
    box_p = box_p + [-0.18, 0.12, 0.70]
    vel, lag_true = np.array([0.5, 0.0, 0.0]), 0.05
    frames_t = np.array([0.0, 0.2, 0.4])
    pot_fits = []
    clouds = []
    for k, t in enumerate(frames_t):
        bp = box_p + vel * t                                                 # where the box really is at video time t
        pc = _occluded_cloud(truth, np.vstack([pot_p, bp]), np.vstack([pot_n, box_n]), 0.001, 500, 1.0, k,
                             occlude=False)
        clouds.append(pc)
        pot_fits.append(register_scene(pot_p, pot_n, pc, _perturb(truth, 3.0, 0.01, seed=40 + k),
                                       estimate_scale=False).cam_T_world)

    def disagreement(lag: float) -> float:
        d = []
        for k, t in enumerate(frames_t):
            vicon_box = box_p + vel * (t + lag - lag_true)                   # Vicon stamps lag the video by lag_true
            f = register_scene(vicon_box, box_n, clouds[k], pot_fits[k], estimate_scale=False).cam_T_world
            d.append(float(np.linalg.norm(f[:3, 3] - pot_fits[k][:3, 3])))
        return float(np.median(d))

    best, curve = latency_by_agreement(np.arange(-0.05, 0.1251, 0.025), disagreement)
    assert abs(best - lag_true) <= 0.0125, curve
    assert curve.max() > 5.0 * curve.min()
