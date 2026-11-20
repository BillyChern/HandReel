"""Tests for `fastdo.retarget.contact`: human/robot contact measures on the object surface.

Two kinds of fixtures:
  (1) a synthetic unit-box mesh with hand-crafted query points (exact, no external data), and
  (2) the baseline's shipped whisking demo (scene.xml, HaWoR hands, object layout, MANO) -- skipped
      with a clear reason wherever any of those real files is unavailable in this environment.

Independent ground truth for "contact frames" (tests 3 & 4). `trajectory_keypoints.npz`'s own
`contact_right`/`contact_pos_right` arrays -- which a test description might take for "the
baseline flags" -- are hardcoded to `np.zeros(...)` by the baseline's own
`retargeting/retargeting/pipeline/process_dataset.py` (see lines ~913-916: `contact_right=np.zeros((N,
10))`, `contact_pos_right=np.zeros((10, 3))`) and are never filled in anywhere else in the shipped
pipeline for this clip -- confirmed by loading the real file and finding every entry exactly 0 over
all 138 frames, both hands. That field therefore carries no information here, so this file uses an
independent, non-circular proxy instead: for each HaWoR frame, map the 778 MANO vertices into the
canonical object frame with the SAME pose/scale convention `human_contact_measure` uses (but not
through `human_contact_measure`, `sdf_lookup`, or `SdfGrid` at all -- through `trimesh`'s own
BVH-based closest-point query on the canonical mesh), and flag the frame "in contact" if the nearest
vertex-to-surface distance is under `tau`. This exercises a completely different
implementation/library from the code under test while asking the same physical question, so it is
suitable as ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from torch import Tensor

from fastdo.retarget.contact import ContactMeasure, human_contact_measure, robot_contact_measure
from fastdo.retarget.labels import (
    FINGER_LABEL_NAMES,
    INDEX,
    MIDDLE,
    PALM,
    PINKY,
    RING,
    THUMB,
    mano_finger_labels,
    robot_link_labels,
)
from fastdo.retarget.mj_model import RobotScene
from fastdo.solver.sdf import mesh_sdf_grid
from fastdo.teacher.layout_reader import read_baseline_layout

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs" / "sharpa"
             / "right" / "whisking" / "0" / "scene.xml")
TRAJECTORY_KINEMATIC_NPZ = SCENE_XML.parent / "trajectory_kinematic.npz"
TRAJECTORY_KEYPOINTS_NPZ = (REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting" / "outputs"
                            / "mano" / "right" / "whisking" / "0" / "trajectory_keypoints.npz")
HAWOR_NPZ = (REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "whisking" / "whisking"
             / "all_hand_meshes.npz")
LAYOUT_JSON = (REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "whisking"
               / "obj_tracking_out" / "whisk" / "combined_visualization"
               / "layout_camera_frame_optimized.json")
WHISK_OBJ = (REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "whisking"
             / "video_segmentation" / "masks" / "frame_000125_masks" / "whisk" / "whisk.obj")

_WHISKING_FILES = (SCENE_XML, TRAJECTORY_KINEMATIC_NPZ, HAWOR_NPZ, LAYOUT_JSON, WHISK_OBJ)
whisking_data_missing = pytest.mark.skipif(
    not all(p.exists() for p in _WHISKING_FILES),
    reason=f"shipped whisking demo data not available: {[str(p) for p in _WHISKING_FILES if not p.exists()]}",
)

_MANO_RIGHT_CANDIDATES = (
    REPO_ROOT / "data" / "mano" / "MANO_RIGHT.pkl",
    Path("/opt/fastdo/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl"),
)
MANO_RIGHT_PKL = next((p for p in _MANO_RIGHT_CANDIDATES if p.exists()), None)
mano_missing = pytest.mark.skipif(
    MANO_RIGHT_PKL is None,
    reason=f"no MANO_RIGHT.pkl found in {[str(p) for p in _MANO_RIGHT_CANDIDATES]}",
)

keypoints_missing = pytest.mark.skipif(
    not TRAJECTORY_KEYPOINTS_NPZ.exists() or not TRAJECTORY_KINEMATIC_NPZ.exists(),
    reason=f"trajectory_keypoints.npz/trajectory_kinematic.npz not both available: "
           f"{[str(p) for p in (TRAJECTORY_KEYPOINTS_NPZ, TRAJECTORY_KINEMATIC_NPZ) if not p.exists()]}",
)

# `trajectory_kinematic.npz` (135 frames, the baseline's mink-IK kinematic solution) and the
# HaWoR/layout frame indexing (138 frames) are offset by one frame -- derived, not assumed, by
# `test_kin_to_hawor_offset_is_derived_from_object_translation_residual` below (which scans offsets
# -2..+2 comparing `qpos[:, 28:31]` against `trajectory_keypoints.npz`'s `qpos_obj_right[:, :3]` and
# asserts the minimizing offset equals this constant); measured
# residuals (8.5 mm at +1 vs. 19.1 mm at 0).
_KIN_TO_HAWOR_OFFSET = 1


# --------------------------------------------------------------------------------------------
# Synthetic fixtures (no external data)
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unit_box_sdf_and_surface() -> tuple[object, Tensor]:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sdf = mesh_sdf_grid(np.asarray(box.vertices), np.asarray(box.faces), resolution=48)
    surf_pts, _ = trimesh.sample.sample_surface(box, 500, seed=np.random.default_rng(0))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    return sdf, surface


def test_synthetic_band_weights_match_closed_form_and_zero_beyond_tau(
    unit_box_sdf_and_surface: tuple[object, Tensor],
) -> None:
    """3 points at known signed distances from the box's +x face (0.5, 0, 0): 0 / 4 / 10 mm outside."""
    sdf, surface = unit_box_sdf_and_surface
    sigma, tau = 0.004, 0.008
    offsets = torch.tensor([0.0, 0.004, 0.010], dtype=torch.float64)
    pts = torch.zeros(1, 3, 3, dtype=torch.float64)
    pts[0, :, 0] = 0.5 + offsets
    labels = torch.tensor([PALM, THUMB, INDEX], dtype=torch.int8)
    poses = torch.eye(4, dtype=torch.float64)[None]

    meas = human_contact_measure(pts, poses, 1.0, labels, sdf, surface, sigma=sigma, tau=tau)

    phi = meas.sdf_values[0]
    expected = torch.where(phi.abs() < tau, torch.exp(-(phi ** 2) / (2 * sigma ** 2)), torch.zeros_like(phi))
    assert torch.allclose(meas.weights[0], expected, atol=1e-9)
    # The offsets were chosen so the geometry itself, not just the formula, lands where expected:
    assert meas.weights[0, 0].item() > 0.99          # ~on the surface
    assert 0.55 < meas.weights[0, 1].item() < 0.65    # 4 mm out, sigma=4mm -> exp(-0.5)=0.6065
    assert meas.weights[0, 2].item() == 0.0           # 10 mm out > tau=8mm
    assert torch.equal(meas.labels, labels)
    # scale=1, identity pose -> the canonical raw (unprojected) position is exactly the input point.
    assert torch.allclose(meas.raw_points[0], pts[0])


def test_synthetic_weight_is_exactly_zero_at_tau_boundary(unit_box_sdf_and_surface: tuple[object, Tensor]) -> None:
    """The band is a strict `|phi| < tau` cutoff: a point exactly at tau=8mm gets weight 0, not exp(-2)."""
    sdf, surface = unit_box_sdf_and_surface
    pts = torch.tensor([[[0.508, 0.0, 0.0]]], dtype=torch.float64)   # exactly 8mm outside the +x face
    labels = torch.tensor([PALM], dtype=torch.int8)
    poses = torch.eye(4, dtype=torch.float64)[None]
    meas = human_contact_measure(pts, poses, 1.0, labels, sdf, surface, sigma=0.004, tau=0.008)
    assert abs(meas.sdf_values[0, 0].item() - 0.008) < 1e-4
    assert meas.weights[0, 0].item() == 0.0


def test_synthetic_nearest_surface_projection_picks_closest_known_sample() -> None:
    """Hand-crafted `surface` (not mesh-sampled): projection must pick the literally-nearest sample."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sdf = mesh_sdf_grid(np.asarray(box.vertices), np.asarray(box.faces), resolution=48)
    surface = torch.tensor([
        [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.5],
    ], dtype=torch.float64)
    # Two query points, each unambiguously closest to one known sample.
    pts = torch.tensor([[[0.49, 0.02, 0.01], [0.01, 0.0, 0.49]]], dtype=torch.float64)
    labels = torch.tensor([INDEX, MIDDLE], dtype=torch.int8)
    poses = torch.eye(4, dtype=torch.float64)[None]
    meas = human_contact_measure(pts, poses, 1.0, labels, sdf, surface, sigma=0.004, tau=0.008)
    assert meas.sample_idx[0].tolist() == [0, 4]
    assert torch.allclose(meas.points[0, 0], surface[0])
    assert torch.allclose(meas.points[0, 1], surface[4])
    # raw_points is the *pre*-projection position -- the input query points themselves here.
    assert torch.allclose(meas.raw_points[0], pts[0])
    assert not torch.allclose(meas.raw_points[0, 0], meas.points[0, 0])


def test_synthetic_vertex_outside_grid_gets_zero_weight_not_an_exception() -> None:
    """A point far outside the padded SDF grid must get weight 0, never raise."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sdf = mesh_sdf_grid(np.asarray(box.vertices), np.asarray(box.faces), resolution=24, padding=0.1)
    surf_pts, _ = trimesh.sample.sample_surface(box, 200, seed=np.random.default_rng(1))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    pts = torch.tensor([[[50.0, 50.0, 50.0]]], dtype=torch.float64)   # far outside any padding
    labels = torch.tensor([PALM], dtype=torch.int8)
    poses = torch.eye(4, dtype=torch.float64)[None]
    meas = human_contact_measure(pts, poses, 1.0, labels, sdf, surface, sigma=0.004, tau=0.008)
    assert meas.weights[0, 0].item() == 0.0
    assert meas.mass[0].item() == 0.0


# --------------------------------------------------------------------------------------------
# Robot Jacobian: finite-difference check (real shipped scene + trajectory)
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def whisk_mesh() -> trimesh.Trimesh:
    return trimesh.load(WHISK_OBJ, process=False)


@pytest.fixture(scope="module")
def whisk_sdf_and_surface(whisk_mesh: trimesh.Trimesh) -> tuple[object, Tensor]:
    """The one canonical-mesh SDF/surface-sample pair every whisking-data test shares.

    Built once (module scope) from `whisk_mesh` and reused by both the human- and robot-side
    fixtures below (`whisking_human_measure`, `whisking_robot_measure`) -- the two measures are only
    comparable if they are evaluated against the exact same canonical frame (same SDF resolution,
    same surface-sample seed/count), not two independently rebuilt ones.
    """
    sdf = mesh_sdf_grid(np.asarray(whisk_mesh.vertices), np.asarray(whisk_mesh.faces), resolution=64)
    surf_pts, _ = trimesh.sample.sample_surface(whisk_mesh, 1024, seed=np.random.default_rng(0))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    return sdf, surface


@pytest.fixture(scope="module")
def scene() -> RobotScene:
    return RobotScene.from_xml(SCENE_XML)


# Metres-per-canonical-unit object scale, from `read_baseline_layout` on the shipped layout JSON
# (`mesh_scale`). Cross-checked against `scene.xml`'s own object mesh: `scene.xml` carries NO
# `scale=` attribute anywhere on the object's `<mesh>` asset (`objects/whisking/visual.obj`), and
# that .obj file's own extents are already `mesh_scale` x the canonical (SAM-3D-reconstruction,
# ~unit-scale) `whisk.obj` used for the SDF -- ratio measured as 0.180863 on all 3 axes, i.e. the
# .obj was pre-baked to metres at export time rather than scaled by MuJoCo at load time. So
# `read_baseline_layout`'s JSON-derived `mesh_scale` is the (only) authoritative source; there is no
# competing `scene.xml` attribute to reconcile it against.
_MESH_SCALE = 0.18086324855404615


@whisking_data_missing
def test_robot_jacobian_matches_central_finite_differences(
    scene: RobotScene, whisk_sdf_and_surface: tuple[object, Tensor],
) -> None:
    sdf, surface = whisk_sdf_and_surface
    with np.load(TRAJECTORY_KINEMATIC_NPZ) as data:
        qpos_real = data["qpos"]

    points, body_ids, _ = scene.surface_points(n_per_link=5, seed=0)
    labels = robot_link_labels(scene.model, body_ids)

    q0_np = qpos_real[60]     # a real, non-degenerate hand-object contact configuration
    q0 = torch.as_tensor(q0_np, dtype=torch.float64)[None, :]
    measure, dw_dq = robot_contact_measure(q0, scene, body_ids, points, labels, sdf, surface,
                                           _MESH_SCALE, jacobian=True)
    assert dw_dq.shape == (1, len(body_ids), 28)
    assert measure.raw_points.shape == (1, len(body_ids), 3)
    assert measure.points.shape == (1, len(body_ids), 3)

    active = torch.nonzero(measure.weights[0] > 1e-6).flatten().tolist()
    assert len(active) >= 5, "fixture frame should have several in-band robot points to check"
    check_points = active[:10]

    eps = 1e-6
    max_abs_err, max_rel_err = 0.0, 0.0
    for k in range(28):
        q_plus, q_minus = q0_np.copy(), q0_np.copy()
        q_plus[k] += eps
        q_minus[k] -= eps
        w_plus = robot_contact_measure(torch.as_tensor(q_plus, dtype=torch.float64)[None], scene, body_ids,
                                       points, labels, sdf, surface, _MESH_SCALE, jacobian=False).weights[0]
        w_minus = robot_contact_measure(torch.as_tensor(q_minus, dtype=torch.float64)[None], scene, body_ids,
                                        points, labels, sdf, surface, _MESH_SCALE, jacobian=False).weights[0]
        fd = (w_plus - w_minus) / (2 * eps)
        for u in check_points:
            a, f = dw_dq[0, u, k].item(), fd[u].item()
            max_abs_err = max(max_abs_err, abs(a - f))
            if abs(f) > 1e-9:
                max_rel_err = max(max_rel_err, abs(a - f) / abs(f))

    assert max_rel_err < 1e-4, f"max relative FD error {max_rel_err}"


# --------------------------------------------------------------------------------------------
# Real whisking-clip data: human measure vs. independent contact ground truth, handle mass, robot side
# --------------------------------------------------------------------------------------------


def _canonical_mesh_axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Principal-axis handle/head split (design: "the end whose per-bin cross-section radius is smaller").

    Returns `(centroid, axis, midpoint, handle_is_low_side)`: a point `p` (canonical frame) is "on
    the handle" iff `((p - centroid) @ axis < midpoint) == handle_is_low_side`.
    """
    centroid = vertices.mean(axis=0)
    centred = vertices - centroid
    cov = centred.T @ centred / len(centred)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    proj = centred @ axis
    n_bins = 20
    bin_edges = np.linspace(proj.min(), proj.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(proj, bin_edges) - 1, 0, n_bins - 1)
    perp_radius = np.linalg.norm(centred - np.outer(proj, axis), axis=1)
    bin_radius = np.array([perp_radius[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan
                           for i in range(n_bins)])
    lo_radius = np.nanmean(bin_radius[: n_bins // 2])
    hi_radius = np.nanmean(bin_radius[n_bins // 2:])
    midpoint = 0.5 * (proj.min() + proj.max())
    return centroid, axis, float(midpoint), bool(lo_radius < hi_radius)


def _reference_contact_frames(mesh: trimesh.Trimesh, hand_verts_canon: np.ndarray, tau_c: float) -> np.ndarray:
    """Independent (non-`SdfGrid`) "in contact" flag per frame: any MANO vertex within `tau_c` of the mesh."""
    query = trimesh.proximity.ProximityQuery(mesh)
    n_frames = hand_verts_canon.shape[0]
    in_contact = np.zeros(n_frames, dtype=bool)
    for t in range(n_frames):
        _, dist, _ = query.on_surface(hand_verts_canon[t])
        in_contact[t] = dist.min() < tau_c
    return in_contact


@pytest.fixture(scope="module")
def whisking_human_measure(
    whisk_mesh: trimesh.Trimesh, whisk_sdf_and_surface: tuple[object, Tensor],
) -> tuple[ContactMeasure, np.ndarray, trimesh.Trimesh]:
    """Human `ContactMeasure` over all 138 shipped frames, plus the independent reference contact flags.

    Shares `whisk_sdf_and_surface` with the robot-side fixture (`whisking_robot_measure`) rather than
    rebuilding its own SDF/surface: both measures must be evaluated against the exact same canonical
    frame (same resolution, same sample seed/count) for the two to be comparable at all.
    """
    with np.load(HAWOR_NPZ) as data:
        verts = torch.as_tensor(np.asarray(data["right_vertices"], dtype=np.float64))
    poses, mesh_scale, _ = read_baseline_layout(LAYOUT_JSON)
    assert abs(mesh_scale - _MESH_SCALE) < 1e-6

    sdf, surface = whisk_sdf_and_surface
    mesh = whisk_mesh

    labels = torch.zeros(verts.shape[1], dtype=torch.int8) if MANO_RIGHT_PKL is None else \
        mano_finger_labels(MANO_RIGHT_PKL, cache_dir=REPO_ROOT / "data" / "mano")
    measure = human_contact_measure(verts, poses, mesh_scale, labels, sdf, surface, sigma=0.004, tau=0.008)

    R, t = poses[:, :3, :3].double().numpy(), poses[:, :3, 3].double().numpy()
    y = np.einsum("tji,tvj->tvi", R, verts.numpy() - t[:, None, :]) / mesh_scale
    reference = _reference_contact_frames(mesh, y, tau_c=0.008 / mesh_scale)
    return measure, reference, mesh


@whisking_data_missing
def test_human_measure_mass_matches_independent_contact_frames(
    whisking_human_measure: tuple[ContactMeasure, np.ndarray, trimesh.Trimesh],
) -> None:
    measure, reference, _ = whisking_human_measure
    our_contact = (measure.mass > 0).numpy()
    disagreement = (our_contact != reference).mean()
    assert disagreement <= 0.10, f"disagreement {disagreement:.3f} exceeds the allowed 10%"


@whisking_data_missing
def test_human_measure_concentrates_on_the_handle_on_contact_frames(
    whisking_human_measure: tuple[ContactMeasure, np.ndarray, trimesh.Trimesh],
) -> None:
    """On (independently-flagged) contact frames, >=80% of the human measure's mass sits on the handle."""
    measure, reference, mesh = whisking_human_measure
    centroid, axis, midpoint, handle_is_lo = _canonical_mesh_axis(np.asarray(mesh.vertices))

    pts = measure.points.numpy()                                     # (T, V, 3) canonical
    proj = (pts - centroid) @ axis                                    # (T, V)
    on_handle = (proj < midpoint) if handle_is_lo else (proj > midpoint)

    weights = measure.weights.numpy()
    mass = measure.mass.numpy()
    contact_idx = np.nonzero(reference & (mass > 0))[0]
    assert len(contact_idx) > 0

    handle_frac = np.array([weights[t][on_handle[t]].sum() / mass[t] for t in contact_idx])
    frac_meeting_80pct = (handle_frac >= 0.8).mean()
    assert frac_meeting_80pct >= 0.90, (
        f"only {frac_meeting_80pct:.3f} of contact frames have >=80% handle mass "
        f"(median handle frac {np.median(handle_frac):.3f})"
    )


@pytest.fixture(scope="module")
def whisking_robot_measure(scene: RobotScene, whisk_sdf_and_surface: tuple[object, Tensor],
                          ) -> tuple[ContactMeasure, np.ndarray, np.ndarray]:
    """Robot `ContactMeasure` over all 135 `trajectory_kinematic.npz` frames, plus per-point finger labels."""
    sdf, surface = whisk_sdf_and_surface
    with np.load(TRAJECTORY_KINEMATIC_NPZ) as data:
        qpos = data["qpos"]
    points, body_ids, _ = scene.surface_points(n_per_link=30, seed=0)
    labels = robot_link_labels(scene.model, body_ids)
    q = torch.as_tensor(qpos, dtype=torch.float64)
    measure = robot_contact_measure(q, scene, body_ids, points, labels, sdf, surface, _MESH_SCALE,
                                    sigma=0.004, tau=0.008, jacobian=False)
    return measure, labels.numpy(), qpos


@whisking_data_missing
def test_robot_measure_has_mass_and_finger_diversity_on_contact_frames(
    whisking_robot_measure: tuple[ContactMeasure, np.ndarray, np.ndarray],
    whisking_human_measure: tuple[ContactMeasure, np.ndarray, trimesh.Trimesh],
) -> None:
    """Robot measure at the baseline's *kinematic-only* solution (mink IK, no MPC) on the same contact
    frames the human measure is on (via `_KIN_TO_HAWOR_OFFSET`).

    Empirically, the baseline's per-frame fingertip-IK kinematic stage
    reliably closes index/middle contact but leaves the thumb measurably farther from the object on
    a large minority of frames (median ~13.5 mm vs. ~2-3 mm for index/middle) -- exactly the
    contact-fidelity gap the contact-transport objective replaces this stage to fix.
    Thresholds below are set from the measured baseline, not chosen to make an arbitrary bar pass:
    thumb mass is required on a solid minority of contact frames (not "most", which the kinematic-
    only baseline does not reach for the thumb) and at least one other finger on a strong majority.
    """
    robot_measure, robot_labels, _ = whisking_robot_measure
    _, human_reference, _ = whisking_human_measure

    n_kin = robot_measure.weights.shape[0]
    kin_reference = human_reference[_KIN_TO_HAWOR_OFFSET:_KIN_TO_HAWOR_OFFSET + n_kin]
    assert kin_reference.shape[0] == n_kin

    mass = robot_measure.mass.numpy()
    contact_mass = mass[kin_reference]
    frac_positive = (contact_mass > 0).mean()
    assert frac_positive >= 0.80, f"only {frac_positive:.3f} of contact frames have positive robot mass"

    weights = robot_measure.weights.numpy()
    thumb_mass = weights[:, robot_labels == THUMB].sum(axis=1)[kin_reference]
    thumb_frac = (thumb_mass > 0).mean()
    assert thumb_frac >= 0.40, f"thumb carries mass on only {thumb_frac:.3f} of contact frames"

    best_other_frac = 0.0
    for finger in (INDEX, MIDDLE, RING, PINKY):
        finger_mass = weights[:, robot_labels == finger].sum(axis=1)[kin_reference]
        best_other_frac = max(best_other_frac, (finger_mass > 0).mean())
    assert best_other_frac >= 0.75, f"best non-thumb finger only active on {best_other_frac:.3f} of contact frames"


@mano_missing
def test_mano_finger_labels_cover_all_fingers_and_palm() -> None:
    assert MANO_RIGHT_PKL is not None
    labels = mano_finger_labels(MANO_RIGHT_PKL, cache_dir=REPO_ROOT / "data" / "mano")
    assert labels.shape == (778,)
    assert labels.dtype == torch.int8
    present = set(torch.unique(labels).tolist())
    assert present == {PALM, THUMB, INDEX, MIDDLE, RING, PINKY}


def test_mano_finger_labels_synthetic_shape_and_dtype_without_a_pkl(tmp_path: Path) -> None:
    """Contact-measure logic must be testable even with no MANO pkl on disk: a synthetic label vector."""
    labels = torch.zeros(778, dtype=torch.int8)
    labels[:100] = THUMB
    labels[100:300] = INDEX
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    sdf = mesh_sdf_grid(np.asarray(box.vertices), np.asarray(box.faces), resolution=24)
    surf_pts, _ = trimesh.sample.sample_surface(box, 200, seed=np.random.default_rng(2))
    surface = torch.as_tensor(np.asarray(surf_pts), dtype=torch.float64)
    verts = torch.zeros(1, 778, 3, dtype=torch.float64)
    verts[0, :, 0] = 0.5
    poses = torch.eye(4, dtype=torch.float64)[None]
    measure = human_contact_measure(verts, poses, 1.0, labels, sdf, surface)
    assert torch.equal(measure.labels, labels)
    assert measure.weights.shape == (1, 778)
    assert measure.raw_points.shape == (1, 778, 3)
    assert measure.points.shape == (1, 778, 3)


def test_finger_label_names_indexable_by_enum() -> None:
    assert FINGER_LABEL_NAMES[PALM] == "palm"
    assert FINGER_LABEL_NAMES[THUMB] == "thumb"
    assert FINGER_LABEL_NAMES[PINKY] == "pinky"


@keypoints_missing
def test_kin_to_hawor_offset_is_derived_from_object_translation_residual() -> None:
    """`_KIN_TO_HAWOR_OFFSET` must not silently go stale: re-derive it here and assert it matches.

    Scans offsets -2..+2 comparing `trajectory_kinematic.npz`'s own object-translation qpos block
    (`qpos[:, 28:31]`) against `trajectory_keypoints.npz`'s `qpos_obj_right[:, :3]`, picks the offset
    minimizing the mean per-frame translation residual, and asserts it equals the module constant --
    so a future change to either shipped file (different frame trimming, a data refresh, ...) fails
    this test loudly instead of silently corrupting `test_robot_measure_has_mass_and_finger_diversity_
    on_contact_frames`'s frame alignment.
    """
    with np.load(TRAJECTORY_KINEMATIC_NPZ) as data:
        kin_obj_pos = data["qpos"][:, 28:31]
    with np.load(TRAJECTORY_KEYPOINTS_NPZ) as data:
        kp_obj_pos = data["qpos_obj_right"][:, :3]
    n_kin, n_kp = kin_obj_pos.shape[0], kp_obj_pos.shape[0]

    best_offset, best_score = None, np.inf
    for offset in range(-2, 3):
        lo, hi = max(0, -offset), min(n_kin, n_kp - offset)
        if hi <= lo:
            continue
        residual = np.linalg.norm(kin_obj_pos[lo:hi] - kp_obj_pos[lo + offset:hi + offset], axis=1)
        score = float(residual.mean())
        if score < best_score:
            best_score, best_offset = score, offset

    assert best_offset == _KIN_TO_HAWOR_OFFSET, (
        f"derived offset {best_offset} (mean residual {best_score:.4f} m) no longer matches the "
        f"hardcoded _KIN_TO_HAWOR_OFFSET={_KIN_TO_HAWOR_OFFSET}"
    )
