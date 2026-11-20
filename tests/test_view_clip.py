"""scripts/view_clip.py on a tiny synthetic clip directory (CPU only, no model, no data).

The loader tests need only numpy, trimesh and OpenCV. The server test starts the viewer on 127.0.0.1 and gets its
page with HTTP 200; it skips when the viser extra is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "view_clip.py"
_SPEC = importlib.util.spec_from_file_location("view_clip", SCRIPT)
V = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = V
_SPEC.loader.exec_module(V)

T, H, W = 3, 48, 64
FRAME_IDX = np.array([0, 2, 4], np.int64)


def _clip(root: Path, *, faces: bool = True, decode: bool = True, contact: bool = False) -> Path:
    """A clip directory as the engine writes it, with a unit cube 0.5 m in front of the camera."""
    import trimesh

    d = root / "tiny_clip"
    d.mkdir()
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(d / "mesh.obj")
    poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    poses[:, 2, 3] = 0.5
    poses[:, 0, 3] = [0.0, 0.01, 0.02]
    np.savez(d / "solve.npz", poses=poses, scale=np.float64(0.1), frame_idx=FRAME_IDX)
    if contact:
        moved = poses.copy()
        moved[:, 1, 3] = 0.03
        np.savez(d / "contact.npz", poses=moved, frame_idx=FRAME_IDX)
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], np.float32)
    np.savez(d / "intrinsics.npz", K=K, source=np.array("moge2"))
    rng = np.random.default_rng(0)
    hands: dict[str, np.ndarray] = {"frame_idx": FRAME_IDX, "camera_K": K.astype(np.float64)}
    for side in ("right", "left"):
        hands[f"{side}_vertices"] = (rng.normal(0, 0.01, (T, 778, 3)) + [0, 0, 0.45]).astype(np.float32)
        hands[f"{side}_valid"] = np.array([True, side == "right", False])
        if faces:
            hands[f"{side}_faces"] = rng.integers(0, 778, (1552, 3)).astype(np.int32)
    np.savez(d / "hands.npz", **hands)
    if decode:
        frames = np.stack([np.full((H, W, 3), 40 * (i + 1), np.uint8) for i in range(T)])
        np.savez(d / "decode.npz", frames=frames, frame_idx=FRAME_IDX, fps=np.float64(20.0), stride=np.int64(2))
    return d


def test_the_loader_places_the_mesh_with_the_stored_pose_and_scale(tmp_path: Path) -> None:
    clip = V.load_clip(_clip(tmp_path))
    assert [f.fid for f in clip.frames] == FRAME_IDX.tolist()
    centre = clip.frames[1].obj_v.mean(0)
    assert np.allclose(centre, [0.01, 0.0, 0.5], atol=1e-6)                  # scale * R @ y + t
    assert np.ptp(clip.frames[0].obj_v[:, 0]) == pytest.approx(0.1)           # the unit cube at scale 0.1
    assert [[s for s, _ in f.hands] for f in clip.frames] == [["right", "left"], ["right"], []]   # valid rows only
    assert clip.pose_source == "solve.npz" and clip.image_source == "decode.npz"
    assert clip.frames[2].rgb is not None and int(clip.frames[2].rgb[0, 0, 0]) == 120


def test_contact_poses_replace_the_solve_poses_when_present(tmp_path: Path) -> None:
    clip = V.load_clip(_clip(tmp_path, contact=True))
    assert clip.pose_source == "contact.npz"
    assert clip.frames[0].obj_v.mean(0)[1] == pytest.approx(0.03)


def test_frames_from_a_video_when_there_is_no_decode_npz(tmp_path: Path) -> None:
    import cv2

    d = _clip(tmp_path, decode=False, faces=False)
    video = tmp_path / "src.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (W, H))
    for i in range(6):
        writer.write(np.full((H, W, 3), 30 * i, np.uint8))
    writer.release()
    clip = V.load_clip(d, video)
    assert clip.image_source == video.name
    assert [abs(int(f.rgb[0, 0, 0]) - 30 * f.fid) <= 3 for f in clip.frames] == [True] * T   # frame n of the video
    assert clip.hand_f == {"right": None, "left": None}                       # no faces: drawn as points
    assert V.load_clip(d).image_source == "none"


def test_mismatched_frame_idx_and_missing_files_exit_with_one_line(tmp_path: Path) -> None:
    d = _clip(tmp_path)
    hands = dict(np.load(d / "hands.npz"))
    hands["frame_idx"] = FRAME_IDX + 1
    np.savez(d / "hands.npz", **hands)
    with pytest.raises(SystemExit, match="hands.npz and solve.npz have different frame_idx"):
        V.main([str(d)])
    with pytest.raises(SystemExit, match="is .* an engine clip directory"):
        V.main([str(tmp_path / "nothing")])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.skipif(importlib.util.find_spec("viser") is None,
                    reason="the viser extra is not installed (uv sync --frozen --extra viser)")
def test_the_viewer_serves_its_page_on_localhost(tmp_path: Path) -> None:
    d = _clip(tmp_path)
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(SCRIPT), str(d), "--port", str(port), "--run-seconds", "60"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline, line = time.monotonic() + 60, ""
        while time.monotonic() < deadline and "serving http://" not in line:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
        assert f"serving http://127.0.0.1:{port}" in line, line
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=20) as r:
            assert r.status == 200
            assert b"<html" in r.read(4096).lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


REFERENCE = REPO / "release" / "reference" / "v2d_t3_ep011_cam_a"


def test_the_shipped_reference_output_loads_and_names_its_clip() -> None:
    """The reference output of the example clip: 69 processed rows (stride 2 over 137 source frames), the object in
    front of the camera, and hands.npz without the MANO face and vertex arrays (its joints are drawn as points)."""
    import json

    clip = V.load_clip(REFERENCE)
    assert [f.fid for f in clip.frames] == list(range(0, 137, 2))
    assert clip.hand_f == {"right": None, "left": None}
    assert not [k for k in np.load(REFERENCE / "hands.npz").files if k.endswith(("_faces", "_vertices"))]
    assert {len(v) for f in clip.frames for _, v in f.hands} == {21}                  # the joints
    report = json.loads((REFERENCE / "example_report.json").read_text())
    assert report["clip_id"] == "v2d_t3_ep011_cam_a" and report["input"]["sha256_matches_pin"] is True
    assert set(report["stage_states"].values()) == {"done"}
    assert "CC BY 4.0" in (REFERENCE / "ATTRIBUTION.txt").read_text()


def test_a_stub_or_partial_run_gives_a_one_line_error(tmp_path: Path) -> None:
    """A --stub output (0-byte placeholders), a run's --out directory and an unreadable npz exit with one
    line that says what is wrong, never a traceback."""
    import subprocess

    stub = tmp_path / "out" / "engine" / "clip"
    stub.mkdir(parents=True)
    for name in ("mesh.obj", "solve.npz", "intrinsics.npz"):
        (stub / name).write_bytes(b"")
    script = Path(__file__).resolve().parents[1] / "scripts" / "view_clip.py"
    for target, want in ((stub, "empty placeholder files: a --stub run"),
                         (tmp_path / "out", "looks like a run's --out directory")):
        r = subprocess.run([sys.executable, str(script), str(target)], capture_output=True, text=True, timeout=120,
                           check=False)
        assert r.returncode == 1 and want in r.stderr and "Traceback" not in r.stderr, r.stderr
    (stub / "mesh.obj").write_text("v 0 0 1\n")
    (stub / "solve.npz").write_bytes(b"PK\x03\x04 truncated")
    (stub / "intrinsics.npz").write_bytes(b"x")
    r = subprocess.run([sys.executable, str(script), str(stub)], capture_output=True, text=True, timeout=120,
                       check=False)
    assert r.returncode == 1 and "not a readable npz file" in r.stderr and "Traceback" not in r.stderr, r.stderr


def test_render_reference_png_draws_the_reference_rows(tmp_path: Path) -> None:
    """scripts/render_reference_png.py on the shipped reference (no video: grey canvas) writes one PNG with a panel per
    requested row, and refuses a row outside the clip with a one-line message."""
    import subprocess

    import cv2

    script = REPO / "scripts" / "render_reference_png.py"
    out = tmp_path / "ref.png"
    r = subprocess.run([sys.executable, str(script), str(REFERENCE), "--out", str(out), "--panel-width", "200"],
                       capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stderr
    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] == 3 * 200
    assert (REFERENCE / "expected_projection.png").is_file()
    r = subprocess.run([sys.executable, str(script), str(REFERENCE), "--out", str(tmp_path / "x.png"), "--rows", "69"],
                       capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 1 and "outside 0..68" in r.stderr and "Traceback" not in r.stderr


def test_render_credits_v2d_only_for_the_example_source(tmp_path: Path) -> None:
    """The V2D CC BY credit goes under the shipped reference's frames, never under another clip's (an own video run
    with `--spec`, or a directory without `source_domain.json`)."""
    spec = importlib.util.spec_from_file_location("render_reference_png", REPO / "scripts" / "render_reference_png.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.credit_line(REFERENCE) == mod.CREDIT
    own = tmp_path / "my_clip"
    own.mkdir()
    (own / "source_domain.json").write_text(json.dumps({"clip_id": "my_clip", "source": "own_video",
                                                        "domain": "egocentric"}))
    line = mod.credit_line(own)
    assert "V2D" not in line and "NVIDIA" not in line and "my_clip" in line
    bare = tmp_path / "bare"
    bare.mkdir()
    assert "V2D" not in mod.credit_line(bare)
    # frames read from --video carry the V2D credit only when the file is the example clip byte for byte
    other = tmp_path / "other.mp4"
    other.write_bytes(b"not the example clip")
    line = mod.credit_line(REFERENCE, other)
    assert "V2D" not in line and "NVIDIA" not in line
    # ... and the example's clip ID is named as the OVERLAY's reconstruction, never as the source of those frames
    assert line.startswith("Frames from the supplied video (credit and licence supplied by its owner).")
    assert line.endswith("Overlay from reconstruction v2d_t3_ep011_cam_a.") and line.count("v2d_t3_ep011_cam_a") == 1
    assert "supplied video" in mod.credit_line(own, other) and "Frames: clip my_clip" in mod.credit_line(own)
    assert len(mod.EXAMPLE_VIDEO_SHA256) == 64
    pinned = json.loads((REPO / "release" / "example" / "example_input.json").read_text())
    assert mod.EXAMPLE_VIDEO_SHA256 in json.dumps(pinned)
    same = tmp_path / "same.bin"
    same.write_bytes(b"x")
    mod.EXAMPLE_VIDEO_SHA256 = mod._sha256(same)
    assert mod.credit_line(REFERENCE, same) == mod.CREDIT


@pytest.mark.parametrize("change, message", [
    ("frame_idx_repeat", "increasing"),
    ("frame_idx_descending_unsigned", "increasing"),
    ("frame_idx_beyond_int64", "increasing"),
    ("frame_idx_float", "increasing"),
    ("poses_short", "poses have shape"),
    ("hands_short", "one row per"),
])
def test_render_refuses_arrays_that_do_not_share_the_solve_rows(tmp_path: Path, change: str, message: str) -> None:
    """A clip directory whose arrays disagree on the rows is refused with one line before any row is indexed."""
    import shutil
    import subprocess

    clip = tmp_path / "clip"
    shutil.copytree(REFERENCE, clip)
    with np.load(clip / "solve.npz", allow_pickle=False) as z:
        solve = {k: z[k] for k in z.files}
    with np.load(clip / "hands.npz", allow_pickle=False) as z:
        hands = {k: z[k] for k in z.files}
    if change == "frame_idx_repeat":
        solve["frame_idx"] = solve["frame_idx"].copy()
        solve["frame_idx"][1] = solve["frame_idx"][0]
    elif change == "frame_idx_descending_unsigned":            # a difference of these wraps to a large positive number
        solve["frame_idx"] = solve["frame_idx"][::-1].astype(np.uint64)
    elif change == "frame_idx_beyond_int64":
        solve["frame_idx"] = solve["frame_idx"].astype(np.uint64)
        solve["frame_idx"][-1] = np.uint64(2 ** 63 + 5)
    elif change == "frame_idx_float":
        solve["frame_idx"] = solve["frame_idx"].astype(np.float64)
    elif change == "poses_short":
        solve["poses"] = solve["poses"][:-1]
    else:
        side = next(k[:-7] for k in hands if k.endswith("_joints"))
        hands[f"{side}_joints"] = hands[f"{side}_joints"][:-1]
    np.savez(clip / "solve.npz", **solve)
    np.savez(clip / "hands.npz", **hands)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "render_reference_png.py"), str(clip), "--out",
                        str(tmp_path / "x.png")], capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 1 and message in r.stderr and "Traceback" not in r.stderr
