#!/usr/bin/env python3
"""View one reconstructed clip in the browser: object and hands in 3D over the source frame, frame by frame.

    uv run --frozen --extra viser python scripts/view_clip.py <clip dir> [--video <source video>] [--port 8080]

`<clip dir>` is an engine clip directory (`<--out>/engine/<clip_id>/` of `release/example/run_example.py`, or the
shipped reference output `release/reference/v2d_t3_ep011_cam_a/`). The script starts a Viser server on 127.0.0.1
(open the printed URL in a browser; `--host 0.0.0.0` to serve other machines) with play/pause, a frame slider and a
playback-rate box; drag to orbit, scroll to zoom. Ctrl-C stops it.

What it reads (nothing is fitted, smoothed or interpolated; every shown frame is one of the run's processed frames):
- `mesh.obj` (canonical units) and `solve.npz` `poses` and `scale`: a mesh vertex y is at `scale * R @ y + t` in the
  camera frame (`docs/output.md`). When the run wrote `contact.npz`, its `poses` are used instead, as the site loader
  does; the scale is always `solve.npz`'s.
- `hands.npz`: `{side}_vertices` of each row the run marked `{side}_valid`, drawn as a mesh with `{side}_faces`, or
  as points when the file has no faces; a file without vertices (the shipped reference output, see its README) has
  its 21 `{side}_joints` drawn as larger points instead.
- `intrinsics.npz` `K`: the source frame is drawn on an image plane behind the scene, sized and offset with `K` so that
  seen from the camera centre it covers the same rays as the reconstruction. The initial view starts at the camera
  centre and looks at the object, with a field of view near the camera's, so the 3D starts roughly over the frame;
  in the browser it need not sit exactly on the source pixels (an offset of the mesh against the frame has been seen
  in the initial view). For an exact overlay, draw the rows with `scripts/render_reference_png.py`.

Colours: the object magenta, the right hand green and the left hand orange, as in `render_reference_png.py`.
- the source frame of each row: `decode.npz` `frames` when the clip directory has it (the frames the stages saw),
  else the frames of `--video` at `frame_idx`, else none. Every row's frames are checked against `frame_idx`.

The world frame is the camera frame (OpenCV axes: x right, y down, z forward), in metres. Adapted from the project
site's viewer (one group of nodes per frame, created up front; playback only switches which group is visible, so the
image, the 3D and the frame number of one step always come from the same frame).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

OBJ_RGB = (230, 60, 150)                 # magenta: distinct from the scene colours of the example clip's cups
HAND_RGB = {"right": (60, 210, 60), "left": (255, 200, 0)}    # green and orange, as render_reference_png.py
TEX_MAX_WIDTH = 960                      # source frames are resized to at most this width for the browser


@dataclass
class Frame:
    fid: int                             # source frame index (`frame_idx`)
    obj_v: np.ndarray                    # (V, 3) object vertices, camera frame, metres
    hands: list = field(default_factory=list)    # [(side, (778, 3) vertices or (21, 3) joints)]
    rgb: np.ndarray | None = None        # (H, W, 3) uint8 source frame, or None


@dataclass
class Clip:
    name: str
    K: np.ndarray
    obj_f: np.ndarray
    hand_f: dict                         # side -> (F, 3) faces, or None when the file has none
    frames: list
    pose_source: str
    image_source: str


def _npz(path: Path) -> dict[str, np.ndarray]:
    """Every array of an npz file; a file numpy cannot read is a ValueError that names it."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as e:
        raise ValueError(f"{path} is not a readable npz file ({type(e).__name__}); was the run complete?") from None


def video_frames(path: Path, want: list[int]) -> dict[int, np.ndarray]:
    """Decode `path` from frame 0 in order (no seeking) and keep the frame numbers in `want`, as RGB."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"{path}: not a readable video")
    out: dict[int, np.ndarray] = {}
    wanted, last, i = set(want), max(want), 0
    while i <= last:
        ok, im = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    missing = sorted(wanted - set(out))
    if missing:
        raise ValueError(f"{path}: frames {missing[:5]} not decoded (the video has {i} frames)")
    return out


def load_clip(clip_dir: Path, video: Path | None = None) -> Clip:
    """Everything the viewer shows, checked for consistency.

    Raises:
        FileNotFoundError: If `mesh.obj`, `solve.npz` or `intrinsics.npz` is missing.
        ValueError: If the files disagree on `frame_idx`, or an object or hand vertex is at or behind the camera.
    """
    import trimesh

    engines = sorted(p for p in (clip_dir / "engine").glob("*") if p.is_dir()) if (clip_dir / "engine").is_dir() else []
    for name in ("mesh.obj", "solve.npz", "intrinsics.npz"):
        if not (clip_dir / name).is_file():
            hint = (f"; this looks like a run's --out directory: pass {engines[0]}" if engines else
                    ": is it an engine clip directory, and did the run finish its solve stage?")
            raise FileNotFoundError(f"{clip_dir / name} not found{hint}")
    empty = [n for n in ("mesh.obj", "solve.npz", "intrinsics.npz") if (clip_dir / n).stat().st_size == 0]
    if empty:
        raise ValueError(f"{', '.join(empty)} in {clip_dir} are empty placeholder files: a --stub run writes no "
                         "reconstruction to view; view a GPU run or release/reference/v2d_t3_ep011_cam_a")
    solve = _npz(clip_dir / "solve.npz")
    fr = np.asarray(solve["frame_idx"], np.int64)
    poses, pose_source = np.asarray(solve["poses"], np.float64), "solve.npz"
    if (clip_dir / "contact.npz").is_file():
        contact = _npz(clip_dir / "contact.npz")
        if not np.array_equal(np.asarray(contact["frame_idx"], np.int64), fr):
            raise ValueError("contact.npz and solve.npz have different frame_idx")
        poses, pose_source = np.asarray(contact["poses"], np.float64), "contact.npz"
    scale = float(np.asarray(solve["scale"]))
    K = np.asarray(_npz(clip_dir / "intrinsics.npz")["K"], np.float64)
    mesh = trimesh.load(str(clip_dir / "mesh.obj"), process=False, force="mesh")
    mv, mf = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)

    hands: dict[str, np.ndarray] = {}
    if (clip_dir / "hands.npz").is_file():
        hands = _npz(clip_dir / "hands.npz")
        if not np.array_equal(np.asarray(hands["frame_idx"], np.int64), fr):
            raise ValueError("hands.npz and solve.npz have different frame_idx")
    hand_key = {s: (f"{s}_vertices" if f"{s}_vertices" in hands else f"{s}_joints") for s in ("right", "left")}
    sides = [s for s in ("right", "left") if hand_key[s] in hands]
    hand_f = {s: (np.asarray(hands[f"{s}_faces"], np.int64) if f"{s}_faces" in hands and hand_key[s].endswith(
        "_vertices") else None) for s in sides}

    rgbs: dict[int, np.ndarray] = {}
    image_source = "none"
    if (clip_dir / "decode.npz").is_file():
        decode = _npz(clip_dir / "decode.npz")
        if "frames" in decode and decode["frames"].ndim == 4:
            if not np.array_equal(np.asarray(decode["frame_idx"], np.int64), fr):
                raise ValueError("decode.npz and solve.npz have different frame_idx")
            rgbs = {int(f): decode["frames"][i] for i, f in enumerate(fr)}
            image_source = "decode.npz"
    if not rgbs and video is not None:
        rgbs = video_frames(video, [int(f) for f in fr])
        image_source = video.name

    frames = []
    for i, f in enumerate(fr):
        P = poses[i]
        ov = scale * mv @ P[:3, :3].T + P[:3, 3]
        hs = [(s, np.asarray(hands[hand_key[s]][i], np.float64)) for s in sides if bool(hands[f"{s}_valid"][i])]
        if (ov[:, 2] <= 0).any() or any((v[:, 2] <= 0).any() for _, v in hs):
            raise ValueError(f"frame {int(f)}: a vertex is at or behind the camera; not a camera-frame clip?")
        frames.append(Frame(int(f), ov, hs, rgbs.get(int(f))))
    return Clip(clip_dir.name, K, mf, hand_f, frames, pose_source, image_source)


def image_size(clip: Clip) -> tuple[int, int]:
    """`(W, H)` of the source frames: from the frames when there are any, else from the principal point."""
    for fr in clip.frames:
        if fr.rgb is not None:
            return int(fr.rgb.shape[1]), int(fr.rgb.shape[0])
    return round(2 * clip.K[0, 2]), round(2 * clip.K[1, 2])


def plane_depth(clip: Clip) -> float:
    """Depth of the image plane: behind every object and hand vertex of the clip."""
    z = max(float(np.max(f.obj_v[:, 2])) for f in clip.frames)
    for f in clip.frames:
        for _, v in f.hands:
            z = max(z, float(np.max(v[:, 2])))
    return round(1.25 * z + 0.05, 3)


def build_scene(server: Any, clip: Clip) -> list[Any]:
    """One hidden group per frame (object, hands, source image, frame label); returns the groups in frame order."""
    import cv2

    W, H = image_size(clip)
    fx, fy = clip.K[0, 0], clip.K[1, 1]
    cx, cy = clip.K[0, 2], clip.K[1, 2]
    D = plane_depth(clip)
    centre = np.mean([f.obj_v.mean(0) for f in clip.frames], axis=0)
    server.scene.set_up_direction("-y")
    server.initial_camera.position = (0.0, 0.0, 0.0)
    server.initial_camera.look_at = tuple(float(x) for x in centre)
    server.initial_camera.up = (0.0, -1.0, 0.0)
    server.initial_camera.fov = float(2 * np.arctan(H / 2 / fy) * 1.08)
    server.initial_camera.near = 0.01
    server.initial_camera.far = 20.0
    zc = float(centre[2])
    label_pos = (float((0.04 * W - cx) / fx * zc), float((0.05 * H - cy) / fy * zc), zc)
    groups = []
    for k, fr in enumerate(clip.frames):
        g = server.scene.add_frame(f"/f{fr.fid:05d}", show_axes=False, visible=(k == 0))
        server.scene.add_mesh_simple(f"{g.name}/object", fr.obj_v.astype(np.float32), clip.obj_f.astype(np.uint32),
                                     color=OBJ_RGB, side="double")
        for side, hv in fr.hands:
            faces = clip.hand_f[side]
            if faces is None:
                server.scene.add_point_cloud(f"{g.name}/hand_{side}", hv.astype(np.float32),
                                             colors=np.tile(np.asarray(HAND_RGB[side], np.uint8), (len(hv), 1)),
                                             point_size=0.0015 if len(hv) > 21 else 0.006)
            else:
                server.scene.add_mesh_simple(f"{g.name}/hand_{side}", hv.astype(np.float32), faces.astype(np.uint32),
                                             color=HAND_RGB[side], side="double")
        if fr.rgb is not None:
            im = fr.rgb
            if im.shape[1] > TEX_MAX_WIDTH:
                s = TEX_MAX_WIDTH / im.shape[1]
                im = cv2.resize(im, (TEX_MAX_WIDTH, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            # The plane spans the image at depth D in the camera's view; its centre is the image centre's ray.
            server.scene.add_image(f"{g.name}/rgb", im, render_width=float(W * D / fx), render_height=float(H * D / fy),
                                   format="jpeg", jpeg_quality=88,
                                   position=(float((W / 2 - cx) / fx * D), float((H / 2 - cy) / fy * D), D))
        server.scene.add_label(f"{g.name}/label", f"frame {fr.fid}", position=label_pos)
        groups.append(g)
    return groups


def serve(clip: Clip, host: str, port: int, fps: float, run_seconds: float | None = None) -> int:
    """Start the server, add the controls, and play until Ctrl-C (or for `run_seconds`, for tests)."""
    import viser

    server = viser.ViserServer(host=host, port=port, verbose=False)
    groups = build_scene(server, clip)
    n = len(groups)
    state = {"shown": 0}
    lock = threading.Lock()
    playing = server.gui.add_checkbox("Play", initial_value=True)
    slider = server.gui.add_slider("Row (0-based)", min=0, max=max(n - 1, 1), step=1, initial_value=0,
                                   disabled=n == 1)
    rate = server.gui.add_number("Playback fps", initial_value=fps, min=0.5, max=60.0, step=0.5)
    info = server.gui.add_markdown("")

    def show(k: int) -> None:
        with lock:
            if k != state["shown"]:
                groups[state["shown"]].visible = False
            groups[k].visible = True
            state["shown"] = k
            fr = clip.frames[k]
            info.content = (f"**{clip.name}**  \nsource frame {fr.fid} (row {k} of rows 0-{n - 1})  \n"
                            f"pose: {clip.pose_source}; image: {clip.image_source}; "
                            f"hands: {', '.join(s for s, _ in fr.hands) or 'none valid'}")

    @slider.on_update
    def _(_event: Any) -> None:
        show(int(slider.value))

    show(0)
    print(f"view_clip: {clip.name}, {n} frames; serving http://{host}:{server.get_port()} (viewer only: nothing is "
          "reconstructed here, and viewer time is not reconstruction time)", flush=True)
    start = time.monotonic()
    try:
        while run_seconds is None or time.monotonic() - start < run_seconds:
            time.sleep(1.0 / max(float(rate.value), 0.5))
            if playing.value and n > 1:
                slider.value = (int(slider.value) + 1) % n
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip_dir", type=Path, help="an engine clip directory (mesh.obj, solve.npz, intrinsics.npz, ...)")
    ap.add_argument("--video", type=Path, help="the source video, for the frames when the directory has no decode.npz")
    ap.add_argument("--host", default="127.0.0.1", help="interface to serve on (default 127.0.0.1: this machine only)")
    ap.add_argument("--port", type=int, default=8080, help="port (default 8080)")
    ap.add_argument("--fps", type=float, default=10.0, help="initial playback rate in rows per second (default 10)")
    ap.add_argument("--run-seconds", type=float, default=None, help=argparse.SUPPRESS)   # tests: stop after this
    a = ap.parse_args(argv)
    try:
        clip = load_clip(a.clip_dir, a.video)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"view_clip: {e}") from None
    try:
        import viser  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit("view_clip: viser is not installed: uv sync --frozen --extra viser (README, View the "
                         "result)") from None
    return serve(clip, a.host, a.port, a.fps, a.run_seconds)


if __name__ == "__main__":
    sys.exit(main())
