#!/usr/bin/env python3
"""Draw one clip directory's reconstruction over its source frames at a few rows, as one PNG.

    uv run --frozen python scripts/render_reference_png.py <clip dir> --out <file.png> [--video <source video>]
        [--rows 0,34,68] [--panel-width 480]

`<clip dir>` is an engine clip directory (`<--out>/engine/<clip_id>/` of `release/example/run_example.py`) or the
shipped reference output `release/reference/v2d_t3_ep011_cam_a/`. For each requested decode row it projects, with
`intrinsics.npz` `K` (a camera-frame point (X, Y, Z) goes to the pixel (fx X / Z + cx, fy Y / Z + cy),
`docs/output.md`, "Units and frames"):

- the object mesh, `mesh.obj` placed by `solve.npz` `poses` and `scale` (a vertex y at `scale * R @ y + t`), as a
  filled silhouette with its outline;
- the hand joints, `hands.npz` `{side}_joints`, as dots, on the rows the run marked `{side}_valid`.

The PNG projects the solve poses. The interactive viewer uses contact-refined poses when available.

The background is the source frame of that row: `decode.npz` `frames` when the directory has it, else the frame of
`--video` at the row's `frame_idx`, else a grey canvas. Nothing is fitted or smoothed. The line under the panels
credits the frames to the V2D Challenge (CC BY 4.0) only when the directory's `source_domain.json` names the example's
source (`v2d_track3_public`) and, when the frames come from `--video`, that file's sha256 is the example clip's
(`EXAMPLE_VIDEO_SHA256`). Under frames decoded inside another clip's directory it names the clip and leaves its
licence and attribution to you; under frames of a `--video` file it cannot verify it says only that they come from
the supplied video and names the reconstruction as the overlay's source. The shipped
`release/reference/v2d_t3_ep011_cam_a/expected_projection.png` was drawn by this script from the shipped reference
output and the example clip (`--video`), with the default rows and panel width. It is a picture to compare with by
eye, not a numeric reference.

CPU only; needs numpy, opencv and trimesh (all in the main env). Exit 1 with a one-line message on an input error.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np

OBJ_BGR = (150, 60, 230)                 # magenta: distinct from the scene's blue and beige cups
HAND_BGR = {"right": (60, 210, 60), "left": (0, 200, 255)}
ALPHA = 0.45
V2D_SOURCE = "v2d_track3_public"         # `source` of the example clip (release/example/example_input.json)
# sha256 of the example clip (release/example/example_input.json); the V2D credit goes under --video frames only then.
EXAMPLE_VIDEO_SHA256 = "508a4fc94747741f35fd12780e6d5e6e7f78a233e2bf379202515cb9bff32585"
CREDIT = ("Frames: Video to Data (V2D) Challenge, Track 3 public split, NVIDIA, CC BY 4.0. "
          "Overlay: this clip directory's reconstruction.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def credit_line(clip_dir: Path, video: Path | None = None) -> str:
    """The text under the panels when they show source frames: the V2D credit only for the example's source, and,
    when the frames were read from `video`, only when that file is the example clip byte for byte."""
    import json

    try:
        rec = json.loads((clip_dir / "source_domain.json").read_text())
    except (OSError, ValueError):
        rec = {}
    if isinstance(rec, dict) and rec.get("source") == V2D_SOURCE and \
            (video is None or _sha256(video) == EXAMPLE_VIDEO_SHA256):
        return CREDIT
    name = rec.get("clip_id", clip_dir.name) if isinstance(rec, dict) else clip_dir.name
    if video is not None:                    # an unverified file: its frames are not known to be this clip's
        return ("Frames from the supplied video (credit and licence supplied by its owner). "
                f"Overlay from reconstruction {name}.")
    return f"Frames: clip {name} (its licence and attribution are yours to state). Overlay: its reconstruction."


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as e:
        raise ValueError(f"{path} is not a readable npz file ({type(e).__name__})") from None


def _frames_from_video(path: Path, want: list[int]) -> dict[int, np.ndarray]:
    """Decode `path` from frame 0 in order (no seeking) and keep the frame numbers in `want` (BGR)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"{path}: not a readable video")
    out: dict[int, np.ndarray] = {}
    wanted, i = set(want), 0
    while i <= max(want):
        ok, im = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = im
        i += 1
    cap.release()
    missing = sorted(wanted - set(out))
    if missing:
        raise ValueError(f"{path}: frames {missing} not decoded (the video has {i} frames)")
    return out


def project(K: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixels; the points must be in front of the camera."""
    if (pts[:, 2] <= 0).any():
        raise ValueError("a point is at or behind the camera")
    return np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2], K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], 1)


def render(clip_dir: Path, rows: list[int], video: Path | None, panel_width: int) -> np.ndarray:
    """The PNG image (BGR): one panel per row, side by side, and a credit line under them."""
    import cv2
    import trimesh

    for name in ("mesh.obj", "solve.npz", "intrinsics.npz"):
        if not (clip_dir / name).is_file() or (clip_dir / name).stat().st_size == 0:
            raise ValueError(f"{clip_dir / name} is missing or empty (a --stub run writes no reconstruction)")
    solve = _npz(clip_dir / "solve.npz")
    for key in ("frame_idx", "poses", "scale"):
        if key not in solve:
            raise ValueError(f"{clip_dir / 'solve.npz'} has no '{key}'")
    fr = np.asarray(solve["frame_idx"])
    # compared element to element, never subtracted: a difference of unsigned IDs wraps instead of going negative
    if fr.ndim != 1 or not np.issubdtype(fr.dtype, np.integer) or len(fr) == 0 or (fr < 0).any() or \
            (fr[1:] <= fr[:-1]).any() or int(fr.max()) > np.iinfo(np.int64).max:
        raise ValueError("solve.npz frame_idx must be a non-empty 1-D array of increasing non-negative integers")
    fr = fr.astype(np.int64)
    bad = [r for r in rows if not 0 <= r < len(fr)]
    if bad:
        raise ValueError(f"rows {bad} are outside 0..{len(fr) - 1}")
    poses, scale = np.asarray(solve["poses"], np.float64), float(np.asarray(solve["scale"]))
    if poses.shape != (len(fr), 4, 4):
        raise ValueError(f"solve.npz poses have shape {poses.shape}, not ({len(fr)}, 4, 4)")
    K = np.asarray(_npz(clip_dir / "intrinsics.npz")["K"], np.float64)
    mesh = trimesh.load(str(clip_dir / "mesh.obj"), process=False, force="mesh")
    mv, mf = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    hands = _npz(clip_dir / "hands.npz") if (clip_dir / "hands.npz").is_file() else {}
    if "frame_idx" in hands and not np.array_equal(np.asarray(hands["frame_idx"]), fr):
        raise ValueError("hands.npz and solve.npz have different frame_idx")
    for side in HAND_BGR:
        if f"{side}_joints" not in hands:
            continue
        j, v = np.asarray(hands[f"{side}_joints"]), np.asarray(hands.get(f"{side}_valid", ()))
        if j.ndim != 3 or j.shape[0] != len(fr) or j.shape[2] != 3 or v.shape != (len(fr),):
            raise ValueError(f"hands.npz {side}_joints and {side}_valid must have one row per solve.npz frame_idx")
    W, H = round(2 * K[0, 2]), round(2 * K[1, 2])

    images: dict[int, np.ndarray] = {}
    if (clip_dir / "decode.npz").is_file():
        dec = _npz(clip_dir / "decode.npz")
        if "frames" in dec and dec["frames"].ndim == 4:
            if len(dec["frames"]) != len(fr) or \
                    ("frame_idx" in dec and not np.array_equal(np.asarray(dec["frame_idx"]), fr)):
                raise ValueError("decode.npz frames and solve.npz frame_idx do not describe the same rows")
            images = {int(fr[r]): cv2.cvtColor(dec["frames"][r], cv2.COLOR_RGB2BGR) for r in rows}
    from_video = not images and video is not None
    if from_video:
        images = _frames_from_video(video, [int(fr[r]) for r in rows])

    panels = []
    for r in rows:
        f = int(fr[r])
        im = images.get(f)
        im = np.full((H, W, 3), 96, np.uint8) if im is None else im.copy()
        P = poses[r]
        uv = project(K, scale * mv @ P[:3, :3].T + P[:3, 3])
        mask = np.zeros(im.shape[:2], np.uint8)
        cv2.fillPoly(mask, list(np.round(uv[mf]).astype(np.int32)), 255)
        im[mask > 0] = ((1 - ALPHA) * im[mask > 0] + ALPHA * np.asarray(OBJ_BGR)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, contours, -1, OBJ_BGR, max(2, W // 400))
        for side, bgr in HAND_BGR.items():
            if f"{side}_joints" in hands and bool(hands[f"{side}_valid"][r]):
                for x, y in np.round(project(K, np.asarray(hands[f"{side}_joints"][r], np.float64))).astype(int):
                    cv2.circle(im, (int(x), int(y)), max(4, W // 200), bgr, -1, lineType=cv2.LINE_AA)
        s = panel_width / im.shape[1]
        im = cv2.resize(im, (panel_width, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"row {r} (source frame {f})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        panels.append(im)
    strip = np.concatenate(panels, axis=1)
    foot = np.full((32, strip.shape[1], 3), 255, np.uint8)
    text = (credit_line(clip_dir, video if from_video else None) if images
            else "No source frames given (--video): grey canvas. Overlay: the reconstruction.")
    cv2.putText(foot, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([strip, foot], axis=0)


def main(argv: list[str] | None = None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip_dir", type=Path, help="an engine clip directory (mesh.obj, solve.npz, intrinsics.npz, ...)")
    ap.add_argument("--out", type=Path, required=True, help="the PNG file to write")
    ap.add_argument("--video", type=Path, help="the source video, for the frames when the directory has no decode.npz")
    ap.add_argument("--rows", default="0,34,68", help="decode rows, comma-separated, 0-based (default 0,34,68)")
    ap.add_argument("--panel-width", type=int, default=480, help="width of each panel in pixels (default 480)")
    a = ap.parse_args(argv)
    try:
        rows = [int(x) for x in a.rows.split(",") if x.strip()]
        img = render(a.clip_dir, rows, a.video, a.panel_width)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"render_reference_png: {e}") from None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(a.out), img, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise SystemExit(f"render_reference_png: could not write {a.out}")
    print(f"render_reference_png: wrote {a.out} ({img.shape[1]}x{img.shape[0]}, rows {rows})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
