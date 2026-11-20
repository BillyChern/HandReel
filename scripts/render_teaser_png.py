"""Draw the README picture: three frames of a clip (top row) and the same frames with the reconstruction drawn over
them (bottom row). It reads an engine clip directory and the source video, like `render_reference_png.py`, whose
checks and projection it reuses; nothing is retouched and every panel is the full crop of one decoded frame.

    python scripts/render_teaser_png.py release/reference/v2d_t3_ep011_cam_a --video <the clip> --out docs/assets/teaser.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_reference_png as RR

OBJ_RGB = (106, 61, 154)                       # the paper's colour for HandReel
HAND_RGB = {"left": (217, 95, 2), "right": (27, 158, 119)}
ALPHA = 0.55


def main(argv: list[str] | None = None) -> int:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", default="0,34,68")
    ap.add_argument("--fps", type=float, default=20.0, help="frame rate of the source video, for the time labels")
    a = ap.parse_args(argv)
    rows = [int(x) for x in a.rows.split(",")]
    solve, hands = RR._npz(a.clip_dir / "solve.npz"), RR._npz(a.clip_dir / "hands.npz")
    fr = np.asarray(solve["frame_idx"]).astype(np.int64)
    poses, scale = np.asarray(solve["poses"], np.float64), float(np.asarray(solve["scale"]))
    K = np.asarray(RR._npz(a.clip_dir / "intrinsics.npz")["K"], np.float64)
    mesh = trimesh.load(str(a.clip_dir / "mesh.obj"), process=False, force="mesh")
    mv, mf = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    frames = RR._frames_from_video(a.video, [int(fr[r]) for r in rows])

    raw, over, pts = [], [], []
    for r in rows:
        im = cv2.cvtColor(frames[int(fr[r])], cv2.COLOR_BGR2RGB)
        P = poses[r]
        uv = RR.project(K, scale * mv @ P[:3, :3].T + P[:3, 3])
        mask = np.zeros(im.shape[:2], np.uint8)
        cv2.fillPoly(mask, list(np.round(uv[mf]).astype(np.int32)), 255)
        out = im.copy()
        out[mask > 0] = ((1 - ALPHA) * out[mask > 0] + ALPHA * np.asarray(OBJ_RGB)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, OBJ_RGB, 4, lineType=cv2.LINE_AA)
        pts.append(uv)
        for side, rgb in HAND_RGB.items():
            if f"{side}_joints" in hands and bool(hands[f"{side}_valid"][r]):
                j = RR.project(K, np.asarray(hands[f"{side}_joints"][r], np.float64))
                pts.append(j)
                for x, y in np.round(j).astype(int):
                    cv2.circle(out, (int(x), int(y)), 9, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(out, (int(x), int(y)), 6, rgb, -1, lineType=cv2.LINE_AA)
        raw.append(im)
        over.append(out)

    # one crop for all six panels: the box of everything drawn, padded, at 4:3, inside the frame
    H, W = raw[0].shape[:2]
    allp = np.concatenate(pts)
    x0, y0, x1, y1 = allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()
    cx, cy, w = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, (y1 - y0) * 4 / 3) * 1.25
    w = min(w, W, H * 4 / 3)
    h = w * 3 / 4
    left = int(np.clip(cx - w / 2, 0, W - w))
    top = int(np.clip(cy - h / 2, 0, H - h))
    box = (slice(top, top + int(h)), slice(left, left + int(w)))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.35), dpi=150)
    for c, r in enumerate(rows):
        for row, imgs in enumerate((raw, over)):
            ax = axes[row, c]
            ax.imshow(imgs[c][box])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, c].set_title(f"t = {int(fr[r]) / a.fps:.1f} s", fontsize=13, pad=6)
    axes[0, 0].set_ylabel("Input video", fontsize=14, labelpad=8)
    axes[1, 0].set_ylabel("HandReel reconstruction", fontsize=14, labelpad=8)
    fig.text(0.5, 0.012, RR.credit_line(a.clip_dir, a.video).replace("Overlay: this clip directory's reconstruction.",
             "Purple: reconstructed object. Dots: hand joints."), ha="center", fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.95, bottom=0.05, wspace=0.02, hspace=0.03)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor="white", pil_kwargs={"quality": 90} if a.out.suffix == ".jpg" else {})
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
