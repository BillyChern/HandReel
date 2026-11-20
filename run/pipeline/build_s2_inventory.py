"""Builds the S2 window inventory from a frozen draw, and reports the corpus the yield law divides by.

The inventory itself is a run input that holds absolute paths, so write it outside version control,
and THIS script is the reproducible artefact. `bench.windows_inventory` then points the registered runner at
the file; the windows themselves are cut by `fastdo.bench.bench_clips.windowed_clips_from_inventory`, so
nothing about the sampling is editable here -- this names which videos take part, which the draw already fixed.

**Planning numbers vs run numbers.** The window COUNTS printed here use the draw's recorded `duration_sec`,
because probing every video's decoded-frame count costs a full decode pass per file. The RUN uses the
pre-registration's own definition (`D_v` = decoded frames / avg_frame_rate, `probe_windowed_clips`). On the
one video checked both ways these agreed to the centisecond (1033.13 s), but the printed counts are still
PLANNING figures and a window near a tail boundary can move.

Usage:
    python run/pipeline/build_s2_inventory.py --draw <draw.json> \\
        --videos <directory of <video_uid>.mp4> --source ego4d --out <windows_inventory.json>

**EK-100 is deliberately not wired in here.** Its TEST videos may not be opened before S2 Freeze-1, and even
a metadata probe opens the file; add it after the freeze.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from fastdo.bench.bench_clips import WINDOW_SECONDS, operating_stride, window_offset
from fastdo.io.video import probe_fps

logger = logging.getLogger(__name__)

SPLITS = ("TEST", "CALIB")


def plan_rows(draw: dict, videos: Path, source: str) -> tuple[list[dict], list[str]]:
    """One row per video of the draw that is present on disk, plus the uids that are missing.

    `fps` is a metadata-only ffprobe (`avg_frame_rate`): it reads the container header, not frames.
    """
    rows: list[dict] = []
    missing: list[str] = []
    for split in SPLITS:
        for entry in draw.get(split, []):
            uid = str(entry["video_uid"])
            path = videos / f"{uid}.mp4"
            if not path.is_file():
                missing.append(uid)
                continue
            fps = probe_fps(path)
            stride = operating_stride(fps)
            windows = max(0, int((float(entry["duration_sec"]) - window_offset(uid)) // WINDOW_SECONDS))
            rows.append({"video_id": uid, "path": str(path), "source": source, "split": split,
                         "fps": fps, "s_v": stride, "duration_s": float(entry["duration_sec"]),
                         "n_windows": windows,
                         "rows": windows * len(range(0, round(WINDOW_SECONDS * fps), stride))})
    return rows, missing


def report(rows: list[dict]) -> str:
    """The corpus as the yield law sees it: windows are the denominator, decode rows the front-end load."""
    lines = [f"fps {sorted({r['fps'] for r in rows})} -> strides {sorted({r['s_v'] for r in rows})}"]
    for split in (*SPLITS, "ALL"):
        group = rows if split == "ALL" else [r for r in rows if r["split"] == split]
        if not group:
            continue
        lines.append(f"{split:5s}: {len(group):3d} videos  {sum(r['duration_s'] for r in group) / 3600:6.2f} h  "
                     f"{sum(r['n_windows'] for r in group):5d} windows  "
                     f"{sum(r['rows'] for r in group):7d} decode rows")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=Path, required=True, help="the frozen draw (TEST/CALIB video lists)")
    parser.add_argument("--videos", type=Path, required=True, help="directory holding <video_uid>.mp4")
    parser.add_argument("--source", required=True, help="manifest source for these clips, e.g. ego4d")
    parser.add_argument("--out", type=Path, required=True, help="inventory to write")
    args = parser.parse_args()

    rows, missing = plan_rows(json.loads(args.draw.read_text()), args.videos, args.source)
    if missing:
        logger.warning("%d of the draw's videos are not on disk: %s", len(missing), missing[:5])
    if not rows:
        raise SystemExit(f"{args.draw}: none of the draw's videos are under {args.videos}")
    logger.info("%s", report(rows))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([{k: r[k] for k in ("video_id", "path", "source", "split")} for r in rows],
                                   indent=1))
    logger.info("inventory -> %s (%d videos)", args.out, len(rows))


if __name__ == "__main__":
    main()
