"""HOI4D GT right hands as HaWoR-format `all_hand_meshes.npz`, one per sequence of a window list (CPU).

    PYTHONPATH=src python run/pipeline/build_hoi4d_hands.py windows.json --pool dev \
        --out-root "$FASTDO_ROOT/outputs/hoi4d_hands" \
        --hawor-python /path/to/hawor/bin/python --mano-right-pkl /path/to/MANO_RIGHT.pkl

Each sequence becomes `<out-root>/<seq>/all_hand_meshes.npz` plus `all_hand_meshes_validation.json`, made by
`_hoi4d_hands_driver.py` in the hawor env (CUDA hidden, 2 threads); the window's object registration shift is
passed for a diagnostic only. GT only: no
pipeline output is read or written. A failing sequence keeps its record with `pass` False; nothing is replaced.
`fastdo.io.datasets.hoi4d.hand_track` reads the file into a `HandTrack`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1].parent
DRIVER = Path(__file__).resolve().parent / "_hoi4d_hands_driver.py"


def nonempty_path(value: str) -> Path:
    if not value:
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value)


def hands_npz(out_root: Path, seq: str) -> Path:
    return Path(out_root) / seq / "all_hand_meshes.npz"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows", type=Path, help="a split file ({windows: [...]}) or a list of window entries")
    ap.add_argument("--pool", default="dev", help="only windows of this pool (FINAL_TEST is refused)")
    ap.add_argument("--clip-ids", nargs="*", default=None)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("FASTDO_ROOT") or REPO),
                    help="project root (default: FASTDO_ROOT or this checkout)")
    ap.add_argument("--hawor-python", type=nonempty_path, default=os.environ.get("FASTDO_HAWOR_PYTHON") or None,
                    help="HaWoR environment interpreter (or FASTDO_HAWOR_PYTHON)")
    ap.add_argument("--hawor-root", type=nonempty_path, default=os.environ.get("FASTDO_HAWOR_ROOT") or None,
                    help="HaWoR checkout (or FASTDO_HAWOR_ROOT; default: under project root)")
    ap.add_argument("--mano-right-pkl", type=nonempty_path,
                    default=(Path(os.environ["FASTDO_MANO_DIR"]) / "MANO_RIGHT.pkl"
                             if os.environ.get("FASTDO_MANO_DIR") else None),
                    help="user-provided MANO_RIGHT.pkl (or FASTDO_MANO_DIR)")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.out_root is None:
        args.out_root = args.root / "outputs/hoi4d_hands"
    if args.hawor_root is None:
        args.hawor_root = args.root / "third_party/do-as-i-do/reconstruction/modules/HaWoR"
    if args.pool.strip().casefold() == "final_test":
        raise SystemExit("REFUSING: FINAL_TEST windows are held out")
    data = json.loads(args.windows.read_text())
    windows = [w for w in (data["windows"] if isinstance(data, dict) else data) if w.get("pool", "dev") == args.pool]
    if args.clip_ids:
        windows = [w for w in windows if w["clip_id"] in set(args.clip_ids)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2"}
    failed = 0
    for w in windows:
        out = hands_npz(args.out_root, w["sequence"])
        if out.is_file() and not args.force:
            print(f"{w['sequence']}: exists")
            continue
        if not args.hawor_python or not args.mano_right_pkl:
            ap.error("set --hawor-python/FASTDO_HAWOR_PYTHON and --mano-right-pkl/FASTDO_MANO_DIR")
        dx, dy = w["shift_px"]
        cmd = [str(args.hawor_python), str(DRIVER), w["sequence"], str(out), "--hoi4d-root", str(args.root / "data/hoi4d"),
               "--hawor-root", str(args.hawor_root), "--mano-right-pkl", str(args.mano_right_pkl), "--shift-dx", str(dx),
               "--shift-dy", str(dy)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False, timeout=900)
        failed += r.returncode != 0
        print(f"{w['sequence']}: rc={r.returncode} {(r.stdout + r.stderr).strip().splitlines()[-1:]}", flush=True)
    print(f"{len(windows)} windows, {failed} failed validation or conversion")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
