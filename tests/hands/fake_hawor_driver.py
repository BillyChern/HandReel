"""Stand-in for run/pipeline/_hawor_batched_driver.py in tests: the same CLI and file contract, no models.

`FAKE_DRIVER_MODE` picks the behaviour:
- `ok`: every clip done, with a valid hand npz (exit 0);
- `one_failed`: the second clip failed with an error message (exit 1);
- `load_error`: a load failure, `error` set and no clips (exit 2);
- `bad_npz`: clips reported done, but the npz is written with raw `np.savez` and `left_joints` of shape (T,20,3);
- `sleep`: starts a sleeping grandchild, writes both pids to `<job_dir>/fake_driver.pids`, then sleeps;
- `crash` (`--serve` only): writes its pid to `<job_dir>/fake_driver.pids` and SIGKILLs itself mid-request.

numpy and `fastdo` are imported lazily. Importing `fastdo.hands` loads torch, which can take longer than a short test
timeout on a loaded host, and `sleep` mode must record its pids before that timeout fires.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from fastdo.hands.hand_npz import SideArrays

MODES = ("ok", "one_failed", "load_error", "bad_npz", "sleep", "crash")
SECONDS_KEYS = ("read", "detect", "box_filter", "crop", "backbone", "heads", "gap_fill", "mano", "write", "total")
PID_FILE = "fake_driver.pids"


def side_arrays(t: int) -> SideArrays:
    """An all-valid side of zeros with the npz contract's shapes and dtypes."""
    import numpy as np

    from fastdo.hands.hand_npz import SideArrays

    return SideArrays(vertices=np.zeros((t, 778, 3), np.float32), joints=np.zeros((t, 21, 3), np.float32),
                      trans=np.zeros((t, 3), np.float32), rot=np.zeros((t, 3), np.float32),
                      hand_pose=np.zeros((t, 45), np.float32), betas=np.zeros((t, 10), np.float32),
                      valid=np.ones(t, bool), infilled=np.zeros(t, bool), box=np.zeros((t, 5), np.float32))


def write_valid_hand_npz(path: Path, frame_idx: np.ndarray, fps: float = 10.0) -> None:
    """A contract-valid hand npz with one row per entry of `frame_idx`."""
    import numpy as np

    from fastdo.hands.hand_npz import write_hand_npz

    faces = {"left": np.zeros((1552, 3), np.int32), "right": np.zeros((1552, 3), np.int32)}
    t = int(np.asarray(frame_idx).shape[0])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_hand_npz(Path(path), np.asarray(frame_idx, np.int64), fps, side_arrays(t), side_arrays(t), faces)


def clip_result(clip_id: str, status: str, error: str, n_frames: int, total: float = 0.9) -> dict[str, Any]:
    seconds = {key: total / 9 for key in SECONDS_KEYS[:-1]} | {"total": total}
    return {"clip_id": clip_id, "status": status, "error": error, "n_frames": n_frames, "seconds": seconds}


def write_response(path: Path, error: str, clips: Sequence[dict[str, Any]], load_seconds: float = 1.0,
                   warmup_seconds: float = 0.5) -> None:
    """A driver-format `response.json`, written atomically."""
    body = {"version": 1, "error": error, "load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
            "compute_seconds": 0.25, "peak_cuda_gb": 0.0, "clips": list(clips)}
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)


def serve(mode: str) -> int:
    """`--serve`: one request path per stdin line, the response next to it, one completion line on stdout.

    Every request appends this process's pid to `<job_dir>/fake_driver.pids`. `crash` SIGKILLs itself mid-request;
    `load_error` answers with `serving: false` and exits 2, as the real driver does.
    """
    for line in sys.stdin:
        request = Path(line.strip())
        if mode == "crash":
            (request.parent / PID_FILE).write_text(str(os.getpid()))
            os.kill(os.getpid(), signal.SIGKILL)
        code = handle(mode, request, request.with_name("response.json"))
        with (request.parent / PID_FILE).open("a") as pids:
            pids.write(f"{os.getpid()}\n")
        print(json.dumps({"request": str(request), "code": code, "serving": mode != "load_error"}), flush=True)
        if mode == "load_error":
            return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args(argv)
    mode = os.environ.get("FAKE_DRIVER_MODE", "ok")
    if mode not in MODES:
        raise SystemExit(f"unknown FAKE_DRIVER_MODE {mode!r}")
    if args.serve:
        return serve(mode)
    return handle(mode, args.request, args.response)


def handle(mode: str, request_path: Path, response: Path) -> int:
    """One request in `mode`; returns the exit code a fresh driver would."""
    if mode == "sleep":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        (request_path.parent / PID_FILE).write_text(f"{os.getpid()} {child.pid}")
        time.sleep(120)
        return 0
    if mode == "load_error":
        write_response(response, "weights missing: /fake/weights/hawor/checkpoints/hawor.ckpt", [])
        return 2
    import numpy as np

    request = json.loads(request_path.read_text())
    results = []
    for i, clip in enumerate(request["clips"]):
        with np.load(clip["frames_path"], allow_pickle=False) as data:
            frame_idx, fps = np.asarray(data["frame_idx"]), float(data["fps"])
        out = Path(clip["out_npz"])
        if mode == "one_failed" and i == 1:
            results.append(clip_result(clip["clip_id"], "failed", f"fake detection failure on {clip['clip_id']}", 0))
            continue
        write_valid_hand_npz(out, frame_idx, fps)
        if mode == "bad_npz":
            with np.load(out, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["left_joints"] = np.zeros((len(frame_idx), 20, 3), np.float32)
            np.savez(out, **arrays)  # type: ignore[arg-type]  # numpy's stub types `allow_pickle` into **kwds
        results.append(clip_result(clip["clip_id"], "done", "", len(frame_idx)))
    write_response(response, "", results)
    return 1 if any(r["status"] == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
