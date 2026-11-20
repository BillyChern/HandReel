"""The detector client's protocol (S2 precondition P3), with the subprocess faked or stubbed.

detectron2 cannot live in the engine venv, so the detector runs in its own interpreter, as the SAM 3D worker does.
What matters here and is testable without a GPU: the request carries a PATH and row indices and never pixels (the
Ego4D licence steps depend on that), the process is built ONCE and reused across windows (the 5.5 s of every 7.5 s
that a one-shot process spent on startup), a failing or silent subprocess raises with something diagnosable, and a
reply that does not answer the question asked is rejected rather than half-used.

The warm-process tests run a STUB server in this interpreter: it speaks the real line protocol, so the client's
handshake, restart and error paths are exercised for real without detectron2 or a card.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from fastdo.prompt.hands23_client import Hands23Client, Hands23Error, WarmDetector

STUB = textwrap.dedent('''
    """A detector that speaks the line protocol and counts its own starts."""
    import argparse, json, sys
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    home = Path(args.weights).parent
    starts = home / "starts.txt"
    n_starts = int(starts.read_text()) + 1 if starts.is_file() else 1
    starts.write_text(str(n_starts))
    mode = (home / "mode.txt").read_text().strip() if (home / "mode.txt").is_file() else "ok"

    if mode == "silent_start":
        sys.exit(0)
    if mode == "no_ready":
        print(json.dumps({"ready": False, "why": "weights do not match the config"}), flush=True)
        sys.exit(0)
    print("detectron2 chatter on stderr, which must not reach stdout", file=sys.stderr, flush=True)
    print(json.dumps({"ready": True, "build_seconds": 0.5, "weights_sha256": "c" * 64}), flush=True)

    served = 0
    for line in sys.stdin:
        path = line.strip()
        if not path or path == "quit":
            break
        served += 1
        if mode == "die_on_second" and served == 2:
            sys.exit(3)
        request = json.loads(Path(path).read_text())
        if mode == "error_once" and served == 1:
            print(json.dumps({"ok": False, "error": "ValueError: no such decode row"}), flush=True)
            continue
        if mode == "ack_without_writing":
            print(json.dumps({"ok": True}), flush=True)
            continue
        rows = {str(r): [] for r in request["rows"]}
        Path(request["reply"]).write_text(json.dumps(
            {"rows": rows, "meta": {"weights_sha256": "c" * 64, "warm": True, "served": served}}))
        print(json.dumps({"ok": True, "reply": request["reply"]}), flush=True)
''')


def _decode(tmp_path: Path, n: int = 8) -> Path:
    p = tmp_path / "decode.npz"
    np.savez(p, frames=np.zeros((n, 12, 16, 3), np.uint8), frame_idx=np.arange(n, dtype=np.int64),
             fps=np.array(10.0), stride=np.array(3, dtype=np.int64))
    return p


def _reply(rows: list[int], *, hands: dict | None = None) -> dict:
    return {"rows": {str(r): (hands or {}).get(r, []) for r in rows}, "meta": {"weights_sha256": "a" * 64}}


def _stub_client(tmp_path: Path, mode: str = "ok") -> Hands23Client:
    """A client whose detector is the stub server above, with `mode` steering how it misbehaves."""
    home = tmp_path / "det"
    home.mkdir(parents=True, exist_ok=True)
    script = home / "stub_detect.py"
    script.write_text(STUB)
    (home / "mode.txt").write_text(mode)
    return Hands23Client(python=sys.executable, script=script, weights=home / "w.pth", device="cpu",
                         startup_timeout_s=60.0, timeout_s=60.0)


def _starts(client: Hands23Client) -> int:
    path = Path(client.weights).parent / "starts.txt"
    return int(path.read_text()) if path.is_file() else 0


def test_the_request_carries_a_path_and_rows_but_no_pixels(tmp_path: Path) -> None:
    seen: dict = {}

    def runner(request: Path) -> None:
        seen["request"] = json.loads(request.read_text())
        Path(seen["request"]["reply"]).write_text(json.dumps(_reply(seen["request"]["rows"])))

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=runner)
    client.detect(_decode(tmp_path), [0, 3, 6])

    req = seen["request"]
    assert req["rows"] == [0, 3, 6]
    assert req["decode_npz"].endswith("decode.npz") and Path(req["decode_npz"]).is_file()
    blob = json.dumps(req)
    assert "frames" not in blob and "pixels" not in blob            # a path, never an array
    assert len(blob) < 2000                                          # nothing image-sized travels in the request


def test_a_reply_for_the_wrong_rows_is_refused(tmp_path: Path) -> None:
    def runner(request: Path) -> None:
        reply = json.loads(request.read_text())["reply"]
        Path(reply).write_text(json.dumps(_reply([0, 1, 2])))        # not the rows we asked for

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=runner)
    with pytest.raises(Hands23Error, match="rows"):
        client.detect(_decode(tmp_path), [0, 3, 6])


def test_a_crash_or_a_missing_reply_raises_with_the_reason(tmp_path: Path) -> None:
    def boom(request: Path) -> None:
        raise RuntimeError("detector exited 1: CUDA out of memory")

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=boom)
    with pytest.raises(Hands23Error, match="CUDA out of memory"):
        client.detect(_decode(tmp_path), [0])

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=lambda request: None)             # acknowledges, writes nothing
    with pytest.raises(Hands23Error, match="no reply"):
        client.detect(_decode(tmp_path), [0])


def test_malformed_json_is_an_error_not_a_traceback(tmp_path: Path) -> None:
    def junk(request: Path) -> None:
        Path(json.loads(request.read_text())["reply"]).write_text("not json{")

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=junk)
    with pytest.raises(Hands23Error, match="reply is not JSON"):
        client.detect(_decode(tmp_path), [0])


def test_the_records_come_back_keyed_by_int_row(tmp_path: Path) -> None:
    rec = {"hand_score": 0.95, "contact_idx": 3, "obj_box": [1, 2, 3, 4], "obj_score": 0.9,
           "interaction_score": 0.8, "p_held_like": 0.7, "side": "right"}

    def runner(request: Path) -> None:
        req = json.loads(request.read_text())
        Path(req["reply"]).write_text(json.dumps(_reply(req["rows"], hands={3: [rec]})))

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=runner)
    out = client.detect(_decode(tmp_path), [0, 3])

    assert set(out.rows) == {0, 3} and out.rows[3][0]["obj_score"] == 0.9
    assert out.meta["weights_sha256"] == "a" * 64


def test_the_client_refuses_rows_outside_the_decode(tmp_path: Path) -> None:
    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=lambda request: None)
    with pytest.raises(ValueError, match="rows"):
        client.detect(_decode(tmp_path, n=4), [0, 9])
    with pytest.raises(ValueError, match="rows"):
        client.detect(_decode(tmp_path, n=4), [])


# -- the warm process: the point of the whole file --------------------------------------------------------------

def test_three_windows_start_the_detector_once(tmp_path: Path) -> None:
    """The reason warm mode exists: the model build is paid per RUN, not per window."""
    client = _stub_client(tmp_path)
    decode = _decode(tmp_path)
    try:
        for _ in range(3):
            out = client.detect(decode, [0, 3])
            assert set(out.rows) == {0, 3}
        assert out.meta["warm"] is True and out.meta["served"] == 3
        assert _starts(client) == 1
    finally:
        client.close()


def test_the_command_is_the_serve_command(tmp_path: Path) -> None:
    client = _stub_client(tmp_path)
    cmd = client.command()
    assert "--serve" in cmd and "--weights" in cmd
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_a_detector_that_dies_is_replaced_on_the_next_window(tmp_path: Path) -> None:
    client = _stub_client(tmp_path, mode="die_on_second")
    decode = _decode(tmp_path)
    try:
        client.detect(decode, [0])                                   # first window: fine
        with pytest.raises(Hands23Error, match="exited 3|closed its output"):
            client.detect(decode, [0])                                # second: the process exits mid-request
        out = client.detect(decode, [0, 1])                           # third: a fresh process serves it
        assert set(out.rows) == {0, 1}
        assert _starts(client) == 2                                   # it restarted rather than staying dead
    finally:
        client.close()


def test_one_unanswerable_window_does_not_cost_the_warm_model(tmp_path: Path) -> None:
    client = _stub_client(tmp_path, mode="error_once")
    decode = _decode(tmp_path)
    try:
        with pytest.raises(Hands23Error, match="no such decode row"):
            client.detect(decode, [0])
        out = client.detect(decode, [0, 1])                          # the same process answers the next window
        assert set(out.rows) == {0, 1}
        assert _starts(client) == 1
    finally:
        client.close()


def test_an_acknowledged_window_with_no_reply_file_is_an_error(tmp_path: Path) -> None:
    client = _stub_client(tmp_path, mode="ack_without_writing")
    try:
        with pytest.raises(Hands23Error, match="no reply"):
            client.detect(_decode(tmp_path), [0])
    finally:
        client.close()


def test_a_detector_that_never_reports_ready_fails_with_its_own_last_words(tmp_path: Path) -> None:
    client = _stub_client(tmp_path, mode="silent_start")
    try:
        with pytest.raises(Hands23Error, match="without sending its ready line|closed its output"):
            client.detect(_decode(tmp_path), [0])
    finally:
        client.close()

    client = _stub_client(tmp_path / "b", mode="no_ready")
    try:
        with pytest.raises(Hands23Error, match="did not report ready"):
            client.detect(_decode(tmp_path), [0])
    finally:
        client.close()


def test_a_wedged_detector_hits_the_timeout_instead_of_hanging_the_run(tmp_path: Path) -> None:
    script = tmp_path / "wedged.py"
    script.write_text('import time\ntime.sleep(300)\n')
    warm = WarmDetector([sys.executable, str(script)], startup_timeout_s=1.0, timeout_s=1.0)
    try:
        with pytest.raises(Hands23Error, match="did not send its ready line within 1s"):
            warm(tmp_path / "request.json")
    finally:
        warm.close()


def test_close_is_idempotent_and_leaves_no_process(tmp_path: Path) -> None:
    client = _stub_client(tmp_path)
    client.detect(_decode(tmp_path), [0])
    client.close()
    client.close()
    assert client._warm is not None and client._warm._proc is None


def _captured_request(tmp_path: Path, **client_kwargs) -> dict:
    seen: dict = {}

    def runner(request: Path) -> None:
        seen["raw"] = request.read_text()
        body = json.loads(seen["raw"])
        Path(body["reply"]).write_text(json.dumps(_reply(body["rows"])))

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=runner, **client_kwargs)
    client.detect(_decode(tmp_path), [0, 3, 6])
    return seen


def test_batch_one_sends_the_pre_234_request_byte_for_byte(tmp_path: Path) -> None:
    """The default must be today's request -- no `batch` key, the same keys in the same order."""
    for kwargs in ({}, {"batch": 1}, {"prefetch": 0}, {"batch": 1, "prefetch": 0}):
        raw = _captured_request(tmp_path, **kwargs)["raw"]
        body = json.loads(raw)
        pre_234 = json.dumps({"decode_npz": str(tmp_path / "decode.npz"), "rows": [0, 3, 6],   # the old expression
                              "weights": str(tmp_path / "w.pth"), "device": "cuda", "reply": body["reply"]})
        assert raw == pre_234


def test_a_batch_above_one_is_sent_to_the_detector(tmp_path: Path) -> None:
    body = json.loads(_captured_request(tmp_path, batch=4)["raw"])
    assert body["batch"] == 4 and body["rows"] == [0, 3, 6]


def test_prefetch_above_zero_is_sent_to_the_detector(tmp_path: Path) -> None:
    body = json.loads(_captured_request(tmp_path, prefetch=2)["raw"])
    assert body["prefetch"] == 2 and "batch" not in body


def test_a_batch_below_one_is_refused_before_the_detector_runs(tmp_path: Path) -> None:
    def runner(request: Path) -> None:
        raise AssertionError("the detector must not be asked")

    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=runner, batch=0)
    with pytest.raises(ValueError, match="batch"):
        client.detect(_decode(tmp_path), [0, 3])
    client = Hands23Client(python="/usr/bin/python3", script=tmp_path / "d.py", weights=tmp_path / "w.pth",
                           runner=runner, prefetch=-1)
    with pytest.raises(ValueError, match="prefetch"):
        client.detect(_decode(tmp_path), [0, 3])
