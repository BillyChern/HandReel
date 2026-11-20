"""Runs the Hands23 hand-object detector in its own interpreter and brings back BOXES ONLY.

detectron2 does not live in the engine venv, so the detector runs the way the SAM 3D backend does: a subprocess in
the Hands23 env (`$FASTDO_HANDS23_ROOT/env_hands23`), JSON in and JSON out.

**The process is WARM: it is started once and kept.** Measured on a 12-clip corpus, a one-shot process spent
5.5 s of every 7.5 s window on interpreter start, detectron2 import, model build and weight hashing, and only 2.0 s
detecting. Those 5.5 s are a per-window tax on a stage that runs on every window of the internet corpus, so the
client keeps one process alive and sends it one request-file path per window. A dead or wedged process is replaced
on the next call, and a request the detector cannot answer comes back as an error line rather than killing it.

The cost of warmth is that the detector's ~1.1 GiB stays resident for the life of the run instead of appearing and
vanishing per window; `close()` releases it, and the peak is unchanged either way.

**The request carries a decode.npz PATH and row indices, never pixels.** The subprocess opens the file itself, and
the reply holds boxes and scores only. That keeps the Ego4D licence steps unchanged: the only
pixel cache remains `decode.npz`, which is already inventoried and deleted with its window, and no crop, thumbnail
or rendered image is ever created or forwarded.

Licence: the code is MIT, detectron2 Apache-2.0, and the weights are
allowed for **non-commercial academic research**, which propagates to data produced with them. Cite Hands23 (Cheng,
Shan, Hassen, Higgins, Fouhey, NeurIPS 2023) and its sources.
"""

from __future__ import annotations

import atexit
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import numpy as np

__all__ = ["Hands23Client", "Hands23Error", "Hands23Result"]

Runner = Callable[[Path], None]
"""Processes one request file, writing the reply to the path named inside it. The default keeps a warm process."""

_STDERR_TAIL = 5


class Hands23Error(RuntimeError):
    """The detector subprocess failed, wrote nothing, or answered a different question."""


@dataclass(frozen=True)
class Hands23Result:
    """`rows`: decode row -> that row's hand records (boxes and scores). `meta`: the detector's own provenance."""

    rows: dict[int, list[dict]]
    meta: dict


class WarmDetector:
    """One detector process, started on first use and kept across windows.

    The protocol is one request-file path per line of stdin and one JSON ack per line of stdout, which is why the
    process must never write anything else to stdout (detectron2's own logging goes to stderr, captured to a file
    so a full pipe buffer can never wedge the detector).
    """

    def __init__(self, cmd: Sequence[str], startup_timeout_s: float = 300.0, timeout_s: float = 600.0) -> None:
        self._cmd = [str(c) for c in cmd]
        self._startup_timeout_s = float(startup_timeout_s)
        self._timeout_s = float(timeout_s)
        self._proc: subprocess.Popen[bytes] | None = None
        self._dir: str | None = None
        self._log: Path | None = None
        self._buf = b""
        self.ready: dict = {}
        atexit.register(self.close)

    def __call__(self, request: Path) -> None:
        proc = self._ensure()
        stdin = proc.stdin
        if stdin is None:
            self._fail("the detector process was started without an input pipe")
        try:
            stdin.write(f"{request}\n".encode())
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._fail(f"the detector process went away while being asked ({exc})")
        ack = self._read_json(self._timeout_s, "a reply")
        if not ack.get("ok"):
            raise Hands23Error(f"the detector refused the request: {ack.get('error', ack)}")

    def close(self) -> None:
        """Stops the process and releases its GPU memory. Safe to call twice."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(b"quit\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                proc.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait(timeout=10)
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    # -- internals ------------------------------------------------------------------------------------------

    def _ensure(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self.close()
        self._dir = tempfile.mkdtemp(prefix="fastdo_h23_serve_")
        self._log = Path(self._dir) / "detector.log"
        self._buf = b""
        with self._log.open("wb") as err:
            self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                          bufsize=0)
        self.ready = self._read_json(self._startup_timeout_s, "its ready line")
        if not self.ready.get("ready"):
            self._fail(f"the detector started but did not report ready: {self.ready}")
        return self._proc

    def _read_json(self, timeout_s: float, what: str) -> dict:
        line = self._read_line(timeout_s, what)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            self._fail(f"the detector's {what} is not JSON ({exc}): {line[:200]!r}")
        if not isinstance(payload, dict):
            self._fail(f"the detector's {what} is not a JSON object: {line[:200]!r}")
        return payload

    def _read_line(self, timeout_s: float, what: str) -> str:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise Hands23Error("the detector process is not running")
        deadline = time.monotonic() + timeout_s
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._fail(f"the detector did not send {what} within {timeout_s:.0f}s")
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                if proc.poll() is not None:
                    self._fail(f"the detector exited {proc.returncode} without sending {what}")
                continue
            chunk = os.read(proc.stdout.fileno(), 1 << 16)
            if not chunk:
                self._fail(f"the detector closed its output without sending {what}")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode("utf-8", "replace").strip()

    def _fail(self, message: str) -> NoReturn:
        """Raises with the detector's own last words, and kills it so the next call starts a fresh one."""
        tail = ""
        if self._log is not None and self._log.is_file():
            lines = self._log.read_text(errors="replace").strip().splitlines()[-_STDERR_TAIL:]
            tail = " | ".join(line.strip() for line in lines if line.strip())
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        raise Hands23Error(message + (f"; detector said: {tail}" if tail else ""))


@dataclass
class Hands23Client:
    """Calls the detector for the given decode rows, keeping one warm process.

    Attributes:
        python: Interpreter of the detectron2 environment.
        script: The detector-side script (reads request files, writes replies).
        weights: `model_hands23.pth`.
        device: Passed through to the script.
        runner: Injection point; the default is a `WarmDetector` built on first use. Tests pass a plain function.
        timeout_s: Wall limit for one window.
        startup_timeout_s: Wall limit for the model build, paid once per process.
        batch: Rows per detector forward (`prompt_stage.detect_batch`). 1 sends the request exactly as before
            the option existed (no `batch` key), so the default path is byte-identical; >1 asks the detector to run
            its backbone and RPN on that many rows at once.
        prefetch: CPU threads preparing rows ahead of the GPU (`prompt_stage.detect_prefetch`). 0 sends no
            `prefetch` key (the original request); >0 overlaps the BGR flip and resize with the previous forward.
    """

    python: str
    script: Path
    weights: Path
    device: str = "cuda"
    runner: Runner | None = None
    timeout_s: float = 600.0
    startup_timeout_s: float = 300.0
    batch: int = 1
    prefetch: int = 0
    _warm: WarmDetector | None = field(default=None, init=False, repr=False, compare=False)

    def command(self) -> list[str]:
        """The warm server's command line."""
        return [str(self.python), str(self.script), "--serve", "--weights", str(self.weights),
                "--device", str(self.device)]

    def _runner(self) -> Runner:
        if self.runner is not None:
            return self.runner
        if self._warm is None:
            self._warm = WarmDetector(self.command(), startup_timeout_s=self.startup_timeout_s,
                                      timeout_s=self.timeout_s)
        return self._warm

    def close(self) -> None:
        """Releases the warm process (and its GPU memory), if one was started."""
        if self._warm is not None:
            self._warm.close()

    def detect(self, decode_npz: Path, rows: Sequence[int]) -> Hands23Result:
        """Hand records for `rows` of `decode_npz`.

        Raises:
            ValueError: If `rows` is empty or steps outside the decode file's rows.
            Hands23Error: If the detector fails, writes no reply, writes unparsable JSON, or answers other rows.
        """
        decode_npz = Path(decode_npz)
        wanted = [int(r) for r in rows]
        if not wanted:
            raise ValueError("rows is empty: the caller must name the candidate rows")
        if int(self.batch) < 1 or int(self.prefetch) < 0:
            raise ValueError(f"batch must be >= 1 and prefetch >= 0, got batch {self.batch}, prefetch {self.prefetch}")
        with np.load(decode_npz) as data:            # header only; the frames are not materialised here
            n_rows = int(data["frame_idx"].shape[0])
        bad = [r for r in wanted if not 0 <= r < n_rows]
        if bad:
            raise ValueError(f"rows {bad} are outside the {n_rows} decode rows of {decode_npz}")

        runner = self._runner()
        with tempfile.TemporaryDirectory(prefix="fastdo_h23_") as tmp:
            request, reply = Path(tmp) / "request.json", Path(tmp) / "reply.json"
            body: dict[str, object] = {"decode_npz": str(decode_npz), "rows": wanted,
                                       "weights": str(self.weights), "device": self.device, "reply": str(reply)}
            if int(self.batch) != 1:                  # absent at 1: the request is the original request, byte for byte
                body["batch"] = int(self.batch)
            if int(self.prefetch) != 0:               # likewise absent at 0
                body["prefetch"] = int(self.prefetch)
            request.write_text(json.dumps(body))
            try:
                runner(request)
            except Hands23Error:
                raise
            except Exception as exc:                  # a runner that raises anything else still gets one error type
                raise Hands23Error(str(exc)) from exc
            if not reply.is_file():
                raise Hands23Error(f"the detector wrote no reply to {reply} (it acknowledged without writing)")
            try:
                payload = json.loads(reply.read_text())
            except json.JSONDecodeError as exc:
                raise Hands23Error(f"the detector's reply is not JSON: {exc}") from exc

        got = {int(k): list(v) for k, v in dict(payload.get("rows", {})).items()}
        if sorted(got) != sorted(wanted):
            raise Hands23Error(f"the detector answered rows {sorted(got)}, not the rows asked for {sorted(wanted)}")
        return Hands23Result(rows=got, meta=dict(payload.get("meta", {})))
