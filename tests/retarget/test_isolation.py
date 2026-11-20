"""Tests for `fastdo.retarget.isolation`'s process-group kill logic.

A real incident during development: `gpu_dry_run_probe`'s original implementation only sent
SIGTERM/SIGKILL to the immediate subprocess's own PID. The baseline invocation's wrapper chain
(`nice` -> `conda run` -> the real Python process) forks rather than exec-replacing at the
`conda run` step, so killing only the wrapper left the real MuJoCo-Warp MPC process running,
un-timeboxed, after the dry run had already reported its (final) peak and "finished". Two such
orphaned runs were caught and killed externally. These tests reproduce the exact
shape of that bug (a process that forks a child and does not forward signals to it) and assert the
fix (`start_new_session=True` + `os.killpg`) kills the whole tree, not just the wrapper.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastdo.retarget.isolation import gpu_dry_run_probe

_FORKING_SCRIPT = """
import json, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[1], "w") as f:
    json.dump({"child_pid": child.pid}, f)
time.sleep(120)
"""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    else:
        return True


def test_gpu_dry_run_probe_kills_a_grandchild_the_wrapper_never_forwards_signals_to(
    tmp_path: Path,
) -> None:
    """Reproduces the real bug shape: a "wrapper" process (standing in for `nice`/`conda run`)
    that forks a real child and does not forward SIGTERM/SIGKILL to it. Only killing the whole
    process group (not just the wrapper's own PID) can reap the child too."""
    script = tmp_path / "forking_wrapper.py"
    script.write_text(_FORKING_SCRIPT)
    child_pid_file = tmp_path / "child_pid.json"
    log_path = tmp_path / "dry_run.log"

    result = gpu_dry_run_probe(
        cmd=[sys.executable, str(script), str(child_pid_file)], cwd=tmp_path, env=dict(os.environ),
        seconds=2.0, poll_seconds=0.5, log_path=log_path,
    )

    assert result.confirmed_dead, "process group was not confirmed dead -- the real bug this guards against"

    # The wrapper had to have actually started and forked before we can check its child died too.
    deadline = time.monotonic() + 5.0
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert child_pid_file.exists(), "wrapper never wrote its child's pid -- test setup is broken"
    import json

    child_pid = json.loads(child_pid_file.read_text())["child_pid"]
    assert not _pid_alive(child_pid), (
        f"grandchild pid {child_pid} is still alive -- killing only the wrapper's own pid is not "
        f"enough (this is the exact bug the process-group kill fixes)"
    )
