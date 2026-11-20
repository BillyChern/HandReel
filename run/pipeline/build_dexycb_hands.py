"""DexYCB ground-truth MANO hands as HaWoR-format `all_hand_meshes.npz`, right and left hands.

  FASTDO_ROOT=/path/to/checkout uv run python run/pipeline/build_dexycb_hands.py 'policy_hands.subjects=[subject-03]'

Every selected sequence becomes `<out_root>/<sequence>/all_hand_meshes.npz`, with `all_hand_meshes_validation.json`
beside it (`fastdo.policy.dexycb_inputs` layout).
- The selection is `policy_hands.sequences`, else every sequence of `subjects` outside `exclude_subjects`, up to
  `max_seqs`.
- Each file is made by `run/pipeline/_dexycb_hands_driver.py` in the hawor conda env: a CPU subprocess with CUDA
  hidden, in the camera frame of the sequence's `serial_index`-th serial.
- The driver runs at `threads` (2) for torch, OpenMP, MKL and OpenBLAS, whatever the parent process asks. Float32 MANO
  output differs by about 1e-5 mm across thread counts, and the h2h and regtracks references ran at 2 threads; pinning
  makes the files the same under either CPU slot. Each record stores the count.
- Both MANO models come from config (`policy_hands.mano_{right,left}_pkl`, interpolated from
  `retarget_stage.mano_pkl_path`); a sequence whose hand has no configured model raises before any subprocess.
- Subjects in `exclude_subjects` (07 and 08, reserved for calibration and confirmation) are refused before any file of
  theirs is read.

Validation per sequence, on 5 frames: joint error <= `joint_tol_mm` against DexYCB `joint_3d`, and vertex-in-hand-label
fraction >= `label_frac_min`. A failing sequence is recorded with `pass` False, never dropped or replaced; the reference
function skips it. A driver that crashes or times out also gives a row with `pass` False and its error, and the build
goes on with the next sequence. It leaves no validation record, so the reference function skips that sequence as not
built. `<out_root>/validation.json` summarizes the run.

A converted sequence is reused unless `force`, when its record matches the serial and the thread count.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from fastdo.io.datasets.dexycb import DexYCBReader, load_dexycb_yaml
from fastdo.policy.dexycb_inputs import hands_npz_path, hands_record_path

logger = logging.getLogger(__name__)

__all__ = ["HandsBuildConfig", "build_hands", "convert_sequence", "select_sequences"]

DRIVER = Path(__file__).resolve().parent / "_dexycb_hands_driver.py"
RESERVED = ("subject-07", "subject-08")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@dataclass(frozen=True)
class HandsBuildConfig:
    """The Hydra block `policy_hands` (module docstring)."""

    out_root: str
    dexycb_root: str
    mano_pca: str
    hawor_python: str
    hawor_root: str
    mano_right_pkl: str
    mano_left_pkl: str
    serial_index: int = 0
    validate_frames: int = 5
    joint_tol_mm: float = 2.0
    label_frac_min: float = 0.8
    exclude_subjects: tuple[str, ...] = RESERVED
    force: bool = False
    timeout_s: float = 600.0
    threads: int = 2

    def __post_init__(self) -> None:
        if self.threads < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads}")

    @classmethod
    def from_mapping(cls, node: Mapping[str, Any]) -> HandsBuildConfig:
        names = ("out_root", "dexycb_root", "mano_pca", "hawor_python", "hawor_root", "mano_right_pkl", "mano_left_pkl")
        return cls(**{name: str(node[name] or "") for name in names}, serial_index=int(node["serial_index"]),
                   validate_frames=int(node["validate_frames"]), joint_tol_mm=float(node["joint_tol_mm"]),
                   label_frac_min=float(node["label_frac_min"]),
                   exclude_subjects=tuple(str(s) for s in node["exclude_subjects"]), force=bool(node["force"]),
                   timeout_s=float(node["timeout_s"]), threads=int(node["threads"]))


def _row(sequence: str, record: Mapping[str, Any], npz: Path) -> dict[str, Any]:
    conversion, validation = record["conversion"], record["validation"]
    return {"sequence": sequence, "hand_side": conversion["hand_side"], "serial": conversion["serial"],
            "num_frames": conversion["num_frames"], "n_valid": conversion["n_valid"],
            "threads": conversion.get("threads"),
            "joint_err_mm_all_frames": conversion["joint3d_err_mm_all_frames"]["mean"],
            "joint_err_mm_mean_5f": validation["joint_err_mm_mean_5f"], "label_frac_5f": validation["frac_all"],
            "frames": validation["frames"], "pass_joint": validation["pass_joint"],
            "pass_label": validation["pass_label"], "pass": bool(validation["pass"]), "npz": str(npz)}


def _failed_row(sequence: str, side: str, serial: str, error: str, log: Path) -> dict[str, Any]:
    """The summary row of a sequence whose driver left no validation record."""
    return {"sequence": sequence, "hand_side": side, "serial": serial, "pass": False, "pass_joint": False,
            "pass_label": False, "error": error, "log": str(log)}


def _text(output: str | bytes | None) -> str:
    return output.decode(errors="replace") if isinstance(output, bytes) else (output or "")


def _driver_env(threads: int) -> dict[str, str]:
    """The driver's environment: CUDA hidden, none of this environment's Python paths, and `threads` for OpenMP, MKL and
    OpenBLAS whatever the parent process asks (module docstring)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")}
    env.update({name: str(threads) for name in THREAD_VARS}, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    return env


def convert_sequence(sequence: str, cfg: HandsBuildConfig) -> dict[str, Any]:
    """Convert and validate `"<subject>/<sequence>"` (a trailing serial is ignored); returns its summary row. A driver
    that crashes or times out gives a row with `pass` False and its `error` (module docstring).

    Raises:
        ValueError: For a reserved subject, or a hand side without a configured MANO model.
    """
    sequence = "/".join(sequence.strip("/").split("/")[:2])
    subject = sequence.split("/")[0]
    if any(subject == s or subject.endswith(s) for s in cfg.exclude_subjects):
        raise ValueError(f"{sequence}: subject {subject} is reserved (policy_hands.exclude_subjects); not opened")
    meta = load_dexycb_yaml(Path(cfg.dexycb_root) / sequence / "meta.yml")
    side, serial = str(next(iter(meta["mano_sides"]))), str(list(meta["serials"])[cfg.serial_index])
    if not {"right": cfg.mano_right_pkl, "left": cfg.mano_left_pkl}.get(side):
        raise ValueError(f"{sequence}: {side} hand, but no MANO model is configured (retarget_stage.mano_pkl_path.{side})")
    npz, record_path = hands_npz_path(cfg.out_root, sequence), hands_record_path(cfg.out_root, sequence)
    if not cfg.force and npz.is_file() and record_path.is_file():
        stored = json.loads(record_path.read_text())
        if str(stored["conversion"]["serial"]) == serial and stored["conversion"].get("threads") == cfg.threads:
            return _row(sequence, stored, npz)
    npz.parent.mkdir(parents=True, exist_ok=True)
    record_path.unlink(missing_ok=True)
    log_path = npz.parent / "convert.log"
    cmd = [cfg.hawor_python, str(DRIVER), sequence, str(npz), "--serial", serial, "--dexycb-root", cfg.dexycb_root,
           "--mano-pca", cfg.mano_pca, "--hawor-root", cfg.hawor_root, "--mano-right-pkl", cfg.mano_right_pkl,
           "--mano-left-pkl", cfg.mano_left_pkl, "--validate-frames", str(cfg.validate_frames), "--joint-tol-mm",
           str(cfg.joint_tol_mm), "--label-frac-min", str(cfg.label_frac_min), "--threads", str(cfg.threads)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, env=_driver_env(cfg.threads), cwd=str(npz.parent),
                              timeout=cfg.timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(_text(exc.stdout) + _text(exc.stderr))
        error = f"hands driver timed out after {cfg.timeout_s:g} s"
        logger.error("%s (%s hand, serial %s): %s; recorded as failed, build continues (%s)", sequence, side, serial,
                     error, log_path)
        return _failed_row(sequence, side, serial, error, log_path)
    log_path.write_text(done.stdout + done.stderr)
    if done.returncode not in (0, 1) or not record_path.is_file():
        error = f"hands driver exited {done.returncode} without a validation record: {done.stderr[-1500:]}"
        logger.error("%s (%s hand, serial %s): %s; recorded as failed, build continues (%s)", sequence, side, serial,
                     error, log_path)
        return _failed_row(sequence, side, serial, error, log_path)
    row = _row(sequence, json.loads(record_path.read_text()), npz)
    logger.info("%s (%s hand, serial %s): joint error %.3f mm, label fraction %.3f, pass %s", sequence, side, serial,
                row["joint_err_mm_mean_5f"], row["label_frac_5f"], row["pass"])
    return row


def build_hands(sequences: Sequence[str], cfg: HandsBuildConfig) -> dict[str, Any]:
    """`convert_sequence` for each sequence; writes and returns the `<out_root>/validation.json` summary of this run."""
    rows = {seq: convert_sequence(seq, cfg) for seq in sequences}
    summary = {"sequences": rows, "n": len(rows), "n_pass": sum(r["pass"] for r in rows.values()),
               "failed": sorted(s for s, r in rows.items() if not r["pass"]),
               "sides": {side: sum(r["hand_side"] == side for r in rows.values()) for side in ("left", "right")},
               "threads": cfg.threads,
               "rule": f"pass = mean joint error over the {cfg.validate_frames} validation frames <= {cfg.joint_tol_mm} mm "
                       f"AND in-hand-label fraction >= {cfg.label_frac_min}; failing sequences, and sequences whose "
                       "driver failed (with their error), are recorded, not kept",
               "updated_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    path = Path(cfg.out_root) / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    return summary


def select_sequences(dexycb_root: str | Path, subjects: Sequence[str], exclude_subjects: Sequence[str],
                     max_seqs: int = 0) -> list[str]:
    """`"<subject>/<sequence>"` ids of the allowed subjects (empty `subjects`: all), in reader order, up to `max_seqs`."""
    reader = DexYCBReader(Path(dexycb_root), subjects=list(subjects) or None, exclude_subjects=exclude_subjects)
    ids = list(dict.fromkeys(seq.rsplit("/", 1)[0] for seq in reader.sequences()))
    return ids[:max_seqs] if max_seqs > 0 else ids


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    node = cfg.policy_hands
    build_cfg = HandsBuildConfig.from_mapping(node)
    sequences = [str(s) for s in node.sequences] or select_sequences(
        build_cfg.dexycb_root, [str(s) for s in node.subjects], build_cfg.exclude_subjects, int(node.max_seqs))
    summary = build_hands(sequences, build_cfg)
    logger.info("hands: %d of %d pass (%s); failed %s", summary["n_pass"], summary["n"], summary["sides"],
                summary["failed"])


if __name__ == "__main__":
    main()
