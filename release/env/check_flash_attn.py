"""Check that the `sam3d` env can run SAM 3D's attention on this GPU. Run it with the env's own python:

    conda run -n sam3d python release/env/check_flash_attn.py [--no-gpu]

SAM 3D Objects picks its attention backend from the name of CUDA device 0: on a card whose name contains A100, H100
or H200 it sets `ATTN_BACKEND=SPARSE_ATTN_BACKEND=flash_attn` and then imports flash_attn
(`sam3d_objects/pipeline/inference_pipeline.py`, `set_attention_backend`). Without flash_attn the shape stage fails
there with `ModuleNotFoundError: No module named 'flash_attn'`. On any other card it uses PyTorch's `sdpa` and does
not import flash_attn.

Exit 1 (FAIL) when device 0 is such a card and flash_attn does not import. Exit 0 otherwise. On another
Ampere-or-newer card without flash_attn it prints a NOTE: SAM 3D runs `sdpa` there, which this release has not run
(every run of ours used an H100 and flash_attn). With --no-gpu no device is queried, and a missing flash_attn is a NOTE.
flash_attn is installed in release/QUICKSTART.md step 2 (the official 2.8.3 wheel; `release/env/sam3d.pip.txt`).
"""
from __future__ import annotations

import argparse
import importlib
import sys

FLASH_ATTN_CARDS = ("A100", "H100", "H200")      # SAM 3D's own rule: a substring of torch.cuda.get_device_name(0)
PINNED = "2.8.3"
INSTALL = "Install it as release/QUICKSTART.md step 2 does (the flash_attn line of release/env/sam3d.pip.txt)"


def flash_attn_status(module: str = "flash_attn") -> tuple[str | None, str | None]:
    """(version, None) when the module imports, else (None, the import error as one line)."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - an ABI or CUDA mismatch raises OSError/ImportError/RuntimeError
        return None, f"{type(exc).__name__}: {exc}".splitlines()[0]
    return str(getattr(mod, "__version__", "unknown")), None


def decide(gpu: tuple[str, tuple[int, int]] | None, version: str | None, error: str | None) -> tuple[bool, str]:
    """(ok, one line) for device 0 = `gpu` ((name, compute capability), or None for --no-gpu)."""
    if version is not None:
        note = "" if version == PINNED else f" (this release pins {PINNED})"
        where = "" if gpu is None else f" on {gpu[0]}"
        return True, f"OK   flash_attn {version} imports in the sam3d env{where}{note}"
    if gpu is None:
        return True, (f"NOTE flash_attn does not import ({error}). SAM 3D needs it on A100/H100/H200: {INSTALL}. "
                      "Rerun without --no-gpu on the GPU host to check the card")
    name, cc = gpu
    if any(card in name for card in FLASH_ATTN_CARDS):
        return False, (f"FAIL flash_attn does not import ({error}), and SAM 3D selects flash_attn on {name}: the shape "
                       f"stage would fail with \"No module named 'flash_attn'\". {INSTALL}")
    if cc >= (8, 0):
        return True, (f"NOTE flash_attn does not import ({error}). On {name} (sm_{cc[0]}{cc[1]}) SAM 3D uses PyTorch "
                      "sdpa instead, a path this release has not run (all our runs: H100 with flash_attn)")
    return True, f"OK   {name} (sm_{cc[0]}{cc[1]}): SAM 3D uses sdpa here and does not import flash_attn"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-gpu", action="store_true", help="do not query a CUDA device")
    a = ap.parse_args(argv)
    gpu = None
    if not a.no_gpu:
        import torch

        if not torch.cuda.is_available():
            print("FAIL no CUDA device is visible to the sam3d env (rerun with --no-gpu for a CPU-only look)")
            return 1
        gpu = (torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0)))
    ok, line = decide(gpu, *flash_attn_status())
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
