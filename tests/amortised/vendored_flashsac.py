"""Import `rsl_rl_flashsac` modules READ-ONLY from a separate FlashDexRetarget clone for the SAC fixed-batch checks.

Nothing of that package is included in this repository. The package (inside FlashDexRetarget @ 7919f8ae,
`third_party/rsl_rl_flashsac`, BSD-3-Clause) is found at `$FASTDO_FLASHSAC_ROOT`; without it the checks skip. It imports
`rsl_rl` and `tensordict`, neither of which is installed in our venv. The functions checked here use neither at run
time except `TensorDict` construction in the buffer's `sample` and module-level type names, so minimal stand-ins are
placed in `sys.modules` for the duration of the import only; `sys.modules` and `sys.path` are restored afterwards
(third-party imports must not leak global state into the rest of the suite). Nothing under third_party is written:
bytecode writing is disabled during the import.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("FASTDO_FLASHSAC_ROOT") or "/nonexistent/flashdexretarget/third_party/rsl_rl_flashsac")
FILES = ("rsl_rl_flashsac/utils/categorical.py", "rsl_rl_flashsac/storage/torch_buffer.py",
         "rsl_rl_flashsac/algorithms/flash_sac.py", "rsl_rl_flashsac/utils/reward_normalization.py",
         "rsl_rl_flashsac/modules/flash_sac_layers.py", "rsl_rl_flashsac/utils/lr_scheduler.py")


def available() -> bool:
    return all((ROOT / f).is_file() for f in FILES)


def digests() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


class _TensorDict(dict):
    """Stand-in for `tensordict.TensorDict`: a dict that accepts (and ignores) batch_size/device."""

    def __init__(self, source: Any = None, batch_size: Any = None, device: Any = None, **kw: Any) -> None:
        super().__init__(source or {}, **kw)
        self.batch_size = batch_size


def _stubs() -> dict[str, types.ModuleType]:
    mods: dict[str, types.ModuleType] = {}
    for name in ("rsl_rl", "rsl_rl.env", "rsl_rl.extensions", "rsl_rl.utils", "rsl_rl.modules", "tensordict"):
        mods[name] = types.ModuleType(name)
    mods["rsl_rl.env"].VecEnv = object
    mods["rsl_rl.extensions"].Symmetry = object
    mods["rsl_rl.utils"].resolve_obs_groups = lambda *a, **k: None
    mods["rsl_rl.utils"].resolve_callable = lambda x: x
    mods["rsl_rl.modules"].HiddenState = Any
    mods["tensordict"].TensorDict = _TensorDict
    return mods


def load() -> dict[str, types.ModuleType]:
    """{'categorical', 'torch_buffer', 'flash_sac', 'reward_normalization', 'layers', 'lr_scheduler'}: the vendored
    modules."""
    saved_modules = dict(sys.modules)
    saved_path = list(sys.path)
    saved_dwb = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.modules.update(_stubs())
        sys.path.insert(0, str(ROOT))
        out = {
            "categorical": importlib.import_module("rsl_rl_flashsac.utils.categorical"),
            "torch_buffer": importlib.import_module("rsl_rl_flashsac.storage.torch_buffer"),
            "flash_sac": importlib.import_module("rsl_rl_flashsac.algorithms.flash_sac"),
            "reward_normalization": importlib.import_module("rsl_rl_flashsac.utils.reward_normalization"),
            "layers": importlib.import_module("rsl_rl_flashsac.modules.flash_sac_layers"),
            "lr_scheduler": importlib.import_module("rsl_rl_flashsac.utils.lr_scheduler"),
        }
    finally:
        for name in list(sys.modules):
            if name not in saved_modules:
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_dwb
    return out
