"""Import DO AS I DO's `retargeting.utils.mjwp` on CPU (no warp in this env) so its functions serve as the oracle.

`warp`, `mujoco_warp` and `loguru` are stubbed only for the duration of the module-scoped fixture; `wp.to_torch` and
`wp.from_torch` are identities (the fake env holds torch tensors directly) and `wp.copy` is a no-op (DO AS I DO only
copies a tensor onto its own storage there). Everything imported under the stubs is removed afterwards.
"""

from __future__ import annotations

import sys
import types

import pytest

from tests.baseline_paths import baseline_path

BASELINE = baseline_path("retargeting")
_STUBBED = ("warp", "mujoco_warp", "loguru")


def _stub_modules() -> dict[str, types.ModuleType]:
    wp = types.ModuleType("warp")
    wp.init = lambda: None
    wp.to_torch = lambda x: x
    wp.from_torch = lambda x: x
    wp.copy = lambda dst, src: None
    wp.ScopedCapture = type("ScopedCapture", (), {})
    mjw = types.ModuleType("mujoco_warp")
    mjw.Model = object
    mjw.Data = object
    lg = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, name: str):
            return lambda *a, **k: None

    lg.logger = _Logger()
    return {"warp": wp, "mujoco_warp": mjw, "loguru": lg}


@pytest.fixture(scope="module")
def daid():
    """(mjwp, math) modules of DO AS I DO, imported with warp stubbed; cleaned up after the module."""
    if not (BASELINE / "retargeting").is_dir():
        pytest.skip("DO AS I DO checkout not available")
    saved = {k: sys.modules.get(k) for k in _STUBBED}
    sys.modules.update(_stub_modules())
    sys.path.insert(0, str(BASELINE))
    try:
        from retargeting.utils import math as dmath
        from retargeting.utils import mjwp

        yield types.SimpleNamespace(M=mjwp, math=dmath)
    finally:
        sys.path.remove(str(BASELINE))
        for name in [m for m in sys.modules if m == "retargeting" or m.startswith("retargeting.")]:
            del sys.modules[name]
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
