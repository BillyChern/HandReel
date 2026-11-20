"""MuJoCo 3.14's multi-contact CCD default switched off for the opt-in 3.14 sim stack.

MuJoCo 3.14 turned multi-contact convex collision from an opt-in ENABLE flag (`mjENBL_MULTICCD`, off in DO AS I DO's
MuJoCo 3.4.0) into an opt-out DISABLE flag (`mjDSBL_MULTICCD`, so ON by default). mujoco_warp 3.14 follows it
(`io.put_model` and `collision_convex`: `use_multiccd = (opt.disableflags & MULTICCD) == 0`). On one failing clip
this turns one support-pedestal contact into four and decides the outcome; CPU 3.14 with the bit set reproduces 3.4.0
on 16/16 gate clips.

`install_noccd(mjwp, mjwarp, mujoco)` makes every MjModel that DO AS I DO's `retargeting.utils.mjwp.setup_mj_model`
returns carry `mjDSBL_MULTICCD`, so the bit is on the model before `setup_env` steps it on the CPU and hands it to
`mujoco_warp.put_model` (`mjwp.py:265-277`; both look the functions up as module attributes at call time). It also
wraps `mujoco_warp.put_model` as a GUARD: a model without the bit is refused (RuntimeError), and the disableflags of
every mujoco_warp Model produced are recorded for `arm_meta`. DO AS I DO's code and the installed packages are never
modified; the wrappers live in the process only. The driver installs this only under `--sim-stack mjwarp314_noccd`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["NoCCDHook", "disable_bit", "disable_multiccd", "install_noccd"]

MARK = "_fastdo_noccd"


def disable_bit(mujoco: Any) -> int:
    """`mujoco.mjtDisableBit.mjDSBL_MULTICCD` as an int (absent before MuJoCo 3.x's opt-out flag: refuse)."""
    bit = getattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD", None)
    if bit is None:
        raise RuntimeError(f"MuJoCo {getattr(mujoco, '__version__', '?')} has no mjDSBL_MULTICCD")
    return int(bit)


def disable_multiccd(model: Any, mujoco: Any) -> Any:
    """Set `mjDSBL_MULTICCD` on `model.opt.disableflags` in place (every other bit unchanged); returns `model`."""
    model.opt.disableflags = int(model.opt.disableflags) | disable_bit(mujoco)
    return model


def _flags_int(x: Any) -> int:
    """mujoco_warp keeps `opt.disableflags` as a Python int; accept a 0-d / 1-element array too."""
    try:
        return int(x)
    except TypeError:
        import numpy as np

        return int(np.asarray(x).reshape(-1)[0])


class NoCCDHook:
    """The installed wrappers and their counters (`stats()` goes to `arm_meta.sim_stack.multiccd`)."""

    def __init__(self, bit: int) -> None:
        self.bit = bit
        self.models_patched = 0
        self.put_model_calls = 0
        self.put_model_disableflags: list[int] = []

    def stats(self) -> dict[str, Any]:
        return {"multiccd": "disabled", "mjDSBL_MULTICCD": self.bit, "setup_mj_model_patched": self.models_patched,
                "put_model_calls": self.put_model_calls,
                "put_model_disableflags": list(self.put_model_disableflags)}


def install_noccd(mjwp: Any, mjwarp: Any, mujoco: Any) -> NoCCDHook:
    """Wrap `mjwp.setup_mj_model` (set the bit) and `mjwarp.put_model` (refuse a model without it; record the
    resulting Model's flags). Refuses a second install (one hook per process, counters stay meaningful)."""
    if getattr(mjwp.setup_mj_model, MARK, False) or getattr(mjwarp.put_model, MARK, False):
        raise RuntimeError("install_noccd: already installed in this process")
    hook = NoCCDHook(disable_bit(mujoco))
    setup_orig = mjwp.setup_mj_model
    put_orig = mjwarp.put_model

    def setup_mj_model(*args: Any, **kwargs: Any) -> Any:
        model = disable_multiccd(setup_orig(*args, **kwargs), mujoco)
        hook.models_patched += 1
        return model

    def put_model(mjm: Any, *args: Any, **kwargs: Any) -> Any:
        if not int(mjm.opt.disableflags) & hook.bit:
            raise RuntimeError("--sim-stack mjwarp314_noccd: a model without mjDSBL_MULTICCD reached "
                               "mujoco_warp.put_model (built outside setup_mj_model?)")
        m = put_orig(mjm, *args, **kwargs)
        hook.put_model_calls += 1
        hook.put_model_disableflags.append(_flags_int(m.opt.disableflags))
        return m

    for fn, orig in ((setup_mj_model, setup_orig), (put_model, put_orig)):
        setattr(fn, MARK, True)
        fn.__wrapped__ = orig  # type: ignore[attr-defined]
    mjwp.setup_mj_model = setup_mj_model
    mjwarp.put_model = put_model
    return hook
