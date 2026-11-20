"""Run a queue of clips by continuous batching on a slot-union batch: the one entry point the GPU driver
(`run/pipeline/_union_physics_batch_driver.py`) and the CPU known-answer harness share.

Each clip is set up by FAST's own `clip.setup` (unchanged; its own RNG slot is captured right after, as in
`clip.run_many`), checked against the registered path, adopted into a free slot, and run by `continuous.slot_run`.
Returns one `Lane`-like record per clip, in queue order: `.result = (errors, timings)` or `.error`.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fastdo.physics import continuous as CT
from fastdo.physics import union_slot as U

__all__ = ["ClipOutcome", "run_union_queue"]


@dataclass
class ClipOutcome:
    name: str
    result: Any = None             # (errors, timings) like `lockstep.Lane.result`
    error: BaseException | None = None
    rng_state: Any = None          # the clip's generator state when it finished (KA-U3)


def run_union_queue(configs: Sequence[Any], n_slots: int, worlds_per_slot: int, device: str, union_dir: str,
                    decide: str = "emulate", rng_slot: Callable[[Any], Any] | None = None,
                    use_graph: bool | None = None, names: Sequence[str] | None = None,
                    stats_out: dict | None = None) -> list[ClipOutcome]:
    """Run every config to completion on `n_slots` slots. `rng_slot(config)` builds a clip's `RngSlot` (default:
    the CUDA generator's, active)."""
    import loguru
    from retargeting.utils import sampling as S
    from retargeting.utils.math import mul_quat

    from fastdo.physics import clip as CL
    from fastdo.physics.lockstep import RngSlot
    from fastdo.physics.union_batch import UnionBatch

    names = list(names) if names is not None else [str(i) for i in range(len(configs))]
    outcomes = {n: ClipOutcome(name=n) for n in names}
    first = configs[0]
    batch = UnionBatch(n_slots, worlds_per_slot, int(first.nconmax_per_env), int(first.njmax_per_env), device,
                       union_dir, use_graph=use_graph)
    make_slot = rng_slot or (lambda cfg: RngSlot(True, device=cfg.device))

    def start(slot: int, item: tuple[str, Any]) -> tuple[CT.Resident, Any]:
        name, cfg = item
        rslot = make_slot(cfg)
        cr = CL.setup(cfg, rslot, decide=decide)
        if cr is None:
            raise RuntimeError(f"{name}: output exists and force is off (refused in the union path)")
        rslot.capture()
        U.check_registered(cr.core.config)
        if cr.core.env.num_worlds != worlds_per_slot:
            raise ValueError(f"{name}: {cr.core.env.num_worlds} worlds, slots hold {worlds_per_slot}")
        core = U.SlotCore(config=cr.core.config, S=S, mul_quat=mul_quat, rslot=rslot, rng=cr.core.rng,
                          plan=cr.core.plan, rstat=cr.core.rstat, rstat_native=cr.core.rstat, views=None,
                          log=loguru.logger)
        out = outcomes[name]

        def gen():
            errors = yield from CT.slot_run(cr, core)
            out.result = (errors, cr.timings)
            out.rng_state = rslot.state
            return errors

        clip = types.SimpleNamespace(scene=cr.core.config.model_path, model=cr.mj_model, core=core)
        return CT.Resident(name=name, gen=gen(), clip=clip), cr

    def on_finish(res: CT.Resident) -> None:
        if res.error is not None:
            outcomes[res.name].error = res.error

    stats = CT.run_continuous(batch, n_slots, list(zip(names, configs, strict=True)), start, on_finish)
    if stats_out is not None:
        stats_out.update({"ticks": stats.ticks, "rebuilds": stats.rebuilds, "seconds": stats.seconds,
                          "busy_fraction": stats.slot_ticks_busy / max(stats.slot_ticks_total, 1),
                          "union_build_seconds": list(batch.build_seconds), "shrinks": stats.shrinks,
                          "checks": batch.diagnostics()})
    return [outcomes[n] for n in names]
