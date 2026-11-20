"""The mujoco_warp slot-union batch behind `continuous.run_continuous` (`retargeting` env only).

S slots of W worlds each (W = num_samples x num_perturb_samples of the registered config) on ONE `mjwarp.Model`/
`Data`. `assign(clips)` (re)builds the union for the resident clips (`union_scene` + `union_rows` + `union_warp`),
carrying every resident slot's state (current and saved STATE fields, `eq_active`, gravity and `eq_data` rows) over,
and re-points each resident `SlotCore`'s views and union geom masks. `adopt(slot, (cr, core))` copies a clip that
`clip.setup` (FAST, unchanged) just built on its own model into the slot: all worlds' state, its saved state, its weld
target (`eq_data`) and gravity rows, then drops the clip's own mujoco_warp env. `step()` launches the union step (a
captured CUDA graph, or `mjwarp.step` on the CPU device). With `check` (default) every step also counts, on the
device, contacts that touch a geom inactive for the contact's world (KA-U2; must stay 0), steps whose contact buffer
is full (nacon >= naconmax) and worlds whose constraint count exceeds njmax; `diagnostics()` reads them once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastdo.physics import union_slot as U
from fastdo.physics.union_rows import union_table
from fastdo.physics.union_scene import build_union_xml
from fastdo.physics.union_warp import EMPTY, put_union_model, union_data0, write_static_poses

__all__ = ["UnionBatch"]

WELD_SOLREF = (0.01, 1.0)                       # mjwp.set_weld_active(True) -> set_weld_solver constants
WELD_SOLIMP = (0.9995, 0.9995, 0.001, 0.5, 2.0)


def _configure_like(union: Any, ref: Any) -> None:
    """The union gets the clip models' (DO AS I DO `setup_mj_model`) options."""
    for f in ("timestep", "iterations", "ls_iterations", "o_solref", "o_solimp", "integrator"):
        setattr(union.opt, f, getattr(ref.opt, f))


class UnionBatch:
    def __init__(self, n_slots: int, worlds_per_slot: int, nconmax: int, njmax: int, device: str,
                 union_dir: str | Path, use_graph: bool | None = None, check: bool = True) -> None:
        self.S, self.W = n_slots, worlds_per_slot
        self.N = n_slots * worlds_per_slot
        self.nconmax, self.njmax, self.device = nconmax, njmax, device
        self.union_dir = Path(union_dir)
        self.use_graph = (str(device).startswith("cuda")) if use_graph is None else use_graph
        self.cores: list[Any | None] = [None] * n_slots
        self.clips: list[Any | None] = [None] * n_slots
        self.model: Any = None
        self.data: Any = None
        self.graph: Any = None
        self.table: Any = None
        self.mjm: Any = None
        self.cur: dict[str, torch.Tensor] = {}
        self.prev: dict[str, torch.Tensor] = {}
        self.build_seconds: list[float] = []
        self.check = check                 # run-time checks after every step (device-side; read once at the end)

    # --- build --------------------------------------------------------------------------------------------------
    def _world_clip(self, scenes: list[Path]) -> np.ndarray:
        wc = np.full(self.N, EMPTY, dtype=np.int64)
        for s, clip in enumerate(self.clips):
            if clip is not None:
                wc[s * self.W:(s + 1) * self.W] = scenes.index(Path(clip.scene).resolve())
        return wc

    def assign(self, clips: list[Any | None]) -> None:
        """Rebuild for `clips` (per slot: an object with `.scene` (scene.xml) and `.model` (its configured MjModel)
        and `.core` (`SlotCore`), or None); `len(clips)` slots of W worlds. Clips already resident keep their state
        (matched by identity, so a shrink may move them to another slot)."""
        import time

        import mujoco
        import mujoco_warp as mjwarp
        import warp as wp

        t0 = time.perf_counter()
        if not clips:
            raise ValueError("assign needs at least one slot")
        # residents keep their state, wherever they sit in the new assignment (a shrink compacts the slots)
        keep = {id(c): self._save_slot(s) for s, c in enumerate(self.clips)
                if c is not None and any(c is x for x in clips)}
        self.S, self.N = len(clips), len(clips) * self.W
        self.clips = list(clips)
        resident = [c for c in clips if c is not None]
        scenes: list[Path] = []
        models: list[Any] = []
        for c in resident:
            p = Path(c.scene).resolve()
            if p not in scenes:
                scenes.append(p)
                models.append(c.model)
        if not scenes:
            self.model = self.data = self.graph = None
            self.cores = [None] * self.S
            return
        spec = build_union_xml(scenes, self.union_dir)
        mjm = mujoco.MjModel.from_xml_path(str(spec.xml_path))
        _configure_like(mjm, models[0])
        table = union_table(mjm, models, spec)
        wc = self._world_clip(scenes)
        with wp.ScopedDevice(self.device):
            model = put_union_model(mjm, table, wc, device=self.device)
            data = mjwarp.put_data(mjm, union_data0(mjm), nworld=self.N, nconmax=self.nconmax, njmax=self.njmax)
            write_static_poses(data, mjm, table, wc)
            t = wp.to_torch
            welds = [i for i in range(mjm.neq) if (mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_EQUALITY, i) or "")
                     == f"{spec.side}_object_weld"]
            t(model.eq_solref)[:, welds, :] = torch.tensor(WELD_SOLREF, device=t(model.eq_solref).device)
            t(model.eq_solimp)[:, welds, :] = torch.tensor(WELD_SOLIMP, device=t(model.eq_solimp).device)
            graph = None
            if self.use_graph:
                mjwarp.step(model, data)
                mjwarp.step(model, data)
                wp.synchronize()
                with wp.ScopedCapture() as cap:
                    mjwarp.step(model, data)
                wp.synchronize()
                graph = cap.graph
        self.mjm, self.table, self.model, self.data, self.graph, self.scenes = mjm, table, model, data, graph, scenes
        self._bind()
        for s, clip in enumerate(self.clips):
            if clip is not None and id(clip) in keep:
                self._restore_slot(s, keep[id(clip)])
        self._bind_diag(wc, spec)
        self.cores = [None] * self.S
        for s, clip in enumerate(self.clips):
            self.cores[s] = None if clip is None else clip.core
            if clip is not None:
                k = scenes.index(Path(clip.scene).resolve())
                clip.core.rstat = U.remap_static(clip.core.rstat_native, table.geom_map[k], mjm.ngeom)
                clip.core.views = self.views(s)
        self.build_seconds.append(time.perf_counter() - t0)

    def _bind(self) -> None:
        import warp as wp

        t = wp.to_torch
        d = self.data
        self.cur = {f: t(getattr(d, f)) for f in set(U.BROADCAST_FIELDS) | set(U.STATE_FIELDS)}
        self.prev = {f: torch.zeros_like(self.cur[f]) for f in U.STATE_FIELDS}
        self.eq_active = t(d.eq_active)
        self.gravity = t(self.model.opt.gravity)
        self.eq_data = t(self.model.eq_data)
        self.contact = (t(d.contact.dist).reshape(-1), t(d.contact.geom).reshape(-1, 2),
                        t(d.contact.worldid).reshape(-1), t(d.nacon))

    def _bind_diag(self, wc: np.ndarray, spec: Any) -> None:
        """Per-world active-geom table for the run-time checks (KA-U2 on the device, buffer overflows)."""
        from fastdo.physics.union_rows import geom_names

        shared = np.array([n not in spec.owner for n in geom_names(self.mjm)])
        rows = np.where((wc == EMPTY)[:, None], shared[None, :], self.table.active[np.maximum(wc, 0)])
        dev = self.contact[0].device
        self._active = torch.as_tensor(rows.reshape(-1), device=dev)
        self._ngeom = int(self.mjm.ngeom)
        if not hasattr(self, "diag"):
            self.diag = {k: torch.zeros((), dtype=torch.int64, device=dev)
                         for k in ("steps", "parked_contacts", "contact_overflow_steps", "efc_overflow_worlds")}
        import warp as wp

        self._nefc = wp.to_torch(self.data.nefc)
        self._naconmax = int(self.contact[0].shape[0])

    def _accumulate_diag(self) -> None:
        dist, geom, wid, nacon = self.contact
        n = nacon.to(torch.int64)
        valid = torch.arange(dist.shape[0], device=dist.device) < n
        w = wid.to(torch.int64).clamp(0, self.N - 1)
        g = geom.to(torch.int64).clamp(0, self._ngeom - 1)
        a0 = self._active[w * self._ngeom + g[:, 0]]
        a1 = self._active[w * self._ngeom + g[:, 1]]
        self.diag["parked_contacts"] += (valid & ~(a0 & a1)).sum()
        self.diag["contact_overflow_steps"] += (n >= self._naconmax).to(torch.int64).sum()
        self.diag["efc_overflow_worlds"] += (self._nefc > self.njmax).sum()
        self.diag["steps"] += 1

    def diagnostics(self) -> dict[str, int]:
        """Host copy of the run-time checks (one read; call at the end of a run)."""
        return {k: int(v.item()) for k, v in getattr(self, "diag", {}).items()}

    def _rng(self, s: int) -> slice:
        return slice(s * self.W, (s + 1) * self.W)

    def _save_slot(self, s: int) -> dict[str, torch.Tensor]:
        r = self._rng(s)
        out = {f"cur.{f}": self.cur[f][r].clone() for f in U.STATE_FIELDS}
        out.update({f"prev.{f}": self.prev[f][r].clone() for f in U.STATE_FIELDS})
        out.update({"eq_active": self.eq_active[r].clone(), "gravity": self.gravity[r].clone(),
                    "eq_data": self.eq_data[r].clone()})
        return out

    def _restore_slot(self, s: int, saved: dict[str, torch.Tensor]) -> None:
        r = self._rng(s)
        for f in U.STATE_FIELDS:
            self.cur[f][r] = saved[f"cur.{f}"]
            self.prev[f][r] = saved[f"prev.{f}"]
        self.eq_active[r] = saved["eq_active"]
        self.gravity[r] = saved["gravity"]
        self.eq_data[r] = saved["eq_data"]

    # --- per slot -------------------------------------------------------------------------------------------------
    def views(self, s: int) -> U.SlotViews:
        r = self._rng(s)
        dist, geom, wid, nacon = self.contact
        return U.SlotViews(lo=r.start, hi=r.stop, cur={f: x[r] for f, x in self.cur.items()},
                           prev={f: x[r] for f, x in self.prev.items()}, eq_active=self.eq_active[r],
                           gravity=self.gravity[r], cv=U.SlotContactView(dist, geom, wid, nacon, r.start, r.stop))

    def adopt(self, s: int, clip_run: Any) -> None:
        """Copy the freshly set-up clip (`clip.ClipRun`, its own env still alive) into slot s, then drop that env."""
        import warp as wp

        t = wp.to_torch
        env = clip_run.core.env
        if env.num_worlds != self.W:
            raise ValueError(f"clip has {env.num_worlds} worlds, slots have {self.W}")
        r = self._rng(s)
        for f in self.cur:
            if f in ("geom_xpos", "geom_xmat"):      # clip geom ids != union geom ids; static rows are the union's
                continue
            self.cur[f][r] = t(getattr(env.data_wp, f)).to(self.cur[f].device)
        for f in U.STATE_FIELDS:
            self.prev[f][r] = t(getattr(env.data_wp_prev, f)).to(self.prev[f].device)
        self.eq_active[r] = t(env.data_wp.eq_active)
        self.gravity[r] = t(env.model_wp.opt.gravity)[0]
        self.eq_data[r] = t(env.model_wp.eq_data)[0]
        for name, val in (("eq_solref", WELD_SOLREF), ("eq_solimp", WELD_SOLIMP)):
            mine = t(getattr(env.model_wp, name))[0]
            ids = list(clip_run.core.config.warmup_weld_eq_ids or [])
            if ids and not torch.equal(mine[ids], torch.tensor(val, device=mine.device).expand(len(ids), -1)):
                raise ValueError(f"clip {name} rows {mine[ids].tolist()} differ from the union constant {val}")
        clip_run.core.env = None
        clip_run.core = None

    def step(self) -> None:
        import mujoco_warp as mjwarp
        import warp as wp

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            mjwarp.step(self.model, self.data)
        if self.check:
            self._accumulate_diag()
