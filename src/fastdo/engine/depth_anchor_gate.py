"""Certificate-gated depth anchor: the clip-level certificate and the re-solve bookkeeping.

The dead-zone depth anchor rescues clip-level depth OFFSETS but harms good clips when always on
(per-frame pointmap noise leaves the band on a few frames, and LM trial steps leave it transiently). Post hoc, gating
the anchor on the clip-level certificate |median_t d_t| >= 0.25 of the production solve selected 3/71 lab clips and
improved all 3. `solve_stage.depth_anchor_gate` (`DepthAnchorGateConfig`) wires that design into both engine solve
paths (`workers_solve.make_solve_worker` per clip, `batched_stages.SolveBatchStage` per group):

1. the production solve runs exactly as without the gate;
2. `depth_certificate` of its FINAL solution: d_t = log max(z_sol(t), 1e-6) - log z_obs(t), with z_sol the camera-frame
   z of the posed template centroid (s R c' + t, c the mean of the clip's `MeshSamples.points`, c' its anisotropic
   correction `aniso.corrected_template_points`, the identity for an isotropic solve) and z_obs `batch.frame_depths` of
   the clip's observations -- exactly the depth-anchor term's two depths (`terms.depth_anchor_term`,
   `batch._depth_anchor_fields`). Frames without z_obs are excluded; fewer than `MIN_VALID_FRAMES` valid frames give no
   certificate (never flagged);
3. flagged (c >= threshold): the clip is re-solved from the SAME `SolveCall` -- the same mesh, observations, initial
   poses and registration scale, and the same solver seed -- with `gated_solver_config` (lam_depth_anchor = lam,
   depth_anchor_band = band), so the anchor is the only difference, and the re-solve is kept;
4. `gate_record` goes into the call's context (`with_gate_record`) and `finish_solve` writes it to solve.json's
   `depth_anchor_gate` block. solve.npz of an unflagged clip is the production solve's: the certificate reads copies
   of the solution and adds no float op to anything that is written.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from fastdo.engine.solve_stage_config import DepthAnchorGateConfig
from fastdo.solver.aniso import corrected_template_points
from fastdo.solver.api import MeshSamples, SolverConfig, TrajectoryObservations, TrajectorySolution
from fastdo.solver.batch import frame_depths, valid_pixels_of
from fastdo.solver.terms import _ANCHOR_MIN_DEPTH

__all__ = ["GATE_KEY", "MIN_VALID_FRAMES", "DepthCertificate", "SolveManyFn", "apply_depth_anchor_gate",
           "depth_certificate", "gate_record", "gated_solver_config", "with_gate_record"]

logger = logging.getLogger(__name__)

GATE_KEY = "depth_anchor_gate"      # the SolveCall context key and the solve.json block
MIN_VALID_FRAMES = 3                 # below this many frames with an observed depth, no certificate


SolveManyFn = Callable[[list[MeshSamples], list[TrajectoryObservations], list[Tensor], list[float], SolverConfig],
                       list[TrajectorySolution]]


class _GateCall(Protocol):
    """What the gate reads of a `workers_solve.SolveCall` (a frozen dataclass)."""

    @property
    def mesh(self) -> MeshSamples: ...
    @property
    def obs(self) -> TrajectoryObservations: ...
    @property
    def init_poses(self) -> Tensor: ...
    @property
    def init_scale(self) -> float: ...
    @property
    def context(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class DepthCertificate:
    """One solve's clip-level depth certificate. `value` = |`median_d`|, None (as `median_d`) when fewer than
    `MIN_VALID_FRAMES` of the `n_frames` frames have an observed depth (`n_valid`)."""

    value: float | None
    median_d: float | None
    n_valid: int
    n_frames: int

    def flagged(self, threshold: float) -> bool:
        """`value >= threshold`; a clip without a certificate is never flagged."""
        return self.value is not None and self.value >= threshold

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def depth_certificate(mesh_points: Tensor, obs: TrajectoryObservations, solution: TrajectorySolution) -> DepthCertificate:
    """The certificate of `solution` on the clip it solved (module docstring, step 2). Float64 on the CPU, on copies:
    nothing it reads is modified. The median of an even count is the mean of the two middle values (numpy's)."""
    points, counts = valid_pixels_of(obs)
    z_obs = frame_depths(points, torch.cat([counts.new_zeros(1), counts.cumsum(0)]))
    centroid = corrected_template_points(mesh_points.detach().cpu().double().mean(0, keepdim=True), solution)[0]
    poses = solution.poses.detach().cpu().double()
    z_sol = (float(solution.scale) * (poses[:, 2, :3] @ centroid) + poses[:, 2, 3]).clamp_min(_ANCHOR_MIN_DEPTH)
    valid = torch.isfinite(z_obs)
    d = (z_sol[valid].log() - z_obs[valid].log()).numpy()
    n_valid, n_frames = int(valid.sum()), int(z_obs.shape[0])
    if n_valid < MIN_VALID_FRAMES:
        return DepthCertificate(value=None, median_d=None, n_valid=n_valid, n_frames=n_frames)
    median = float(np.median(d))
    return DepthCertificate(value=abs(median), median_d=median, n_valid=n_valid, n_frames=n_frames)


def gated_solver_config(base: SolverConfig, gate: DepthAnchorGateConfig) -> SolverConfig:
    """The re-solve's config: `base` with `lam_depth_anchor = gate.lam` and `depth_anchor_band = gate.band`, nothing
    else changed (the seed included, so the observation and template subsamples are the production solve's).

    Raises:
        ValueError: If `base` already builds the anchor (its first solve would not be the production solve).
    """
    if base.lam_depth_anchor > 0.0:
        raise ValueError(f"the depth-anchor gate re-solves a production solve, but its SolverConfig already has "
                         f"lam_depth_anchor={base.lam_depth_anchor}")
    return dataclasses.replace(base, lam_depth_anchor=gate.lam, depth_anchor_band=gate.band)


def gate_record(gate: DepthAnchorGateConfig, first: DepthCertificate, first_solution: TrajectorySolution,
                resolve: DepthCertificate | None = None) -> MappingProxyType[str, Any]:
    """solve.json's `depth_anchor_gate` block: the gate's constants, the production solve's certificate and whether it
    flagged; when flagged, the production solve's scale and iterations (the kept solution is the re-solve, which the
    `solver` block describes) and the re-solve's certificate.

    Raises:
        ValueError: If `resolve` is given for an unflagged certificate or missing for a flagged one.
    """
    flagged = first.flagged(gate.threshold)
    if flagged != (resolve is not None):
        raise ValueError(f"gate_record: flagged={flagged} but resolve certificate given={resolve is not None}")
    record: dict[str, Any] = {"threshold": gate.threshold, "lam": gate.lam, "band": gate.band,
                              "min_valid_frames": MIN_VALID_FRAMES, "certificate": first.as_dict(), "flagged": flagged,
                              "init": "the production solve's SolveCall (same init poses, registration scale, seed)"}
    if resolve is not None:
        record["production_solve"] = {"scale": float(first_solution.scale),
                                      "iterations": int(first_solution.iterations)}
        record["resolve_certificate"] = resolve.as_dict()
    return MappingProxyType(record)


def with_gate_record[CallT: _GateCall](call: CallT, record: Mapping[str, Any]) -> CallT:
    """A copy of the frozen `workers_solve.SolveCall` `call` whose context also carries `record` under `GATE_KEY`."""
    return dataclasses.replace(call, context=MappingProxyType({**call.context, GATE_KEY: record}))  # type: ignore[type-var]


def apply_depth_anchor_gate[CallT: _GateCall](
        clip_ids: Sequence[str], calls: Sequence[CallT], solutions: Sequence[TrajectorySolution],
        gate: DepthAnchorGateConfig, base: SolverConfig,
        solve_many: SolveManyFn) -> tuple[list[CallT], list[TrajectorySolution]]:
    """Steps 2-4 of the module docstring for a group of production solves (one clip in the per-clip path).

    Every flagged clip is re-solved in ONE `solve_many` call with `gated_solver_config(base, gate)` from its own
    call's `mesh`, `obs`, `init_poses` and `init_scale` (the production solve's arguments). Returns the calls with
    their `gate_record` and the solutions to keep: an unflagged clip's is the production solution object itself.
    A re-solve that raises propagates (the clips fail as their production solve would have).

    Raises:
        ValueError: If `base` already builds the anchor (`gated_solver_config`).
        RuntimeError: If `solve_many` returns a different number of solutions than it was given clips.
    """
    config = gated_solver_config(base, gate)
    certs = [depth_certificate(c.mesh.points, c.obs, s) for c, s in zip(calls, solutions, strict=True)]
    flagged = [i for i, cert in enumerate(certs) if cert.flagged(gate.threshold)]
    resolved: dict[int, TrajectorySolution] = {}
    if flagged:
        sub = [calls[i] for i in flagged]
        out = solve_many([c.mesh for c in sub], [c.obs for c in sub], [c.init_poses for c in sub],
                         [c.init_scale for c in sub], config)
        if len(out) != len(sub):
            raise RuntimeError(f"the gated re-solve returned {len(out)} solutions for {len(sub)} clips")
        resolved = dict(zip(flagged, out, strict=True))
    new_calls: list[CallT] = []
    kept: list[TrajectorySolution] = []
    for i, (clip_id, call, solution) in enumerate(zip(clip_ids, calls, solutions, strict=True)):
        again = resolved.get(i)
        after = None if again is None else depth_certificate(call.mesh.points, call.obs, again)
        new_calls.append(with_gate_record(call, gate_record(gate, certs[i], solution, after)))
        kept.append(solution if again is None else again)
        logger.info("depth-anchor gate %s: certificate %s over %d/%d frames -> %s%s", clip_id,
                    "n/a" if certs[i].value is None else f"{certs[i].value:.4f}", certs[i].n_valid, certs[i].n_frames,
                    "FLAGGED, re-solved" if again is not None else "unflagged, production solve kept",
                    "" if after is None else f" (re-solve certificate "
                    f"{'n/a' if after.value is None else f'{after.value:.4f}'})")
    return new_calls, kept
