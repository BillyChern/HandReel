"""Physics stage for DO AS I DO's MuJoCo-Warp sampling MPC: fast executors and our own planners.

DO AS I DO (`third_party/do-as-i-do/retargeting`) is imported, never modified.

Sync-free executor: re-implements only the HOST side of its MPPI rollout loop so that no per-step
device->host synchronisation remains, and lets several clips run in lock-step on separate CUDA streams of one
process. Every per-clip quantity (samples, horizon, iterations, noise, random draws, reward, weights, criterion) is
DO AS I DO's, computed by the same kernels.
- `sync_free`: pure-torch reward pieces (reward, contact masks, quat_to_vel, RewardStatic); no warp import.
- `lockstep`: the cooperative scheduler (clips as generators, one CUDA stream each) and per-clip RNG slots.
- `perturb`: the pre-built, verified perturbation schedule (Philox emulation); pure torch.
- `executor`: the warp-dependent rollout/optimize loop; `clip`: setup + outer loop + single/lock-step runners
  (both run in the `retargeting` env only).
- `warp_compat`: Warp 1.10 `wp.copy` semantics for DO AS I DO's code under `--sim-stack mjwarp314` (opt-in).
- `multiccd`: MuJoCo 3.14's multi-contact CCD default switched off under `--sim-stack mjwarp314_noccd` (opt-in).

Our planners (default OFF, layered on the stock executor; each module importable without the others):
- `planner_config`: frozen opt-in configs (`D1Config`, `C2Config`; both disabled by default).
- `rank_estimator`: antithetic knot noise, Blom rank weights, DO AS I DO's weights (equivalence mode), acceptance.
- `d1`: drop-in replacements of DO AS I DO's `make_optimize_once_fn` / `make_optimize_fn` (variance-reduced MPC).
- `c2_terms`: CPU structured-controller terms (gravity feedforward, contact geometry, grasp-force NNLS, squeeze).
- `certificate`: the C2 certificate on per-step error series (DO AS I DO's success thresholds plus guards).
- `c2_rollout` / `c2_plan`: whole-clip rollouts and the tiered C2 planner (retargeting env only).

Continuous batching on a slot-union model (opt-in research path; not yet validated on the GPU):
- `union_scene` / `union_rows`: the slot-union MJCF of K same-hand clips and each clip's per-world rows (no warp).
- `union_warp`: one mujoco_warp Model with per-world rows; `union_batch`: the batch (rebuild, adopt, step).
- `union_slot`: the MPC of one clip on a world range; `continuous`: the scheduler and `clip.run` mirror;
  `union_run`: the shared entry point of the driver and the CPU known answers.
"""

__all__ = [
    "c2_plan",
    "c2_rollout",
    "c2_terms",
    "certificate",
    "clip",
    "continuous",
    "d1",
    "executor",
    "lockstep",
    "multiccd",
    "perturb",
    "planner_config",
    "rank_estimator",
    "sync_free",
    "union_batch",
    "union_rows",
    "union_run",
    "union_scene",
    "union_slot",
    "union_warp",
    "warp_compat",
]
