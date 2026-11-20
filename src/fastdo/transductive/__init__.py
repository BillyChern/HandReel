"""Transductive residual policy on DO AS I DO's physics executor.

One shared PPO policy per hand side, trained ON the clips it converts, adds a residual to DO AS I DO's kinematic warm
start; every stored success is re-executed deterministically and scored by the unchanged physics scorer.

Modules (import the one you need; the GPU backend imports warp lazily and runs only in the `retargeting` env):
- `clip`: a clip's executor problem (`ClipSpec`) and its cache;
- `config`: `EnvConfig` (control, reward, episode end, observation, assist curriculum);
- `reward`, `obs`: DO AS I DO's reward/tracking terms and the observation (numpy, the reference implementation);
- `sim_cpu`, `group`, `env`: the CPU backend (MuJoCo 3.4.0), per-world episode logic, multi-process frontend;
- `torch_terms`, `batch_group`, `sim_warp`, `batch_env`: the same logic batched in torch, the mujoco_warp backend
  (DO AS I DO's own simulator) and the GPU frontend;
- `certify`: deterministic re-execution and the scorer's run artifacts;
- `trainer`: the seeded, resumable PPO loop (the tracking-policy learner) with the success store.
"""

__all__ = ["batch_env", "batch_group", "certify", "clip", "config", "env", "group", "obs", "reward", "sim_cpu",
           "sim_warp", "torch_terms", "trainer"]
