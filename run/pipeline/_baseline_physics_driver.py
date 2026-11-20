# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""Isolation study: invoke the DO AS I DO baseline's physics stage directly.

Runs *inside the `retargeting` conda env* -- it has no dependency on `fastdo`,
`hydra`, or anything else from this repo's own uv environment, only on the vendored
`third_party/do-as-i-do/retargeting` package. It is deliberately a thin, standalone script rather
than a `fastdo` module: `retarget_whisking.py` (this repo's own Hydra entry point) shells out to it
via `conda run -n retargeting python run/pipeline/_baseline_physics_driver.py ...` because the two
environments are not the same Python and cannot import each other's packages.

`launch.py` does not thread `--output-root-dir` into every pipeline stage
(a real bug), so this script calls `optimize_physics.main` directly on an
already-prepared scene directory (built by `fastdo.retarget.isolation.setup_arm_directory`) instead
of going through `launch.py`'s stages 1-4.5 again. It never writes to, or imports anything that
would require modifying, `third_party/do-as-i-do` itself -- only reads its `config/*.yaml` and
calls its `retargeting.pipeline.optimize_physics.main`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Keys the driver sets itself (dedicated flags or pinned run-mode values); `--override` may not set them, so one key is
# never set two ways. `noise_scale` is derived by `process_config` and popped before `Config(**...)`.
DRIVER_MANAGED_KEYS = frozenset({
    "task", "output_root_dir", "data_id", "robot_type", "embodiment_type", "seed", "max_sim_steps", "force",
    "show_viewer", "wait_on_finish", "save_video", "num_samples", "num_perturb_samples", "noise_scale"})


# The physics executor. "reference" runs DO AS I DO's `optimize_physics.main`; "fast" runs `fastdo.physics`
# (no per-step host syncs; identical registered MPC). Run metadata of a reference run is unchanged.
# "fast" is the production default -- its exactness gate passed on 16/16 clips (outcomes identical to the
# reference runs on both criteria; traces within the reference executor's own run-to-run spread) at 1.67x
# (conservative) to 1.91x GPU-s per clip. Registered studies that measured the reference pin --executor reference.
# The lock-step multi-clip path (`_fast_physics_batch_driver.py`) stays OPT-IN (1.40x conservative < 1.5x).
EXECUTORS = ("reference", "fast")
DEFAULT_EXECUTOR = "fast"
SRC = Path(__file__).resolve().parents[2] / "src"


# The simulator stack under the physics executor. "installed" = the `retargeting` env's own build (MuJoCo 3.4.0,
# mujoco_warp 0.0.1, Warp 1.10.1: DO AS I DO's). "mjwarp314" = MuJoCo 3.14.0 + mujoco_warp 3.14.0 + Warp 1.17.0 taken
# from a SITE directory (env FASTDO_MJWARP314_SITE) that holds ONLY those three packages and their dist-info; it is put
# first on sys.path before anything imports them, so torch, numpy and DO AS I DO stay the env's own. OPT-IN: the
# default stays "installed" (promotion is a separate decision). Only with --executor fast (the combination that was
# gated). Build the site with run/setup/make_mjwarp314_site.sh. Under it, `wp.copy` gets Warp 1.10.1's raw-memcpy
# semantics for differing element sizes (`fastdo.physics.warp_compat`).
# "mjwarp314_noccd" = the same site and versions, plus MuJoCo 3.14's multi-contact CCD switched OFF
# (`mjDSBL_MULTICCD` set on every MjModel DO AS I DO's `setup_mj_model` returns, before `mujoco_warp.put_model`;
# `fastdo.physics.multiccd`), as in DO AS I DO's MuJoCo 3.4.0. OPT-IN; "mjwarp314" is unchanged.
SIM_STACKS = ("installed", "mjwarp314", "mjwarp314_noccd")
DEFAULT_SIM_STACK = "installed"
NOCCD_STACKS = frozenset({"mjwarp314_noccd"})
SIM_STACK_SITE_ENV = "FASTDO_MJWARP314_SITE"
SIM_STACK_MODULES = ("mujoco", "mujoco_warp", "warp")
SIM_STACK_DISTS = {"mujoco": "mujoco", "mujoco_warp": "mujoco_warp", "warp": "warp_lang"}
_V314 = {"mujoco": "3.14.0", "mujoco_warp": "3.14.0", "warp": "1.17.0"}
SIM_STACK_VERSIONS = {"mjwarp314": dict(_V314), "mjwarp314_noccd": dict(_V314)}


def _is_stack_entry(name: str) -> bool:
    if name in SIM_STACK_MODULES:
        return True
    return name.endswith(".dist-info") and name.split("-", 1)[0] in set(SIM_STACK_DISTS.values())


def activate_sim_stack(stack: str, environ: Any = None, path: list[str] | None = None,
                       modules: Any = None) -> str | None:
    """Put `stack`'s site directory first on `path` (sys.path); returns it (None for "installed"). Must run before
    anything imports mujoco / mujoco_warp / warp. Refuses a site holding anything but those three packages (it
    would shadow the env's torch / numpy / DO AS I DO dependencies)."""
    environ = os.environ if environ is None else environ
    path = sys.path if path is None else path
    modules = sys.modules if modules is None else modules
    if stack not in SIM_STACKS:
        raise ValueError(f"unknown sim stack {stack!r}; choose from {SIM_STACKS}")
    if stack == "installed":
        return None
    loaded = [m for m in SIM_STACK_MODULES if m in modules]
    if loaded:
        raise RuntimeError(f"--sim-stack {stack}: {loaded} already imported; the stack must be selected first")
    site = environ.get(SIM_STACK_SITE_ENV, "")
    if not site:
        raise RuntimeError(f"--sim-stack {stack} needs {SIM_STACK_SITE_ENV}=<dir holding only "
                           f"{', '.join(SIM_STACK_MODULES)} (+ dist-info)>")
    site_p = Path(site)
    missing = [m for m in SIM_STACK_MODULES if not (site_p / m / "__init__.py").is_file()]
    if missing:
        raise RuntimeError(f"{SIM_STACK_SITE_ENV}={site}: missing packages {missing}")
    extra = sorted(e.name for e in site_p.iterdir() if not _is_stack_entry(e.name))
    if extra:
        raise RuntimeError(f"{SIM_STACK_SITE_ENV}={site}: holds more than the simulator stack: {extra}")
    path.insert(0, str(site_p))
    return str(site_p)


def sim_stack_versions() -> dict[str, str]:
    """Versions (dist metadata, as resolved on sys.path) and module files of the imported simulator stack."""
    from importlib import metadata

    out: dict[str, str] = {}
    for mod, dist in SIM_STACK_DISTS.items():
        m = __import__(mod)
        out[mod] = metadata.version(dist)
        out[f"{mod}_file"] = str(getattr(m, "__file__", ""))
    return out


def check_sim_stack(stack: str, versions: dict[str, str]) -> None:
    """Refuse a run whose imported stack is not the requested one (e.g. the site was shadowed)."""
    want = SIM_STACK_VERSIONS.get(stack, {})
    bad = {k: (versions.get(k), v) for k, v in want.items() if versions.get(k) != v}
    if bad:
        raise RuntimeError(f"--sim-stack {stack}: imported versions differ from the requested stack: {bad}")


def install_noccd_patch(mjwp: Any = None, mjwarp: Any = None, mujoco: Any = None) -> Any:
    """Under `--sim-stack mjwarp314_noccd`, wrap DO AS I DO's `setup_mj_model` so every MjModel carries
    `mjDSBL_MULTICCD` before `mujoco_warp.put_model` (which is wrapped as a guard). Our code only: third_party and the
    installed packages are untouched. Returns the hook (`stats()` goes to `arm_meta.sim_stack.multiccd`)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if mjwp is None:
        from retargeting.utils import mjwp
    if mjwarp is None:
        import mujoco_warp as mjwarp
    if mujoco is None:
        import mujoco
    from fastdo.physics.multiccd import install_noccd

    return install_noccd(mjwp, mjwarp, mujoco)


def noccd_reached(stats: dict[str, Any]) -> bool:
    """The patch reached the run: >= 1 patched model, >= 1 put_model, and every mujoco_warp Model has the bit."""
    flags = stats.get("put_model_disableflags", [])
    return (stats.get("setup_mj_model_patched", 0) >= 1 and stats.get("put_model_calls", 0) >= 1
            and len(flags) == stats["put_model_calls"] and all(int(f) & int(stats["mjDSBL_MULTICCD"]) for f in flags))


def install_stack_patches(stack: str) -> tuple[Any, Any]:
    """The process-level patches a non-installed stack needs: (warp_compat module or None, NoCCDHook or None).
    "installed" imports and patches NOTHING (its path is the original code). Any 3.14 stack gets the `warp_compat` copy
    shim; "mjwarp314_noccd" also gets the MULTICCD patch. Must run before the executor's setup."""
    if stack == "installed":
        return None, None
    # DO AS I DO copies float32 buffers into vector-typed arrays, which Warp 1.10.1 does as a raw
    # memcpy and Warp 1.17.0 refuses; reproduce 1.10.1's copy for exactly that case (fastdo.physics.warp_compat)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import warp

    from fastdo.physics import warp_compat as compat

    compat.install_copy_compat(warp)
    noccd = install_noccd_patch() if stack in NOCCD_STACKS else None
    return compat, noccd


def sim_stack_meta(stack: str, site: str | None, versions: dict[str, str], compat: Any, noccd: Any) -> dict[str, Any]:
    """`arm_meta.sim_stack`: name, site, versions (+ the shim's and the MULTICCD patch's counters when installed)."""
    return {"name": stack, "site": site, **versions,
            **({"warp_copy_compat": compat.stats()} if compat is not None else {}),
            **({"multiccd": noccd.stats()} if noccd is not None else {})}


def load_fast_executor() -> Any:
    """Import the sync-free executor (`fastdo.physics.clip.run_single`, plus the DO AS I DO modules its setup uses)
    so that no import lands inside the timed region. Returns `run_single(config) -> (errors, timings)`."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import retargeting.utils.mjwp
    import retargeting.utils.sampling  # noqa: F401

    from fastdo.physics.clip import run_single

    return run_single


def resolve_reward_stride(executor: str, stride: int | None) -> int:
    """Reward-stride lever (opt-in): the reward stride the run uses. None (flag absent) and 1 are the registered reward;
    s > 1 evaluates the step reward on every s-th rollout step only (`fastdo.physics.executor` docstring) and exists
    on the fast executor only (the reference executor is DO AS I DO's own code, never patched)."""
    if stride is None:
        return 1
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"--reward-stride must be a positive int, got {stride!r}")
    if stride != 1 and executor != "fast":
        raise ValueError(f"--reward-stride {stride} needs --executor fast (the reference executor is unpatched)")
    return stride


def resolve_fallback(executor: str, warm_start: str | None, iteration_cap: int | None,
                     hold_cap_s: float | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(d) capped MPC fallback (opt-in, fast executor only; `fastdo.amortised.fallback`): the keyword
    arguments for `run_single` and the `arm_meta.fallback` record ({}, None when no flag is given, so an unflagged
    run's call and metadata are unchanged). `--warm-start` is an npz with `knots` (K,) executor rows and `U` (K, R)
    residuals: the control plan becomes ctrl_ref + U interpolated (`fastdo.physics.clip` docstring);
    `--iteration-cap` caps the MPC iterations per control step (at most `fastdo.amortised.fallback.ITERATION_CAP`);
    `--hold-cap-s` stops the run at the first control step past that many seconds of the loop. A fallback hold
    (`--warm-start`) is refused without BOTH caps: an uncapped warm-started hold is not the registered fallback. The
    caps alone (no warm start) stay allowed (the cap known answers)."""
    if warm_start is None and iteration_cap is None and hold_cap_s is None:
        return {}, None
    if executor != "fast":
        raise ValueError("--warm-start / --iteration-cap / --hold-cap-s need --executor fast")
    from fastdo.amortised.fallback import ITERATION_CAP

    if warm_start is not None and (iteration_cap is None or hold_cap_s is None):
        raise ValueError("--warm-start (a fallback hold) needs --iteration-cap and --hold-cap-s")
    kw: dict[str, Any] = {}
    rec: dict[str, Any] = {"warm_start": warm_start, "iteration_cap": iteration_cap, "hold_cap_s": hold_cap_s}
    if warm_start is not None:
        import hashlib

        import numpy as np

        raw = Path(warm_start).read_bytes()
        with np.load(warm_start) as z:
            knots, U = np.asarray(z["knots"], np.int64), np.asarray(z["U"], np.float64)
        kw["warm_start"] = (knots, U)
        rec.update(warm_start_sha256=hashlib.sha256(raw).hexdigest(), knots=int(knots.size),
                   residual_max_abs=float(np.abs(U).max()) if U.size else 0.0)
    if iteration_cap is not None:
        if isinstance(iteration_cap, bool) or not isinstance(iteration_cap, int) or iteration_cap < 1:
            raise ValueError(f"--iteration-cap must be a positive int, got {iteration_cap!r}")
        if iteration_cap > ITERATION_CAP:
            raise ValueError(f"--iteration-cap {iteration_cap} > the design's cap {ITERATION_CAP} (design §4.4)")
        kw["iteration_cap"] = iteration_cap
    if hold_cap_s is not None:
        if not hold_cap_s >= 0.0:
            raise ValueError(f"--hold-cap-s must be >= 0, got {hold_cap_s!r}")
        kw["hold_cap_s"] = float(hold_cap_s)
    return kw, rec


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """`["key=value", ...]` -> `{key: value}`. Values are YAML scalars/lists (`1.6` -> float, `12` -> int,
    `true` -> bool, `[a, b]` -> list), as the baseline's own YAML config load types them. Refuses a malformed item,
    a key given twice and a driver-managed key (`DRIVER_MANAGED_KEYS`)."""
    import yaml

    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key or key != item.split("=", 1)[0] or not raw.strip():
            raise ValueError(f"--override expects key=value, got {item!r}")
        if key in out:
            raise ValueError(f"--override sets {key!r} twice")
        if key in DRIVER_MANAGED_KEYS:
            raise ValueError(f"--override {key!r}: driver-managed key (use its dedicated flag)")
        out[key] = yaml.safe_load(raw)
    return out


def check_override_keys(overrides: dict[str, Any], allowed: set[str] | frozenset[str]) -> None:
    """Refuse override keys that are not `Config` fields: `filter_config_fields` would silently drop them."""
    unknown = sorted(k for k in overrides if k not in allowed)
    if unknown:
        raise ValueError(f"--override keys are not baseline Config fields: {unknown}")


def _load_config_dict(baseline_root: Path, overrides: dict[str, Any]) -> Any:
    """Reproduces `launch.py::load_mjwp_config` (YAML defaults + do_as_i_do override + explicit
    overrides -> `retargeting.config.Config`) without importing `launch.py` itself, since that
    top-level script name would otherwise shadow/confuse `sys.modules` for a driver script that
    also needs `retargeting.config`."""
    from retargeting.config import Config, filter_config_fields, load_config_yaml

    config_dir = baseline_root / "config"
    cfg_dict = load_config_yaml(str(config_dir / "default.yaml"))
    override_path = config_dir / "override" / "do_as_i_do.yaml"
    if override_path.exists():
        cfg_dict.update(load_config_yaml(str(override_path)))
    cfg_dict.update(overrides)

    filtered = filter_config_fields(cfg_dict)
    if "pair_margin_range" in filtered:
        filtered["pair_margin_range"] = tuple(filtered["pair_margin_range"])
    if "xy_offset_range" in filtered:
        filtered["xy_offset_range"] = tuple(filtered["xy_offset_range"])
    filtered.pop("noise_scale", None)
    return Config(**filtered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True,
                        help="third_party/do-as-i-do/retargeting (contains config/ and retargeting/)")
    parser.add_argument("--output-root-dir", required=True, help="absolute; the arm's own output root")
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--robot-type", default="sharpa")
    parser.add_argument("--embodiment-type", default="right")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="override the do_as_i_do default (1024); None keeps it")
    parser.add_argument("--num-perturb-samples", type=int, default=None,
                        help="override the do_as_i_do default (4); None keeps it")
    parser.add_argument("--max-sim-steps", type=int, default=0, help="0 = full length (baseline default)")
    parser.add_argument("--override", action="append", default=None, metavar="KEY=VALUE",
                        help="set any other baseline Config field (repeatable; YAML-typed value)")
    parser.add_argument("--executor", choices=EXECUTORS, default=DEFAULT_EXECUTOR,
                        help="reference = DO AS I DO's optimize_physics.main; fast = the sync-free executor "
                             "(fastdo.physics; same samples/horizon/iterations/noise/criterion)")
    parser.add_argument("--sim-stack", choices=SIM_STACKS, default=DEFAULT_SIM_STACK,
                        help="installed = the env's MuJoCo 3.4.0 / mujoco_warp 0.0.1 / Warp 1.10.1; mjwarp314 = "
                             f"MuJoCo + mujoco_warp 3.14.0 / Warp 1.17.0 from ${SIM_STACK_SITE_ENV} (opt-in; "
                             "--executor fast only); mjwarp314_noccd = the same with multi-contact CCD off "
                             "(mjDSBL_MULTICCD, as MuJoCo 3.4.0; opt-in)")
    parser.add_argument("--reward-stride", type=int, default=None,
                        help="RS lever (opt-in, fast executor only): evaluate the step reward on every s-th "
                             "rollout step (x s; terminal step always). Absent = the registered reward")
    parser.add_argument("--warm-start", default=None,
                        help="fallback hold (opt-in, fast executor only): npz of knot rows + residuals; the "
                             "control plan becomes ctrl_ref + the interpolated residual")
    parser.add_argument("--iteration-cap", type=int, default=None,
                        help="fallback hold: cap on MPC iterations per control step (fast executor only)")
    parser.add_argument("--hold-cap-s", type=float, default=None,
                        help="fallback hold: stop at the first control step past this many loop seconds")
    parser.add_argument("--errors-out", required=True, help="path to save the tracking-error npz")
    parser.add_argument("--meta-out", required=True, help="path to save a small run-metadata json")
    args = parser.parse_args(argv)

    if args.sim_stack != "installed" and args.executor != "fast":
        parser.error(f"--sim-stack {args.sim_stack} is gated with --executor fast only")
    sim_site = activate_sim_stack(args.sim_stack)
    cli_overrides = parse_overrides(args.override)
    reward_stride = resolve_reward_stride(args.executor, args.reward_stride)
    fb_kw, fb_rec = resolve_fallback(args.executor, args.warm_start, args.iteration_cap, args.hold_cap_s)
    baseline_root = Path(args.baseline_root).resolve()
    sys.path.insert(0, str(baseline_root))

    overrides: dict[str, Any] = {
        "task": args.task,
        "output_root_dir": args.output_root_dir,
        "data_id": args.data_id,
        "robot_type": args.robot_type,
        "embodiment_type": args.embodiment_type,
        "seed": args.seed,
        "max_sim_steps": args.max_sim_steps,
        "force": True,
        "show_viewer": False,
        "wait_on_finish": False,
        "save_video": False,
    }
    if args.num_samples is not None:
        overrides["num_samples"] = args.num_samples
    if args.num_perturb_samples is not None:
        overrides["num_perturb_samples"] = args.num_perturb_samples
    if cli_overrides:
        from dataclasses import fields

        from retargeting.config import Config

        check_override_keys(cli_overrides, {f.name for f in fields(Config)})
        overrides.update(cli_overrides)

    config = _load_config_dict(baseline_root, overrides)

    import numpy as np
    from retargeting.pipeline.optimize_physics import main as optimize_physics

    # every import happens BEFORE t0, so `wall_seconds` times only the physics run, for both
    # executors (the reference path is byte-for-byte the original code; the fast path's modules load here too).
    run_single = load_fast_executor() if args.executor == "fast" else None
    stack_versions = sim_stack_versions()
    check_sim_stack(args.sim_stack, stack_versions)
    compat, noccd = install_stack_patches(args.sim_stack)

    t0 = time.perf_counter()
    executor_meta: dict[str, Any] | None = None
    if run_single is not None:
        kw: dict[str, Any] = {} if reward_stride == 1 else {"reward_stride": reward_stride}
        kw.update(fb_kw)
        errors, timings = run_single(config, **kw)
        executor_meta = {"executor": "fast", "fast_timings": timings}
    else:
        errors = optimize_physics(config)
    wall_s = time.perf_counter() - t0

    errors_out = Path(args.errors_out)
    errors_out.parent.mkdir(parents=True, exist_ok=True)
    if errors is not None:
        np.savez(errors_out, obj_pos_err=errors["obj_pos_err"], obj_quat_err=errors["obj_quat_err"],
                 pos_err_frames=errors["pos_err_frames"], quat_err_frames=errors["quat_err_frames"])

    meta_out = Path(args.meta_out)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta = _run_meta(task=args.task, wall_s=wall_s, overrides=overrides, config=config, errors=errors,
                     cli_overrides=cli_overrides)
    if executor_meta is not None:
        meta.update(executor_meta)
    meta["sim_stack"] = sim_stack_meta(args.sim_stack, sim_site, stack_versions, compat, noccd)
    if args.reward_stride is not None:          # recorded only when the flag is given (default metadata unchanged)
        meta["reward_stride"] = reward_stride
    if fb_rec is not None:                      # (d): only when a fallback flag is given
        meta["fallback"] = {**fb_rec, "capped": (executor_meta or {}).get("fast_timings", {}).get("capped"),
                            "max_num_iterations": int(config.max_num_iterations)}
    meta_out.write_text(json.dumps(meta, indent=2))
    if noccd is not None and not noccd_reached(noccd.stats()):
        # a patch that was installed but never reached the model mujoco_warp ran is a failed run, not a pass
        logger.error("--sim-stack %s: the MULTICCD patch did not reach mujoco_warp: %s", args.sim_stack, noccd.stats())
        return 1

    logger.info("=== driver done: task=%s wall=%.1fs errors=%s ===", args.task, wall_s, errors)
    return 0 if errors is not None else 1


def _run_meta(task: str, wall_s: float, overrides: dict[str, Any], config: Any,
              errors: dict[str, Any] | None, cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run metadata for `arm_meta.json`, read from the baseline's `Config` after `optimize_physics.
    main` returns: `process_config` and `main` mutate it in place, so `warmup_steps`, `output_dir`
    and the resolved `max_sim_steps` (`len(reference) - horizon - ctrl` when uncapped) are final.
    `raw_reference_frames` is the input `trajectory_kinematic.npz`'s frame count before resampling;
    `executed_steps` is the length of the per-step error series (warmup included); `ref_dt`/`sim_dt`
    give the resampled clip length, so a capped `max_sim_steps` is distinguishable from the clip. With `--override`,
    `cli_overrides` holds the requested values and `resolved_overrides` each key's value on the final
    `Config` (absent without overrides, so the metadata of an un-overridden run is unchanged)."""
    import numpy as np

    with np.load(config.data_path) as ref:
        raw_reference_frames = int(ref["qpos"].shape[0])
    meta = {
        "task": task, "wall_seconds": float(wall_s),
        "num_samples": overrides.get("num_samples", "default(1024)"),
        "num_perturb_samples": overrides.get("num_perturb_samples", "default(4)"),
        "has_errors": errors is not None,
        "output_dir": str(config.output_dir),
        "warmup_steps": int(config.warmup_steps),
        "max_sim_steps": int(config.max_sim_steps),
        "ctrl_steps": int(config.ctrl_steps),
        "max_num_iterations": int(config.max_num_iterations),
        "improvement_threshold": float(config.improvement_threshold),
        "improvement_check_steps": int(config.improvement_check_steps),
        "raw_reference_frames": raw_reference_frames,
        "executed_steps": 0 if errors is None else len(errors["pos_err_frames"]),
        "ref_dt": float(config.ref_dt), "sim_dt": float(config.sim_dt),
    }
    if cli_overrides:
        meta["cli_overrides"] = dict(cli_overrides)
        meta["resolved_overrides"] = {k: _jsonable(getattr(config, k, None)) for k in cli_overrides}
    return meta


def _jsonable(x: Any) -> Any:
    """A resolved Config value as JSON (numpy scalars/arrays and tuples become plain Python)."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")
    raise SystemExit(main())
