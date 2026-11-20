"""C2 certificate (criteria C1-C7 + robustness): stricter than DO AS I DO's success, on one trajectory's numbers.

The same function serves the PREDICTED certificate (world 0 of a whole-clip rollout, inside the driver; C6 is not
available there, robustness is) and the POST-HOC certificate (the unchanged scorer's row of the executed run; C6 is,
robustness is not). The success thresholds equal `fastdo.retarget.isolation_report`'s (a test pins the equality;
that module is not importable in DO AS I DO's env, so they are repeated here).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fastdo.physics.planner_config import CertificateConfig

__all__ = ["CertificateInput", "certificate", "displacement_after", "post_warmup_means", "repair_checkpoint"]


@dataclass(frozen=True)
class CertificateInput:
    completed: bool
    whole_pos_m: float
    whole_rot_rad: float
    post_pos_m: float
    post_rot_rad: float
    max_pen_m: float
    obj_disp_m: float
    ref_disp_m: float
    contact_kept: float | None = None    # post-hoc only
    robust_frac: float | None = None     # predicted only


def certificate(x: CertificateInput, cfg: CertificateConfig) -> dict:
    """Item-wise verdicts and `passed` (all available items true). NaN anywhere fails its item."""
    ok = lambda v: bool(np.isfinite(v))  # noqa: E731
    items = {
        "C1_completed": bool(x.completed),
        "C2_whole": ok(x.whole_pos_m) and ok(x.whole_rot_rad) and x.whole_pos_m < cfg.success_pos_m
        and x.whole_rot_rad < cfg.success_rot_rad,
        "C3_post": ok(x.post_pos_m) and ok(x.post_rot_rad) and x.post_pos_m < cfg.success_pos_m
        and x.post_rot_rad < cfg.success_rot_rad,
        "C4_post_pos": ok(x.post_pos_m) and x.post_pos_m <= cfg.post_pos_max_m,
        "C5_penetration": ok(x.max_pen_m) and x.max_pen_m <= cfg.penetration_max_m,
        "C7_displacement": ok(x.obj_disp_m) and ok(x.ref_disp_m)
        and x.obj_disp_m >= cfg.displacement_ratio_min * x.ref_disp_m,
    }
    if x.contact_kept is not None:
        items["C6_contact"] = ok(x.contact_kept) and x.contact_kept >= cfg.contact_kept_min
    if x.robust_frac is not None:
        items["R_robust"] = ok(x.robust_frac) and x.robust_frac >= cfg.robust_frac_min
    return {"items": items, "passed": all(items.values()), "input": asdict(x)}


def post_warmup_means(pos: np.ndarray, rot: np.ndarray, warmup_steps: int, end_step: int) -> tuple[float, float]:
    """Means over executed steps [warmup, min(end, T)) (the scorer's `post_warmup_tracking_errors` window)."""
    end = min(int(end_step), len(pos))
    if end <= warmup_steps:
        return float("nan"), float("nan")
    return float(np.mean(pos[warmup_steps:end])), float(np.mean(rot[warmup_steps:end]))


def displacement_after(obj_pos: np.ndarray, start: int) -> float:
    """max_t |p(t) - p(start)| over t >= start (object position series (T, 3))."""
    if start >= len(obj_pos):
        return float("nan")
    return float(np.max(np.linalg.norm(obj_pos[start:] - obj_pos[start], axis=1)))


def repair_checkpoint(pos_err: np.ndarray, dropped_at: int | None, warmup_steps: int, cfg: CertificateConfig,
                      ctrl_steps: int = 100) -> tuple[int, int]:
    """(t_fail, t_c): first post-release step with object error > fail_err_m (or the drop step); the release step if
    none. t_c = max(0, ctrl_steps * floor((t_fail - repair_lead_steps) / ctrl_steps))."""
    post = np.nonzero(pos_err[warmup_steps:] > cfg.fail_err_m)[0]
    cands = [warmup_steps + int(post[0])] if len(post) else []
    if dropped_at is not None:
        cands.append(int(dropped_at))
    t_fail = min(cands) if cands else warmup_steps
    t_c = max(0, ctrl_steps * int(np.floor((t_fail - cfg.repair_lead_steps) / ctrl_steps)))
    return t_fail, t_c
