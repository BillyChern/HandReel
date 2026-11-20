"""C2 planner: tier 0a (theta0) -> tier 0b (CEM over theta) -> tier 1 (D1 repair from a checkpoint).

Installed into DO AS I DO's `optimize_physics` from our side (see `install`): its `optimize` is replaced by
`C2Planner.optimize`, its `step_env` by `C2Planner.step_env` (adds the per-step servo), its `load_data` by a wrapper
that keeps the reference tensors. The executor loop, warmup, artifacts and errors stay DO AS I DO's.
At the first control step (sim step 0) the planner decides the tier with whole-clip rollouts and the predicted
certificate; afterwards each control step returns the tier-0 plan (no rollout) until the repair checkpoint t_c,
and D1's optimizer from there (warm start: the tier-0 plan, squeeze included).
"""

from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np
import torch

from fastdo.physics.c2_rollout import RolloutResult, TierZeroController, whole_clip_rollout
from fastdo.physics.c2_terms import frames_to_steps, hand_info, squeeze_trajectory
from fastdo.physics.certificate import CertificateInput, certificate, displacement_after, repair_checkpoint
from fastdo.physics.planner_config import C2Config

__all__ = ["C2Planner", "HandObjectReplayer"]

N_THETA = 7


class HandObjectReplayer:
    """The scorer's `Replayer.min_dist`, for the predicted certificate's penetration."""

    def __init__(self, scene_xml: str, side: str) -> None:
        self.m = mujoco.MjModel.from_xml_path(scene_xml)
        self.d = mujoco.MjData(self.m)
        body = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(self.m.nbody)]
        obj = body.index(f"{side}_object")
        self.obj = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == obj and self.m.geom_group[g] == 3}
        self.hand = {g for g in range(self.m.ngeom) if self.m.geom_group[g] == 3
                     and body[self.m.geom_bodyid[g]].startswith(f"{side}_") and g not in self.obj
                     and "arm_cyl" not in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")}

    def min_dist(self, q: np.ndarray) -> float:
        self.d.qpos[:] = q[: self.m.nq]
        mujoco.mj_forward(self.m, self.d)
        best = float("inf")
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.hand and g2 in self.obj) or (g2 in self.hand and g1 in self.obj):
                best = min(best, float(c.dist))
        return best

    def max_penetration(self, qpos: np.ndarray, start: int, end: int, stride: int = 5) -> float:
        d = [self.min_dist(qpos[i]) for i in range(start, min(end, len(qpos)), stride)]
        d = [x for x in d if np.isfinite(x)]
        return float(max(0.0, -min(d))) if d else 0.0


class C2Planner:
    def __init__(self, cfg: C2Config, d1_cfg: Any, m: Any, error_fn: Any) -> None:
        self.cfg = cfg
        self.d1_cfg = d1_cfg
        self.m = m                      # retargeting.utils.mjwp
        self.error_fn = error_fn        # retargeting.utils.tracking_error.compute_object_tracking_error
        self.ref: tuple | None = None   # (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos), device tensors
        self.ctl: TierZeroController | None = None
        self.theta: torch.Tensor | None = None
        self.mode = "undecided"
        self.t_c = 0
        self.d1_optimize: Any = None
        self.last_plan: torch.Tensor | None = None
        self.record: dict[str, Any] = {"tiers": [], "world_steps": {}, "walls": {}}
        self.pred_qpos0: np.ndarray | None = None   # world 0 of the rollout behind the chosen theta

    # ------------------------------------------------------------------ hooks installed into optimize_physics
    def wrap_load_data(self, load_data: Any) -> Any:
        def load(config: Any, data_path: str = "") -> Any:
            out = load_data(config, data_path)
            self.ref = out
            return out
        return load

    def wrap_step_env(self, step_env: Any) -> Any:
        import warp as wp

        def step(config: Any, env: Any, ctrl: torch.Tensor, perturb: bool = True, sim_step: int = 0) -> Any:
            if self.ctl is not None and self.theta is not None and not self.cfg.terms_off:
                if ctrl.dim() == 1:
                    ctrl = ctrl.unsqueeze(0).repeat(env.num_worlds, 1)
                else:
                    ctrl = ctrl.clone()
                qpos = wp.to_torch(env.data_wp.qpos)
                t_idx = torch.round(wp.to_torch(env.data_wp.time) / config.sim_dt).long()
                gains = self.theta[5].expand(env.num_worlds)
                ctrl[:, :3] += self.ctl.servo(qpos, t_idx, gains)
            return step_env(config, env, ctrl, perturb=perturb, sim_step=sim_step)
        return step

    def make_optimize(self, d1_optimize: Any) -> Any:
        self.d1_optimize = d1_optimize

        def optimize(config: Any, env: Any, ctrls: torch.Tensor, ref_slice: tuple,
                     progress_callback: Any = None) -> tuple[torch.Tensor, dict]:
            s = int(config.warmup_sim_step)
            if self.mode == "undecided":
                self._decide(config, env)
            h = int(config.horizon_steps)
            if self.mode == "tier0" or s < self.t_c:
                plan = self.ctl.plan(s, h, self.theta)
                self.last_plan = plan
                return plan, self._empty_info(config)
            warm = self._warm_start(config, s, h)
            new, info = self.d1_optimize(config, env, warm, ref_slice, progress_callback)
            self.last_plan = new
            return new, info
        return optimize

    # ------------------------------------------------------------------ internals
    def _empty_info(self, config: Any) -> dict:
        z = np.zeros(int(config.max_num_iterations))
        return {"improvement": z.copy(), "rew_max": z.copy(), "rew_min": z.copy(), "rew_median": z.copy(),
                "rew_mean": z.copy(), "d1_reverted": z.copy(), "opt_steps": np.array([0])}

    def _warm_start(self, config: Any, s: int, h: int) -> torch.Tensor:
        """Tier-0 plan at the first repair step; after it, DO AS I DO's receding update on the offset reference."""
        full = self.ctl.plan(s, h, self.theta)
        if s == self.t_c or self.last_plan is None:
            return full
        k = int(config.ctrl_steps)
        prev = self.last_plan[k:]
        tail = full[h - k:].clone()
        if config.blend_reference and prev.shape[0] > 0:
            decay = torch.linspace(1.0, 0.0, k, device=tail.device).unsqueeze(1)
            tail = tail + (prev[-1] - tail[0]) * decay
        return torch.cat([prev, tail], dim=0)

    def _build_controller(self, config: Any, env: Any) -> None:
        cfg = self.cfg
        side = "left" if config.embodiment_type == "left" else "right"
        model = mujoco.MjModel.from_xml_path(config.model_path)
        info = hand_info(model, side)
        with np.load(config.data_path) as z:
            frames = np.asarray(z["qpos"], dtype=np.float64)
        mu = float(model.pair_friction[0, 0]) * cfg.squeeze.mu_scale if model.npair else 0.7
        sq = squeeze_trajectory(model, frames, info, cfg.squeeze, float(config.ref_dt), mu)
        qpos_ref, _, ctrl_ref, _, _ = self.ref
        n_t = int(ctrl_ref.shape[0])
        w = int(config.warmup_steps)
        spf = int(config.ref_steps)
        sm = cfg.squeeze.smooth_frames
        gap = frames_to_steps(sq["gap"], w, spf, n_t, cfg.squeeze.ramp_steps, sm)
        fin = frames_to_steps(sq["finger"], w, spf, n_t, cfg.squeeze.ramp_steps, sm)          # (T, 5, n_f)
        held = frames_to_steps(sq["held"][:, None], w, spf, n_t, 0, sm)[:, 0]
        held[:w] = 0.0
        nu = int(config.nu)
        base = np.zeros((n_t, nu))
        finger = np.zeros((5, n_t, nu))
        if not cfg.terms_off:
            base[:, info.finger_ctrl] = gap
            finger[:, :, info.finger_ctrl] = np.transpose(fin, (1, 0, 2))
            if cfg.gravity_ff:      # gravity is zero in DO AS I DO's welded warmup: feedforward from the release on
                base[w:, info.z_ctrl] = (info.m_hand + info.m_obj * held[w:]) * 9.81 / info.kp_z
        dev = config.device
        nq = int(qpos_ref.shape[1])
        self.ctl = TierZeroController(
            ctrl_ref=ctrl_ref, base=torch.tensor(base, dtype=torch.float32, device=dev),
            finger=torch.tensor(finger, dtype=torch.float32, device=dev),
            finger_cols=torch.tensor(info.finger_ctrl, dtype=torch.long, device=dev), z_col=info.z_ctrl,
            joint_max=cfg.squeeze.joint_offset_max_rad, ref_obj_pos=qpos_ref[:, nq - 7:nq - 4].float(),
            obj_pos_slice=slice(nq - 7, nq - 4), release_step=w, servo_sat=cfg.servo_sat_m)
        self.record["controller"] = {
            "m_hand": info.m_hand, "m_obj": info.m_obj, "kp_z": info.kp_z, "mu": mu,
            "frames": int(frames.shape[0]), "frames_with_contact": int((sq["n_contacts"] > 0).sum()),
            "contacts_median": float(np.median(sq["n_contacts"])), "nnls_residual_median":
            float(np.median(sq["residual"][sq["n_contacts"] > 0])) if (sq["n_contacts"] > 0).any() else None,
            "gravity_ff_m": float(info.m_hand * 9.81 / info.kp_z), "terms_off": cfg.terms_off}

    def _theta0(self, dev: str) -> torch.Tensor:
        th = torch.zeros(N_THETA, device=dev)
        th[5] = 0.0 if self.cfg.terms_off else self.cfg.servo_gain0
        return th

    def _evaluate(self, config: Any, env: Any, r: RolloutResult, world_rows: np.ndarray, n_steps: int,
                  replayer: HandObjectReplayer | None) -> dict:
        """Predicted certificate of world 0 (+ robustness over `world_rows`)."""
        qpos_ref = self.ref[0].detach().cpu().numpy().astype(np.float64)
        err = self.error_fn(r.qpos0, qpos_ref[:n_steps], config.embodiment_type, "mjwp")
        pos, rot = np.asarray(err["pos_err_frames"]), np.asarray(err["quat_err_frames"])
        w = int(config.warmup_steps)
        end = int(config.max_sim_steps)
        nq = qpos_ref.shape[1]
        op = slice(nq - 7, nq - 4)
        post_p = float(np.mean(pos[w:end])) if end > w else float("nan")
        post_r = float(np.mean(rot[w:end])) if end > w else float("nan")
        c = self.cfg.certificate
        ok_w = (~r.dropped[world_rows]) & (r.whole_pos[world_rows] < c.success_pos_m) & \
            (r.whole_rot[world_rows] < c.success_rot_rad) & (r.post_pos[world_rows] < c.success_pos_m) & \
            (r.post_rot[world_rows] < c.success_rot_rad)
        pen = replayer.max_penetration(r.qpos0, w, end) if replayer is not None else 0.0
        z = r.qpos0[:, op.start + 2]
        drop_rows = np.nonzero((z < qpos_ref[: len(z), op.start + 2] - c.drop_below_ref_m) | ~np.isfinite(z))[0]
        x = CertificateInput(
            completed=not bool(r.dropped[0]), whole_pos_m=float(err["obj_pos_err"]), whole_rot_rad=float(err["obj_quat_err"]),
            post_pos_m=post_p, post_rot_rad=post_r, max_pen_m=pen,
            obj_disp_m=displacement_after(r.qpos0[:end, op], w), ref_disp_m=displacement_after(qpos_ref[:end, op], w),
            robust_frac=float(ok_w.mean()) if len(world_rows) else None)
        out = certificate(x, c)
        out["pos_err"] = pos
        out["drop_step"] = int(drop_rows[0]) if len(drop_rows) else None
        return out

    def _decide(self, config: Any, env: Any) -> None:
        t0 = time.perf_counter()
        cfg = self.cfg
        dev = config.device
        nw = int(env.num_worlds)
        self._build_controller(config, env)
        if cfg.force_repair_from is not None:
            if cfg.force_theta is None:
                raise ValueError("force_repair_from needs force_theta")
            self.mode, self.t_c = "repair", int(cfg.force_repair_from)
            self.theta = torch.tensor(cfg.force_theta, dtype=torch.float32, device=dev)
            self.record.update({"mode": self.mode, "t_c": self.t_c, "theta": list(cfg.force_theta),
                                "forced": True, "decide_wall_s": time.perf_counter() - t0})
            return
        n_steps = int(np.ceil(config.max_sim_steps / config.ctrl_steps) * config.ctrl_steps)
        replayer = HandObjectReplayer(config.model_path, "left" if config.embodiment_type == "left" else "right")
        det = torch.zeros(nw, dtype=torch.bool, device=dev)
        det[0] = True
        th0 = self._theta0(dev)
        r = whole_clip_rollout(config, env, self.m, self.ctl, th0.unsqueeze(0).repeat(nw, 1), det, n_steps,
                               self.ref[0], self.ref, False, int(config.max_sim_steps),
                               cfg.certificate.drop_below_ref_m)
        ev = self._evaluate(config, env, r, np.arange(1, nw), n_steps, replayer)
        self._log_tier("0a", th0, ev, r)
        best_theta, best_ev, tier = th0, ev, "0a"
        self.pred_qpos0 = r.qpos0
        if not ev["passed"] and not cfg.terms_off:
            best_theta, best_ev = self._search(config, env, n_steps, replayer, th0)
            tier = "0b"
        self.theta = best_theta
        if best_ev["passed"] or cfg.terms_off:
            self.mode, self.t_c = "tier0", 0
        else:
            t_fail, self.t_c = repair_checkpoint(best_ev["pos_err"], best_ev["drop_step"], int(config.warmup_steps),
                                                 cfg.certificate, int(config.ctrl_steps))
            self.mode = "repair"
            self.record["t_fail"] = t_fail
        self.record.update({"mode": self.mode, "tier": tier if self.mode == "tier0" else "1", "t_c": self.t_c,
                            "predicted_pass": bool(best_ev["passed"]), "theta": self.theta.cpu().tolist(),
                            "decide_wall_s": time.perf_counter() - t0})

    def _search(self, config: Any, env: Any, n_steps: int, replayer: HandObjectReplayer,
                th0: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        dev = config.device
        nw = int(env.num_worlds)
        per = nw // cfg.search_candidates
        lo = torch.tensor(cfg.search_lo, device=dev)
        hi = torch.tensor(cfg.search_hi, device=dev)
        mu = th0.clone()
        sig = torch.tensor(cfg.search_sigma, device=dev)
        det = torch.zeros(nw, dtype=torch.bool, device=dev)
        det[::per] = True
        gen = torch.Generator(device=dev).manual_seed(int(config.seed) + 270)
        ranked: list[tuple[bool, float, torch.Tensor]] = []
        for it in range(cfg.search_iterations):
            cand = mu + sig * torch.randn((cfg.search_candidates, N_THETA), generator=gen, device=dev)
            cand[0] = mu
            cand = torch.maximum(torch.minimum(cand, hi), lo)
            theta = cand.repeat_interleave(per, dim=0)
            r = whole_clip_rollout(config, env, self.m, self.ctl, theta, det, n_steps, self.ref[0], self.ref, True,
                                   int(config.max_sim_steps), cfg.certificate.drop_below_ref_m)
            obj = r.reward.reshape(cfg.search_candidates, per).mean(1)
            c = cfg.certificate
            ok = (~r.dropped) & (r.whole_pos < c.success_pos_m) & (r.whole_rot < c.success_rot_rad) & \
                (r.post_pos <= c.post_pos_max_m) & (r.post_rot < c.success_rot_rad)
            ok_c = ok.reshape(cfg.search_candidates, per).all(1)
            for i in range(cfg.search_candidates):
                ranked.append((bool(ok_c[i]), float(obj[i]), cand[i].clone()))
            elite = torch.tensor(np.argsort(-obj)[: cfg.search_elite].copy(), device=dev)
            mu = cand[elite].mean(0)
            sig = torch.maximum(cand[elite].std(0), 0.1 * torch.tensor(cfg.search_sigma, device=dev))
            self.record["tiers"].append({"tier": f"0b_it{it}", "obj_max": float(obj.max()), "n_ok": int(ok_c.sum()),
                                         "rollout_wall_s": r.wall_s, "world_steps": r.world_steps})
            self.record["world_steps"]["search"] = self.record["world_steps"].get("search", 0) + r.world_steps
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_theta, best_ev = ranked[0][2], None
        for _, _, th in ranked[:3]:
            theta = th.unsqueeze(0).repeat(nw, 1)
            det0 = torch.zeros(nw, dtype=torch.bool, device=dev)
            det0[0] = True
            r = whole_clip_rollout(config, env, self.m, self.ctl, theta, det0, n_steps, self.ref[0], self.ref, False,
                                   int(config.max_sim_steps), cfg.certificate.drop_below_ref_m)
            ev = self._evaluate(config, env, r, np.arange(1, nw), n_steps, replayer)
            self._log_tier("0b_verify", th, ev, r)
            if best_ev is None:
                best_theta, best_ev = th, ev
                self.pred_qpos0 = r.qpos0
            if ev["passed"]:
                self.pred_qpos0 = r.qpos0
                return th, ev
        return best_theta, best_ev

    def _log_tier(self, name: str, theta: torch.Tensor, ev: dict, r: RolloutResult) -> None:
        self.record["tiers"].append({
            "tier": name, "theta": theta.cpu().tolist(), "passed": bool(ev["passed"]),
            "items": ev["items"], "input": ev["input"], "drop_step": ev["drop_step"],
            "rollout_wall_s": r.wall_s, "world_steps": r.world_steps})
        self.record["world_steps"][name] = self.record["world_steps"].get(name, 0) + r.world_steps
