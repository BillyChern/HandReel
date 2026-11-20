# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
# Portions adapted from DO AS I DO (https://github.com/malik-group/do-as-i-do, commit 824591b),
# Copyright (c) 2026 malik-group, MIT License; see licenses/DO-AS-I-DO-MIT.txt and THIRD_PARTY_LICENSES.md.
"""Sync-free equivalents of DO AS I DO's per-step reward (`retargeting/utils/mjwp.py::get_reward`).

DO AS I DO's reward issues ~12 host synchronisations per rollout step: `nacon.item()` to slice the flat contact buffer,
`bool(tensor.any())` early exits, `bool(gate[sim_step])` reads of per-frame reference masks, boolean-mask indexing in
`quat_to_vel` and list indexing (`x[:, [i, j]]`, a pageable host->device index copy). Each is replaced here by an
operation that yields the SAME bits without leaving the device:

- the contact buffer is used whole with a validity mask `arange(L) < nacon`; masked entries contribute exact zeros
  to non-negative `amax` scatters (`include_self=True` on zeros), so every per-world maximum is unchanged;
- per-frame reference masks are copied to the host ONCE per clip (they are properties of the reference, never of
  the simulation), so every branch is decided on the host from a numpy lookup;
- `quat_to_vel` evaluates its elementwise formula on all rows and selects with `torch.where`; the compacted rows of
  the original see the same elementwise ops on the same inputs, so the selected values are bitwise identical;
- list indices become cached device index tensors (`index_select` gives the same contiguous (N, k) gather).

Pure torch/numpy: importable without warp, so it is unit-tested on CPU against DO AS I DO's own functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = [
    "ContactView",
    "RefMasks",
    "RewardStatic",
    "diff_qpos",
    "pair_contact_worlds",
    "penetration",
    "quat_sub",
    "quat_to_vel",
    "reward_static",
    "reward_step",
]


@dataclass
class ContactView:
    """Torch views of an mjwarp contact buffer: flat `dist` (L,), `geom` (L, 2), `worldid` (L,), `nacon` (1,)."""

    dist: torch.Tensor
    geom: torch.Tensor
    worldid: torch.Tensor
    nacon: torch.Tensor
    arange: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.arange = torch.arange(self.dist.shape[0], device=self.dist.device)

    def valid(self) -> torch.Tensor:
        """(L,) bool: entry is one of the first `nacon` (active) contacts."""
        return self.arange < self.nacon.to(self.arange.dtype)


@dataclass
class RefMasks:
    """Host copies of the per-frame reference gates of one object body (`MJWPEnv._inhand_gate_mask` etc.)."""

    inhand: np.ndarray | None
    pedestal: np.ndarray | None
    floor: np.ndarray | None

    @staticmethod
    def _at(mask: np.ndarray | None, step: int) -> bool:
        if mask is None or mask.size == 0:
            return False
        return bool(mask[min(step, mask.shape[0] - 1)])

    def state(self, step: int) -> tuple[bool, bool, bool, bool]:
        """`mjwp._ref_contact_state`: (has_gate, in_hand_ref, near_ped_ref, near_floor_ref)."""
        has_gate = self.inhand is not None and self.inhand.size > 0
        return has_gate, self._at(self.inhand, step), self._at(self.pedestal, step), self._at(self.floor, step)


@dataclass
class RewardStatic:
    """Per-clip constants the reward reads, resolved once (names, masks, index tensors, host flags)."""

    group_index: dict[str, torch.Tensor]
    weight: torch.Tensor
    is_hand: torch.Tensor | None
    is_object: torch.Tensor | None
    any_pedestal: torch.Tensor
    any_pedestal_host: bool
    is_floor: torch.Tensor
    is_floor_host: bool
    hand_host: bool
    side_body: dict[str, int]
    body_obj_mask: dict[int, torch.Tensor]
    body_obj_mask_host: dict[int, bool]
    ref_masks: dict[int, RefMasks]
    contact_site_index: torch.Tensor | None = None


def reward_static(config: Any, env: Any, body_name: Any, device: Any = None) -> RewardStatic:
    """Resolve a clip's `RewardStatic` from DO AS I DO's config (after `precompute_hand_object_geom_mask` and the
    `_qpos_*` caches) and env (`_side_object_body_id` and the per-frame gate dicts). `body_name(body_id) -> str`
    (mujoco's mj_id2name). Pure torch: no warp, so it is tested on CPU."""
    dev = torch.device(device if device is not None else config.device)
    body_mask: dict[int, torch.Tensor] = {}
    body_mask_host: dict[int, bool] = {}
    ref_masks: dict[int, RefMasks] = {}

    def host(d: dict, b: int) -> np.ndarray | None:
        m = d.get(b)
        return None if m is None else m.detach().cpu().numpy().astype(bool)

    for side in ("right", "left"):
        bid = env._side_object_body_id.get(side, -1)
        if bid < 0 or bid in body_mask:
            continue
        mask = config._is_right_object_geom if body_name(bid).startswith("right") else config._is_left_object_geom
        body_mask[bid], body_mask_host[bid] = mask, bool(mask.any())
        ref_masks[bid] = RefMasks(inhand=host(env._inhand_gate_mask, bid),
                                  pedestal=host(env._obj_pedestal_gate_mask, bid),
                                  floor=host(env._obj_floor_gate_mask, bid))
    any_ped = config._is_right_pedestal_geom | config._is_left_pedestal_geom
    csi = None
    if config.contact_rew_scale > 0.0 and len(config.contact_site_ids) > 0:
        csi = torch.as_tensor(list(config.contact_site_ids), device=dev, dtype=torch.long)
    return RewardStatic(
        group_index={k: torch.as_tensor(v, device=dev, dtype=torch.long)
                     for k, v in config._qpos_group_slices.items()},
        weight=config._qpos_weight_cache, is_hand=config._is_hand_geom, is_object=config._is_object_geom,
        any_pedestal=any_ped, any_pedestal_host=bool(any_ped.any()), is_floor=config._is_floor_geom,
        is_floor_host=bool(config._is_floor_geom.any()), hand_host=bool(config._is_hand_geom.any()),
        side_body=dict(env._side_object_body_id), body_obj_mask=body_mask, body_obj_mask_host=body_mask_host,
        ref_masks=ref_masks, contact_site_index=csi)


def quat_to_vel(quat: torch.Tensor) -> torch.Tensor:
    """`retargeting.utils.math.quat_to_vel` without the `torch.any` sync and boolean-mask compaction."""
    axis = quat[..., 1:4]
    sin_a_2 = torch.norm(axis, dim=-1, keepdim=True)
    zero_mask = sin_a_2[..., 0] == 0.0
    non_zero_mask = ~zero_mask
    # contiguous operands, as the original's boolean-indexed ones are (same kernel path on CPU; no-op for values)
    speed = 2.0 * torch.atan2(sin_a_2[..., 0].contiguous(), quat[..., 0].contiguous())
    speed = torch.where(speed > torch.pi, speed - 2.0 * torch.pi, speed)
    val = axis * speed[..., None] / sin_a_2
    return torch.where(non_zero_mask[..., None], val, torch.zeros_like(axis))


def quat_sub(qa: torch.Tensor, qb: torch.Tensor, mul_quat: Any) -> torch.Tensor:
    """`retargeting.utils.math.quat_sub` (DO AS I DO's `mul_quat` passed in) with the sync-free `quat_to_vel`."""
    qneg = qb.clone()
    qneg[..., 1:] = -qneg[..., 1:]
    qdif = mul_quat(qneg, qa)
    return quat_to_vel(qdif)


def diff_qpos(config: Any, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor, mul_quat: Any) -> torch.Tensor:
    """`mjwp._diff_qpos`, line for line, with the sync-free `quat_sub`."""
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=qpos_sim.device)
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            qpos_diff[:, :-12] = qpos_sim[:, :-12] - qpos_ref[:, :-12]
            qpos_diff[:, -12:-9] = qpos_sim[:, -12:-9] - qpos_ref[:, -12:-9]
            qpos_diff[:, -9:-6] = qpos_sim[:, -9:-6] - qpos_ref[:, -9:-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7], mul_quat)
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:], mul_quat)
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            qpos_diff[:, :-6] = qpos_sim[:, :-6] - qpos_ref[:, :-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:], mul_quat)
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return qpos_diff


def penetration(st: RewardStatic, cv: ContactView, num_worlds: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`mjwp.check_penetration` over the WHOLE buffer: inactive entries are exact zeros (not dropped)."""
    pen = torch.clamp(-cv.dist, min=0.0)
    pen = torch.nan_to_num(pen, nan=0.0)
    if st.is_hand is not None:
        ngeom = st.is_hand.shape[0]
        g1 = cv.geom[:, 0].long().clamp(0, ngeom - 1)
        g2 = cv.geom[:, 1].long().clamp(0, ngeom - 1)
        ho_mask = (st.is_hand[g1] & st.is_object[g2]) | (st.is_object[g1] & st.is_hand[g2])
        pen = pen * ho_mask
    pen = torch.where(cv.valid(), pen, torch.zeros_like(pen))
    return pen, cv.worldid.long().clamp(0, num_worlds - 1)


def pair_contact_worlds(cv: ContactView, a_mask: torch.Tensor, b_mask: torch.Tensor, a_any: bool, b_any: bool,
                        num_worlds: int) -> torch.Tensor:
    """`mjwp._pair_contact_worlds`; `a_any`/`b_any` are the host-cached `mask.any()` of the two STATIC geom masks."""
    out = torch.zeros(num_worlds, dtype=torch.bool, device=a_mask.device)
    if not (a_any and b_any):
        return out
    worldid = cv.worldid.long().clamp(0, num_worlds - 1)
    touching = (cv.dist <= 0.0) & ~torch.isnan(cv.dist)
    ngeom = a_mask.shape[0]
    g1 = cv.geom[:, 0].long().clamp(0, ngeom - 1)
    g2 = cv.geom[:, 1].long().clamp(0, ngeom - 1)
    pair = (a_mask[g1] & b_mask[g2]) | (b_mask[g1] & a_mask[g2])
    hit = pair & touching & cv.valid()
    per_world = torch.zeros(num_worlds, device=a_mask.device)
    per_world.scatter_reduce_(0, worldid, hit.float(), reduce="amax", include_self=True)
    return per_world > 0.5


def _rest_surface_penalty(reward: torch.Tensor, st: RewardStatic, cv: ContactView, config: Any, sim_step: int, *,
                          scale: float, surface_mask: torch.Tensor, surface_any: bool, surface: str,
                          num_worlds: int) -> dict[str, torch.Tensor]:
    """`mjwp._rest_surface_penalty` with host-side branch decisions (module docstring)."""
    out: dict[str, torch.Tensor] = {}
    if not (scale > 0.0 and surface_any):
        return out
    in_warmup = int(sim_step) < int(config.warmup_steps)
    seen: set[int] = set()
    for side in ("right", "left"):
        obj_body_id = st.side_body.get(side, -1)
        if obj_body_id < 0 or obj_body_id in seen:
            continue
        seen.add(obj_body_id)
        has_gate, in_hand_ref, near_ped_ref, near_floor_ref = st.ref_masks[obj_body_id].state(sim_step)
        if not has_gate:
            continue
        pen = torch.zeros_like(reward)
        if not in_warmup:
            near_ref = near_floor_ref if surface == "floor" else near_ped_ref
            obj_mask = st.body_obj_mask[obj_body_id]
            obj_any = st.body_obj_mask_host[obj_body_id]
            if in_hand_ref and not near_ref:
                mismatch = ~pair_contact_worlds(cv, st.is_hand, obj_mask, st.hand_host, obj_any, num_worlds)
                pen = scale * mismatch.float()
            elif near_ref and not in_hand_ref:
                mismatch = ~pair_contact_worlds(cv, surface_mask, obj_mask, surface_any, obj_any, num_worlds)
                pen = scale * mismatch.float()
        out[side] = pen
    return out


def reward_step(config: Any, st: RewardStatic, qpos_sim: torch.Tensor, qvel_sim: torch.Tensor,
                site_xpos: torch.Tensor | None, cv: ContactView, ref: tuple[torch.Tensor, ...], sim_step: int,
                mul_quat: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`mjwp.get_reward`, op for op, with no host synchronisation (module docstring)."""
    qpos_ref, qvel_ref, _ctrl_ref, contact_ref, contact_pos_ref = ref
    n = qpos_sim.shape[0]
    qpos_diff = diff_qpos(config, qpos_sim, qpos_ref.unsqueeze(0).expand(n, -1), mul_quat)
    delta_qpos = qpos_diff * st.weight
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)
    qpos_rew = -qpos_dist * 1.0
    qvel_rew = -config.vel_rew_scale * qvel_dist * 1.0
    qpos_group_rew = {name: -torch.norm(delta_qpos.index_select(1, idx), p=2, dim=1)
                      for name, idx in st.group_index.items()}
    if config.contact_rew_scale > 0.0 and len(config.contact_site_ids) > 0:
        contact_pos = site_xpos.index_select(1, st.contact_site_index)
        contact_dist = torch.norm(contact_pos - contact_pos_ref, p=2, dim=-1)
        contact_dist_masked = contact_dist * contact_ref.unsqueeze(0)
        contact_rew: torch.Tensor | float = -contact_dist_masked.sum(dim=1)
    else:
        contact_rew = 0.0
    reward = qpos_rew + qvel_rew + contact_rew

    pen_penalty = torch.zeros_like(reward)
    if config.penetration_penalty_scale > 0.0:
        pen, wid = penetration(st, cv, n)
        pen = torch.clamp(pen - config.penetration_margin, min=0.0)
        pen_sq = pen * pen
        pen_penalty = torch.zeros(n, device=qpos_sim.device)
        pen_penalty.scatter_reduce_(0, wid, pen_sq, reduce="amax", include_self=True)
        pen_penalty = config.penetration_penalty_scale * pen_penalty
        reward = reward - pen_penalty

    drop_penalty = torch.zeros_like(reward)
    if config.drop_penalty_scale > 0.0:
        obj_z = _object_z(config, qpos_sim)
        drop = torch.clamp(config.drop_z_thresh - obj_z, min=0.0)
        drop_penalty = config.drop_penalty_scale * (drop * drop).sum(dim=-1)
        reward = reward - drop_penalty

    pedestal_pens = _rest_surface_penalty(reward, st, cv, config, sim_step, scale=config.pedestal_penalty_scale,
                                          surface_mask=st.any_pedestal, surface_any=st.any_pedestal_host,
                                          surface="pedestal", num_worlds=n)
    floor_pens = _rest_surface_penalty(reward, st, cv, config, sim_step, scale=config.floor_penalty_scale,
                                       surface_mask=st.is_floor, surface_any=st.is_floor_host, surface="floor",
                                       num_worlds=n)
    for pens in (pedestal_pens, floor_pens):
        for p in pens.values():
            reward = reward - p
    info = {"qpos_dist": qpos_dist, "qvel_dist": qvel_dist, "qpos_rew": qpos_rew, "qvel_rew": qvel_rew,
            "pen_penalty": pen_penalty, "drop_penalty": drop_penalty,
            **{f"pedestal_{s}": p for s, p in pedestal_pens.items()},
            **{f"floor_{s}": p for s, p in floor_pens.items()}, **qpos_group_rew}
    return reward, info


def _object_z(config: Any, qpos_sim: torch.Tensor) -> torch.Tensor:
    """`mjwp._get_object_z`."""
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            return torch.stack([qpos_sim[:, -10], qpos_sim[:, -4]], dim=1)
        return torch.stack([qpos_sim[:, -12], qpos_sim[:, -5]], dim=1)
    if config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            return qpos_sim[:, -4].unsqueeze(1)
        return qpos_sim[:, -5].unsqueeze(1)
    return torch.zeros(qpos_sim.shape[0], 1, device=qpos_sim.device)
