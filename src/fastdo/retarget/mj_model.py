"""MuJoCo scene adapter: qpos layout, forward kinematics, point Jacobians, hand surface sampling.

Wraps the baseline's generated `scene.xml` (Sharpa Wave hand + floating wrist + object) with
`mujoco`'s Python bindings. The qpos/qvel layout below was *derived* from the shipped whisking
demo's `scene.xml` via `model.jnt_qposadr`/`model.jnt_type`/joint names, not assumed from the plan:

    joints (in model order)              qpos          qvel (dof)
    ------------------------------------  ------------  ------------
    right_pos_x/y/z   (slide x3)          0:3           0:3
    right_rot_x/y/z   (hinge x3)          3:6           3:6
    -> wrist total                        0:6  (6)      0:6  (6)
    right_thumb_{CMC_FE,CMC_AA,MCP_FE,
      MCP_AA,IP}                          6:11          6:11
    right_index_{MCP_FE,MCP_AA,PIP,DIP}   11:15         11:15
    right_middle_{...}                    15:19         15:19
    right_ring_{...}                      19:23         19:23
    right_pinky_{CMC,MCP_FE,MCP_AA,
      PIP,DIP}                            23:28         23:28
    -> hand total (hinge x22)             6:28 (22)     6:28 (22)
    right_object_joint (free)             28:35 (7)     28:34 (6)
    -> nq = 6 + 22 + 7 = 35, nv = 6 + 22 + 6 = 34, matching `trajectory_kinematic.npz`.

Two points the plan's own description got wrong, confirmed by inspecting the compiled model
rather than assumed: the wrist is *not* a MuJoCo free joint -- it is a chain of 3 `slide` + 3
`hinge` joints, i.e. world-frame translation followed by an intrinsic X-Y-Z Euler rotation (this
IS the wrist body's pose directly: no extra offset is introduced between `right_base_yaw` and
`right_hand_C_MC`). Only the object has a genuine 7-qpos/6-dof free joint.

Hand-link collision geometry in this scene is primitive shapes (box/cylinder/capsule), not
meshes -- the mesh assets referenced by the XML are visual-only (`group=1`); collision geoms use
`group=3`. `surface_points` builds trimesh primitives that match MuJoCo's own geom parameterisation
(box `size` = half-extents, cylinder/capsule `size` = (radius, half-length) along local +Z) and
samples them directly; a `MESH`-type collision geom would need separate handling and is not
exercised by the shipped scene, so it raises `NotImplementedError` rather than guessing.

`RobotScene.robot_dof` is the wrist+hand velocity block (`slice(0, 28)` here, formed by joining
`wrist_dof` and `hand_dof`, which are adjacent by construction) -- the tangent space the kinematic retargeting's
per-frame LM blocks are sized over. `point_jacobians` returns columns over exactly this slice, not
the full `nv`: the object free joint's 6 columns are structurally zero for any point rigidly
attached to the hand subtree (the object is a separate branch of the kinematic tree, parented
directly to world), which `tests/retarget/test_mj_model.py` verifies against the full-width
`mj_jac` output before asserting the two are consistent. `point_kinematics` is the shared,
batched-over-frames core (one `mj_kinematics`/`mj_comPos` forward pass per frame) that both
`point_jacobians` and `world_points` are built on, so a caller needing both world points and
Jacobians for the same `q` -- C2's per-frame contact measure and C3's per-frame LM residual both
do -- never pays for forward kinematics twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

__all__ = ["RobotScene"]

_WRIST_JOINT_SUFFIXES = ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")
_COLLISION_GEOM_GROUP = 3


def _joint_name(model: mujoco.MjModel, jid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    return name if name is not None else f"joint_{jid}"


def _contiguous_slice(addrs: list[int], label: str, kind: str) -> slice:
    """Smallest slice spanning sorted `addrs`, after checking they are exactly consecutive."""
    ordered = sorted(addrs)
    expected = list(range(ordered[0], ordered[0] + len(ordered)))
    if ordered != expected:
        raise ValueError(f"{label} {kind} addresses are not contiguous: {ordered}")
    return slice(ordered[0], ordered[0] + len(ordered))


def _derive_joint_layout(model: mujoco.MjModel) -> dict[str, slice]:
    """Partition joints into wrist/hand/object by type and name, then verify qpos/dof are packed.

    This does not assume joint order or a fixed hand DoF count: it (1) requires exactly one free
    joint (the object), (2) requires exactly one non-free joint per wrist-suffix name, (3) treats
    every remaining non-free joint as "hand", and (4) fails loudly if any group's qpos/dof
    addresses are not a contiguous block -- which would mean this scene's layout does not match
    the assumptions documented above.
    """
    free_jids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free_jids) != 1:
        raise ValueError(f"expected exactly one free joint (the object), found {len(free_jids)}")
    object_jid = free_jids[0]

    wrist_jids: list[int] = []
    for suffix in _WRIST_JOINT_SUFFIXES:
        matches = [j for j in range(model.njnt)
                   if j != object_jid and _joint_name(model, j).endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one wrist joint ending in {suffix!r}, "
                              f"found {len(matches)}: {[_joint_name(model, j) for j in matches]}")
        wrist_jids.append(matches[0])

    hand_jids = [j for j in range(model.njnt) if j != object_jid and j not in wrist_jids]
    if not hand_jids:
        raise ValueError("no hand joints found after removing the wrist and object joints")

    wrist_qpos = _contiguous_slice([int(model.jnt_qposadr[j]) for j in wrist_jids], "wrist", "qpos")
    hand_qpos = _contiguous_slice([int(model.jnt_qposadr[j]) for j in hand_jids], "hand", "qpos")
    object_qpos = slice(int(model.jnt_qposadr[object_jid]), int(model.jnt_qposadr[object_jid]) + 7)

    wrist_dof = _contiguous_slice([int(model.jnt_dofadr[j]) for j in wrist_jids], "wrist", "dof")
    hand_dof = _contiguous_slice([int(model.jnt_dofadr[j]) for j in hand_jids], "hand", "dof")

    covered = set(range(wrist_qpos.start, wrist_qpos.stop))
    covered |= set(range(hand_qpos.start, hand_qpos.stop))
    covered |= set(range(object_qpos.start, object_qpos.stop))
    if covered != set(range(model.nq)):
        raise ValueError(f"wrist+hand+object qpos slices do not exactly cover nq={model.nq}: "
                          f"covered {sorted(covered)}")

    return {
        "wrist": wrist_qpos,
        "hand": hand_qpos,
        "object": object_qpos,
        "wrist_dof": wrist_dof,
        "hand_dof": hand_dof,
    }


def _robot_dof_slice(layout: dict[str, slice]) -> slice:
    """The wrist+hand velocity block (`nq_hand+6` wide), joining `wrist_dof` and `hand_dof`.

    Requires the two to be adjacent (in either order); this holds for the derived layout above
    (`wrist_dof=0:6`, `hand_dof=6:28`) but is checked rather than assumed, since the kinematic retargeting sizes its
    per-frame LM blocks directly off this slice's width.
    """
    wrist, hand = layout["wrist_dof"], layout["hand_dof"]
    if wrist.stop == hand.start:
        return slice(wrist.start, hand.stop)
    if hand.stop == wrist.start:
        return slice(hand.start, wrist.stop)
    raise ValueError(f"wrist_dof {wrist} and hand_dof {hand} are not adjacent; "
                      "cannot form a contiguous robot_dof block")


def _hand_link_body_ids(model: mujoco.MjModel, layout: dict[str, slice]) -> list[int]:
    """Bodies in the hand kinematic subtree that carry at least one collision-group geom.

    The hand subtree is found structurally, not by name: it is every descendant of the body
    directly attached to the *last* wrist joint (the wrist's terminal frame), excluding the
    object's own body (defensive -- the object is parented directly to world in this scene, so it
    would not be reached by the descent anyway).
    """
    last_wrist_dof = layout["wrist_dof"].stop - 1
    last_wrist_jid = next(j for j in range(model.njnt) if int(model.jnt_dofadr[j]) == last_wrist_dof)
    wrist_terminal_body = int(model.jnt_bodyid[last_wrist_jid])

    object_jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    object_body = int(model.jnt_bodyid[object_jid])

    children: dict[int, list[int]] = {}
    for b in range(model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)

    subtree: set[int] = set()
    stack = list(children.get(wrist_terminal_body, []))
    while stack:
        b = stack.pop()
        if b in subtree or b == object_body:
            continue
        subtree.add(b)
        stack.extend(children.get(b, []))

    hand_links = [b for b in sorted(subtree)
                  if any(model.geom_bodyid[g] == b and model.geom_group[g] == _COLLISION_GEOM_GROUP
                         for g in range(model.ngeom))]
    return hand_links


def _object_contact_geoms(model: mujoco.MjModel, geom_ids: list[int], hand_links: list[int]) -> list[int]:
    """The subset of `geom_ids` (order kept) the simulator can collide with geometry off the hand links: a
    geom joined to a non-hand geom by an explicit `<contact><pair>`, or one whose contype/conaffinity bits
    let MuJoCo's dynamic collision filter consider it. The baseline's scenes use explicit pairs only."""
    hand = set(hand_links)
    paired: set[int] = set()
    for g1, g2 in zip(model.pair_geom1.tolist(), model.pair_geom2.tolist(), strict=True):
        on_hand1, on_hand2 = int(model.geom_bodyid[g1]) in hand, int(model.geom_bodyid[g2]) in hand
        if on_hand1 != on_hand2:
            paired.add(g1 if on_hand1 else g2)
    return [g for g in geom_ids if g in paired or model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0]


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz quaternion (4,) -> rotation matrix (3,3), via MuJoCo's own convention."""
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def _geom_local_mesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    """Collision geom -> `trimesh.Trimesh` in its parent body's local frame (applies geom pos/quat)."""
    gtype = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * np.asarray(size, dtype=np.float64))
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.uv_sphere(radius=float(size[0]))
    else:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        raise NotImplementedError(f"unsupported collision geom type {int(gtype)} on {name!r}; "
                                   "only box/cylinder/capsule/sphere are handled")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_matrix(model.geom_quat[geom_id])
    transform[:3, 3] = model.geom_pos[geom_id]
    mesh.apply_transform(transform)
    return mesh


@dataclass(frozen=True)
class RobotScene:
    """A loaded MuJoCo scene plus its resolved wrist/hand/object qpos-qvel layout.

    `model`/`data` are the live `mujoco` bindings; methods that need forward kinematics mutate
    `data.qpos` and re-run `mj_kinematics`/`mj_comPos` in place, so a `RobotScene` is a stateful,
    non-reentrant scratch object (not thread-safe) -- matching typical MuJoCo usage.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    xml_path: Path
    layout: dict[str, slice]

    @classmethod
    def from_xml(cls, path: str | Path) -> RobotScene:
        xml_path = Path(path)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        layout = _derive_joint_layout(model)
        _robot_dof_slice(layout)   # fail fast if wrist_dof/hand_dof can't form one contiguous block
        return cls(model=model, data=data, xml_path=xml_path, layout=layout)

    @property
    def robot_dof(self) -> slice:
        """The wrist+hand velocity block (`slice(0, 28)` here) that `point_jacobians` columns span."""
        return _robot_dof_slice(self.layout)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    def joint_layout(self) -> dict[str, slice]:
        """qpos slices `hand`/`wrist`/`object` and velocity slices `hand_dof`/`wrist_dof`."""
        return dict(self.layout)

    def _forward(self, q: np.ndarray) -> None:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)   # required before mj_jac (cdof/subtree_com)

    def fk(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """FK for every named body: q (T,nq) or (nq,) -> {body_name: (T,4,4) world transforms}."""
        q2 = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if q2.shape[1] != self.model.nq:
            raise ValueError(f"expected q with {self.model.nq} columns, got {q2.shape}")
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody)]
        out = {name: np.tile(np.eye(4, dtype=np.float64), (q2.shape[0], 1, 1))
               for name in names if name is not None}
        for t in range(q2.shape[0]):
            self._forward(q2[t])
            for b, name in enumerate(names):
                if name is None:
                    continue
                out[name][t, :3, :3] = self.data.xmat[b].reshape(3, 3)
                out[name][t, :3, 3] = self.data.xpos[b]
        return out

    def surface_points(self, n_per_link: int, seed: int | None = None
                        ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Sample `n_per_link` surface points (area-weighted) per hand link's collision geoms.

        Returns `(points, body_ids, body_names)`: `points` (U,3) are in each point's own body's
        local frame (geom pos/quat already applied); `body_ids` (U,) are MuJoCo body ids, ready to
        pass into `point_jacobians`/`world_points`; `body_names[i]` is the name of the i-th
        distinct hand-link body, in the same ascending body-id order the points are grouped by
        (i.e. `body_names == [name(b) for b in sorted(set(body_ids))]`).
        """
        rng = np.random.default_rng(seed)
        hand_links = _hand_link_body_ids(self.model, self.layout)
        points_chunks: list[np.ndarray] = []
        body_ids_chunks: list[np.ndarray] = []
        body_names: list[str] = []
        for b in hand_links:
            geom_ids = [g for g in range(self.model.ngeom)
                        if self.model.geom_bodyid[g] == b and self.model.geom_group[g] == _COLLISION_GEOM_GROUP]
            meshes = [_geom_local_mesh(self.model, g) for g in geom_ids]
            combined = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
            pts, _ = trimesh.sample.sample_surface(combined, n_per_link, seed=rng)
            points_chunks.append(np.asarray(pts, dtype=np.float64))
            body_ids_chunks.append(np.full(n_per_link, b, dtype=np.int64))
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            body_names.append(name if name is not None else f"body_{b}")
        points = np.concatenate(points_chunks, axis=0)
        body_ids = np.concatenate(body_ids_chunks, axis=0)
        return points, body_ids, body_names

    def hand_collision_geoms(self) -> list[int]:
        """Collision-group geom ids on the hand-link bodies (the bodies `surface_points` samples) that the
        simulator can collide with the object (`_object_contact_geoms`), ordered by body id then geom id --
        the primitives the clearance-band penetration term covers. On the Sharpa scene this drops the arm cylinder
        on the palm body, which is in none of the scene's contact pairs (`surface_points` still samples it)."""
        hand_links = _hand_link_body_ids(self.model, self.layout)
        geoms = [g for b in hand_links for g in range(self.model.ngeom)
                 if self.model.geom_bodyid[g] == b and self.model.geom_group[g] == _COLLISION_GEOM_GROUP]
        return _object_contact_geoms(self.model, geoms, hand_links)

    def point_kinematics(self, q: np.ndarray, body_ids: np.ndarray, local_pts: np.ndarray,
                          jacobian: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """World points (and, if requested, robot-only point Jacobians) for `local_pts`/`body_ids`.

        q is (T,nq) or (nq,) (promoted to T=1). Returns `(world, jac)`: `world` is (T,U,3);
        `jac`, when `jacobian` is True, is (T,U,3,nq_hand+6) over `robot_dof` (else `None`). Runs
        exactly one `mj_kinematics`/`mj_comPos` forward pass per frame -- the shared core
        `point_jacobians`/`world_points` are built on, so callers needing both world points and
        Jacobians for the same `q` never pay for forward kinematics twice.
        """
        q2 = np.atleast_2d(np.asarray(q, dtype=np.float64))
        n_frames = q2.shape[0]
        n_pts = len(body_ids)
        dof = self.robot_dof
        world = np.empty((n_frames, n_pts, 3), dtype=np.float64)
        jac: np.ndarray | None = np.empty((n_frames, n_pts, 3, dof.stop - dof.start), dtype=np.float64) \
            if jacobian else None
        for t in range(n_frames):
            self._forward(q2[t])
            for u in range(n_pts):
                b = int(body_ids[u])
                xmat = self.data.xmat[b].reshape(3, 3)
                world[t, u] = self.data.xpos[b] + xmat @ local_pts[u]
                if jacobian:
                    assert jac is not None
                    jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                    mujoco.mj_jac(self.model, self.data, jacp, None, world[t, u], b)
                    jac[t, u] = jacp[:, dof]
        return world, jac

    def world_points(self, q: np.ndarray, body_ids: np.ndarray, local_pts: np.ndarray) -> np.ndarray:
        """local_pts (U,3) attached to body_ids (U,) at a single configuration q (nq,) -> (U,3)."""
        q1 = np.asarray(q, dtype=np.float64)
        if q1.ndim != 1:
            raise ValueError(f"world_points expects a single configuration q (nq,), got shape {q1.shape}")
        world, _ = self.point_kinematics(q1[None, :], body_ids, local_pts, jacobian=False)
        return world[0]

    def point_jacobians(self, q: np.ndarray, body_ids: np.ndarray, local_pts: np.ndarray) -> np.ndarray:
        """Robot-only positional Jacobians of `local_pts`/`body_ids`: q (T,nq) or (nq,) -> (T,U,3,nq_hand+6).

        Columns span `robot_dof` (wrist+hand); the object free joint's columns are dropped since
        they are structurally zero for points rigidly attached to the hand (verified in
        `tests/retarget/test_mj_model.py` against the full-width `mj_jac` output).
        """
        _, jac = self.point_kinematics(q, body_ids, local_pts, jacobian=True)
        assert jac is not None
        return jac
