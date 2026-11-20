"""Row-level stages of the kinematic robot path (`fastdo.robot.kinematic_path` runs them): wrist targets,
wrist-spike rows, the human hand-object distance, the contact anchor, the robot-frame targets, and the export warnings.
The stage descriptions (numbers, smoke-clip evidence) are in `fastdo.robot.kinematic_path`'s docstring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from fastdo.robot.dex3_retarget import mano_hand_frame
from fastdo.robot.placement import transform_poses

__all__ = ["StaticWindow", "contact_anchor", "hand_object_distance", "object_motion", "path_warnings",
           "robot_frame_targets", "spike_rows", "static_window", "task_span", "wrist_targets_world"]


def path_warnings(camera_source: str, gravity_source: str, pitch_deg: float | None,
                  wrist_above_table_at_anchor: float, up_face_note: str | None = None) -> tuple[str, ...]:
    """Human-readable warnings for the package (module docstring, "Up direction"); `up_face_note` describes an up
    taken from a resting face of the object (`fastdo.robot.rest`)."""
    out = []
    if gravity_source == "default" and up_face_note is not None:
        out.append(f"up from an ASSUMED resting face of the object ({up_face_note}), not an estimate")
    elif gravity_source == "default" and pitch_deg is None:
        out.append("GRAVITY NOT ESTIMATED (qc gravity_source=default): up is the camera's -y at row 0, wrong by the "
                   "wearer's head pitch, so the table plane and every wrist orientation are tilted by it; re-export "
                   "with --camera-pitch-deg or from a reconstruction with geometry-stage gravity")
    elif gravity_source == "default":
        out.append(f"up from an ASSUMED head pitch of {pitch_deg:g} deg (given with --camera-pitch-deg or chosen by the "
                   "placement search; see the adaptations), not an estimate")
    if camera_source == "gravity_aligned_static":
        out.append("CAMERA ASSUMED STATIC (qc camera_source=gravity_aligned_static): head motion appears as world "
                   "motion of the hand and the object")
    if wrist_above_table_at_anchor < 0.0:
        out.append(f"the wrist target at the contact onset is {-wrist_above_table_at_anchor * 100:.1f} cm BELOW the "
                   "table top: the human did not take the object from a support at this row")
    return tuple(out)


def wrist_targets_world(joints_world: np.ndarray) -> np.ndarray:
    """`(T,4,4)` Dex3 `base_link` targets: MANO joint 0 position, `mano_hand_frame` rotation."""
    out = np.tile(np.eye(4), (len(joints_world), 1, 1))
    out[:, :3, :3] = mano_hand_frame(joints_world)
    out[:, :3, 3] = joints_world[:, 0]
    return out


def hand_object_distance(points_world: np.ndarray, object_world: np.ndarray, mesh_vertices: np.ndarray,
                         mesh_faces: np.ndarray, scale: float, rows: np.ndarray, stride: int = 1) -> np.ndarray:
    """`(T,)` distance (m) from the closest of `points_world[t, ::stride]` `(T,N,3)` to the object surface at
    `object_world[t]` `(T,4,4)` (rigid; mesh vertices x `scale`), on `rows` (bool `(T,)`), NaN elsewhere."""
    mesh = trimesh.Trimesh(scale * np.asarray(mesh_vertices, dtype=np.float64), np.asarray(mesh_faces), process=False)
    out = np.full(len(rows), np.nan)
    for t in np.flatnonzero(rows):
        pose = object_world[t]
        local = (np.asarray(points_world[t, ::max(stride, 1)], dtype=np.float64) - pose[:3, 3]) @ pose[:3, :3]
        _, dist, _ = trimesh.proximity.closest_point(mesh, local)
        out[t] = float(dist.min())
    return out


def contact_anchor(dist: np.ndarray, used: np.ndarray, contact_m: float) -> tuple[int, int | None]:
    """`(anchor row, contact row or None)`: the first used row with `dist < contact_m`, else the first used row."""
    rows = np.flatnonzero(np.asarray(used, dtype=bool))
    if rows.size == 0:
        raise ValueError("no used row")
    hit = rows[np.asarray(dist)[rows] < contact_m]
    if hit.size == 0:
        return int(rows[0]), None
    return int(hit[0]), int(hit[0])


def robot_frame_targets(T_rv: np.ndarray, object_start: np.ndarray, wrist_world: np.ndarray, object_world: np.ndarray,
                        anchor: int) -> tuple[np.ndarray, np.ndarray]:
    """`(wrist targets (T,4,4), reference object poses (T,4,4))` in the robot frame (stage 4 of the module docstring).

    From `anchor` on: `T_rv @ pose`. Before it: the object holds `object_start` and the wrist keeps its pose relative
    to the object, `object_start @ inv(object_world[t]) @ wrist_world[t]`. With `object_start = T_rv @
    object_world[anchor]` both rules give the same wrist at the anchor.
    """
    wrist_r = transform_poses(T_rv, wrist_world)
    obj_r = transform_poses(T_rv, object_world)
    if anchor > 0:
        rel = np.linalg.solve(object_world[:anchor], wrist_world[:anchor])
        wrist_r[:anchor] = np.einsum("ij,tjk->tik", object_start, rel)
        obj_r[:anchor] = object_start
    return wrist_r, obj_r


def _speed_spikes(positions: np.ndarray, rows: np.ndarray, times: np.ndarray, max_speed: float,
                  min_segment: int) -> np.ndarray:
    """Bool over `rows`: True for rows of short speed-consistent segments that are not reachable at `max_speed` from
    the kept rows on both sides (`spike_rows`)."""
    def ok(a: int, b: int) -> bool:
        return float(np.linalg.norm(positions[a] - positions[b])) <= max_speed * abs(float(times[a] - times[b]))

    seg = np.zeros(len(rows), dtype=np.int64)
    for i in range(1, len(rows)):
        seg[i] = seg[i - 1] + (0 if ok(int(rows[i - 1]), int(rows[i])) else 1)
    sizes = np.bincount(seg)
    keep = sizes[seg] >= min_segment
    if not keep.any():
        keep = seg == int(np.argmax(sizes))
    for i in range(len(rows)):                       # restore short-segment rows consistent with both kept sides
        if keep[i]:
            continue
        before = [j for j in range(i) if keep[j]]
        after = [j for j in range(i + 1, len(rows)) if keep[j]]
        if (not before or ok(int(rows[before[-1]]), int(rows[i]))) and (not after or ok(int(rows[i]),
                                                                                         int(rows[after[0]]))):
            keep[i] = True
    return ~keep


def spike_rows(positions: np.ndarray, valid: np.ndarray, window: int, threshold: float,
               times: np.ndarray | None = None, max_speed: float | None = None, min_segment: int = 4) -> np.ndarray:
    """`(T,)` spike rows among the valid ones (stage 1 of the module docstring).

    With `times` and `max_speed`, first the speed gate: the valid rows are cut wherever the wrist would have to move
    faster than `max_speed` between consecutive rows; rows of segments shorter than `min_segment` are spikes unless
    reachable at `max_speed` from the kept rows on both sides. Then, among the rows left, a row farther than `threshold`
    from the median of the `window` rows around it is a spike.
    """
    rows = np.flatnonzero(valid)
    out = np.zeros(len(valid), dtype=bool)
    if rows.size == 0:
        return out
    if times is not None and max_speed is not None:
        out[rows] = _speed_spikes(positions, rows, np.asarray(times, dtype=np.float64), max_speed, min_segment)
        rows = rows[~out[rows]]
    half = window // 2
    for i, r in enumerate(rows):
        nb = rows[max(0, i - half): i + half + 1]
        out[r] = float(np.linalg.norm(positions[r] - np.median(positions[nb], axis=0))) > threshold
    return out


def task_span(dist: np.ndarray, used: np.ndarray, times: np.ndarray, contact_m: float, lead_s: float, lag_s: float,
              rest_s: float, trim: bool) -> tuple[int, int, bool]:
    """`(first row, last row, rest_check)` of the task (stage 1b of the kinematic path).

    Contact rows are the used rows with `dist < contact_m`. With `trim` and at least one contact row, the task keeps the
    rows from `lead_s` before the first contact to `lag_s` after the last one, within the used rows: outside that
    span the human hand is away from the object and its motion does not act on the task. Otherwise every row is kept
    (0 .. T-1). `rest_check`: the first contact comes at least `rest_s` after the first used row, i.e. the object rests
    before it is touched.
    """
    rows = np.flatnonzero(np.asarray(used, dtype=bool))
    if rows.size == 0:
        raise ValueError("no used row")
    t = np.asarray(times, dtype=np.float64)
    hit = rows[np.asarray(dist)[rows] < contact_m]
    if hit.size == 0:
        return 0, len(t) - 1, False
    rest = bool(t[hit[0]] - t[rows[0]] >= rest_s - 1e-9)
    if not trim:
        return 0, len(t) - 1, rest
    lo = rows[t[rows] >= t[hit[0]] - lead_s - 1e-9][0]
    hi = rows[t[rows] <= t[hit[-1]] + lag_s + 1e-9][-1]
    return int(lo), int(hi), rest


def object_motion(object_world: np.ndarray, centre: np.ndarray, ref: int, rows: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """`(displacement (R,) m, rotation (R,) deg)` of the object between row `ref` and each of `rows`: the
    displacement of its hull centroid `centre` (3,) (metric object frame) and the angle of the relative rotation."""
    rows = np.asarray(rows, dtype=np.int64)
    pts = np.einsum("tij,j->ti", object_world[:, :3, :3], np.asarray(centre, dtype=np.float64)) + object_world[:, :3, 3]
    disp = np.linalg.norm(pts[rows] - pts[ref], axis=1)
    rel = np.einsum("ji,tjk->tik", object_world[ref, :3, :3], object_world[rows, :3, :3])
    cos = (np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0
    return disp, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@dataclass(frozen=True)
class StaticWindow:
    """The task rows of `static_window`: clip row of every task row, synthetic rows, and the measured object motion
    (m / deg) over the kept and the dropped rows before the contact onset and after the last contact."""
    index: np.ndarray                  # (T,) clip row of each task row; synthetic rows repeat the first / last kept
    synthetic: np.ndarray              # (T,) bool
    times: np.ndarray                  # (T,) s, synthetic rows at the clip's median row interval
    first: int                         # first / last kept clip row
    last: int
    anchor: int                        # clip row of the contact onset (the first used row without a contact)
    contact_end: int | None            # clip row of the last contact
    pre_kept: tuple[float, float]      # max motion from the contact-onset pose over the kept rows before it
    pre_dropped: tuple[float, float]   # ... over the dropped rows (0, 0: none dropped)
    post_kept: tuple[float, float]     # max motion from the last-contact pose over the kept rows after it
    post_dropped: tuple[float, float]
    tol: tuple[float, float]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"tolerance_m": self.tol[0], "tolerance_deg": self.tol[1],
                "pre_contact_kept_m": self.pre_kept[0], "pre_contact_kept_deg": self.pre_kept[1],
                "pre_contact_dropped_m": self.pre_dropped[0], "pre_contact_dropped_deg": self.pre_dropped[1],
                "post_contact_kept_m": self.post_kept[0], "post_contact_kept_deg": self.post_kept[1],
                "post_contact_dropped_m": self.post_dropped[0], "post_contact_dropped_deg": self.post_dropped[1],
                "first_row": self.first, "last_row": self.last, "anchor_row": self.anchor,
                "contact_end_row": self.contact_end, "synthetic_rows_before": self.pads[0],
                "synthetic_rows_after": self.pads[1]}

    @property
    def pads(self) -> tuple[int, int]:
        """`(synthetic rows before, after)` the kept clip rows."""
        real = np.flatnonzero(~self.synthetic)
        return int(real[0]), int(len(self.synthetic) - 1 - real[-1])


def static_window(object_world: np.ndarray, centre: np.ndarray, used: np.ndarray, dist: np.ndarray,
                  times: np.ndarray, r0: int, r1: int, contact_m: float, tol_m: float, tol_deg: float,
                  approach: bool) -> StaticWindow:
    """The task rows within clip rows `r0`-`r1` (`task_span`) on which the object reference may be held static.

    Before the contact onset the reference object is held at its contact-onset pose (`robot_frame_targets`), and with
    `approach` the human's hand path before the first contact and after the last is replaced by the robot's approach /
    retreat (`fastdo.robot.reach.approach_retreat`). Both need the reconstructed object to be static there, which is
    MEASURED, not assumed (`object_motion`: hull-centroid displacement and rotation): the rows up to the last one where
    the object is more than `tol_m` / `tol_deg` from its contact-onset pose are dropped and, with `approach`, rows after
    the first one where it is that far from its last-contact pose. With `approach` the dropped time is refilled with
    synthetic rows (copies of the first / last kept row, the object static by construction), so the approach / retreat
    keeps its duration. Each drop is a warning naming the displacement.
    """
    t = np.asarray(times, dtype=np.float64)
    used = np.asarray(used, dtype=bool)
    rows = np.flatnonzero(used)
    rows = rows[(rows >= r0) & (rows <= r1)]
    if rows.size == 0:
        raise ValueError("no used row in the task span")
    hits = rows[np.asarray(dist)[rows] < contact_m]
    anchor = int(hits[0]) if hits.size else int(rows[0])
    c1 = int(hits[-1]) if hits.size else None
    first, last = r0, r1
    pre_kept = pre_drop = post_kept = post_drop = (0.0, 0.0)
    notes = []
    pre = rows[rows < anchor]
    if pre.size:
        d, a = object_motion(object_world, centre, anchor, pre)
        bad = (d > tol_m) | (a > tol_deg)
        if bad.any():
            last_bad = int(pre[bad][-1])
            first = int(rows[rows > last_bad][0])
            gone = pre <= last_bad
            pre_drop = (float(d[gone].max()), float(a[gone].max()))
        keep = pre >= first
        if keep.any():
            pre_kept = (float(d[keep].max()), float(a[keep].max()))
    if approach and c1 is not None:
        post = rows[rows > c1]
        if post.size:
            d, a = object_motion(object_world, centre, c1, post)
            bad = (d > tol_m) | (a > tol_deg)
            if bad.any():
                first_bad = int(post[bad][0])
                last = int(rows[rows < first_bad][-1])
                gone = post >= first_bad
                post_drop = (float(d[gone].max()), float(a[gone].max()))
            keep = post <= last
            if keep.any():
                post_kept = (float(d[keep].max()), float(a[keep].max()))
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    pad_before = round((t[first] - t[r0]) / dt) if approach and hits.size and first > r0 else 0
    pad_after = round((t[r1] - t[last]) / dt) if approach and hits.size and last < r1 else 0
    if pre_drop != (0.0, 0.0):
        notes.append(f"the reconstructed object moves up to {pre_drop[0] * 100:.1f} cm / {pre_drop[1]:.0f} deg from its "
                     f"contact-onset pose (clip row {anchor}) on clip rows {r0}-{first - 1}, before the right hand "
                     f"touches it (tolerance {tol_m * 100:g} cm / {tol_deg:g} deg): those rows are dropped and the "
                     f"task starts at row {first}" + (f" after {pad_before} synthetic approach rows" if pad_before
                                                       else "") + "; the robot does not reproduce that motion")
    if post_drop != (0.0, 0.0):
        notes.append(f"after the last contact (clip row {c1}) the reconstructed object moves up to "
                     f"{post_drop[0] * 100:.1f} cm / {post_drop[1]:.0f} deg on clip rows {last + 1}-{r1} while the "
                     f"right hand is away from it: those rows are dropped and the task ends at row {last}"
                     + (f", followed by {pad_after} synthetic retreat rows" if pad_after else ""))
    index = np.concatenate([np.full(pad_before, first), np.arange(first, last + 1), np.full(pad_after, last)])
    synthetic = np.concatenate([np.ones(pad_before, bool), np.zeros(last - first + 1, bool), np.ones(pad_after, bool)])
    times_out = np.concatenate([t[first] - dt * np.arange(pad_before, 0, -1), t[first:last + 1],
                                t[last] + dt * np.arange(1, pad_after + 1)])
    return StaticWindow(index=index.astype(np.int64), synthetic=synthetic, times=times_out, first=first, last=last,
                        anchor=anchor, contact_end=c1, pre_kept=pre_kept, pre_dropped=pre_drop, post_kept=post_kept,
                        post_dropped=post_drop, tol=(tol_m, tol_deg), warnings=tuple(notes))
