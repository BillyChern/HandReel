"""DexYCB GT MANO of one sequence -> HaWoR's `all_hand_meshes.npz`, right or left hand (runs in the hawor env).

`run/pipeline/build_dexycb_hands.py` starts it as a CPU subprocess. It ports an earlier GT-hands converter (not
shipped), which handled left hands too and was validated on 38 sequences, 15 of them left-handed. The one change:
every path comes from the command line (from config), none is hard-coded. HaWoR's MANO wrapper and geometry helpers
are imported from `--hawor-root` at runtime without running its package `__init__`.

Models. The right hand is HaWoR's MANO layer on `--mano-right-pkl`. The left hand uses `--mano-left-pkl` with
`is_rhand=False`, as HaWoR's `run_mano_left` does. The PCA basis comes from `--mano-pca`: `hands_components` and
`hands_mean`, or their `_left` versions.
- HaWoR's `run_mano_left` flips `shapedirs[:, 0, :]` by default; that flip is NOT applied. DexYCB's GT fit
  (dex-ycb-toolkit, manopth on the official pickle) uses the pickle's shapedirs as they are.
- On one left-hand DexYCB development sequence, the joint error without the flip is 0.686 mm; with it, 4.05 mm.
  The flipped error is recorded as `shapedirs_fix_diagnostic`.

Format (HaWoR demo.py writer):
- `{left,right}_vertices` (T,778,3) float32, camera frame, metres;
- `{left,right}_faces` (1552,3) int32: right = MANO faces plus HaWoR's 14 wrist faces, left = right[:, [0,2,1]];
- `{left,right}_joints` (T,21,3) float32, OpenPose order;
- `{left,right}_trans` (T,3), `_rot` (T,3), `_hand_pose` (T,45), `_betas` (T,10) float32, and `_valid` (T,) bool.
The GT side's arrays are filled; the absent side's are zeros with valid False. Rows without a GT hand take the
parameters of the nearest valid row.

Validation, as in the regtracks pre-registration. It uses 5 frames: GT hand valid, >= 50 % of the vertices in the
image, evenly spaced.
- Joint error: the mean over those frames' 21 joints against DexYCB `joint_3d`, <= `--joint-tol-mm`.
- Label fraction: over all in-image vertices of those frames, the fraction inside the hand label (255) of `seg`,
  >= `--label-frac-min`.
Writes the npz and `<stem>_validation.json` next to it. Exit code: 0 when both pass, 1 when validation fails.

Threads: `--threads` (the builder's `policy_hands.threads`, 2) sets torch's thread count, and the builder sets OpenMP,
MKL and OpenBLAS to the same count; float32 MANO output depends on it. It is recorded as `conversion.threads`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger("dexycb_hands_driver")

FACES_NEW = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279], [122, 118, 279],
                      [279, 118, 215], [118, 117, 215], [215, 117, 214], [117, 119, 214], [214, 119, 121],
                      [119, 120, 121], [121, 120, 78], [120, 108, 78], [78, 108, 79]])  # hawor/utils/process.py L70-83
KEY_ORDER = ["left_vertices", "left_faces", "left_joints", "right_vertices", "right_faces", "right_joints",
             "left_trans", "left_rot", "left_hand_pose", "left_betas", "left_valid",
             "right_trans", "right_rot", "right_hand_pose", "right_betas", "right_valid"]


class _TupleSafeLoader(yaml.SafeLoader):
    """SafeLoader plus DexYCB's !!python/tuple tag only."""


_TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                                 lambda loader, node: tuple(loader.construct_sequence(node)))


def load_yaml(path: Path) -> Any:
    with path.open() as f:
        return yaml.load(f, Loader=_TupleSafeLoader)


def _load_module(name: str, path: Path) -> Any:
    """Import one HaWoR file without executing its package __init__; no bytecode lands in third_party."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Converter:
    """The regtracks conversion and validation with configured paths (module docstring)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dexycb = Path(args.dexycb_root)
        hawor = Path(args.hawor_root)
        self.wrapper = _load_module("hawor_mano_wrapper", hawor / "lib" / "models" / "mano_wrapper.py")
        self.geometry = _load_module("hawor_geometry", hawor / "hawor" / "utils" / "geometry.py")

    def mano(self, side: str, fix_shapedirs: bool = False) -> Any:
        pkl = Path(self.args.mano_right_pkl if side == "right" else self.args.mano_left_pkl)
        if not pkl.is_file():
            raise SystemExit(f"MANO {side} model not found at {pkl} (retarget_stage.mano_pkl_path.{side})")
        cfg: dict[str, Any] = {"data_dir": str(pkl.parent.parent) + "/", "model_path": str(pkl.parent),
                               "gender": "neutral", "num_hand_joints": 15, "create_body_pose": False}
        if side == "left":
            cfg["is_rhand"] = False
        model = self.wrapper.MANO(**cfg)
        if fix_shapedirs:
            model.shapedirs[:, 0, :] *= -1
        return model

    @torch.no_grad()
    def run_mano(self, mano: Any, trans: np.ndarray, rot: np.ndarray, hand_pose: np.ndarray,
                 betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """HaWoR's run_mano(use_cuda=False) for B = 1: quaternion aa_to_rotmat, pose2rot False."""
        t = len(rot)
        go = self.geometry.aa_to_rotmat(torch.from_numpy(np.ascontiguousarray(rot))).view(t, 1, 3, 3)
        hp = self.geometry.aa_to_rotmat(torch.from_numpy(np.ascontiguousarray(hand_pose)).reshape(t * 15, 3))
        out = mano(global_orient=go.float(), hand_pose=hp.view(t, 15, 3, 3).float(), betas=torch.from_numpy(betas).float(),
                   transl=torch.from_numpy(trans).float(), pose2rot=False)
        return out.vertices.numpy().astype(np.float32), out.joints.numpy().astype(np.float32)

    def intrinsics(self, serial: str) -> np.ndarray:
        path = min((self.dexycb / "calibration" / "intrinsics").glob(f"{serial}_*.yml"))
        c = load_yaml(path)["color"]
        return np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], dtype=np.float64)

    def convert(self, seq: str, serial: str) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
        subject, sequence = seq.strip("/").split("/")[:2]
        seq_dir = self.dexycb / subject / sequence
        meta = load_yaml(seq_dir / "meta.yml")
        side = str(next(iter(meta["mano_sides"])))
        if side not in ("right", "left"):
            raise SystemExit(f"{seq}: mano_sides[0]={side!r}")
        mano_right = self.mano("right")                     # its faces are HaWoR's faces for both sides
        mano = mano_right if side == "right" else self.mano("left")
        betas = np.asarray(load_yaml(self.dexycb / "calibration" / f"mano_{meta['mano_calib'][0]}" / "mano.yml")["betas"],
                           dtype=np.float32)
        suffix = "" if side == "right" else "_left"
        with np.load(self.args.mano_pca) as p:
            comps, mean = p[f"hands_components{suffix}"].astype(np.float32), p[f"hands_mean{suffix}"].astype(np.float32)
        pca_check = {"pca_keys": [f"hands_components{suffix}", f"hands_mean{suffix}"],
                     "hands_mean_equals_mano_layer": bool(np.allclose(mean, mano.hand_mean.numpy(), atol=1e-6)),
                     "first6_components_equal_mano_layer": bool(np.allclose(comps[:6], mano.hand_components.numpy(),
                                                                             atol=1e-6))}
        n = int(meta["num_frames"])
        pose_m = np.zeros((n, 51), np.float32)
        joint_3d = np.full((n, 21, 3), -1.0, np.float32)
        for t in range(n):
            with np.load(seq_dir / serial / f"labels_{t:06d}.npz") as lab:
                pose_m[t], joint_3d[t] = lab["pose_m"][0], lab["joint_3d"][0]
        valid = np.any(pose_m != 0, axis=1)
        if valid.sum() < 2:
            raise SystemExit(f"{seq}: only {int(valid.sum())} frames with a GT hand")
        rows = np.nonzero(valid)[0]
        src = rows[np.abs(np.arange(n)[:, None] - rows[None, :]).argmin(1)]
        rot, trans = pose_m[src, 0:3], pose_m[src, 48:51]
        hand_pose = (pose_m[src, 3:48] @ comps + mean).astype(np.float32)
        betas_t = np.tile(betas, (n, 1))
        verts, joints = self.run_mano(mano, trans, rot, hand_pose, betas_t)
        faces = np.concatenate([np.asarray(mano_right.faces), FACES_NEW], axis=0).astype(np.int32)
        gj = valid & np.all(joint_3d != -1, axis=(1, 2))
        jerr = np.linalg.norm(joints[gj] - joint_3d[gj], axis=-1) * 1000
        other = "left" if side == "right" else "right"
        out = {f"{side}_vertices": verts, f"{side}_joints": joints, f"{other}_vertices": np.zeros_like(verts),
               f"{other}_joints": np.zeros_like(joints), "left_faces": faces[:, [0, 2, 1]].astype(np.int32),
               "right_faces": faces, f"{side}_trans": trans.astype(np.float32), f"{side}_rot": rot.astype(np.float32),
               f"{side}_hand_pose": hand_pose, f"{side}_betas": betas_t.astype(np.float32), f"{side}_valid": valid,
               f"{other}_trans": np.zeros((n, 3), np.float32), f"{other}_rot": np.zeros((n, 3), np.float32),
               f"{other}_hand_pose": np.zeros((n, 45), np.float32), f"{other}_betas": np.zeros((n, 10), np.float32),
               f"{other}_valid": np.zeros(n, bool)}
        info: dict[str, Any] = {
            "seq": f"{subject}/{sequence}", "serial": serial, "num_frames": n, "hand_side": side,
            "threads": int(torch.get_num_threads()),
            "mano_model": str(self.args.mano_right_pkl if side == "right" else self.args.mano_left_pkl),
            "shapedirs_fix_applied": False, "n_valid": int(valid.sum()), "first_valid": int(rows[0]),
            "filled_from_nearest_valid": [int(i) for i in np.nonzero(~valid)[0]], "pca_basis_check": pca_check,
            "joint3d_err_mm_all_frames": {"n_frames": int(gj.sum()), "mean": float(jerr.mean()), "max": float(jerr.max())}}
        if side == "left":
            info["left_pkl_faces_same_triangles_as_right"] = (
                {tuple(sorted(map(int, f))) for f in np.asarray(mano.faces)}
                == {tuple(sorted(map(int, f))) for f in np.asarray(mano_right.faces)})
        aux = {"joint_3d": joint_3d, "gj": gj, "side": side, "params": (trans, rot, hand_pose, betas_t)}
        return {k: out[k] for k in KEY_ORDER}, info, aux

    def validate(self, npz: dict[str, np.ndarray], aux: dict[str, Any], seq: str, serial: str) -> dict[str, Any]:
        subject, sequence = seq.strip("/").split("/")[:2]
        side, K = aux["side"], self.intrinsics(serial)

        def project(t: int, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            v = npz[f"{side}_vertices"][t].astype(np.float64)
            u = np.rint(K[0, 0] * v[:, 0] / v[:, 2] + K[0, 2]).astype(int)
            w = np.rint(K[1, 1] * v[:, 1] / v[:, 2] + K[1, 2]).astype(int)
            return u, w, (v[:, 2] > 0) & (u >= 0) & (u < shape[1]) & (w >= 0) & (w < shape[0])

        visible = [int(t) for t in np.nonzero(npz[f"{side}_valid"])[0] if project(int(t), (480, 640))[2].mean() >= 0.5]
        n = int(self.args.validate_frames)
        frames = sorted({visible[i] for i in np.linspace(0, len(visible) - 1, n).round().astype(int)})
        rows, tot_in, tot_hit, jerrs = [], 0, 0, []
        for t in frames:
            with np.load(self.dexycb / subject / sequence / serial / f"labels_{t:06d}.npz") as lab:
                seg, joint_2d = lab["seg"], lab["joint_2d"][0].astype(np.float64)
            u, w, inside = project(t, seg.shape)
            hit = seg[w[inside], u[inside]] == 255
            row: dict[str, Any] = {"frame": t, "n_inside": int(inside.sum()), "frac_in_hand_label": float(hit.mean())}
            jc = npz[f"{side}_joints"][t].astype(np.float64)
            row["joint_2d_reproj_px_mean"] = float(np.linalg.norm((jc @ K.T)[:, :2] / jc[:, 2:3] - joint_2d, axis=1).mean())
            if aux["gj"][t]:
                e = np.linalg.norm(npz[f"{side}_joints"][t] - aux["joint_3d"][t], axis=-1) * 1000
                row.update(joint_err_mm_mean=float(e.mean()), joint_err_mm_max=float(e.max()))
                jerrs.append(e)
            rows.append(row)
            tot_in, tot_hit = tot_in + int(inside.sum()), tot_hit + int(hit.sum())
        frac = tot_hit / max(tot_in, 1)
        jmean = float(np.concatenate(jerrs).mean()) if jerrs else float("inf")
        res: dict[str, Any] = {
            "frames": frames, "per_frame": rows, "frac_all": frac, "joint_err_mm_mean_5f": jmean,
            "joint_err_mm_max_5f": float(np.concatenate(jerrs).max()) if jerrs else None, "n_joint_frames": len(jerrs),
            "joint_tol_mm": self.args.joint_tol_mm, "label_frac_min": self.args.label_frac_min,
            "pass_joint": bool(jmean <= self.args.joint_tol_mm), "pass_label": bool(frac >= self.args.label_frac_min)}
        res["pass"] = res["pass_joint"] and res["pass_label"]
        if side == "left":                                   # diagnostic: HaWoR's shapedirs flip on the same frames
            trans, rot, hand_pose, betas_t = aux["params"]
            _, jfix = self.run_mano(self.mano("left", fix_shapedirs=True), trans, rot, hand_pose, betas_t)
            e2 = [np.linalg.norm(jfix[t] - aux["joint_3d"][t], axis=-1) * 1000 for t in frames if aux["gj"][t]]
            res["shapedirs_fix_diagnostic"] = {"joint_err_mm_mean_5f": float(np.concatenate(e2).mean()) if e2 else None}
        return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq", help="subject/sequence, e.g. 20200820-subject-03/20200820_135250")
    ap.add_argument("out_npz", type=Path)
    for name in ("--serial", "--dexycb-root", "--mano-pca", "--hawor-root", "--mano-right-pkl", "--mano-left-pkl"):
        ap.add_argument(name, required=True)
    ap.add_argument("--validate-frames", type=int, default=5)
    ap.add_argument("--joint-tol-mm", type=float, default=2.0)
    ap.add_argument("--label-frac-min", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=2, help="torch threads (policy_hands.threads)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[dexycb_hands] %(message)s")
    torch.set_num_threads(args.threads)
    converter = Converter(args)
    npz, info, aux = converter.convert(args.seq, args.serial)
    report: dict[str, Any] = {"conversion": info, "validation": converter.validate(npz, aux, args.seq, args.serial)}
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **npz)                               # type: ignore[arg-type]
    report["out"] = {k: [list(v.shape), str(v.dtype)] for k, v in npz.items()}
    args.out_npz.with_name(args.out_npz.stem + "_validation.json").write_text(json.dumps(report, indent=1))
    val = report["validation"]
    logger.info("%s (%s): T=%d valid=%d joint_err_mm 5f mean=%.3f | in-hand-label frac=%.3f -> pass=%s %s", args.seq,
                info["hand_side"], info["num_frames"], info["n_valid"], val["joint_err_mm_mean_5f"], val["frac_all"],
                val["pass"], args.out_npz)
    return 0 if val["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
