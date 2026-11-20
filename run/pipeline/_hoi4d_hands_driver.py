"""HOI4D GT right-hand MANO of one sequence -> HaWoR-format `all_hand_meshes.npz` (runs in the hawor env, CPU).

`run/pipeline/build_hoi4d_hands.py` starts it as a subprocess, one per sequence. The DexYCB driver
(`_dexycb_hands_driver.py`) is the model: HaWoR's MANO wrapper and geometry helpers are imported from `--hawor-root`
without running its package `__init__`, and the output has the same keys, so `fastdo.policy.dexycb_inputs`-style
readers and `fastdo.io.datasets.hoi4d.hand_track` read it unchanged.

Source: `Hand_pose/handpose_right_hand/<seq>/<t>.pickle` -- `poseCoeff` (48: global orient + 15 finger joints,
axis-angle), `beta` (10), `trans` (camera frame, m), `kps2D` (21, 2). The official loader builds manopth's layer with
`use_pca=False, flat_hand_mean=True`, so the 45 finger values ARE the full finger pose: HaWoR's layer (pose2rot=False,
no mean added -- the DexYCB driver adds the mean itself) takes them as they are. A frame without a pickle is invalid
and takes the parameters of the nearest valid frame, as in the DexYCB driver.

Validation (all recorded; `pass` needs both gates):
- kps2D: the FK joints projected with the SHIPPED intrinsics against the pickle's own `kps2D` (manopth's joint order is
  the OpenPose order HaWoR returns): mean px over valid frames <= `--kps-tol-px`. A known answer for the MANO
  convention (hand mean, joint order, units, trans): the kps2D were projected by the dataset from the same parameters.
- hand label: over 5 evenly spaced valid frames, the fraction of in-image FK vertices (SHIPPED intrinsics: the hands,
  unlike the object poses, are registered to the RGB -- fastdo.io.datasets.hoi4d) inside the right-hand label
  (0,128,0) of the 2D mask, >= `--label-frac-min`. The same fraction under the object's registration shift
  (`--shift-dx/--shift-dy`) is recorded as a diagnostic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("hoi4d_hands_driver")

FACES_NEW = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279], [122, 118, 279],
                      [279, 118, 215], [118, 117, 215], [215, 117, 214], [117, 119, 214], [214, 119, 121],
                      [119, 120, 121], [121, 120, 78], [120, 108, 78], [78, 108, 79]])  # hawor/utils/process.py L70-83
KEY_ORDER = ["left_vertices", "left_faces", "left_joints", "right_vertices", "right_faces", "right_joints",
             "left_trans", "left_rot", "left_hand_pose", "left_betas", "left_valid",
             "right_trans", "right_rot", "right_hand_pose", "right_betas", "right_valid"]
HAND_COLOUR = (0, 128, 0)


def _load_module(name: str, path: Path) -> Any:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mask_path(root: Path, seq: str, t: int) -> Path:
    seg = root / "HOI4D_annotations" / seq / "2Dseg"
    sub = "shift_mask" if (seg / "shift_mask").is_dir() else "mask"
    return seg / sub / f"{t:05d}.png"


def convert(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root, seq = Path(args.hoi4d_root), args.seq
    hawor = Path(args.hawor_root)
    wrapper = _load_module("hawor_mano_wrapper", hawor / "lib" / "models" / "mano_wrapper.py")
    geometry = _load_module("hawor_geometry", hawor / "hawor" / "utils" / "geometry.py")
    pkl = Path(args.mano_right_pkl)
    mano = wrapper.MANO(data_dir=str(pkl.parent.parent) + "/", model_path=str(pkl.parent), gender="neutral",
                        num_hand_joints=15, create_body_pose=False)
    n = int(args.n_frames)
    coeff = np.zeros((n, 48), np.float32)
    beta = np.zeros((n, 10), np.float32)
    trans = np.zeros((n, 3), np.float32)
    kps = np.full((n, 21, 2), np.nan, np.float32)
    valid = np.zeros(n, bool)
    for t in range(n):
        p = root / "Hand_pose" / "handpose_right_hand" / seq / f"{t}.pickle"
        if not p.is_file():
            continue
        with p.open("rb") as fh:   # HOI4D's own release format (no JSON alternative); a local, integrity-checked download
            d = pickle.load(fh, encoding="latin1")
        coeff[t], beta[t], trans[t] = d["poseCoeff"], d["beta"], np.asarray(d["trans"]).reshape(3)
        kps[t] = np.asarray(d["kps2D"], np.float32).reshape(21, 2)
        valid[t] = bool(np.all(np.isfinite(coeff[t])) and np.all(np.isfinite(trans[t])))
    if valid.sum() < 2:
        raise SystemExit(f"{seq}: only {int(valid.sum())} frames with a GT right hand")
    rows = np.nonzero(valid)[0]
    src = rows[np.abs(np.arange(n)[:, None] - rows[None, :]).argmin(1)]
    rot, hand_pose, tr, betas = coeff[src, :3], coeff[src, 3:], trans[src], beta[src]
    with torch.no_grad():
        go = geometry.aa_to_rotmat(torch.from_numpy(np.ascontiguousarray(rot))).view(n, 1, 3, 3)
        hp = geometry.aa_to_rotmat(torch.from_numpy(np.ascontiguousarray(hand_pose)).reshape(n * 15, 3))
        out = mano(global_orient=go.float(), hand_pose=hp.view(n, 15, 3, 3).float(),
                   betas=torch.from_numpy(betas).float(), transl=torch.from_numpy(tr).float(), pose2rot=False)
    verts, joints = out.vertices.numpy().astype(np.float32), out.joints.numpy().astype(np.float32)
    faces = np.concatenate([np.asarray(mano.faces), FACES_NEW], axis=0).astype(np.int32)
    z = lambda shape: np.zeros(shape, np.float32)
    npz = {"right_vertices": verts, "right_joints": joints, "left_vertices": np.zeros_like(verts),
           "left_joints": np.zeros_like(joints), "left_faces": faces[:, [0, 2, 1]].astype(np.int32),
           "right_faces": faces, "right_trans": tr, "right_rot": rot, "right_hand_pose": hand_pose,
           "right_betas": betas, "right_valid": valid, "left_trans": z((n, 3)), "left_rot": z((n, 3)),
           "left_hand_pose": z((n, 45)), "left_betas": z((n, 10)), "left_valid": np.zeros(n, bool)}
    info = {"seq": seq, "num_frames": n, "hand_side": "right", "n_valid": int(valid.sum()),
            "threads": int(torch.get_num_threads()), "mano_model": str(pkl),
            "beta_std_max": float(beta[valid].std(0).max()),
            "filled_from_nearest_valid": [int(i) for i in np.nonzero(~valid)[0]]}
    return {k: npz[k] for k in KEY_ORDER}, {"info": info, "kps": kps}


def validate(args: argparse.Namespace, npz: dict[str, np.ndarray], kps: np.ndarray) -> dict[str, Any]:
    root = Path(args.hoi4d_root)
    K = np.load(root / "camera_params" / args.seq.split("/")[0] / "intrin.npy").astype(np.float64)
    Kr = K.copy()
    Kr[0, 2] += args.shift_dx
    Kr[1, 2] += args.shift_dy
    valid = npz["right_valid"]

    def proj(x: np.ndarray, k: np.ndarray) -> np.ndarray:
        return (x @ k.T)[..., :2] / x[..., 2:3]

    j = npz["right_joints"].astype(np.float64)
    ok = valid & np.all(np.isfinite(kps), axis=(1, 2))
    err = np.linalg.norm(proj(j[ok], K) - kps[ok], axis=-1)
    err_reg = np.linalg.norm(proj(j[ok], Kr) - kps[ok], axis=-1)
    frames = [int(t) for t in np.nonzero(valid)[0][np.linspace(0, valid.sum() - 1, 5).round().astype(int)]]
    labs = {}
    for t in frames:
        with Image.open(mask_path(root, args.seq, t)) as im:
            labs[t] = np.asarray(im.convert("RGB"))

    def label_fraction(k: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
        tot_in = tot_hit = 0
        per = []
        for t in frames:
            lab = labs[t]
            uv = np.rint(proj(npz["right_vertices"][t].astype(np.float64), k)).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < lab.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < lab.shape[0])
            hit = np.all(lab[uv[inside, 1], uv[inside, 0]] == np.asarray(HAND_COLOUR, np.uint8), axis=-1)
            per.append({"frame": t, "n_inside": int(inside.sum()),
                        "frac_in_hand_label": float(hit.mean()) if inside.any() else None})
            tot_in, tot_hit = tot_in + int(inside.sum()), tot_hit + int(hit.sum())
        return tot_hit / max(tot_in, 1), per

    frac, per = label_fraction(K)
    frac_shifted, _ = label_fraction(Kr)
    res = {"kps2d_px_mean": float(err.mean()), "kps2d_px_p95": float(np.percentile(err, 95)),
           "kps2d_px_mean_registered_K": float(err_reg.mean()), "n_kps_frames": int(ok.sum()),
           "label_frames": per, "frac_all": frac, "frac_all_object_shift_K": frac_shifted, "kps_tol_px": args.kps_tol_px, "label_frac_min": args.label_frac_min,
           "pass_kps": bool(err.mean() <= args.kps_tol_px), "pass_label": bool(frac >= args.label_frac_min)}
    res["pass"] = res["pass_kps"] and res["pass_label"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq")
    ap.add_argument("out_npz", type=Path)
    for name in ("--hoi4d-root", "--hawor-root", "--mano-right-pkl"):
        ap.add_argument(name, required=True)
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--shift-dx", type=float, default=0.0)
    ap.add_argument("--shift-dy", type=float, default=0.0)
    ap.add_argument("--kps-tol-px", type=float, default=2.0)
    ap.add_argument("--label-frac-min", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[hoi4d_hands] %(message)s")
    torch.set_num_threads(args.threads)
    npz, aux = convert(args)
    report: dict[str, Any] = {"conversion": aux["info"], "validation": validate(args, npz, aux["kps"]),
                              "registration_px": [args.shift_dx, args.shift_dy]}
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **npz)                               # type: ignore[arg-type]
    report["out"] = {k: [list(v.shape), str(v.dtype)] for k, v in npz.items()}
    args.out_npz.with_name(args.out_npz.stem + "_validation.json").write_text(json.dumps(report, indent=1))
    val = report["validation"]
    logger.info("%s: valid=%d kps2D px mean=%.3f | in-hand-label frac=%.3f -> pass=%s", args.seq,
                aux["info"]["n_valid"], val["kps2d_px_mean"], val["frac_all"], val["pass"])
    return 0 if val["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
