# Copyright 2026 The alongray authors
# SPDX-License-Identifier: Apache-2.0
"""Mutant check: every non-equivalent mutant of the split, the reprojection, the CLI, the shift reading, the
alignments, the scene gauge, the pooling and the change estimand must fail the own tests.

Usage: ``python tools/mutate_split.py`` from anywhere (needs pytest). Optionally, ``ALONGRAY_EXTRA_TESTS=<file>``
names one more test file, outside the package, to run against each mutant, with ``ALONGRAY_EXTRA_PYTHONPATH`` added
to its import path (for example a bit-exact parity suite against another implementation of the same arithmetic).
Each mutant runs in a temporary copy of the package; the source tree is never modified. An unmutated BASELINE must
pass first. Mutants named ``EQUIV_*`` are equivalent (they change only float rounding, or nothing at all) and are
expected to survive the own tests; only a bit-exact extra suite can catch the rounding ones. Exit 0 iff the baseline passes and every non-equivalent mutant is killed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

# name: (file under src/alongray, old text, new text); each old text must occur exactly once
MUTANTS: dict[str, tuple[str, str, str]] = {
    "BASELINE_unmutated": ("core.py", "", ""),
    "lam_denominator_pred": ("core.py", "lam = float(np.sum(p * g) / gg)",
                             "lam = float(np.sum(p * g) / np.sum(p * p))"),
    "gauge_part_drops_minus_one": ("core.py", "gauge_part=abs(dec.lam - 1.0) * z_rms",
                                   "gauge_part=abs(dec.lam) * z_rms"),
    "off_orbit_uses_lam_1": ("core.py", 'o = np.einsum("ij,ij->i", p - dec.lam * g', 'o = np.einsum("ij,ij->i", p - g'),
    "off_orbit_mean_abs": ("core.py", "off = float(np.sqrt(np.mean(o ** 2)))", "off = float(np.mean(np.abs(o)))"),
    "z_rms_is_mean": ("core.py", "z_rms = float(np.sqrt(np.mean(z ** 2)))", "z_rms = float(np.mean(z))"),
    "along_on_pred_ray": ("core.py", 'along = np.einsum("ij,ij->i", d, u)',
                          'along = np.einsum("ij,ij->i", d, p / np.linalg.norm(p, axis=1)[:, None])'),
    "along_sign_flipped": ("core.py", "d = p - g\n", "d = g - p\n"),
    "lam_unfiltered_nan": ("core.py", "    p, g = p[ok], g[ok]\n    gg = ", "    gg = "),
    "lam_check_removed": ("core.py", "if not (np.isfinite(lam) and lam > 0.0):", "if False:"),
    "kappa_check_removed": ("core.py", "if not (np.isfinite(kappa) and kappa > 0.0):", "if False:"),
    "e_par_is_mean_abs": ("core.py", "e_par = float(np.sqrt(np.mean(ar.signed_along ** 2)))",
                          "e_par = float(np.mean(np.abs(ar.signed_along)))"),
    "oracle_floor_times_lam": ("core.py", "oracle_floor=off / dec.lam", "oracle_floor=off * dec.lam"),
    "kappa_ignored": ("core.py", '_traj3(c_pred, "c_pred") * float(kappa)', '_traj3(c_pred, "c_pred") * 1.0'),
    "o_projects_on_pred_ray": ("core.py", "p - dec.lam * g, g / z[:, None])",
                               "p - dec.lam * g, p / np.linalg.norm(p, axis=1)[:, None])"),
    "mm_warning_removed": ("core.py", "if z_rms > _Z_RMS_WARN:", "if False:"),
    "fx_fy_swapped": ("score.py", "fx, fy, cx, cy = a\n", "fy, fx, cx, cy = a\n"),
    "hand_root_ignored": ("score.py", 'root=hand_root, kind="hand"', 'root=0, kind="hand"'),
    "pose_bottom_row_unchecked": ("score.py", "if not np.allclose(a[fin, 3, :]",
                                  "if False and not np.allclose(a[fin, 3, :]"),
    "object_accepts_joints": ("score.py", 'and kind != "object":', ":"),
    "hand_shape_check_removed": ("score.py", "if (_is_joints(pred) or _is_joints(gt)) and", "if False and"),
    "joint_pair_check_removed": ("aligned.py", "if p.ndim != 3 or p.shape[2] != 3 or p.shape != g.shape:", "if False:"),
    "zero_fill_warning_removed": ("score.py", "if np.any(np.all(p_all == 0.0, axis=1)):", "if False:"),
    "cli_hand_root_dropped": ("cli.py", "hand_root=args.hand_root", "hand_root=0"),
    "cli_exit_1_on_error": ("cli.py", "        return 2\n", "        return 1\n"),
    "shift_part_is_rms": ("core.py", "return abs(float(np.mean(a))), both,", "return float(np.sqrt(np.mean(a * a))), both,"),
    "collinearity_inverted": ("core.py", "float(np.mean(z) ** 2 / np.mean(z * z))", "float(np.mean(z * z) / np.mean(z) ** 2)"),
    "low_n_never_set": ("core.py", "low_n=dec.n_frames < MIN_ROWS)", "low_n=False)"),
    "low_n_warning_removed": ("core.py", "if dec.n_frames < MIN_ROWS:", "if False:"),
    "sens_scale_direction": ("sensitivity.py", "kappa = float(np.sum(fp * fg) / np.sum(fp * fp))",
                             "kappa = float(np.sum(fp * fg) / np.sum(fg * fg))"),
    "sens_mask_shape_unchecked": ("sensitivity.py", "if v.shape != (p.shape[0],):", "if False:"),
    "scene_residual_undivided": ("scene.py", "res[n] = off[n] / shared", "res[n] = off[n]"),
    "pool_share_unsquared": ("pool.py", '"gauge": sum(s.gauge_part ** 2 for s in splits) / e2',
                             '"gauge": sum(s.gauge_part for s in splits) ** 2 / e2'),
    "pool_draw_short": ("pool.py", "pick = gen.integers(0, len(labels), len(labels))",
                        "pick = gen.integers(0, len(labels), len(labels) - 1)"),
    "pool_percentile_level": ("pool.py", "lo, hi = (100 - 100 * level) / 2,", "lo, hi = (100 - 100 * level) / 3,"),
    "pool_cli_boot_unchecked": ("pool.py", "if a.boot > 0 and not a.cluster_key:", "if False:"),
    "cli_one_sided_frame_idx": ("cli.py", 'if ("frame_idx" in pred) != ("frame_idx" in gt):', "if False:"),
    "cli_pred_mask_ignored": ("cli.py", "keep = vp[ip] & vg[ig]", "keep = vg[ig]"),
    "cli_sign_change_unchecked": ("cli.py", "if minority.size and float(np.max(np.abs(minority))) > tol:", "if False:"),
    "cli_sign_tolerance_zero": ("cli.py", "tol = max(0.05, 0.1 * float(np.median(np.abs(z))))", "tol = 0.0"),
    "cli_mirror_check_removed": ("cli.py", "if far_opposite.mean() >= 0.9:", "if False:"),
    "cli_mirror_threshold_half": ("cli.py", "if far_opposite.mean() >= 0.9:", "if far_opposite.mean() >= 0.5:"),
    "cli_wrong_side_warning_removed": ("cli.py", "        if n_opp:\n", "        if False:\n"),
    "cli_image_check_pooled": ("cli.py", "front = gr[np.sign(gr[:, 2]) == sign]",
                               "front = g[np.sign(g[:, 2]) == sign]"),
    "cli_image_size_ignored": ("cli.py", "_image_size(pred, gt))", "None)"),
    "cli_image_check_removed": ("cli.py", "if inside.mean() < 0.5:", "if False:"),
    "cli_wide_cone_warning_removed": ("cli.py", "if ratio > 3.0:", "if False:"),
    "cli_01_passthrough_removed": ("cli.py",
                                   'if not any(k in d for k in ("frame_ids", "frame_idx", "valid") for d in (pred, gt)):',
                                   "if False:"),
    "cli_float_frame_idx_accepted": ("cli.py", "if f.ndim != 1 or not np.issubdtype(f.dtype, np.integer):",
                                     "if f.ndim != 1:"),
    "cli_mask_values_unchecked": ("cli.py", "if v.dtype != np.bool_ and not", "if False and not"),
    "cli_scene_without_hand_ok": ("cli.py", "if args.scene and ph is None:", "if False:"),
    "pool_gram_b2_uses_scale_only": ("pool.py", "b2 = b0 * rbar + b1 * rz\n", "b2 = b1 * rz\n"),
    # regression mutants: the ported 0.3.0 fixes (A1-A5, intrinsics)
    "pool_cauchy_schwarz_unchecked": ("pool.py", "if pz * pz > pp * zz * (1 + MOMENT_RTOL):", "if False:"),
    "pool_split_identity_unchecked": ("pool.py", "if abs(e * e - g * g - o * o) > rtol", "if False and abs(e * e - g * g - o * o) > rtol"),
    "pool_missing_uncounted": ("pool.py", "missing = sorted(manifest - set(ids)) if manifest is not None else []",
                               "missing = []"),
    "pool_refuse_policy_ignored": ("pool.py", 'if policy == "refuse" and (n["failed"] or missing):', "if False:"),
    "score_k_bottom_row_unchecked": ("score.py", "if not np.array_equal(a[2], [0.0, 0.0, 1.0]):", "if False:"),
    "score_joint_denominator_is_centre": ("score.py", "n_al[name] = int(np.isfinite(per_frame).sum())",
                                          "n_al[name] = len(p)"),
    "pool_split_fields_unchecked": ("pool.py", "        if missing:\n", "        if False:\n"),
    "pool_split_finite_unchecked": ("pool.py", "        if bad:\n", "        if False:\n"),
    "pool_row_z_unchecked": ("pool.py", "if not (np.isfinite(z).all() and (z > 0).all()):", "if False:"),
    "pool_a_alias_silent": ("pool.py", 'if "a_r" not in r:', "if False:"),
    "pool_cli_typeerror_uncaught": ("pool.py", "except (ValueError, KeyError, TypeError, OSError) as e:",
                                    "except (ValueError, KeyError, OSError) as e:"),
    "pool_rows_key_list_ok": ("pool.py", "if not isinstance(doc, dict):", "if False:"),
    # regression mutants: alongray.change (the solve-change estimand) and the cloud-length check
    "chg_ray_ignored": ("change.py", 'r = g if ray == "gt" else b', "r = g"),
    "chg_share_mean_of_cos2": ("change.py", "share_along=(s_par / s_tot) if",
                               "share_along=float(np.mean(par ** 2 / np.where(d2 > 0, d2, 1.0))) if"),
    "chg_rows_positional": ("change.py", "b, a, g, ids = _ordered(b, ib), _ordered(a, ia), _ordered(g, ig), sg",
                            "b, a, g, ids = b, a, g, ib"),
    "chg_id_sets_unchecked": ("change.py", "if not (np.array_equal(sb, sg) and np.array_equal(sa, sg)):", "if False:"),
    "chg_duplicate_ids_ok": ("change.py", "if np.unique(xi).size != xi.size:", "if False:"),
    "chg_float_ids_ok": ("change.py", 'if x.dtype.kind not in "iu":', "if False:"),
    "chg_mask_ignores_after": ("change.py", "use = fin_g & fin_b & fin_a & ok_ray", "use = fin_g & fin_b & ok_ray"),
    "chg_error_on_change_ray": ("change.py", "ut = g / np.linalg.norm(g, axis=1, keepdims=True)", "ut = u"),
    "chg_zero_change_scored": ("change.py", 'status = "scored" if s_tot > 0.0 else "zero_change"', 'status = "scored"'),
    "chg_degenerate_ray_kept": ("change.py", "ok_ray = np.isfinite(rn) & (rn > min_range) & np.isfinite(gn) & (gn > min_range)",
                                "ok_ray = np.isfinite(rn) & np.isfinite(gn)"),
    "chg_signed_change_abs": ("change.py", "signed_change_along_mean=float(par.mean())",
                              "signed_change_along_mean=float(np.abs(par).mean())"),
    "chg_abs_along_signed": ("change.py", "delta_abs_along_mean=float(np.abs(aa).mean() - np.abs(ab).mean())",
                             "delta_abs_along_mean=float(aa.mean() - ab.mean())"),
    "chg_median_share_is_mean": ("change.py", '"median_window_share": float(np.median(sh))',
                                 '"median_window_share": float(np.mean(sh))'),
    "chg_paired_median_unpaired": ("change.py", '"median_paired_delta_centre": float(np.median(de))',
                                   '"median_paired_delta_centre": float(np.median([w.e_after_mean for w in framed]) '
                                   '- np.median([w.e_before_mean for w in framed]))'),
    "chg_spearman_is_pearson": ("change.py", "rx, ry = rankdata(x), rankdata(y)", "rx, ry = x, y"),
    "chg_mixed_rays_pooled": ("change.py", "if len(rays) > 1:", "if False:"),
    "chg_unassigned_ok": ("change.py", "        if extra:\n", "        if False:\n"),
    "chg_duplicate_windows_ok": ("change.py", "if len(set(ids)) != len(ids):", "if False:"),
    "chg_missing_ignored": ("change.py", "missing = [x for x in asg if x not in set(ids)]", "missing = []"),
    "chg_nonfinite_counts_swapped": ("change.py", '"n_nonfinite_before": int((~fin_b).sum()),',
                                     '"n_nonfinite_before": int((~fin_a).sum()),'),
    "chg_dropped_ids_wrong": ("change.py", "used, dropped = tuple(int(i) for i in ids[use]), tuple(int(i) for i in ids[~use])",
                              "used, dropped = tuple(int(i) for i in ids[use]), tuple(int(i) for i in ids[use])"),
    "chg_zero_change_frames_uncounted": ("change.py", 'flow["n_zero_change"] = int((d2 == 0.0).sum())',
                                         'flow["n_zero_change"] = 0'),
    "chg_min_range_unchecked": ("change.py", "if not (np.isfinite(min_range) and min_range > 0):", "if False:"),
    "score_cloud_length_vs_T_unchecked": ("score.py", "if not (len(pred_points) == len(gt_points) == p_all.shape[0]):",
                                          "if len(pred_points) != len(gt_points):"),
    # regression mutants: the scale-gauge illustration
    "gd_depth_unchecked": ("gauge_demo.py", "if np.any(x[:, 2] <= MIN_DEPTH_M):", "if False:"),
    "gd_lam_unchecked": ("gauge_demo.py", "if not (np.isfinite(lam) and lam > 0):", "if False:"),
    "gd_anchor_follows_gauge": ("gauge_demo.py", "moved = a - c + lam * c", "moved = lam * a"),
    "gd_baseline_ignored": ("gauge_demo.py", "project(k, x - b) - project(k, xs - b)", "project(k, x) - project(k, xs)"),
    "gd_scaled_about_centroid": ("gauge_demo.py", "    xs = lam * x\n", "    xs = lam * (x - x.mean(0)) + x.mean(0)\n"),
    # the missing-outcome sensitivity card
    "mb_lower_ignores_b_unknowns": ("missing_bounds.py", 'lo = ca["s"] - cb["s"] - cb["u"]', 'lo = ca["s"] - cb["s"]'),
    "mb_upper_adds_b_unknowns": ("missing_bounds.py", 'hi = ca["s"] - cb["s"] + ca["u"]',
                                 'hi = ca["s"] - cb["s"] + ca["u"] + cb["u"]'),
    "mb_pass_upper_float_wrong": ("missing_bounds.py", '"pass_upper": (s + u) / n', '"pass_upper": s / n'),
    "mb_delivery_failure_as_unknown": ("missing_bounds.py",
                                       's, u, dfail = v.count("pass"), v.count("unknown"), v.count("delivery_failure")',
                                       's, u, dfail = v.count("pass"), v.count("unknown") + v.count("delivery_failure"), '
                                       'v.count("delivery_failure")'),
    "mb_rule_unchecked": ("missing_bounds.py", "if rule not in DELIVERY_RULES:", "if False:"),
    "mb_duplicates_ok": ("missing_bounds.py", "if len(set(ids)) != len(ids):", "if False:"),
    "mb_status_unchecked": ("missing_bounds.py", "    if bad:\n", "    if False:\n"),
    "mb_definition_unchecked": ("missing_bounds.py", "if m.get(k) != protocol[k]:", "if False:"),
    "mb_cover_unchecked": ("missing_bounds.py", "if set(st) != set(scorable):", "if False:"),
    "mb_N_counts_nonscorable": ("missing_bounds.py", "    n = len(scorable)\n    lo = ",
                                "    n = len(scorable) + len(excluded)\n    lo = "),
    "mb_gate_ignores_unknowns": ("missing_bounds.py", 'if ca["u"] + cb["u"] == 0:', "if False:"),
    "mb_same_method_ok": ("missing_bounds.py", 'if a.get("method") == b.get("method"):', "if False:"),
    "mb_nonscorable_status_ok": ("missing_bounds.py", "    if off:\n", "    if False:\n"),
    "mb_cli_exit_zero_on_error": ("missing_bounds.py", "        return 2\n", "        return 0\n"),
    # the rev_* mutants: each survived an earlier version of the own tests
    "rev_degenerate_double_count": ("change.py", '"n_ray_degenerate": int((fin_ray & ~ok_ray).sum())',
                                    '"n_ray_degenerate": int((~ok_ray).sum())'),
    "rev_z1_refusal_removed": ("pool.py", "if math.isfinite(z1) and not (z1 > 0 and z1 * z1 <= zz * (1 + MOMENT_RTOL)):",
                               "if False:"),
    "rev_rbar_refusal_removed": ("pool.py", "if rbar * rbar > e2 + MOMENT_RTOL * scale:", "if False:"),
    "rev_diff_medians_over_scored": ("change.py", "np.median([w.e_after_mean for w in framed])",
                                     "np.median([w.e_after_mean for w in scored])"),
    "rev_share_count_strict": ("change.py", "int((sh >= 0.9).sum())", "int((sh > 0.9).sum())"),
    "rev_ids_not_str": ("pool.py", 'ids = [str(r["unit_id"]) for r in rows if "unit_id" in r]',
                        'ids = [r["unit_id"] for r in rows if "unit_id" in r]'),
    "rev_group_mismatch_ok": ("pool.py", "        if wrong:\n", "        if False:\n"),
    # the best gauge move of the prediction and the least e_par it reaches
    "kstar_is_the_oracle": ("core.py", "return lam * a / (lam * lam * a + b)", "return 1.0 / lam"),
    "kstar_numerator_drops_lam": ("core.py", "return lam * a / (lam * lam * a + b)", "return a / (lam * lam * a + b)"),
    "kstar_lam_unsquared": ("core.py", "return lam * a / (lam * lam * a + b)", "return lam * a / (lam * a + b)"),
    "emin_is_oracle_floor": ("core.py", "return z * off / math.hypot(lam * z, off)", "return off / lam"),
    "emin_is_off_orbit_part": ("core.py", "return z * off / math.hypot(lam * z, off)", "return off"),
    "emin_lam_dropped": ("core.py", "math.hypot(lam * z, off)", "math.hypot(z, off)"),
    "kstar_split_unchecked": ("core.py", "if not (math.isfinite(lam) and lam > 0.0 and math.isfinite(z) and z > 0.0",
                              "if False and not (math.isfinite(lam) and lam > 0.0 and math.isfinite(z) and z > 0.0"),
    "EQUIV_along_rewritten": ("core.py", 'along = np.einsum("ij,ij->i", d, u)',
                              'along = np.einsum("ij,ij->i", p, u) - np.linalg.norm(g, axis=1)'),
    # 0.4.0: optional frame ids in score and the command line (rules of alongray.change)
    "ids_ignored": ("score.py", "    if pred_ids is not None:\n        op, og = _frame_join(",
                    "    if False:\n        op, og = _frame_join("),
    "ids_same_set_becomes_same_length": ("score.py", "    if not np.array_equal(fp[op], fg[og]):",
                                         "    if fp.size != fg.size:"),
    "ids_intersection_instead_of_identity": (
        "score.py",
        ('    op, og = np.argsort(fp, kind="stable"), np.argsort(fg, kind="stable")\n'
         "    if not np.array_equal(fp[op], fg[og]):"),
        ('    cp, cg = np.isin(fp, fg), np.isin(fg, fp)\n'
         '    op = np.flatnonzero(cp)[np.argsort(fp[cp], kind="stable")]\n'
         '    og = np.flatnonzero(cg)[np.argsort(fg[cg], kind="stable")]\n    if op.size == 0:')),
    "ids_rows_not_reordered": ("score.py", "        pred, gt = np.asarray(pred)[op], np.asarray(gt)[og]",
                               "        pred, gt = np.asarray(pred), np.asarray(gt)"),
    "ids_clouds_not_reordered": (
        "score.py", "            pred_points, gt_points = [pred_points[i] for i in op], [gt_points[i] for i in og]",
        "            pred_points, gt_points = list(pred_points), list(gt_points)"),
    "cli_frame_ids_ignored": ("cli.py", '    if "frame_ids" in pred:\n        if "frame_idx" in pred',
                              '    if False:\n        if "frame_idx" in pred'),
    "cli_require_frame_ids_ignored": ("cli.py", "    if args.require_frame_ids and not (", "    if False and not ("),
    # 0.4.0: fewer than two clusters give NOT_ESTIMABLE, never a zero-width interval
    "boot_one_cluster_guard_removed": ("pool.py", "    if len(labels) < 2:\n        return NOT_ESTIMABLE",
                                       "    if False:\n        return NOT_ESTIMABLE"),
    "boot_zero_cluster_guard_removed": ("pool.py", "    if not labels:\n        return NOT_ESTIMABLE",
                                        "    if False:\n        return NOT_ESTIMABLE"),
    # equivalent: with one cluster, rng.integers(0, 1, 1) draws no bits, so skipping those draws changes nothing
    "EQUIV_boot_one_cluster_skips_its_draws": (
        "pool.py", "    by = {c: [i for i, x in enumerate(clusters) if x == c] for c in labels}\n",
        ("    if len(labels) < 2:\n        return NOT_ESTIMABLE\n"
         "    by = {c: [i for i, x in enumerate(clusters) if x == c] for c in labels}\n")),
}


def _pytest(cwd: Path, args: list[str], env: dict[str, str]) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *args], cwd=cwd,
                       env=env, capture_output=True, text=True, check=False)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return r.returncode, lines[-1] if lines else r.stderr.strip()[-200:]


def main() -> int:
    extra = os.environ.get("ALONGRAY_EXTRA_TESTS")
    extra_path = os.environ.get("ALONGRAY_EXTRA_PYTHONPATH", "")
    rows: dict[str, dict[str, tuple[int, str]]] = {}
    for name, (fname, old, new) in MUTANTS.items():
        src = (PKG / "src/alongray" / fname).read_text()
        if old and src.count(old) != 1:
            print(f"{name}: anchor occurs {src.count(old)} times in {fname}; update the mutant table")
            return 1
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "pkg"
            shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
            (pkg / "src/alongray" / fname).write_text(src.replace(old, new) if old else src)
            env = dict(os.environ, PYTHONPATH=str(pkg / "src"), CUDA_VISIBLE_DEVICES="")
            res = {"own": _pytest(pkg, [], env)}
            if extra:
                penv = dict(env, PYTHONPATH=os.pathsep.join([str(pkg / "src"), *filter(None, [extra_path])]))
                res["extra"] = _pytest(pkg, ["--rootdir", str(Path(extra).parent), extra], penv)
        rows[name] = res
        print(f"{name:28s} " + "  ".join(f"{k}: rc={v[0]} ({v[1]})" for k, v in res.items()), flush=True)
    base = rows["BASELINE_unmutated"]
    if any(rc != 0 for rc, _ in base.values()):
        print("BASELINE FAILS: the check is void")
        return 1
    real = [n for n in rows if not n.startswith(("BASELINE", "EQUIV"))]
    # a kill is a FAILED test: rc 1 with "failed" in the summary (a collection error or crash is not a kill)
    survivors = [n for n in real if not (rows[n]["own"][0] == 1 and "failed" in rows[n]["own"][1])]
    print(f"own tests killed {len(real) - len(survivors)}/{len(real)} non-equivalent mutants; survivors: {survivors}")
    if extra:
        killed = sum(rows[n]["extra"][0] == 1 and "failed" in rows[n]["extra"][1] for n in rows
                     if n != "BASELINE_unmutated")
        print(f"the extra suite killed {killed}/{len(rows) - 1} mutants (incl. equivalent)")
    return 0 if not survivors else 1


if __name__ == "__main__":
    import argparse

    # No options: parsing only answers --help (and refuses stray arguments) before the campaign starts.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
