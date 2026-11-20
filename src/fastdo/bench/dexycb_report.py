"""Printed summary of the DexYCB object-metric evaluation under the strict and scaled-ICP protocols.

Used by `run/pipeline/eval_solver_dexycb.py`: one labelled block per protocol whose sequence means are present,
with a published baseline row printed only under the per-frame scaled-ICP block, the protocol DO AS I DO
Table 2 follows (MCC-HO / DiffHOI / G-HOP), so it is never read against the strict camera-frame metric.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

__all__ = ["GT_MESH_CAVEAT", "PROTOCOL_BLOCKS", "PROTOCOL_KEYS", "summary_lines"]

# Printed under the per-frame scaled-ICP block when every summarised row scored the GT mesh as the predicted shape.
GT_MESH_CAVEAT = ("caveat: predicted shape = GT mesh; scaled ICP per frame forgives almost all pose error, so this "
                  "block is NOT comparable with DO AS I DO Table 2 (reconstructed shapes); compare tracking with "
                  "the per-clip block")

# (block title, result-field suffix); the strict fields keep their original names f5 / f10 / chamfer_cm2.
PROTOCOL_BLOCKS: tuple[tuple[str, str], ...] = (
    ("strict (camera frame, no alignment)", ""),
    ("scaled ICP per frame (Table 2 protocol)", "_icp_frame"),
    ("scaled ICP per clip", "_icp_clip"),
)
PROTOCOL_KEYS: tuple[str, ...] = ("f5_icp_frame", "f10_icp_frame", "chamfer_icp_frame_cm2",
                                  "f5_icp_clip", "f10_icp_clip", "chamfer_icp_clip_cm2")


def summary_lines(means: Mapping[str, float], n_rows: int, n_failed: int, baseline: Mapping[str, float],
                  baseline_label: str, *, shape_sources: Collection[str]) -> list[str]:
    """Summary table lines from sequence means keyed like the result rows (`f5`, `f10_icp_clip`, `seconds`, ...).

    A protocol block is printed when its `f5<suffix>` mean is present; `baseline` (`f5`, `f10`, `chamfer_cm2`)
    is printed, labelled `baseline_label`, only under the per-frame scaled-ICP block. `shape_sources` holds the
    `shape_source` of every summarised row: when all of them are "gt_mesh", `GT_MESH_CAVEAT` is printed directly
    under the per-frame scaled-ICP block, before the baseline row.
    """
    header = f"{'':10s}{'F-5':>8s}{'F-10':>8s}{'Chamfer(cm^2)':>15s}{'sec':>8s}"
    gt_mesh_only = bool(shape_sources) and all(source == "gt_mesh" for source in shape_sources)
    lines: list[str] = []
    for title, sfx in PROTOCOL_BLOCKS:
        if f"f5{sfx}" not in means:
            continue
        mean = (f"{'mean':10s}{means[f'f5{sfx}']:8.3f}{means[f'f10{sfx}']:8.3f}{means[f'chamfer{sfx}_cm2']:15.3f}"
                f"{means['seconds']:8.2f}  (over {n_rows} evaluated sequences; {n_failed} failed and excluded)")
        lines += [title, header, mean]
        if sfx == "_icp_frame":
            if gt_mesh_only:
                lines.append(GT_MESH_CAVEAT)
            lines.append(f"{'baseline':10s}{baseline['f5']:8.3f}{baseline['f10']:8.3f}"
                         f"{baseline['chamfer_cm2']:15.3f}{'':8s}  {baseline_label}")
    if "f5_icp_frame" not in means:
        lines.append("(protocol metrics off: no row is comparable with the baseline, which is scored after ICP)")
    return lines
