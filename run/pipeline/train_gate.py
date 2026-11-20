"""Fit, validate and calibrate the interaction gate from the token index; write the gate bundle and its card.

Runs on CPU. Each partition of `gate_train.index_path` loads separately. The model is
built after `torch.manual_seed(gate_train.seed)` and fit on the `fit` partition only (DexYCB windows; the 100DOH clips
never move the weights). `val` (the held-out DexYCB subject) gives `val_auroc`, the promotion metric;
`val_seq` gives `val_seq_auroc`, reported only (null when empty). The `calib` partition (100DOH train clips with
portable contact) sets the split-conformal threshold at `gate_train.alpha`; the `check` clips' scores are listed by
clip name. The card (`gate_train.card_path`, JSON) records everything; the bundle (`gate_train.bundle_path`) is written
only when `val_auroc >= gate_train.min_val_auroc` and the conformal k >= 1. Otherwise the card says why, no bundle file
exists, and the script exits non-zero. Nothing is retuned on the calibration clips.

  cd ${FASTDO_ROOT} && env CUDA_VISIBLE_DEVICES="" uv run --extra dev python \
    run/pipeline/train_gate.py paths.root=${FASTDO_ROOT}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from fastdo.bench.bench_stages import resolve_path
from fastdo.bench.gpu_budget import apply_cpu_budget, apply_gpu_budget
from fastdo.gate.bundle import save_gate_bundle
from fastdo.gate.calibrate import GateCalibration, auroc, calibrate_gate
from fastdo.gate.dataset import (
    DOH_ITEM_PREFIX,
    PARTITIONS,
    GateDataset,
    item_counts,
    load_gate_dataset,
    read_gate_index,
)
from fastdo.gate.model import GateConfig, InteractionGate
from fastdo.gate.train import GateTrainConfig, gate_scores, train_gate

logger = logging.getLogger(__name__)

HAND_CONF_SOURCE = "none"      # zeros until a hand detector runs before the gate


def _scores(model: InteractionGate, data: GateDataset) -> Tensor:
    return gate_scores(model, data.features) if data.features.shape[0] else torch.zeros(0)


def _auroc_or_none(model: InteractionGate, data: GateDataset) -> float | None:
    positive = data.labels.interact > 0.5
    if not bool(positive.any()) or bool(positive.all()):
        return None
    return auroc(_scores(model, data), data.labels.interact)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def false_positive_rates(data: Mapping[str, Any], scores_of: Callable[[Any], Any],
                         threshold: float | None) -> dict[str, Any]:
    """Acceptance rate of the NEGATIVE items, by partition, at the calibrated threshold.

    The conformal calibration fixes RECALL and says nothing about this. An earlier card carried no
    false-positive column at all, so a reader could infer only the fit-partition rate -- 0.004 against
    0.123 out of fold, a 30x understatement of the deployable number. `min_val_auroc` cannot stand in:
    AUROC is threshold-free ranking and this threshold is placed by recall, so a gate can rank almost
    perfectly and still admit one negative in eight where it ships.

    `out_of_fold` pools `val` and `val_seq`, the partitions the fit never saw. That is the figure the
    yield law needs: `a = pi*r + (1-pi)*f`, and `(1-pi)*f / (pi*r)` diverges as `pi -> 0`, so the
    false-positive rate costs most exactly where usable windows are rarest.
    """
    if threshold is None:
        return {"threshold": None, "by_partition": {}, "out_of_fold": None, "n_out_of_fold": 0}
    by_partition: dict[str, Any] = {}
    pooled_above = pooled_n = 0
    for name, dataset in data.items():
        labels = dataset.labels.interact.tolist()
        negatives = [s for s, y in zip(scores_of(dataset).tolist(), labels, strict=True) if y <= 0.5]
        if not negatives:
            continue
        above = sum(1 for s in negatives if s >= threshold)
        by_partition[name] = {"n": len(negatives), "accepted": above, "rate": above / len(negatives)}
        if name in {"val", "val_seq"}:
            pooled_above, pooled_n = pooled_above + above, pooled_n + len(negatives)
    return {"threshold": threshold, "by_partition": by_partition,
            "out_of_fold": (pooled_above / pooled_n) if pooled_n else None, "n_out_of_fold": pooled_n}


def run_train_gate(cfg: DictConfig, root: Path) -> dict[str, Any]:
    """Fits, validates and calibrates the gate, writes the card (and the bundle when promoted); returns the card.

    Raises:
        FileExistsError: If the bundle or card exists and `gate_train.overwrite` is false.
        ValueError: If the index has no `fit` item or its provenance has no crop.
    """
    gt = cfg.gate_train
    index_path, bundle_path = resolve_path(root, gt.index_path), resolve_path(root, gt.bundle_path)
    card_path = resolve_path(root, gt.card_path)
    for path in (bundle_path, card_path):
        if path.exists() and not bool(gt.overwrite):
            raise FileExistsError(f"{path} exists; set gate_train.overwrite=true to replace it")
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    items, provenance = read_gate_index(index_path)
    if "crop" not in provenance:
        raise ValueError(f"{index_path}: the index provenance records no encoder crop")
    crop = str(provenance["crop"])
    data = {partition: load_gate_dataset(index_path, {partition}) for partition in PARTITIONS}
    fit = data["fit"]
    if fit.features.shape[0] == 0:
        raise ValueError(f"{index_path} has no fit items")

    alpha, min_auroc, seed = float(gt.alpha), float(gt.min_val_auroc), int(gt.seed)
    train_cfg = GateTrainConfig(steps=int(gt.steps), batch_size=int(gt.batch_size), lr=float(gt.lr),
                                weight_decay=float(gt.weight_decay), seed=seed)
    torch.manual_seed(seed)
    model = InteractionGate(GateConfig(token_dim=int(fit.features.shape[2]) - 1, hidden=int(gt.hidden)))
    result = train_gate(model, fit.features, fit.labels, train_cfg)
    val_auroc, val_seq_auroc = _auroc_or_none(model, data["val"]), _auroc_or_none(model, data["val_seq"])

    reasons: list[str] = []
    if val_auroc is None:
        reasons.append("val_auroc is undefined: the subject-held-out val partition lacks a positive or a negative")
    elif val_auroc < min_auroc:
        reasons.append(f"val_auroc {val_auroc:.4f} < min_val_auroc {min_auroc:g}")
    calib = data["calib"]
    calibration: GateCalibration | None = None
    try:
        calibration = calibrate_gate(_scores(model, calib), calib.labels.interact, alpha)
    except ValueError as exc:
        reasons.append(f"conformal k = 0 at alpha {alpha:g}, so no recall guarantee: {exc}")
    check = data["check"]
    check_scores = {item_id.removeprefix(DOH_ITEM_PREFIX): float(score)
                    for item_id, score in zip(check.item_ids, _scores(model, check).tolist(), strict=True)}
    threshold = calibration.threshold if calibration else None
    fpr = false_positive_rates(data, lambda ds: _scores(model, ds), threshold)
    # `f` was never a promotion criterion. A bound is OPT-IN so this change breaks no existing
    # run, but its absence is now RECORDED rather than invisible -- which was the whole defect.
    max_fpr = None if gt.get("max_fpr") is None else float(gt.max_fpr)
    if max_fpr is not None and fpr["out_of_fold"] is not None and fpr["out_of_fold"] > max_fpr:
        reasons.append(f"out-of-fold false-positive rate {fpr['out_of_fold']:.4f} > max_fpr {max_fpr:g}")
    promoted = not reasons
    card: dict[str, Any] = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "promoted": promoted, "reasons": reasons,
        "index_path": str(index_path), "index_sha256": index_sha, "bundle_path": str(bundle_path),
        "bundle_sha256": None, "alpha": alpha, "k": calibration.k if calibration else 0,
        "n_calib_pos": int((calib.labels.interact > 0.5).sum()),
        "threshold": calibration.threshold if calibration else None,
        "guaranteed_recall": round(calibration.guaranteed_recall, 6) if calibration else None,
        "empirical_recall": calibration.empirical_recall if calibration else None,
        "empirical_recall_basis": "in-sample: the calibration clips themselves",
        "val_auroc": val_auroc, "val_seq_auroc": val_seq_auroc, "min_val_auroc": min_auroc,
        # the column the card used to lack: what this threshold actually admits, by partition
        "false_positive_rates": fpr, "max_fpr": max_fpr, "fpr_bounded": max_fpr is not None,
        "val_subject": provenance.get("val_subject"), "check_scores": check_scores,
        "counts": item_counts(items), "gate_train": OmegaConf.to_container(gt, resolve=True),
        "fit": {"n_items": int(fit.features.shape[0]), "pos_weight": result.pos_weight,
                "first_loss": result.losses[0] if result.losses else None, "final_loss": result.final_loss},
        "token_dim": model.cfg.token_dim, "crop": crop, "hand_conf_source": HAND_CONF_SOURCE,
        "rigid_calibrated": False, "index_provenance": provenance,
    }
    if promoted and calibration is not None:
        card["bundle_sha256"] = save_gate_bundle(bundle_path, model, calibration, crop, card,
                                                 hand_conf_source=HAND_CONF_SOURCE)
    else:
        bundle_path.unlink(missing_ok=True)          # never leave an earlier bundle beside a refusing card
    _write_json(card_path, card)
    logger.info("gate card %s: promoted %s, val_auroc %s, val_seq_auroc %s, k %s, threshold %s", card_path, promoted,
                val_auroc, val_seq_auroc, card["k"], card["threshold"])
    return card


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    apply_cpu_budget()
    apply_gpu_budget()
    try:
        root = Path(hydra.utils.get_original_cwd())
    except ValueError:                    # called with a composed config, outside a Hydra run (tests)
        root = Path.cwd()
    card = run_train_gate(cfg, root)
    if not card["promoted"]:
        logger.error("the gate was not promoted and no bundle was written: %s", "; ".join(card["reasons"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
