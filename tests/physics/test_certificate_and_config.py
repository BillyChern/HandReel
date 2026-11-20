"""Certificate items, repair checkpoint, planner config defaults (OFF) and JSON round trip."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fastdo.physics.certificate import (
    CertificateInput,
    certificate,
    displacement_after,
    post_warmup_means,
    repair_checkpoint,
)
from fastdo.physics.planner_config import C2Config, CertificateConfig, D1Config, PlannerConfig, from_dict

GOOD = dict(completed=True, whole_pos_m=0.02, whole_rot_rad=0.1, post_pos_m=0.03, post_rot_rad=0.15, max_pen_m=0.004,
            obj_disp_m=0.2, ref_disp_m=0.22)


def test_certificate_passes_and_each_item_can_fail() -> None:
    cfg = CertificateConfig()
    assert certificate(CertificateInput(**GOOD), cfg)["passed"]
    bad = {
        "C1_completed": {"completed": False},
        "C2_whole": {"whole_rot_rad": 0.5},
        "C3_post": {"post_pos_m": 0.1},
        "C4_post_pos": {"post_pos_m": 0.07},
        "C5_penetration": {"max_pen_m": 0.0081},
        "C7_displacement": {"obj_disp_m": 0.1},
    }
    for item, change in bad.items():
        out = certificate(CertificateInput(**{**GOOD, **change}), cfg)
        assert not out["passed"], item
        assert not out["items"][item], item
    nan = certificate(CertificateInput(**{**GOOD, "post_rot_rad": math.nan}), cfg)
    assert not nan["items"]["C3_post"]


def test_optional_items() -> None:
    cfg = CertificateConfig()
    base = certificate(CertificateInput(**GOOD), cfg)
    assert "C6_contact" not in base["items"] and "R_robust" not in base["items"]
    assert not certificate(CertificateInput(**GOOD, contact_kept=0.4), cfg)["passed"]
    assert certificate(CertificateInput(**GOOD, contact_kept=0.5), cfg)["passed"]
    assert not certificate(CertificateInput(**GOOD, robust_frac=0.86), cfg)["passed"]
    assert certificate(CertificateInput(**GOOD, robust_frac=0.9), cfg)["passed"]


def test_thresholds_equal_the_scorer_definitions() -> None:
    from fastdo.retarget.isolation_report import SUCCESS_POS_M, SUCCESS_ROT_RAD

    cfg = CertificateConfig()
    assert cfg.success_pos_m == SUCCESS_POS_M and cfg.success_rot_rad == SUCCESS_ROT_RAD


def test_post_warmup_means_and_displacement() -> None:
    pos = np.r_[np.zeros(10), np.full(10, 0.2)]
    assert post_warmup_means(pos, pos, 10, 15) == (pytest.approx(0.2), pytest.approx(0.2))
    assert math.isnan(post_warmup_means(pos, pos, 10, 10)[0])
    p = np.zeros((5, 3))
    p[3:, 2] = [0.1, 0.05]
    assert displacement_after(p, 1) == pytest.approx(0.1)


def test_repair_checkpoint_rule() -> None:
    cfg = CertificateConfig()
    err = np.r_[np.zeros(600), np.full(150, 0.01), np.full(50, 0.2)]
    t_fail, t_c = repair_checkpoint(err, None, 600, cfg)
    assert t_fail == 750 and t_c == 500                    # >= 1 s before, on the 0.5 s grid
    assert repair_checkpoint(err, 700, 600, cfg) == (700, 500)
    assert repair_checkpoint(np.zeros(800), None, 600, cfg) == (600, 400)
    assert repair_checkpoint(np.zeros(800), 100, 600, cfg)[1] == 0


def test_planner_defaults_are_off_and_round_trip() -> None:
    p = PlannerConfig()
    assert p.planner == "ref" and not p.d1.enabled and not p.c2.enabled
    q = PlannerConfig(planner="c2", d1=D1Config(enabled=True), c2=C2Config(enabled=True, force_theta=(0.0,) * 7,
                                                                            force_repair_from=400))
    back = from_dict(json.loads(json.dumps(q.to_dict())))
    assert back == q
