"""KA-ID at the config/registration level and the opt-in `tba` stage's wiring.

With `tba_stage: null` (the shipped value) nothing is registered: the stage list, the stage functions and the labels
are exactly those of the config without the key (so every artefact is BASE's). With a block the stage sits right
after contact (or solve) and before qc.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from fastdo.bench.bench_stages import build_stage_fns
from fastdo.engine.orchestrator import STAGE_GPU
from fastdo.engine.workers_tba import with_tba
from fastdo.tba.config import TBA_KEY, TbaConfig, tba_from_mapping

_REPO = Path(__file__).resolve().parents[2]


def _cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_REPO / "run/conf"), version_base=None):
        return compose(config_name="config", overrides=[f"paths.root={_REPO}", *overrides])


def test_the_shipped_value_is_null_and_null_means_off() -> None:
    assert OmegaConf.load(_REPO / "run/conf/config.yaml")[TBA_KEY] is None
    assert tba_from_mapping(None) is None
    assert tba_from_mapping({}) == TbaConfig()
    assert tba_from_mapping({"carve": False}).carve is False


@pytest.mark.parametrize("bad", [{"gauge": "pointmap"}, {"unknown_key": 1}, {"vote_q": 0.0}, {"min_live": 0},
                                 {"gnc_c_px": [8.0, 2.0]}, [1, 2], "on"])
def test_bad_blocks_are_refused(bad) -> None:
    with pytest.raises(ValueError):
        tba_from_mapping(bad)


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_ka_id_null_registers_exactly_what_the_config_without_the_key_registers() -> None:
    on_null = _cfg()
    without = _cfg()
    with open_dict(without):
        del without[TBA_KEY]
    fns_a, labels_a = build_stage_fns(on_null, _REPO, [])
    fns_b, labels_b = build_stage_fns(without, _REPO, [])
    assert "tba" not in fns_a and "tba" not in labels_a
    assert list(labels_a) == list(labels_b) and labels_a == labels_b
    assert set(fns_a) == set(fns_b)
    # mutation: a block registers the stage, after contact (DexYCB profile) / solve and before qc
    with open_dict(on_null):
        on_null[TBA_KEY] = {}
    fns_c, labels_c = build_stage_fns(on_null, _REPO, [])
    names = list(labels_c)
    assert "tba" in fns_c and "TBA-V" in labels_c["tba"]
    at = names.index("contact") if "contact" in names else names.index("solve")
    assert names.index("tba") == at + 1
    assert "qc" not in names or names.index("qc") > names.index("tba")
    assert STAGE_GPU["tba"] is False


def test_with_tba_places_the_stage_once() -> None:
    assert with_tba(("solve", "qc")) == ("solve", "tba", "qc")
    assert with_tba(("solve", "contact", "qc")) == ("solve", "contact", "tba", "qc")
    assert with_tba(("solve", "contact", "tba", "qc")) == ("solve", "contact", "tba", "qc")
    with pytest.raises(ValueError):
        with_tba(("decode", "qc"))
    with pytest.raises(ValueError):
        with_tba(("solve", "tba", "contact", "qc"))
