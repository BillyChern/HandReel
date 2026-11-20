"""The CoTracker3 torch.hub code is loaded at the pinned commit by default; `main` opts into the moving branch."""
from __future__ import annotations

import pytest
import torch

from fastdo.tracks import cotracker as ct


class _Model:
    def to(self, _device: str) -> _Model:
        return self

    def eval(self) -> _Model:
        return self


def test_default_is_the_pinned_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTDO_COTRACKER_HUB_REF", raising=False)
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(torch.hub, "load", lambda repo, entry: seen.append((repo, entry)) or _Model())
    ct._load_cotracker_model("cpu")
    assert seen == [(f"facebookresearch/co-tracker:{ct.COTRACKER_HUB_COMMIT}", "cotracker3_online")]


def test_main_opts_into_the_unpinned_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTDO_COTRACKER_HUB_REF", "main")
    assert ct.cotracker_hub_ref() == "facebookresearch/co-tracker"


def test_an_explicit_ref_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTDO_COTRACKER_HUB_REF", ct.COTRACKER_HUB_COMMIT)
    seen: list[str] = []
    monkeypatch.setattr(torch.hub, "load", lambda repo, entry: seen.append(repo) or _Model())
    ct._load_cotracker_model("cpu")
    assert seen == [f"facebookresearch/co-tracker:{ct.COTRACKER_HUB_COMMIT}"]


def test_empty_ref_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTDO_COTRACKER_HUB_REF", "")
    with pytest.raises(ValueError, match="set but empty"):
        ct.cotracker_hub_ref()
