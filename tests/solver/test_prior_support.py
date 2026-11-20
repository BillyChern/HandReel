"""What the temporal priors' support scale averages over."""
from __future__ import annotations

import pytest
import torch

from fastdo.solver.batch import prior_support


def test_the_modes_agree_when_every_frame_is_observed() -> None:
    valid = torch.ones(10, 64, dtype=torch.bool)
    assert prior_support(valid, "all_frames") == prior_support(valid, "observed_frames") == 64.0


def test_unobserved_frames_dilute_only_the_shipped_mode() -> None:
    """Pointmaps on 3 of 24 frames: `all_frames` weakens the priors 8x, `observed_frames` does not."""
    valid = torch.zeros(24, 64, dtype=torch.bool)
    valid[[0, 12, 23]] = True
    assert prior_support(valid, "all_frames") == pytest.approx(64.0 * 3 / 24)
    assert prior_support(valid, "observed_frames") == pytest.approx(64.0)


def test_the_floor_and_the_default_are_unchanged() -> None:
    empty = torch.zeros(5, 8, dtype=torch.bool)
    assert prior_support(empty) == prior_support(empty, "observed_frames") == 1.0
    valid = torch.ones(4, 3, dtype=torch.bool)
    valid[0, :2] = False                         # the shipped formula, exactly: sum / T
    assert prior_support(valid) == pytest.approx(max(1.0, 10 / 4))


def test_an_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="prior_support"):
        prior_support(torch.ones(2, 2, dtype=torch.bool), "per_clip")
