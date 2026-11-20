"""fastdo.audit on the synthetic known answers of `_ka.py` (CPU, MuJoCo, no stored data)."""

from __future__ import annotations

import pytest

from tests.audit._ka import CHECKS, Impl


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_known_answer(name: str) -> None:
    CHECKS[name](Impl())
