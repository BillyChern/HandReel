"""Smoke test: the package imports and exposes its public subpackages."""

import fastdo


def test_package_importable_and_versioned() -> None:
    assert fastdo.__version__ == "0.1.0"
    for name in ("io", "bench", "teacher", "tracker", "retarget"):
        assert name in fastdo.__all__
