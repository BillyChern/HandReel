"""`fastdo.physics.warp_compat` reproduces Warp 1.10.1's contiguous `copy` across element sizes.

Runs wherever `warp` imports (the `retargeting` env, or with FASTDO_MJWARP314_SITE on sys.path), on the CPU device;
skipped in environments without warp. The expected bytes are computed independently with numpy (a raw memcpy of
count * sizeof(src.dtype) bytes at byte offsets offset * sizeof(dtype)).
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")


@pytest.fixture(scope="module")
def compat():
    import importlib

    wp.config.quiet = True
    wp.init()
    mod = importlib.import_module("fastdo.physics.warp_compat")
    original = wp.copy
    mod.install_copy_compat(wp)
    yield mod
    wp.copy = original


def test_float_into_spatial_vector_is_a_raw_byte_copy(compat) -> None:
    src_np = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6) * 0.5
    dest = wp.zeros((3, 4), dtype=wp.spatial_vectorf, device="cpu")
    wp.copy(dest, wp.array(src_np, dtype=wp.float32, device="cpu"))
    assert np.array_equal(dest.numpy().reshape(-1), src_np.reshape(-1))


def test_offsets_and_count_follow_warp_1_10_byte_arithmetic(compat) -> None:
    src_np = np.arange(30, dtype=np.float32)
    dest = wp.zeros(8, dtype=wp.vec3f, device="cpu")              # 24 floats
    wp.copy(dest, wp.array(src_np, dtype=wp.float32, device="cpu"), dest_offset=2, src_offset=5, count=9)
    want = np.zeros(24, np.float32)
    want[2 * 3: 2 * 3 + 9] = src_np[5:14]                       # bytes: dest 2*12, src 5*4, n 9*4
    assert np.array_equal(dest.numpy().reshape(-1), want)


def test_overflow_is_still_refused(compat) -> None:
    dest = wp.zeros(2, dtype=wp.vec3f, device="cpu")              # 6 floats
    with pytest.raises(RuntimeError):
        wp.copy(dest, wp.array(np.zeros(7, np.float32), dtype=wp.float32, device="cpu"))


def test_matching_dtypes_pass_through_unchanged(compat) -> None:
    a = wp.array(np.arange(5, dtype=np.float32), dtype=wp.float32, device="cpu")
    b = wp.zeros(5, dtype=wp.float32, device="cpu")
    n0 = compat.stats()["compat_copies"]
    wp.copy(b, a)
    assert np.array_equal(b.numpy(), a.numpy()) and compat.stats()["compat_copies"] == n0


def test_install_is_idempotent(compat) -> None:
    assert compat.install_copy_compat(wp) is False
