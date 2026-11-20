"""`read_npz_rows` / `read_decode_rows`: the selected rows, byte-identical to a full `np.load`, read alone."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fastdo.engine import npz_rows
from fastdo.engine.npz_rows import npz_array_shape, read_npz_rows
from fastdo.engine.worker_support import DECODE_FILE, read_decode_rows, save_npz
from fastdo.engine.workers_frames import make_decode_worker
from tests.engine.fakes import forbid_full_frames, random_frames, write_decode

N, H, W = 9, 7, 5


def _decode(tmp_path: Path, n: int = N, seed: int = 0) -> tuple[Path, np.ndarray]:
    """A decode.npz written by the decode stage's writer, `save_npz`, with the decode stage's keys."""
    frames = random_frames(n, h=H, w=W, seed=seed)
    save_npz(tmp_path / DECODE_FILE, frames=frames, frame_idx=np.arange(n, dtype=np.int64) * 3,
             fps=np.float64(10.0), stride=np.int64(3))
    return tmp_path / DECODE_FILE, frames


def _full(path: Path, rows: object) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["frames"])[np.asarray(rows, dtype=np.intp)]


def _same(got: np.ndarray, expected: np.ndarray) -> None:
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert got.tobytes() == expected.tobytes()
    assert type(got) is np.ndarray and got.flags.c_contiguous and got.flags.writeable


@pytest.mark.parametrize("rows", [[0], [N - 1], [3, 1, 7], [4, 4, 0, 4], [-1, 2], list(range(N)),
                                  np.array([8, 0, 5], dtype=np.int32), np.array([2, 6], dtype=np.uint64)])
def test_rows_match_the_full_load_byte_for_byte(tmp_path: Path, rows: object,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _decode(tmp_path)
    expected = _full(path, rows)
    forbid_full_frames(monkeypatch)
    _same(read_npz_rows(path, "frames", rows), expected)       # type: ignore[arg-type]


def test_random_row_sets_match_the_full_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, frames = _decode(tmp_path, n=31, seed=3)
    rng = np.random.default_rng(147)
    cases = [rng.integers(0, 31, size=int(rng.integers(1, 40))) for _ in range(25)]
    forbid_full_frames(monkeypatch)
    for rows in cases:
        _same(read_npz_rows(path, "frames", rows), frames[rows])


def test_decode_rows_returns_the_frame_idx_unchanged(tmp_path: Path) -> None:
    path, frames = _decode(tmp_path)
    got, frame_idx = read_decode_rows(path, [5, 2])
    _same(got, frames[[5, 2]])
    np.testing.assert_array_equal(frame_idx, np.arange(N, dtype=np.int64) * 3)
    assert frame_idx.dtype == np.int64


def test_an_empty_selection_is_a_zero_row_array(tmp_path: Path) -> None:
    path, _ = _decode(tmp_path)
    for rows in ([], np.array([], dtype=np.int64)):
        _same(read_npz_rows(path, "frames", rows), _full(path, []))
        assert read_npz_rows(path, "frames", rows).shape == (0, H, W, 3)


def test_out_of_range_rows_raise_like_numpy(tmp_path: Path) -> None:
    path, _ = _decode(tmp_path)
    with pytest.raises(IndexError):
        read_npz_rows(path, "frames", [N])
    with pytest.raises(IndexError):
        read_npz_rows(path, "frames", [-N - 1])
    with pytest.raises(TypeError, match="1-D sequence of integers"):
        read_npz_rows(path, "frames", [True, False])
    with pytest.raises(TypeError, match="1-D sequence of integers"):
        read_npz_rows(path, "frames", [[0, 1]])
    with pytest.raises(KeyError):
        read_npz_rows(path, "pixels", [0])


def test_a_compressed_archive_falls_back_to_the_full_load(tmp_path: Path) -> None:
    frames = random_frames(N, h=H, w=W, seed=1)
    path = tmp_path / "decode.npz"
    np.savez_compressed(path, frames=frames, frame_idx=np.arange(N, dtype=np.int64))
    assert zipfile.ZipFile(path).getinfo("frames.npy").compress_type == zipfile.ZIP_DEFLATED
    assert npz_rows._stored_layout(path, "frames") is None
    for rows in ([2, 0, 2], [N - 1], []):
        _same(read_npz_rows(path, "frames", rows), frames[np.asarray(rows, dtype=np.intp)])
    assert npz_array_shape(path, "frames") == (N, H, W, 3)


def test_a_fortran_ordered_array_falls_back_to_the_full_load(tmp_path: Path) -> None:
    frames = np.asfortranarray(random_frames(N, h=H, w=W, seed=2))
    path = tmp_path / "decode.npz"
    save_npz(path, frames=frames, frame_idx=np.arange(N, dtype=np.int64))
    assert npz_rows._stored_layout(path, "frames") is None
    for rows in ([4, 1, 4], [0], [-1]):
        _same(read_npz_rows(path, "frames", rows), np.ascontiguousarray(frames[rows]))


def test_the_stored_path_is_taken_and_leaves_no_mapping(tmp_path: Path) -> None:
    path, frames = _decode(tmp_path)
    offset, shape, dtype = npz_rows._stored_layout(path, "frames")    # type: ignore[misc]
    assert shape == (N, H, W, 3) and dtype == np.uint8
    raw = path.read_bytes()
    assert raw[offset:offset + frames.nbytes] == frames.tobytes()       # the region IS the array's bytes
    got = read_npz_rows(path, "frames", [1, 3])
    del got
    maps = Path("/proc/self/maps")
    if maps.is_file():
        assert str(path.resolve()) not in maps.read_text()


def test_a_truncated_member_is_refused(tmp_path: Path) -> None:
    path, _ = _decode(tmp_path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("frames.npy")
    info.file_size -= 1

    def short(self: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
        return info

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zipfile.ZipFile, "getinfo", short)
        with pytest.raises(ValueError, match="fewer than"):
            read_npz_rows(path, "frames", [0])


def test_other_members_and_dtypes_read_the_same(tmp_path: Path) -> None:
    path = tmp_path / "a.npz"
    joints = np.random.default_rng(0).normal(size=(6, 21, 3)).astype(np.float32)
    save_npz(path, frame_idx=np.arange(6, dtype=np.int64), joints=joints)
    _same(read_npz_rows(path, "joints", [5, 0, 5]), joints[[5, 0, 5]])
    _same(read_npz_rows(path, "frame_idx", [2]), np.array([2], dtype=np.int64))
    assert npz_array_shape(path, "joints") == (6, 21, 3)


def test_the_decode_stage_output_and_the_test_fakes_read_the_same(tmp_path: Path) -> None:
    """The decode worker end to end (ffmpeg testsrc) and `write_decode`'s plain `np.savez`, both on the row path."""
    video = tmp_path / "testsrc.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "30",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    make_decode_worker({"clip0": video}, stride=3)("clip0", tmp_path / "clip0")
    write_decode(tmp_path / "fake", random_frames(N, h=H, w=W, seed=4))
    for path in (tmp_path / "clip0" / DECODE_FILE, tmp_path / "fake" / DECODE_FILE):
        assert npz_rows._stored_layout(path, "frames") is not None
        n = npz_array_shape(path, "frames")[0]
        cases = [[0], [n - 1], [n - 1, 0, n // 2, 0], list(range(n))[::-1]]
        expected = [_full(path, rows) for rows in cases]
        with pytest.MonkeyPatch.context() as mp:
            forbid_full_frames(mp)
            for rows, want in zip(cases, expected, strict=True):
                _same(read_decode_rows(path, rows)[0], want)
