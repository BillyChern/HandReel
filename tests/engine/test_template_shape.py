"""Tests for the template shape worker: a metric template through the merged D4 shape path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import trimesh

from fastdo.engine.worker_support import save_npz
from fastdo.engine.workers_shape import (
    TEMPLATE_BACKEND,
    make_template_shape_worker,
    read_shape_meta,
)
from tests.engine.fakes import random_frames, write_decode


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    lines = [f"v {x!r} {y!r} {z!r}" for x, y, z in vertices.tolist()]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces.tolist()]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_template_shape_worker_keeps_the_metric_frame_recoverable(tmp_path: Path) -> None:
    box = trimesh.creation.box(extents=(0.12, 0.08, 0.04))
    box.apply_translation((0.01, -0.02, 0.03))
    raw = np.asarray(box.vertices, dtype=np.float64)
    template = _write_obj(tmp_path / "box.obj", raw, np.asarray(box.faces))
    clip = tmp_path / "engine" / "clip0"
    write_decode(clip, random_frames(3, 24, 32), stride=3)
    masks = np.zeros((3, 24, 32), dtype=bool)
    masks[:, 8:16, 10:22] = True
    save_npz(clip / "mask.npz", masks=masks, frame_idx=3 * np.arange(3, dtype=np.int64),
             keyframe_index=np.array(0, dtype=np.int64))

    make_template_shape_worker({"clip0": template}, sdf_resolution=16, n_samples=256)("clip0", clip)

    meta = read_shape_meta(clip)
    assert (meta["backend"], meta["backend_class"]) == (TEMPLATE_BACKEND, "TemplateMesh") == ("template", "TemplateMesh")
    assert meta["normalization_scale"] == pytest.approx(1.0 / math.sqrt(0.0224), rel=0.0, abs=1e-9)
    np.testing.assert_allclose(meta["normalization_center"], [0.01, -0.02, 0.03], rtol=0.0, atol=1e-9)
    assert meta["phase_seconds"].keys() == {"load"} and meta["mesh_postprocess"] is False
    mesh = cast("trimesh.Trimesh", trimesh.load(str(clip / "mesh.obj"), process=False, force="mesh"))
    expected = (raw - np.asarray(meta["normalization_center"])) * meta["normalization_scale"]
    np.testing.assert_allclose(np.asarray(mesh.vertices), expected, rtol=0.0, atol=1e-6)
    with pytest.raises(KeyError, match="clip1"):
        make_template_shape_worker({"clip0": template})("clip1", clip)
