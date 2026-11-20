"""F2 deterministic SAM 3D: the order-fixed segment mean, the runtime patch and
its scoping, the worker protocol, and the real-weights gate.

Only `test_real_sam3d_deterministic_gates` touches real weights. It skips without CUDA, `FASTDO_SAM3D_PYTHON`,
`FASTDO_GPU_MEM_GB >= 19` or `FASTDO_ENGINE_DET_INPUTS`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.shape import ShapeConfig
from fastdo.shape import sam3d_determinism as det
from fastdo.shape.sam3d_backend import WORKER_SCRIPT, Sam3dBackend, Sam3dWorkerClient
from fastdo.shape.sam3d_worker import SAM3D_MIN_GPU_GB, deterministic_sam3d, reconstruct_geometry
from fastdo.shape.trellis_backend import TrellisBackend
from tests.shape.fake_sam3d import FakeSam3dPipeline, det_patch_installed, make_fake_sam3d_modules

_REPO = Path(__file__).resolve().parents[2]
_PARAMS: dict[str, Any] = {"seed": 0, "sparse_structure_steps": 25, "structured_latent_steps": 25, "solver": "euler",
                           "mesh_postprocess": False}


def _scatter_mean(inp: torch.Tensor, index: torch.Tensor, src: torch.Tensor, include_self: bool) -> torch.Tensor:
    return torch.scatter_reduce(inp, 0, index.unsqueeze(1).expand(-1, src.shape[1]), src, reduce="mean",
                                include_self=include_self)


def _frame_and_mask() -> tuple[np.ndarray, np.ndarray]:
    image = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:15, 8:22] = True
    return image, mask


# --- the kernel ---------------------------------------------------------------------------------------


def test_segment_mean_matches_scatter_reduce_bitwise_fp32_fp64() -> None:
    m, n, c = 500, 3000, 7
    g = torch.Generator()
    for dtype in (torch.float32, torch.float64):
        for include_self in (True, False):
            for random_input in (False, True):
                for seed in range(10):
                    g.manual_seed(seed)
                    index = torch.randint(0, m, (n,), generator=g)
                    src = (torch.randn(n, c, generator=g) * 2.15).to(dtype)
                    inp = torch.randn(m, c, generator=g).to(dtype) if random_input else torch.zeros(m, c, dtype=dtype)
                    out = det.segment_mean(inp, index, src, include_self)
                    ref = _scatter_mean(inp, index, src, include_self)
                    assert out.numpy().tobytes() == ref.numpy().tobytes(), (dtype, include_self, random_input, seed)


def test_segment_mean_include_self_semantics() -> None:
    index, src = torch.tensor([0, 0, 1, 2]), torch.tensor([2.0, 4.0, 6.0, -0.0])
    assert det.segment_mean(torch.zeros(3), index, src, True).tolist() == [2.0, 3.0, 0.0]      # (0+2+4)/3
    assert det.segment_mean(torch.full((4,), 7.0), index, src, False).tolist() == [3.0, 6.0, 0.0, 7.0]  # empty keeps


def test_segment_mean_fp16_is_the_ascending_fold() -> None:
    m, n, c = 40, 300, 5
    g = torch.Generator().manual_seed(3)
    index = torch.randint(0, m, (n,), generator=g)
    src = (torch.randn(n, c, generator=g) * 50).to(torch.float16)
    counts = torch.bincount(index, minlength=m)
    k, src_max = int(counts.max()), float(src.abs().max())
    assert k * src_max < 65504                                    # the fold cannot overflow (Check L5)
    out = det.segment_mean(torch.zeros(m, c, dtype=torch.float16), index, src, False)

    acc = torch.zeros(m, c, dtype=torch.float16)
    for i in range(n):                                            # ascending original index, one add at a time
        acc[index[i]] = acc[index[i]] + src[i]
    expected = torch.where((counts > 0)[:, None], acc / counts.clamp_min(1).to(torch.float16)[:, None], acc)
    assert out.numpy().tobytes() == expected.numpy().tobytes()
    exact = torch.zeros(m, c, dtype=torch.float64).index_add_(0, index, src.double())
    exact = exact / counts.clamp_min(1).double()[:, None]
    assert float((out.double() - exact).abs().max()) <= k * 2**-10 * src_max


def test_segment_mean_is_order_fixed() -> None:
    g = torch.Generator().manual_seed(5)
    index = torch.randint(0, 64, (1000,), generator=g)
    src, inp = torch.randn(1000, 3, generator=g), torch.randn(64, 3, generator=g)
    outs = {det.segment_mean(inp, index, src, True).numpy().tobytes() for _ in range(50)}
    assert len(outs) == 1
    for order in (torch.argsort(index, stable=True), torch.argsort(-index, stable=True)):   # within-segment order kept
        assert det.segment_mean(inp, index[order], src[order], True).numpy().tobytes() in outs


def test_deterministic_scatter_reduce_rejects_unsupported() -> None:
    c = 3
    src, zeros = torch.randn(10, c), torch.zeros(4, c)
    index = torch.randint(0, 4, (10,)).unsqueeze(1).expand(-1, c)
    with pytest.raises(NotImplementedError, match="dim"):
        det.deterministic_scatter_reduce(zeros, 1, index, src, "mean")
    with pytest.raises(NotImplementedError, match="reduce"):
        det.deterministic_scatter_reduce(zeros, 0, index, src, "sum")
    with pytest.raises(NotImplementedError, match="expand"):
        det.deterministic_scatter_reduce(zeros, 0, index.contiguous(), src, "mean")
    with pytest.raises(NotImplementedError, match="CUDA"):
        det.deterministic_scatter_reduce(zeros, 0, index, src, "mean")
    one = torch.randint(0, 4, (10, 1))                            # C == 1: a contiguous column passes the expand guard
    with pytest.raises(NotImplementedError, match="CUDA"):
        det.deterministic_scatter_reduce(torch.zeros(4, 1), 0, one, torch.randn(10, 1), "mean")


# _reference_cubes_to_verts reproduces SAM 3D Objects (https://github.com/facebookresearch/
# sam-3d-objects, commit 875b0100, sam3d_objects/model/backbone/tdfy_dit/representations/mesh/
# utils_cube.py `cubes_to_verts`), Copyright (c) Meta Platforms, Inc. and affiliates., a SAM
# Material distributed under the SAM License; see licenses/SAM_LICENSE.
def _reference_cubes_to_verts(num_verts: int, cubes: torch.Tensor, value: torch.Tensor,
                              reduce: str = "mean") -> torch.Tensor:
    """utils_cube.cubes_to_verts at do-as-i-do 875b010, verbatim."""
    m = value.shape[2]
    reduced = torch.zeros(num_verts, m, device=cubes.device, dtype=value.dtype)
    return torch.scatter_reduce(reduced, 0, cubes.unsqueeze(-1).expand(-1, -1, m).flatten(0, 1), value.flatten(0, 1),
                                reduce=reduce, include_self=False)


def test_deterministic_cubes_to_verts_matches_reference() -> None:
    g = torch.Generator().manual_seed(7)
    cubes, value = torch.randint(0, 300, (200, 8), generator=g), torch.randn(200, 8, 4, generator=g)
    out = det.deterministic_cubes_to_verts(310, cubes, value)
    assert out.numpy().tobytes() == _reference_cubes_to_verts(310, cubes, value).numpy().tobytes()
    with pytest.raises(NotImplementedError, match="mean"):
        det.deterministic_cubes_to_verts(310, cubes, value, reduce="sum")


# --- install and uninstall ----------------------------------------------------------------------------


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    modules = make_fake_sam3d_modules()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


def _state(modules: dict[str, ModuleType]) -> tuple[Any, ...]:
    return (modules[det.SPATIAL_MODULE].SparseDownsample.forward, modules[det.UTILS_CUBE_MODULE].cubes_to_verts,
            modules[det.FLEXICUBES_MODULE].FlexiCubes.__call__, torch.scatter_reduce,
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


def test_install_uninstall_restores_identity(fake_modules: dict[str, ModuleType]) -> None:
    before = _state(fake_modules)
    with deterministic_sam3d():
        assert getattr(fake_modules[det.SPATIAL_MODULE].SparseDownsample.forward, det.PATCH_SENTINEL, False)
        assert fake_modules[det.UTILS_CUBE_MODULE].cubes_to_verts is det.deterministic_cubes_to_verts
        assert getattr(fake_modules[det.FLEXICUBES_MODULE].FlexiCubes.__call__, det.PATCH_SENTINEL, False)
    after = _state(fake_modules)
    assert all(a is b for a, b in zip(after[:4], before[:4], strict=True)) and after[4:] == before[4:]


def test_install_restores_on_exception(fake_modules: dict[str, ModuleType]) -> None:
    before = _state(fake_modules)
    with pytest.raises(ValueError, match="inside"), deterministic_sam3d():
        raise ValueError("inside")
    assert _state(fake_modules) == before


def test_double_install_raises(fake_modules: dict[str, ModuleType]) -> None:
    before = _state(fake_modules)
    with deterministic_sam3d():
        patched = _state(fake_modules)
        with pytest.raises(RuntimeError, match="already installed"), deterministic_sam3d():
            pass
        assert _state(fake_modules) == patched                    # the refused install changed nothing
    assert _state(fake_modules) == before


def _global_modes() -> dict[str, Any]:
    """Every process-global flag the scope touches: a leak from the P3 scope would slow later requests and stages."""
    import torch.utils.deterministic

    return {"enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


@pytest.mark.parametrize("start", ["default", "enabled_strict"])
@pytest.mark.parametrize("scope", ["deterministic_sam3d", "scoped_deterministic"])
def test_global_modes_restored_on_exit_and_on_exception(fake_modules: dict[str, ModuleType], scope: str,
                                                        start: str) -> None:
    flexi = fake_modules[det.FLEXICUBES_MODULE]
    context = deterministic_sam3d if scope == "deterministic_sam3d" else det.scoped_deterministic
    saved = _global_modes()
    try:
        if start == "enabled_strict":           # a caller's own non-default mode must come back, not the defaults
            torch.use_deterministic_algorithms(True, warn_only=False)
        before = _global_modes()

        def inside() -> None:                   # the P3 path: FlexiCubes flips the mode inside the context
            if scope == "deterministic_sam3d":
                flexi.FlexiCubes()(torch.zeros(8, 3), torch.zeros(8))
            else:
                flexi.SEEN.append(torch.are_deterministic_algorithms_enabled())

        with context():
            inside()
        assert flexi.SEEN[-1] is True and _global_modes() == before
        with pytest.raises(ValueError, match="inside"), context():
            inside()
            raise ValueError("inside")
        assert _global_modes() == before
        real_call = flexi.FlexiCubes.__call__

        def failing(self: Any, *args: Any, **kwargs: Any) -> Any:
            real_call(self, *args, **kwargs)
            raise ValueError("inside FlexiCubes")

        flexi.FlexiCubes.__call__ = failing     # raising inside P3's own scope, not only in the outer block
        try:
            with pytest.raises(ValueError, match="inside FlexiCubes"), context():
                flexi.FlexiCubes()(torch.zeros(8, 3), torch.zeros(8))
        finally:
            flexi.FlexiCubes.__call__ = real_call
        assert _global_modes() == before
        flexi.FlexiCubes()(torch.zeros(8, 3), torch.zeros(8))     # outside any context: the caller's mode
        assert flexi.SEEN[-1] is before["enabled"]
        if start == "default":
            assert flexi.SEEN[-1] is False
    finally:
        torch.use_deterministic_algorithms(saved["enabled"], warn_only=saved["warn_only"])
    assert _global_modes() == saved


def test_scatter_reduce_swapped_only_inside_downsample_forward(fake_modules: dict[str, ModuleType]) -> None:
    torch_orig, spatial, flexi = torch.scatter_reduce, fake_modules[det.SPATIAL_MODULE], fake_modules[det.FLEXICUBES_MODULE]
    with deterministic_sam3d():
        assert torch.scatter_reduce is torch_orig
        spatial.SparseDownsample().forward("feats")
        assert torch.scatter_reduce is torch_orig
        flexi.FlexiCubes()(torch.zeros(8, 3), torch.zeros(8))
    flexi.FlexiCubes()(torch.zeros(8, 3), torch.zeros(8))
    assert spatial.SEEN == [det.deterministic_scatter_reduce]
    assert flexi.SEEN == [True, torch.are_deterministic_algorithms_enabled()]   # the P3 flag only inside FlexiCubes
    assert torch.scatter_reduce is torch_orig


def test_reconstruct_geometry_scopes_patch_to_slat_and_decode(fake_modules: dict[str, ModuleType]) -> None:
    for deterministic in (True, False):
        pipe, post = FakeSam3dPipeline(), []

        def _post(vertices: np.ndarray, faces: np.ndarray, post: list[bool] = post,
                  **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            post.append(det_patch_installed())
            return vertices, faces

        image, mask = _frame_and_mask()
        reconstruct_geometry(pipe, image, mask, **(_PARAMS | {"mesh_postprocess": True}), postprocess_fn=_post,
                             deterministic=deterministic)
        assert {e["event"]: e["det"] for e in pipe.events} == {
            "pointmap": False, "preprocess": False, "sparse_structure": False, "structured_latent": deterministic,
            "decode": deterministic}
        assert post == [False]
        assert not det_patch_installed()


# --- the worker protocol and the config field ---------------------------------------------------------


def test_worker_protocol_passes_deterministic_and_reports_peak(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    env = dict(os.environ, FAKE_SAM3D_LOG=str(log), CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")]))
    command = [sys.executable, str(WORKER_SCRIPT), "--config", "fake.yaml", "--device", "cpu",
               "--factory", "tests.shape.fake_sam3d:make_fake_pipeline"]
    client = Sam3dWorkerClient(command, env=env, startup_timeout_s=120.0, request_timeout_s=120.0)
    image, mask = _frame_and_mask()
    try:
        client.reconstruct(image, mask, _PARAMS | {"deterministic": True})
        assert "peak_gpu_gib" in client.last_reply and client.last_reply["peak_gpu_gib"] is None
        assert client.last_peak_gpu_gib is None and client.last_peak_gpu_gib_reserved is None
        assert client.last_digests is None and "digests" not in client.last_reply   # digests are opt-in (Check M2)
        client.reconstruct(image, mask, _PARAMS | {"digest": True})              # no deterministic key: False
        digests = client.last_digests
    finally:
        assert client.close() == 0
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["det"] for e in events if e["event"] == "structured_latent"] == [True, False]
    # "params" is the fast-path options' request digest: the resolved parameters that can change the mesh.
    assert digests is not None and set(digests) == {"slat_feats", "flexicubes_input", "mesh_vertices", "mesh_faces",
                                                    "params"}
    assert all(isinstance(value, str) and len(value) == 64 for value in digests.values())


def test_shape_config_deterministic_field() -> None:
    assert ShapeConfig().deterministic is False                     # trellis: unpatched, so off by default
    assert ShapeConfig(backend="sam3d").deterministic is True       # the default flip after the F2.5 gates
    assert ShapeConfig(backend="template").deterministic is False
    with pytest.raises(ValueError, match="deterministic"):
        ShapeConfig(deterministic=1)  # type: ignore[arg-type]
    assert Sam3dBackend(ShapeConfig(backend="sam3d")).sampling_params["deterministic"] is True
    assert Sam3dBackend(ShapeConfig(backend="sam3d", deterministic=False)).sampling_params["deterministic"] is False
    with pytest.raises(ValueError, match="deterministic"):
        TrellisBackend(ShapeConfig(deterministic=True))


# --- real weights (GPU gate F2.5) ---------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SAM 3D Objects needs CUDA (and > 17 GiB of allocator memory)")
def test_real_sam3d_deterministic_gates() -> None:
    if not os.environ.get("FASTDO_SAM3D_PYTHON"):
        pytest.skip("FASTDO_SAM3D_PYTHON (the SAM 3D environment's interpreter) is not set")
    if float(os.environ.get("FASTDO_GPU_MEM_GB", "12")) < SAM3D_MIN_GPU_GB:
        pytest.skip("SAM 3D peaks at 18.67 GiB allocated; run with FASTDO_GPU_MEM_GB >= 19")
    inputs = os.environ.get("FASTDO_ENGINE_DET_INPUTS")
    if not inputs:
        pytest.skip("FASTDO_ENGINE_DET_INPUTS (the directory holding kf_135545.npz and kf_135841.npz) is not set")
    from tests.shape.sam3d_det_gate import run_f2_gates

    record = run_f2_gates(Path(inputs), ("135545", "135841"))
    log = os.environ.get("FASTDO_ENGINE_DET_LOG")
    if log:
        Path(log).write_text(json.dumps(record, indent=1))
    assert record["gates"] == {"G1": "pass", "G2": "pass", "G3": "pass"}, record["gates"]
