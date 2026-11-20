"""The opt-in CUDA-graph replay of SAM 3D's structured-latent flow transformer blocks.

CPU tests: the `sam3d_cuda_graph_slat` flag from `ShapeConfig` to the worker's `reconstruct_geometry` and its reply,
the runner's signature keying and eager fallback, the `blocks` shadowing and the attention replacement restored by
identity, and the device `cu_seqlens` cache checked against the original expression. They use stand-ins for SAM 3D's
sparse modules; the same checks against the REAL `full_attn` and `ModulatedSparseTransformerCrossBlock` stack were run
once, ad hoc, on the CPU in the sam3d env (it has no pytest). The CUDA tests (skipped without a GPU) check replay against the eager stack
bit for bit: a stand-in stack anywhere, and a small real `SLatFlowModel` when SAM 3D, spconv and flash_attn import
(the sam3d env). The real SAM 3D bit-identity gate is a GPU study.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fastdo.shape import ShapeConfig
from fastdo.shape import sam3d_cuda_graph_slat as slat_mod
from fastdo.shape import sam3d_worker as worker_mod
from fastdo.shape.backend import non_default_shape_options
from fastdo.shape.sam3d_backend import Sam3dBackend
from fastdo.shape.sam3d_worker import (
    reconstruct_geometry,
    request_params_digest,
    resolve_shape_options,
    serve,
)
from tests.shape.fake_sam3d import FakeSam3dPipeline

BASIC_MODULE = "sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic"


def _params(**overrides: Any) -> dict[str, Any]:
    return {"seed": 42, "sparse_structure_steps": 25, "structured_latent_steps": 25, "solver": "euler",
            "mesh_postprocess": True} | overrides


# --- the flag ------------------------------------------------------------------------------------------


def test_the_flag_is_off_by_default_and_a_default_request_is_unchanged_on_the_wire() -> None:
    cfg = ShapeConfig(backend="sam3d")
    assert cfg.sam3d_cuda_graph_slat is False
    assert "sam3d_cuda_graph_slat" not in Sam3dBackend(cfg).sampling_params
    assert non_default_shape_options(cfg) == {}
    assert resolve_shape_options(_params()).cuda_graph_slat is False
    assert worker_mod._call_kwargs(_params(), resolve_shape_options(_params()), None, None, None)[
        "cuda_graph_slat"] is False


def test_the_flag_reaches_the_request_and_is_recorded_when_set() -> None:
    cfg = ShapeConfig(backend="sam3d", sam3d_cuda_graph_slat=True)
    params = Sam3dBackend(cfg).sampling_params
    assert params["sam3d_cuda_graph_slat"] is True and "sam3d_cuda_graph_ss" not in params
    options = resolve_shape_options(params)
    assert options.cuda_graph_slat is True and options.cuda_graph_ss is False
    assert worker_mod._call_kwargs(params, options, None, None, None)["cuda_graph_slat"] is True
    assert non_default_shape_options(cfg) == {"sam3d_cuda_graph_slat": True}


def test_the_flag_does_not_move_the_params_digest() -> None:
    assert request_params_digest(_params(sam3d_cuda_graph_slat=True)) == request_params_digest(_params())


@pytest.mark.parametrize("value", [1, "true", None])
def test_a_non_bool_flag_is_refused_with_the_same_message_at_both_boundaries(value: Any) -> None:
    with pytest.raises(ValueError) as config_error:
        ShapeConfig(backend="sam3d", sam3d_cuda_graph_slat=value)
    with pytest.raises(ValueError) as worker_error:
        resolve_shape_options(_params(sam3d_cuda_graph_slat=value))
    assert str(worker_error.value) == str(config_error.value)
    assert "sam3d_cuda_graph_slat" in str(worker_error.value)


def test_the_flag_is_refused_off_sam3d() -> None:
    with pytest.raises(ValueError, match="sam3d_cuda_graph_slat"):
        ShapeConfig(backend="trellis", sam3d_cuda_graph_slat=True)
    ShapeConfig(backend="trellis", sam3d_cuda_graph_slat=False)


def test_the_hydra_default_is_the_promoted_value() -> None:
    """Promoted after 8/8 objects were bit-identical on the GPU gate; the ShapeConfig field itself stays False."""
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2] / "run" / "conf" / "config.yaml"
    assert OmegaConf.load(root).bench.shape.sam3d_cuda_graph_slat is True


# --- the graph path is entered only when enabled -------------------------------------------------------


class _RecordingGraph:
    def __init__(self, stats: dict[str, Any]) -> None:
        self.entered: list[Any] = []
        self.active = False
        self.stats = stats

    @contextmanager
    def __call__(self, generator: Any) -> Any:
        self.entered.append(generator)
        self.active = True
        try:
            yield SimpleNamespace(take_stats=lambda: dict(self.stats))
        finally:
            self.active = False


_SS_STATS = {"captures": 1, "capture_seconds": 0.5, "replays": 38, "eager_calls": 0}
_SLAT_STATS = {"captures": 1, "capture_seconds": 0.7, "replays": 75, "eager_calls": 0, "release_seconds": 0.01}


def _reconstruct(pipe: FakeSam3dPipeline, **overrides: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    image = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:15, 8:22] = True
    params: dict[str, Any] = {"seed": 0, "sparse_structure_steps": 25, "structured_latent_steps": 25,
                              "solver": "euler", "mesh_postprocess": False} | overrides
    return reconstruct_geometry(pipe, image, mask, **params)


@pytest.mark.parametrize(("ss", "slat"), [(False, False), (False, True), (True, True), (True, False)])
def test_the_slat_graph_wraps_exactly_the_structured_latent_flow_when_enabled(monkeypatch: pytest.MonkeyPatch,
                                                                             ss: bool, slat: bool) -> None:
    ss_recorder, slat_recorder = _RecordingGraph(_SS_STATS), _RecordingGraph(_SLAT_STATS)
    monkeypatch.setattr(worker_mod.sam3d_cuda_graph, "graphed_ss_backbone", ss_recorder)
    monkeypatch.setattr(worker_mod.sam3d_cuda_graph_slat, "graphed_slat_blocks", slat_recorder)
    pipe = FakeSam3dPipeline()
    seen: dict[str, tuple[bool, bool]] = {}
    for name in ("sample_sparse_structure", "sample_slat", "decode_slat"):
        original = getattr(pipe, name)

        def spy(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            seen[_name] = (ss_recorder.active, slat_recorder.active)
            return _original(*args, **kwargs)

        setattr(pipe, name, spy)
    graph_stats: dict[str, Any] = {}
    _reconstruct(pipe, cuda_graph_ss=ss, cuda_graph_slat=slat, graph_stats=graph_stats)
    assert slat_recorder.entered == ([pipe.models["slat_generator"]] if slat else [])
    assert seen == {"sample_sparse_structure": (ss, False), "sample_slat": (False, slat), "decode_slat": (False, False)}
    expected = (dict(_SS_STATS) if ss else {}) | ({f"slat_{k}": v for k, v in _SLAT_STATS.items()} if slat else {})
    assert graph_stats == expected


@pytest.mark.parametrize(("ss", "slat"), [(False, False), (False, True), (True, True), (True, False)])
def test_the_worker_replies_with_each_runners_counters_only_when_asked(monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path, ss: bool, slat: bool) -> None:
    calls: list[dict[str, Any]] = []

    def fake_reconstruct(pipeline: Any, image: Any, mask: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs["graph_stats"] is not None:
            if kwargs["cuda_graph_ss"]:
                kwargs["graph_stats"].update(captures=0, replays=38)
            if kwargs["cuda_graph_slat"]:
                kwargs["graph_stats"].update(slat_captures=1, slat_replays=75, slat_eager_calls=0)
        return np.zeros((3, 3)), np.zeros((1, 3), np.int64), {"structured_latent_flow": 1.0}

    monkeypatch.setattr(worker_mod, "reconstruct_geometry", fake_reconstruct)
    request_npz = tmp_path / "in.npz"
    np.savez(request_npz, image=np.zeros((4, 4, 3), np.uint8), mask=np.ones((4, 4), bool))
    params = _params(**({"sam3d_cuda_graph_ss": True} if ss else {}), **({"sam3d_cuda_graph_slat": True} if slat else {}))
    line = json.dumps({"cmd": "reconstruct", "id": 1, "input": str(request_npz),
                       "output": str(tmp_path / "out.npz"), "params": params})
    replies = io.StringIO()
    assert serve(object(), io.StringIO(line + "\n"), replies) == 0
    reply = json.loads(replies.getvalue())
    assert reply["ok"] and calls[0]["cuda_graph_slat"] is slat and calls[0]["cuda_graph_ss"] is ss
    assert (calls[0]["graph_stats"] is None) is (not ss and not slat)
    assert reply.get("cuda_graph_ss") == ({"captures": 0, "replays": 38} if ss else None)
    assert reply.get("cuda_graph_slat") == ({"captures": 1, "replays": 75, "eager_calls": 0} if slat else None)


# --- stand-ins for SAM 3D's sparse modules -------------------------------------------------------------


class _Data:
    """A backend data object: the features plus state `replace` carries over by identity."""

    def __init__(self, features: torch.Tensor, indices: torch.Tensor, spatial_shape: list[int],
                 indice_dict: dict[str, Any]) -> None:
        self._features, self.indices, self.spatial_shape, self.indice_dict = (features, indices, spatial_shape,
                                                                               indice_dict)


class FakeSparseTensor:
    __module__ = BASIC_MODULE           # `signature` reads the sparse DEBUG flag from the class's module

    def __init__(self, data: _Data, shape: torch.Size, layout: list[slice], scale: tuple[int, ...] = (1, 1, 1),
                 spatial_cache: dict[str, Any] | None = None) -> None:
        self.data, self._shape, self._layout, self._scale = data, shape, layout, scale
        self._spatial_cache = {} if spatial_cache is None else spatial_cache

    @classmethod
    def make(cls, feats: torch.Tensor) -> FakeSparseTensor:
        n = feats.shape[0]
        data = _Data(feats, torch.zeros(n, 4, dtype=torch.int32, device=feats.device), [8, 8, 8], {})
        return cls(data, torch.Size([1, *feats.shape[1:]]), [slice(0, n)])

    @property
    def feats(self) -> torch.Tensor:
        return self.data._features

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def layout(self) -> list[slice]:
        return self._layout

    @property
    def device(self) -> torch.device:
        return self.feats.device

    def replace(self, feats: torch.Tensor) -> FakeSparseTensor:
        data = _Data(feats, self.data.indices, list(self.data.spatial_shape), self.data.indice_dict)
        return FakeSparseTensor(data, torch.Size([self._shape[0], *feats.shape[1:]]), self._layout, self._scale,
                                self._spatial_cache)


def _flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_q: torch.Tensor) -> torch.Tensor:
    """One-segment varlen attention without host syncs; `cu_q` is read on the device."""
    out = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
    return out.transpose(0, 1) + cu_q[-1:].to(q.dtype) * 0


def _original_attention(full_attn: ModuleType) -> Any:
    """The flash_attn path of SAM 3D's `sparse_scaled_dot_product_attention`, CPU-built cu_seqlens included."""

    def sparse_scaled_dot_product_attention(*args: Any) -> Any:
        s = args[0]
        q_seqlen = [s.layout[i].stop - s.layout[i].start for i in range(s.shape[0])]
        cu_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(s.device)
        if len(args) == 1:
            out = full_attn.flash_attn.flash_attn_varlen_qkvpacked_func(s.feats, cu_q, max(q_seqlen))
        elif len(args) == 2:
            n, length, _, heads, channels = args[1].shape
            kv_seqlen = [length] * n
            cu_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(s.device)
            out = full_attn.flash_attn.flash_attn_varlen_kvpacked_func(
                s.feats, args[1].reshape(n * length, 2, heads, channels), cu_q, cu_kv, max(q_seqlen), max(kv_seqlen))
        else:
            out = args[0].feats + args[1].feats + args[2].feats                 # a form the replacement leaves alone
        return s.replace(out)

    return sparse_scaled_dot_product_attention


@pytest.fixture
def fake_sparse(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stand-ins for `full_attn`, `attention.modules` and `basic`, registered under SAM 3D's module names."""
    full_attn = ModuleType(slat_mod.FULL_ATTN_MODULE)
    attention = ModuleType(slat_mod.ATTENTION_MODULE)
    basic = ModuleType(BASIC_MODULE)
    full_attn.ATTN, full_attn.DEBUG, basic.DEBUG = "flash_attn", False, False  # type: ignore[attr-defined]
    full_attn.SparseTensor = FakeSparseTensor  # type: ignore[attr-defined]
    full_attn.flash_attn = SimpleNamespace(  # type: ignore[attr-defined]
        flash_attn_varlen_qkvpacked_func=lambda qkv, cu, m: _flash(*qkv.unbind(1), cu),
        flash_attn_varlen_kvpacked_func=lambda q, kv, cu_q, cu_kv, mq, mk: _flash(q, *kv.unbind(1), cu_q))
    original = _original_attention(full_attn)
    full_attn.sparse_scaled_dot_product_attention = original  # type: ignore[attr-defined]
    attention.sparse_scaled_dot_product_attention = original  # type: ignore[attr-defined]
    for module in (full_attn, attention, basic):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(full_attn=full_attn, attention=attention, basic=basic, original=original)


class _Block(torch.nn.Module):
    """Self- then cross-attention through the module-level attention function, as SAM 3D's block resolves it."""

    def __init__(self, channels: int = 16, heads: int = 2) -> None:
        super().__init__()
        self.heads, self.use_checkpoint = heads, False
        self.self_attn = SimpleNamespace(attn_mode="full", use_rope=False)
        self.cross_attn = SimpleNamespace(attn_mode="full", use_rope=False)
        self.mod, self.qkv, self.q = (torch.nn.Linear(channels, channels), torch.nn.Linear(channels, 3 * channels),
                                      torch.nn.Linear(channels, channels))
        self.kv, self.out = torch.nn.Linear(channels, 2 * channels), torch.nn.Linear(channels, channels)

    def forward(self, h: FakeSparseTensor, t_emb: torch.Tensor, cond: torch.Tensor) -> FakeSparseTensor:
        attend = sys.modules[slat_mod.ATTENTION_MODULE].sparse_scaled_dot_product_attention
        x = h.replace(h.feats + self.mod(t_emb))
        a = attend(x.replace(self.qkv(x.feats).reshape(x.feats.shape[0], 3, self.heads, -1)))
        x = x.replace(x.feats + self.out(a.feats.reshape(a.feats.shape[0], -1)))
        kv = self.kv(cond).reshape(cond.shape[0], cond.shape[1], 2, self.heads, -1)
        c = attend(x.replace(self.q(x.feats).reshape(x.feats.shape[0], self.heads, -1)), kv)
        return x.replace(x.feats + c.feats.reshape(c.feats.shape[0], -1))


class _Model(torch.nn.Module):
    def __init__(self, n_blocks: int = 3) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(n_blocks)])

    def forward(self, h: Any, t_emb: torch.Tensor, cond: torch.Tensor) -> Any:
        for block in self.blocks:                   # SLatFlowModel.forward's loop
            h = block(h, t_emb, cond)
        return h


def _generator(model: torch.nn.Module) -> Any:
    return SimpleNamespace(reverse_fn=SimpleNamespace(backbone=model))


def _inputs(n: int = 7, length: int = 5, device: str = "cpu", seed: int = 0) -> tuple[Any, torch.Tensor,
                                                                                         torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (FakeSparseTensor.make(torch.randn(n, 16, generator=g).to(device)),
            torch.randn(1, 16, generator=g).to(device), torch.randn(1, length, 16, generator=g).to(device))


# --- the runner: shadowing, eager fallback, keying -----------------------------------------------------


def test_uncapturable_calls_run_eager_and_the_blocks_are_restored(fake_sparse: SimpleNamespace) -> None:
    model = _Model()
    blocks, keys = model.blocks, list(model.state_dict())
    h, t_emb, cond = _inputs()
    with torch.no_grad():
        expected = model(h, t_emb, cond)
        with slat_mod.graphed_slat_blocks(_generator(model)) as runner:
            assert model.blocks is runner and model._modules["blocks"] is blocks
            assert list(model.state_dict()) == keys and list(model.parameters()) == list(blocks.parameters())
            out = model(h, t_emb, cond)                         # CPU tensors: never captured
            out_cfg = model(h, t_emb, cond * 0)
            stats = runner.take_stats()
    assert "blocks" not in vars(model) and model.blocks is blocks
    assert stats == {"captures": 0, "capture_seconds": 0.0, "replays": 0, "eager_calls": 2, "release_seconds": 0.0}
    assert torch.equal(out.feats, expected.feats) and torch.equal(out_cfg.feats, model(h, t_emb, cond * 0).feats)
    assert runner.graphs == {} and runner.cu_seqlens == {}
    assert fake_sparse.attention.sparse_scaled_dot_product_attention is fake_sparse.original   # never installed


def test_nesting_is_refused_and_the_blocks_survive_it() -> None:
    model = _Model()
    blocks = model.blocks
    with slat_mod.graphed_slat_blocks(_generator(model)), pytest.raises(RuntimeError, match="already overridden"), \
            slat_mod.graphed_slat_blocks(_generator(model)):
        pass
    assert "blocks" not in vars(model) and model.blocks is blocks


def test_a_warmup_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="warmup"):
        slat_mod.GraphedSlatBlocks(_Model().blocks, warmup=0)


def test_close_without_a_capture_does_not_touch_the_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> None:
        raise AssertionError("empty_cache without a capture")

    monkeypatch.setattr(torch.cuda, "empty_cache", refuse)
    runner = slat_mod.GraphedSlatBlocks(_Model().blocks)
    runner.graphs["eager signature"] = None
    runner.close()
    assert runner.graphs == {} and runner.stats["release_seconds"] == 0.0


@pytest.fixture
def cpu_signatures(monkeypatch: pytest.MonkeyPatch, fake_sparse: SimpleNamespace) -> SimpleNamespace:
    """Keying on CPU tensors: `_tensor_sig` without its CUDA and alignment conditions, a settable autocast state."""
    autocast = {"state": (False, torch.float16)}
    monkeypatch.setattr(slat_mod, "_tensor_sig", lambda x: None if not isinstance(x, torch.Tensor) or x.requires_grad
                        else (tuple(x.shape), tuple(x.stride()), x.dtype, x.device))
    monkeypatch.setattr(slat_mod, "_autocast_state", lambda: autocast["state"])
    return SimpleNamespace(autocast=autocast, **vars(fake_sparse))


def test_the_signature_keys_on_tokens_layout_shapes_and_autocast(cpu_signatures: SimpleNamespace) -> None:
    runner = slat_mod.GraphedSlatBlocks(_Model().blocks)
    with torch.no_grad():
        h, t_emb, cond = _inputs(seed=0)
        key = runner.signature(h, t_emb, cond)
        assert key is not None
        h2, t2, c2 = _inputs(seed=1)
        assert runner.signature(h2, t2, c2 * 0) == key                  # values and the CFG zeros do not matter
        assert runner.signature(*_inputs(n=9)) != key                   # another object's token count
        assert runner.signature(*_inputs(length=6)) != key              # another condition length
        moved = h.replace(h.feats)
        moved._layout = [slice(1, 8)]
        assert runner.signature(moved, t_emb, cond) != key              # the layout sets cu_seqlens
        assert runner.signature(h, t_emb.double(), cond) != key         # dtype
        assert runner.signature(h, t_emb, cond.transpose(1, 2).contiguous().transpose(1, 2)) != key   # stride
        cpu_signatures.autocast["state"] = (True, torch.float16)
        assert runner.signature(h, t_emb, cond) != key
    assert runner.signature(h, t_emb, cond) is None                     # grad mode


@pytest.mark.parametrize("change", ["sdpa", "full_attn_debug", "basic_debug", "not_sparse", "rope", "checkpoint",
                                    "windowed"])
def test_uncapturable_configurations_have_no_signature(cpu_signatures: SimpleNamespace, change: str) -> None:
    model = _Model()
    h, t_emb, cond = _inputs()
    if change == "sdpa":
        cpu_signatures.full_attn.ATTN = "sdpa"
    elif change == "full_attn_debug":
        cpu_signatures.full_attn.DEBUG = True
    elif change == "basic_debug":
        cpu_signatures.basic.DEBUG = True
    elif change == "not_sparse":
        h = SimpleNamespace(feats=h.feats, layout=h.layout, shape=h.shape)
    elif change == "rope":
        model.blocks[1].self_attn.use_rope = True
    elif change == "checkpoint":
        model.blocks[2].use_checkpoint = True
    else:
        model.blocks[0].self_attn.attn_mode = "windowed"
    runner = slat_mod.GraphedSlatBlocks(model.blocks)
    with torch.no_grad():
        assert runner.signature(h, t_emb, cond) is None


def test_a_real_cpu_tensor_is_never_capturable() -> None:
    assert slat_mod._tensor_sig(torch.zeros(4, 16)) is None
    assert slat_mod._tensor_sig("not a tensor") is None


# --- the device cu_seqlens attention ------------------------------------------------------------------


def test_the_replacement_equals_the_original_and_caches_the_original_cu_seqlens(fake_sparse: SimpleNamespace) -> None:
    h, _, _ = _inputs(n=7, length=5)
    qkv = h.replace(torch.randn(7, 3, 2, 8))
    q = h.replace(torch.randn(7, 2, 8))
    kv = torch.randn(1, 5, 2, 2, 8)
    cache: dict[Any, torch.Tensor] = {}
    unhandled: list[str] = []
    with slat_mod.device_cu_seqlens_attention(cache, unhandled):
        patched = fake_sparse.attention.sparse_scaled_dot_product_attention
        assert patched is not fake_sparse.original and getattr(patched, slat_mod.PATCH_SENTINEL)
        for args in ((qkv,), (q, kv)):
            out, expected = patched(*args), fake_sparse.original(*args)
            assert torch.equal(out.feats, expected.feats) and out.shape == expected.shape
            assert out.layout is expected.layout
    assert fake_sparse.attention.sparse_scaled_dot_product_attention is fake_sparse.original   # by identity
    assert set(cache) == {((7,), torch.device("cpu")), ((5,), torch.device("cpu"))}
    assert cache[((7,), torch.device("cpu"))].tolist() == [0, 7] and cache[((5,), torch.device("cpu"))].tolist() == [0, 5]
    assert all(value.dtype == torch.int32 for value in cache.values())
    assert unhandled == []


def test_a_cached_cu_seqlens_that_differs_from_the_original_raises(fake_sparse: SimpleNamespace) -> None:
    h, _, _ = _inputs(n=7)
    qkv = h.replace(torch.randn(7, 3, 2, 8))
    cache = {((7,), torch.device("cpu")): torch.tensor([0, 6], dtype=torch.int32)}
    with slat_mod.device_cu_seqlens_attention(cache, []), pytest.raises(RuntimeError, match="cu_seqlens differ"):
        fake_sparse.attention.sparse_scaled_dot_product_attention(qkv)
    assert fake_sparse.attention.sparse_scaled_dot_product_attention is fake_sparse.original   # restored on raise


def test_a_replacement_that_differs_from_the_original_raises(fake_sparse: SimpleNamespace) -> None:
    h, _, _ = _inputs(n=7)
    qkv = h.replace(torch.randn(7, 3, 2, 8))
    flash = fake_sparse.full_attn.flash_attn
    calls = {"n": 0}

    def drifting(qkv: torch.Tensor, cu: torch.Tensor, m: int) -> torch.Tensor:  # differs on every other call
        calls["n"] += 1
        return _flash(*qkv.unbind(1), cu) + calls["n"] % 2

    fake_sparse.full_attn.flash_attn = SimpleNamespace(flash_attn_varlen_qkvpacked_func=drifting,
                                                       flash_attn_varlen_kvpacked_func=flash.flash_attn_varlen_kvpacked_func)
    with slat_mod.device_cu_seqlens_attention({}, []), pytest.raises(RuntimeError, match="differs from SAM 3D"):
        fake_sparse.attention.sparse_scaled_dot_product_attention(qkv)


def test_unhandled_call_forms_run_the_original_and_are_reported(fake_sparse: SimpleNamespace) -> None:
    h, _, _ = _inputs(n=7)
    q, k, v = (h.replace(torch.randn(7, 2, 8)) for _ in range(3))
    unhandled: list[str] = []
    with slat_mod.device_cu_seqlens_attention({}, unhandled):
        out = fake_sparse.attention.sparse_scaled_dot_product_attention(q, k, v)
        fake_sparse.full_attn.ATTN = "xformers"
        fake_sparse.attention.sparse_scaled_dot_product_attention(q.replace(torch.randn(7, 3, 2, 8)))
    assert torch.equal(out.feats, q.feats + k.feats + v.feats)
    assert len(unhandled) == 2 and "3 args" in unhandled[0] and "xformers" in unhandled[1]


def test_the_attention_replacement_does_not_nest(fake_sparse: SimpleNamespace) -> None:
    with slat_mod.device_cu_seqlens_attention({}, []), pytest.raises(RuntimeError, match="already installed"), \
            slat_mod.device_cu_seqlens_attention({}, []):
        pass
    assert fake_sparse.attention.sparse_scaled_dot_product_attention is fake_sparse.original


# --- the rebuilt output ----------------------------------------------------------------------------


def test_the_output_check_accepts_exactly_a_replace_of_the_input(fake_sparse: SimpleNamespace) -> None:
    h, t_emb, cond = _inputs()
    with torch.no_grad():
        out = _Model()(h, t_emb, cond)
    assert slat_mod._is_replace_of(out, h)
    copied = out.replace(out.feats)
    copied._layout = list(h.layout)                     # plain values (slices, sizes) need only be equal ...
    assert slat_mod._is_replace_of(copied, h)
    relaid = out.replace(out.feats)
    relaid._layout = [slice(0, 6)]
    assert not slat_mod._is_replace_of(relaid, h)
    reindexed = out.replace(out.feats)                  # ... tensors and state dicts must be the same objects
    reindexed.data.indices = h.data.indices.clone()
    assert not slat_mod._is_replace_of(reindexed, h)
    recached = out.replace(out.feats)
    recached._spatial_cache = {}
    assert not slat_mod._is_replace_of(recached, h)
    rescaled = out.replace(out.feats)
    rescaled._scale = (2, 2, 2)
    assert not slat_mod._is_replace_of(rescaled, h)
    assert not slat_mod._is_replace_of(SimpleNamespace(feats=out.feats), h)


# --- CUDA: replay equals the eager stack bit for bit ---------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_replay_equals_the_eager_stack_bit_for_bit(fake_sparse: SimpleNamespace) -> None:
    model = _Model().cuda().half()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        with slat_mod.graphed_slat_blocks(_generator(model)) as runner:
            for step in range(4):
                for n in (7, 7, 11):                                    # a second signature: a second capture
                    h, t_emb, cond = _inputs(n=n, device="cuda", seed=10 * step + n)
                    h, t_emb, cond = h.replace(h.feats.half()), t_emb.half(), cond.half()
                    for c in (cond, cond * 0):
                        runner_out = model(h, t_emb, c)
                        expected = runner.eager(h, t_emb, c)
                        assert torch.equal(runner_out.feats, expected.feats)
                        assert slat_mod._is_replace_of(runner_out, h)
            stats = runner.take_stats()
            assert fake_sparse.attention.sparse_scaled_dot_product_attention is fake_sparse.original
        assert runner.graphs == {} and runner.cu_seqlens == {}
    assert stats["captures"] == 2 and stats["eager_calls"] == 0 and stats["replays"] == 24
    assert "blocks" not in vars(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_slat_flow_model_replay_equals_eager_bit_for_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small real `SLatFlowModel` (spconv IO blocks, 2 real transformer blocks, flash_attn): eager vs graphed forward.
    Runs where SAM 3D, spconv and flash_attn import (the sam3d env with the worker's PYTHONPATH)."""
    flash_attn = pytest.importorskip("flash_attn")
    slf = pytest.importorskip("sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow")
    import importlib

    sp = importlib.import_module("sam3d_objects.model.backbone.tdfy_dit.modules.sparse")
    full_attn = importlib.import_module(slat_mod.FULL_ATTN_MODULE)
    monkeypatch.setattr(full_attn, "ATTN", "flash_attn")
    monkeypatch.setattr(full_attn, "flash_attn", flash_attn, raising=False)
    torch.manual_seed(0)
    model = slf.SLatFlowModel(resolution=16, in_channels=8, model_channels=128, cond_channels=64, out_channels=8,
                              num_blocks=2, num_heads=2, io_block_channels=[32], qk_rms_norm=True, pe_mode="ape",
                              use_fp16=True)
    with torch.no_grad():
        for p in model.parameters():                   # initialize_weights zeroes the output layer and adaLN
            p.normal_(0.0, 0.05)
    model = model.cuda().eval()
    coords = torch.unique(torch.cat([torch.zeros(300, 1, dtype=torch.int32),
                                     torch.randint(0, 16, (300, 3), dtype=torch.int32)], 1), dim=0).cuda()
    x = sp.SparseTensor(feats=torch.randn(coords.shape[0], 8, device="cuda"), coords=coords)
    cond = torch.randn(1, 12, 64, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        eager = [model(x, torch.tensor([t], device="cuda"), c).feats for t in (900.0, 500.0, 100.0)
                 for c in (cond, cond * 0)]
        with slat_mod.graphed_slat_blocks(_generator(model)) as runner:
            graphed = [model(x, torch.tensor([t], device="cuda"), c).feats for t in (900.0, 500.0, 100.0)
                       for c in (cond, cond * 0)]
            stats = runner.take_stats()
    assert stats["captures"] == 1 and stats["replays"] == 6 and stats["eager_calls"] == 0
    assert all(torch.equal(a, b) for a, b in zip(eager, graphed, strict=True))
