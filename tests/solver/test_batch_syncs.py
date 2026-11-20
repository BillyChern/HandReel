"""Host reads inside the batched LM loop, counted on CPU.

`test_batch.py::test_host_syncs_per_iteration` counts real CUDA synchronizations and needs the GPU. This proxy runs in
the CPU suite. Inside every `lm_optimize_batch` call it counts the operations that read a device value back to the
host when the tensors live on CUDA: aten scalar reads (`item`, `bool`, `float` and `int` all reach
`_local_scalar_dense`), `nonzero`, `masked_select`, `unique`, bool-mask indexing and `repeat_interleave` without
`output_size`, plus `Tensor.cpu()`, `to("cpu")` (the host named literally), `tolist()`, `numpy()` and `item()` on a
tensor that is not already a host copy. The loop's one designated read (the per-clip flags, `.cpu().tolist()`) counts
once. Run against BASE f8c8878 the same count gives 2.25 reads per iteration, the figure CUDA hooks measured there.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import weakref
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from fastdo.solver import api
from fastdo.solver.api import SolverConfig, solve_trajectories
from tests.solver.test_batch import _clip, _inputs, _unzip

logger = logging.getLogger(__name__)
_ATEN = torch.ops.aten
_READS = {_ATEN._local_scalar_dense.default, _ATEN.nonzero.default, _ATEN.masked_select.default,
          _ATEN._unique2.default, _ATEN.unique_consecutive.default, _ATEN.unique_dim.default}
_INDEX = {_ATEN.index.Tensor, _ATEN.index_put_.default, _ATEN.index_put.default}
_REPEAT = {_ATEN.repeat_interleave.Tensor, _ATEN.repeat_interleave.self_Tensor, _ATEN.repeat_interleave.self_int}


class HostReads(TorchDispatchMode):
    """Counts host reads (see the module docstring) while `counting()` is active."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: collections.Counter[str] = collections.Counter()
        self._host: dict[int, weakref.ref[Tensor]] = {}
        self.active = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        if self.active:
            if func in _READS:
                self.counts[str(func)] += 1
            elif func in _INDEX and any(isinstance(i, Tensor) and i.dtype == torch.bool for i in args[1] if i is not None):
                self.counts["bool index"] += 1
            elif func in _REPEAT and kwargs.get("output_size") is None and (len(args) < 4 or args[3] is None):
                self.counts[str(func)] += 1
        return func(*args, **kwargs)

    def is_host_copy(self, t: Tensor) -> bool:
        ref = self._host.get(id(t))
        return ref is not None and ref() is t

    def mark_host_copy(self, t: Tensor) -> None:
        self._host[id(t)] = weakref.ref(t)

    @contextlib.contextmanager
    def counting(self) -> Iterator[HostReads]:
        self.counts.clear()
        self._host.clear()
        self.active = True
        try:
            with self:
                yield self
        finally:
            self.active = False


def _to_cpu(args: tuple[object, ...], kwargs: dict[str, object]) -> bool:
    """`to("cpu", ...)` with the host named literally. A `to(device)` whose `torch.device` is the solver's own device
    moves nothing on CUDA, and on this CPU run it cannot be told apart from a host copy, so it is not counted."""
    return any(isinstance(x, str) and x.startswith("cpu") for x in (*args, kwargs.get("device")))


def _hook_methods(monkeypatch: pytest.MonkeyPatch, reads: HostReads) -> None:
    for name in ("cpu", "to", "tolist", "numpy", "item"):
        original = getattr(Tensor, name)

        def hooked(self: Tensor, *args: object, _name: str = name, _original=original, **kwargs: object) -> object:
            copies = _name == "cpu" or (_name == "to" and _to_cpu(args, kwargs))
            if reads.active and (copies or _name in ("tolist", "numpy", "item")) and not reads.is_host_copy(self):
                reads.counts[f"Tensor.{_name}"] += 1
            out = _original(self, *args, **kwargs)
            if reads.active and copies and isinstance(out, Tensor):
                reads.mark_host_copy(out)
            return out

        monkeypatch.setattr(Tensor, name, hooked)


def test_proxy_sees_scalar_reads_and_masked_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = HostReads()
    _hook_methods(monkeypatch, reads)
    x = torch.linspace(-1.0, 1.0, 5)
    with reads.counting():
        assert bool((x > 0).any())
        _ = x[x > 0]
        flags = torch.stack([x > 0, x < 0]).cpu().tolist()
    assert len(flags) == 2
    assert reads.counts[str(_ATEN._local_scalar_dense.default)] == 1 and reads.counts["bool index"] == 1
    assert reads.counts["Tensor.cpu"] == 1 and reads.counts["Tensor.tolist"] == 0, reads.counts


def test_host_reads_per_iteration_cpu_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every `lm_optimize_batch` call of a 2-clip solve reads the device back at most once per loop iteration."""
    reads = HostReads()
    _hook_methods(monkeypatch, reads)
    records: list[tuple[int, int, dict[str, int]]] = []
    real_lm = api.lm_optimize_batch

    def counted_lm(*args, **kwargs):  # type: ignore[no-untyped-def]
        with reads.counting():
            res = real_lm(*args, **kwargs)
        records.append((res.loop_iterations, sum(reads.counts.values()), dict(reads.counts)))
        return res

    monkeypatch.setattr(api, "lm_optimize_batch", counted_lm)
    inputs = [_inputs(_clip(T, 256, seed=100 + T), seed=T) for T in (9, 12)]
    solve_trajectories(*_unzip(inputs), SolverConfig(n_obs=128, n_mesh=256, device="cpu"))
    logger.info("(loop iterations, host reads, by kind) per level: %s", records)
    assert len(records) == 3
    for iterations, n_reads, kinds in records:
        assert n_reads <= iterations, records
        assert set(kinds) <= {"Tensor.cpu"}, records
