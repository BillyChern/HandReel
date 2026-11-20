import pytest
import torch

from fastdo.bench.gpu_budget import (
    CPU_ENV_VAR,
    DEFAULT_GB,
    ENV_VAR,
    apply_cpu_budget,
    apply_gpu_budget,
)


def test_opt_out_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "0")
    assert apply_gpu_budget() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_default_cap_is_at_most_12_gib_of_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    fraction = apply_gpu_budget()
    total = torch.cuda.get_device_properties(0).total_memory
    assert fraction is not None
    assert abs(fraction - min(1.0, DEFAULT_GB * 2**30 / total)) < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_explicit_cap_is_enforced_by_the_allocator() -> None:
    # The cap limits NEW allocations from the driver; blocks an earlier test left in the caching allocator would serve
    # this request without ever reaching it (after test_throughput_curve, ~9 GB reserved: DID NOT RAISE). Empty first.
    torch.cuda.empty_cache()
    apply_gpu_budget(gb=0.25)
    try:
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(int(0.5 * 2**30), dtype=torch.uint8, device="cuda")
    finally:
        apply_gpu_budget(gb=DEFAULT_GB)  # restore the default for the rest of the session


def test_cpu_budget_sets_torch_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CPU_ENV_VAR, "3")
    assert apply_cpu_budget() == 3
    assert torch.get_num_threads() == 3
    apply_cpu_budget(4)
