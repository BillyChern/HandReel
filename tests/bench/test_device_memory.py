"""The NVML process-memory sampler (`fastdo.bench.device_memory`) on canned nvidia-smi output."""

from __future__ import annotations

import threading

from fastdo.bench.device_memory import ProcessMemorySampler, nvidia_smi_process_mib


def _rows(*pairs: tuple[int, int]) -> str:
    return "\n".join(f"{pid}, {mib}" for pid, mib in pairs)


def test_process_mib_sums_the_rows_of_one_pid() -> None:
    out = _rows((111, 500), (222, 9000), (111, 700))
    assert nvidia_smi_process_mib(111, lambda: out) == (1200, 2)        # a pid listed twice counts twice, as the guard
    assert nvidia_smi_process_mib(333, lambda: out) is None
    assert nvidia_smi_process_mib(111, lambda: "No running processes found") is None


def test_process_mib_without_nvidia_smi() -> None:
    def missing() -> str:
        raise FileNotFoundError("nvidia-smi")

    assert nvidia_smi_process_mib(1, missing) is None


def test_sampler_keeps_the_peak_and_calls_the_limit_once() -> None:
    readings = iter([100, 4000, 5000, 250])
    lock = threading.Lock()

    def query() -> str:
        with lock:
            return _rows((42, next(readings, 300)))

    hits: list[int] = []
    past_limit = threading.Event()

    def on_limit(mib: int) -> None:
        hits.append(mib)
        past_limit.set()

    with ProcessMemorySampler(interval_s=0.001, stop_mib=3000, on_limit=on_limit, query=query, pid=42) as sampler:
        assert past_limit.wait(5.0)
        while sampler.samples < 5:
            threading.Event().wait(0.001)
    assert hits == [4000] and sampler.peak_mib == 5000 and sampler.max_rows == 1


def test_sampler_without_rows_reports_none() -> None:
    with ProcessMemorySampler(interval_s=0.001, stop_mib=1, on_limit=lambda mib: None, query=lambda: "",
                              pid=7) as sampler:
        pass
    assert sampler.peak_mib is None and sampler.max_rows == 0 and sampler.samples >= 2
