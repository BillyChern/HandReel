"""The lock-step scheduler interleaves clips round-robin, isolates failures, and keeps per-clip RNG streams."""

from __future__ import annotations

from fastdo.physics.lockstep import Lane, RngSlot, run_lockstep, wait_event


def _clip(name: str, n: int, log: list, fail_at: int | None = None):
    total = 0
    for i in range(n):
        if fail_at is not None and i == fail_at:
            raise ValueError(f"{name} failed")
        log.append((name, i))
        total += i
        yield
    return total


def test_round_robin_and_results() -> None:
    log: list = []
    seen: list = []
    lanes = [Lane("a", _clip("a", 3, log), streams="sa"), Lane("b", _clip("b", 1, log), streams="sb"),
             Lane("c", _clip("c", 2, log), streams="sc")]
    run_lockstep(lanes, set_streams=lambda ln: seen.append(ln.streams))
    assert log == [("a", 0), ("b", 0), ("c", 0), ("a", 1), ("c", 1), ("a", 2)]
    assert [ln.result for ln in lanes] == [3, 0, 1]
    assert all(ln.done and ln.error is None for ln in lanes)
    assert seen[:3] == ["sa", "sb", "sc"]          # streams switched before every resume


def test_failure_is_isolated() -> None:
    log: list = []
    lanes = [Lane("a", _clip("a", 4, log, fail_at=1)), Lane("b", _clip("b", 3, log))]
    run_lockstep(lanes)
    assert isinstance(lanes[0].error, ValueError) and lanes[0].result is None
    assert lanes[1].error is None and lanes[1].result == 3
    assert ("b", 2) in log


def test_wait_event_yields_until_done() -> None:
    class Ev:
        def __init__(self) -> None:
            self.n = 0

        def query(self) -> bool:
            self.n += 1
            return self.n > 3

    assert sum(1 for _ in wait_event(Ev())) == 3


def test_rng_slot_keeps_streams_apart() -> None:
    """Two clips drawing alternately from one shared generator each see their own sequence."""
    state = {"g": 0}
    a = RngSlot(True, get_state=lambda: state["g"], set_state=lambda s: state.__setitem__("g", s))
    b = RngSlot(True, get_state=lambda: state["g"], set_state=lambda s: state.__setitem__("g", s))
    state["g"] = 100
    a.capture()
    state["g"] = 200
    b.capture()
    draws = []
    for _ in range(3):
        for slot, name in ((a, "a"), (b, "b")):
            with slot:
                draws.append((name, state["g"]))
                state["g"] += 1
    assert draws == [("a", 100), ("b", 200), ("a", 101), ("b", 201), ("a", 102), ("b", 202)]
    inactive = RngSlot(False)
    with inactive:
        pass


class _Ev:
    def __init__(self, ready_after: int) -> None:
        self.n, self.k, self.synced, self.queries = 0, ready_after, 0, 0

    def query(self) -> bool:
        self.n += 1
        self.queries += 1
        return self.n > self.k

    def synchronize(self) -> None:
        self.synced += 1
        self.n = self.k


def test_drive_single_blocks_instead_of_spinning() -> None:
    from fastdo.physics.lockstep import drive_single

    ev = _Ev(ready_after=10**6)

    def clip():
        yield
        yield from wait_event(ev)
        return "done"

    assert drive_single(clip()) == "done"
    assert ev.synced == 1 and ev.queries == 2                 # one blocking sync, not a million polls


def test_lockstep_sleeps_only_when_no_lane_progressed(monkeypatch) -> None:
    import fastdo.physics.lockstep as L

    sleeps: list[float] = []
    monkeypatch.setattr(L.time, "sleep", sleeps.append)
    ev_a, ev_b = _Ev(3), _Ev(5)

    def clip(ev):
        yield from wait_event(ev)
        return ev.k

    lanes = [Lane("a", clip(ev_a)), Lane("b", clip(ev_b))]
    run_lockstep(lanes)
    assert [ln.result for ln in lanes] == [3, 5]
    # rounds 1-3 both lanes only waited; round 4 a finished (progress); round 5 b alone still waited
    assert len(sleeps) == 4 and all(s == L.IDLE_SLEEP_S for s in sleeps)
