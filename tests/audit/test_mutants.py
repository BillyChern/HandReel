"""Mutation test of the key checks: each mutant is the real source of one function with one plausible error
substituted, compiled in a copy of its module's namespace. A mutant is killed only if at least one known answer of
`_ka.py` FAILS on it (AssertionError); a crash is an error of this test, not a kill."""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import replace
from types import ModuleType
from typing import Any

import pytest

from fastdo.audit import census, certificates, runner, scene
from tests.audit._ka import Impl, run_ka


def mutant(module: ModuleType, target: str, old: str, new: str, rebind: tuple[str, ...] = ()) -> dict[str, Any]:
    """Namespace with `target` recompiled from its source with `old` -> `new` (exactly one occurrence), and each
    name in `rebind` recompiled unchanged so that it calls the mutant."""
    src = textwrap.dedent(inspect.getsource(getattr(module, target)))
    assert src.count(old) == 1, f"{target}: {old!r} occurs {src.count(old)} times"
    ns = dict(module.__dict__)
    exec(compile(src.replace(old, new), f"<mutant {target}>", "exec"), ns)  # noqa: S102 (mutation testing)
    for name in rebind:
        code = compile(textwrap.dedent(inspect.getsource(getattr(module, name))), f"<rebind {name}>", "exec")
        exec(code, ns)  # noqa: S102 (mutation testing)
    return ns


def _census(old: str, new: str, target: str = "grip_census") -> Impl:
    rebind = ("grip_census",) if target != "grip_census" else ()
    return replace(Impl(), grip_census=mutant(census, target, old, new, rebind)["grip_census"])


def _cert(old: str, new: str, target: str = "C289Certificate") -> Impl:
    rebind = ("C289Certificate", "make_certificate") if target == "mt_null" else ("make_certificate",)
    return replace(Impl(), make_certificate=mutant(certificates, target, old, new, rebind)["make_certificate"])


MUTANTS = {
    # grip census
    "census samples the warm-up": lambda: _census("range(int(release_row)", "range(0"),
    "census reads the tangential force": lambda: _census("fs.append(float(f6[0]))", "fs.append(float(f6[1]))"),
    "census counts every object contact": lambda: _census(
        "if obj[g1] == obj[g2] or not led[g2 if obj[g1] else g1]:", "if obj[g1] == obj[g2]:"),
    "census takes the mean": lambda: _census("grip_over_weight_p50=float(np.median(t / weight))",
                                             "grip_over_weight_p50=float(np.mean(t / weight))"),
    "census divides by mass, not weight": lambda: _census("np.linalg.norm(gravity)", "1.0", "object_weight"),
    "census keeps the largest contact only": lambda: _census("tot.append(sum(fs))",
                                                             "tot.append(max(fs) if fs else 0.0)"),
    "census ignores the stride": lambda: _census("qpos.shape[0], int(stride))", "qpos.shape[0], 1)"),
    "census drops the controls": lambda: _census("d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[r], qvel[r], ctrl[r]",
                                                 "d.qpos[:], d.qvel[:] = qpos[r], qvel[r]"),
    "census keeps the model's gravity": lambda: _census("        model.opt.gravity[:] = g\n", ""),
    # C_289
    "C_289 strict position threshold": lambda: _cert("p_err <= th.obj_pos_m", "p_err < th.obj_pos_m"),
    "C_289 without grace rows": lambda: _cert("a, b = W + th.grace_rows, min(T, n)", "a, b = W, min(T, n)"),
    "C_289 without M_i": lambda: _cert("ok, detail = obj and motion and informative, {}",
                                       "ok, detail = obj and informative, {}"),
    "C_289 ignores informativeness": lambda: _cert("ok, detail = obj and motion and informative, {}",
                                                   "ok, detail = obj and motion, {}"),
    "C_289 scores a prefix": lambda: _cert("full = n >= T and", "full = n > 0 and"),
    "MT-null over the whole trace": lambda: _cert("seg = ref[warmup_rows + th.grace_rows:required_rows]",
                                                  "seg = ref[:required_rows]", "mt_null"),
    # scene edits
    "removal takes every object pair": lambda: replace(Impl(), hand_object_pair_names=mutant(
        scene, "hand_object_pair_names", 'p["kinds"] == ("hand", "object")', '"object" in p["kinds"]')[
        "hand_object_pair_names"]),
    "pedestal recognised by body name only": lambda: replace(Impl(), classify_pairs=mutant(
        scene, "classify_pairs", 'if "pedestal" in n or "pedestal" in b:', 'if "pedestal" in b:')["classify_pairs"]),
    "removal accepts missing names": lambda: replace(Impl(), remove_pairs=mutant(
        scene, "remove_pairs", "    if bad:\n", "    if False:\n")["remove_pairs"]),
    # runner
    "runner census from row 0": lambda: replace(Impl(), audit_run=mutant(
        runner, "_census", "object_geom=obj, release_row=W,", "object_geom=obj, release_row=0,",
        ("audit_run",))["audit_run"]),
    "runner contact-removed null keeps contact": lambda: replace(Impl(), audit_run=mutant(
        runner, "_absent_simulator", 'absent_mode="displace" if removal == "displace" else "pairs"',
        'absent_mode="pairs"', ("audit_run",))["audit_run"]),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name: str) -> None:
    failed = run_ka(MUTANTS[name]())
    assert failed, f"mutant survived: {name}"


def test_real_implementation_passes() -> None:
    assert run_ka(Impl()) == []
