"""Batched world-state copies of the tracking environments (`fastdo.policy.env_state`), with which the particle
system of `fastdo.policy.smc` copies particles between worlds.

On the CPU twin a restored or copied world reproduces its next 10 steps bit-exactly, and every copied field reads back
as written; on MuJoCo Warp (skipped without CUDA) a restored world reproduces to float32 solver precision.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import mujoco
import numpy as np
import pytest
import torch
from torch import Tensor

# Measured (2026-09-22): GPU contact order makes two identical warp rollouts differ by up to 2.0e-4 (30/30 runs
# exceed 1e-5, so 1e-5 failed ~50% of the time). Omitting one WorldState field from the restore changes the output by
# 2.7e-3 (prev_delta), 6.9e-2 (phi), >= 4.9 (k, t, backend); so atol 1e-3 is 5x the noise and still catches every
# field that CAN be seen on the GPU. warm_start, prev_x, force, obs, priv and reference move it by only 7e-5..1.7e-4,
# inside the noise at ANY tolerance -- their restore is proven only by the bit-exact CPU-twin tests above.
WARP_RTOL, WARP_ATOL = 0.0, 1e-3
_EPISODE_BUFFERS = ("warm_start", "prev_delta", "prev_x", "force", "obs", "priv", "phi")


def _record(env: Any, actions: Tensor) -> list[dict[str, Tensor]]:
    """Clones of everything a step returns or exposes, per step."""
    out = []
    for action in actions:
        obs, reward, done, info = env.step(action.to(env.device))
        row = {"obs": obs, "priv": env.privileged(), "reward": reward, "done": done}
        for key, value in info.items():
            items = value.items() if isinstance(value, dict) else [("", value)]
            row.update({f"{key}/{name}": tensor for name, tensor in items})
        out.append({key: value.detach().clone().cpu() for key, value in row.items()})
    return out


def _staggered_env(device: str, n_worlds: int) -> tuple[Any, Tensor]:
    """Worlds on distinct clips (start frames 0, 20, 40, 10; lengths 6, 14, 12, 9; the third yawed and shifted) with
    open-hand warm-ups of W = 5 steps. The last world restarts after two steps, so at the snapshot it is on warm-up
    step t = 1 and the others on t = 3."""
    from fastdo.policy.env import make_env
    from tests.policy.env_fixtures import SCENE_XML, env_config, moved_reference, whisking_reference

    env = make_env(env_config(n_worlds=n_worlds, warmup_duration=0.1), SCENE_XML, device)
    refs = [whisking_reference(n_frames=6), whisking_reference(n_frames=14, start=20),
            moved_reference(whisking_reference(n_frames=12, start=40), 0.5, (0.05, -0.04, 0.0)),
            whisking_reference(n_frames=9, start=10)][:n_worlds]
    for world, ref in enumerate(refs):
        env.reset(ref, torch.tensor([world]))
    actions = 0.3 * torch.randn(13, n_worlds, env.act_dim, generator=torch.Generator().manual_seed(0))
    env.step(actions[0].to(env.device))
    env.step(actions[1].to(env.device))
    env.reset(refs[-1], torch.tensor([n_worlds - 1]))
    env.step(actions[2].to(env.device))
    return env, actions[3:]


def _check_window(steps: list[dict[str, Tensor]]) -> None:
    """The 10 steps start in every world's warm-up, cross every world's weld release and end world 0's clip (a 6-frame
    clip: a time-out on its fifth free step, or a failure before it), so an auto-reset happens inside them."""
    warm = torch.stack([s["warmup/"] for s in steps])
    assert bool(warm[:2].all()) and bool((warm[:-1] & ~warm[1:]).any(0).all())
    assert bool(torch.stack([s["done"] for s in steps])[:, 0].any())


def _skip_without_scene() -> None:
    from tests.policy.env_fixtures import TRAJECTORY_NPZ

    if not TRAJECTORY_NPZ.exists():
        pytest.skip("baseline submodule whisking scene not checked out")


def test_state_copy_roundtrip_cpu_env() -> None:
    """get_state, 10 steps, set_state, the same 10 steps on the CPU twin: every observation, privileged feature,
    reward, flag and info entry is bit-identical, through the weld release and an auto-reset."""
    _skip_without_scene()
    env, actions = _staggered_env("cpu", 3)
    ids = torch.arange(3)
    state = env.get_state(ids)
    first = _record(env, actions)
    env.set_state(ids, state)
    second = _record(env, actions)
    _check_window(first)
    for t, (a, b) in enumerate(zip(first, second, strict=True)):
        for key in a:
            assert torch.equal(a[key], b[key]), (t, key)
    env.set_state(ids, state)                                         # a snapshot restores any number of times
    assert all(torch.equal(x, y) for x, y in zip(_record(env, actions[:2])[1].values(), first[1].values(),
                                                  strict=True))


def test_state_copy_between_worlds_cpu_env() -> None:
    """set_state writes a snapshot into other worlds: the cyclic permutation world i -> world (i + 1) % 3 of three
    staggered worlds on distinct clips reproduces each source world's next 10 steps bit-exactly in its new world.

    The 10-step window cannot see every copied field: the previous canonical x-axis matters only near a degenerate
    frame, the next step overwrites the last contact forces and ignores the time, and the Newton solver converges to
    the same step from any warm start. So synthetic values of the episode buffers, the simulator time and warm start,
    the weld target and gravity are written into the permuted worlds and must read back exactly (`MjData` included)."""
    _skip_without_scene()
    env, actions = _staggered_env("cpu", 3)
    state = env.get_state(torch.arange(3))
    first = _record(env, actions)
    perm = torch.tensor([1, 2, 0])                                    # world perm[i] receives source row i
    env.set_state(perm, state)
    moved = torch.empty_like(actions)
    moved[:, perm] = actions
    second = _record(env, moved)
    for t, (a, b) in enumerate(zip(first, second, strict=True)):
        for key in a:
            assert torch.equal(a[key], b[key][perm]), (t, key)

    gen, model = torch.Generator().manual_seed(1), env.scene.model
    buffers = {name: torch.randn(getattr(state, name).shape, generator=gen, dtype=getattr(state, name).dtype)
               for name in _EPISODE_BUFFERS}
    backend = {name: np.array(value, copy=True) for name, value in state.backend.items()}
    slices = {}
    for flag, width, attribute in ((mujoco.mjtState.mjSTATE_TIME, 1, "time"),
                                   (mujoco.mjtState.mjSTATE_WARMSTART, model.nv, "qacc_warmstart")):
        offset = mujoco.mj_stateSize(model, int(mujoco.mjtState.mjSTATE_INTEGRATION) & (int(flag) - 1))
        slices[attribute] = slice(offset, offset + width)        # state elements follow the order of the flag bits
        backend["integration"][:, slices[attribute]] = torch.randn(3, width, generator=gen, dtype=torch.float64)
    backend["weld"] = backend["weld"] + torch.randn(backend["weld"].shape, generator=gen, dtype=torch.float64).numpy()
    backend["gravity"] = backend["gravity"] + np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.0, -0.5, 0.0]])
    written = dataclasses.replace(state, backend=backend, **buffers)
    env.set_state(perm, written)
    read = env.get_state(perm)
    for name in _EPISODE_BUFFERS:
        assert torch.equal(getattr(read, name), buffers[name]), name
    for name, value in backend.items():
        np.testing.assert_array_equal(read.backend[name], value, err_msg=name)
    for attribute, part in slices.items():
        simulator = np.stack([np.atleast_1d(getattr(env._datas[w], attribute)) for w in perm.tolist()])
        np.testing.assert_array_equal(simulator, backend["integration"][:, part], err_msg=attribute)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_state_copy_roundtrip_warp_env() -> None:
    """MuJoCo Warp: the staggered roundtrip on 4 worlds, then worlds 0 and 1 copied into worlds 3 and 2. Flags and
    frame indices agree exactly; floats within 1e-4 relative / 1e-5 absolute, since GPU contact detection does not
    promise the same contact order (hence solver summation order) on every run."""
    pytest.importorskip("mujoco_warp", reason="the `policy` extra (mujoco-warp, warp-lang) is not installed")
    _skip_without_scene()
    env, actions = _staggered_env("cuda", 4)
    ids = torch.arange(4, device="cuda")
    state = env.get_state(ids)
    first = _record(env, actions)
    env.set_state(ids, state)
    second = _record(env, actions)
    _check_window(first)

    def same(a: Tensor, b: Tensor) -> None:
        if a.dtype in (torch.bool, torch.long):
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=WARP_RTOL, atol=WARP_ATOL)

    for a, b in zip(first, second, strict=True):
        for key in a:
            same(a[key], b[key])
    env.set_state(ids, state)
    env.set_state(torch.tensor([3, 2], device="cuda"), env.get_state(torch.tensor([0, 1], device="cuda")))
    copied = actions.clone()
    copied[:, [3, 2]] = actions[:, [0, 1]]
    third = _record(env, copied)
    for a, c in zip(first, third, strict=True):
        for key in a:
            same(a[key][[0, 1]], c[key][[3, 2]])


def test_snapshot_restored_into_a_longer_bank_reproduces_its_steps() -> None:
    """The `ReferenceBank.set_rows` padding branch: a world on a 6-frame clip is snapshotted while the
    bank holds 6 frames; after the other world's 30-frame clip grows the bank, restoring the snapshot pads its rows to
    30, and the world's next 7 steps (through its time-out and restart) reproduce the steps recorded right after the
    snapshot bit for bit."""
    _skip_without_scene()
    from fastdo.policy.env import make_env
    from tests.policy.env_fixtures import SCENE_XML, env_config, whisking_reference

    env = make_env(env_config(n_worlds=2, warmup_duration=0.04), SCENE_XML, "cpu")         # W = 2
    env.reset(whisking_reference(n_frames=6), None)
    env.step(torch.zeros(2, env.act_dim))
    assert env._bank.t_max == 6
    world = torch.tensor([0])
    state = env.get_state(world)
    actions = 0.3 * torch.randn(7, 2, env.act_dim, generator=torch.Generator().manual_seed(2))
    first = _record(env, actions)
    env.reset(whisking_reference(n_frames=30, start=10), torch.tensor([1]))
    assert env._bank.t_max == 30 and state.reference.obj.shape[1] == 6
    env.set_state(world, state)
    assert env.get_state(world).reference.obj.shape[1] == 30
    second = _record(env, actions)
    assert bool(torch.stack([s["done"][0] for s in first]).any())                  # the window crosses the restart
    for t, (a, b) in enumerate(zip(first, second, strict=True)):
        for key in a:
            assert torch.equal(a[key][0], b[key][0]), (t, key)
