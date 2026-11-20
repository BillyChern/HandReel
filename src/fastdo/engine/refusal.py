"""What a REFUSAL is, in one place both the workers and the runners can import.

A stage that declines a clip on the merits -- the interaction gate, the keyframe prompt -- raises an
ordinary exception whose message starts with a stable prefix, so the manifest needs no
extra state: the row is `failed` and the prefix says the failure was a verdict rather than an error.

This module exists because of the import direction. Every `workers_*` module imports
`fastdo.engine.orchestrator` for `StageFn`, so the orchestrator cannot import the prefixes back from
them. A leaf with no engine imports can be read by both, and by `manifest`.

Why the distinction earns its keep: a refusal is a DETERMINISTIC function of artifacts already on disk,
so re-running it buys an identical refusal. A genuine failure -- an OOM, a GPU-guard kill, a truncated
read -- is worth retrying. `ClipManifest.pending` treats every non-`done` row as work, which is right for
the second and pure waste for the first.
"""
from __future__ import annotations

__all__ = ["GATE_REJECTED_PREFIX", "PROMPT_REFUSED_PREFIX", "REFUSAL_PREFIXES", "is_refusal"]

#: The interaction gate's verdict prefix (`workers_gate.GateRejected`).
GATE_REJECTED_PREFIX = "gate rejected"

#: The keyframe prompt stage's verdict prefix (`workers_prompt.PromptRefused`).
PROMPT_REFUSED_PREFIX = "prompt refused"

#: Every prefix that marks a recorded error as a verdict rather than a fault.
REFUSAL_PREFIXES: tuple[str, ...] = (GATE_REJECTED_PREFIX, PROMPT_REFUSED_PREFIX)


#: The prefixes landed in `1d9c6e5`. A manifest written BEFORE that records the same refusals without
#: them -- one Ego4D refusal run's manifest is an example: all 13 of its failures are prompt refusals
#: ("no confident hand-object contact", "the box prompt gave an implausible ...") and not one matches.
#: Such a manifest gets no benefit from the resume policy and will keep re-running its refusals; that is
#: the correct conservative behaviour, since an unrecognised message may be a genuine fault.


def is_refusal(error: str) -> bool:
    """Is this recorded manifest error a stage's verdict on the clip, rather than a fault?

    Matches on the message prefix rather than an exception type because the manifest stores the message,
    not the exception: by the time a resume reads the row, the object is long gone.
    """
    return any(error.startswith(prefix) for prefix in REFUSAL_PREFIXES)
