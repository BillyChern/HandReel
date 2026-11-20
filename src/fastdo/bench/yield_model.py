"""The corpus-economics model, with each constant's PROVENANCE attached to it.

Three findings of 2026-09-21 are encoded here rather than left in prose, because all three were mistakes
that a number alone cannot prevent:

* **Contended card** — the per-stage budget was measured on a contended card. Shape's recorded 84.83 s is **20.1 s**
  on a quiet one. A `Seconds` therefore carries a `card` field, and `yield_per_gpu_hour` REFUSES a
  contended constant unless the caller says so in as many words.
* **Per-video decode** — decode is a **per-video** cost, not a per-window one: 258.07 s for the first window of a video
  against 3.22 s for every later one. So the cost model has two terms and `windows_per_video` is an
  argument, not an assumption.
* **Amdahl both ways** — Amdahl cuts both ways. `speedup_of_improving` is monotonically decreasing in the remainder, so
  an inflated remainder UNDER-values work already done. The helper returns the remainder it used alongside
  the ratio, because a speedup quoted without its denominator is the shape of that error.

Nothing here measures anything. It is the arithmetic plus the bookkeeping that stops the arithmetic being
applied to numbers that cannot support it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "MEASURED",
    "Seconds",
    "seconds_per_window",
    "speedup_of_improving",
    "windows_per_gpu_hour",
    "yield_per_gpu_hour",
]

CardState = Literal["exclusive", "contended", "unknown"]


@dataclass(frozen=True)
class Seconds:
    """A duration and the conditions that produced it.

    `card` is not documentation. `exclusive` means foreign compute processes were checked and there were
    none; `contended` means they were present or a peer was known to be running; `unknown` means nobody
    recorded it — which is the state every pre-2026-09-21 figure is in, and the reason the contended-card
    finding took a replay rather than an inspection to settle.
    """

    seconds: float
    source: str
    card: CardState = "unknown"
    note: str = ""

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError(f"{self.source}: seconds must be >= 0, got {self.seconds}")

    @property
    def trustworthy(self) -> bool:
        """True only for a duration measured with the card verified empty."""
        return self.card == "exclusive"

    def __float__(self) -> float:
        return float(self.seconds)


# Every constant this project has that feeds an economic argument, with where it came from. A figure whose
# card state is not `exclusive` is kept, not deleted -- the point is to make its status visible at the call
# site, not to pretend it does not exist.
MEASURED: dict[str, Seconds] = {
    # --- measured on a card/machine checked empty (2026-09-21) -------------------------------------
    "shape_ego4d": Seconds(17.8, "measured: shape stage, two Ego4D windows (mean of k000/k001)", "exclusive",
                           "84.83 s and 152.53 s on the 2026-09-20 pilot; cold start excluded at 1.05x"),
    "prompt_21_rows": Seconds(1.40, "measured: prompt cost fit 0.363 + 0.049*n s, at n=21", "exclusive",
                              "the pilot said 14.81 s (10.6x); the config docstring's 1.39 s was RIGHT"),
    "decode_cold_frame_table": Seconds(204.2, "measured: cold decode + frame-table build, median of 2 windows",
                                       "exclusive",
                                       "a 1033 s and a 1161 s video; the ONLY pilot figure that "
                                       "reproduced (258 s). Scales with video duration, not universal"),
    "decode_warm_window": Seconds(2.45, "measured: the window decode alone, warm frame table", "exclusive",
                                  "the floor: no addressing change can remove this"),
    "decode_packet_route": Seconds(3.42, "measured: packet table + window decode", "exclusive",
                                   "60x on the cold path, byte-identical; flat in W, within 1.4x of the "
                                   "floor. Opt-in per source: screened 21/21 on Ego4D only"),
    "c_pre_packet_route": Seconds(4.82, "derived: decode_packet_route + prompt_21_rows", "exclusive",
                                  "flat in W -- the packet route removes windows-per-video from the model"),
    # --- superseded, kept because earlier arithmetic needs them ------------------------------------
    "shape_ego4d_pilot": Seconds(118.7, "pilot: S2 pilot stage profile (mean of k000/k001)", "contended",
                                 "SUPERSEDED by shape_ego4d; kept because earlier arithmetic needs it"),
    "prompt_pilot": Seconds(14.81, "pilot: S2 manifest, median over 45 refused windows", "contended",
                            "SUPERSEDED by prompt_21_rows; 10.6x inflated"),
    "c_video_decode": Seconds(258.07, "pilot: S2 manifest, mean over the 9 first-of-video windows", "unknown",
                              "SUPERSEDED by decode_cold_frame_table (204.2 s measured); kept because "
                              "the published per-video arithmetic is pinned against it"),
    "c_window_decode": Seconds(3.22, "pilot: S2 manifest, mean over the 38 later windows", "unknown",
                               "SUPERSEDED by decode_warm_window (2.45 s measured); same reason"),
    "c_post_pilot": Seconds(161.3, "note: the 2026-09-21 c_pre estimate", "contended",
                            "the figure that made the packet-table work look like 1.05x-1.15x"),
    # --- still unmeasured: the largest open number in the model ------------------------------------
    "c_post_partly_corrected": Seconds(92.4, "pilot: S2 stage row, batch-corrected, shape measured quiet", "unknown",
                                       "ONLY shape is exclusive; geometry 47.0, mask 21.3, hands 20.7, "
                                       "solve 21.6 and qc 13.3 s are all pilot values. If they share "
                                       "shape's ~4x, c_post is ~23 s and S_max goes 20x -> ~85x"),
}


def seconds_per_window(c_window: float, c_video: float, windows_per_video: float) -> float:
    """`c_window + c_video / W` — the two-term cost (a per-video term, amortised over W).

    Raises:
        ValueError: On `windows_per_video <= 0`. Zero windows per video is not a limit to take; it is a
            corpus that indexes videos and looks at none of them, whose cost per window is undefined
            rather than infinite.
    """
    if windows_per_video <= 0:
        raise ValueError(f"windows_per_video must be > 0, got {windows_per_video}")
    if c_window < 0 or c_video < 0:
        raise ValueError(f"costs must be >= 0, got c_window={c_window}, c_video={c_video}")
    return c_window + c_video / windows_per_video


def windows_per_gpu_hour(c_window: float, c_video: float, windows_per_video: float) -> float:
    """Windows finished per GPU-hour under the two-term model."""
    return 3600.0 / seconds_per_window(c_window, c_video, windows_per_video)


def yield_per_gpu_hour(pi: float, r: float, q: float, c_pre: float, a: float, c_post: Seconds | float,
                       *, allow_contended: bool = False) -> float:
    """`levers.md`'s yield law, `3600·π·r·q / (c_pre + a·c_post)`.

    Args:
        pi: Fraction of windows the gate admits.
        r: Fraction of admitted windows that survive QC.
        q: Usable clips per surviving window.
        c_pre: Per-window cost paid on EVERY window (screen + amortised index).
        a: `π·r + (1−π)·f` — the expected share of `c_post` actually paid.
        c_post: Per-window cost of the full chain. Pass a `Seconds` to have its provenance checked.
        allow_contended: Permit a `c_post` that was not measured on a verified-empty card. Required, and
            named, because an inflated `c_post` UNDER-states the value of every `c_pre`
            improvement — a signed error, in the direction that abandons good work.

    Raises:
        ValueError: If `c_post` is a `Seconds` that is not `exclusive` and `allow_contended` is False.
    """
    if isinstance(c_post, Seconds) and not c_post.trustworthy and not allow_contended:
        raise ValueError(
            f"c_post came from {c_post.source!r} with card={c_post.card!r}, not 'exclusive'. An inflated "
            f"c_post makes every c_pre improvement look smaller than it is. Pass "
            f"allow_contended=True to proceed anyway, and quote the card state with the result.")
    denominator = c_pre + a * float(c_post)
    if denominator <= 0:
        raise ValueError(f"denominator must be > 0, got {denominator}")
    return 3600.0 * pi * r * q / denominator


@dataclass(frozen=True)
class Speedup:
    """A ratio and the remainder it was taken against — never one without the other."""

    ratio: float
    remainder: float
    before: float
    after: float
    card: CardState = "unknown"
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        tail = f" [{'; '.join(self.caveats)}]" if self.caveats else ""
        return (f"{self.ratio:.2f}x ({self.before:.2f} -> {self.after:.2f} against a remainder of "
                f"{self.remainder:.2f}, card={self.card}){tail}")


def speedup_of_improving(x_before: float, x_after: float, remainder: float | Seconds,
                         *, card: CardState = "unknown") -> Speedup:
    """`(x_before + R) / (x_after + R)` — one additive term improved, against everything else.

    **Monotonically decreasing in `R`.** That is the whole point: Amdahl is usually quoted to deflate a
    ratio, and the same algebra deflates in the other direction when `R` is over-measured. The remainder
    travels with the ratio in the returned `Speedup` so the number cannot be quoted without it.
    """
    R = float(remainder)
    if R < 0 or x_before < 0 or x_after < 0:
        raise ValueError(f"all terms must be >= 0, got {x_before=}, {x_after=}, R={R}")
    if x_after + R <= 0:
        raise ValueError("the improved denominator must be > 0")
    state: CardState = remainder.card if isinstance(remainder, Seconds) else card
    caveats: list[str] = []
    if state != "exclusive":
        caveats.append(f"remainder card={state}: this ratio is a LOWER bound")
    return Speedup((x_before + R) / (x_after + R), R, x_before, x_after, state, tuple(caveats))
