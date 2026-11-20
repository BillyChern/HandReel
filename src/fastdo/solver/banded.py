"""Block-tridiagonal normal equations: O(T) assembly, block-Thomas solve and marginal covariances.

With the smoothness prior coupling
only adjacent frames, the (6T+1)-dimensional Gauss-Newton system is block-tridiagonal in the 6T pose
coordinates plus one dense arrow row/column for the shared log-scale parameter s:

    H = [[D_1, U_1,           , c_1],
         [U_1^T, D_2, U_2,     , c_2],
         [     , ...,  ...,    , ...],
         [     ,     , U_{T-1}^T, D_T, c_T],
         [c_1^T, c_2^T, ..., c_T^T, h_ss]]

`assemble_block_tridiagonal` builds `D`, `U`, `c`, `h_ss`, `g`, `g_s` as batched contractions (no Python
loop over frames): every term's per-frame (7,7)/(7,) blocks sum directly into `D`/`c`/`h_ss`/`g`/`g_s`.
The temporal priors are the only source of the `U` off-diagonal blocks, since data terms never couple
frames. `solve_block_tridiagonal` eliminates the pose block by the block-Thomas (block-LDL^T) recursion
and then the scale by a rank-one Schur complement, never materializing the (6T+1)^2 matrix;
`marginal_covariances` reuses the forward Schur factors for the backward (Rauch-Tung-Striebel, "RTS")
covariance recursion, plus the scale parameter's rank-one contribution to each block.

The constant-velocity prior couples frames (k-1, k, k+1), adding (k, k+2) blocks
`U2`. Grouping consecutive frames (2j, 2j+1) into 12x12 super-blocks makes the system exactly
block-tridiagonal again, so the same recursion runs at block size 12, and each frame's 6x6 marginal is a
diagonal sub-block of the 12x12 one. Without a velocity prior (`U2 is None`) blocks stay 6x6.

The frames of B clips are concatenated in order and described by a
`ClipLayout`. The joint system has one scale unknown per clip (`h_ss`, `g_s` are (B,); frame k's arrow
entry `c_k` couples it to its own clip's scale), and every prior block that would couple two clips is
exactly zero because the evaluate callback weights those residuals by zero (`batch.batch_priors`).
Block-Thomas elimination with a zero coupling block creates no fill across it, and each clip's scale is
eliminated by its own Schur complement, so the batched solve equals B separate solves up to round-off;
the RTS recursion decouples the same way.

Anisotropic scale (`SolverConfig.anisotropic_scale`): each clip's one
log-scale becomes three log-stretches, so the arrow border grows from 1 to K = 3 columns: `c` (T,6,K), `h_ss` (B,K,K),
`g_s` (B,K). The pose band is untouched; block-Thomas solves K + 1 right-hand sides instead of 2 and the Schur
complement is a KxK SPD system per clip instead of a scalar (the isotropy prior's lam P sits in `h_ss`, assembled from
its term like any data term). A 1-column system keeps the historical shapes and runs the historical code unchanged. The recursion therefore runs on a clip-parallel layout: every
clip is padded to the same even length W with decoupled identity dummy frames (the odd-T padding of Task
15, so a frame pair never straddles two clips), and `_block_thomas` runs one sequential loop over the
block positions with one batched kernel per step for all clips. The solve runs on the system's device
without host synchronization (the LM loop's one-sync-per-iteration budget: cuSOLVER Cholesky factors,
pinned once per solve by `api.solve_trajectories`, and cuBLAS triangular solves, never MAGMA, see
`_block_thomas`); `marginal_covariances`, called once per solve, runs on the CPU where the sequential
small-block recursion is faster. A system without a layout is the single-clip case
with a scalar `h_ss`/`g_s`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from fastdo.solver._block_thomas import forward_schur, pair_frames, rts_covariances, thomas_solve
from fastdo.solver.terms import TermOutput

Smooth = tuple[Tensor, Tensor, Tensor]
Smooth2 = tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class Priors:
    """Temporal priors on the trajectory, accepted wherever a bare `Smooth` triple is.

    `smooth` is `smoothness_term`'s constant-pose triple `(r (T-1,6), J_k, J_{k+1})`; `velocity` is
    `velocity_term`'s second-difference quadruple `(r (T-2,6), J_{k-1}, J_k, J_{k+1})`. An evaluate
    callback may return either a `Priors` or (as before) a `Smooth` triple or `None`.
    """

    smooth: Smooth | None = None
    velocity: Smooth2 | None = None


PriorsLike = Smooth | Priors | None


def clip_sum(values: Tensor, index: Tensor, n: int) -> Tensor:
    """`torch.zeros(n).index_add(0, index, values)` for 1-D `values`, deterministically: a masked row sum.

    CUDA `index_add` on floats sums with atomics, in a different order on every run, and the LM accepts or
    rejects steps by comparing these costs, so a 1-ulp difference can flip a decision and move a pose by ~1 cm on
    bit-identical inputs. Selecting with `where` (never multiplying by a one-hot, since 0 * inf is NaN)
    and reducing along a fixed axis sums in a fixed order.
    """
    member = index[None, :] == torch.arange(n, device=index.device)[:, None]
    return torch.where(member, values[None, :], torch.zeros((), dtype=values.dtype, device=values.device)).sum(1)


def as_priors(priors: PriorsLike) -> Priors:
    """Normalize the `smooth` slot of an evaluate callback's return value to a `Priors`."""
    if priors is None:
        return Priors()
    if isinstance(priors, Priors):
        return priors
    return Priors(smooth=priors)


def _index(values: Sequence[int], device: torch.device) -> Tensor:
    return torch.tensor(list(values), dtype=torch.long, device=device)


@dataclass(frozen=True, eq=False)
class ClipLayout:
    """Host-known structure of B clips whose frames are concatenated in order.

    `clip_index` (T,) names each frame's clip. `pair_rows` (P,) / `triple_rows` (Q,) are the rows k
    whose frames k, k+1 (resp. k, k+2) lie in one clip, i.e. the prior blocks `U[k]` / `U2[k]` that may
    be nonzero. Every clip is padded to `width` W frames (the longest clip, rounded up to even); `slot`
    (T,), `u_slot` (P,), `u2_slot` (Q,) place frames and coupling blocks in the flattened (B*W),
    (B*(W-1)), (B*(W-2)) padded arrays. `clip_frames[b]` is clip b's frame index (arange). All index
    tensors are built once, outside any LM loop.
    """

    lengths: tuple[int, ...]
    width: int
    clip_index: Tensor
    pair_rows: Tensor
    triple_rows: Tensor
    slot: Tensor
    u_slot: Tensor
    u2_slot: Tensor
    clip_frames: tuple[Tensor, ...] = ()

    @staticmethod
    def build(lengths: Sequence[int], device: torch.device | str) -> ClipLayout:
        lengths = tuple(int(n) for n in lengths)
        if not lengths or min(lengths) < 1:
            raise ValueError(f"every clip needs at least one frame, got lengths {lengths}")
        dev = torch.device(device)
        longest = max(lengths)
        W = longest + longest % 2
        starts = [sum(lengths[:b]) for b in range(len(lengths))]
        clip_frames = tuple(torch.arange(s, s + n, dtype=torch.long, device=dev)
                            for s, n in zip(starts, lengths, strict=True))   # device kernels, no copy
        if len(lengths) == 1:                                            # no host-to-device copy
            T = lengths[0]
            ar = clip_frames[0]
            zeros = torch.zeros(T, dtype=torch.long, device=dev)
            return ClipLayout(lengths, W, zeros, ar[:-1], ar[:max(T - 2, 0)], ar, ar[:-1], ar[:max(T - 2, 0)],
                              clip_frames)
        clip, slot, pairs, triples, u_slot, u2_slot = [], [], [], [], [], []
        k = 0
        for b, n in enumerate(lengths):
            for pos in range(n):
                clip.append(b)
                slot.append(b * W + pos)
                if pos + 1 < n:
                    pairs.append(k)
                    u_slot.append(b * (W - 1) + pos)
                if pos + 2 < n:
                    triples.append(k)
                    u2_slot.append(b * (W - 2) + pos)
                k += 1
        return ClipLayout(lengths, W, _index(clip, dev), _index(pairs, dev), _index(triples, dev),
                          _index(slot, dev), _index(u_slot, dev), _index(u2_slot, dev), clip_frames)

    def frames_of(self, clips: Sequence[int]) -> Tensor | None:
        """Device index of the frames of `clips` (ascending clip ids), or None when `clips` names every clip.
        Concatenates the prebuilt per-clip indices: device kernels only, no host-device transfer."""
        if len(clips) == self.n_clips:
            return None
        if len(clips) == 1:
            return self.clip_frames[clips[0]]
        return torch.cat([self.clip_frames[b] for b in clips])

    @property
    def n_clips(self) -> int:
        return len(self.lengths)

    @property
    def n_frames(self) -> int:
        return sum(self.lengths)

    def frame_slices(self) -> tuple[slice, ...]:
        out, k = [], 0
        for n in self.lengths:
            out.append(slice(k, k + n))
            k += n
        return tuple(out)


@dataclass(frozen=True)
class BlockTridiagonal:
    """Block-tridiagonal-plus-arrow normal equations, all tensors float64 on the terms' device.

    `D` (T,6,6) are the per-frame pose diagonal blocks, `U` (T-1,6,6) the prior-only off-diagonal blocks
    (`U[k]` couples frame `k` to frame `k+1`), `c` (T,6) each frame's arrow column coupling it to its
    (clip's) log-scale parameter, `h_ss` the scale corner and `g` (T,6) / `g_s` the gradient entries.
    None of these are damped; damping (`lam * diag(H)` plus the numerical floor) is applied inside
    `solve_block_tridiagonal`/`marginal_covariances`, as `lm.py`'s dense path only damps at solve time.

    `U2` (T-2,6,6), present only when a velocity prior was assembled, holds the (k, k+2)
    blocks; `None` means the system is block-tridiagonal. `layout` describes B concatenated
    clips: `h_ss`/`g_s` are then (B,), one scale per clip, and the blocks coupling two clips are zero.
    Without a layout the system is one clip with scalar `h_ss`/`g_s`.

    Anisotropic scale: a K-column border (`border` > 1) has `c` (T,6,K), `h_ss` (B,K,K) and `g_s` (B,K) ((K,K), (K,)
    without a layout); see the module docstring.
    """

    D: Tensor
    U: Tensor
    c: Tensor
    h_ss: Tensor
    g: Tensor
    g_s: Tensor
    U2: Tensor | None = None
    layout: ClipLayout | None = None

    @property
    def border(self) -> int:
        """Border columns per clip: 1 (the log-scale), or the anisotropic scale's 3 log-stretches."""
        return 1 if self.c.dim() == 2 else int(self.c.shape[-1])


def _clip_sum_rows(values: Tensor, index: Tensor, n: int) -> Tensor:
    """`clip_sum` of (T, ...) `values` row by row: (n, ...), the same fixed-order masked sum over the frame axis."""
    member = index[None, :] == torch.arange(n, device=index.device)[:, None]
    flat = values.reshape(values.shape[0], -1)
    out = torch.where(member[:, :, None], flat[None], torch.zeros((), dtype=values.dtype, device=values.device)).sum(1)
    return out.reshape(n, *values.shape[1:])


def term_blocks(term: TermOutput) -> tuple[Tensor, Tensor]:
    """Per-frame (7,7) Hessian blocks and (7,) gradients from one term (moved here from `lm.py`)."""
    sw = term.weight.clamp_min(0.0).sqrt()
    Jw = (term.jacobian * sw[..., None, None]).flatten(1, 2)            # (T, n*d, 7)
    rw = (term.residual * sw[..., None]).flatten(1, 2)                  # (T, n*d)
    Jw = Jw.to(torch.float64)
    rw = rw.to(torch.float64)
    return Jw.transpose(1, 2) @ Jw, (Jw.transpose(1, 2) @ rw.unsqueeze(-1)).squeeze(-1)


def assemble_block_tridiagonal(terms: list[TermOutput], smooth: PriorsLike, n_frames: int,
                               layout: ClipLayout | None = None) -> BlockTridiagonal:
    """Batched assembly: sum every term's (T,7,7)/(T,7) blocks into (D,c,h_ss,g,g_s), then add the
    smoothness off-diagonal (U) and its diagonal/gradient contributions via three batched `bmm`s over
    the T-1 adjacent pairs. No Python loop over frames. `smooth` may also be a `Priors`; its velocity
    quadruple (A, B, C) = (J_{k-1}, J_k, J_{k+1}) adds A^T A, B^T B, C^T C to the three diagonals, A^T B
    and B^T C to the adjacent blocks and A^T C to `U2`. With a `layout` the scale corner and
    gradient are summed per clip (`clip_sum` over `layout.clip_index`), giving (B,) `h_ss`/`g_s`."""
    return assemble_from_blocks([term_blocks(term) for term in terms], smooth, n_frames, layout)


def assemble_from_blocks(blocks: Sequence[tuple[Tensor, Tensor]], smooth: PriorsLike, n_frames: int,
                         layout: ClipLayout | None = None) -> BlockTridiagonal:
    """`assemble_block_tridiagonal` from every term's precomputed per-frame `term_blocks` (in term order).
    The batched LM keeps these per-frame blocks and overwrites only the rows of the clips it re-evaluated. Blocks of
    6 + K columns (anisotropic scale: K = 3) assemble a K-column border (`BlockTridiagonal.border`)."""
    T = n_frames
    priors = as_priors(smooth)
    tri, vel = priors.smooth, priors.velocity
    anchor = tri[0] if tri is not None else (vel[0] if vel is not None else None)
    device = blocks[0][0].device if blocks else (anchor.device if anchor is not None else torch.device("cpu"))
    if blocks and blocks[0][0].shape[-1] > 7:
        return _assemble_bordered(blocks, tri, vel, T, layout, device)
    D = torch.zeros(T, 6, 6, dtype=torch.float64, device=device)
    c = torch.zeros(T, 6, dtype=torch.float64, device=device)
    g = torch.zeros(T, 6, dtype=torch.float64, device=device)
    n_scale = () if layout is None else (layout.n_clips,)
    h_ss = torch.zeros(n_scale, dtype=torch.float64, device=device)
    g_s = torch.zeros(n_scale, dtype=torch.float64, device=device)
    for Hk, gk in blocks:                                                # (T,7,7), (T,7)
        D = D + Hk[:, :6, :6]
        c = c + Hk[:, :6, 6]
        g = g + gk[:, :6]
        if layout is None:
            h_ss = h_ss + Hk[:, 6, 6].sum()
            g_s = g_s + gk[:, 6].sum()
        else:
            h_ss = h_ss + clip_sum(Hk[:, 6, 6], layout.clip_index, layout.n_clips)
            g_s = g_s + clip_sum(gk[:, 6], layout.clip_index, layout.n_clips)
    D, U, g, U2 = _add_priors(D, g, tri, vel, T, device)
    return BlockTridiagonal(D=D, U=U, c=c, h_ss=h_ss, g=g, g_s=g_s, U2=U2, layout=layout)


def _assemble_bordered(blocks: Sequence[tuple[Tensor, Tensor]], tri: Smooth | None, vel: Smooth2 | None, T: int,
                       layout: ClipLayout | None, device: torch.device) -> BlockTridiagonal:
    """`assemble_from_blocks` for (T,6+K,6+K) blocks, a K-column border per clip (module docstring)."""
    K = blocks[0][0].shape[-1] - 6
    D = torch.zeros(T, 6, 6, dtype=torch.float64, device=device)
    c = torch.zeros(T, 6, K, dtype=torch.float64, device=device)
    g = torch.zeros(T, 6, dtype=torch.float64, device=device)
    n_scale = () if layout is None else (layout.n_clips,)
    h_ss = torch.zeros(*n_scale, K, K, dtype=torch.float64, device=device)
    g_s = torch.zeros(*n_scale, K, dtype=torch.float64, device=device)
    for Hk, gk in blocks:                                                # (T,6+K,6+K), (T,6+K)
        D = D + Hk[:, :6, :6]
        c = c + Hk[:, :6, 6:]
        g = g + gk[:, :6]
        if layout is None:
            h_ss = h_ss + Hk[:, 6:, 6:].sum(0)
            g_s = g_s + gk[:, 6:].sum(0)
        else:
            h_ss = h_ss + _clip_sum_rows(Hk[:, 6:, 6:], layout.clip_index, layout.n_clips)
            g_s = g_s + _clip_sum_rows(gk[:, 6:], layout.clip_index, layout.n_clips)
    D, U, g, U2 = _add_priors(D, g, tri, vel, T, device)
    return BlockTridiagonal(D=D, U=U, c=c, h_ss=h_ss, g=g, g_s=g_s, U2=U2, layout=layout)


def _add_priors(D: Tensor, g: Tensor, tri: Smooth | None, vel: Smooth2 | None, T: int, device: torch.device
                ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """The temporal priors' pose-band contributions: `(D, U, g, U2)` (split out of `assemble_from_blocks` for the
    anisotropic scale; the operations are the historical ones, in the same order)."""
    if tri is not None:
        r, Jk, Jk1 = (x.to(torch.float64) for x in tri)
        JkT, Jk1T = Jk.transpose(-1, -2), Jk1.transpose(-1, -2)
        D = D.clone()
        D[:-1] += JkT @ Jk
        D[1:] += Jk1T @ Jk1
        U = JkT @ Jk1
        g = g.clone()
        g[:-1] += (JkT @ r.unsqueeze(-1)).squeeze(-1)
        g[1:] += (Jk1T @ r.unsqueeze(-1)).squeeze(-1)
    else:
        U = torch.zeros(max(T - 1, 0), 6, 6, dtype=torch.float64, device=device)
    U2: Tensor | None = None
    if vel is not None:
        rv, A, B, C = (x.to(torch.float64) for x in vel)
        AT, BT, CT = A.transpose(-1, -2), B.transpose(-1, -2), C.transpose(-1, -2)
        D, U, g = D.clone(), U.clone(), g.clone()
        D[:-2] += AT @ A
        D[1:-1] += BT @ B
        D[2:] += CT @ C
        U[:-1] += AT @ B
        U[1:] += BT @ C
        U2 = AT @ C
        rv1 = rv.unsqueeze(-1)
        g[:-2] += (AT @ rv1).squeeze(-1)
        g[1:-1] += (BT @ rv1).squeeze(-1)
        g[2:] += (CT @ rv1).squeeze(-1)
    return D, U, g, U2


def densify(sys: BlockTridiagonal) -> tuple[Tensor, Tensor]:
    """`BlockTridiagonal` -> dense `H` and `g` of size 6T + B (B = 1 without a layout; the scale of clip
    b sits at index 6T + b). O(T) Python loop: tests and small problems only, never an LM hot loop. An anisotropic
    K-column border gives size 6T + K B, clip b's border at 6T + K b ... 6T + K b + K - 1."""
    if sys.border > 1:
        return _densify_bordered(sys)
    T = sys.D.shape[0]
    lengths = sys.layout.lengths if sys.layout is not None else (T,)
    clip_of = [b for b, n in enumerate(lengths) for _ in range(n)]
    B = len(lengths)
    n = 6 * T + B
    device, dtype = sys.D.device, sys.D.dtype
    H = torch.zeros(n, n, dtype=dtype, device=device)
    g = torch.zeros(n, dtype=dtype, device=device)
    h_ss, g_s = sys.h_ss.reshape(B), sys.g_s.reshape(B)
    for k in range(T):
        sl = slice(6 * k, 6 * k + 6)
        col = 6 * T + clip_of[k]
        H[sl, sl] = sys.D[k]
        H[sl, col] = sys.c[k]
        H[col, sl] = sys.c[k]
        g[sl] = sys.g[k]
    for k in range(T - 1):
        a, b = slice(6 * k, 6 * k + 6), slice(6 * k + 6, 6 * k + 12)
        H[a, b] = sys.U[k]
        H[b, a] = sys.U[k].transpose(-1, -2)
    if sys.U2 is not None:
        for k in range(T - 2):
            a, b = slice(6 * k, 6 * k + 6), slice(6 * k + 12, 6 * k + 18)
            H[a, b] = sys.U2[k]
            H[b, a] = sys.U2[k].transpose(-1, -2)
    for j in range(B):
        H[6 * T + j, 6 * T + j] = h_ss[j]
        g[6 * T + j] = g_s[j]
    return H, g


def _densify_bordered(sys: BlockTridiagonal) -> tuple[Tensor, Tensor]:
    """`densify` of an anisotropic K-column border."""
    T, K = sys.D.shape[0], sys.border
    lengths = sys.layout.lengths if sys.layout is not None else (T,)
    clip_of = [b for b, n in enumerate(lengths) for _ in range(n)]
    B = len(lengths)
    n = 6 * T + K * B
    H = torch.zeros(n, n, dtype=sys.D.dtype, device=sys.D.device)
    g = torch.zeros(n, dtype=sys.D.dtype, device=sys.D.device)
    h_ss, g_s = sys.h_ss.reshape(B, K, K), sys.g_s.reshape(B, K)
    for k in range(T):
        sl, cols = slice(6 * k, 6 * k + 6), slice(6 * T + K * clip_of[k], 6 * T + K * clip_of[k] + K)
        H[sl, sl] = sys.D[k]
        H[sl, cols] = sys.c[k]
        H[cols, sl] = sys.c[k].T
        g[sl] = sys.g[k]
    for k in range(T - 1):
        a, b = slice(6 * k, 6 * k + 6), slice(6 * k + 6, 6 * k + 12)
        H[a, b] = sys.U[k]
        H[b, a] = sys.U[k].transpose(-1, -2)
    if sys.U2 is not None:
        for k in range(T - 2):
            a, b = slice(6 * k, 6 * k + 6), slice(6 * k + 12, 6 * k + 18)
            H[a, b] = sys.U2[k]
            H[b, a] = sys.U2[k].transpose(-1, -2)
    for j in range(B):
        cols = slice(6 * T + K * j, 6 * T + K * j + K)
        H[cols, cols] = h_ss[j]
        g[cols] = g_s[j]
    return H, g


def _scatter(n_rows: int, index: Tensor, src: Tensor, fill: Tensor | None = None) -> Tensor:
    """(n_rows, *src.shape[1:]) tensor holding `src` at rows `index` and `fill` (default zeros) elsewhere."""
    base = (torch.zeros(n_rows, *src.shape[1:], dtype=src.dtype, device=src.device) if fill is None
            else fill.expand(n_rows, *src.shape[1:]).contiguous())
    return base.index_copy(0, index, src)


def _clip_blocks(sys: BlockTridiagonal, layout: ClipLayout) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The system in the clip-parallel layout the recursion runs on: `(D (B,N,b,b), U (B,N-1,b,b),
    c (B,N,b), g (B,N,b))`, b = 6 (N = W) or, with a velocity prior, b = 12 over frame pairs (N = W/2).
    Padding frames are decoupled identity blocks; cross-clip coupling blocks are dropped (they are zero
    by construction). Results return per frame via `_frames`."""
    B, W = layout.n_clips, layout.width
    eye = torch.eye(6, dtype=sys.D.dtype, device=sys.D.device)
    D = _scatter(B * W, layout.slot, sys.D, eye).reshape(B, W, 6, 6)
    c = _scatter(B * W, layout.slot, sys.c).reshape(B, W, *sys.c.shape[1:])     # (B,W,6,K) for K columns
    g = _scatter(B * W, layout.slot, sys.g).reshape(B, W, 6)
    U = _scatter(B * (W - 1), layout.u_slot, sys.U.index_select(0, layout.pair_rows)).reshape(B, W - 1, 6, 6)
    if sys.U2 is None:
        return D, U, c, g
    U2 = _scatter(B * (W - 2), layout.u2_slot, sys.U2.index_select(0, layout.triple_rows))
    return pair_frames(D, U, U2.reshape(B, W - 2, 6, 6), c, g)


def _frames(x: Tensor, layout: ClipLayout) -> Tensor:
    """(B,N,b) clip-layout vectors -> (T,6) per-frame vectors (drops padding); K columns: (B,N,b,K) -> (T,6,K)."""
    return x.reshape(layout.n_clips * layout.width, 6, *x.shape[3:]).index_select(0, layout.slot)


def _per_clip(lam: float | Tensor, B: int, like: Tensor) -> Tensor:
    if isinstance(lam, Tensor):
        return lam.to(like.dtype).reshape(B)
    return torch.full((B,), lam, dtype=like.dtype, device=like.device)


def solve_block_tridiagonal(sys: BlockTridiagonal, lam: float | Tensor, floor: float = 1e-9
                            ) -> tuple[Tensor, Tensor]:
    """Solve `(H + lam*diag(H) + floor*I) delta = -g` without forming the dense `H`: block-Thomas
    eliminates the pose block (`A x = c`, `A y = -g_pose` in one forward/backward pass over both
    right-hand sides), then each clip's scale is recovered by its own rank-one Schur complement
    `delta_sigma_b = (-g_s_b - c_b.y_b) / (h_ss_b' - c_b.x_b)`, `delta_pose = y - delta_sigma * x`. `lam`
    is a float or a (B,) tensor, one damping per clip. Returns `(delta_pose (T,6), delta_sigma)`
    on the system's device, `delta_sigma` (B,) with a layout and 0-dim without. Runs on the system's
    device with no host synchronization; differentiable (plain torch throughout).

    Anisotropic scale (a K-column border): one pass over K + 1 right-hand sides (`A X = c`, `A y = -g`), then each
    clip's KxK Schur complement `S_b = h_ss_b' - c_b^T X_b` (h_ss' damped like every diagonal: `lam * diag` plus the
    floor) gives `delta_l_b = S_b^{-1} (-g_s_b - c_b^T y_b)`, `delta_pose = y - X delta_l`; `delta_sigma` is then (B,K)
    or (K,)."""
    layout = sys.layout if sys.layout is not None else ClipLayout.build((sys.D.shape[0],), sys.D.device)
    B = layout.n_clips
    D, U, c, g = _clip_blocks(sys, layout)
    lam_b = _per_clip(lam, B, sys.D)
    if sys.border > 1:
        return _solve_bordered(sys, layout, D, U, c, g, lam_b, floor)
    _Dtilde, L, M = forward_schur(D, U, lam_b, floor)
    X = thomas_solve(L, M, torch.stack([c, -g], dim=-1))               # (B,N,b,2)
    x, y = X[..., 0], X[..., 1]                                          # A x = c, A y = -g
    h_ss_damped = sys.h_ss.reshape(B) * (1.0 + lam_b) + floor
    cx = (c * x).sum(dim=(1, 2))
    cy = (c * y).sum(dim=(1, 2))
    delta_sigma = (-sys.g_s.reshape(B) - cy) / (h_ss_damped - cx)
    delta_pose = _frames(y - delta_sigma[:, None, None] * x, layout)
    return delta_pose, (delta_sigma if sys.layout is not None else delta_sigma.reshape(()))


def _damped_border(h_ss: Tensor, lam_b: Tensor, floor: float) -> Tensor:
    """(B,K,K) border corner with `lam * diag` and the floor on its diagonal, the pose blocks' damping convention."""
    eye = torch.eye(h_ss.shape[-1], dtype=h_ss.dtype, device=h_ss.device)
    return h_ss + lam_b[:, None, None] * torch.diag_embed(torch.diagonal(h_ss, dim1=-2, dim2=-1)) + floor * eye


def _solve_bordered(sys: BlockTridiagonal, layout: ClipLayout, D: Tensor, U: Tensor, c: Tensor, g: Tensor,
                    lam_b: Tensor, floor: float) -> tuple[Tensor, Tensor]:
    """`solve_block_tridiagonal` for an anisotropic K-column border (see its docstring)."""
    B, K = layout.n_clips, sys.border
    _Dtilde, L, M = forward_schur(D, U, lam_b, floor)
    X = thomas_solve(L, M, torch.cat([c, -g.unsqueeze(-1)], dim=-1))    # (B,N,b,K+1)
    x, y = X[..., :K], X[..., K]                                         # A x = c, A y = -g
    S = _damped_border(sys.h_ss.reshape(B, K, K), lam_b, floor) - torch.einsum("bnik,bnil->bkl", c, x)
    rhs = -sys.g_s.reshape(B, K) - torch.einsum("bnik,bni->bk", c, y)
    delta_l = torch.linalg.solve_ex(S, rhs.unsqueeze(-1)).result.squeeze(-1)   # no info read-back (host sync)
    delta_pose = _frames(y - torch.einsum("bnik,bk->bni", x, delta_l), layout)
    return delta_pose, (delta_l if sys.layout is not None else delta_l.reshape(K))


def marginal_covariances(sys: BlockTridiagonal, lam: float = 0.0, floor: float = 1e-9) -> Tensor:
    """Per-frame (T,6,6) marginal covariance blocks of `(H + lam*diag(H) + floor*I)^{-1}` (a Laplace
    approximation's precision matrix), via the O(T) backward RTS recursion on the forward pass's Schur
    factors plus each clip's scale rank-one correction: `Sigma_k += x_k x_k^T /
    (h_ss_b' - c_b.x_b)` with `x` solving `A x = c`. Never forms the dense matrix or its pseudo-inverse.
    Runs on CPU float64 (once per solve, where the sequential recursion is fastest) and returns on the
    system's device; `.to("cpu", torch.float64)` keeps the autograd graph intact. With a velocity prior
    each frame's 6x6 marginal is a diagonal sub-block of its frame pair's 12x12 marginal."""
    device = sys.D.device
    T = sys.D.shape[0]
    moved = {name: getattr(sys, name).to("cpu", torch.float64) for name in ("D", "U", "c", "g", "h_ss", "g_s")}
    U2 = sys.U2.to("cpu", torch.float64) if sys.U2 is not None else None
    layout = ClipLayout.build(sys.layout.lengths if sys.layout is not None else (T,), "cpu")
    cpu_sys = replace(sys, U2=U2, layout=layout, **moved)
    B, W = layout.n_clips, layout.width
    D, U, c, _g = _clip_blocks(cpu_sys, layout)
    lam_b = _per_clip(lam, B, cpu_sys.D)
    _Dtilde, L, M = forward_schur(D, U, lam_b, floor)
    Sigma = rts_covariances(L, M)                                        # (B,N,b,b)
    if U2 is not None:                                                   # frame-pair super-blocks
        Sigma = torch.stack([Sigma[..., :6, :6], Sigma[..., 6:, 6:]], dim=2)
    Sigma_f = Sigma.reshape(B * W, 6, 6).index_select(0, layout.slot)
    if cpu_sys.border > 1:                                               # rank-K border correction
        K = cpu_sys.border
        xk = thomas_solve(L, M, c)                                       # (B,N,b,K), A x = c
        S = _damped_border(cpu_sys.h_ss.reshape(B, K, K), lam_b, floor) - torch.einsum("bnik,bnil->bkl", c, xk)
        xf = _frames(xk, layout)                                         # (T,6,K)
        S_inv = torch.linalg.inv(S)[layout.clip_index]
        return (Sigma_f + xf @ S_inv @ xf.transpose(-1, -2)).to(device=device)
    xb = thomas_solve(L, M, c.unsqueeze(-1)).squeeze(-1)                 # A x = c
    denom = cpu_sys.h_ss.reshape(B) * (1.0 + lam_b) + floor - (c * xb).sum(dim=(1, 2))
    x = _frames(xb, layout)
    Sigma_f = Sigma_f + torch.einsum("ki,kj->kij", x, x) / denom[layout.clip_index][:, None, None]
    return Sigma_f.to(device=device)
