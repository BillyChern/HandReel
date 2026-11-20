"""Conditional flow-matching residual prior over knot-space control residuals.

Rectified flow, sigma_min = 0, with the convention used everywhere in this module: tau = 0 is NOISE and tau = 1 is
DATA. x_tau = (1 - tau) x0 + tau x1, x0 ~ N(0, I), tau ~ U[0, 1]; the network regresses the velocity x1 - x0:
    L = E || m (.) (v_theta(x_tau, tau, c) - (x1 - x0)) ||^2   (mean over valid knots and dimensions).
Sampling integrates dx/dtau = v from tau = 0 (a standard-normal draw) to tau = 1 with `n_euler` (16) explicit Euler
steps. Every sample index s draws its x0 from its own generator seeded by (seed, s), so the first S samples are the
same whatever the total requested ("prefix-consistent seeds", the pass@S curve).

Network (`FlowNet`): a 1-D temporal U-Net over knots (`levels` down/up stages, residual conv blocks with FiLM), the
per-knot conditioning concatenated to x at the input, and one FiLM vector per sample = an MLP of the sinusoidal tau
embedding + an MLP of the global conditioning (scalars: mass, bounding box, side; plus a PointNet over the object
point cloud when given). GNR's architecture class with PointNet in place of PTv3.

Padding invariance: training pads every clip to the data's K_max while sampling runs at the clip's own K, so the
network must compute the SAME function of the valid knots whatever the padding. Every activation is zeroed on the
padded knots after every convolution, pooling and upsampling (the mask is max-pooled to each level), and GroupNorm's
statistics are taken over the valid knots only (`_masked_group_norm`). A valid prefix of K knots then sees exactly the
zero boundary it sees at its own length, at every level (the stride-2 pooling is aligned at knot 0), so v on the valid
knots does not depend on the pad length (float summation order aside). Padded knots are also masked at the output and
in the loss.

Training data (`PriorData`, `sample_batch`): clips are drawn UNIFORMLY, then one solution uniformly within the clip,
so a clip with many solutions does not dominate.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["FlowConfig", "FlowNet", "PriorClip", "PriorData", "flow_loss", "sample", "sample_batch", "train_prior"]


@dataclass(frozen=True)
class FlowConfig:
    R: int                      # residual dimensions
    knot_cond: int              # per-knot conditioning width
    glob_cond: int              # global scalar conditioning width
    use_points: bool = False    # PointNet over the object cloud
    point_feat: int = 128
    channels: int = 64
    levels: int = 2
    film: int = 128
    n_euler: int = 16


class PointNet(nn.Module):
    def __init__(self, out: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, out))

    def forward(self, pts: Tensor) -> Tensor:          # (B, P, 3) -> (B, out)
        return self.mlp(pts).max(dim=1).values


def _masked_group_norm(gn: nn.GroupNorm, h: Tensor, m: Tensor) -> Tensor:
    """GroupNorm with statistics over the VALID positions only (m (B, 1, L) in {0, 1}); padded positions are zeroed.
    With every position valid it equals `gn(h)` (biased variance, the same eps and affine)."""
    B, C, L = h.shape
    G = gn.num_groups
    hg = h.reshape(B, G, C // G, L)
    mg = m.reshape(B, 1, 1, L)
    n = (mg.sum((2, 3), keepdim=True) * (C // G)).clamp(min=1.0)
    mean = (hg * mg).sum((2, 3), keepdim=True) / n
    var = (((hg - mean) * mg) ** 2).sum((2, 3), keepdim=True) / n
    out = ((hg - mean) / torch.sqrt(var + gn.eps)).reshape(B, C, L)
    if gn.affine:
        out = out * gn.weight[None, :, None] + gn.bias[None, :, None]
    return out * m


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, film: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.c2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, c_out), c_out)
        self.n2 = nn.GroupNorm(min(8, c_out), c_out)
        self.film = nn.Linear(film, 2 * c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: Tensor, e: Tensor, m: Tensor) -> Tensor:
        """x (B, C, L) is zero on the padded positions (m == 0); so is the output (module docstring)."""
        h = _masked_group_norm(self.n1, self.c1(x), m)
        s, b = self.film(e).chunk(2, -1)
        h = F.silu(h * (1.0 + s[..., None]) + b[..., None]) * m
        h = F.silu(_masked_group_norm(self.n2, self.c2(h), m)) * m
        return (h + self.skip(x)) * m


def _sinusoid(tau: Tensor, dim: int) -> Tensor:
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, dtype=tau.dtype) / half)
    a = tau[:, None] * 100.0 * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class FlowNet(nn.Module):
    def __init__(self, cfg: FlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        C, E = cfg.channels, cfg.film
        self.time = nn.Sequential(nn.Linear(64, E), nn.SiLU(), nn.Linear(E, E))
        g_in = cfg.glob_cond + (cfg.point_feat if cfg.use_points else 0)
        self.points = PointNet(cfg.point_feat) if cfg.use_points else None
        self.glob = nn.Sequential(nn.Linear(max(g_in, 1), E), nn.SiLU(), nn.Linear(E, E))
        self.inp = nn.Conv1d(cfg.R + cfg.knot_cond, C, 3, padding=1)
        chans = [C * 2 ** i for i in range(cfg.levels + 1)]
        self.down = nn.ModuleList([_Block(chans[i], chans[i + 1], E) for i in range(cfg.levels)])
        self.pool = nn.ModuleList([nn.Conv1d(chans[i + 1], chans[i + 1], 3, stride=2, padding=1)
                                   for i in range(cfg.levels)])
        self.mid = _Block(chans[-1], chans[-1], E)
        self.up = nn.ModuleList([_Block(chans[i + 1] * 2, chans[i], E) for i in reversed(range(cfg.levels))])
        self.out = nn.Conv1d(C, cfg.R, 3, padding=1)

    def _conditioning(self, kcond: Tensor, gcond: Tensor, points: Tensor | None) -> tuple[Tensor, Tensor]:
        g = gcond if self.points is None or points is None else torch.cat([gcond, self.points(points)], -1)
        if g.shape[-1] == 0:
            g = torch.zeros(g.shape[0], 1)
        return kcond, self.glob(g)

    def forward(self, x: Tensor, tau: Tensor, kcond: Tensor, gcond: Tensor, mask: Tensor,
                points: Tensor | None = None) -> Tensor:
        K = x.shape[1]
        kcond, gvec = self._conditioning(kcond, gcond, points)
        e = self.time(_sinusoid(tau, 64)) + gvec
        m = mask[..., None]
        h = torch.cat([x * m, kcond * m], -1).transpose(1, 2)            # (B, C, K)
        step = 2 ** self.cfg.levels
        Kp = -(-K // step) * step
        h = F.pad(h, (0, Kp - K))
        ms = [F.pad(mask.to(h.dtype), (0, Kp - K))[:, None]]               # (B, 1, L) per level
        for _ in range(self.cfg.levels):
            ms.append(F.max_pool1d(ms[-1], 2, 2))
        h = self.inp(h) * ms[0]
        skips = []
        for lvl, (blk, pool) in enumerate(zip(self.down, self.pool, strict=True)):
            h = blk(h, e, ms[lvl])
            skips.append(h)
            h = pool(h) * ms[lvl + 1]
        h = self.mid(h, e, ms[-1])
        for lvl, blk in zip(reversed(range(self.cfg.levels)), self.up, strict=True):
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-1], mode="nearest") * ms[lvl]
            h = blk(torch.cat([h, s], 1), e, ms[lvl])
        return self.out(h)[..., :K].transpose(1, 2) * m


def flow_loss(net: FlowNet, x1: Tensor, kcond: Tensor, gcond: Tensor, mask: Tensor, gen: torch.Generator,
              points: Tensor | None = None) -> Tensor:
    """Rectified-flow loss (module docstring) on a batch of scaled knot residuals x1 (B, K, R)."""
    x0 = torch.randn(x1.shape, generator=gen)
    tau = torch.rand(x1.shape[0], generator=gen)
    t = tau[:, None, None]
    xt = (1.0 - t) * x0 + t * x1
    v = net(xt, tau, kcond, gcond, mask, points)
    m = mask[..., None]
    return (((v - (x1 - x0)) * m) ** 2).sum() / (m.sum() * x1.shape[-1]).clamp(min=1.0)


def _noise(seed: int, index: int, shape: tuple[int, ...]) -> Tensor:
    h = hashlib.sha256(f"fastdo.amortised.flow_prior:{int(seed)}:{int(index)}".encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
    return torch.randn(shape, generator=g)


@torch.no_grad()
def sample(net: FlowNet, kcond: Tensor, gcond: Tensor, mask: Tensor, n: int, seed: int,
           points: Tensor | None = None, start: int = 0) -> Tensor:
    """n samples (n, K, R) for ONE clip's conditioning (kcond (K, Ck), gcond (G,), mask (K,)); sample index
    start + s uses x0 from seed (seed, start + s) (prefix-consistent). Euler from tau = 0 to 1."""
    net.eval()
    K = kcond.shape[0]
    x = torch.stack([_noise(seed, start + s, (K, net.cfg.R)) for s in range(n)])
    kc, gc, mk = kcond[None].expand(n, -1, -1), gcond[None].expand(n, -1), mask[None].expand(n, -1)
    pts = None if points is None else points[None].expand(n, -1, -1)
    N = net.cfg.n_euler
    for i in range(N):
        tau = torch.full((n,), i / N)
        x = x + net(x, tau, kc, gc, mk, pts) / N
    return x * mk[..., None]


@dataclass
class PriorClip:
    """One clip's conditioning and its solutions (scaled knot residuals, each (K, R))."""

    kcond: np.ndarray                 # (K, Ck)
    gcond: np.ndarray                 # (G,)
    solutions: list[np.ndarray] = field(default_factory=list)
    points: np.ndarray | None = None  # (P, 3)


@dataclass
class PriorData:
    clips: list[PriorClip]

    @property
    def k_max(self) -> int:
        return max(c.kcond.shape[0] for c in self.clips)


def sample_batch(data: PriorData, batch: int, rng: np.random.Generator) -> tuple[Tensor, ...]:
    """(x1, kcond, gcond, mask, points | None, clip ids): clips uniformly, then a solution uniformly within the clip."""
    usable = [i for i, c in enumerate(data.clips) if c.solutions]
    ids = np.asarray(usable)[rng.integers(0, len(usable), batch)]
    K = data.k_max
    c0 = data.clips[usable[0]]
    R, Ck, G = c0.solutions[0].shape[1], c0.kcond.shape[1], c0.gcond.shape[0]
    x1, kc, mk = np.zeros((batch, K, R)), np.zeros((batch, K, Ck)), np.zeros((batch, K))
    gc = np.zeros((batch, G))
    pts = None if c0.points is None else np.zeros((batch, *c0.points.shape))
    for b, i in enumerate(ids):
        c = data.clips[i]
        k = c.kcond.shape[0]
        x1[b, :k] = c.solutions[int(rng.integers(0, len(c.solutions)))]
        kc[b, :k], mk[b, :k], gc[b] = c.kcond, 1.0, c.gcond
        if pts is not None:
            pts[b] = c.points
    f = lambda a: torch.as_tensor(a, dtype=torch.float32)
    return f(x1), f(kc), f(gc), f(mk), (None if pts is None else f(pts)), ids


def train_prior(net: FlowNet, data: PriorData, steps: int, batch: int = 128, lr: float = 1e-3, seed: int = 0,
                log_every: int = 0) -> list[float]:
    """Adam on `flow_loss` with batches from `sample_batch`; returns the loss trace (every `log_every` steps)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.1)
    net.train()
    trace = []
    for s in range(steps):
        x1, kc, gc, mk, pts, _ = sample_batch(data, batch, rng)
        loss = flow_loss(net, x1, kc, gc, mk, gen, pts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and s % log_every == 0:
            trace.append(float(loss.detach()))
    return trace
