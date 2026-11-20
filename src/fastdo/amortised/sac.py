"""Learner: an off-policy max-entropy actor-critic that replicates FlashSAC's TARGET exactly.

Our own re-implementation (FlashDexRetarget MIT; `rsl_rl_flashsac` BSD-3-Clause, Holiday Robotics / RSL-RL), checked
against the vendored modules imported read-only on fixed batches (`tests/amortised/test_m4_sac.py`).

Target semantics (rsl_rl_flashsac @ FDR 7919f8ae: `storage/torch_buffer.py::_get_n_step_prev_transition`,
`utils/categorical.py`, `algorithms/flash_sac.py::_update_critic/_update_actor/_update_temperature`):
- n-step aggregation with FIRST-DONE truncation (`nstep_aggregate`): j* = first j with term_j or trunc_j (n - 1 if
  none); R = sum_{j <= j*} gamma^j r_j (rewards after the first done are dropped); next obs, term and trunc are those
  of step j*;
- bootstrap mask: TRUE terminations only (a time-out bootstraps); the bootstrap discount is the fixed gamma^n even
  when j* < n - 1 (as the reference);
- distributional target (`categorical_target`): atoms z_1..z_M uniform on [-G_max, G_max]; target atom values
  clip(R + gamma^n (z_j - alpha log pi(a'|o')) (1 - term)) (the entropy shift INSIDE the atoms); the probabilities of
  the critic with the lower MEAN (`select_min_q`) are projected onto the support by linear splitting (C51);
- critic loss: cross-entropy of every critic's predicted distribution against that target, averaged; the critic
  forward is ONE joint pass over (o, a) and (o', a') so BatchNorm statistics cover both halves (CrossQ); the next
  action is sampled with BatchNorm in eval mode, the target critic runs in train mode;
- actor loss mean(alpha log pi - min(Q_1, Q_2)) on a joint actor pass over (o, o') (only the first half enters);
  temperature loss alpha (H - H_target), H_target = 1/2 A log(2 pi e sigma^2) (`target_entropy`, FlashSAC's
  `temp_target_sigma` rule); actor and temperature every `actor_update_period` critic updates, before the critic;
- Polyak tau after every critic update; return-based reward normalisation (`RewardNormalizer`, FlashSAC's);
- learning rates: FlashSAC's warmup-cosine schedule (`warmup_cosine`), stepped PER OPTIMIZER as its `LambdaLR`s are:
  the critic's by critic updates, the actor's and the temperature's by actor updates (every `actor_update_period`).
- UTD per TRANSITION (`UtdCounter`): u = 2/2048 gradient updates per environment transition (FDR: 2 updates per
  vectorised step at 2,048 worlds); at W worlds a vectorised step earns u W updates (8 at 8,192).
Not replicated (networks are ours): FlashSAC's unit-norm weight re-normalisation, AMP and torch.compile; the
exploration-noise repeat (zeta) belongs to the acting loop.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "SAC",
    "Actor",
    "Critic",
    "RewardNormalizer",
    "SacConfig",
    "UtdCounter",
    "actor_loss",
    "categorical_target",
    "critic_loss",
    "nstep_aggregate",
    "select_min_q",
    "split_dones",
    "tanh_log_det_jacobian",
    "target_entropy",
    "temperature_loss",
    "warmup_cosine",
]


@dataclass(frozen=True)
class SacConfig:
    gamma: float = 0.98 ** (5 * 0.005 * 60)     # per policy step of k = 5 rows (0.970)
    n_steps: int = 5
    num_bins: int = 101
    g_max: float = 5.0
    tau: float = 0.01
    num_qs: int = 2
    actor_hidden: int = 128
    actor_blocks: int = 2
    critic_hidden: int = 1024
    critic_blocks: int = 2
    actor_update_period: int = 2
    batch_size: int = 4096
    lr: float = 3e-4                            # peak
    lr_init: float = 3e-4
    lr_end: float = 1.5e-4
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 1_000_000
    temp_init: float = 0.01
    temp_target_sigma: float = 0.15
    updates_per_transition: float = 2.0 / 2048.0
    normalize_reward: bool = True
    log_std_min: float = -10.0
    log_std_max: float = 2.0


# ------------------------------------------------------------------------------------------------ target pieces


def nstep_aggregate(rew: Tensor, term: Tensor, trunc: Tensor, gamma: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(R, j*, term at j*, trunc at j*) of windows rew/term/trunc (n, B) (module docstring)."""
    n = rew.shape[0]
    done = (term.bool() | trunc.bool())
    alive = torch.ones_like(rew)
    R = torch.zeros_like(rew[0])
    for j in range(n):
        R = R + (gamma ** j) * rew[j] * alive[j]
        if j + 1 < n:
            alive[j + 1] = alive[j] * (~done[j]).to(rew.dtype)
    first = torch.where(done.any(0), done.to(torch.int64).argmax(0), torch.full_like(done[0], n - 1, dtype=torch.int64))
    pick = lambda x: x.gather(0, first[None])[0]
    return R, first, pick(term.to(rew.dtype)), pick(trunc.to(rew.dtype))


def split_dones(dones: Tensor, time_outs: Tensor | None) -> tuple[Tensor, Tensor]:
    """(terminated, truncated) from the env's episode ends and its time-out flags: a time-out is a TRUNCATION (it
    bootstraps), every other episode end a termination (FlashSAC `process_env_step`)."""
    dones = dones.float().reshape(-1)
    trunc = torch.zeros_like(dones) if time_outs is None else time_outs.float().reshape(-1)
    return (dones - trunc).clamp(min=0.0), trunc


def select_min_q(next_qs: Tensor, next_log_probs: Tensor) -> Tensor:
    """Log-probs (B, M) of the critic with the lower mean Q per sample: next_qs (Q, B), next_log_probs (Q, B, M)."""
    idx = next_qs.argmin(dim=0)
    return next_log_probs.gather(0, idx[None, :, None].expand(1, -1, next_log_probs.shape[-1]))[0]


def categorical_target(target_log_probs: Tensor, reward: Tensor, term: Tensor, entropy_term: Tensor,
                       gamma_n: float, num_bins: int, g_max: float) -> Tensor:
    """C51 projection of clip(R + gamma^n (z - alpha log pi)(1 - term)) with the selected distribution's probabilities.
    `entropy_term` = alpha log pi(a'|o') (B,)."""
    B = reward.shape[0]
    width = 2.0 * g_max / (num_bins - 1)
    z = torch.linspace(-g_max, g_max, num_bins, dtype=target_log_probs.dtype, device=target_log_probs.device)
    zt = reward[:, None] + gamma_n * (z[None, :] - entropy_term[:, None]) * (1.0 - term[:, None])
    zt = zt.clamp(-g_max, g_max)
    b = (zt + g_max) / width
    lo = b.floor().long()
    hi = (lo + 1).clamp(0, num_bins - 1)
    frac = b - lo.to(b.dtype)
    p = target_log_probs.exp()
    out = torch.zeros(B, num_bins, dtype=p.dtype, device=p.device)
    out.scatter_add_(1, lo, p * (1.0 - frac))
    out.scatter_add_(1, hi, p * frac)
    return out


def critic_loss(target_probs: Tensor, pred_log_probs: Tensor) -> Tensor:
    """Mean over critics and samples of -sum_j target_j log p_j; pred_log_probs (Q, B, M)."""
    return -(target_probs[None] * pred_log_probs).sum(-1).mean()


def actor_loss(log_probs: Tensor, qs: Tensor, alpha: Tensor) -> Tensor:
    return (log_probs * alpha - torch.minimum(qs[0], qs[1])).mean()


def temperature_loss(temp: Tensor, entropy: Tensor, target: float) -> Tensor:
    return temp * (entropy.detach() - target).mean()


def target_entropy(action_dim: int, sigma: float) -> float:
    return 0.5 * action_dim * math.log(2.0 * math.pi * math.e * sigma ** 2)


def warmup_cosine(step: int, init: float, peak: float, end: float, warmup: int, decay: int) -> float:
    """FlashSAC's `warmup_cosine_decay_scheduler`: linear warmup init -> peak, cosine decay to end at `decay`."""
    if step < warmup:
        return init + (peak - init) * (step / warmup)
    if step < decay:
        return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (decay - warmup)))
    return end


def tanh_log_det_jacobian(x: Tensor) -> Tensor:
    """log(1 - tanh(x)^2), stable: 2 (log 2 - x - softplus(-2x))."""
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class RewardNormalizer:
    """Divide rewards by max(sqrt(Var[G_running] + eps), max|G_running| / G_max) (FlashSAC's `normalize_reward`)."""

    def __init__(self, gamma: float, g_max: float, eps: float = 1e-8, rms_eps: float = 1e-4) -> None:
        self.gamma, self.g_max, self.eps, self.rms_eps = gamma, g_max, eps, rms_eps
        self.G = torch.zeros(1)
        self.G_max = torch.zeros(1)
        self.mean, self.var, self.count = torch.zeros(1), torch.ones(1), torch.tensor(0.0)

    def update(self, reward: Tensor, term: Tensor, trunc: Tensor) -> None:
        done = (term.bool() | trunc.bool()).float()
        self.G = self.gamma * (1.0 - done) * self.G + reward
        self.G_max = torch.maximum(self.G_max, self.G.abs().max())
        x = self.G.reshape(-1, 1)
        n = torch.tensor(float(x.shape[0]))
        m = x.sum(0) / n
        v = (x.square().sum(0) / n - m.square()).clamp(min=0.0)
        delta = m - self.mean
        tot = self.count + n
        ratio = n / tot
        M2 = self.var * (self.count + self.rms_eps) + v * n + delta.square() * self.count * ratio
        self.mean, self.var, self.count = self.mean + delta * ratio, M2 / tot, tot

    def normalize(self, r: Tensor) -> Tensor:
        return r / torch.maximum(torch.sqrt(self.var + self.eps), self.G_max / self.g_max)

    def state_dict(self) -> dict:
        return {k: getattr(self, k).clone() for k in ("G", "G_max", "mean", "var", "count")}

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v.clone())


class UtdCounter:
    """Gradient updates per vectorised step at a FIXED rate u per transition (fractional credit carried over)."""

    def __init__(self, u: float, n_worlds: int) -> None:
        self.u, self.n_worlds, self.credit = float(u), int(n_worlds), 0.0

    def step(self) -> int:
        self.credit += self.u * self.n_worlds
        k = math.floor(self.credit + 1e-9)
        self.credit -= k
        return k


# ------------------------------------------------------------------------------------------------ networks


class _BN(nn.Module):
    """BatchNorm whose mode is chosen per call (train statistics or running statistics), momentum 0.01."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: Tensor, training: bool) -> Tensor:
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training, 0.01, 1e-5)


class _Trunk(nn.Module):
    """Input BN -> Linear -> residual blocks (BN, Linear 4h, ReLU, Linear h) -> LayerNorm."""

    def __init__(self, in_dim: int, hidden: int, blocks: int) -> None:
        super().__init__()
        self.bn_in = _BN(in_dim)
        self.embed = nn.Linear(in_dim, hidden)
        self.bns = nn.ModuleList([_BN(hidden) for _ in range(blocks)])
        self.up = nn.ModuleList([nn.Linear(hidden, 4 * hidden) for _ in range(blocks)])
        self.down = nn.ModuleList([nn.Linear(4 * hidden, hidden) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor, training: bool) -> Tensor:
        h = self.embed(self.bn_in(x, training))
        for bn, up, down in zip(self.bns, self.up, self.down, strict=True):
            h = h + down(F.relu(up(bn(h, training))))
        return self.norm(h)


class Actor(nn.Module):
    """Tanh-Gaussian policy; log std squashed into [log_std_min, log_std_max]; actions in [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: SacConfig) -> None:
        super().__init__()
        self.trunk = _Trunk(obs_dim, cfg.actor_hidden, cfg.actor_blocks)
        self.mean = nn.Linear(cfg.actor_hidden, act_dim)
        self.std = nn.Linear(cfg.actor_hidden, act_dim)
        self.lo, self.hi = cfg.log_std_min, cfg.log_std_max
        self.act_dim = act_dim

    def mean_std(self, obs: Tensor, training: bool) -> tuple[Tensor, Tensor]:
        h = self.trunk(obs, training)
        log_std = self.lo + (self.hi - self.lo) * 0.5 * (1.0 + torch.tanh(self.std(h)))
        return self.mean(h), log_std.exp()

    def sample(self, obs: Tensor, training: bool, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mean, std = self.mean_std(obs, training)
        eps = torch.randn_like(mean) if noise is None else noise
        raw = mean + std * eps
        logp = torch.distributions.Normal(mean, std).log_prob(raw) - tanh_log_det_jacobian(raw)
        return torch.tanh(raw), logp.sum(-1)


class Critic(nn.Module):
    """`num_qs` categorical critics; forward -> (Q (num_qs, B), log-probs (num_qs, B, M))."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: SacConfig) -> None:
        super().__init__()
        self.nets = nn.ModuleList([nn.ModuleDict({"trunk": _Trunk(obs_dim + act_dim, cfg.critic_hidden,
                                                                   cfg.critic_blocks),
                                                  "head": nn.Linear(cfg.critic_hidden, cfg.num_bins)})
                                   for _ in range(cfg.num_qs)])
        self.register_buffer("z", torch.linspace(-cfg.g_max, cfg.g_max, cfg.num_bins))

    def forward(self, obs: Tensor, act: Tensor, training: bool) -> tuple[Tensor, Tensor]:
        x = torch.cat([obs, act], -1)
        logp = torch.stack([F.log_softmax(n["head"](n["trunk"](x, training)), -1) for n in self.nets])
        return (logp.exp() * self.z).sum(-1), logp


# ------------------------------------------------------------------------------------------------ the learner


class SAC:
    """Actor, online and target critic, temperature, optimizers, reward normaliser (module docstring)."""

    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, act_dim: int, cfg: SacConfig | None = None,
                 device: str = "cpu") -> None:
        cfg = cfg if cfg is not None else SacConfig()
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = Actor(actor_obs_dim, act_dim, cfg).to(self.device)
        self.critic = Critic(critic_obs_dim, act_dim, cfg).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.log_temp = torch.tensor([math.log(cfg.temp_init)], device=self.device, requires_grad=True)
        self.h_target = target_entropy(act_dim, cfg.temp_target_sigma)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.opt_temp = torch.optim.Adam([self.log_temp], lr=cfg.lr)
        self.normalizer = RewardNormalizer(cfg.gamma ** 1, cfg.g_max) if cfg.normalize_reward else None
        self.update_step = 0
        self.actor_step = 0

    @property
    def temperature(self) -> Tensor:
        return self.log_temp.exp()

    def _lr(self, step: int) -> float:
        c = self.cfg
        return warmup_cosine(step, c.lr_init, c.lr, c.lr_end, c.lr_warmup_steps, c.lr_decay_steps)

    @staticmethod
    def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
        for g in opt.param_groups:
            g["lr"] = lr

    def actor_step_loss(self, b: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """(actor loss, log pi of the first half) from one joint actor pass over (o, o')."""
        B = b["action"].shape[0]
        a_all, lp_all = self.actor.sample(torch.cat([b["actor_obs"], b["next_actor_obs"]], 0), training=True)
        a, lp = a_all[:B], lp_all[:B]
        self.critic.requires_grad_(False)
        qs, _ = self.critic(b["critic_obs"], a, training=False)
        self.critic.requires_grad_(True)
        return actor_loss(lp, qs, self.temperature.detach()), lp

    def critic_step_loss(self, b: dict[str, Tensor], rewards: Tensor) -> Tensor:
        cfg = self.cfg
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(b["next_actor_obs"], training=False)
            ent = self.temperature.detach() * next_logp
            obs_all = torch.cat([b["critic_obs"], b["next_critic_obs"]], 0)
            act_all = torch.cat([b["action"], next_a], 0)
            qs_all, lp_all = self.critic_target(obs_all, act_all, training=True)
            B = b["action"].shape[0]
            sel = select_min_q(qs_all[:, B:], lp_all[:, B:])
            target = categorical_target(sel, rewards, b["terminated"], ent, cfg.gamma ** cfg.n_steps, cfg.num_bins,
                                        cfg.g_max)
        _, pred = self.critic(obs_all, act_all, training=True)
        return critic_loss(target, pred[:, :B])

    def update(self, b: dict[str, Tensor]) -> dict[str, float]:
        """One gradient update on a sampled n-step batch: keys actor_obs, critic_obs, next_actor_obs,
        next_critic_obs, action (in [-1, 1]), reward (n-step aggregated), terminated."""
        cfg = self.cfg
        # the normaliser's running state lives on the host; the target is formed on the learner's device (a no-op
        # on the CPU path)
        rewards = (self.normalizer.normalize(b["reward"].cpu()) if self.normalizer is not None
                   else b["reward"]).to(self.device)
        out: dict[str, float] = {}
        if self.update_step % cfg.actor_update_period == 0:
            self._set_lr(self.opt_actor, self._lr(self.actor_step))
            self._set_lr(self.opt_temp, self._lr(self.actor_step))
            la, lp = self.actor_step_loss(b)
            self.opt_actor.zero_grad(set_to_none=True)
            la.backward()
            self.opt_actor.step()
            ent = -lp.detach().mean()
            lt = temperature_loss(self.temperature, ent, self.h_target)
            self.opt_temp.zero_grad(set_to_none=True)
            lt.backward()
            self.opt_temp.step()
            self.actor_step += 1
            out.update(actor=float(la.detach()), temperature=float(lt.detach()), entropy=float(ent))
        self._set_lr(self.opt_critic, self._lr(self.update_step))
        lc = self.critic_step_loss(b, rewards)
        self.opt_critic.zero_grad(set_to_none=True)
        lc.backward()
        self.opt_critic.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(self.critic_target.parameters()), list(self.critic.parameters()), cfg.tau)
        self.update_step += 1
        out["critic"] = float(lc.detach())
        return out

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(), "log_temp": self.log_temp.detach().clone(),
                "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
                "opt_temp": self.opt_temp.state_dict(), "update_step": self.update_step, "actor_step": self.actor_step,
                "normalizer": self.normalizer.state_dict() if self.normalizer is not None else None}

    def load_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        with torch.no_grad():
            self.log_temp.copy_(d["log_temp"])
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        self.opt_temp.load_state_dict(d["opt_temp"])
        self.update_step = int(d["update_step"])
        self.actor_step = int(d["actor_step"])
        if self.normalizer is not None and d.get("normalizer") is not None:
            self.normalizer.load_state_dict(d["normalizer"])
