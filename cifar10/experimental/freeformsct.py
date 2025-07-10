from dataclasses import dataclass
from typing import Callable, Iterable, Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import (
    ForwardProcessSupports,
    PredictionSupports,
    VelocitySupports,
)
from flowmodels.cm import MultistepConsistencySampler
from flowmodels.sct import _AdaptiveWeights
from flowmodels.utils import EMASupports


class _PositiveLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        if bias is not None:
            bias = F.relu(bias)
        return F.linear(input, F.relu(self.weight), bias)


@dataclass
class _Coeff:
    g: torch.Tensor
    a: torch.Tensor
    s: torch.Tensor
    dg: torch.Tensor | None = None
    da: torch.Tensor | None = None
    ds: torch.Tensor | None = None
    ddg: torch.Tensor | None = None
    dda: torch.Tensor | None = None
    dds: torch.Tensor | None = None


class _LearnableInterpolant(nn.Module):
    def __init__(self, h: int = 1024, dt: float = 1e-2):
        super().__init__()
        self.dt = dt
        self.l1 = _PositiveLinear(1, 1)
        self.l2 = nn.Sequential(
            _PositiveLinear(1, h),
            nn.Sigmoid(),
            _PositiveLinear(h, 1),
        )
        with torch.no_grad():
            self.l1.weight.copy_(torch.tensor(1.0))
            self.l2[2].weight.mul_(1e-5)  # pyright: ignore
        self.i = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_placeholder", torch.tensor([0, 1]), persistent=False)

    def forward(self, t: torch.Tensor):
        assert isinstance(self._placeholder, torch.Tensor)
        (b,) = t.shape
        i = self.i.sigmoid()
        t = torch.cat(
            [self._placeholder, t.to(self._placeholder.device)], dim=0
        )  # pyright: ignore
        t = i * t + (1 - i) * (np.pi * 0.5 * t).sin().square()
        l1 = self.l1.forward(t[:, None])
        g0, g1, gamma = (l1 + self.l2.forward(l1)).squeeze(dim=-1).split([1, 1, b])
        gamma = (gamma - g0) / (g1 - g0).clamp_min(1e-8)
        return gamma

    def coeff(
        self,
        t: torch.Tensor,
        with_1st: bool = False,
        with_2nd: bool = False,
    ) -> _Coeff:
        c, g = 0.5 + self.c.sigmoid() * 0.5, self.forward(t)
        a, s = (1 - g) ** c, g**c
        if not with_1st and not with_2nd:
            return _Coeff(g, a, s)

        g_dt = self.forward(t + self.dt)
        dg = (g_dt - g) / self.dt
        da = -c * (1 - g) ** (c - 1) * dg
        ds = c * g ** (c - 1) * dg
        if not with_2nd:
            return _Coeff(g, a, s, dg, da, ds)

        g_2dt = self.forward(t + 2 * self.dt)
        ddg = (g_2dt - 2 * g_dt + g) * self.dt**-2
        dda = (
            c * (c - 1) * (1 - g) ** (c - 2) * dg.square()
            - c * (1 - g) ** (c - 1) * ddg
        )
        dds = c * (c - 1) * g ** (c - 2) * dg.square() + c * g ** (c - 1) * ddg
        return _Coeff(g, a, s, dg, da, ds, ddg, dda, dds)

    def _broadcast(
        self,
        c: _Coeff | torch.Tensor,
        ref: torch.Tensor,
        with_1st: bool = False,
        with_2nd: bool = False,
    ) -> _Coeff:
        b, *_ = ref.shape
        rdim = [b] + [1] * (ref.dim() - 1)
        if isinstance(c, torch.Tensor):
            c = self.coeff(c, with_1st, with_2nd)
        return _Coeff(
            **{
                k: (v.view(rdim) if v is not None else None)  # pyright: ignore
                for k, v in vars(c).items()
            }
        )

    def interp(self, x: torch.Tensor, e: torch.Tensor, c: _Coeff | torch.Tensor):
        c = self._broadcast(c, ref=x)
        return c.a * x + c.s * e

    def velocity(self, x: torch.Tensor, e: torch.Tensor, c: _Coeff | torch.Tensor):
        c = self._broadcast(c, ref=x, with_1st=True)
        assert c.da is not None and c.ds is not None
        return c.da * x + c.ds * e

    def predict(self, x_t: torch.Tensor, v_t: torch.Tensor, c: _Coeff | torch.Tensor):
        c = self._broadcast(c, ref=x_t, with_2nd=True)
        assert c.da is not None and c.ds is not None
        return (c.ds * x_t - c.s * v_t) / (c.ds * c.a - c.da * c.s)

    def grad(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        F: torch.Tensor,
        dF: torch.Tensor,
        c: _Coeff | torch.Tensor,
    ) -> torch.Tensor:
        c = self._broadcast(c, ref=x_t, with_2nd=True)
        assert c.da is not None and c.ds is not None
        assert c.dda is not None and c.dds is not None
        nu = c.ds * c.a - c.s * c.da
        dnu = c.dds * c.a - c.s * c.dda
        return (
            (c.dds * x_t + c.ds * v_t - c.ds * F - c.s * dF) * nu
            - dnu * (c.ds * x_t - c.s * F)
        ) / nu.square().clamp_min(1e-7)


class FreeformCT(
    nn.Module,
    ForwardProcessSupports,
    PredictionSupports,
    VelocitySupports,
):
    """sCT: Simplifying, Stabilizing & Scailing Continuous-Time Consistency Models, Lu & Song, 2024.[arXiv:2410.11081]"""

    def __init__(
        self,
        module: nn.Module,
        _ada_weight_size: int = 128,
        _hidden_interpolant: int = 1024,
        _eps: float = 1e-2,
    ):
        super().__init__()
        self.F0 = module
        self.sampler = MultistepConsistencySampler()
        self._ada_weight = _AdaptiveWeights(_ada_weight_size)
        self._interpolant = _LearnableInterpolant(_hidden_interpolant, dt=_eps)
        self._eps = _eps

        self._debug_from_loss = {}

    # debug purpose
    @property
    def _debug_purpose(self):
        return {
            "ffsct/i": self._interpolant.i.sigmoid().item(),
            "ffsct/c": self._interpolant.c.item(),
            **self._debug_from_loss,
            **getattr(self.F0, "_debug_purpose", {}),
        }

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated sample from the given `x_t`.
        """
        v_t = self.F0.forward(x_t, t)
        return self._interpolant.predict(x_t, v_t, t)

    def predict(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the sample points `x_0` from the `x_t` w.r.t. the timestep `t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            the predicted sample points `x_0`.
        """
        return self.forward(x_t, t)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        return self.F0.forward(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        ema: Self | EMASupports[Self] | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from the proposal distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the prior distribution, `X_0`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        # shortcut
        batch_size, *_ = sample.shape
        device = sample.device
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            t = torch.rand(batch_size, device=device)
        # for numerical stability
        t = t.clamp(self._eps, 1 - self._eps)
        c = self._interpolant.coeff(t, with_2nd=True)
        # [B, ...]
        x_t = self._interpolant.interp(sample, prior, c)
        # [B, ...]
        v_t = self._interpolant.velocity(sample, prior, c)
        with torch.no_grad():
            F, jvp, *_ = torch.func.jvp(  # pyright: ignore [reportPrivateImportUsage]
                EMASupports[Self].reduce(self, ema).F0.forward,
                (x_t, t),
                (v_t, torch.ones_like(t)),
            )
            F: torch.Tensor = F.detach()
            jvp: torch.Tensor = jvp.detach()
            # df/dt
            grad = self._interpolant.grad(x_t, v_t, F, jvp, c)
            f = self._interpolant.predict(x_t, F, c)
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # normalized tangent
        normalized_tangent = grad / (
            g_norm := (grad.norm(p=2, dim=rdim, keepdim=True) + 0.1)
        )
        # [B, ...]
        estim: torch.Tensor = self._interpolant.predict(x_t, self.F0.forward(x_t, t), c)
        # [B]
        mse = (estim - f + normalized_tangent).square().mean(dim=rdim)
        # [B], adaptive weighting
        logvar = self._ada_weight.forward(t)
        # [B], different with
        loss = mse * logvar.exp() - logvar
        with torch.no_grad():
            self._debug_from_loss = {
                "sct/mse": mse.mean().item(),
                "sct/logvar": logvar.mean().item(),
                "sct/tangent-norm": g_norm.mean().item(),
            }
        # []
        return loss.mean()

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        # T
        steps = steps or 4
        # single-step sampler
        batch_size, *_ = prior.shape
        if steps <= 1:
            x_0 = self.predict(prior, torch.ones(batch_size) - self._eps)
            return x_0, [x_0]
        # multi-step sampler
        x_t, x_ts = prior, []
        if verbose is None:
            verbose = lambda x: x
        for t in verbose(range(steps, 0, -1)):
            # [B]
            t = torch.full((batch_size,), t / steps, dtype=torch.float32)
            # [B, ...]
            x_0 = self.predict(x_t, t.clamp(self._eps, 1 - self._eps))
            # [B, ...]
            x_t = self.noise(
                x_0, (t - 1 / steps).clamp(self._eps, 1 - self._eps), prior
            )
            x_ts.append(x_t)
        return x_t, x_ts

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the `prior`.
        Args:
            x_0: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [torch.long; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        if prior is None:
            prior = torch.randn_like(x_0)
        return self._interpolant.interp(x_0, prior, t)
