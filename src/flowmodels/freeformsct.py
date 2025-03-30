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
            bias = F.softplus(bias)
        return F.linear(input, F.softplus(self.weight), bias)


class _LearnableInterpolant(nn.Module):
    def __init__(self, h: int = 1024):
        super().__init__()
        self.l1 = _PositiveLinear(1, 1)
        self.l2 = nn.Sequential(
            _PositiveLinear(1, h),
            nn.Sigmoid(),
            _PositiveLinear(h, 1),
        )
        self.gamma_min = nn.Parameter(torch.tensor(-10.))
        self.gamma_gap = nn.Parameter(torch.tensor(20.))
        self.c = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("_placeholder", torch.tensor([0, 1]), persistent=False)

        self._grad_fn = torch.func.grad(lambda x: self.forward(x).sum())
        self._2nd_fn = torch.func.grad(lambda x: self._grad_fn(x).sum())

    def forward(self, t: torch.Tensor):
        b, = t.shape
        t = torch.cat([self._placeholder, t], dim=0)
        l1 = self.l1.forward(t[:, None])
        g0, g1, gamma = (l1 + self.l2.forward(l1)).squeeze(dim=-1).split([1, 1, b])
        return (gamma - g0) / (g1 - g0) * self.gamma_gap + self.gamma_min

    def coeff(
        self,
        t: torch.Tensor,
        with_1st: bool = False,
        with_2nd: bool = False,
    ):
        gamma = self.forward(t)
        # sigmoid(-gamma) ** c
        alpha = (-self.c * (1 + gamma.exp()).log()).exp()
        # sigmoid(gamma) ** c
        sigma = (-self.c * (1 + (-gamma).exp()).log()).exp()
        if not with_1st and not with_2nd:
            return gamma, alpha, sigma

        dgamma = self._grad_fn(t)
        m_gamma_sigmoid = (-gamma).sigmoid()
        gamma_sigmoid = gamma.sigmoid()
        dalpha = -self.c * alpha * (1 - m_gamma_sigmoid) * dgamma
        dsigma = self.c * sigma * (1 - gamma_sigmoid) * dgamma
        if not with_2nd:
            return gamma, alpha, sigma, dgamma, dalpha, dsigma
        
        ddgamma = self._2nd_fn(t)
        ddalpha = -self.c * (
            dalpha * (1 - m_gamma_sigmoid) * dgamma
            + alpha * m_gamma_sigmoid * (1 - m_gamma_sigmoid) * dgamma.square()
            + alpha * (1 - m_gamma_sigmoid) * ddgamma
        )
        ddsigma = self.c * (
            dsigma * (1 - gamma_sigmoid) * dgamma
            - sigma * gamma_sigmoid * (1 - gamma_sigmoid) * dgamma.square()
            * sigma * (1 - gamma) * ddgamma
        )
        return (
            gamma, alpha, sigma, dgamma, dalpha, dsigma, ddgamma, ddalpha, ddsigma,
        )

    def interp(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor):
        b, = t.shape
        rdim = [b] + [1] * (x.dim() - 1)
        _, alpha, sigma = self.coeff(self.forward(t))
        return alpha.view(rdim) * x + sigma.view(rdim) * e

    def velocity(self, x: torch.Tensor, e: torch.Tensor, t: torch.Tensor):
        b, = t.shape
        rdim = [b] + [1] * (x.dim() - 1)
        _, _, _, _, dalpha, dsigma = self.coeff(t, with_1st=True)
        return dalpha.view(rdim) * x + dsigma.view(rdim) * e

    def predict(self, x_t: torch.Tensor, v_t: torch.Tensor, t: torch.Tensor):
        b, = t.shape
        rdim = [b] + [1] * (x_t.dim() - 1)
        _, alpha, sigma, _, dalpha, dsigma = [
            tensor.view(rdim) for tensor in self.coeff(t, with_1st=True)
        ]
        return (sigma * v_t - dsigma * x_t) / (sigma * dalpha - alpha * dsigma).clamp_min(1e-8)

    def grad(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        a_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        b, = t.shape
        rdim = [b] + [1] * (x_t.dim() - 1)
        _, alpha, sigma, _, dalpha, dsigma, _, ddalpha, ddsigma = [
            tensor.view(rdim) for tensor in self.coeff(t, with_2nd=True)
        ]
        return (
            (sigma * a_t - ddsigma * x_t) * (sigma * dalpha - alpha * dsigma)
            - (sigma * v_t - dsigma * x_t) * (sigma * ddalpha - ddsigma * alpha)
        ) / (sigma * dalpha - alpha * dsigma).square().clamp_min(1e-7)


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
    ):
        super().__init__()
        self.F0 = module
        self.sampler = MultistepConsistencySampler()
        self._ada_weight = _AdaptiveWeights(_ada_weight_size)
        self._interpolant = _LearnableInterpolant(_hidden_interpolant)

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
        # [B, ...], `self.noise` automatically scale the prior with `sigma_d`
        x_t = self._interpolant.interp(sample, prior, t)
        with torch.no_grad():
            td = (t - 1e-5).clamp_min(0)
            teacher = self.forward(
                self._interpolant.interp(sample, prior, td),
                td
            )
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        mse = (teacher - self.forward(x_t, t)).square().mean(rdim) / 1e-5
        # # [B], adaptive weighting
        # logvar = self._ada_weight.forward(t)
        # # [B], different with
        # loss = mse * logvar.exp() - logvar
        # []
        return loss.mean()

        # [B, ...]
        v_t = self._interpolant.velocity(sample, prior, t)
        with torch.no_grad():
            F, jvp, *_ = torch.func.jvp(  # pyright: ignore [reportPrivateImportUsage]
                EMASupports[Self].reduce(self, ema).F0.forward,
                (x_t, t),
                (v_t, torch.ones_like(t)),
            )
            F: torch.Tensor = F.detach()
            jvp: torch.Tensor = jvp.detach()
            # df/dt
            grad = self._interpolant.grad(x_t, F, jvp, t)
            f = self._interpolant.predict(x_t, F, t)
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # normalized tangent
        normalized_tangent = grad / (grad.norm(p=2, dim=rdim, keepdim=True) + 0.1)
        # [B, ...]
        estim: torch.Tensor = self.forward(x_t, t)
        # [B]
        mse = (estim - f + normalized_tangent).square().mean(dim=rdim)
        # [B], adaptive weighting
        logvar = self._ada_weight.forward(t)
        # [B], different with
        loss = mse * logvar.exp() - logvar
        # []
        return loss.mean()

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        return self.sampler.sample(self, prior, steps, verbose)

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
