from dataclasses import dataclass
from typing import Callable, Iterable, Self

import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import (
    ContinuousScheduler,
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
    ScoreModel,
    VelocitySupports,
)
from flowmodels.cm import MultistepConsistencySampler
from flowmodels.utils import EMASupports


@dataclass
class ScaledContinuousCMScheduler(ContinuousScheduler):
    vp: bool = True

    sigma_d: float = 0.5  # standard deviation of the prior distribution
    p_mean: float = -1.0  # default values from Appendix G.2. CIFAR-10
    p_std: float = 1.4

    def var(self, t: torch.Tensor) -> torch.Tensor:
        return (t * np.pi * 0.5).sin().square()


class ScaledContinuousCM(
    nn.Module,
    ScoreModel,
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
    VelocitySupports,
):
    """sCT: Simplifying, Stabilizing & Scailing Continuous-Time Consistency Models, Lu & Song, 2024.[arXiv:2410.11081]"""

    def __init__(self, module: nn.Module, scheduler: ScaledContinuousCMScheduler):
        super().__init__()
        self.F0 = module
        self._ada_weight = nn.Linear(1, 1)
        self.scheduler = scheduler
        self.sampler = MultistepConsistencySampler()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated sample from the given `x_t`.
        """
        # shortcut
        (bsize,) = t.shape
        sigma_d = self.scheduler.sigma_d
        backup = t
        # [B, ...], scale t to range[0, pi/2]
        t = (t * np.pi * 0.5).view([bsize] + [1] * (x_t.dim() - 1))
        return t.cos() * x_t - t.sin() * sigma_d * self.F0(x_t / sigma_d, backup)

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
        sigma_d = self.scheduler.sigma_d
        return sigma_d * self.F0(x_t / sigma_d, t)

    def score(self, x_t: torch.Tensor, t: torch.Tensor):
        """Estimate the stein score from the given sample `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated score.
        """
        (bsize,) = t.shape
        # [B, ...]
        x_0 = self.forward(x_t, t)
        # [B, ...]
        var = self.scheduler.var(t).view([bsize] + [1] * (x_0.dim() - 1))
        # simplified:
        # t = t * np.pi * 0.5 >>= (t.cos() * x_0 - x_t) / t.sin().square()
        return ((1 - var).sqrt().to(x_0) * x_0 - x_t) / var.clamp_min(1e-7).to(x_0)

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
        sigma_d = self.scheduler.sigma_d
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            # shortcut
            p_std, p_mean = self.scheduler.p_std, self.scheduler.p_std
            # sample from log-normal
            rw_t = (torch.randn(batch_size) * p_std + p_mean).exp()
            # [T], in range[0, pi/2]
            t = (rw_t / sigma_d).atan()
        else:
            # scale into range[0, pi/2]
            t = t * np.pi * 0.5
            # compute a reciprocal of the prior weighting term from the given t
            rw_t = t.tan() * sigma_d
        # [B, ...], `self.noise` automatically scale the prior with `sigma_d`
        x_t = self.noise(sample, t / np.pi * 2, prior)
        # [B, ...]
        _t = t.view([batch_size] + [1] * (x_t.dim() - 1))
        # [B, ...]
        v_t = _t.cos() * prior * sigma_d - _t.sin() * sample
        # [B, ...], [B, ...], jvp = t.cos() * t.sin() * dF/dt
        F, jvp, *_ = torch.func.jvp(  # pyright: ignore [reportPrivateImportUsage]
            EMASupports[Self].reduce(self, ema).F0.forward,
            (x_t / sigma_d, t / np.pi * 2),
            (_t.cos() * _t.sin() * v_t, t.cos() * t.sin() * sigma_d),
        )
        F: torch.Tensor = F.detach()
        jvp: torch.Tensor = jvp.detach()
        # df/dt = -t.cos() * (sigma_d * F(x_t/sigma_d, t) - dx_t/dt) - t.sin() * (x_t + sigma_d * dF/dt)
        cos_mul_grad = (
            -_t.cos().square() * (sigma_d * F - v_t)
            - _t.sin() * _t.cos() * x_t
            - sigma_d * jvp
        )
        # [B, ...]
        estim: torch.Tensor = self.F0.forward(x_t / sigma_d, t / np.pi * 2)
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # normalized tangent
        normalized_tangent = cos_mul_grad / (
            cos_mul_grad.norm(p=2, dim=rdim, keepdim=True) + 0.1
        )
        # [B]
        mse = (estim - F - normalized_tangent).square().mean(dim=rdim)
        # [B], adaptive weighting
        logvar = self._ada_weight.forward(_t).squeeze(dim=-1)
        # [B]
        loss = (mse * logvar.exp() - logvar) / rw_t.clamp_min(1e-7)
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
        (bsize,) = t.shape
        if prior is None:
            prior = torch.randn_like(x_0)
        # [B, ...], scale in range [0, pi/2]
        t = (t * np.pi * 0.5).view([bsize] + [1] * (x_0.dim() - 1))
        # [B, ...]
        return t.cos() * x_0 + t.sin() * prior * self.scheduler.sigma_d
