from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

import torch
import torch.nn as nn

from flowmodels.basis import (
    ContinuousScheduler,
    ContinuousSchedulerProtocol,
    ForwardProcessSupports,
    Sampler,
    SamplingSupports,
    ScoreModel,
    ScoreSupports,
)


@dataclass
class VPSDEScheduler(ContinuousScheduler):
    """Variance-preserving scheduler for SDE-based score model,
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    beta_min: float = 0.1
    beta_max: float = 20

    def betas(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the betas at each timesteps `t`.
        Args:
            t: [FloatLike; [B]], the target timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B]], the betas at each timesteps.
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the variance of the prior distribution at each timesteps, `t`.
        Args:
            t: [FloatLike; [B]], the target timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B]], the variances at each timesteps.
        """
        # beta = beta_min + t(beta_max - beta_min)
        b_max, b_min = self.beta_max, self.beta_min
        return 1 - (-0.5 * t.square() * (b_max - b_min) - t * b_min).exp()


class VPSDE(nn.Module, ScoreModel, ForwardProcessSupports, SamplingSupports):
    """Score modeling with variance-preserving SDE.
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    def __init__(self, module: nn.Module, scheduler: ContinuousScheduler):
        super().__init__()
        self.score_estim = module
        self.scheduler = scheduler
        assert self.scheduler.vp, "unsupported scheduler; variance-exploding scheduler"

        self.sampler = None
        if isinstance(scheduler, VPSDEAncestralSamplerSupports):
            self.sampler = VPSDEAncestralSampler(scheduler)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the gradient of the log-likelihood(Stein Score) from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]], the given noised sample, `x_t`.
            t: [torch.long; [B]], the current timestep of the noised sample in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated score from the given sample `x_t`.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return self.score_estim(x_t, t, **kwargs)

    def score(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the stein score from the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated stein scores.
        """
        return self.forward(x_t, t, label=label)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the training data, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from the uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], the sample from the prior distribution,
                sample from the gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        bsize, *_ = sample.shape
        # sample
        if t is None:
            t = torch.rand(bsize)
        if prior is None:
            prior = torch.randn_like(sample)
        # compute objective
        noised = self.noise(sample, t, prior=prior)
        estim = self.forward(noised, t, label=label)
        # [B], zero-based
        sigma = self.scheduler.var(t).clamp(1e-10, 1.0).sqrt().to(estim)
        # [B, ...]
        sigma = sigma.view([bsize] + [1] * (sample.dim() - 1))
        # [B, ...], apply `\lambda(\sigma) = \sigma^2`
        score_div_std = (noised - sample) / sigma
        return 0.5 * (estim + score_div_std).square().mean()

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the VPSDEAncestralSampler."""
        assert self.sampler is not None
        return self.sampler.sample(self, prior, label, steps, verbose=verbose)

    def noise(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given samples `x_0` to the `x_t` w.r.t. the timesteps `t` and the noise `eps`.
        Args:
            sample: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [torch.long; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        if (t <= 0).all():
            return sample
        # assign default value
        if prior is None:
            prior = torch.randn_like(sample)
        # B
        bsize, *_ = sample.shape
        # [B]
        var = self.scheduler.var(t)
        # [B, ...]
        var = var.view([bsize] + [1] * (sample.dim() - 1))
        # [B, ...], variance exploding
        return (1 - var).sqrt().to(sample) * sample + var.sqrt().to(sample) * prior.to(
            sample
        )


@runtime_checkable
class VPSDEAncestralSamplerSupports(ContinuousSchedulerProtocol, Protocol):

    def betas(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the betas at each timesteps `t`.
        Args:
            t: [FloatLike; [B]], the target timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B]], the betas at each timesteps.
        """
        ...


class VPSDEAncestralSampler(Sampler):
    """Ancestral sampler for score model formed variance-preserving SDE,
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    DEFAULT_STEPS: int = 1000

    def __init__(self, scheduler: VPSDEAncestralSamplerSupports):
        super().__init__()
        self.scheduler = scheduler
        assert self.scheduler.vp, "unsupported scheduler; variance-exploding scheduler"

    def sample(
        self,
        model: ScoreSupports,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        eps: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample from the prior distribution to the trained distribution.
        Args:
            model: the score estimation model.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        # assign default values
        steps = steps or self.DEFAULT_STEPS
        if verbose is None:
            verbose = lambda x: x
        if eps is None:
            eps = [torch.randn_like(prior) for _ in range(steps)]
        # B
        bsize, *_ = prior.shape
        x_t, x_ts = prior, []
        with torch.inference_mode():
            for t in verbose(range(steps, 0, -1)):
                # []
                b = (
                    self.scheduler.betas(torch.tensor(t / steps, dtype=torch.float32))
                    / steps
                )
                # [B, ...]
                score = model.score(x_t, torch.full((bsize,), t / steps), label=label)
                # [B, ...]
                x_t = (
                    (2 - (1 - b).sqrt()).to(x_t) * x_t
                    + b.to(x_t) * score
                    + b.sqrt().to(x_t) * eps[t - 1]
                )
                x_ts.append(x_t)
        return x_t, x_ts
