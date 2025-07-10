from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn

from flowmodels.basis import (
    ContinuousScheduler,
    ForwardProcessSupports,
    Sampler,
    SamplingSupports,
    ScoreModel,
    ScoreSupports,
)


@dataclass
class VESDEScheduler(ContinuousScheduler):
    """Variance-exploding scheduler for SDE-based score model,
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    vp: bool = False  # variance-exploding scheduler

    sigma_min: float = 0.01
    sigma_max: float = 1.0

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the variance of the prior distribution at each timesteps, `t`.
        Args:
            t: [FloatLike; [B]], the target timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B]], the variances at each timesteps.
        """
        return (self.sigma_min * (self.sigma_max / self.sigma_min) ** t).square()


class VESDE(nn.Module, ScoreModel, ForwardProcessSupports, SamplingSupports):
    """Score modeling with variance-expldoing SDE.
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    def __init__(self, module: nn.Module, scheduler: ContinuousScheduler):
        super().__init__()
        self.score_estim = module
        self.scheduler = scheduler
        assert (
            not self.scheduler.vp
        ), "unsupported scheduler; variance-preserving scheduler"
        self.sampler = VESDEAncestralSampler(scheduler)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the gradient of the log-likelihood(Stein Score) from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]], the given noised sample, `x_t`.
            t: [torch.long; [B]], the current timestep of the noised sample in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated score from the given sample `x_t`.
        """
        return self.score_estim(x_t, t)

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the stein score from the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated stein scores.
        """
        return self.forward(x_t, t)

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
        estim = self.forward(noised, t)
        # [B], zero-based
        sigma = self.scheduler.var(t).sqrt().to(estim)
        # [B, ...]
        sigma = sigma.view([bsize] + [1] * (sample.dim() - 1))
        # [B, ...], apply `\lambda(\sigma) = \sigma^2`
        score_div_std = (noised - sample) / sigma
        return 0.5 * (sigma * estim + score_div_std).square().mean()

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the VESDEAncestralSampler."""
        return self.sampler.sample(self, prior, steps, verbose=verbose)

    def noise(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given samples `x_0` to the `x_t` w.r.t. the timesteps `t` and the noise `eps`.
        Args:
            x_0: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [torch.long; [B]], the target timesteps in range[0, 1].
            eps: [FloatLike; [B, ...]], the samples from the prior distribution.
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
        # [B], zero-based
        sigma = self.scheduler.var(t).sqrt().to(sample)
        # [B, ...]
        sigma = sigma.view([bsize] + [1] * (sample.dim() - 1))
        # [B, ...], variance exploding
        return sample + sigma * prior.to(sample)


class VESDEAncestralSampler(Sampler):
    """Ancestral sampler for score model formed variance-exploding SDE,
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    DEFAULT_STEPS: int = 1000

    def __init__(self, scheduler: ContinuousScheduler):
        super().__init__()
        self.scheduler = scheduler
        assert (
            not self.scheduler.vp
        ), "unsupported scheduler; variance-preserving scheduler"

    def sample(
        self,
        model: ScoreSupports,
        prior: torch.Tensor,
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
                # [], []
                curr, prev = self.scheduler.var(
                    torch.tensor([t / steps, (t - 1) / steps], dtype=torch.float32)
                )
                # [B, ...]
                score = model.score(x_t, torch.full((bsize,), t / steps))
                # [B, ...]
                x_t = (
                    x_t
                    + (curr - prev).to(x_t) * score
                    + (prev * (curr - prev) / curr.clamp_min(1e-7)).sqrt().to(x_t)
                    * eps[t - 1]
                )
                x_ts.append(x_t)
        return x_t, x_ts
