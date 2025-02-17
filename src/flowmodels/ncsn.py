from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import (
    ForwardProcessSupports,
    Sampler,
    Scheduler,
    SchedulerProtocol,
    ScoreModel,
    ScoreSupports,
)


@dataclass
class NCSNScheduler(Scheduler):
    """Variance scheduler,
    Generative Modeling By Estimating Gradients of the Data Distribution, Song et al., 2019.[arXiv:1907.05600]
    """

    vp: bool = False  # variance-exploding scheduler

    T: int = 10  # denoted by `L` in Song et al., 2019.
    R: int = 100  # denoted by `T` in Song et al., 2019.
    sigma_1: float = 1.0
    sigma_L: float = 0.01
    eps: float = 2e-5

    def var(self) -> torch.Tensor:
        factor = np.exp(np.log(self.sigma_1 / self.sigma_L) / (self.T - 1))
        # [T], from sigma_L to sigma_1(reverse order)
        sigma = self.sigma_L * (factor ** torch.arange(self.T, dtype=torch.float32))
        # [T]
        return sigma.square()


class NCSN(nn.Module, ScoreModel, ForwardProcessSupports):
    """Generative Modeling By Estimating Gradients of the Data Distribution, Song et al., 2019.[arXiv:1907.05600]"""

    def __init__(self, module: nn.Module, scheduler: Scheduler):
        super().__init__()
        self.score_estim = module
        self.scheduler = scheduler
        assert (
            not self.scheduler.vp
        ), "unsupported scheduler; variance-preserving scheduler"

        self.sampler = None
        if isinstance(scheduler, AnnealedLangevinDynamicsSamplerSupports):
            self.sampler = AnnealedLangevinDynamicsSampler(scheduler)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the gradient of the log-likelihood(Stein Score) from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [torch.long; [B]], the current timestep of the noised sample in range[1, T].
        Returns:
            estimated score from the given sample `x_t`.
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
        # discretize in range[0, T]
        t = (t * self.scheduler.T).long()
        # [B, ...]
        return self.forward(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
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
        batch_size, *_ = sample.shape
        # sample
        if t is None:
            t = torch.rand(batch_size)
        if prior is None:
            prior = torch.randn_like(sample)
        # discretize in range[1, T]
        t = ((t * self.scheduler.T).long() + 1).clamp_max(self.scheduler.T)
        # compute objective
        noised = self.noise(sample, t, prior=prior)
        estim = self.forward(noised, t)
        # [T], zero-based
        sigma = self.scheduler.var().sqrt().to(estim)
        # [B, ...]
        sigma = sigma[t - 1].view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...], apply `\lambda(\sigma) = \sigma^2`
        score_div_std = (noised - sample) / sigma
        return 0.5 * (sigma * estim + score_div_std).square().mean()

    def sample(
        self,
        prior: torch.Tensor,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        """Forward to the DDPMSampler."""
        if self.sampler is None:
            return None
        return self.sampler.sample(self, prior, verbose=verbose)

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the noise `eps`.
        Args:
            x_0: [FloatLike; [B, ...]], the given sample, `x_0`.
            t: [torch.long; [B]], the target timestep in range[1, T].
            prior: [FloatLike; [B, ...]], the sample from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        if (t <= 0).all():
            return x_0
        # assign default value
        if prior is None:
            prior = torch.randn_like(x_0)
        # [T], zero-based
        sigma = self.scheduler.var().sqrt().to(x_0)
        # [T, ...]
        sigma = sigma.view([self.scheduler.T] + [1] * (x_0.dim() - 1))
        # [B, ...], variance exploding
        return x_0 + sigma[t - 1] * prior.to(x_0)


@runtime_checkable
class AnnealedLangevinDynamicsSamplerSupports(SchedulerProtocol, Protocol):
    R: int
    eps: float


class AnnealedLangevinDynamicsSampler(Sampler):
    """Annealed Langevin Dynamics Sampler from NCSN,
    Generative Modeling By Estimating Gradients of the Data Distribution, Song et al., 2019.[arXiv:1907.05600]
    """

    def __init__(self, scheduler: AnnealedLangevinDynamicsSamplerSupports):
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
        """Transfer the samples from the prior distribution to the trained distribution.
        Args:
            prior: [FloatLike; [B, ...]], samples from the prior distribution.
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        total = self.scheduler.T * self.scheduler.R
        assert steps is None or steps == total
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        if eps is None:
            eps = [torch.randn_like(prior) for _ in range(total)]
        # [T]
        var = self.scheduler.var().tolist()
        # loop
        x_t, x_ts = prior, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(self.scheduler.T, 0, -1)):
                alpha = self.scheduler.eps * (var[i - 1] / max(var[0], 1e-7))
                for _ in verbose(range(self.scheduler.R)):
                    score = model.score(
                        x_t,
                        torch.full((bsize,), i / self.scheduler.T).to(x_t),
                    )
                    x_t = x_t + 0.5 * alpha * score + np.sqrt(alpha) * eps[len(x_ts)]
                    x_ts.append(x_t)
        return x_t, x_ts
