from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import (
    ContinuousScheduler,
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
    ScoreModel,
)
from flowmodels.cm import MultistepConsistencySampler


@dataclass
class ECTScheduler(ContinuousScheduler):
    vp: bool = False

    sigma_d: float = 0.5  # standard deviation of the prior distribution
    p_mean: float = -1.1  # default values from Appendix G.2. CIFAR-10
    p_std: float = 2.0

    # hyperparameters for shrinking
    k: float = 8.0
    b: float = 1.0
    q: float = 2.0
    total_training_steps: int | None = None
    d_factor: int = 8

    def cskip(self, t: torch.Tensor) -> torch.Tensor:
        """Differentiable functions s.t. `cskip(0) = 1`.
        Args:
            t: [FloatLike; [B]], the given timesteps, in range[0, 1].
        Returns:
            coefficients for `x`, s.t. f(x, t) = cskip(t) x + cout(t) * F(x, t).
        """
        return self.sigma_d**2 / (t.square() + self.sigma_d**2)

    def cout(self, t: torch.Tensor) -> torch.Tensor:
        """Differentiable functions s.t. `cout(0) = 0`.
        Args:
            t: [FloatLike; [B]], the given timesteps, in range[0, 1].
        Returns:
            coefficients for `F`, s.t. f(x, t) = cskip(t) x + cout(t) * F(x, t).
        """
        return t * self.sigma_d**2 / (t.square() + self.sigma_d**2)

    def r(self, t: torch.Tensor, step: int) -> torch.Tensor:
        """A step size scheduler.
        Args:
            t: [FloatLike; [B]], the given timesteps, in range[0, 1].
            step: the current training steps.
        Returns:
            a step size vector `r`.
        """
        assert self.total_training_steps is not None
        n_t = 1 + self.k * (-self.b * t).sigmoid()
        d = self.total_training_steps // self.d_factor
        a = step // d
        return (1 - self.q ** (-a) * n_t) * t

    def var(self, t: torch.Tensor) -> torch.Tensor:
        return t.square()


class EasyConsistencyTraining(
    nn.Module,
    ScoreModel,
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
):
    """ECT: Consistency Models Made Easy, Geng et al., ICLR 2025.[arXiv:2406.14548]"""

    def __init__(self, module: nn.Module, scheduler: ECTScheduler):
        super().__init__()
        self.F0 = module
        self.scheduler = scheduler
        self.sampler = MultistepConsistencySampler()
        self._step = 0

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
        rdim = [bsize] + [1] * (x_t.dim() - 1)
        # [B, ...]
        cskip, cout = self.scheduler.cskip(t).view(rdim), self.scheduler.cout(t).view(
            rdim
        )
        # [B, ...]
        return cskip * x_t + cout * self.F0.forward(x_t, t)

    def predict(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the sample points `x_0` from the `x_t` w.r.t. the timestep `t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            the predicted sample points `x_0`.
        """
        return self.forward(x_t, t)

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
        # [B, ...], score of variance-exploding scheme
        return (x_0 - x_t) / var.clamp_min(1e-7).to(x_0)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from the proposal distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the prior distribution,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        # shortcut
        batch_size, *_ = sample.shape
        sigma_d = self.scheduler.sigma_d
        device = sample.device
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            # shortcut
            p_std, p_mean = self.scheduler.p_std, self.scheduler.p_mean
            # sample from log-normal
            rw_t = (torch.randn(batch_size, device=device) * p_std + p_mean).exp()
            # [T]
            t = rw_t.atan() / np.pi * 2
        # [B, ...]
        x_t = self.noise(sample, t, prior)
        # [B]
        r = self.scheduler.r(t, self._step)
        # [B, ...]
        x_r = self.noise(sample, r, prior)
        with torch.no_grad():
            x_r0 = self.forward(x_r, r)
        # []
        loss = (self.forward(x_t, t) - x_r0.detach()).square().mean()
        # update internal state
        self._step += 1
        return loss

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
        sample: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the `prior`.
        Args:
            sample: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [torch.long; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        (bsize,) = t.shape
        if prior is None:
            prior = torch.randn_like(sample)
        return sample + t.view([bsize] + [1] * (sample.dim() - 1)).to(prior) * prior
