from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import Sampler, Scheduler, ScoreModel


@dataclass
class DDPMScheduler(Scheduler):
    """Variance scheduler,
    Denoising Diffusion Probabilistic Models, Ho et al., 2020.[arXiv:2006.11239]
    """

    T: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 0.02

    def betas(self) -> list[float]:
        # increasing linearly from `beta_1` to `beta_t`
        return [
            self.beta_1 + i / (self.T - 1) * (self.beta_T - self.beta_1)
            for i in range(self.T - 1)
        ] + [self.beta_T]

    def var(self) -> torch.Tensor:
        # [T]
        beta = torch.tensor(self.betas(), dtype=torch.float32)
        # [T]
        alpha_bar = (1 - beta).cumprod(dim=0)
        return 1 - alpha_bar


class DDPM(nn.Module, ScoreModel):
    """Denoising Diffusion Probabilistic Models, Ho et al., 2020.[arXiv:2006.11239]"""

    def __init__(self, module: nn.Module, scheduler: Scheduler):
        super().__init__()
        self.noise_estim = module
        self.scheduler = scheduler
        self.sampler = None
        try:
            self.sampler = DDPMSampler(scheduler)
        except AssertionError:
            pass

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the noise from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]], the given noised samples, `x_t`.
            t: [torch.long; [B]], the current timesteps of the noised sample in range[0, T].
        Returns:
            estimated noise from the given sample `x_t`.
        """
        return self.noise_estim(x_t, t)

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
        estim = self.forward(x_t, t)
        # [T + 1], one-based
        var = F.pad(
            self.scheduler.var().to(estim.device, torch.float32),
            [1, 0],
            mode="constant",
            value=0.0,
        )
        # [T + 1, ...]
        var = var.view([self.scheduler.T + 1] + [1] * (estim.dim() - 1))
        return estim * var[t].clamp_min(1e-7).rsqrt().to(estim.dtype)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the training data, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from the uniform distribution if not provided.
            eps: [FloatLike; [B, ...]], the samples from the prior distribution,
                sample from the gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        # sample
        if t is None:
            t = torch.rand(batch_size)
        if eps is None:
            eps = torch.randn_like(sample)
        # discretize in range [1, T]
        t = ((t * self.scheduler.T).long() + 1).clamp_max(self.scheduler.T)
        # compute objective
        noised = self.noise(sample, t, eps=eps)
        estim = self.forward(noised, t)
        return (eps - estim).square().mean()

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        """Forward to the DDPMSampler."""
        if self.sampler is None:
            return None
        return self.sampler.sample(self, prior, steps, verbose)

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the noise `eps`.
        Args:
            x_0: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [torch.long; [B]], the target timesteps in range[1, T].
            eps: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        if (t <= 0).all():
            return x_0
        # assign default value
        if eps is None:
            eps = torch.randn_like(x_0)
        # settings
        dtype, device = x_0.dtype, x_0.device
        # [T], zero-based
        alpha_bar = 1 - self.scheduler.var().to(device, torch.float32)
        # [T, ...]
        alpha_bar = alpha_bar.view([self.scheduler.T] + [1] * (x_0.dim() - 1))
        # [B, ...], variance-preserving scheduler
        if self.scheduler.vp:
            return alpha_bar[t - 1].sqrt().to(dtype) * x_0 + (
                1 - alpha_bar[t - 1]
            ).sqrt().to(dtype) * eps.to(x_0)
        # variance-exploding scheduler
        return x_0 + (1 - alpha_bar[t - 1]).sqrt().to(dtype) * eps.to(x_0)


class DDPMSampler(Sampler):
    """DDPM Sampler,
    Denoising Diffusion Probabilistic Models, Ho et al., 2020.[arXiv:2006.11239]
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        assert (
            self.scheduler.vp
        ), "varaince-exploding diffusion denoiser is not implemented yet."

    def sample(
        self,
        model: ScoreModel,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        eps: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample from the prior distribution to the trained distribution.
        Args:
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        assert steps is None or steps == self.scheduler.T, "unsupported steps"
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        if eps is None:
            eps = [None] * self.scheduler.T
        # [T, ...]
        std = (
            self.scheduler.var()
            .sqrt()
            .view([self.scheduler.T] + [1] * (prior.dim() - 1))
        )
        # loop
        x_t, x_ts = prior, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(self.scheduler.T, 0, -1)):
                t = torch.full((bsize,), i, dtype=torch.long)
                score = model.score(x_t, t / self.scheduler.T)
                e_t = score * std[i - 1, None].to(score.device, score.dtype)
                x_t = self.denoise(x_t, e_t, t, eps=eps[i - 1])
                x_ts.append(x_t)
        return x_t, x_ts

    def denoise(
        self,
        x_t: torch.Tensor,
        e_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoise the given sample `x_t` to the single-step backward `x_{t-1}`
            w.r.t. the current timestep `t` and the additional noise `eps`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            e_t: [FloatLike, [B, ...]], the estimated noise of the `x_t`.
            t: [torch.long; [B]], the current timesteps in range[1, T].
            eps: [FloatLike; [...]], the additional noise from the prior.
        Returns:
            denoised sample, `x_{t_1}`.
        """
        if (t <= 0).all():
            return x_t
        # assign default value
        if eps is None:
            eps = torch.randn_like(x_t)
        # settings
        dtype, device = x_t.dtype, x_t.device
        # [T], zero-based
        alpha_bar = 1 - self.scheduler.var().to(device, torch.float32)
        # [T, ...]
        alpha_bar = alpha_bar.view([self.scheduler.T] + [1] * (x_t.dim() - 1))
        # [T, ...], zero-based
        alpha = alpha_bar / F.pad(
            alpha_bar[:-1], [0, 0] * (x_t.dim() - 1) + [1, 0], "constant", 1.0
        )
        # [T, ...], zero-based
        beta = 1 - alpha
        # [B, ...], denoised
        mean = alpha[t - 1].clamp_min(1e-7).rsqrt().to(dtype) * (
            x_t
            - beta[t - 1].to(dtype)
            * (1 - alpha_bar[t - 1].to(dtype)).clamp_min(1e-7).rsqrt()
            * e_t
        )
        # B
        (batch_size,) = t.shape
        # [B, ...]
        mask = t.view([batch_size] + [1] * (x_t.dim() - 1)) > 1
        return mean + mask * beta[t - 1].sqrt().to(dtype) * eps.to(mean)
