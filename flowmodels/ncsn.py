from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Config:
    """Variance scheduler,
    Generative Modeling By Estimating Gradients of the Data Distribution, Song et al., 2019.[arXiv:1907.05600]
    """

    L: int = 10
    T: int = 100
    sigma_1: float = 1.0
    sigma_L: float = 0.01
    eps: float = 2e-5

    def sigmas(self) -> list[float]:
        factor = np.exp(np.log(self.sigma_L / self.sigma_1) / (self.L - 1))
        return [self.sigma_1 * (factor**i) for i in range(self.L)]


class NCSN(nn.Module):
    """Generative Modeling By Estimating Gradients of the Data Distribution, Song et al., 2019.[arXiv:1907.05600]"""

    def __init__(self, module: nn.Module, config: Config):
        super().__init__()
        self.score_estim = module
        self.config = config

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the gradient of the log-likelihood(Stein Score) from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [torch.long; [B]], the current timestep of the noised sample in range[1, L].
        Returns:
            estimated score from the given sample `x_t`.
        """
        return self.score_estim(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `x_0`.
            timesteps: [torch.long; [B]], target timesteps in range[1, L],
                sample from uniform distribution if not provided.
            eps: [FloatLike; [B, ...]], sample from prior distribution,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        # sample
        if timesteps is None:
            timesteps = torch.randint(1, self.config.L + 1, (batch_size,))
        if eps is None:
            eps = torch.randn_like(sample)
        # compute objective
        noised = self.noise(sample, timesteps, eps=eps)
        estim = self.forward(noised, timesteps)
        # [T], zero-based
        sigma = estim.new_tensor(self.config.sigmas())
        # [B, ...]
        sigma = sigma[timesteps - 1].view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...], apply `\lambda(\sigma) = \sigma^2`
        score_div_std = (noised - sample) / sigma
        return 0.5 * (sigma * estim + score_div_std).square().mean()

    def sample(
        self,
        prior: torch.Tensor,
        verbose: Callable[[range], Iterable] | None = None,
        eps: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Transfer the samples from the prior distribution to the trained distribution.
        Args:
            prior: [FloatLike; [B, ...]], samples from the prior distribution.
        """
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        if eps is None:
            eps = [
                torch.randn_like(prior) for _ in range(self.config.T * self.config.L)
            ]
        # [L]
        var = self.config.sigmas()
        # loop
        x_t, x_ts = prior, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(self.config.L)):
                alpha = self.config.eps * (var[i] / var[-1]) ** 2
                for _ in verbose(range(self.config.T)):
                    score = self.forward(
                        x_t,
                        torch.full((bsize,), i, dtype=torch.long),
                    )
                    x_t = x_t + 0.5 * alpha * score + np.sqrt(alpha) * eps[len(x_ts)]
                    x_ts.append(x_t)
        return x_t, x_ts

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the noise `eps`.
        Args:
            x_0: [FloatLike; [B, ...]], the given sample, `x_0`.
            t: [torch.long; [B]], the target timestep in range[1, L].
            eps: [FloatLike; [B, ...]], the sample from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        if (t <= 0).all():
            return x_0
        # assign default value
        if eps is None:
            eps = torch.randn_like(x_0)
        # [L], zero-based
        sigma = x_0.new_tensor(self.config.sigmas())
        # L
        (timesteps,) = sigma.shape
        # [L, ...]
        sigma = sigma.view([timesteps] + [1] * (x_0.dim() - 1))
        # [B, ...]
        return x_0 + sigma[t - 1] * eps.to(x_0)
