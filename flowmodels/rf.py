from typing import Callable, Iterable

import torch
import torch.nn as nn


class RectifiedFlow(nn.Module):
    """Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022.[arXiv:2209.03003]"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.velocity_estim = module

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the causalized velocity from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [torch.float32; [B]], the current timestep in range[0, 1].
        Returns:
            estimated velocity from the given sample `x_t`.
        """
        return self.velocity_estim(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        src: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            timesteps: [torch.float32; [B]], target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
            src: [FloatLike; [B, ...]], sample from the source distribution, `X_0`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        # sample
        if timesteps is None:
            timesteps = torch.rand(batch_size)
        if src is None:
            src = torch.randn_like(sample)
        # compute objective
        # [B, ...]
        t = timesteps.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = t * sample + (1 - t) * src
        # [B, ...]
        estim = self.forward(x_t, timesteps)
        return ((sample - src) - estim).square().mean()

    def sample(
        self,
        src: torch.Tensor,
        steps: int = 3,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Transfer the samples from the prior distribution to the trained distribution, using vanilla Euler method.
        Args:
            src: [FloatLike; [B, ...]], samples from the source distribution, `X_0`.
            steps: the number of the steps.
        """
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        # loop
        x_t, x_ts = src, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(steps)):
                velocity = self.forward(
                    x_t,
                    torch.full((bsize,), i / steps, dtype=torch.float32),
                )
                x_t = x_t + velocity / steps
                x_ts.append(x_t)
        return x_t, x_ts
