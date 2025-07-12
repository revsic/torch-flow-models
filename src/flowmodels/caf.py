from copy import deepcopy
from typing import Callable, Iterable

import torch
import torch.nn as nn

from flowmodels.basis import VelocitySupports, SamplingSupports


class _AccelerationWrapper(nn.Module):
    def __init__(self, in_channels: int, backbone: nn.Module, dim: int = -1):
        super().__init__()
        self.proj = nn.Linear(in_channels * 2, in_channels)
        self.backbone = backbone
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor | None = None,
    ):
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return self.backbone(
            self.proj.forward(torch.cat([x, v], dim=self.dim)), t, **kwargs
        )


class ConstantAccelerationFlow(nn.Module, VelocitySupports, SamplingSupports):
    """Constant Acceleration Flow, Park et al., 2024.[arXiv:2411.00322]"""

    def __init__(self, channels: int, module: nn.Module, h: float = 1.5):
        super().__init__()
        self.v_0 = module
        self.a_t = _AccelerationWrapper(channels, deepcopy(module))
        self.h = h

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the causalized velocity from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]], the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            estimated velocity from the given sample `x_t`.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        (bsize,) = t.shape
        v0 = self.v_0.forward(x_t, t, **kwargs)
        return v0 + (t.view([bsize] + [1] * (x_t.dim() - 1))) * self.a_t.forward(
            x_t, v0, t, **kwargs
        )

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        return self.forward(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the source distribution, `X_0`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        # sample
        if t is None:
            t = torch.rand(batch_size)
        if prior is None:
            prior = torch.randn_like(sample)
        # compute objective
        backup = t
        # [B, ...]
        t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        v_0 = self.h * (sample - prior)
        # [B, ...]
        x_t = (1 - t.square()) * prior + t.square() * sample + (t - t.square()) * v_0
        # [B, ...]
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        v_0_estim = self.v_0.forward(x_t, backup, **kwargs)
        # []
        v_loss = (v_0 - v_0_estim).square().mean()
        # [B, ...]
        target = 2 * (sample - prior) - 2 * v_0_estim.detach()
        estim = self.a_t.forward(x_t, v_0_estim.detach(), backup, **kwargs)
        # []
        a_loss = (target - estim).square().mean()
        return v_loss, a_loss

    DEFAULT_STEPS = 100

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        # assign default values
        steps = steps or self.DEFAULT_STEPS
        if verbose is None:
            verbose = lambda x: x
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        # loop
        x_t, x_ts = prior, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(steps)):
                t = torch.full((bsize,), i / steps, dtype=torch.float32)
                v_0 = self.v_0.forward(x_t, t, **kwargs)
                a_t = self.a_t.forward(x_t, v_0, t, **kwargs)
                x_t = x_t + v_0 / steps + (2 * i + 1) / (2 * steps) * a_t / steps
                x_ts.append(x_t)
        return x_t, x_ts
