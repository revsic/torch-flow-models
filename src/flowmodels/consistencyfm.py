from typing import Callable, Iterable, Self

import torch
import torch.nn as nn

from flowmodels.basis import ODEModel, SamplingSupports
from flowmodels.euler import VanillaEulerSolver
from flowmodels.utils import EMASupports


class ConsistencyFlowMatching(nn.Module, ODEModel, SamplingSupports):
    """Consistency Flow Matching: Defining Straight Flows with Velocity Consistency, Yang et al., 2024.[arXiv:2407.02398]"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.velocity_estim = module
        self.solver = VanillaEulerSolver()

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the causalized velocity from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            estimated velocity from the given sample `x_t`.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return self.velocity_estim(x_t, t, **kwargs)

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
        return self.forward(x_t, t, label=label)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        ema: Self | EMASupports[Self] | None = None,
        delta_t: float = 1e-3,
        alpha: float = 1e-5,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            t: [FloatLike; [B]], target timesteps in range[0, 1 - delta_t],
                sample from uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the source distribution, `X_0`,
                sample from gaussian if not provided.
            ema: the EMA model for consistency training, use `self` if not provided.
            delta_t: the small value for time difference,
                default value 1e-3 is given by the official repository,
                github+YangLing0818/consistency_flow_matching
            alpha: the weighting term for direct flow match object,
                default value 1e-5 is given by the official repository.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        # sample
        if t is None:
            t = torch.rand(batch_size).clamp_max(1 - delta_t)
        if prior is None:
            prior = torch.randn_like(sample)
        # reduce to the Consistency FM
        ema = EMASupports[Self].reduce(self, ema)
        # compute objective
        backup = t
        # [B, ...]
        t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = t * sample + (1 - t) * prior
        x_td = ((t + delta_t) * sample) + (1 - t - delta_t) * prior
        # [B, ...]
        estim = self.forward(x_t, backup, label=label)
        with torch.inference_mode():
            estim_ema = ema.forward(x_td, backup + delta_t, label=label)
        # []
        flowmatch = (estim_ema - estim).square().mean()
        # [], estimated endpoint
        estim = x_t + (1 - t) * estim
        estim_ema = x_td + (1 - t - delta_t) * estim_ema
        # [], velocity consistency
        consistency = (estim - estim_ema).square().mean()
        return consistency + alpha * flowmatch

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = 1,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Transfer the samples from the prior distribution to the trained distribution, using vanilla Euler method.
        Args:
            prior: [FloatLike; [B, ...]], samples from the source distribution, `X_0`.
            steps: the number of the steps.
        """
        return self.solver.solve(self, prior, label, steps, verbose)
