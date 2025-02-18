from typing import Callable, Iterable, Self

import torch
import torch.nn as nn

from flowmodels.basis import ODEModel
from flowmodels.cm import EMASupports
from flowmodels.euler import VanillaEulerSolver


class ConsistencyFlowMatching(nn.Module, ODEModel):
    """Consistency Flow Matching: Defining Straight Flows with Velocity Consistency, Yang et al., 2024.[arXiv:2407.02398]"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.velocity_estim = module
        self.solver = VanillaEulerSolver()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the causalized velocity from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            estimated velocity from the given sample `x_t`.
        """
        return self.velocity_estim(x_t, t)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        src: torch.Tensor | None = None,
        ema: Self | EMASupports[Self] | None = None,
        delta_t: float = 1e-3,
        alpha: float = 1e-5,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            t: [FloatLike; [B]], target timesteps in range[0, 1 - delta_t],
                sample from uniform distribution if not provided.
            src: [FloatLike; [B, ...]], sample from the source distribution, `X_0`,
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
        if src is None:
            src = torch.randn_like(sample)
        # reduce to the Consistency FM
        ema = self._reduce_ema(ema)
        # compute objective
        backup = t
        # [B, ...]
        t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = t * sample + (1 - t) * src
        x_td = ((t + delta_t) * sample) + (1 - t - delta_t) * src
        # [B, ...]
        estim = self.forward(x_t, backup)
        with torch.inference_mode():
            estim_ema = ema.forward(x_td, backup + delta_t)
        # []
        flowmatch = (estim_ema - estim).square().mean()
        # [], estimated endpoint
        estim = x_t + (1 - t) * estim
        estim_ema = x_td + (1 - t - delta_t) * estim_ema
        # [], velocity consistency
        consistency = (estim - estim_ema).square().mean()
        return consistency + alpha * flowmatch

    def _reduce_ema(self, ema: Self | EMASupports[Self] | None = None) -> Self:
        match ema:
            case ConsistencyFlowMatching():
                pass
            case None:
                ema = self
            case EMASupports():
                ema = ema.module
        return ema

    def sample(
        self,
        src: torch.Tensor,
        steps: int = 1,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Transfer the samples from the prior distribution to the trained distribution, using vanilla Euler method.
        Args:
            src: [FloatLike; [B, ...]], samples from the source distribution, `X_0`.
            steps: the number of the steps.
        """
        return self.solver.solve(self, src, steps, verbose)
