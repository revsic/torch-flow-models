from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import ODEModel, SamplingSupports
from flowmodels.euler import VanillaEulerSolver


class ConstantVelocityConsistencyModels(
    nn.Module, ODEModel, SamplingSupports,
):
    def __init__(self, module: nn.Module, p_mean: float = -1.0, p_std: float = 1.4):
        super().__init__()
        self.F0 = module
        self.p_mean = p_mean
        self.p_std = p_std
        self._ada_weight = nn.Linear(1, 1)
        self.solver = VanillaEulerSolver()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated sample from the given `x_t`.
        """
        (bsize,) = t.shape
        return x_t + (1 - t.view([bsize] + [1] * (x_t.dim() - 1))) * self.F0.forward(x_t, t)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        return self.F0.forward(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from the proposal distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the prior distribution, `X_0`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        # shortcut
        batch_size, *_ = sample.shape
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            # sample from log-normal
            rw_t = (torch.randn(batch_size) * self.p_std + self.p_std).exp()
            # [T], in range[0, 1]
            t = rw_t.atan() / np.pi * 2
        # [B, ...]
        _t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = _t * sample + (1 - _t) * prior
        # [B, ...]
        v_t = sample - prior
        with torch.no_grad():
            # [B, ...], [B, ...], jvp = dF/dt
            F, jvp, *_ = torch.func.jvp(  # pyright: ignore [reportPrivateImportUsage]
                self.F0.forward,
                (x_t, t),
                (v_t, torch.ones_like(t)),
            )
        F: torch.Tensor = F.detach()
        jvp: torch.Tensor = jvp.detach()
        # df/dt = (1 - t) * [(x_1 - x_0) - F(x_t, t) + (1 - t) * dF/dt]
        grad = (1 - _t) * (v_t - F + (1 - _t) * jvp)
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # normalized tangent
        normalized_tangent = grad / (
            grad.norm(p=2, dim=rdim, keepdim=True) + 0.1
        )
        # [B, ...]
        estim: torch.Tensor = self.F0.forward(x_t, t)
        # [B]
        mse = (estim - F - normalized_tangent).square().mean(dim=rdim)
        # [B], adaptive weighting
        logvar = self._ada_weight.forward(_t).squeeze(dim=-1)
        # [B]
        loss = (mse * logvar.exp() - logvar)
        # []
        return loss.mean()

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        # pre-scale the prior
        return self.solver.solve(self, prior, steps, verbose=verbose)
