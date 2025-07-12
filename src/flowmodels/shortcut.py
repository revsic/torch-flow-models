from typing import Callable, Iterable

import torch
import torch.nn as nn

from flowmodels.basis import ODEModel, ODESolver, SamplingSupports


class ShortcutModel(nn.Module, ODEModel, SamplingSupports):
    """One Step Diffusion via Shortcut Models, Frans et al., 2024.[arXiv:2410.12557]"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.velocity_estim = module
        self.solver = ShortcutEulerSolver()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate the causalized velocity from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
            d: [FloatLike; [B]], the current stepsize in range[0, 1].
        Returns:
            estimated velocity from the given sample `x_t`.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return self.velocity_estim(x_t, t, d, **kwargs)

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
        return self.forward(x_t, t, torch.zeros_like(t), label=label)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the source distribution, `X_0`,
                sample from gaussian if not provided.
            d: [FloatLike; [B]], target step sizes in range[0, 1],
                sample from U[1/128, ..., 1/2, 1] if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        # sample
        if t is None:
            t = torch.rand(batch_size)
        if d is None:
            d = 1 / (2 ** torch.randint(0, 8, size=(batch_size,)))
        if prior is None:
            prior = torch.randn_like(sample)
        # compute objective
        backup = t
        _expand: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.view(
            [batch_size] + [1] * (sample.dim() - 1)
        )
        # [B, ...]
        t = _expand(t)
        # [B, ...]
        x_t = t * sample + (1 - t) * prior
        # [B, ...], instantaneous flow
        estim = self.forward(x_t, backup, torch.zeros_like(backup), label=label)
        # []
        flowmatch = ((sample - prior) - estim).square().mean()
        # [B, ...]
        s_t = self.forward(x_t, backup, d, label=label)
        s_next = self.forward(x_t + s_t * _expand(d), backup + d, d, label=label)
        s_target = 0.5 * (s_t + s_next)
        # []
        estim = self.forward(x_t, backup, 2 * d, label=label)
        consistency = (estim - s_target.detach()).square().mean()
        return flowmatch + consistency

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
        return self.solver.solve_shortcut(self, prior, label, steps, verbose)


class ShortcutEulerSolver(ODESolver):
    """Euler Solver for Shortcut Model."""

    DEFAULT_STEPS = 128

    def solve_shortcut(
        self,
        model: ShortcutModel,
        init: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the shortcut model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps, default 128 iterations.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        # assign default values
        steps = steps or self.DEFAULT_STEPS
        if verbose is None:
            verbose = lambda x: x
        # loop
        x_t, x_ts = init, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(steps)):
                unit_velocity = model.forward(
                    x_t,
                    torch.full((bsize,), i / steps, dtype=torch.float32),
                    torch.full((bsize,), 1 / steps, dtype=torch.float32),
                    label=label,
                )
                x_t = x_t + unit_velocity / steps
                x_ts.append(x_t)
        return x_t, x_ts
