from typing import Callable, Iterable

import torch
import torch.nn as nn

from flowmodels.basis import ODEModel, SamplingSupports
from flowmodels.euler import VanillaEulerSolver


class RectifiedFlow(nn.Module, ODEModel, SamplingSupports):
    """
    Rectified Flow: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022.[arXiv:2209.03003],
    Flow Matching: Flow Matching for Generative Modeling, Lipman et al., 2022.[arXiv:2210.02747]
    """

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
    ) -> torch.Tensor:
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
        device = sample.device
        # sample
        if t is None:
            t = torch.rand(batch_size, device=device)
        if prior is None:
            prior = torch.randn_like(sample)
        # compute objective
        backup = t
        # [B, ...]
        t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = t * sample + (1 - t) * prior
        # [B, ...]
        estim = self.forward(x_t, backup, label=label)
        return ((sample - prior) - estim).square().mean()

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

    def distillation(
        self,
        optim: torch.optim.Optimizer,
        prior: torch.Tensor,
        training_steps: int,
        batch_size: int,
        sample: torch.Tensor | int = 1000,
        label: torch.Tensor | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        """Distillation for single-step generation."""
        return self.reflow(
            optim=optim,
            prior=prior,
            training_steps=training_steps,
            batch_size=batch_size,
            sample=sample,
            label=label,
            timesteps=torch.zeros(batch_size, dtype=prior.dtype, device=prior.device),
            verbose=verbose,
        )

    def reflow(
        self,
        optim: torch.optim.Optimizer,
        prior: torch.Tensor,
        training_steps: int,
        batch_size: int,
        sample: torch.Tensor | int = 1000,
        label: torch.Tensor | None = None,
        timesteps: Callable[[], torch.Tensor] | torch.Tensor | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        """Optimize the model with the reflow-procedure.
        Args:
            optim: optimizer constructed with `self.parameters()`.
            prior: [FloatLike; [D, ...]], samples from the source distribution, `X_0`.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            sample: [FloatLike; [D, ...]], corresponding samples transfered from `src`,
                sample just-in-time if given is integer (assume as the number of the steps for sampling iterations).
        """
        if isinstance(sample, int):
            with torch.inference_mode():
                sample, _ = self.sample(prior, label, sample, verbose)

        if verbose is None:
            verbose = lambda x: x

        losses = []
        self.train()
        for i in verbose(range(training_steps)):
            indices = torch.randint(0, len(sample), (batch_size,))
            t = timesteps() if callable(timesteps) else timesteps
            loss = self.loss(sample[indices], t=t, prior=prior[indices], label=label)
            # update
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log
            losses.append(loss.detach().item())

        return losses
