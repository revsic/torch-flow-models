import copy
from typing import Callable, Iterable, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import (
    ContinuousScheduler,
    ForwardProcessSupports,
    SamplingSupports,
    ScoreModel,
    ScoreSupports,
    Scheduler,
)
from flowmodels.cm import MultistepConsistencySampler
from flowmodels.pfode import ProbabilityFlowODESampler


@runtime_checkable
class RectificationSupports(ScoreSupports, ForwardProcessSupports, Protocol):
    scheduler: ContinuousScheduler | Scheduler

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the noise, score, or data sample from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]], the given noised samples, `x_t`.
            t: [torch.long; [B]], the current timesteps of the noised sample in range[0, 1].
        Returns:
            estimated noise, score, or data sample from the given sample `x_t`.
        """
        ...

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the training data, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from the uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution,
                sample from the gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        ...


class RecitifedDiffusion(nn.Module, ScoreModel, SamplingSupports):
    """Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow, Wang et al., 2024. [arXiv:2410.07303]"""

    def __init__(self, model: RectificationSupports):
        super().__init__()
        if not isinstance(model, RectificationSupports):
            raise TypeError("model must be suitable with `RectificationSupports`")
        self.model = model
        self.sampler = MultistepConsistencySampler(self.model.scheduler)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward to `self.model.forward`."""
        return self.model.forward(x_t, t)

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward to `self.model.score`."""
        return self.model.score(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward to `self.model.loss`."""
        return self.model.loss(sample, t, prior=prior)

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        return self.sampler.sample(self, prior, steps, verbose=verbose)

    def rectify(
        self,
        optim: torch.optim.Optimizer,
        prior: torch.Tensor,
        training_steps: int,
        batch_size: int,
        sample: torch.Tensor | int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[list[float], torch.Tensor]:
        """Optimize the model with the reflow-procedure.
        Args:
            optim: optimizer constructed with `self.parameters()`.
            prior: [FloatLike; [D, ...]], samples from the prior distribution.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            sample: [FloatLike; [D, ...]], corresponding samples transfered from `src`,
                sample just-in-time if given is integer (assume as the number of the steps for sampling iterations).
        """
        if sample is None or isinstance(sample, int):
            with torch.inference_mode():
                sampler = ProbabilityFlowODESampler(self.model.scheduler)
                sample, _ = sampler.sample(self.model, prior, sample, verbose)

        if verbose is None:
            verbose = lambda x: x

        losses = []
        self.train()
        for i in verbose(range(training_steps)):
            indices = torch.randint(0, len(sample), (batch_size,))
            loss = self.loss(sample[indices], prior=prior[indices])
            # update
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log
            losses.append(loss.detach().item())

        return losses, sample

    def consistency_distillation(
        self,
        optim: torch.optim.Optimizer,
        sample: torch.Tensor,
        training_steps: int,
        batch_size: int,
        mu: float = 0.99,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        """Optimize the model with the reflow-procedure.
        Args:
            optim: optimizer constructed with `self.parameters()`.
            sample: [FloatLike; [D, ...]], samples from the data distribution.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
        """
        sampler = ProbabilityFlowODESampler(self.model.scheduler)
        # [T]
        var = sampler._discretized_var()
        # T
        (T,) = var.shape
        # EMA purpose
        ema = copy.deepcopy(self)
        # ([FloatLike; [B, ...]], [torch.long; [B]]) -> [FloatLike, [B, ...]]
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        if self.model.scheduler.vp:
            # [T]
            beta = sampler._compute_vp_beta(1 - var)
            # variance-preserving denoiser
            denoiser = lambda x_t, t: sampler._denoise_vp(ema, x_t, t, beta)
        else:
            # variance-exploding denoiser
            denoiser = lambda x_t, t: sampler._denoise_ve(ema, x_t, t, var)

        if verbose is None:
            verbose = lambda x: x

        losses = []
        self.train()
        for i in verbose(range(training_steps)):
            # [B, ...]
            x = sample[torch.randint(0, len(sample), (batch_size,))]
            # [B]
            t = torch.randint(1, T + 1, (batch_size,))
            # [B, ...]
            x_t = self.model.noise(x, t / T)
            with torch.inference_mode():
                # [B, ...]
                x_prev = denoiser(x_t, t)
                # [B, ...]
                ema_prev = ema.forward(x_prev, (t - 1) / T)
            # [], TODO: replace `self.forward(x_prev, ...)` to EMA
            loss = loss_fn(self.forward(x_t, t / T), ema_prev.clone().detach())
            # update
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log
            losses.append(loss.detach().item())
            # ema
            with torch.no_grad():
                for _ema, _self in zip(ema.parameters(), self.parameters()):
                    _ema.copy_(mu * _ema.data + (1 - mu) * _self.data)

        # assign
        with torch.no_grad():
            for _ema, _self in zip(ema.parameters(), self.parameters()):
                _ema.copy_(_self.data)

        return losses
