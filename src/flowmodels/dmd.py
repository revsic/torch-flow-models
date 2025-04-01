from copy import deepcopy
from typing import Callable, Iterable, Protocol

import torch
import torch.nn as nn

from flowmodels.basis import (
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
    ScoreModel,
    ScoreSupports,
)
from flowmodels.cm import MultistepConsistencySampler


class DMDSupports(ForwardProcessSupports, SamplingSupports, ScoreSupports, Protocol):
    pass


class DistributionMatchingDistillation(
    nn.Module, ForwardProcessSupports, PredictionSupports
):
    """
    One-step Diffusion with Distribution Matching Distillation, Yin et al., 2023. [arXiv:2311.18828]
    Improved Distribution Matching Distillation for Fast Image Synthesis, Yin et al., 2024. [arXiv:2405.14867]
    One-step Diffusion Models with f-Divergence Distribution Matching, Xu et al., 2025. [arXiv:2502.15681]
    """

    def __init__(self, module: nn.Module, fake_score: ScoreModel):
        super().__init__()
        self.generator = module
        self.fake_score = fake_score
        self._noiser: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
            | None
        ) = None
        self.sampler = MultistepConsistencySampler()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep of the given sample `x_t`, in range[0, 1].
        Returns:
            estimated `x_0`.
        """
        return self.generator.forward(x_t, t)

    def predict(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep of the given sample `x_t`, in range[0, 1].
        Returns:
            estimated `x_0`.
        """
        return self.forward(x_t, t)

    def noise(
        self, x_0: torch.Tensor, t: torch.Tensor, prior: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the `prior`.
        Args:
            x_0: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        assert self._noiser is not None
        return self._noiser(x_0, t, prior)

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        if steps is None or steps <= 1:
            bsize, *_ = prior.shape
            x_0 = self.predict(prior, torch.ones(bsize, device=prior.device))
            return x_0, [x_0]

        return self.sampler.sample(self, prior, steps, verbose)

    def dmd(
        self,
        teacher: DMDSupports,
        optim_g: torch.optim.Optimizer,
        optim_s: torch.optim.Optimizer,
        training_steps: int,
        batch_size: int,
        prior: torch.Tensor,
        samples: torch.Tensor | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        _sampling_steps: int | None = None,
        _lambda: float = 0.25,
    ):
        """Distribution matching distillation from the given teacher score-model.
        Args:
            teacher: the target score model.
            optim_g, optim_f: optimizers constructed
                with `self.generator.parameters()` and `self.fake_score.parameters()`.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            TBD
        """
        if verbose is None:
            verbose = lambda x: x
        # generate the datasets first
        if samples is None:
            with torch.inference_mode():
                samples, _ = teacher.sample(
                    prior, steps=_sampling_steps, verbose=verbose
                )
        # inherit teacher's forward process
        self._noiser = teacher.noise

        g_losses: list[float] = []
        s_losses: list[float] = []
        for i in verbose(range(training_steps)):
            indices = torch.randint(0, len(samples), (batch_size,))
            # [B, ...], [B, ...]
            x_0, z = samples[indices], prior[indices]
            # [B, ...]
            x = self.generator.forward(z)
            # [B]
            t = torch.rand(batch_size)
            # [B, ...]
            x_t = teacher.noise(x, t, prior=None)
            # [B, ...]
            with torch.inference_mode():
                s_real = teacher.score(x_t, t)
                s_fake = self.fake_score.score(x_t, t)
            # update generator
            _dkl = (x * (s_fake - s_real).detach()).mean()
            _reg = (x - x_0).square().mean()
            g_loss = _dkl / (x_0 - x).abs().mean().detach() + _lambda * _reg
            g_loss.backward()
            optim_g.step()
            optim_g.zero_grad()

            with torch.inference_mode():
                # [B, ...]
                x = self.generator.forward(z)
            # []
            s_loss = self.fake_score.loss(x.detach())
            # update fake score
            s_loss.backward()
            optim_s.step()
            optim_s.zero_grad()

            # log
            g_losses.append(g_loss.detach().item())
            s_losses.append(s_loss.detach().item())

        return g_losses, s_losses

    def dmd2(self):
        """
        Improved Distribution Matching Distillation for Fast Image Synthesis
        Adversarial Score identity Distillation: Rapidly Surpassing the Teacher in One Step
        """
        ...

    def fdmd(self):
        """One-step Diffusion Models with f-Divergence Distribution Matching"""
        ...
