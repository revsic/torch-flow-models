from typing import Callable, Iterable, Literal, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DMD2Supports(ForwardProcessSupports, ScoreSupports, Protocol):
    pass


_WEIGHTING_H = Callable[[torch.Tensor], torch.Tensor | float]


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

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep of the given sample `x_t`, in range[0, 1].
        Returns:
            estimated `x_0`.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return self.generator.forward(x_t, t, **kwargs)

    def predict(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep of the given sample `x_t`, in range[0, 1].
        Returns:
            estimated `x_0`.
        """
        return self.forward(x_t, t, label=label)

    def noise(
        self, sample: torch.Tensor, t: torch.Tensor, prior: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the `prior`.
        Args:
            sample: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        assert self._noiser is not None
        return self._noiser(sample, t, prior)

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        if steps is None or steps <= 1:
            bsize, *_ = prior.shape
            x_0 = self.predict(prior, torch.ones(bsize, device=prior.device), label)
            return x_0, [x_0]

        return self.sampler.sample(self, prior, label, steps, verbose)

    def dmd(
        self,
        teacher: DMDSupports,
        optim_g: torch.optim.Optimizer,
        optim_s: torch.optim.Optimizer,
        training_steps: int,
        batch_size: int,
        prior: torch.Tensor,
        samples: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
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
            lists of generator losses and fake score losses.
        """
        if verbose is None:
            verbose = lambda x: x
        # generate the datasets first
        if samples is None:
            with torch.inference_mode():
                samples, _ = teacher.sample(
                    prior, label, steps=_sampling_steps, verbose=verbose
                )
        # inherit teacher's forward process
        self._noiser = teacher.noise

        _label, kwargs = None, {}
        g_losses: list[float] = []
        s_losses: list[float] = []
        for i in verbose(range(training_steps)):
            indices = torch.randint(0, len(samples), (batch_size,))
            # [B, ...], [B, ...]
            x_0, z = samples[indices], prior[indices]
            # [B, ...]
            if label is not None:
                kwargs["label"] = _label = label[indices]
            x = self.generator.forward(z, **kwargs)
            # [B]
            t = torch.rand(batch_size)
            # [B, ...]
            x_t = teacher.noise(x, t, prior=None)
            # [B, ...]
            with torch.inference_mode():
                s_real = teacher.score(x_t, t, label=_label)
                s_fake = self.fake_score.score(x_t, t, label=_label)
            # update generator
            _dkl = (x * (s_fake - s_real).detach()).mean()
            _reg = (x - x_0).square().mean()
            g_loss = _dkl / (x_0 - x).abs().mean().detach() + _lambda * _reg
            g_loss.backward()
            optim_g.step()
            optim_g.zero_grad()

            with torch.inference_mode():
                # [B, ...]
                x = self.generator.forward(z, **kwargs)
            # []
            s_loss = self.fake_score.loss(x.detach(), label=_label)
            # update fake score
            s_loss.backward()
            optim_s.step()
            optim_s.zero_grad()

            # log
            g_losses.append(g_loss.detach().item())
            s_losses.append(s_loss.detach().item())

        return g_losses, s_losses

    _DEFAULT_DMD2_TIME_SAMPLER = lambda s: 0.25 * torch.randint(1, 4 + 1, (s,))

    SUPPORTING_DIV: dict[str, _WEIGHTING_H] = {
        "reverse-kl": lambda r: 1.0,
        "softened-rkl": lambda r: 1 / (r + 1),
        "jensen-shannon": lambda r: r / (r + 1),
        "squared-helligner": lambda r: 0.25 * r**0.5,
        "forward-kl": lambda r: r,
        "jeffreys": lambda r: r + 1,
    }

    _SUPPORTING_DIV = Literal[
        "reverse-kl",
        "softened-rkl",
        "jensen-shannon",
        "squared-helligner",
        "forward-kl",
        "jeffreys",
    ]

    def dmd2(
        self,
        teacher: DMD2Supports,
        discriminator: nn.Module,
        optim_g: torch.optim.Optimizer,
        optim_s: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        training_steps: int,
        batch_size: int,
        samples: torch.Tensor,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        h: _SUPPORTING_DIV | _WEIGHTING_H = "reverse-kl",
        verbose: Callable[[range], Iterable] | None = None,
        _score_updates: int = 5,
        _time_sampler: Callable[[int], torch.Tensor] = _DEFAULT_DMD2_TIME_SAMPLER,
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Improved Distribution Matching Distillation for Fast Image Synthesis, Yin et al., 2024.[arXiv:2405.14867]
        Adversarial Score identity Distillation: Rapidly Surpassing the Teacher in One Step, Zhou et al., 2024.[arXiv:2410.14919]
        One-step Diffusion Models with f-Divergence Distribution Matching, Xu et al., 2025.[arXiv:2502.15681]

        Args:
            teacher: the target score model.
            discriminator: the diffusion-gan discriminator.
            optim_g, optim_f, optim_d: optimizers constructed
                with `self.generator.parameters()`, `self.fake_score.parameters()` and `discriminator.parameters()`.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            samples: [FloatLike; [B, ...]], the samples from the data distribution.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            h: weighting term for f-divergence matching (default 1.0 for DMD2).
        Returns:
            lists of generator losses, fake score losses and discriminator losses.
        """
        if verbose is None:
            verbose = lambda x: x
        # inherit teacher's forward process
        self._noiser = teacher.noise

        assert callable(h) or h in self.SUPPORTING_DIV
        _h = h if callable(h) else lambda d: self.SUPPORTING_DIV[h](self._compute_r(d))

        _label, kwargs = None, {}
        g_losses: list[float] = []
        s_losses: list[float] = []
        d_losses: list[float] = []
        for i in verbose(range(training_steps)):
            # two time-scale update rule
            for _ in range(_score_updates):
                indices = torch.randint(0, len(prior), (batch_size,))
                z = prior[indices]
                if label is not None:
                    kwargs["label"] = _label = label[indices]
                with torch.inference_mode():
                    # [B, ...]
                    x = self.generator.forward(z, **kwargs)
                # []
                s_loss = self.fake_score.loss(x.detach(), label=_label)
                # update fake score
                s_loss.backward()
                optim_s.step()
                optim_s.zero_grad()

                s_losses.append(s_loss.detach().item())

            # discriminator updates
            indices = torch.randint(0, len(prior), (batch_size,))
            # [B, ...], [B, ...]
            x_0, z = samples[indices], prior[indices]
            if label is not None:
                kwargs["label"] = _label = label[indices]
            with torch.inference_mode():
                # [B, ...]
                x = self.generator.forward(z, **kwargs)
                # [B, ...]
                real_t = teacher.noise(x_0, _r_t := _time_sampler(batch_size).to(x_0))
                fake_t = teacher.noise(x, _f_t := _time_sampler(batch_size).to(x))
            # log D(x_t, t) = discriminator.forward(x_t, t)
            d_loss = (
                F.softplus(discriminator.forward(real_t, _r_t, **kwargs)).mean()
                + F.softplus(-discriminator.forward(fake_t, _f_t, **kwargs)).mean()
            ) * 1e-2
            d_loss.backward()
            optim_d.step()
            optim_d.zero_grad()

            d_losses.append(d_loss.detach().item())

            # generator loss
            indices = torch.randint(0, len(prior), (batch_size,))
            # [B, ...], [B, ...]
            x_0, z = samples[indices], prior[indices]
            if label is not None:
                kwargs["label"] = _label = label[indices]
            # [B, ...]
            x = self.generator.forward(z, **kwargs)
            # [B]
            t = _time_sampler(batch_size).to(x)
            # [B, ...]
            x_t = teacher.noise(x, t, prior=None)
            # [B, ...]
            with torch.inference_mode():
                s_real = teacher.score(x_t, t, label=_label)
                s_fake = self.fake_score.score(x_t, t, label=_label)
            # [B, ...]
            d = discriminator.forward(x_t, t, **kwargs)
            # update generator
            match = _h(d) * (s_fake - s_real)
            _dkl = (x * match.detach()).mean() / (x_0 - x).abs().mean().detach()
            _gan = F.softplus(d).mean()
            g_loss = _dkl + _gan * 3e-3
            g_loss.backward()
            optim_g.step()
            optim_g.zero_grad()

            g_losses.append(g_loss.detach().item())

        return g_losses, s_losses, d_losses

    @classmethod
    def _compute_r(cls, d: torch.Tensor) -> torch.Tensor:
        # reformulate from softplus objective to logits
        # log(1 + exp(d)) = log(d') => 1 + exp(d) = d'
        # log(1 + exp(-d)) = log(1 - d') => 1 + exp(-d) = 1 - d'
        # => exp(d) - exp(-d) = 2d' - 1
        # => d' = 0.5 * (1 + exp(d) - exp(-d))
        d = 0.5 * (1 + d.exp() - (-d).exp())
        # for numerical stability
        d[(d - 1.0).abs() < 1e-8] = 1 + 1e-8
        return (1 - d) / d
