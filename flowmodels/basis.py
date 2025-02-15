from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass
class Scheduler:
    """Variance scheduler"""

    T: int
    vp: bool = True

    def var(self) -> torch.Tensor:
        """Return the variances of the discrete-time diffusion models.
        Returns:
            [FloatLike; [T]], list of the time-dependent variances.
        """
        raise NotImplementedError("Scheduler.var is not implemented.")


class ScoreModel:
    """Basis of the score models."""

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the stein score from the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated stein scores.
        """
        raise NotImplementedError("ScoreModel.score is not implemented.")

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the training data, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        raise NotImplementedError("ScoreModel.loss is not implemented.")


class Sampler:
    """Score-based sampler."""

    def sample(
        self,
        model: ScoreModel,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample from the prior distribution to the trained distribution.
        Args:
            model: the score estimation model.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        raise NotImplementedError("Sampler.sample is not implemented.")
