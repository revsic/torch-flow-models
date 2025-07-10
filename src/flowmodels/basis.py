from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

import torch


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol for variance scheduler."""

    T: int
    vp: bool

    def var(self) -> torch.Tensor:
        """Return the variances of the discrete-time diffusion models.
        Returns:
            [FloatLike; [T]], list of the time-dependent variances.
        """
        ...


@dataclass
class Scheduler(SchedulerProtocol):
    """Variance schedule.
    Attributes:
        T: the number of the sampling iterations.
        vp: whether the scheduler is for variance-preserving score model or variance-exploding score model.
    """

    T: int
    vp: bool = True

    def var(self) -> torch.Tensor:
        """Return the variances of the discrete-time diffusion models.
        Returns:
            [FloatLike; [T]], list of the time-dependent variances.
        """
        raise NotImplementedError("Scheduler.var is not implemented.")


@runtime_checkable
class ContinuousSchedulerProtocol(Protocol):
    """Protocol for variance scheduler."""

    vp: bool

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the variance of the prior distribution at each timesteps, `t`.
        Args:
            t: [FloatLike; [B]], the target timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B]], the variances at each timesteps.
        """
        ...


@dataclass
class ContinuousScheduler(ContinuousSchedulerProtocol):
    """Continuous-time scheduler."""

    vp: bool = True

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the variance of the prior distribution at each timesteps, `t`.
        Args:
            t: [FloatLike; [B]], the target timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B]], the variances at each timesteps.
        """
        raise NotImplementedError("ContinuousScheduler.var is not implemented.")


@runtime_checkable
class ScoreSupports(Protocol):

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the stein score from the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated stein scores.
        """
        ...


@runtime_checkable
class ForwardProcessSupports(Protocol):

    def noise(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample to the `x_t` w.r.t. the timestep `t` and the `prior`.
        Args:
            sample: [FloatLike; [B, ...]], the given samples.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        ...


@runtime_checkable
class PredictionSupports(Protocol):

    def predict(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the sample points from the `x_t` w.r.t. the timestep `t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            the predicted sample points.
        """
        ...


class ObjectiveSupports(Protocol):

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the training data.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution,
                sample from the gaussian if not provided.
            label: [FloatLike; [B, ...]], additional conditions.
        Returns:
            [FloatLike; []], loss value.
        """
        ...


class ScoreModel(ScoreSupports, ObjectiveSupports):
    """Basis of the score models."""


@runtime_checkable
class VelocitySupports(Protocol):

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        ...


class ODEModel(VelocitySupports, ObjectiveSupports):
    """Basis of the ODE models."""


@runtime_checkable
class SamplingSupports(Protocol):

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample from the trained distribution from the prior samples.
        Args:
            prior: [FloatLike; [B, ...]], the given prior samples.
            steps: the number of the steps to sample.
        Returns:
            [FloatLike; [B, ...]], sampled data.
            `T` x [FloatLike; [B, ...]], sampling trajectories.
        """
        ...


class Sampler:
    """Score-based sampler."""

    def sample(
        self,
        model: ScoreSupports,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample from the prior distribution to the trained distribution.
        Args:
            model: the score estimation model.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        raise NotImplementedError("Sampler.sample is not implemented.")


class ODESolver:
    """ODE Solver."""

    def solve(
        self,
        model: VelocitySupports,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the velocity estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        raise NotImplementedError("ODESolver.solve is not implemented.")
