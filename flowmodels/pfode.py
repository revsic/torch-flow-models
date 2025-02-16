from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from flowmodels.basis import ContinuousScheduler, Sampler, Scheduler, ScoreModel


class ProbabilityFlowODESampler(Sampler):
    """Probability flow ODE (ancestral) sampler for score model.
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    DEFAULT_STEPS: int = 1000

    def __init__(self, scheduler: Scheduler | ContinuousScheduler):
        self.scheduler = scheduler

    def _discretized_var(self, steps: int | None = None) -> torch.Tensor:
        """Discretize the variance sequence.
        Args:
            steps: the number of the steps.
        Returns:
            [FloatLike; [steps]], discretized variance sequence.
        """
        # discrete-time scheduler
        if isinstance(self.scheduler, Scheduler):
            assert steps is None or self.scheduler.T == steps
            return self.scheduler.var()
        # continuous-time scheduler
        steps = steps or self.DEFAULT_STEPS
        # [T], in range[1 / T, 1]
        timesteps = torch.arange(1, steps + 1, dtype=torch.float32) / steps
        return self.scheduler.var(timesteps)

    def sample(
        self,
        model: ScoreModel,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the Probability Flow ODE defined in the range of t; [0, 1].
        Args:
            model: the score estimation model.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        # variance preserving
        if self.scheduler.vp:
            return self._solve_variance_preserving_score_model(
                model, prior, steps, verbose
            )
        # variance exploding
        return self._solve_variance_exploding_score_model(model, prior, steps, verbose)

    def _solve_variance_preserving_score_model(
        self,
        model: ScoreModel,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the Probability Flow ODE defined in the range of t; [0, 1].
        Args:
            model: the variance-preserving score estimation model.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        assert self.scheduler.vp, "unsupported scheduler; variance-exploding scheduler"
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        # B
        bsize, *_ = prior.shape
        # [T]
        alpha_bar = 1 - self._discretized_var(steps)
        # T
        (T,) = alpha_bar.shape
        # [T]
        beta = 1 - alpha_bar / F.pad(alpha_bar[:-1], [1, 0], "constant", 1.0)
        x_t, x_ts = prior, []
        with torch.inference_mode():
            for t in verbose(range(T, 0, -1)):
                score = model.score(x_t, torch.full((bsize,), t / T))
                b = beta[t - 1]
                x_t = (2 - (1 - b).sqrt().to(x_t)) * x_t + 0.5 * b.to(x_t) * score
                x_ts.append(x_t)
        return x_t, x_ts

    def _solve_variance_exploding_score_model(
        self,
        model: ScoreModel,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the Probability Flow ODE defined in the range of t; [0, 1].
        Args:
            model: the variance-exploding score estimation model.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        assert (
            not self.scheduler.vp
        ), "unsupported scheduler; variance-preserving scheduler"
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        # B
        bsize, *_ = prior.shape
        # [T]
        var = self._discretized_var(steps)
        # T
        (T,) = var.shape
        x_t, x_ts = prior, []
        with torch.inference_mode():
            for t in verbose(range(T, 1, -1)):
                score = model.score(x_t, torch.full((bsize,), t / T))
                x_t = x_t + 0.5 * (var[t - 1] - var[t - 2]).to(score) * score
                x_ts.append(x_t)
        return x_t, x_ts
