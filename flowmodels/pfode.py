from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from flowmodels.basis import ODEModel, ODESolver, Scheduler, ScoreModel


class DiscretizedProbabilityFlowODE(ODEModel, ScoreModel):
    """Converter from discretized score model to probability flow ODE,
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    def __init__(self, model: ScoreModel, scheduler: Scheduler):
        self.score_model = model
        self.scheduler = scheduler
        assert self.scheduler.vp, "variance-exploding scheduler is not supported yet."

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the stein score from the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated stein scores.
        """
        return self.score_model.score(x_t, t)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        # [B, ...]
        score = self.score(x_t, t)
        # [T, ...]
        alpha_bar = (1 - self.scheduler.var()).view(
            [self.scheduler.T] + [1] * (x_t.dim() - 1)
        )
        # [T, ...]
        beta = 1 - alpha_bar / F.pad(
            alpha_bar[:-1], [0, 0] * (x_t.dim() - 1) + [1, 0], "constant", 1.0
        )
        # discretize in range[0, T]
        t = (t * self.scheduler.T).long()
        # zero-based
        t = (t - 1).clamp_min(0)
        return 0.5 * beta[t] * (x_t + score)


class DiscretizedProbabilityFlowODESolver(ODESolver):
    """Discretized probability flow ODE (ancestral) solver for time-discretized score model.
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def solve(
        self,
        model: ODEModel,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the discretized score estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        assert isinstance(
            model, DiscretizedProbabilityFlowODE
        ), "supports only models typed `DiscretizedProbabilityFlowODE`"
        # variance preserving
        if self.scheduler.vp:
            return self._solve_variance_preserving_score_model(
                model,
                init,
                steps,
                verbose,
            )
        # variance exploding
        return self._solve_variance_exploding_score_model(
            model,
            init,
            steps,
            verbose,
        )

    def _solve_variance_preserving_score_model(
        self,
        model: DiscretizedProbabilityFlowODE,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the discretized time-preserving score estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        assert self.scheduler.vp, "unsupported scheduler; variance-exploding scheduler"
        assert steps is None or steps == self.scheduler.T, "unsupported steps"
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        # B
        bsize, *_ = init.shape
        # [T, ...]
        alpha_bar = (1 - self.scheduler.var()).view(
            [self.scheduler.T] + [1] * (init.dim() - 1)
        )
        # [T, ...]
        beta = 1 - alpha_bar / F.pad(
            alpha_bar[:-1], [0, 0] * (init.dim() - 1) + [1, 0], "constant", 1.0
        )
        x_t, x_ts = init, []
        with torch.inference_mode():
            for t in verbose(range(self.scheduler.T, 0, -1)):
                score = model.score(
                    x_t,
                    torch.full((bsize,), t / self.scheduler.T),
                )
                b = beta[t - 1, None]
                x_t = (2 - (1 - b).sqrt().to(x_t)) * x_t + 0.5 * b.to(x_t) * score
                x_ts.append(x_t)
        return x_t, x_ts

    def _solve_variance_exploding_score_model(
        self,
        model: DiscretizedProbabilityFlowODE,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the discretized variance-exploding score estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        assert (
            not self.scheduler.vp
        ), "unsupported scheduler; variance-preserving scheduler"
        assert steps is None or steps == self.scheduler.T, "unsupported steps"
        # assign default values
        if verbose is None:
            verbose = lambda x: x
        # B
        bsize, *_ = init.shape
        # [T + 1, ...]
        var = F.pad(self.scheduler.var(), [1, 0], "constant", 0.0).view(
            [self.scheduler.T + 1] + [1] * (init.dim() - 1)
        )
        x_t, x_ts = init, []
        with torch.inference_mode():
            for t in verbose(range(self.scheduler.T, 0, -1)):
                score = model.score(
                    x_t,
                    torch.full((bsize,), t / self.scheduler.T),
                )
                x_t = x_t + 0.5 * (var[t - 1] - var[t - 2]).to(score) * score
                x_ts.append(x_t)
        return x_t, x_ts
