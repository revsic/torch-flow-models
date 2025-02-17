from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from flowmodels.basis import ContinuousScheduler, Sampler, Scheduler, ScoreSupports
from flowmodels.vpsde import VPSDEAncestralSamplerSupports


class ProbabilityFlowODESampler(Sampler):
    """Probability flow ODE (ancestral) sampler for score model.
    Score-Based Generative Modeling Through Stochastic Differential Equations, Song et al., 2021.[arXiv:2011.13456]
    """

    DEFAULT_STEPS: int = 1000

    def __init__(self, scheduler: Scheduler | ContinuousScheduler):
        self.scheduler = scheduler

    def sample(
        self,
        model: ScoreSupports,
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

    def _compute_vp_beta(self, alpha_bar: torch.Tensor) -> torch.Tensor:
        """Compute the beta sequence.
        Args:
            alpha_bar: a list of the `1 - var_t` where `var_t` is discretized variances.
        Returns:
            a list of the betas.
        """
        (T,) = alpha_bar.shape
        if isinstance(self.scheduler, VPSDEAncestralSamplerSupports):
            _t = torch.arange(1, T + 1, dtype=torch.float32) / T
            beta = self.scheduler.betas(_t) / T
        else:
            # [T], emperical estimation
            beta = 1 - alpha_bar / F.pad(alpha_bar[:-1], [1, 0], "constant", 1.0)
        return beta

    def _denoise_vp(
        self,
        model: ScoreSupports,
        x_t: torch.Tensor,
        t: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise the given sample `x_t` to the single-step backward `x_{t-1}`.
        Args:
            model: variance-preserving score model.
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [torch.long; [B]], the current timesteps in range[1, T].
            beta: [FloatLike; [T]], a list of betas.
        Returns:
            [FloatLike; [B, ...]], denoised sample, `x_{t-1}`.
        """
        # T
        (T,) = beta.shape
        # B
        (bsize,) = t.shape
        # [B, ...]
        b = beta[t - 1].view([bsize] + [1] * (x_t.dim() - 1))
        # [B, ...]
        score = model.score(x_t, t / T)
        return (2 - (1 - b).sqrt()).to(x_t) * x_t + 0.5 * b.to(x_t) * score

    def _solve_variance_preserving_score_model(
        self,
        model: ScoreSupports,
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
        beta = self._compute_vp_beta(_alpha_bar := 1 - self._discretized_var(steps))
        # T
        (T,) = beta.shape
        x_t, x_ts = prior, []
        with torch.inference_mode():
            for t in verbose(range(T, 0, -1)):
                t = torch.full((bsize,), t, dtype=torch.long)
                x_t = self._denoise_vp(model, x_t, t, beta)
                x_ts.append(x_t)
        return x_t, x_ts

    def _denoise_ve(
        self,
        model: ScoreSupports,
        x_t: torch.Tensor,
        t: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise the given sample `x_t` to the single-step backward `x_{t-1}`.
        Args:
            model: variance-exploding score model.
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [torch.long; [B]], the current timesteps in range[1, T].
            var: [FloatLike; [T]] a list of variances.
        Returns:
            [FloatLike; [B, ...]], denoised sample, `x_{t-1}`.
        """
        # T
        (T,) = var.shape
        # [T, ...]
        var = var.view([T] + [1] * (x_t.dim() - 1))
        # [B, ...]
        score = model.score(x_t, t / T)
        return x_t + 0.5 * (var[t - 1] - var[(t - 2).clamp_min(0)]).to(score) * score

    def _solve_variance_exploding_score_model(
        self,
        model: ScoreSupports,
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
            for t in verbose(range(T, 0, -1)):
                t = torch.full((bsize,), t, dtype=torch.long)
                x_t = self._denoise_ve(model, x_t, t, var)
                x_ts.append(x_t)
        return x_t, x_ts
