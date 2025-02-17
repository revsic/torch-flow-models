from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from flowmodels.basis import ContinuousScheduler, Sampler, Scheduler, ScoreModel


class MultistepConsistencySampling(Sampler):

    DEFAULT_STEPS = 4

    def __init__(self, scheduler: Scheduler | ContinuousScheduler):
        self.scheduler = scheduler

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
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        # [T]
        var: torch.Tensor
        match self.scheduler:
            case Scheduler():
                var = self.scheduler.var()
                if steps is not None:
                    # for easier step-sampling
                    assert self.scheduler.T % steps == 0
                    # [steps], step-size sampling from the last variance
                    # e.g. [3, 7, 11] will be sampled from [0, 1, ..., 11] with 3-steps
                    var = var.view(steps, -1).T[-1]
                steps = len(var)
            case ContinuousScheduler():
                steps = steps or self.DEFAULT_STEPS
                # [T], in range[1/T, 1]
                steps = torch.arange(1, steps + 1, dtype=torch.float32) / steps
                var = self.scheduler.var(steps)
        # single-step sampler
        batch_size, *_ = prior.shape
        if steps <= 1:
            x_0 = self._denoise(
                model, prior, torch.ones(batch_size), var[-1, None].repeat(batch_size)
            )
            return x_0, [x_0]
        # multi-step sampler
        x_t, x_ts = prior, []
        # [T + 1], one-based
        var = F.pad(var, [1, 0], mode="constant", value=1e-8)
        for t in range(steps, 0, -1):
            # [B], [B]
            v_t_d, v_t = var[t - 1 : t + 1, None].repeat(1, batch_size)
            # [B, ...]
            x_t = self._denoise(
                model,
                x_t,
                torch.full((batch_size,), t / steps, dtype=torch.float32),
                v_t,
                v_t_d,
            )
            x_ts.append(x_t)
        return x_t, x_ts

    def _denoise(
        self,
        model: ScoreModel,
        x_t: torch.Tensor,
        t: torch.Tensor,
        var: torch.Tensor,
        var_prev: torch.Tensor | None = None,
    ):
        """Single step sampler.
        Args:
            model: the score-estimation model.
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, `t` in range[0, 1].
            var: [FloatLike; [B]], the variance of the noise in time `t`.
            var_prev: [FloatLike; [B]], the variance of the noise in single step backward at time `t`,
                i.e. `var_{t - d}` of step size `d`.
        Returns:
            [FloatLike; [B, ...]], estimated sample `x_0` if var_prev is not given,
                otherwise `x_{t-d}` of implicit step size `d` given by `var_prev`.
        """
        # B
        batch_size, *_ = x_t.shape
        # [B, ...]
        var = var.view([batch_size] + [1] * (x_t.dim() - 1))
        # [B, ...]
        eps = -model.score(x_t, t) * var.sqrt().to(x_t)
        # [B, ...]
        x_0 = x_t - var.sqrt().to(x_t) * eps
        # variance exploding
        if not self.scheduler.vp:
            # return x_0 directly
            if var_prev is None:
                return x_0
            # [B, ...]
            var_prev = var_prev.view([batch_size] + [1] * (x_t.dim() - 1))
            # single-step backward to `x_{t-d}` where step size `d` is implicity given by `var_prev`
            return x_0 + var_prev.sqrt().to(x_t) * eps
        # variance preserving
        x_0 = x_0 * (1 - var).clamp_min(1e-7).rsqrt().to(x_0)
        if var_prev is None:
            return x_0
        # [B, ...]
        var_prev = var_prev.view([batch_size] + [1] * (x_t.dim() - 1))
        # single-step backward
        return (1 - var_prev).sqrt().to(x_0) * x_0 + var_prev.sqrt().to(x_0) * eps
