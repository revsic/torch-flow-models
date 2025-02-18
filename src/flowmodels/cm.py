import copy
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import (
    ContinuousScheduler,
    ForwardProcessSupports,
    Sampler,
    Scheduler,
    ScoreModel,
    ScoreSupports,
)


class ConsistencyDistillationSupports(ScoreSupports, ForwardProcessSupports, Protocol):
    scheduler: Scheduler | ContinuousScheduler


@dataclass
class ConsistencyModelScheduler(Scheduler):
    """Variance scheduler with a coefficient scheduler for ensuring the boundary condition,
    Consistency Models, Song et al., 2023. [arXiv:2303.01469]
    """

    T: int = 18 - 1
    vp: bool = False

    eps: float = 1e-7
    rho: float = 7.0
    var_max: float = 1.0

    sigma_data: float = 0.5

    def cskip(self, t: torch.Tensor) -> torch.Tensor:
        """Differentiable functions such that `cskip(eps) = 1` for a small constant `eps`.
        Args:
            t: [torch.long; [B]], the given timesteps, in range(eps, T].
        Returns:
            coefficients for `x` (s.t. f(x, t) = cskip(t) * x + cout(t) * F(x, t))
        """
        return (self.sigma_data**2) / ((t - self.eps).square() + self.sigma_data**2)

    def cout(self, t: torch.Tensor) -> torch.Tensor:
        """Differentiable functions such that `cout(eps) = 0` for a small constant `eps`.
        Args:
            t: [torch.long; [B]], the given timesteps, in range(eps, T].
        Returns:
            coefficients for `x` (s.t. f(x, t) = cskip(t) * x + cout(t) * F(x, t))
        """
        return (self.sigma_data * (t - self.eps)) * (
            self.sigma_data**2 + t.square()
        ).clamp_min(1e-7).rsqrt()

    def var(self) -> torch.Tensor:
        # shortcut
        eps, rho, N = self.eps, self.rho, self.T + 1
        # [T + 1]
        i = torch.arange(1, N + 1)
        # [T + 1]
        t_i = (
            eps ** (1 / rho)
            + (i - 1) / (N - 1) * (self.var_max ** (1 / rho) - eps ** (1 / rho))
        ) ** rho
        # [T], drop the t_0 = eps
        return t_i[1:]


class EMASupports[T: nn.Module](nn.Module):
    def __init__(self, module: T):
        super().__init__()
        self.module = copy.deepcopy(module)

    @torch.no_grad()
    def update(self, module: nn.Module, mu: float | torch.Tensor):
        given = dict(module.named_parameters())
        for name, param in self.module.named_parameters():
            assert name in given, f"parameters not found; named `{name}`"
            param.copy_(mu * param.data + (1 - mu) * given[name].data)

    @classmethod
    def reduce(cls, self_: T, ema: T | Self | None = None) -> T:
        if ema is None:
            return self_
        if isinstance(ema, EMASupports):
            return ema.module
        return ema


class ConsistencyModel(nn.Module, ScoreModel):
    """Consistency Models, Song et al., 2023. [arXiv:2303.01469]"""

    def __init__(self, module: nn.Module, scheduler: ConsistencyModelScheduler):
        super().__init__()
        self.module = module
        self.scheduler = scheduler
        self.sampler = MultistepConsistencySampler(scheduler)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [torch.long; [B]], the current timestep of the given sample `x_t`, in range[0, T].
        Returns:
            estimated `x_0`.
        """
        # B
        (bsize,) = t.shape
        # [B], clamp to slightly larger than eps
        t = t.clamp_min(self.scheduler.eps * (1 + 0.1))
        # [B]
        cskip, cout = self.scheduler.cskip(t), self.scheduler.cout(t)
        # [B, ...]
        cskip = cskip.view([bsize] + [1] * (x_t.dim() - 1))
        cout = cout.view([bsize] + [1] * (x_t.dim() - 1))
        # [B, ...]
        return cskip * x_t + cout * self.module(x_t, t)

    def score(self, x_t: torch.Tensor, t: torch.Tensor):
        """Estimate the stein score from the given sample `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep of the given sample `x_t`, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated score.
        """
        # discretize in range[0, T]
        t = (t * self.scheduler.T).long()
        # [B, ...]
        x_0 = self.forward(x_t, t)
        # [T + 1, ...]
        var = F.pad(self.scheduler.var(), [1, 0], "constant", 1e-7).view(
            [self.scheduler.T + 1] + [1] * (x_0.dim() - 1)
        )
        return ((1 - var[t]).sqrt().to(x_0) * x_0 - x_t) / var[t].to(x_0).clamp_min(
            1e-7
        )

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
                sample from uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution,
                sample from the gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        raise NotImplementedError("ScoreModel.loss is not implemented.")

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        return self.sampler.sample(self, prior, steps, verbose=verbose)

    def distillation(
        self,
        score_model: ConsistencyDistillationSupports,
        optim: torch.optim.Optimizer,
        training_steps: int,
        batch_size: int,
        sample: torch.Tensor,
        mu: float = 0.999,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[list[float], nn.Module]:
        """Consistency distillation from the score-model.
        Args:
            score_model: the target score model.
            optim: optimizer constructed with `self.parameters()`.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            sample: [FloatLike; [B, ...]], the samples from the data distribution.
            mu: the factor of an EMA model updates.
            loss_fn: the loss objective from estimated x_0 and EMA-estimated x_0.
        Returns:
            a list of the loss values and EMA-updated consistency models.
        """
        assert self.scheduler.vp == score_model.scheduler.vp, (
            "both schedulers should be same variance-manaing scheme; "
            "both variance-exploding or both variance-preserving"
        )
        # assign default values
        if verbose is None:
            verbose = lambda x: x

        # prepare the sampler
        sampler = MultistepConsistencySampler(score_model.scheduler)
        # [T + 1]
        var = F.pad(self.scheduler.var(), [1, 0], "constant", self.scheduler.eps)
        # EMA: Exponential Moving Average
        ema = EMASupports(self)

        losses: list[float] = []
        for i in verbose(range(training_steps)):
            # [B, ...]
            x_0 = sample[torch.randint(0, len(sample), (batch_size,))]
            # [B, ...]
            eps = torch.randn_like(x_0)
            # [B]
            t = torch.randint(1, self.scheduler.T + 1, size=(batch_size,))
            # [B], [B]
            v_t, v_td = var[t], var[t - 1]
            # noising
            x_t: torch.Tensor
            v_t_view = v_t.view([batch_size] + [1] * (x_0.dim() - 1))
            if self.scheduler.vp:
                x_t = (1 - v_t_view).sqrt() * x_0 + v_t_view.sqrt() * eps
            else:
                x_t = x_0 + v_t_view.sqrt() * eps
            with torch.inference_mode():
                # [B, ...]
                x_td = sampler._denoise(
                    score_model, x_t, t / self.scheduler.T, v_t, v_td
                )
                # [B, ...]
                x_0_ema = ema.module.forward(x_td, t - 1)
            # [B, ...]
            x_0_estim = self.forward(x_t, t)
            # []
            loss = loss_fn(x_0_estim, x_0_ema.clone().detach())
            # update
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log
            losses.append(loss.detach().item())
            # ema
            ema.update(self, mu)

        return losses, ema.module


class MultistepConsistencySampler(Sampler):

    DEFAULT_STEPS = 4

    def __init__(self, scheduler: Scheduler | ContinuousScheduler):
        self.scheduler = scheduler

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
        # [T]
        var = self._discretized_var(steps)
        # T
        steps = len(var)
        # single-step sampler
        batch_size, *_ = prior.shape
        if steps <= 1:
            x_0 = self._denoise(
                model, prior, torch.ones(batch_size), var[-1, None].repeat(batch_size)
            )
            return x_0, [x_0]
        # multi-step sampler
        x_t, x_ts = prior, []
        if verbose is None:
            verbose = lambda x: x
        # [T + 1], one-based
        var = F.pad(var, [1, 0], mode="constant", value=1e-8)
        for t in verbose(range(steps, 0, -1)):
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

    def _discretized_var(self, steps: int | None = None) -> torch.Tensor:
        """Construct the sequence of discretized variances."""
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
            case ContinuousScheduler():
                steps = steps or self.DEFAULT_STEPS
                # [T], in range[1/T, 1]
                var = self.scheduler.var(
                    torch.arange(1, steps + 1, dtype=torch.float32) / steps
                )
        return var

    def _denoise(
        self,
        model: ScoreSupports,
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
