from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import (
    ContinuousScheduler,
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
    Scheduler,
    ScoreSupports,
)
from flowmodels.utils import EMASupports, backward_process


class ConsistencyDistillationSupports(ScoreSupports, Protocol):
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


class ConsistencyModel(
    nn.Module, PredictionSupports, ForwardProcessSupports, SamplingSupports
):
    """Consistency Models, Song et al., 2023. [arXiv:2303.01469]"""

    def __init__(self, module: nn.Module, scheduler: ConsistencyModelScheduler):
        super().__init__()
        self.module = module
        self.scheduler = scheduler
        self.sampler = MultistepConsistencySampler()

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

    def predict(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the sample points `x_0` from the `x_t` w.r.t. the timestep `t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            the predicted sample points `x_0`.
        """
        return self.forward(x_t, (t * self.scheduler.T).long())

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the `prior`.
        Args:
            x_0: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        (batch_size,) = t.shape
        if prior is None:
            prior = torch.randn_like(x_0)
        # [T + 1]
        var = F.pad(self.scheduler.var(), [1, 0], "constant", self.scheduler.eps)
        # [B]
        t = (t * self.scheduler.T).long()
        # noising
        v_t_view = var[t].view([batch_size] + [1] * (x_0.dim() - 1))
        if self.scheduler.vp:
            return (1 - v_t_view).sqrt() * x_0 + v_t_view.sqrt() * prior
        return x_0 + v_t_view.sqrt() * prior

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
            "both schedulers should be same variance-managing scheme; "
            "both variance-exploding or both variance-preserving"
        )
        # assign default values
        if verbose is None:
            verbose = lambda x: x

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
            # [B, ...]
            x_t = self.noise(x_0, t / self.scheduler.T, eps)
            with torch.inference_mode():
                # [B, ...]
                x_td = backward_process(
                    score_model,
                    x_t,
                    t / self.scheduler.T,
                    v_t,
                    v_td,
                    self.scheduler.vp,
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


class MultistepConsistencySamplerSupports(
    PredictionSupports, ForwardProcessSupports, Protocol
):
    pass


class MultistepConsistencySampler:

    DEFAULT_STEPS = 4

    def sample(
        self,
        model: MultistepConsistencySamplerSupports,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample from the prior distribution to the trained distribution.
        Args:
            model: the point prediction model with forward process supports.
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
            steps: the number of the sampling steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], generated samples.
            `T` x [FloatLike; [B, ...]], trajectories.
        """
        # T
        steps = steps or self.DEFAULT_STEPS
        # single-step sampler
        batch_size, *_ = prior.shape
        if steps <= 1:
            x_0 = model.predict(prior, torch.ones(batch_size))
            return x_0, [x_0]
        # multi-step sampler
        x_t, x_ts = prior, []
        if verbose is None:
            verbose = lambda x: x
        for t in verbose(range(steps, 0, -1)):
            # [B]
            t = torch.full((batch_size,), t / steps, dtype=torch.float32)
            # [B, ...]
            x_0 = model.predict(x_t, t)
            # [B, ...]
            x_t = model.noise(x_0, t - 1 / steps, prior)
            x_ts.append(x_t)
        return x_t, x_ts
