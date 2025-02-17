import copy
from typing import Callable, Iterable, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import ContinuousScheduler, Sampler, ScoreModel, Scheduler
from flowmodels.pfode import ProbabilityFlowODESampler


@runtime_checkable
class ScoreModelProtocol(Protocol):
    scheduler: ContinuousScheduler | Scheduler

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the noise, score, or data sample from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]], the given noised samples, `x_t`.
            t: [torch.long; [B]], the current timesteps of the noised sample in range[0, 1].
        Returns:
            estimated noise, score, or data sample from the given sample `x_t`.
        """
        ...

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the stein score from the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated stein scores.
        """

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the training data, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from the uniform distribution if not provided.
            eps: [FloatLike; [B, ...]], the samples from the prior distribution,
                sample from the gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        ...

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the noise `eps`.
        Args:
            x_0: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [torch.long; [B]], the target timesteps in range[0, 1].
            eps: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        ...


class RecitifedDiffusion(nn.Module, ScoreModel):
    """Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow, Wang et al., 2024. [arXiv:2410.07303]"""

    def __init__(self, model: ScoreModelProtocol):
        super().__init__()
        if not isinstance(model, ScoreModelProtocol):
            raise TypeError("model must be suitable with `ScoreModelProtocol`")
        self.model = model
        self.sampler = MultistepConsistencySampling(self.model.scheduler)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward to `self.model.forward`."""
        return self.model.forward(x_t, t)

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward to `self.model.score`."""
        return self.model.score(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward to `self.model.loss`."""
        return self.model.loss(sample, t, eps=eps)

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        return self.sampler.sample(self, prior, steps, verbose=verbose)

    def rectify(
        self,
        optim: torch.optim.Optimizer,
        prior: torch.Tensor,
        training_steps: int,
        batch_size: int,
        sample: torch.Tensor | int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[list[float], torch.Tensor]:
        """Optimize the model with the reflow-procedure.
        Args:
            optim: optimizer constructed with `self.parameters()`.
            prior: [FloatLike; [D, ...]], samples from the prior distribution.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            sample: [FloatLike; [D, ...]], corresponding samples transfered from `src`,
                sample just-in-time if given is integer (assume as the number of the steps for sampling iterations).
        """
        if sample is None or isinstance(sample, int):
            with torch.inference_mode():
                sampler = ProbabilityFlowODESampler(self.model.scheduler)
                sample, _ = sampler.sample(self.model, prior, sample, verbose)

        losses = []
        self.train()
        for i in verbose(range(training_steps)):
            indices = torch.randint(0, len(sample), (batch_size,))
            loss = self.loss(sample[indices], eps=prior[indices])
            # update
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log
            losses.append(loss.detach().item())

        return losses, sample

    def consistency_distillation(
        self,
        optim: torch.optim.Optimizer,
        sample: torch.Tensor,
        training_steps: int,
        batch_size: int,
        mu: float = 0.99,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        """Optimize the model with the reflow-procedure.
        Args:
            optim: optimizer constructed with `self.parameters()`.
            sample: [FloatLike; [D, ...]], samples from the data distribution.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
        """
        sampler = ProbabilityFlowODESampler(self.model.scheduler)
        # [T]
        var = sampler._discretized_var()
        # T
        (T,) = var.shape
        # EMA purpose
        ema = copy.deepcopy(self)
        # ([FloatLike; [B, ...]], [torch.long; [B]]) -> [FloatLike, [B, ...]]
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        if self.model.scheduler.vp:
            # [T]
            beta = sampler._compute_vp_beta(1 - var)
            # variance-preserving denoiser
            denoiser = lambda x_t, t: sampler._denoise_vp(ema, x_t, t, beta)
        else:
            # variance-exploding denoiser
            denoiser = lambda x_t, t: sampler._denoise_ve(ema, x_t, t, var)

        losses = []
        self.train()
        for i in verbose(range(training_steps)):
            # [B, ...]
            x = sample[torch.randint(0, len(sample), (batch_size,))]
            # [B]
            t = torch.randint(1, T + 1, (batch_size,))
            # [B, ...]
            x_t = self.model.noise(x, t / T)
            with torch.inference_mode():
                # [B, ...]
                x_prev = denoiser(x_t, t)
                # [B, ...]
                ema_prev = ema.forward(x_prev, (t - 1) / T)
            # [], TODO: replace `self.forward(x_prev, ...)` to EMA
            loss = loss_fn(self.forward(x_t, t / T), ema_prev.clone().detach())
            # update
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log
            losses.append(loss.detach().item())
            # ema
            with torch.no_grad():
                for _ema, _self in zip(ema.parameters(), self.parameters()):
                    _ema.copy_(mu * _ema.data + (1 - mu) * _self.data)

        # assign
        with torch.no_grad():
            for _ema, _self in zip(ema.parameters(), self.parameters()):
                _ema.copy_(_self.data)

        return losses


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
