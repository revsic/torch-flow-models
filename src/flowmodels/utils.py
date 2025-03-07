import copy
from typing import Self

import torch
import torch.nn as nn

from flowmodels.basis import (
    ContinuousSchedulerProtocol,
    SchedulerProtocol,
    ScoreSupports,
    VelocitySupports,
)


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


class VelocityInverter(VelocitySupports):
    def __init__(self, model: VelocitySupports):
        self.model = model

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Invert the estimated velocity"""
        return -self.model.velocity(x_t, 1 - t)


def discretize_variance(
    scheduler: ContinuousSchedulerProtocol | SchedulerProtocol | None = None,
    steps: int | None = None,
    _default_steps=4,
) -> torch.Tensor:
    """Construct the sequence of discretized variances."""
    var: torch.Tensor
    match scheduler:
        case None:
            # uniform space, in range[1/T, 1]
            steps = steps or _default_steps
            return torch.arange(1, steps + 1, dtype=torch.float32) / steps
        case SchedulerProtocol():
            var = scheduler.var()
            if steps is not None:
                # for easier step-sampling
                assert scheduler.T % steps == 0
                # [steps], step-size sampling from the last variance
                # e.g. [3, 7, 11] will be sampled from [0, 1, ..., 11] with 3-steps
                var = var.view(steps, -1).T[-1]
        case ContinuousSchedulerProtocol():
            steps = steps or _default_steps
            # [T], in range[1/T, 1]
            var = scheduler.var(torch.arange(1, steps + 1, dtype=torch.float32) / steps)
    return var


def backward_process(
    model: ScoreSupports,
    x_t: torch.Tensor,
    t: torch.Tensor,
    var: torch.Tensor,
    var_prev: torch.Tensor | None = None,
    vp: bool = True,
):
    """Single step sampler.
    Args:
        model: the score-estimation model.
        x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
        t: [FloatLike; [B]], the current timesteps, `t` in range[0, 1].
        var: [FloatLike; [B]], the variance of the noise in time `t`.
        var_prev: [FloatLike; [B]], the variance of the noise in single step backward at time `t`,
            i.e. `var_{t - d}` of step size `d`.
        vp: whether the given model is formulated with variance preserving process or not.
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
    if not vp:
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
