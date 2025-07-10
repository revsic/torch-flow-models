from typing import Callable, Iterable

import torch
import torch.nn as nn

from flowmodels.basis import (
    ODEModel,
    PredictionSupports,
    SamplingSupports,
    VelocitySupports,
)


class AlignYourFlow(nn.Module, ODEModel, PredictionSupports, SamplingSupports):
    """Align Your Flow: Scaling Continuous-Time Flow Map Distillation, Sabour et al., 2025.[arXiv:2506.14603]"""

    def __init__(
        self,
        module: nn.Module,
        p_mean: float = -0.8,
        p_std: float = 1.0,
        tangent_warmup: int | None = None,
        c: float = 0.1,
        max_r: float = 0.99,
        teacher: VelocitySupports | None = None,
    ):
        super().__init__()
        self.F0 = module
        self.p_mean = p_mean
        self.p_std = p_std
        self.tangent_warmup, self._steps = tangent_warmup, 0
        self.c = c
        self.max_r = max_r
        self.teacher = teacher
        # debug purpose
        self._debug_from_loss = {}

    # debug purpose
    @property
    def _debug_purpose(self):
        return {**self._debug_from_loss, **getattr(self.F0, "_debug_purpose", {})}

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the mean velocity from the given `t` to `r`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
            r: [FloatLike; [B]], the terminal timestep in range[0, 1]; r < t.
        Returns:
            estimated mean velocity from the given sample `x_t`.
        """
        return self.F0(x_t, t, t - r)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t` (assume the terminal is zero).
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        # targeting the origin point
        return self.forward(x_t, t, torch.zeros_like(t))

    def predict(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the sample points `x_0` from the `x_t` w.r.t. the timestep `t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            the predicted sample points `x_0`.
        """
        (bsize,) = t.shape
        return x_t - t.view([bsize] + [1] * (x_t.dim() - 1)) * self.velocity(x_t, t)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `x_0`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the source distribution, `x_1`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        device = sample.device
        # sample
        if t is None:
            t = torch.sigmoid(
                torch.randn(batch_size, device=device) * self.p_std + self.p_mean
            )
        if prior is None:
            prior = torch.randn_like(sample)
        if s is None:
            s = torch.sigmoid(
                torch.randn(batch_size, device=device) * self.p_std + self.p_mean
            )
            t, s = torch.maximum(t, s), torch.minimum(t, s)
        # [B, ...]
        _t = t.view([batch_size] + [1] * (prior.dim() - 1))
        _s = s.view([batch_size] + [1] * (prior.dim() - 1))
        # [B, ...]
        x_t = (1 - _t) * sample + _t * prior
        if self.teacher is None:
            v_t = prior - sample
        else:
            with torch.no_grad():
                v_t = self.teacher.velocity(x_t, t)
        # [B, ...], [B, ...]
        jvp_fn = torch.compiler.disable(
            torch.func.jvp, recursive=False  # pyright: ignore
        )
        F, jvp = jvp_fn(
            self.forward,
            (x_t, t, s),  # pyright: ignore
            (v_t, torch.ones_like(t), torch.zeros_like(s)),
        )
        # [B, ...]
        f = x_t - (_t - _s) * F
        # warmup scaler
        r = self.max_r
        if self.tangent_warmup:
            r = min(self._steps / self.tangent_warmup, self.max_r)
            self._steps += 1
        # [B, ...]
        dfdt = (v_t - F) + r * (_s - _t) * jvp
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # normalized tangent
        normalized_tangent = dfdt / (
            _norm := dfdt.norm(p=2, dim=rdim, keepdim=True) + self.c
        )
        # []
        loss = (f - f.detach() + normalized_tangent.detach()).square().mean()
        with torch.no_grad():
            self._debug_from_loss = {
                "sct/mse": loss.item(),
                "sct/tangent-norm": _norm.mean().item(),
            }
        # []
        return loss

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = 1,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Transfer the samples from the prior distribution to the trained distribution, using vanilla Euler method.
        Args:
            prior: [FloatLike; [B, ...]], samples from the source distribution, `X_0`.
            steps: the number of the steps.
        """
        steps = steps or self.DEFAULT_STEPS  # pyright: ignore
        assert isinstance(steps, int)
        if verbose is None:
            verbose = lambda x: x
        # loop
        x_t, x_ts = prior, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(steps, 0, -1)):
                t = torch.full((bsize,), i / steps, dtype=torch.float32)
                velocity = self.forward(x_t, t, t - 1 / steps)
                x_t = x_t - velocity / steps
                x_ts.append(x_t)

        return x_t, x_ts
