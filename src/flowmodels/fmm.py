from typing import Callable, Iterable, Literal

import torch
import torch.nn as nn


from flowmodels.basis import (
    ODEModel,
    PredictionSupports,
    SamplingSupports,
    VelocitySupports,
)


class FlowMapMatching(nn.Module, ODEModel, PredictionSupports, SamplingSupports):
    """
    Flow map matching with stochastic interpolants: A mathematical framework for consistency models, Boffi et al., 2024.[arXiv:2406.07507]
    """

    def __init__(
        self,
        module: nn.Module,
        method: Literal["EMD", "LMD", "FMM"] = "FMM",
        teacher: VelocitySupports | None = None,
    ):
        super().__init__()
        self.F0 = module
        self.method = method
        self.teacher = teacher

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

    def predict(
        self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the sample points `x_s` from the `x_t` w.r.t. the timestep `t`,
            assuming `s` to zero if not provided.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the current timesteps in range[0, 1].
            s: [FloatLike; [B]], the terminal timesteps in range[0, 1].
        Returns:
            the predicted sample points `x_s`.
        """
        (bsize,) = t.shape
        if s is None:
            s = torch.zeros_like(t)
        bt = t.view([bsize] + [1] * (x_t.dim() - 1))
        bs = s.view([bsize] + [1] * (x_t.dim() - 1))
        return x_t - (bt - bs) * self.forward(x_t, t, s)

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
            t = torch.rand(batch_size, device=device)
        if s is None:
            s = torch.rand(batch_size, device=device)
            t, s = torch.maximum(t, s), torch.minimum(t, s)
        if prior is None:
            prior = torch.randn_like(sample)
        # [B, ...]
        _t = t.view([batch_size] + [1] * (prior.dim() - 1))
        # [B, ...]
        x_t = (1 - _t) * sample + _t * prior
        # for torch-dynamo inductor supports
        jvp_fn = torch.compiler.disable(
            torch.func.jvp, recursive=False  # pyright: ignore
        )
        loss: torch.Tensor
        match self.method:
            case "LMD":
                assert self.teacher is not None
                estim, jvp = jvp_fn(
                    lambda s: self.predict(x_t, t, s),
                    (s,),  # pyright: ignore
                    (torch.ones_like(s),),
                )
                loss = (jvp - self.teacher.velocity(estim, s)).square().mean()
            case "EMD":
                assert self.teacher is not None
                _, jvp = jvp_fn(
                    self.predict,
                    (x_t, t, s),  # pyright: ignore
                    (
                        self.teacher.velocity(x_t, t),
                        torch.ones_like(t),
                        torch.zeros_like(s),
                    ),
                )
                loss = jvp.square().mean()
            case "FMM":
                v_t = prior - sample
                x_s = self.predict(x_t, t, s)
                estim, jvp = jvp_fn(
                    lambda t: self.predict(x_s, s, t),
                    (t,),  # pyright: ignore
                    (torch.ones_like(t),),
                )
                loss = (jvp - v_t).square().mean() + (estim - x_t).square().mean()
            case _:
                assert False
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
