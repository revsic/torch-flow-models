from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import ODEModel, PredictionSupports, SamplingSupports


class EBFMM(nn.Module, ODEModel, PredictionSupports, SamplingSupports):
    def __init__(
        self,
        module: nn.Module,
        _hinge: bool = True,
        _lambda: float = 0.001,
        _entropy_reg: float | None = None,
    ):
        super().__init__()
        self.F0 = module
        self._hinge = _hinge
        self._lambda = _lambda
        self._entropy_reg = _entropy_reg

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
            assuming s to zero if it is not provided.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
            s: [FloatLike; [B]], the target timesteps in range[0, 1], terminal point.
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
        src: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `x_0`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
            src: [FloatLike; [B, ...]], sample from the source distribution, `x_1`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        batch_size, *_ = sample.shape
        device = sample.device
        # sample
        # k > t > s
        if k is None:
            k = torch.rand(batch_size, device=device)
        if t is None:
            t = torch.rand(batch_size, device=device) * k
        if s is None:
            s = torch.rand(batch_size, device=device) * t
        if src is None:
            src = torch.randn_like(sample)
        # [B, ...]
        _k = k.view([batch_size] + [1] * (src.dim() - 1))
        _t = t.view([batch_size] + [1] * (src.dim() - 1))
        # [B, ...]
        x_k = (1 - _k) * sample + _k * src
        x_t = (1 - _t) * sample + _t * src
        # prior assumption
        jvp_fn = torch.compiler.disable(
            torch.func.jvp, recursive=False  # pyright: ignore
        )
        u_t, jvp = jvp_fn(
            self.forward,
            (x_k, k, t),  # pyright: ignore
            (src - sample, torch.ones_like(k), torch.zeros_like(t)),
        )
        y_t = x_k - (_k - _t) * u_t
        # [B, ...]
        dfdt = (src - sample - u_t) + (_t - _k) * jvp
        # tangent normalization
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        dfdt = F.normalize(dfdt, p=2, dim=rdim)  # pyright: ignore
        # [B], prior#1: u is the integration of v
        prior_loss = (y_t - y_t.detach() + dfdt.detach()).square().mean(dim=rdim)
        # [B, ...]
        y_k = y_t - (_t - _k) * self.forward(y_t, t, k)
        # [B], prior#2: f is invertible
        prior_loss = prior_loss + (y_k - x_k).square().mean(dim=rdim)
        # [B], discriminator loss
        disc_loss = -((x_t - y_t).detach() * u_t).mean(dim=rdim)
        if self._hinge:
            disc_loss = F.softplus(disc_loss)
        # [B], R1, R2 Regularization
        R1 = self.forward(x_t, t, s).square().mean(dim=rdim)
        R2 = self.forward(y_t.detach(), t, s).square().mean(dim=rdim)
        # [B], generator loss
        gen_loss = -((y_t - x_t) * u_t.detach()).mean(dim=rdim)
        if self._entropy_reg:
            # [B], entropy regularizer w/hutchinson estimation
            y_k, logdet = jvp_fn(
                self.forward,
                (y_t, t, k),  # pyright: ignore
                (torch.randn_like(y_t), torch.zeros_like(t), torch.zeros_like(k)),
            )
            entropy = (y_k - (1 - _k) * sample).square().mean(dim=rdim) / (
                2 * k.square()
            ) - logdet
            gen_loss = gen_loss - self._entropy_reg * entropy
        if self._hinge:
            gen_loss = F.softplus(gen_loss)
        # aggregation
        loss = prior_loss + gen_loss + disc_loss + (R1 + R2) * self._lambda
        return loss.mean()

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
