from typing import Callable, Iterable, Literal
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import (
    ForwardProcessSupports,
    ODEModel,
    PredictionSupports,
    SamplingSupports,
)
from flowmodels.cm import MultistepConsistencySampler


@dataclass
class EDMCoeff:
    in_: torch.Tensor
    skip: torch.Tensor
    out: torch.Tensor
    noise: torch.Tensor


@dataclass
class IMMScheulder:

    trajectory: Literal["cosine", "ot-fm"] = "ot-fm"
    network: Literal["identity", "simple-edm", "euler-fm"] = "simple-edm"

    sigma_d: float = 0.5
    c: float = 1000.0  # a coefficient for `c_noise`
    T: float = 0.994  # boundary of time distributions
    eps: float = 0.001

    eta_max: float = 160.0  # hyperparameters of mapping function
    eta_min: float = 0.0
    k: int = 15

    M: int = 4  # split a batch into B/M groups, each group shares a (s, r, t)

    a: int = 1  # hyperparameters for weighting functions
    b: int = 5

    def coeff_interp(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolation coefficients.
        Args:
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            alpah_t: [FloatLike; [B]], coefficients of samples from data distribution.
            sigma_t: [FloatLike; [B]], coefficients of samples from noise distribution.
        """
        match self.trajectory:
            case "ot-fm":
                return (1 - t), t
            case "cosine":
                return (0.5 * np.pi * t).cos(), (0.5 * np.pi * t).sin()

    def coeff_edm(self, t: torch.Tensor) -> EDMCoeff:
        """EDM coefficients.
        Args:
            t: [FloatLike; [B]], the given timesteps, in range[0, 1].
        Returns:
            coefficients for the EDM formulation, `cin`, `cskip`, `cout` and `cnoise`.
                s.t. f(x, t) = cskip(t) * x + cout(t) * F(cin(t) * x, cnoise(t)).
        """
        # [B], [B]
        a, s = self.coeff_interp(t)
        # [B]
        var = a.square() + s.square()
        # [B], [B]
        c_in, c_noise = var.rsqrt() / self.sigma_d, self.c * t
        match self.network:
            case "identity":
                return EDMCoeff(
                    in_=c_in,
                    skip=torch.zeros_like(t),
                    out=torch.ones_like(t),
                    noise=c_noise,
                )
            case "simple-edm":
                return EDMCoeff(
                    in_=c_in,
                    skip=a / var,
                    out=-self.sigma_d * s * var.rsqrt(),
                    noise=c_noise,
                )
            case "euler-fm":
                return EDMCoeff(
                    in_=c_in, skip=torch.ones_like(t), out=-t * s, noise=c_noise
                )

    def r(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mapping function, from s, t to r.
        Args:
            s: [FloatLike; [B]], intermediate timesteps, in range[eps, t].
            t: [FloatLike; [B]], targetted timesteps, in range[eps, T].
        Returns:
            mapping value `r`.
        """
        a_t, s_t = self.coeff_interp(t)
        eta_t = s_t / a_t
        return torch.maximum(
            s, self._eta_inv(eta_t - (self.eta_max - self.eta_min) / (2**self.k))
        )

    def w(self, t: torch.Tensor) -> torch.Tensor:
        """Weighting function.
        Args:
            t: [FloatLike; [B]], the current timesteps.
        Returns:
            weighting value `w`.
        """
        a_t, s_t = self.coeff_interp(t)
        logsnr, dlogsnr = self._logsnr(t)
        var = a_t.square() * s_t.square()
        return -0.5 * (self.b - logsnr).sigmoid() * dlogsnr * a_t**self.a / var

    def _eta_inv(self, eta: torch.Tensor) -> torch.Tensor:
        match self.trajectory:
            case "cosine":
                # (sin(pi/2 t)/cos(pi/2 t))^{-1} = tan(pi/2 t)^-1
                # => atan(y) = pi/2 t
                # => t = 2/pi atan(y)
                return eta.atan() * 2 / np.pi
            case "ot-fm":
                # (t/(1-t))^{-1} => (1 - t)y = t
                # => y = (1 + y)t
                # => t = y / (1 + y)
                return eta / (1 + eta)

    def _logsnr(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_t, s_t = self.coeff_interp(t)
        logsnr = 2 * (a_t / s_t).log()
        match self.trajectory:
            case "cosine":
                # d/dt 2log(cos(pi/2 t)/sin(pi/2 t))
                # = 2tan(pi/2 t) x d/dt cot(pi/2 t)
                # = -pi tan(pi/2 t)csc^2(pi/2 t) = -pi sin(t')/cos(t') 1/sin^2(t')
                # = -pi/(sin(t')cos(t'))
                _t = t * np.pi * 0.5
                return logsnr, -np.pi / (_t.sin() * _t.cos())
            case "ot-fm":
                # d/dt 2log((1-t)/t)
                # = 2t/(1-t) x d/dt (1-t)/t
                # = 2t/(1-t) x (-t-(1-t))/t^2
                # = -2/(t(1 - t))
                return logsnr, -2 / (t * (1 - t))


class InductivMomentMatching(
    nn.Module,
    ODEModel,
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
):
    """Inductive Moment Matching, Zhou et al., 2025.[arXiv:2503.07565]"""

    def __init__(self, module: nn.Module, scheduler: IMMScheulder):
        super().__init__()
        self.F0 = module
        self.scheduler = scheduler
        self.sampler = MultistepConsistencySampler()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate the `x_s` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep ins range[0, 1].
            s: [FloatLike; [B]], the target timesteps in range[0, t], assume as all zeros if not provided.
        Returns:
            [FloatLike; [B, ...]], estimated sample `x_s` from the given `x_t`.
        """
        # shortcut
        (bsize,) = t.shape
        rdim = [bsize] + [1] * (x_t.dim() - 1)
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        # [B, ...]
        c = self.scheduler.coeff_edm(t)
        # [B, ...]
        x_0 = c.skip.view(rdim) * x_t + c.out.view(rdim) * self.F0.forward(
            c.in_.view(rdim) * x_t, c.noise, **kwargs
        )
        if s is None:
            return x_0
        return self._ddim(x_t, x_0, s, t)

    def predict(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the sample points `x_0` from the `x_t` w.r.t. the timestep `t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            the predicted sample points `x_0`.
        """
        return self.forward(x_t, t, label=label)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B x 2, ...]], training data.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from the proposal distribution if not provided.
            prior: [FloatLike; [B x 2, ...]], sample from the source distribution,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        # shortcut
        device = sample.device
        batch_size, *_ = sample.shape
        _min, _max = self.scheduler.eps, self.scheduler.T
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            # uniform sampling
            t = torch.rand(batch_size // 2, device=device).clamp(_min, _max)
        # uniform sampling
        s = torch.rand(batch_size // 2, device=device).clamp_min(_min) * t
        # [B]
        r = self.scheduler.r(s, t)
        # [B, ...], [B, ...]
        x_t, xp_t = self.noise(sample, t.repeat(2), prior).chunk(2)
        x_r, xp_r = self.noise(sample, r.repeat(2), prior).chunk(2)
        if label is not None:
            l, lp = label.chunk(2)
        else:
            l, lp = None, None
        y_st = self.forward(x_t, t, s, label=l)
        yp_st = self.forward(xp_t, t, s, label=lp)
        with torch.no_grad():
            y_sr = self.forward(x_r, r, s, label=l)
            yp_sr = self.forward(xp_r, r, s, label=lp)
        # [B]
        losses = self.scheduler.w(t) * (
            self._laplace_kernel(y_st, yp_st, t)
            + self._laplace_kernel(y_sr, yp_sr, t)
            - self._laplace_kernel(y_st, yp_sr, t)
            - self._laplace_kernel(yp_st, y_sr, t)
        )
        return losses.mean()

    def noise(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the `prior`.
        Args:
            x_0: [FloatLike; [B, ...]], the given samples, `x_0`.
            t: [torch.long; [B]], the target timesteps in range[0, 1].
            prior: [FloatLike; [B, ...]], the samples from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        (bsize,) = t.shape
        rdim = [bsize] + [1] * (sample.dim() - 1)
        a_t, s_t = self.scheduler.coeff_interp(t)
        if prior is None:
            prior = torch.randn_like(sample)
        return a_t.view(rdim) * sample + s_t.view(rdim) * prior

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        return self.sampler.sample(self, prior, label, steps, verbose)

    def _ddim(
        self, x_t: torch.Tensor, x_0: torch.Tensor, s: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        (bsize,) = t.shape
        rdim = [bsize] + [1] * (x_t.dim() - 1)
        a_s, s_s = self.scheduler.coeff_interp(s)
        a_t, s_t = self.scheduler.coeff_interp(t)
        return (a_s - s_s / s_t * a_t).view(rdim) * x_0 + (s_s / s_t).view(rdim) * x_t

    def _laplace_kernel(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        rdim = [i + 1 for i in range(x.dim() - 1)]
        c = self.scheduler.coeff_edm(t)
        w_tilde = 1 / c.out.abs()
        return torch.exp(-w_tilde * (x - y).square().mean(dim=rdim))
