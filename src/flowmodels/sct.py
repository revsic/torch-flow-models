from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import (
    ContinuousScheduler,
    PredictionSupports,
    SamplingSupports,
    ScoreModel,
    VelocitySupports,
)


@dataclass
class ScaledContinuousCMScheduler(ContinuousScheduler):
    vp: bool = True

    sigma_d: float = 0.5  # standard deviation of the prior distribution
    p_mean: float = -1.0  # default values from Appendix G.2. CIFAR-10
    p_std: float = 1.4

    def var(self, t: torch.Tensor) -> torch.Tensor:
        return (t * np.pi * 0.5).sin().square()


class _AdaptiveWeights(nn.Linear):
    def __init__(
        self, channels: int = 128, max_period: int = 10000, scale: float = 1.0
    ):
        assert channels % 2 == 0
        super().__init__(channels, 1)
        denom = -np.log(max_period) / (channels // 2 - 1)
        # [C // 2]
        freqs = torch.exp(denom * torch.arange(0, channels // 2, dtype=torch.float32))
        self.register_buffer("freqs", freqs, persistent=False)
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.freqs, torch.Tensor)
        # [T, E // 2]
        args = input[:, None].float() * self.freqs * self.scale
        # [T, E] > [T, 1] > [T]
        weight = super().forward(torch.cat([args.cos(), args.sin()], dim=-1))
        return weight.to(input.dtype).squeeze(dim=-1)


class ScaledContinuousCM(
    nn.Module,
    ScoreModel,
    PredictionSupports,
    SamplingSupports,
    VelocitySupports,
):
    """sCT: Simplifying, Stabilizing & Scailing Continuous-Time Consistency Models, Lu & Song, 2024.[arXiv:2410.11081]"""

    DEFAULT_STEPS = 2

    def __init__(
        self,
        module: nn.Module,
        scheduler: ScaledContinuousCMScheduler,
        tangent_warmup: int | None = None,
        _warmup_max: float = 1.0,
        _ada_weight_size: int = 128,
        _approx_jvp: bool = True,
        _dt: float = 0.005,
    ):
        super().__init__()
        self.F0 = module
        self.scheduler = scheduler
        self._tangent_warmup = tangent_warmup
        self._warmup_max = _warmup_max
        self.register_buffer("_steps", torch.tensor(0, requires_grad=False))
        self._ada_weight = _AdaptiveWeights(_ada_weight_size)
        # debug purpose
        self._debug_from_loss = {}
        self._approx_jvp = _approx_jvp
        self._dt = _dt

    # debug purpose
    @property
    def _debug_purpose(self):
        return {**self._debug_from_loss, **getattr(self.F0, "_debug_purpose", {})}

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, pi/2].
        Returns:
            [FloatLike; [B, ...]], estimated sample from the given `x_t`.
        """
        # shortcut
        (bsize,) = t.shape
        sigma_d = self.scheduler.sigma_d
        # [B, ...], scale t to range[0, pi/2]
        bt = t.view([bsize] + [1] * (x_t.dim() - 1))
        # condition
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return bt.cos() * x_t - bt.sin() * sigma_d * self.F0(x_t / sigma_d, t, **kwargs)

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
        return self.forward(x_t, t * np.pi * 0.5, label=label)

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        sigma_d = self.scheduler.sigma_d
        return sigma_d * self.F0(x_t / sigma_d, t * np.pi * 0.5, **kwargs)

    def score(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ):
        """Estimate the stein score from the given sample `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated score.
        """
        (bsize,) = t.shape
        # [B, ...]
        x_0 = self.predict(x_t, t, label=label)
        # [B, ...]
        var = self.scheduler.var(t).view([bsize] + [1] * (x_0.dim() - 1))
        # simplified:
        # t = t * np.pi * 0.5 >>= (t.cos() * x_0 - x_t) / t.sin().square()
        return ((1 - var).sqrt().to(x_0) * x_0 - x_t) / var.clamp_min(1e-5).to(x_0)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_0`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from the proposal distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the prior distribution, `X_1`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        # shortcut
        batch_size, *_ = sample.shape
        sigma_d = self.scheduler.sigma_d
        device = sample.device
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            # shortcut
            p_std, p_mean = self.scheduler.p_std, self.scheduler.p_mean
            # sample from log-normal
            rw_t = (torch.randn(batch_size, device=device) * p_std + p_mean).exp()
            # [T], in range[0, pi/2]
            t = (rw_t / sigma_d).atan()
        else:
            # scale into range[0, pi/2]
            t = t.to(device) * np.pi * 0.5
            # compute a reciprocal of the prior weighting term from the given t
            rw_t = t.tan() * sigma_d
        # [B, ...]
        _t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = _t.cos() * sample + _t.sin() * sigma_d * prior
        # [B, ...]
        v_t = _t.cos() * sigma_d * prior - _t.sin() * sample
        # [B, ...], [B, ...], jvp = sigma_d * t.cos() * t.sin() * dF/dt
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        if self._approx_jvp:
            # shortcut
            dt = self._dt
            fwd = lambda t, bt: self.F0.forward(
                bt.cos() * sample / sigma_d + bt.sin() * prior, t, **kwargs
            )
            # approximation w/o JVP
            dFdt = (fwd(t + dt, _t + dt) - fwd(t - dt, _t - dt)) / (2 * dt)
            jvp = dFdt * _t.cos() * _t.sin() * sigma_d
            estim = self.F0.forward(x_t / sigma_d, t, **kwargs)
        else:
            jvp_fn = torch.compiler.disable(
                torch.func.jvp, recursive=False  # pyright: ignore
            )
            estim, jvp, *_ = jvp_fn(
                lambda x, t: self.F0.forward(x, t, **kwargs),
                (x_t / sigma_d, t),  # pyright: ignore
                (_t.cos() * _t.sin() * v_t, t.cos() * t.sin() * sigma_d),
            )
        # warmup scaler
        r = 1.0
        if self._tangent_warmup:
            self._steps.add_(1)
            r = (self._steps / self._tangent_warmup).clamp_max(self._warmup_max)
        # stop grad
        F = estim.detach()
        # df/dt = -t.cos() * (sigma_d * F(x_t/sigma_d, t) - dx_t/dt) - t.sin() * (x_t + sigma_d * dF/dt)
        cos_mul_grad = -_t.cos().square() * (sigma_d * F - v_t) - r * (
            _t.sin() * _t.cos() * x_t + jvp.detach()
        )
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # normalized tangent
        normalized_tangent = cos_mul_grad / (
            _norm := cos_mul_grad.norm(p=2, dim=rdim, keepdim=True) + 0.1
        )
        # [B]
        mse = (estim - F - normalized_tangent).square().mean(dim=rdim)
        # [B], adaptive weighting
        logvar = self._ada_weight.forward(t)
        # [B], different with
        loss = mse * logvar.exp() - logvar
        with torch.no_grad():
            self._debug_from_loss = {
                "sct/mse": mse.mean().item(),
                "sct/logvar": logvar.mean().item(),
                "sct/tangent-norm": _norm.mean().item(),
            }
        # []
        return loss.mean()

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | list[float] | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        sigma_max: float = 80.0,
        _prescale_prior: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Customized MultistepConsistencySampler."""
        # shortcuts
        sigma_d = self.scheduler.sigma_d
        steps = steps or self.DEFAULT_STEPS
        dtype, device = prior.dtype, prior.device
        # e.g., default atan(80 / 0.5) = 1.5645
        t_max = np.arctan(sigma_max / sigma_d).item()
        # prescaling
        if _prescale_prior:
            prior = prior * sigma_d
        # single-step sampler
        batch_size, *_ = prior.shape
        if isinstance(steps, int) and steps <= 1:
            t = torch.full((batch_size,), t_max, dtype=dtype, device=device)
            x_0 = self.forward(prior, t, label=label)
            return x_0, [x_0]
        # proposed steps
        if steps == 2:
            steps = [t_max, 1.1]
        elif isinstance(steps, int):
            # uniform timesteps for otherwise
            steps = np.linspace(t_max, 0.0, steps + 1)[:-1].tolist()
        assert isinstance(steps, list)
        # multi-step sampler
        x_t, x_ts = prior, []
        rdim = [batch_size] + [1 for _ in range(x_t.dim() - 1)]
        if verbose is None:
            verbose = lambda x: x
        for i in verbose(range(len(steps))):
            # [B]
            t = torch.full((batch_size,), steps[i], dtype=dtype, device=device)
            # [B, ...]
            x_0 = self.forward(x_t, t, label=label)
            if i < len(steps) - 1:
                t_next = torch.full(rdim, steps[i + 1], dtype=dtype, device=device)
                x_t = t_next.cos() * x_0 + t_next.sin() * prior
            else:
                # last inference
                x_t = x_0
            x_ts.append(x_t)
        return x_t, x_ts


class TrigFlow(ScaledContinuousCM):

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # shortcut
        batch_size, *_ = sample.shape
        sigma_d = self.scheduler.sigma_d
        device = sample.device
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            # shortcut
            p_std, p_mean = self.scheduler.p_std, self.scheduler.p_mean
            # sample from log-normal
            rw_t = (torch.randn(batch_size, device=device) * p_std + p_mean).exp()
            # [T], in range[0, pi/2]
            t = (rw_t / sigma_d).atan()
        else:
            # scale into range[0, pi/2]
            t = t * np.pi * 0.5
        # [B, ...]
        _t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = _t.cos() * sample + _t.sin() * sigma_d * prior
        # [B, ...]
        v_t = _t.cos() * sigma_d * prior - _t.sin() * sample
        # [B, ...]
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        estim = sigma_d * self.F0.forward(x_t / sigma_d, t, **kwargs)
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # [B]
        mse = (estim - v_t).square().mean(dim=rdim)
        # [B], adaptive weighting
        logvar = self._ada_weight.forward(t)
        # [B], different with
        loss = mse * logvar.exp() - logvar
        with torch.no_grad():
            self._debug_from_loss = {
                "sct/mse": mse.mean().item(),
                "sct/logvar": logvar.mean().item(),
            }
        return loss.mean()

    def sample(  # pyright: ignore
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        sigma_min: float = 0.02,
        sigma_max: float = np.pi * 0.5,
        rho: float = 7,
        dtype: torch.dtype = torch.float64,
        correction: bool = True,
        cfg_scale: float | None = None,
        uncond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """EDM Sampler, reference git+NVlabs/edm/generate.py."""
        batch_size, *_ = prior.shape
        # assign default values
        steps = steps or 18
        sigma_d = self.scheduler.sigma_d
        if verbose is None:
            verbose = lambda x: x
        # sanity check
        if cfg_scale is not None:
            assert label is not None and uncond is not None
            rdim = [1 for _ in range(uncond.dim())]
            uncond = uncond[None].repeat([batch_size] + rdim)
        # shortcut
        if steps <= 1:
            t = torch.full((batch_size,), np.pi * 0.5)
            x = self.predict(prior, t, label=label)
            # since the prediction is linear combination of x_t and F0
            # , sample level CFG equals to the velocity level CFG
            if cfg_scale is not None:
                u = self.predict(prior, t, label=uncond)
                x = u + cfg_scale * (x - u)
            return x, [x]
        # sample parameter for moving device
        p = next(self.parameters())
        # [T], pi/2(=sigma_max) to 0.02(=sigma_min)
        t_steps = (
            sigma_max ** (1 / rho)
            + torch.arange(steps, dtype=dtype, device=prior.device)
            / (steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        # [T + 1]
        t_steps = F.pad(t_steps, [0, 1], "constant", 0.0)
        # condition supports
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        # Main sampling loop.
        x, xs = prior.to(dtype) * sigma_d, []
        for i in verbose(range(steps)):
            # [], []
            t_cur, t_next = t_steps[i : i + 2]
            # [B, ...], []
            x_hat, t_hat = x, t_cur
            # [B]
            _t = t_hat.repeat(batch_size)
            # [B, ...], rescale t-steps into range[0, 1]
            d_cur = sigma_d * self.F0.forward(
                x_hat.to(p) / sigma_d, _t.to(p), **kwargs
            ).to(dtype)
            if cfg_scale is not None:
                u = sigma_d * self.F0.forward(
                    x_hat.to(p) / sigma_d, _t.to(p), label=uncond
                ).to(dtype)
                d_cur = u + cfg_scale * (d_cur - u)
            x = torch.cos(t_hat - t_next) * x_hat - torch.sin(t_hat - t_next) * d_cur
            # 2nd-order midpoint correction
            if i < steps - 1 and correction:
                # [B]
                _t = t_next.repeat(batch_size)
                # [B, ...], rescale t-steps into range[0, 1]
                d_prime = sigma_d * self.F0.forward(
                    x.to(p) / sigma_d, _t.to(p), **kwargs
                ).to(dtype)
                if cfg_scale is not None:
                    u = sigma_d * self.F0.forward(
                        x.to(p) / sigma_d, _t.to(p), label=uncond
                    ).to(dtype)
                    d_prime = u + cfg_scale * (d_prime - u)
                x = torch.cos(t_hat - t_next) * x_hat - torch.sin(t_hat - t_next) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

            xs.append(x)

        return x, xs
