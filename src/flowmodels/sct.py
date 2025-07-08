from dataclasses import dataclass
from typing import Callable, Iterable, Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmodels.basis import (
    ContinuousScheduler,
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
    ScoreModel,
    VelocitySupports,
)
from flowmodels.cm import MultistepConsistencySampler
from flowmodels.utils import EMASupports


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
    ForwardProcessSupports,
    PredictionSupports,
    SamplingSupports,
    VelocitySupports,
):
    """sCT: Simplifying, Stabilizing & Scailing Continuous-Time Consistency Models, Lu & Song, 2024.[arXiv:2410.11081]"""

    def __init__(
        self,
        module: nn.Module,
        scheduler: ScaledContinuousCMScheduler,
        tangent_warmup: int | None = None,
        _ada_weight_size: int = 128,
    ):
        super().__init__()
        self.F0 = module
        self.scheduler = scheduler
        self.sampler = MultistepConsistencySampler()
        self._tangent_warmup, self._steps = tangent_warmup, 0
        self._ada_weight = _AdaptiveWeights(_ada_weight_size)
        # debug purpose
        self._debug_from_loss = {}

    # debug purpose
    @property
    def _debug_purpose(self):
        return {**self._debug_from_loss, **getattr(self.F0, "_debug_purpose", {})}

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated sample from the given `x_t`.
        """
        # shortcut
        (bsize,) = t.shape
        sigma_d = self.scheduler.sigma_d
        backup = t.to(x_t.device) * np.pi * 0.5
        # [B, ...], scale t to range[0, pi/2]
        t = backup.view([bsize] + [1] * (x_t.dim() - 1))
        return t.cos() * x_t - t.sin() * sigma_d * self.F0(x_t / sigma_d, backup)

    def predict(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the sample points `x_0` from the `x_t` w.r.t. the timestep `t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given points, `x_t`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1].
        Returns:
            the predicted sample points `x_0`.
        """
        return self.forward(x_t, t)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        sigma_d = self.scheduler.sigma_d
        return sigma_d * self.F0(x_t / sigma_d, t * np.pi * 0.5)

    def score(self, x_t: torch.Tensor, t: torch.Tensor):
        """Estimate the stein score from the given sample `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], sample from the trajectory at time `t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated score.
        """
        (bsize,) = t.shape
        # [B, ...]
        x_0 = self.forward(x_t, t)
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
        ema: Self | EMASupports[Self] | None = None,
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
        # ENGINEERING: Out-of-inference mode multiplication for torch-dynamo inductor support
        _tangents = (_t.cos() * _t.sin() * v_t, t.cos() * t.sin() * sigma_d)
        # [B, ...], [B, ...], jvp = sigma_d * t.cos() * t.sin() * dF/dt
        with torch.no_grad():
            jvp_fn = torch.compiler.disable(torch.func.jvp, recursive=False)
            F, jvp, *_ = jvp_fn(
                EMASupports[Self].reduce(self, ema).F0.forward,
                (x_t / sigma_d, t),
                _tangents,
            )
        # warmup scaler
        r = 1.0
        if self._tangent_warmup:
            r = min(self._steps / self._tangent_warmup, 1.0)
            self._steps += 1
        # df/dt = -t.cos() * (sigma_d * F(x_t/sigma_d, t) - dx_t/dt) - t.sin() * (x_t + sigma_d * dF/dt)
        cos_mul_grad = -_t.cos().square() * (sigma_d * F - v_t) - r * (
            _t.sin() * _t.cos() * x_t + jvp
        )
        # [B, ...]
        estim: torch.Tensor = self.F0.forward(x_t / sigma_d, t)
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
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        sigma_d = self.scheduler.sigma_d
        return self.sampler.sample(self, prior * sigma_d, steps, verbose)

    def noise(
        self,
        x_0: torch.Tensor,
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
        if prior is None:
            prior = torch.randn_like(x_0)
        # [B, ...], scale in range [0, pi/2]
        t = (t * np.pi * 0.5).view([bsize] + [1] * (x_0.dim() - 1))
        # [B, ...]
        return t.cos().to(x_0) * x_0 + t.sin().to(x_0) * prior * self.scheduler.sigma_d


class TrigFlow(ScaledContinuousCM):

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        ema: Self | EMASupports[Self] | None = None,
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
        estim = sigma_d * self.F0.forward(x_t / sigma_d, t)
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

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        sigma_min: float = 0.02,
        sigma_max: float = np.pi * 0.5,
        rho: float = 7,
        dtype: torch.dtype = torch.float64,
        correction: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """EDM Sampler, reference git+NVlabs/edm/generate.py."""
        batch_size, *_ = prior.shape
        # assign default values
        steps = steps or 18
        sigma_d = self.scheduler.sigma_d
        if verbose is None:
            verbose = lambda x: x
        # shortcut
        if steps <= 1:
            x = self.predict(prior, torch.full((batch_size,), np.pi * 0.5))
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
            d_cur = sigma_d * self.F0.forward(x_hat.to(p) / sigma_d, _t.to(p)).to(dtype)
            x = torch.cos(t_hat - t_next) * x_hat - torch.sin(t_hat - t_next) * d_cur
            # 2nd-order midpoint correction
            if i < steps - 1 and correction:
                # [B]
                _t = t_next.repeat(batch_size)
                # [B, ...], rescale t-steps into range[0, 1]
                d_prime = sigma_d * self.F0.forward(x.to(p) / sigma_d, _t.to(p)).to(
                    dtype
                )
                x = torch.cos(t_hat - t_next) * x_hat - torch.sin(t_hat - t_next) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

            xs.append(x)

        return x, xs
