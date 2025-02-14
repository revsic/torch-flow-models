from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    """Variance scheduler,
    Denoising Diffusion Probabilistic Models, Ho et al., 2020.[arXiv:2006.11239]
    """

    T: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 0.02

    def betas(self) -> list[float]:
        # increasing linearly from `beta_1` to `beta_t`
        return [
            self.beta_1 + i / (self.T - 1) * (self.beta_T - self.beta_1)
            for i in range(self.T - 1)
        ] + [self.beta_T]


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Models, Ho et al., 2020.[arXiv:2006.11239]"""

    def __init__(self, module: nn.Module, config: Config):
        super().__init__()
        self.noise_estim = module
        self.config = config

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the noise from the given x_t; t.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [torch.long; [B]], the current timestep of the noised sample in range [0, T].
        Returns:
            estimated noise from the given sample `x_t`.
        """
        return self.noise_estim(x_t, t)

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the noise `eps`.
        Args:
            x_0: [FloatLike; [B, ...]], the given sample, `x_0`.
            t: [torch.long; [B]], the target timestep in range [0, T].
            eps: [FloatLike; [B, ...]], the sample from the prior distribution.
        Returns:
            noised sample, `x_t`.
        """
        if (t <= 0).all():
            return x_0
        # assign default value
        if eps is None:
            eps = torch.randn_like(x_0)
        # settings
        dtype, device = x_0.dtype, x_0.device
        # [T], zero-based
        beta = torch.tensor(self.config.betas(), dtype=torch.float32, device=device)
        # [T], zero-based
        alpha_bar = torch.cumprod(1 - beta, dim=0)
        # T
        (timesteps,) = alpha_bar.shape
        # [T, ...]
        alpha_bar = alpha_bar.view([timesteps] + [1] * (x_0.dim() - 1))
        # [B, ...]
        return alpha_bar[t - 1].sqrt().to(dtype) * x_0 + (
            1 - alpha_bar[t - 1]
        ).sqrt().to(dtype) * eps.to(x_0)

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoise the given sample `x_t` to the single-step backward `x_{t-1}`
            w.r.t. the current timestep `t` and the additional noise `eps`.
        Args:
            x_t: [FloatLike; [B, ...]], the given sample, `x_t`.
            t: [torch.long; [B]], the current timestep in range [0, T].
            eps: [FloatLike; [...]], the additional noise from the prior.
        Returns:
            denoised sample, `x_{t_1}`.
        """
        if (t <= 0).all():
            return x_t
        # assign default value
        if eps is None:
            eps = torch.randn_like(x_t)
        # settings
        dtype, device = x_t.dtype, x_t.device
        # [T], zero-based
        beta = torch.tensor(self.config.betas(), dtype=torch.float32, device=device)
        # T
        (timesteps,) = beta.shape
        # [T, ...]
        beta = beta.view([timesteps] + [1] * (x_t.dim() - 1))
        # [T, ...], zero-based
        alpha = 1 - beta
        # [T, ...], zero-based
        alpha_bar = torch.cumprod(alpha, dim=0)
        # [B, ...], denoised
        mean = alpha[t - 1].clamp_min(1e-7).rsqrt().to(dtype) * (
            x_t
            - beta[t - 1].to(dtype)
            * (1 - alpha_bar[t - 1].to(dtype)).clamp_min(1e-7).rsqrt()
            * self.forward(x_t, t)
        )
        # B
        (batch_size,) = t.shape
        # [B, ...]
        mask = t.view([batch_size] + [1] * (x_t.dim() - 1)) > 1
        return mean + mask * beta[t - 1].sqrt().to(dtype) * eps.to(mean)
