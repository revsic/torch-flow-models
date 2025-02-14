from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ddpm import DDPM, Config as DDPMConfig


@dataclass
class Config(DDPMConfig):
    """Variance scheduler,
    Improved Denoising Diffusion Probabilistic Models, Nichol & Dhariwal, 2021.[arXiv:2102.09672]
    """

    s: float = 0.008
    T: int = 40
    eta: float = 0.0
    # unused
    beta_1: float | None = None
    beta_T: float | None = None

    def alphas(self) -> list[float]:
        T, s = self.T, self.s
        # [T]
        timesteps = torch.arange(T, dtype=torch.float32)
        # [T]
        schedule = ((timesteps / T + s) / (1 + s) * np.pi / 2).cos().square()
        # [T]
        return (schedule / schedule[0]).tolist()

    def betas(self) -> list[float]:
        # [T]
        alpha_bar = torch.tensor(self.alphas(), dtype=torch.float32)
        # [T]
        beta = 1 - alpha_bar / F.pad(alpha_bar[:-1], [1, 0], mode="constant", value=1.0)
        return beta.tolist()

    def sigmas(self) -> list[float]:
        # [T + 1]
        alphas = F.pad(
            torch.tensor(self.alphas(), dtype=torch.float32),
            [1, 0],
            mode="constant",
            value=1.0,
        )
        # [T]
        sigmas = (
            self.eta
            # \sqrt{(1 - \alpha_{t-1}) / (1 - \alpha_t)}
            * ((1 - alphas[:-1]) / (1 - alphas[1:]).clamp_min(1e-7)).sqrt()
            # \sqrt{1 - \alpha_t / \alpha_{t-1}}
            * (1 - alphas[1:] / alphas[:-1].clamp_min(1e-7)).sqrt()
        )
        return sigmas.tolist()


class DDIM(DDPM):
    """Denoising Diffusion Implicit Models, Song et al., 2020.[arXiv:2010.02502]"""

    def __init__(self, module: nn.Module, config: Config):
        super().__init__(module, config)

    def noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noise the given sample `x_0` to the `x_t` w.r.t. the timestep `t` and the noise `eps`.
        Args:
            x_0: [FloatLike; [B, ...]], the given sample, `x_0`.
            t: [torch.long; [B]], the target timestep in range[1, T].
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
        alpha = torch.tensor(self.config.alphas(), dtype=torch.float32, device=device)
        # T
        (timesteps,) = alpha.shape
        # [T, ...]
        alpha = alpha.view([timesteps] + [1] * (x_0.dim() - 1))
        # [B, ...]
        return alpha[t - 1].sqrt().to(dtype) * x_0 + (1 - alpha[t - 1]).sqrt().to(
            dtype
        ) * eps.to(x_0)

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
            t: [torch.long; [B]], the current timestep in range[1, T].
            eps: [FloatLike; [...]], the additional noise from the prior.
            _to: [torch.long; [B]], target timestep,
                returns `x_{_to}` instead if given, otherwise `x_{t-1}`.
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
        alpha = torch.tensor(self.config.alphas(), dtype=torch.float32, device=device)
        # T
        (timesteps,) = alpha.shape
        # [T + 1, ...], one-based
        alpha = F.pad(alpha, [1, 0], mode="constant", value=1.0).view(
            [timesteps + 1] + [1] * (x_t.dim() - 1)
        )
        # [T], zero-based
        sigma = torch.tensor(self.config.sigmas(), dtype=torch.float32, device=device)
        # [T, ...], zero-based
        sigma = sigma.view([timesteps] + [1] * (x_t.dim() - 1))
        # [B, ...], predicted noise
        estim = self.forward(x_t, t)
        # [B, ...], predicted x_0
        x_0 = alpha[t].clamp_min(1e-7).rsqrt().to(dtype) * (
            x_t - (1 - alpha[t]).sqrt().to(dtype) * estim
        )
        mean = (
            alpha[t - 1].sqrt().to(dtype) * x_0
            + (1 - alpha[t - 1] - sigma[t - 1].square()).sqrt().to(dtype) * estim
        )
        return mean + sigma[t - 1].to(dtype) * eps.to(mean)
