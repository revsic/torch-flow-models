from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from flowmodels.basis import Scheduler, SchedulerProtocol
from flowmodels.ddpm import DDPMSampler


@dataclass
class DDIMScheduler(Scheduler):
    """Variance scheduler,
    Improved Denoising Diffusion Probabilistic Models, Nichol & Dhariwal, 2021.[arXiv:2102.09672]
    """

    T: int = 40
    s: float = 0.008
    eta: float = 0.0

    def var(self) -> torch.Tensor:
        T, s = self.T, self.s
        # [T]
        t = torch.arange(T, dtype=torch.float32)
        # [T]
        schedule = ((t / T + s) / (1 + s) * np.pi / 2).cos().square()
        # [T]
        return 1 - (schedule / schedule[0].clamp_min(1e-7))

    def sigmas(self) -> torch.Tensor:
        # [T + 1]
        alphas = F.pad(1 - self.var(), [1, 0], "constant", 1.0)
        # [T]
        return (
            self.eta
            # \sqrt{(1 - \alpha_{t-1}) / (1 - \alpha_t)}
            * ((1 - alphas[:-1]) / (1 - alphas[1:]).clamp_min(1e-7)).sqrt()
            # \sqrt{1 - \alpha_t / \alpha_{t-1}}
            * (1 - alphas[1:] / alphas[:-1].clamp_min(1e-7)).sqrt()
        )


class DDIMSamplerSupports(SchedulerProtocol, Protocol):
    def sigmas(self) -> torch.Tensor:
        """Stochasity controller.
        Returns:
            list of standard deviations that controlls the stochasity.
        """
        ...


class DDIMSampler(DDPMSampler):
    """Denoising Diffusion Implicit Models, Song et al., 2020.[arXiv:2010.02502]"""

    def __init__(self, scheduler: DDIMSamplerSupports):
        self.scheduler = scheduler
        assert self.scheduler.vp, "unsupported scheduler; variance-exploding scheduler"

    def denoise(
        self,
        x_t: torch.Tensor,
        e_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoise the given sample `x_t` to the single-step backward `x_{t-1}`
            w.r.t. the current timestep `t` and the additional noise `eps`.
        Args:
            x_t: [FloatLike; [B, ...]], the given sample, `x_t`.
            e_t: [FloatLike; [B, ...]], the estimated noise of the `x_t`.
            t: [torch.long; [B]], the current timestep in range[1, T].
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
        alpha = 1 - self.scheduler.var().to(device)
        # [T + 1, ...], one-based
        alpha = F.pad(alpha, [1, 0], mode="constant", value=1.0).view(
            [self.scheduler.T + 1] + [1] * (x_t.dim() - 1)
        )
        # [T], zero-based
        sigma = self.scheduler.sigmas().to(device)
        # [T, ...], zero-based
        sigma = sigma.view([self.scheduler.T] + [1] * (x_t.dim() - 1))
        # [B, ...], predicted x_0
        x_0 = alpha[t].clamp_min(1e-7).rsqrt().to(dtype) * (
            x_t - (1 - alpha[t]).sqrt().to(dtype) * e_t
        )
        mean = (
            alpha[t - 1].sqrt().to(dtype) * x_0
            + (1 - alpha[t - 1] - sigma[t - 1].square()).sqrt().to(dtype) * e_t
        )
        return mean + sigma[t - 1].to(dtype) * eps.to(mean)
