from typing import Callable, Iterable

import torch
import torch.nn as nn

from flowmodels.basis import ODEModel, PredictionSupports, SamplingSupports


class MeanFlow(nn.Module, ODEModel, PredictionSupports, SamplingSupports):
    """
    Mean Flows for One-step Generative Modeling, Geng et al., 2025.
    """

    DEFAULT_STEPS = 4

    def __init__(
        self,
        module: nn.Module,
        p_mean: float = -0.4,
        p_std: float = 1.0,
        r_mask: float = 0.75,
    ):
        super().__init__()
        self.velocity_estim = module
        self.p_mean = p_mean
        self.p_std = p_std
        self.r_mask = r_mask
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
        return self.velocity_estim(x_t, t, t - r)

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
        src: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
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
        if t is None:
            t = torch.sigmoid(
                torch.randn(batch_size, device=device) * self.p_std + self.p_mean
            )
        if src is None:
            src = torch.randn_like(sample)
        if r is None:
            r = torch.sigmoid(
                torch.randn(batch_size, device=device) * self.p_std + self.p_mean
            )
            t, r = torch.maximum(t, r), torch.minimum(t, r)
            # masking for instantaneous velocity learning (case; r = t)
            mask = torch.arange(batch_size) < int(batch_size * self.r_mask)
            r = torch.where(mask, t, r)
        # [B, ...]
        bt = t.view([batch_size] + [1] * (sample.dim() - 1))
        br = r.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = (1 - bt) * sample + bt * src
        v_t = src - sample
        # jvp for meanflow identity
        jvp_fn = torch.compiler.disable(
            torch.func.jvp, recursive=False  # pyright: ignore
        )
        u, dudt = jvp_fn(
            self.forward,
            (x_t, t, r),  # pyright: ignore
            (v_t, torch.ones_like(t), torch.zeros_like(r)),
        )
        # [B, ...]
        u_tgt = v_t - (bt - br) * dudt
        # [B]
        rdim = [i + 1 for i in range(u.dim() - 1)]
        loss = (u - u_tgt.detach()).square().mean(dim=rdim)
        # [B]
        adp_wt = (loss + 0.01).detach()
        with torch.no_grad():
            self._debug_from_loss = {
                "meanflow/mse": loss.mean().item(),
            }
        return (loss / adp_wt).mean()

    def sample(
        self,
        prior: torch.Tensor,
        steps: int | None = 1,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Transfer the samples from the prior distribution to the trained distribution, using vanilla Euler method.
        Args:
            prior: [FloatLike; [B, ...]], samples from the prior distribution.
            steps: the number of the steps.
        """
        steps = steps or self.DEFAULT_STEPS
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
