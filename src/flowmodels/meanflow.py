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
        p: float = 1.0,
        tangent_warmup: int | None = None,
        _warmup_max: float = 1.0,
        _approx_jvp: bool = True,
        _dt: float = 0.005,
    ):
        super().__init__()
        self.velocity_estim = module
        self.p_mean = p_mean
        self.p_std = p_std
        self.p = p
        # debug purpose
        self._debug_from_loss = {}
        self._tangent_warmup = tangent_warmup
        self.register_buffer("_steps", torch.tensor(0, requires_grad=False))
        self._warmup_max = _warmup_max
        self._approx_jvp = _approx_jvp
        self._dt = _dt

    # debug purpose
    def _debug_purpose(self):
        return {**self._debug_from_loss, **getattr(self.velocity_estim, "_debug_purpose", {})}

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate the mean velocity from the given `t` to `r`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
            r: [FloatLike; [B]], the terminal timestep in range[0, 1]; r < t.
        Returns:
            estimated mean velocity from the given sample `x_t`.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        return self.velocity_estim(x_t, t, t - r, **kwargs)

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t` (assume the terminal is zero).
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        # targeting the origin point
        return self.forward(x_t, t, torch.zeros_like(t), label=label)

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
        (bsize,) = t.shape
        return x_t - t.view([bsize] + [1] * (x_t.dim() - 1)) * self.velocity(
            x_t, t, label=label
        )

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
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
        if r is None:
            r = torch.sigmoid(
                torch.randn(batch_size, device=device) * self.p_std + self.p_mean
            )
            t, r = torch.maximum(t, r), torch.minimum(t, r)
        # [B, ...]
        bt = t.view([batch_size] + [1] * (sample.dim() - 1))
        br = r.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = (1 - bt) * sample + bt * prior
        # warmup scaler
        w = 1.0
        if self._tangent_warmup:
            self._steps.add_(1)
            w = (self._steps / self._tangent_warmup).clamp_max(self._warmup_max)
        v = self.forward(x_t, t, t, label=label)
        v_t = w * v + (1 - w) * (prior - sample)
        if self._approx_jvp:
            # shortcut
            dt = self._dt
            dudt = (
                self.forward(x_t + dt * v_t, t + dt, r, label=label)
                - self.forward(x_t - dt * v_t, t - dt, r, label=label)
            ) / (2 * dt)
            u = self.forward(x_t, t, r, label)
        else:
            jvp_fn = torch.compiler.disable(
                torch.func.jvp, recursive=False  # pyright: ignore
            )
            # [B, ...], [B, ...], jvp = dF/dt
            u, dudt = jvp_fn(
                lambda x, t, r: self.forward(x, t, r, label=label),
                (x_t, t, r),  # pyright: ignore
                (v_t, torch.ones_like(t), torch.zeros_like(r)),
            )
        # [B, ...]
        u_tgt = v_t - (bt - br) * dudt
        # [B]
        rdim = [i + 1 for i in range(u.dim() - 1)]
        meanid = (u - u_tgt.detach()).square().mean(dim=rdim)
        v_loss =  (v - (sample - prior)).square().mean(dim=rdim)
        # [B]
        loss = meanid + v_loss
        adp_wt = (loss + 0.01).detach() ** self.p
        with torch.no_grad():
            self._debug_from_loss = {
                "meanflow/mse": meanid.mean().item(),
                "meanflow/v_loss": v_loss.mean().item(),
                "meanflow/adp_wt": adp_wt.mean().item(),
            }
        return (loss / adp_wt).mean()

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
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
                velocity = self.forward(x_t, t, t - 1 / steps, label=label)
                x_t = x_t - velocity / steps
                x_ts.append(x_t)

        return x_t, x_ts
