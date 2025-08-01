import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import ODEModel, SamplingSupports
from flowmodels.euler import VanillaEulerSolver
from flowmodels.sct import _AdaptiveWeights
from models.lightningdit import LightningDiT_XL_1
from tokenizer.vavae import VA_VAE
from trainer import Cifar10Trainer, TrainConfig


class LinearScaledConsistencyModel(
    nn.Module,
    ODEModel,
    SamplingSupports,
):
    def __init__(
        self,
        module: nn.Module,
        p_mean: float = -1.0,
        p_std: float = 1.4,
        tangent_warmup: int | None = None,
        _warmup_max: float = 1.0,
        _ada_weight_size: int = 128,
        _approx_jvp: bool = True,
        _dt: float = 0.005,
    ):
        super().__init__()
        self.F0 = module
        self.p_mean = p_mean
        self.p_std = p_std
        self._tangent_warmup = tangent_warmup
        self._warmup_max = _warmup_max
        self.register_buffer("_steps", torch.tensor(0, requires_grad=False))
        self._ada_weight = _AdaptiveWeights(_ada_weight_size)
        self.solver = VanillaEulerSolver()
        # debug purpose
        self._debug_from_loss = {}
        self._approx_jvp = _approx_jvp
        self._dt = _dt

    # debug purpose
    def _debug_purpose(self):
        return {**self._debug_from_loss, **getattr(self.F0, "_debug_purpose", {})}

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the `x_0` from the given `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]] the given noised sample, `x_t`.
            t: [FloatLike; [B]], the current timestep in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], estimated sample from the given `x_t`.
        """
        (bsize,) = t.shape
        return x_t + (1 - t.view([bsize] + [1] * (x_t.dim() - 1))) * self.velocity(
            x_t, t, label=label
        )

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
            kwargs["y"] = label.to(x_t.device)
        return -self.F0.forward(x_t, 1 - t.to(x_t.device), **kwargs)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], training data, `X_1`.
            t: [FloatLike; [B]], target timesteps in range[0, 1],
                sample from the proposal distribution if not provided.
            prior: [FloatLike; [B, ...]], sample from the prior distribution, `X_0`,
                sample from gaussian if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        # shortcut
        batch_size, *_ = sample.shape
        device = sample.device
        # sample
        if prior is None:
            prior = torch.randn_like(sample)
        if t is None:
            # sample from log-normal
            rw_t = (
                torch.randn(batch_size, device=device) * self.p_std + self.p_mean
            ).exp()
            # [T], in range[0, 1]
            t = (rw_t / 0.5).atan() / np.pi * 2
        # [B, ...]
        _t = t.view([batch_size] + [1] * (sample.dim() - 1))
        # [B, ...]
        x_t = _t * sample + (1 - _t) * prior
        # [B, ...]
        v_t = sample - prior
        if self._approx_jvp:
            # shortcut
            dt = self._dt
            fwd = lambda t, bt: self.velocity(
                bt * sample + (1 - bt) * prior, t, label=label
            )
            # approximation w/o JVP
            jvp = (fwd(t + dt, _t + dt) - fwd(t - dt, _t - dt)) / (2 * dt)
            estim = self.velocity(x_t, t, label=label)
        else:
            jvp_fn = torch.compiler.disable(
                torch.func.jvp, recursive=False  # pyright: ignore
            )
            # [B, ...], [B, ...], jvp = dF/dt
            estim, jvp, *_ = jvp_fn(
                lambda x, t: self.velocity(x, t, label=label),
                (x_t, t),  # pyright: ignore
                (v_t, torch.ones_like(t)),
            )
        # warmup scaler
        r = 1.0
        if self._tangent_warmup:
            self._steps.add_(1)
            r = (self._steps / self._tangent_warmup).clamp_max(self._warmup_max)
        # stop grad
        F = estim.detach()
        # df/dt = (x_1 - x_0) - F(x_t, t) + (1 - t) * dF/dt
        grad = (1 - _t) * (v_t - F + r * (1 - _t) * jvp.detach())
        # reducing dimension
        rdim = [i + 1 for i in range(x_t.dim() - 1)]
        # normalized tangent
        normalized_tangent = grad / (
            _norm := grad.norm(p=2, dim=rdim, keepdim=True) + 0.1
        )
        # [B]
        mse = (estim - F - normalized_tangent).square().mean(dim=rdim)
        # [B], adaptive weighting
        logvar = self._ada_weight.forward(t)
        # [B]
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
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        cfg_scale: float | None = None,
        uncond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward to the MultistepConsistencySampler."""
        # pre-scale the prior
        return self.solver.solve(
            self,
            prior,
            label,
            steps,
            verbose=verbose,
            cfg_scale=cfg_scale,
            uncond=uncond,
        )


class VAVAEWrapper:
    def __init__(self):
        self.vavae = VA_VAE("./tokenizer/configs/vavae_f16d32.yaml")
        
        stats = torch.load("/data2/shared/LightningDit/latents_stats.pt")
        self.latent_mean = stats["mean"].squeeze(dim=0)
        self.latent_std = stats["std"].squeeze(dim=0)
        assert self.latent_mean.dim() == self.latent_std.dim() == 3

    @torch.no_grad()
    def postproc(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self.latent_std.to(z) + self.latent_mean.to(z)
        if (require_squeeze := z.dim() == 3):
            z = z[None]
        img = (self.vavae.model.decode(z).clamp(-1.0, 1.0) + 1) * 0.5
        if require_squeeze:
            img = img.squeeze(dim=0)
        return img

@dataclass
class Config:
    p_mean: float = -1.0
    p_std: float = 1.4
    tangent_warmup: int = 10000
    warmup_max: float = 1.0
    approx_jvp: bool = True
    dt: float = 0.005

    train: TrainConfig = field(default_factory=TrainConfig)  # pyright: ignore


def reproduce_linear_sct_cifar10():
    config = Config(
        train=TrainConfig(
            n_gpus=1,
            n_grad_accum=32,
            mixed_precision="bf16",
            batch_size=768,
            n_classes=1000 + 1,
            lr=0.0002,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
            clip_grad_norm=0.1,
            nan_to_num=0.0,
            total=400000,
            half_life_ema=500000,
            label_dropout=0.1,
            uncond_label=1000,
            fid_steps=2,
            eval_interval=80,
        ),
    )
    # model definition
    backbone = LightningDiT_XL_1(
        input_size=256 // 16,
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        in_channels=32,
        learn_sigma=False,
    )
    ckpt = torch.load(
        "/data2/shared/LightningDit/lightningdit-xl-imagenet256-800ep.pt",
        map_location=lambda storage, loc: storage,
    )
    if "ema" in ckpt:
        ckpt = ckpt["ema"]
    backbone.load_state_dict(ckpt)
    del ckpt

    model = LinearScaledConsistencyModel(
        backbone,
        p_mean=config.p_mean,
        p_std=config.p_std,
        tangent_warmup=config.tangent_warmup,
        _warmup_max=config.warmup_max,
        _approx_jvp=config.approx_jvp,
        _dt=config.dt,
    )
    vavae = VAVAEWrapper()

    # timestamp
    workspace = Path(f"./test.workspace/linear-sct-cifar10-cond/{config.train.stamp}")
    workspace.mkdir(parents=True, exist_ok=True)
    with open(workspace / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    config.train.workspace = workspace.as_posix()
    trainer = Cifar10Trainer(model, config.train, _postproc=vavae.postproc)
    try:
        trainer.train(sampling_batch_size=128)
    except:
        with open(workspace / "exception", "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    reproduce_linear_sct_cifar10()
