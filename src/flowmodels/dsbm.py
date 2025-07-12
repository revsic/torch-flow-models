import copy
from typing import Callable, Iterable, Literal, TypedDict

import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import ODESolver, SamplingSupports, ScoreModel, VelocitySupports


class DiffusionSchrodingerBridgeMatching(
    nn.Module, ScoreModel, SamplingSupports, VelocitySupports
):
    def __init__(self, module: nn.Module, std: float = 1.0):
        super().__init__()
        self.fwd = module  # obj: (x_1 - x_0)
        self.bwd = copy.deepcopy(module)  # obj: (x_0 - x_1)
        self.std = std
        self.sampler = ModifiedVanillaEulerSolver()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor | None = None,
        direction: Literal["fwd", "bwd"] = "bwd",
    ) -> torch.Tensor:
        """Estimate the score from the given `x_t` w.r.t. the given direction.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [torch.long; [B]], the current timesteps of the sample in range[0, 1].
        Returns:
            estimated score from the given sample `x_t`.
        """
        kwargs = {}
        if label is not None:
            kwargs["label"] = label
        if direction == "fwd":
            return self.fwd(x_t, t, **kwargs)
        return self.bwd(x_t, t, **kwargs)

    def score(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor | None = None,
        direction: Literal["fwd", "bwd"] = "bwd",
    ):
        """Estimate the stein score from the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the etimated scores.
        """
        # ideal case: fwd = -bwd
        mean = 0.5 * (self.fwd(x_t, t, label) - self.bwd(x_t, t, label))
        if direction == "fwd":
            return mean
        # reverse direction
        return -mean

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor | None = None,
        direction: Literal["fwd", "bwd"] = "bwd",
    ) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        return self.score(x_t, t, label, direction)

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        direction: Literal["fwd", "bwd"] = "fwd",
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the samples from the distribution `z_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from the uniform distribution if not provided.
            prior: [FloatLike; [B, ...]], the samples from the distribution `z_1`,
                sample from the gaussian if not provided.
            direction: compute the loss
                that matches the score from `z_t` to `z_1` with `self.fwd` if given is `fwd`,
                otherwise matches the score from `z_t` to `z_0` with `self.bwd`.
        Returns:
            [FloatLike; []], loss value.
        """
        bsize, *_ = sample.shape
        if t is None:
            t = torch.rand(bsize)
        if prior is None:
            prior = torch.randn_like(sample)
        # rename for clarifying
        x_0, x_1 = sample, prior
        # [B, ...]
        _t = t.view([bsize] + [1] * (sample.dim() - 1))
        # sample from brownian motion (B_t - tB_1) ~ N(0, t(1 - t)I)
        z = (_t * (1 - _t)).sqrt().to(x_1) * torch.randn_like(x_1)
        # brownian bridge between z_1 and z_0
        x_t = _t.to(x_1) * x_1 + (1 - _t).to(x_1) * x_0 + self.std * z
        # estimate
        estim = self.forward(x_t, t, label, direction)
        # direction-aware targets
        target: torch.Tensor
        if direction == "fwd":
            # score from `z_t` to `z_1`
            ## grad Q_{1|t}(X_1|X_t) = (X_1 - X_t) / (1 - t)
            ## = (X_1 - (tX_1 + (1-t)X_0 + std * sqrt(t(1 - t))Z)) / (1 - t)
            ## = ((1-t)X_1 - (1-t)X_0 - std * sqrt(t(1 - t))Z) / (1 - t)
            ## = X_1 - X_0 - std * sqrt(t / (1 - t))Z
            target = (
                x_1
                - x_0
                - self.std * (_t / (1.0 - _t).clamp_min(1e-7)).sqrt().to(x_1) * z
            )
        else:
            # score from `z_t` to `z_0`
            ## grad Q_{0|t}(X_0|X_t) = (X_0 - X_t) / t
            ## = (X_0 - (tX_1 + (1-t)X_0 + std * sqrt(t(1 - t))Z)) / t
            ## = X_0 - X_1 - std * sqrt((1-t)/t) Z
            target = (
                x_0
                - x_1
                - self.std * ((1.0 - _t) / _t.clamp_min(1e-7)).sqrt().to(x_1) * z
            )
        return (estim - target).square().mean()

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        direction: Literal["fwd", "bwd"] = "bwd",
    ):
        """Forward to the ModifiedVanillaEulerSolver."""
        return self.sampler.solve(
            self,
            prior,
            label,
            steps,
            verbose,
            std=0.0,
            direction=direction,
        )

    class Logs(TypedDict):
        direction: str
        losses: list[float]

    def imf(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        optim: torch.optim.Optimizer,
        batch_size: int,
        inner_steps: int,
        outer_steps: int = 40,
        steps: int = 20,
        label: torch.Tensor | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> list[Logs]:
        """Training loop for the DSBM, Iterative Markovian Fitting.
        Args:
            x_0: [FloatLike; [D, ...]], the samples from the distribution `pi_0`.
            x_1: [FloatLike; [D, ...]], the samples from the distribution `pi_1`.
            optim: an optimizer constructed with `self.parameters`.
            inner_steps: the number of the training steps for markovian projection.
            outer_steps: the number of the training steps for each forward/backward process models iteratively(#reciprocal projection).
            steps: the number of the sampling steps for the reciprocal projection.
            batch_size: the number of the batch.
        Returns:
            a list of the loss values.
        """
        if verbose is None:
            verbose = lambda x: x

        losses = []
        _label = None
        # backward first (since we assume `x_0` is the data distribution, and `x_1` is the prior)
        direction = "bwd"
        # independent coupling for initializing training loop
        pi_0, pi_1 = x_0, x_1
        for outer in verbose(range(outer_steps)):
            _logs = {"direction": direction, "losses": []}
            for inner in verbose(range(inner_steps)):
                # [B]
                i = torch.randint(0, len(x_0), (batch_size,))
                if label is not None:
                    _label = label[i]
                # [B, ...], sampling
                z_0, z_1 = pi_0[i], pi_1[i]
                # []
                loss = self.loss(z_0, prior=z_1, label=_label, direction=direction)
                # update
                optim.zero_grad()
                loss.backward()
                optim.step()
                # log
                _logs["losses"].append(loss.detach().item())

            losses.append(_logs)
            # reciprocal projection
            if direction == "fwd":
                pi_0 = x_0
                pi_1, _ = self.sample(x_0, label, steps, verbose, direction="fwd")
            else:
                pi_1 = x_1
                pi_0, _ = self.sample(x_1, label, steps, verbose, direction="bwd")

            # flip
            direction = "bwd" if direction == "fwd" else "fwd"

        return losses


class ModifiedVanillaEulerSolver(ODESolver):
    """Modified version of vanilla Euler method"""

    DEFAULT_STEPS: int = 20

    def solve(
        self,
        model: VelocitySupports,
        init: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        std: float = 0.0,
        direction: Literal["fwd", "bwd"] = "bwd",
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the velocity estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps, default 100 iterations.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        # assign default values
        steps = steps or self.DEFAULT_STEPS
        if verbose is None:
            verbose = lambda x: x
        # loop
        x_t, x_ts = init, []
        bsize, *_ = x_t.shape
        with torch.inference_mode():
            for i in verbose(range(steps)):
                t = i / steps
                # invert the timesteps
                if direction == "bwd":
                    t = 1 - t

                velocity = model.velocity(
                    x_t,
                    torch.full((bsize,), t, dtype=torch.float32),
                    label=label,
                )
                x_t = (
                    x_t
                    + velocity / steps
                    + std * torch.randn_like(x_t) * np.sqrt(1 / steps)
                )
                x_ts.append(x_t)
        return x_t, x_ts
