from typing import Callable, Iterable

import torch


class ODEModel:
    """Basis of the ODE models."""

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        raise NotImplementedError("ODEModel.velocity is not implemented.")

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss from the sample.
        Args:
            sample: [FloatLike; [B, ...]], the training data, `x_0`.
            t: [FloatLike; [B]], the target timesteps in range[0, 1],
                sample from uniform distribution if not provided.
        Returns:
            [FloatLike; []], loss value.
        """
        raise NotImplementedError("ODEModel.loss is not implemented.")


class ODESolver:
    """ODE Solver."""

    def solve(
        self,
        model: ODEModel,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the velocity estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps.
            verbose: whether writing the progress of the generations or not.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        raise NotImplementedError("ODESolver.solve is not implemented.")


class VanillaEulerSolver(ODESolver):
    """Vanilla Euler Method for solving ODE."""

    DEFAULT_STEPS = 100

    def solve(
        self,
        model: ODEModel,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
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
                velocity = model.velocity(
                    x_t,
                    torch.full((bsize,), i / steps, dtype=torch.float32),
                )
                x_t = x_t + velocity / steps
                x_ts.append(x_t)
        return x_t, x_ts
