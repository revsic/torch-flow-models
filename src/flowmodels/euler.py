from typing import Callable, Iterable

import torch

from flowmodels.basis import ODESolver, VelocitySupports


class VanillaEulerSolver(ODESolver):
    """Vanilla Euler Method for solving ODE."""

    DEFAULT_STEPS = 100

    def solve(
        self,
        model: VelocitySupports,
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
