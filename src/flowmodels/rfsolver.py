from typing import Callable, Iterable

import torch

from flowmodels.basis import ODESolver, VelocitySupports
from flowmodels.utils import VelocityInverter


class RFSolver(ODESolver):
    """Taming Rectified Flow for Inversion Editing, Wang et al., 2024.[arXiv:2411.04746]"""

    DEFAULT_STEPS = 10

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
            delta_t = 0.5 / steps
            for i in verbose(range(steps)):
                velocity = model.velocity(
                    x_t, torch.full((bsize,), i / steps, dtype=torch.float32)
                )
                x_t_half = x_t + velocity * delta_t
                velocity_half = model.velocity(
                    x_t_half,
                    torch.full((bsize,), i / steps + delta_t, dtype=torch.float32),
                )
                # computing acceleration
                accel = (velocity_half - velocity) / delta_t
                # second-order updates
                x_t = x_t + velocity / steps + 0.5 * accel / (steps**2)
                x_ts.append(x_t)
        return x_t, x_ts


class RFInversion(RFSolver):

    def solve(
        self,
        model: VelocitySupports,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Invert the velocity and pass to solver."""
        return super().solve(
            VelocityInverter(model),
            init,
            steps,
            verbose,
        )
