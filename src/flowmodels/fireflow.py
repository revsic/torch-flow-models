from typing import Callable, Iterable

import torch

from flowmodels.basis import ODESolver, VelocityInverter, VelocitySupports


class FireFlowSolver(ODESolver):
    """FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing, Deng et al., 2024.[arXiv:2412.07517]"""

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
            # cached velocity from the previous midpoint estimation
            # , initialize with t=0
            velocity = model.velocity(
                x_t,
                torch.zeros(bsize, dtype=torch.float32),
            )
            for i in verbose(range(steps)):
                # load the cache
                x_t_half = x_t + velocity * 0.5 / steps
                # save the cache
                velocity = model.velocity(
                    x_t_half,
                    torch.full(
                        (bsize,),
                        (i + 0.5) / steps,
                        dtype=torch.float32,
                    ),
                )
                x_t = x_t + velocity / steps
                x_ts.append(x_t)
        return x_t, x_ts


class FireFlowInversion(FireFlowSolver):

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
