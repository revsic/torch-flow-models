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
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        cfg_scale: float | None = None,
        uncond: torch.Tensor | None = None,
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
        bsize, *_ = init.shape
        # assign default values
        steps = steps or self.DEFAULT_STEPS
        if verbose is None:
            verbose = lambda x: x
        # sanity check
        if cfg_scale is not None:
            assert label is not None and uncond is not None
            rdim = [1 for _ in range(uncond.dim())]
            uncond = uncond[None].repeat([bsize] + rdim)
        # loop
        x_t, x_ts = init, []
        with torch.inference_mode():
            for i in verbose(range(steps)):
                t = torch.full((bsize,), i / steps, dtype=torch.float32)
                velocity = model.velocity(x_t, t, label=label)
                if cfg_scale:
                    uncond_vel = model.velocity(x_t, t, label=uncond)
                    velocity = uncond_vel + cfg_scale * (velocity - uncond_vel)
                x_t = x_t + velocity / steps
                x_ts.append(x_t)
        return x_t, x_ts
