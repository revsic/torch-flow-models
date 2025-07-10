from typing import Callable, Iterable

import torch

from flowmodels.basis import ODESolver, VelocitySupports
from flowmodels.utils import VelocityInverter


class ControlledODESolver(ODESolver):
    """Semantic Image Inversion And Editing Using Rectified Stochastic Differential Equations,
    Rout et al., 2024.[arXiv:2410.10792]"""

    DEFAULT_STEPS = 10

    def solve(
        self,
        model: VelocitySupports,
        init: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        reference: torch.Tensor | None = None,
        tau: float = 0.3,
        eta: float = 0.6,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the velocity estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps, default 100 iterations.
            verbose: whether writing the progress of the generations or not.
            reference: the reference samples for controlling ODE, assertion failure if not given.
            tau, eta: hyperparameters for controlling ODE, time truncator and controller guidance.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        assert (
            reference is not None
        ), "reference sample should be given to contorl ODE trajectories."
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
                # truncation
                if t > tau:
                    eta = 0
                # u_t(Y_t|y_1) = c(Y_t, t) = (y_1 - Y_t) / (1 - t)
                controller = (reference - x_t) / (1 - t)
                velocity = model.velocity(
                    x_t,
                    torch.full((bsize,), t, dtype=torch.float32),
                    label=label,
                )
                # controlled velocity
                velocity = velocity + eta * (controller - velocity)
                # updates
                x_t = x_t + velocity / steps
                x_ts.append(x_t)
        return x_t, x_ts


class ControlledODEInversion(ControlledODESolver):

    def solve(
        self,
        model: VelocitySupports,
        init: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        reference: torch.Tensor | None = None,
        tau: float = 0.3,
        eta: float = 0.6,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Invert the velocity and pass to solver."""
        # sample from the prior distribution
        if reference is None:
            reference = torch.randn_like(init)

        return super().solve(
            VelocityInverter(model),
            init,
            label,
            steps,
            verbose,
            reference=reference,
            tau=tau,
            eta=eta,
        )
