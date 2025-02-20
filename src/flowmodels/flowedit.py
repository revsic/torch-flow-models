from typing import Callable, Iterable, Protocol

import torch


class ConditionalVelocitySupports(Protocol):

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Estimate the conditional velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
            c: [FloatLike; [B, ...]], the conditional vectors.
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        ...


class FlowEditSolver:
    """FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models."""

    DEFAULT_STEPS = 10

    def solve(
        self,
        model: ConditionalVelocitySupports,
        init: torch.Tensor,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        src: torch.Tensor | None = None,
        tgt: torch.Tensor | None = None,
        n_avg: int = 1,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE defined in the range of t; [0, 1].
        Args:
            model: the velocity estimation model.
            init: [FloatLike; [B, ...]], starting point of the ODE.
            steps: the number of the steps, default 100 iterations.
            verbose: whether writing the progress of the generations or not.
            src, tgt: [FloatLike; [B, ...]] the source and target conditional vectors.
        Returns:
            [FloatLike; [B, ...]], the solution.
            `steps` x [FloatLike; [B, ...]], trajectories.
        """
        assert src is not None, "source conditional vectors should be given"
        assert tgt is not None, "target conditional vectors should be given"
        # assign default values
        steps = steps or self.DEFAULT_STEPS
        if verbose is None:
            verbose = lambda x: x
        # loop
        z_t_fe, x_ts = init, []
        bsize, *_ = z_t_fe.shape
        with torch.inference_mode():
            for i in verbose(range(steps)):
                # [B]
                t = torch.full((bsize,), i / steps, dtype=torch.float32)
                # [B, ...]
                bt = t.view([bsize] + [1] * (z_t_fe.dim() - 1))
                # velocity accumulation
                v = 0
                for _ in range(n_avg):
                    # sample from gaussian process
                    n = torch.randn_like(z_t_fe)
                    # forward process
                    z_t_src = bt.to(n) * init + (1 - bt).to(n) * n
                    z_t_tgt = z_t_fe + z_t_src - init
                    v = (
                        v
                        + model.velocity(z_t_tgt, t, tgt)
                        - model.velocity(z_t_src, t, src)
                    )
                # averaging the direction
                v = v / n_avg
                # updates
                z_t_fe = z_t_fe + v / steps
                x_ts.append(z_t_fe)
        return z_t_fe, x_ts
