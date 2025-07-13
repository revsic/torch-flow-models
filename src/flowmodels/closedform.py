from typing import Callable, Iterable

import torch

from flowmodels.basis import SamplingSupports, VelocitySupports
from flowmodels.euler import VanillaEulerSolver


class VelocityField(VelocitySupports, SamplingSupports):
    """Closed-form Velocity field.
    On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity, Bertrand et al., 2025.[arXiv:2506.03719]
    """

    def __init__(
        self,
        data: torch.Tensor,
        max_chunk: int = 10000,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        self.data = data
        self.max_chunk = max_chunk
        self.solver = VanillaEulerSolver()
        self.verbose = verbose

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        label: torch.Tensor | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        """Estimate the velocity of the given samples, `x_t`.
        Args:
            x_t: [FloatLike; [B, ...]], the given samples, `x_t`.
            t: [FloatLike; [B]], the current timesteps, in range[0, 1].
            label: [Any, [B, ...]], additional conditions.
        Returns:
            [FloatLike; [B, ...]], the estimated velocity.
        """
        if verbose is None:
            verbose = self.verbose
        return self.chunked(self.data, x_t, t.to(x_t), self.max_chunk, verbose=verbose)

    def sample(
        self,
        prior: torch.Tensor,
        label: torch.Tensor | None = None,
        steps: int | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ):
        """Forward to the vanilla Euler solver."""
        return self.solver.solve(self, prior, label, steps, verbose)

    @classmethod
    def closedform(
        cls, data: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the closed-form unconditional velocities.
        Args:
            data: [FloatLike; [N, C]], reference data, flattened.
            x_t: [FloatLike; [B, C]], query data.
            t: [FloatLike; [B]], current timesteps.
        Returns:
            [FloatLike; [B]], computed velocities.
        """
        # [N, B, C]
        z = (data[:, None] - x_t) / (1 - t[:, None])
        # [N, B, C] > [N, B]
        l2 = (x_t - t[:, None] * data[:, None]).square().sum(dim=-1)
        # [N, B]
        lambda_ = torch.softmax(-l2 / (2 * (1 - t).square()), dim=0)
        # [B, C]
        return (lambda_[..., None] * z).sum(dim=0)

    @classmethod
    def chunked(
        cls,
        data: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        max_chunk: int = 10000,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> torch.Tensor:
        """Compute the closed-form unconditional velocities.
        Args:
            data: [FloatLike; [N, C]], reference data, flattened.
            x_t: [FloatLike; [B, C]], query data.
            t: [FloatLike; [B]], current timesteps.
            max_chunk: the size of the chunk.
        Returns:
            [FloatLike; [B]], computed velocities.
        """
        if verbose is None:
            verbose = lambda x: x
        B, _ = x_t.shape
        if B > max_chunk:
            v_ts = []
            for i in verbose(range(0, B, max_chunk)):
                v_ts.append(
                    cls.chunked(
                        data, x_t[i : i + max_chunk], t[i : i + max_chunk], max_chunk
                    )
                )
            return torch.cat(v_ts, dim=0)

        N, _ = data.shape
        log_units, log_denom = [], torch.zeros_like(t)
        for i in verbose(range(0, N, max_chunk)):
            chunk = data[i : i + max_chunk]
            # [max_chunk, B, C] > [max_chunk, B]
            l2 = (x_t - t[:, None] * chunk[:, None]).square().sum(dim=-1)
            # [max_chunk, B]
            log_unit = -l2 / (2 * (1 - t).square())
            log_units.append(log_unit)
            # [B]
            log_denom.add_(torch.logsumexp(log_unit, dim=0))
        # update
        v = 0
        for i, log_unit in zip(
            verbose(range(0, N, max_chunk)),
            log_units,
        ):
            chunk = data[i : i + max_chunk]
            # [max_chunk, B]
            lambda_ = (log_unit - log_denom).exp()
            # [max_chunk, B, C]
            z = (chunk[:, None] - x_t) / (1 - t[:, None])
            # [B, C]
            chunked_v = (lambda_[..., None] * z).sum(dim=0)
            v = v + chunked_v
        return v  # pyright: ignore
