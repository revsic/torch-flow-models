from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from flowmodels.basis import ContinuousScheduler, ScoreModel, Scheduler
from flowmodels.rf import RectifiedFlow
from flowmodels.pfode import ProbabilityFlowODESampler


class InstaFlow(RectifiedFlow):
    """InstaFlow: One Step is Enough For High-Quality Diffusion-Based Text-to-Image Generation, Liu et al., ICLR 2024.[arXiv:2309.06380]"""

    def distill_from_score(
        self,
        score_model: ScoreModel,
        scheduler: Scheduler | ContinuousScheduler,
        optim: torch.optim.Optimizer,
        prior: torch.Tensor,
        training_steps: int,
        batch_size: int,
        steps: int | None = None,
        label: torch.Tensor | None = None,
        verbose: Callable[[range], Iterable] | None = None,
    ) -> tuple[list[float], torch.Tensor]:
        """Distill the knowledge from the given score model.
        Args:
            score_model: the target score model.
            scheduler: the noise scheduler for sampling score model.
            optim: optimizer constructed with `self.parameters()`.
            prior: [FloatLike; [D, ...]], samples from the prior distribution.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            steps: the number of the sampling iterations (for score model).
        """
        sampler = ProbabilityFlowODESampler(scheduler)
        sample, _ = sampler.sample(
            model=score_model, prior=prior, label=label, steps=steps, verbose=verbose
        )
        losses = super().reflow(
            optim=optim,
            prior=prior,
            training_steps=training_steps,
            batch_size=batch_size,
            sample=sample,
            label=label,
            verbose=verbose,
        )
        return losses, sample

    def distillation(
        self,
        optim: torch.optim.Optimizer,
        prior: torch.Tensor,
        training_steps: int,
        batch_size: int,
        sample: torch.Tensor | int = 1000,
        label: torch.Tensor | None = None,
        verbose: Callable[[range], Iterable] | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ):
        """Distillation for single-step generation.
        Args:
            optim: optimizer constructed with `self.parameters()`.
            prior: [FloatLike; [D, ...]], samples from the source distribution, `X_0`.
            training_steps: the number of the training steps.
            batch_size: the size of the batch.
            sample: [FloatLike; [D, ...]], corresponding samples transfered from `prior`,
                sample just-in-time if given is integer (assume as the number of the steps for sampling iterations).
            loss_fn: similarity measure between the sample and single-step generated one.
        """
        if isinstance(sample, int):
            with torch.inference_mode():
                sample, _ = self.sample(prior, label, sample, verbose)

        if verbose is None:
            verbose = lambda x: x

        t = torch.zeros(batch_size, dtype=sample.dtype, device=sample.device)

        losses = []
        self.train()
        for i in verbose(range(training_steps)):
            indices = torch.randint(0, len(sample), (batch_size,))
            # [B, ...]
            estim = prior[indices] + self.forward(prior[indices], t)
            loss = loss_fn(sample[indices], estim)
            # update
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log
            losses.append(loss.detach().item())

        return losses
