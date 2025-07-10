from pathlib import Path
from typing import Callable, Generic, Type, TypeVar

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from flowmodels.utils import EMASupports


class BaseTestbed:
    def dataset(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | Callable[[int], torch.Tensor]]:
        raise NotImplementedError("BaseTestbed.dataset is not implemented")

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        ema: EMASupports | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("BaseTestbed.loss is not implemented")

    def visualize(
        self,
        name: str = "model",
        steps: list[int | None] = [100, 4, 1],
        _sigma_data: float = 1.0,
        _save_fig: Path | None = None,
    ):
        raise NotImplementedError("BaseTestbed.visualize is not implemented")

    def evaluate(self, steps: list[int | None] = [100, 4, 1]) -> dict:
        raise NotImplementedError("BaseTestbed.evaluate is not implemented")

    def ema_update(self, ema: EMASupports, mu: float):
        raise NotImplementedError("BaseTestbed.ema_update is not implemented")

    def sample_t(self, batch_size: int):
        return None

    def train_loop(
        self,
        optim: torch.optim.Optimizer,
        train_steps: int = 1000,
        batch_size: int = 2048,
        mu: float = 0.0,
        ema: EMASupports | None = None,
    ) -> list[float]:
        losses = []
        data, prior = self.dataset()
        with tqdm(range(train_steps), leave=False) as pbar:
            for i in pbar:
                indices = torch.randint(0, len(data), (batch_size,))
                sample = data[indices]
                if callable(prior):
                    src = prior(batch_size)
                else:
                    src = prior[indices]
                # []
                loss = self.loss(sample, self.sample_t(batch_size), src, ema)
                # update
                optim.zero_grad()
                loss.backward()
                optim.step()

                if ema is not None:
                    self.ema_update(ema, mu)

                # log
                loss = loss.detach().item()
                losses.append(loss)
                pbar.set_postfix_str(f"loss: {loss:.2f}")

        return losses


Testable = nn.Module

T = TypeVar("T", bound=Testable)


class Testbed(BaseTestbed, Generic[T]):
    @classmethod
    def default_network(cls) -> nn.Module:
        raise NotImplementedError("Testbed.default_network is not implemented")

    def __init__(self, model: T):
        self.model = model

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        ema: EMASupports | None = None,
    ):
        if ema is None:
            return self.model.loss(sample, t, prior)  # pyright: ignore
        return self.model.loss(sample, t, prior, ema)  # pyright: ignore

    def ema_update(self, ema: EMASupports, mu: float):
        ema.update(self.model, mu)

    def train(
        self,
        learning_rate: float = 0.001,
        train_steps: int = 1000,
        batch_size: int = 2048,
        mu: float = 0.0,
        ema: EMASupports | None = None,
    ) -> list[float]:
        return self.train_loop(
            torch.optim.Adam(self.model.parameters(), learning_rate),
            train_steps,
            batch_size,
            mu,
            ema,
        )

    @classmethod
    def inherit(cls, Super: Type[T], *args, **kwargs):
        return cls(Super(cls.default_network(), *args, **kwargs))
