from typing import Callable, Iterable, Iterator, Protocol, Self, Type, TypeVar

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn

from flowmodels.basis import SamplingSupports
from flowmodels.utils import EMASupports


class _Share(SamplingSupports, Protocol):
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


class _Loss1(_Share, Protocol):
    def loss(
        self, sample: torch.Tensor, t: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor: ...


class _Loss2(_Share, Protocol):
    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        src: torch.Tensor,
        ema: EMASupports,
    ) -> torch.Tensor: ...


class _Loss3(_Share, Protocol):
    def loss(
        self, sample: torch.Tensor, t: torch.Tensor, prior: torch.Tensor
    ) -> torch.Tensor: ...


class _Loss4(_Share, Protocol):
    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        prior: torch.Tensor,
        ema: EMASupports,
    ) -> torch.Tensor: ...


_InheritanceSupports = _Loss1 | _Loss2 | _Loss3 | _Loss4

S = TypeVar("S", bound=_InheritanceSupports)


class Testbed(nn.Module, SamplingSupports):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    @classmethod
    def dataset(cls):
        mixture = D.MixtureSameFamily(
            D.Categorical(torch.tensor([0.3, 0.7])),
            D.Normal(torch.tensor([-0.5, 1.0]), torch.tensor([0.1, 0.2])),
        )
        # target distribution
        return mixture.sample((100000, 1))

    @classmethod
    def inherit(
        cls, Super: Type[S], dim: int = 1, retn_class: bool = False, *args, **kwargs
    ):
        class Inherited(cls):
            def __init__(self, dim: int = 1, *args, **kwargs):
                super().__init__(dim)
                self._super = (Super(self, *args, **kwargs),)  # pyright: ignore

            @property
            def base(self) -> S:
                (_super,) = self._super
                return _super

            def parameters(self, recurse=True) -> Iterator[nn.Parameter]:
                return self.base.parameters(recurse)

            def loss(
                self,
                sample: torch.Tensor,
                t: torch.Tensor,
                src: torch.Tensor,
                ema: EMASupports | None = None,
            ) -> torch.Tensor:
                if ema is None:
                    return self.base.loss(sample, t, src)  # pyright: ignore
                return self.base.loss(sample, t, src, ema)  # pyright: ignore

            def sample(
                self,
                prior: torch.Tensor,
                steps: int | None = None,
                verbose: Callable[[range], Iterable] | None = None,
            ):
                return self.base.sample(prior, steps, verbose)

        if retn_class:
            return Inherited
        return Inherited(dim, *args, **kwargs)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net.forward(torch.cat([x_t, t[:, None].to(x_t)], dim=-1))

    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor,
        src: torch.Tensor,
        ema: EMASupports[Self] | None = None,
    ) -> torch.Tensor: ...

    def train_loop(
        self,
        dataset: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        lr: float = 0.001,
        train_steps: int = 1000,
        batch_size: int = 2048,
        optim: Type[torch.optim.Optimizer] | torch.optim.Optimizer = torch.optim.Adam,
        mu: float = 0.0,
        ema: EMASupports[Self] | None = None,
    ) -> list[float]:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = lambda x: x
        # train
        self.train()
        if isinstance(optim, type):
            optim = optim(self.parameters(), lr)  # pyright: ignore

        _length = len(dataset)
        if isinstance(dataset, tuple):
            _data, _prior = dataset
            _length = len(_data)

        losses = []
        with tqdm(range(train_steps)) as pbar:
            for i in pbar:
                indices = torch.randint(0, _length, (batch_size,))
                if isinstance(dataset, torch.Tensor):
                    sample = dataset[indices]
                    prior = torch.randn_like(sample)
                else:
                    sample, prior = _data[indices], _prior[indices]  # pyright: ignore
                # []
                loss = self.loss(sample, torch.rand(batch_size), prior, ema)
                # update
                optim.zero_grad()
                loss.backward()
                optim.step()

                if ema is not None:
                    ema.update(self, mu=mu)

                # log
                loss = loss.detach().item()
                losses.append(loss)
                pbar.set_postfix_str(f"loss: {loss:.2f}")

        return losses

    def visualize(
        self,
        name: str,
        losses: list[float],
        gt: torch.Tensor,
        prior: torch.Tensor | None = None,
        steps: list[int] = [100, 4, 1],
        n: int = 10000,
        histspace: torch.Tensor = torch.linspace(-3, 3, 200),
        trajspace: torch.Tensor = torch.linspace(-3, 3, 10)[:, None],
        verbose: Callable[[range], Iterable] | None = None,
        _sigma_data: float = 1.0,
    ):
        import matplotlib.pyplot as plt

        plt.plot(losses)

        # plot histogram
        if prior is None:
            prior = torch.randn(n, *gt.shape[1:])
        with torch.no_grad():
            x_ts = [
                x_t for step in steps for x_t, _ in (self.sample(prior, step, verbose),)
            ]
        plt.figure()
        plt.hist(prior, bins=histspace, label="prior")  # pyright: ignore
        plt.hist(gt[:n], bins=histspace, label="gt", alpha=0.7)  # pyright: ignore
        for step, x_t in zip(steps, x_ts):
            plt.hist(
                x_t.view(-1),
                bins=histspace,  # pyright: ignore
                label=f"{name}-{step}",
                alpha=0.5,
            )
        plt.legend()
        _xticks, _ = plt.xticks()

        # plt.trajectory
        with torch.no_grad():
            x_trajs = [
                x_ts
                for step in steps
                for _, x_ts in (self.sample(trajspace, step, verbose),)
            ]
        plt.figure()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i in range(len(trajspace)):
            for j, (step, x_ts) in enumerate(zip(steps, x_trajs)):
                plt.plot(
                    torch.tensor(
                        [_sigma_data * trajspace[i].item()]
                        + [x_t[i].item() for x_t in x_ts]
                    ),
                    np.linspace(0, 1, len(x_ts) + 1),
                    colors[j % len(colors)],
                    **({} if i > 0 else {"label": f"{name}-{step}"}),  # pyright: ignore
                    alpha=0.5,
                )
                plt.xticks(_xticks)  # pyright: ignore
        plt.legend()
