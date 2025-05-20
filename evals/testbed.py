import traceback
from pathlib import Path
from typing import Callable, Iterable, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from tqdm.auto import tqdm

from flowmodels.basis import SamplingSupports
from flowmodels.utils import EMASupports


def visualize_1d(
    model: SamplingSupports,
    dataset: tuple[torch.Tensor, torch.Tensor],
    trajspace: torch.Tensor,
    name: str = "model",
    steps: list[int | None] = [100, 4, 1],
    histspace: torch.Tensor = torch.linspace(-3, 3, 200),
    verbose: Callable[[range], Iterable] | None = None,
    _sigma_data: float = 1.0,
    _save_fig: Path | None = None,
):
    gt, prior = dataset
    with torch.no_grad():
        x_ts = [
            x_t for step in steps for x_t, _ in (model.sample(prior, step, verbose),)
        ]
    plt.figure()
    plt.hist(prior, bins=histspace, label="prior")  # pyright: ignore
    plt.hist(gt, bins=histspace, label="gt", alpha=0.7)  # pyright: ignore
    for step, x_t in zip(steps, x_ts):
        plt.hist(
            x_t.view(-1),
            bins=histspace,  # pyright: ignore
            label=f"{name}-{step}",
            alpha=0.5,
        )
    plt.legend()
    _xticks, _ = plt.xticks()
    if _save_fig:
        plt.savefig(_save_fig / "hist.png")

    # plot trajectory
    with torch.no_grad():
        x_trajs = [
            x_ts
            for step in steps
            for _, x_ts in (model.sample(trajspace, step, verbose),)
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
    if _save_fig:
        plt.savefig(_save_fig / "traj.png")


def kde_stats(
    data: np.ndarray,
    sample: np.ndarray,
    grid: np.ndarray,
    dxdy: float | None = None,
    max_num: int = 1000,
    bandwidth: float = 0.2,
):
    kde_data = KernelDensity(bandwidth=bandwidth).fit(data[:max_num])
    kde_sample = KernelDensity(bandwidth=bandwidth).fit(sample[:max_num])

    logp = kde_data.score_samples(grid)
    logq = kde_sample.score_samples(grid)

    if dxdy is None:
        # assume that the grid is uniformly distributed
        dxdy = np.abs(np.prod(grid[1] - grid[0])).item()

    kld = lambda logp, logq: np.sum(np.exp(logp) * (logp - logq)).item() * dxdy
    logm = np.log((np.exp(logp) + np.exp(logq)) * 0.5 + 1e-10)
    return {
        "nll": -kde_data.score_samples(sample).mean(),
        "kld-fwd": kld(logp, logq),
        "kld-bwd": kld(logq, logp),
        "jsd": 0.5 * (kld(logp, logm) + kld(logq, logm)),
    }


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


class _MScoreModel(nn.Module, SamplingSupports):
    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class _MODEModel(nn.Module, SamplingSupports):
    def loss(
        self,
        sample: torch.Tensor,
        t: torch.Tensor | None = None,
        src: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


type Testable = _MScoreModel | _MODEModel


class Testbed[T: Testable](BaseTestbed):
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


class TestGaussianMixture1D(Testbed):
    class ConcatenationWrapper(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            self.backbone = backbone

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.backbone.forward(torch.cat([x, t[:, None]], dim=-1))

    @classmethod
    def default_backbone(cls, dim: int = 1):
        return nn.Sequential(
            nn.Linear(1 + dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @classmethod
    def default_network(cls) -> nn.Module:
        return cls.ConcatenationWrapper(cls.default_backbone())

    def dataset(
        self, size: int = 100000
    ) -> tuple[torch.Tensor, Callable[[int], torch.Tensor]]:
        mixture = D.MixtureSameFamily(
            D.Categorical(torch.tensor([0.3, 0.7])),
            D.Normal(torch.tensor([-0.5, 1.0]), torch.tensor([0.1, 0.2])),
        )
        # target distribution
        return mixture.sample((size, 1)), lambda bs: torch.randn(bs, 1)

    def visualize(
        self,
        name: str = "model",
        steps: list[int | None] = [100, 4, 1],
        _sigma_data: float = 1.0,
        _save_fig: Path | None = None,
    ):
        data, prior = self.dataset()
        if callable(prior):
            prior = prior(len(data))
        return visualize_1d(
            self.model,
            (data, prior),
            torch.linspace(-3, 3, 10)[:, None],
            name,
            steps,
            verbose=lambda x: tqdm(x, leave=False),
            _sigma_data=_sigma_data,
            _save_fig=_save_fig,
        )

    @torch.no_grad()
    def evaluate(self, steps: list[int | None] = [100, 4, 1]) -> dict:
        data, prior = self.dataset()
        if callable(prior):
            prior = prior(len(data))

        results = {}
        grid = np.linspace(-3, 3, 200)[:, None]
        for step in steps:
            try:
                r, _ = self.model.sample(prior, step, lambda x: tqdm(x, leave=False))
                results[step] = kde_stats(data.numpy(), r.numpy(), grid)
            except:
                results[step] = traceback.format_exc()
        return results
