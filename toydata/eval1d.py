import traceback
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from tqdm.auto import tqdm

from flowmodels.basis import SamplingSupports

from testbed import Testbed
from testutils import inherit_testbed, kde_stats, main


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
    def evaluate(
        self,
        steps: list[int | None] = [100, 4, 1],
        grid: np.ndarray = np.linspace(-3, 3, 200)[:, None],
        dxdy: float | None = None,
    ) -> dict:
        data, prior = self.dataset()
        if callable(prior):
            prior = prior(len(data))

        results = {}
        for step in steps:
            try:
                r, _ = self.model.sample(prior, step, lambda x: tqdm(x, leave=False))
                results[step] = kde_stats(data.numpy(), r.numpy(), grid, dxdy)
            except:
                results[step] = traceback.format_exc()
        return results


if __name__ == "__main__":
    main(
        Path("./results/gm1d"),
        inherit_testbed(TestGaussianMixture1D),
        using_stamp=True,
    )
