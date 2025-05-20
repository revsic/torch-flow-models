from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import torch
import torch.nn as nn
from tqdm import tqdm

from flowmodels.basis import SamplingSupports

from gm1d import TestGaussianMixture1D
from testutils import inherit_testbed, main


def _grid(
    x_range: tuple[int, int],
    x_samples: int,
    y_range: tuple[int, int],
    y_samples: int,
):
    xspace = np.linspace(*x_range, x_samples)
    yspace = np.linspace(*y_range, y_samples)
    stacked = np.stack([
        np.tile(xspace[None], [y_samples, 1]),
        np.tile(yspace[:, None], [1, x_samples]),
    ], axis=-1)
    return stacked.reshape(-1, 2)


def visualize_2d(
    model: SamplingSupports,
    dataset: tuple[torch.Tensor, torch.Tensor],
    name: str = "model",
    steps: list[int | None] = [100, 4, 1],
    verbose: Callable[[range], Iterable] | None = None,
    _sigma_data: float = 1.0,
    _save_fig: Path | None = None,
):
    gt, prior = dataset
    with torch.no_grad():
        for step in steps:
            x_t, x_ts = model.sample(prior, step, verbose)

            plt.figure()
            plt.scatter(*x_t.T, marker='x', label=f"{name}-{step}")
            plt.scatter(*gt.T, marker='x', label="gt", alpha=0.5)
            plt.legend()

            if _save_fig:
                plt.savefig(_save_fig / f"{name}-{step}.png")


class TestMoon2D(TestGaussianMixture1D):
    @classmethod
    def default_backbone(cls, dim: int = 1):
        return nn.Sequential(
            nn.Linear(2 + dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def dataset(self, size: int = 100000):
        points, _ = sklearn.datasets.make_moons(size, noise=0.005)
        return torch.tensor(points, dtype=torch.float32), lambda bs: torch.randn(bs, 2)

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
        return visualize_2d(
            self.model,
            (data, prior),
            name=name,
            steps=steps,
            verbose=lambda x: tqdm(x, leave=False),
            _sigma_data=_sigma_data,
            _save_fig=_save_fig
        )

    @torch.no_grad()
    def evaluate(
        self,
        steps: list[int | None] = [100, 4, 1],
        grid: np.ndarray = _grid([-1.5, 1.5], 200, [-1.5, 1.5], 200),
    ):
        return super().evaluate(steps, grid)


if __name__ == "__main__":
    main(
        Path("./results/moon"),
        inherit_testbed(TestMoon2D),
        using_stamp=True,
    )
