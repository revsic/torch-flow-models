import sys
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import torch
import torch.nn as nn
from tqdm import tqdm

from flowmodels.basis import SamplingSupports

from eval1d import TestGaussianMixture1D
from testutils import inherit_testbed, main


def _grid(
    x_range: tuple[int, int],
    x_samples: int,
    y_range: tuple[int, int],
    y_samples: int,
):
    xspace = np.linspace(*x_range, x_samples)
    yspace = np.linspace(*y_range, y_samples)
    stacked = np.stack(
        [
            np.tile(xspace[None], [y_samples, 1]),
            np.tile(yspace[:, None], [1, x_samples]),
        ],
        axis=-1,
    )
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
            plt.scatter(*x_t.T, marker="x", label=f"{name}-{step}")
            plt.scatter(*gt[:500].T, marker="x", label="gt", alpha=0.5)
            plt.legend()

            if _save_fig:
                plt.savefig(_save_fig / f"{name}-{step}.png")


class TestMoon2D(TestGaussianMixture1D):
    GRIDS = _grid([-1.5, 1.5], 200, [-1.5, 1.5], 200)

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
            _save_fig=_save_fig,
        )

    @torch.no_grad()
    def evaluate(
        self,
        steps: list[int | None] = [100, 4, 1],
        grid: np.ndarray = GRIDS,
    ):
        _nx = np.sort(np.unique(TestMoon2D.GRIDS[:, 0]))
        _ny = np.sort(np.unique(TestMoon2D.GRIDS[:, 0]))
        dxdy = np.abs((_nx[1] - _nx[0]) * (_ny[1] - _ny[0]))
        return super().evaluate(steps, grid, dxdy)


class TestSCurve2D(TestMoon2D):
    GRIDS = _grid([-1.5, 1.5], 200, [-2.5, 2.5], 200)

    def dataset(self, size: int = 100000):
        points, _ = sklearn.datasets.make_s_curve(size, noise=0.05, random_state=0)
        return torch.tensor(
            points[:, [0, 2]], dtype=torch.float32
        ), lambda bs: torch.randn(bs, 2)


class TestSwissRoll2D(TestMoon2D):
    GRIDS = _grid([-1.5, 1.5], 200, [-1.5, 1.5], 200)

    def dataset(self, size: int = 100000):
        points, _ = sklearn.datasets.make_swiss_roll(size, noise=0.05, random_state=0)
        return torch.tensor(
            points[:, [0, 2]] / 10, dtype=torch.float32
        ), lambda bs: torch.randn(bs, 2)


class TestCircles2D(TestMoon2D):
    GRIDS = _grid([-1.5, 1.5], 200, [-1.5, 1.5], 200)

    def dataset(self, size: int = 100000):
        points, _ = sklearn.datasets.make_circles(size, noise=0.05, random_state=0)
        return torch.tensor(points, dtype=torch.float32), lambda bs: torch.randn(bs, 2)


class TestCheckerboard(TestMoon2D):
    GRIDS = _grid([-2.5, 2.5], 200, [-2.5, 2.5], 200)

    def dataset(self, size: int = 100000):
        grid = np.stack(
            [
                np.tile(np.arange(4)[None], [4, 1]),
                np.tile(np.arange(4)[:, None], [1, 4]),
            ],
            axis=-1,
        ).reshape(-1, 2)
        grid = grid[grid.sum(axis=-1) % 2 == 1]
        per_grid, residue = divmod(size, len(grid))
        assert residue == 0

        samples = np.random.rand(size, 2)
        samples = samples + np.tile(grid[:, None], (1, per_grid, 1)).reshape(-1, 2)
        np.random.shuffle(samples)
        return torch.tensor(samples - 2.0, dtype=torch.float32), lambda bs: torch.randn(
            bs, 2
        )


EVAL2D_TESTBEDS = {
    "moon": TestMoon2D,
    "scurve": TestSCurve2D,
    "swissroll": TestSwissRoll2D,
    "circles": TestCircles2D,
    "checkerboard": TestCheckerboard,
}


if __name__ == "__main__":
    bed = "moon"
    if len(sys.argv) > 1:
        bed = sys.argv[1]

    beds = [bed]
    if bed == "all":
        beds = list(EVAL2D_TESTBEDS)

    with tqdm(beds) as pbar:
        for bed in pbar:
            pbar.set_description_str(bed)
            main(
                Path(f"./results/{bed}"),
                inherit_testbed(EVAL2D_TESTBEDS[bed]),
                using_stamp=True,
                leave=False,
            )
