from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from flowmodels.basis import SamplingSupports


def gmm(
    size: int = 100000,
    weights: list[float] = [0.3, 0.7],
    mean: list[float] = [-0.5, 1.0],
    std: list[float] = [0.1, 0.2],
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = weights / np.sum(weights)
    sizes = [int(size * w) for w in weights[:-1]]
    sizes.append(size - np.sum(sizes))
    points = torch.cat(
        [torch.randn(n, 1) * s + m for n, s, m in zip(sizes, std, mean)],
        dim=0,
    )
    labels = []
    for i, n in enumerate(sizes):
        labels.extend([i for _ in range(n)])
    return points, torch.tensor(labels)


def vis(
    model: SamplingSupports,
    prior: torch.Tensor = torch.randn(10000, 1),
    conditional: bool = False,
    dataset: tuple[torch.Tensor, torch.Tensor] = gmm(),
    name: str = "model",
    steps: list[int | None] = [100, 4, 1],
    histspace: torch.Tensor = torch.linspace(-5, 5, 200),
    trajspace: torch.Tensor = torch.linspace(-3, 3, 10)[:, None],
    verbose: Callable[[range], Iterable] | None = None,
    _sigma_data: float = 1.0,
):
    X, Y = dataset
    perm = torch.randperm(len(X))[: len(prior)]

    testlabel = None
    if conditional:
        testlabel = Y[perm]
    with torch.no_grad():
        x_ts = [
            x_t
            for step in steps
            for x_t, _ in (model.sample(prior, testlabel, step, verbose),)
        ]
    plt.figure()
    plt.hist(prior, bins=histspace, label="prior")  # pyright: ignore
    plt.hist(X[perm], bins=histspace, label="gt", alpha=0.7)  # pyright: ignore
    for step, x_t in zip(steps, x_ts):
        plt.hist(
            x_t.view(-1),
            bins=histspace,  # pyright: ignore
            label=f"{name}-{step}",
            alpha=0.5,
        )
    plt.legend()
    _xticks, _ = plt.xticks()

    if conditional:
        plt.figure()
        plt.title("Conditional")
        for l in torch.unique(Y):
            plt.hist(
                X[perm][Y[perm] == l],
                bins=histspace,  # pyright: ignore
                label=f"gt-m{l.item()}",
                alpha=0.7,
            )
            for step, x_t in zip(steps, x_ts):
                plt.hist(
                    x_t[Y[perm] == l],
                    bins=histspace,  # pyright: ignore
                    label=f"{name}-{step}-m{l.item()}",
                    alpha=0.5,
                )
        plt.legend()
        plt.xticks(_xticks)  # pyright: ignore

    # plot trajectory
    if conditional:
        u = torch.unique(Y)
        if len(u) >= len(trajspace):
            testlabel = u[: len(trajspace)]
        else:
            testlabel = torch.cat(
                [u, testlabel[: len(trajspace) - len(u)]], dim=0  # pyright: ignore
            )
    with torch.no_grad():
        x_trajs = [
            x_ts
            for step in steps
            for _, x_ts in (model.sample(trajspace, testlabel, step, verbose),)
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


class Backbone(nn.Module):
    def __init__(
        self,
        dim: int = 1,
        aux: int = 1,
        layers: int = 5,
        hiddens: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + aux, hiddens),
            *[
                submodule
                for _ in range(layers - 2)
                for submodule in (nn.ReLU(), nn.Linear(hiddens, hiddens))
            ],
            nn.ReLU(),
            nn.Linear(hiddens, dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is not None:
            if t.dim() == 1:
                t = t[:, None]
            x_t = torch.cat([x_t, t.to(x_t)], dim=-1)
        if label is not None:
            if label.dim() == 1:
                label = label[:, None]
            x_t = torch.cat([x_t, label.to(x_t)], dim=-1)
        return self.net.forward(x_t)
