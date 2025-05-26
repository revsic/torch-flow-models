from dataclasses import dataclass, field
from datetime import timedelta, timezone, datetime
from pathlib import Path
from typing import Any, Callable, Type

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from tqdm.auto import tqdm

from flowmodels import (
    ConstantAccelerationFlow,
    ConsistencyFlowMatching,
    DDPM,
    DDIMScheduler,
    DiffusionSchrodingerBridgeMatching,
    EasyConsistencyTraining,
    ECTScheduler,
    InductivMomentMatching,
    IMMScheulder,
    NCSN,
    NCSNScheduler,
    RectifiedFlow,
    ScaledContinuousCM,
    ScaledContinuousCMScheduler,
    ShortcutModel,
    VESDE,
    VESDEScheduler,
    VPSDE,
    VPSDEScheduler,
)
from flowmodels.sct import TrigFlow
from flowmodels.utils import EMASupports

from testbed import Testable, Testbed


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


@dataclass
class TestSuite:
    bed: Testbed
    steps: list[int | None] = field(default_factory=lambda: [100, 4, 1])

    def test(self, figpath: Path | None = None):
        losses = self.bed.train()
        plt.figure()
        plt.plot(losses)
        plt.title("training loss")
        if figpath:
            plt.savefig(figpath / "loss.png")

        self.bed.visualize(steps=self.steps, _save_fig=figpath)
        result = self.bed.evaluate(steps=self.steps)
        if figpath:
            with open(figpath / "eval.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return result


def main(
    path: Path,
    models: dict[str, TestSuite],
    using_stamp: bool = False,
    leave: bool = True,
):
    if using_stamp:
        stamp = datetime.now(timezone(timedelta(hours=9))).strftime(
            "%Y.%m.%d.KST%H:%M:%S"
        )
        path = path / stamp

    path.mkdir(exist_ok=True, parents=True)

    results = {}
    with tqdm(models.items(), total=len(models), leave=leave) as pbar:
        for name, suite in pbar:
            pbar.set_description_str(name)
            p = path / name
            p.mkdir(exist_ok=True)
            result = suite.test(p)
            results[name] = result

    with open(path / "result.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def default_factory(
    TestableModel: Type[Testable],
    *args,
    steps: list[int | None] = [100, 4, 1],
    **kwargs,
):
    def inner(GivenTestbed: Type[Testbed[Testable]]):
        network = GivenTestbed.default_network()
        model = TestableModel(network, *args, **kwargs)
        return TestSuite(GivenTestbed(model), steps=steps)

    return inner


FACTORIES: dict[str, Callable[[Type[Testbed[Testable]]], TestSuite]] = {
    "cfm": default_factory(ConsistencyFlowMatching),
    "ddpm": default_factory(DDPM, DDIMScheduler(T=40), steps=[40]),
    "dsbm": default_factory(DiffusionSchrodingerBridgeMatching),
    "ect": default_factory(
        EasyConsistencyTraining, ECTScheduler(total_training_steps=1000)
    ),
    "imm": default_factory(InductivMomentMatching, IMMScheulder()),
    "ncsn": default_factory(NCSN, NCSNScheduler(T=10, R=100), steps=[10 * 100]),
    "trigflow": default_factory(TrigFlow, ScaledContinuousCMScheduler()),
    "rf": default_factory(RectifiedFlow),
    "vesde": default_factory(VESDE, VESDEScheduler()),
    "vpsde": default_factory(VPSDE, VPSDEScheduler()),
    # custom factory: caf, shortcut-model, sct
}


def register_factory(
    name: str, kwargs: dict[str, Any] = {}, steps: list[int | None] = [100, 4, 1]
):
    def wrapper(func: Callable[[Type[Testbed[Testable]]], Type[Testbed[Testable]]]):
        def wrapped(GivenTestbed: Type[Testbed[Testable]]) -> TestSuite:
            Inherited = func(GivenTestbed)
            return TestSuite(Inherited(**kwargs), steps)

        global FACTORIES
        FACTORIES[name] = wrapped
        return func

    return wrapper


def inherit_testbed[T: Testbed](GivenTestbed: Type[T]):
    return {name: factory(GivenTestbed) for name, factory in FACTORIES.items()}


FACTORY_TESTBED_CAF_ARGS = dict(channels=1)


@register_factory(name="caf", kwargs=FACTORY_TESTBED_CAF_ARGS)
def factory_testbed_caf[T: Testbed](GivenTestbed: Type[T]):
    class TestbedCAF(GivenTestbed):
        def __init__(self, channels: int):
            super().__init__(ConstantAccelerationFlow(channels, self.default_network()))

        def train_loop(
            self,
            optim: torch.optim.Optimizer,
            train_steps: int = 1000,
            batch_size: int = 2048,
            mu: float = 0,
            ema: EMASupports | None = None,
        ):
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
                    v_loss, _ = self.loss(sample, self.sample_t(batch_size), src)
                    # update
                    optim.zero_grad()
                    v_loss.backward()
                    optim.step()
                    # log
                    v_loss = v_loss.detach().item()
                    losses.append(v_loss)
                    pbar.set_postfix_str(f"v_loss: {v_loss:.2f}")

            with tqdm(range(train_steps), leave=False) as pbar:
                for i in pbar:
                    indices = torch.randint(0, len(data), (batch_size,))
                    sample = data[indices]
                    if callable(prior):
                        src = prior(batch_size)
                    else:
                        src = prior[indices]
                    # []
                    _, a_loss = self.loss(sample, self.sample_t(batch_size), src)
                    # update
                    optim.zero_grad()
                    a_loss.backward()
                    optim.step()
                    # log
                    a_loss = a_loss.detach().item()
                    losses.append(a_loss)
                    pbar.set_postfix_str(f"a_loss: {a_loss:.2f}")

            return losses

    return TestbedCAF


@register_factory(name="shortcut")
def factory_testbed_shortcut_model[T: Testbed](GivenTestbed: Type[T]):
    class TestbedShortcutModel(GivenTestbed):
        class ShortcutWrapper(nn.Module):
            def __init__(self, backbone: nn.Module):
                super().__init__()
                self.backbone = backbone

            def forward(
                self, x: torch.Tensor, t: torch.Tensor, d: torch.Tensor
            ) -> torch.Tensor:
                return self.backbone.forward(
                    torch.cat([x, t[:, None], d[:, None]], dim=-1)
                )

        @classmethod
        def default_network(cls):
            return cls.ShortcutWrapper(cls.default_backbone(dim=2))  # pyright: ignore

        def __init__(self):
            super().__init__(ShortcutModel(self.default_network()))

    return TestbedShortcutModel


@register_factory(name="sct", steps=[4, 2, 1])
def factory_testbed_sct[T: Testbed](GivenTestbed: Type[T]):
    class TestbedSCT(GivenTestbed):
        def __init__(self):
            super().__init__(
                ScaledContinuousCM(
                    self.default_network(), ScaledContinuousCMScheduler()
                )
            )

        def train(
            self,
            learning_rate: float = 0.001,
            train_steps: int = 1000,
            batch_size: int = 2048,
            mu: float = 0,
            ema: EMASupports | None = None,
        ):
            pretrainer = GivenTestbed(TrigFlow(self.model.F0, self.model.scheduler))
            pretrainer.train(learning_rate, train_steps, batch_size)

            return super().train(learning_rate, train_steps, batch_size, mu, ema)

    return TestbedSCT
