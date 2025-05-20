import json
from dataclasses import dataclass, field
from datetime import timedelta, timezone, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from flowmodels.utils import EMASupports
from testbed import TestGaussianMixture1D


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


@dataclass
class TestSuite:
    bed: TestGaussianMixture1D
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


class TestbedCAF(TestGaussianMixture1D):
    def __init__(self):
        super().__init__(
            ConstantAccelerationFlow(1, self.default_network())
        )  # pyright: ignore

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


class TestbedShortcutModel(TestGaussianMixture1D):
    class ShortcutWrapper(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            self.backbone = backbone

        def forward(
            self, x: torch.Tensor, t: torch.Tensor, d: torch.Tensor
        ) -> torch.Tensor:
            return self.backbone.forward(torch.cat([x, t[:, None], d[:, None]], dim=-1))

    @classmethod
    def default_network(cls):
        return cls.ShortcutWrapper(cls.default_backbone(dim=2))

    def __init__(self):
        super().__init__(ShortcutModel(self.default_network()))


class TestbedSCT(TestGaussianMixture1D):
    def __init__(self):
        super().__init__(
            ScaledContinuousCM(self.default_network(), ScaledContinuousCMScheduler())
        )

    def train(
        self,
        learning_rate: float = 0.001,
        train_steps: int = 1000,
        batch_size: int = 2048,
        mu: float = 0,
        ema: EMASupports | None = None,
    ):
        pretrainer = TestGaussianMixture1D(
            TrigFlow(self.model.F0, self.model.scheduler)
        )
        pretrainer.train(learning_rate, train_steps, batch_size)

        return super().train(learning_rate, train_steps, batch_size, mu, ema)


MODELS = {
    "caf": TestSuite(TestbedCAF()),
    "cfm": TestSuite(TestGaussianMixture1D.inherit(ConsistencyFlowMatching)),
    "ddpm": TestSuite(
        TestGaussianMixture1D.inherit(DDPM, DDIMScheduler(T=40)), steps=[40]
    ),
    "dsbm": TestSuite(
        TestGaussianMixture1D.inherit(DiffusionSchrodingerBridgeMatching)
    ),
    "ect": TestSuite(
        TestGaussianMixture1D.inherit(
            EasyConsistencyTraining, ECTScheduler(total_training_steps=1000)
        )
    ),
    "imm": TestSuite(
        TestGaussianMixture1D.inherit(InductivMomentMatching, IMMScheulder())
    ),
    "ncsn": TestSuite(
        TestGaussianMixture1D.inherit(NCSN, NCSNScheduler(T=10, R=100)),
        steps=[10 * 100],
    ),
    "trigflow": TestSuite(TestGaussianMixture1D.inherit(TrigFlow, ScaledContinuousCMScheduler())),
    "rf": TestSuite(TestGaussianMixture1D.inherit(RectifiedFlow)),
    "sct": TestSuite(TestbedSCT()),
    "shortcut": TestSuite(TestbedShortcutModel()),
    "vesde": TestSuite(TestGaussianMixture1D.inherit(VESDE, VESDEScheduler())),
    "vpsde": TestSuite(TestGaussianMixture1D.inherit(VPSDE, VPSDEScheduler())),
}


def main():
    PATH = Path("./test.gm1d")
    STAMP = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%d.KST%H:%M:%S")

    path = PATH / STAMP
    path.mkdir(exist_ok=True, parents=True)

    results = {}
    with tqdm(MODELS.items(), total=len(MODELS)) as pbar:
        for name, suite in pbar:
            pbar.set_description_str(name)
            p = path / name
            p.mkdir(exist_ok=True)
            result = suite.test(p)
            results[name] = result

    with open(path / "result.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
