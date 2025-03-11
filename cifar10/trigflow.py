from datetime import datetime, timezone, timedelta
from pathlib import Path

from ddpmpp import DDPMpp
from flowmodels.sct import TrigFlow, ScaledContinuousCMScheduler
from trainer import Cifar10Trainer


def reproduce_trigflow_cifar10():
    # model definition
    backbone = DDPMpp(resolution=32, in_channels=3, dropout=0.13)
    model = TrigFlow(backbone, ScaledContinuousCMScheduler())

    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=1,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        shuffle=True,
        dataset_path=Path("./"),
        workspace=Path(f"./test.workspace/trigflow-cifar10/{stamp}"),
    )

    trainer.train(total=400000, mixed_precision="no", gradient_accumulation_steps=1)


if __name__ == "__main__":
    reproduce_trigflow_cifar10()
