import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from nvidiaedm import SongUNet
from flowmodels import RectifiedFlow
from trainer import Cifar10Trainer, TrainConfig


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)  # pyright: ignore

def reproduce_rectifiedflow():
    config = Config(
        train=TrainConfig(
            n_gpus=1,
            n_grad_accum=2,
            mixed_precision="no",
            batch_size=1024,
            n_classes=10 + 1,
            lr=2e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
            clip_grad_norm=None,
            nan_to_num=None,
            total=400000,
            label_dropout=0.1,
            uncond_label=10,
            fid_steps=18,
        ),
    )
    # model definition
    backbone = SongUNet(num_classes=10)
    model = RectifiedFlow(backbone)

    # timestamp
    workspace = Path(f"./test.workspace/rf-cifar10-cond/{config.train.stamp}")
    workspace.mkdir(parents=True, exist_ok=True)
    with open(workspace / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    config.train.workspace = workspace.as_posix()
    trainer = Cifar10Trainer(model, config.train)
    trainer.optim = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=config.train.lr,
        betas=(config.train.beta1, config.train.beta2),
        eps=config.train.eps,
        weight_decay=config.train.weight_decay,
    )
    try:
        trainer.train()
    except:
        with open(workspace / "exception", "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    reproduce_rectifiedflow()
