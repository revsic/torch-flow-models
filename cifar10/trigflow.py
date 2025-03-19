from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch

from ddpmpp_cm import UNetModel
from flowmodels.sct import TrigFlow, ScaledContinuousCMScheduler
from trainer import Cifar10Trainer


class InverseSquareRootScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        alpha_ref: float = 0.001,
        t_ref: int = 70000,
        last_epoch: int = -1,
    ):
        self.alpha_ref = alpha_ref
        self.t_ref = t_ref
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.alpha_ref / np.sqrt(max(self.last_epoch / self.t_ref, 1.0))
            for _ in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return self.get_lr()


def reproduce_trigflow_cifar10():
    # model definition
    backbone = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=4,
        attention_resolutions=[16],
        dropout=0.13,
        channel_mult=[1, 2, 2, 2],
        conv_resample=True,  # Unknown
        num_heads=1,
        use_scale_shift_norm=True,
        use_double_norm=True,
        resblock_updown=False,
        temb_scale=0.02,
    )
    model = TrigFlow(
        backbone,
        ScaledContinuousCMScheduler(
            p_mean=-1.0,
            p_std=1.4,
        ),
    )

    n_gpus = 2
    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=512 // n_gpus,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        shuffle=True,
        dataset_path=Path("./"),
        workspace=Path(f"./test.workspace/trigflow-cifar10/{stamp}"),
    )
    trainer.scheduler = InverseSquareRootScheduler(
        trainer.optim,
        0.001,
        t_ref=70000,
    )

    trainer.train(
        total=400000,
        mixed_precision="no",
        gradient_accumulation_steps=1,
    )


if __name__ == "__main__":
    reproduce_trigflow_cifar10()
