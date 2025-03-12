from datetime import datetime, timezone, timedelta
from pathlib import Path

from ddpmpp_cm import UNetModel
from flowmodels.sct import TrigFlow, ScaledContinuousCMScheduler
from trainer import Cifar10Trainer


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
    model = TrigFlow(backbone, ScaledContinuousCMScheduler())

    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=256,  # => 512-sized batch on 2 3090 GPUs
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
