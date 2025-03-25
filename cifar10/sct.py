from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from safetensors.torch import load_file

from ddpmpp_cm import UNetModel
from flowmodels.sct import ScaledContinuousCM, ScaledContinuousCMScheduler
from trainer import Cifar10Trainer
from trigflow import InverseSquareRootScheduler


def reproduce_sct_cifar10():
    # model definition
    backbone = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=4,
        attention_resolutions=[16],
        dropout=0.20,
        channel_mult=[1, 2, 2, 2],
        conv_resample=True,  # Unknown
        num_heads=1,
        use_scale_shift_norm=True,
        use_double_norm=True,
        resblock_updown=False,
        temb_scale=0.02,
        attn_module="legacy",
        attn_dtype=torch.float16,
    )
    model = ScaledContinuousCM(
        backbone,
        ScaledContinuousCMScheduler(),
        tangent_warmup=10000,
    )
    state_dict = load_file(
        # "./test.workspace/trigflow-cifar10/2025.03.15KST22:04:04/ckpt/2040/model.safetensors",
        "./test.workspace/sct-cifar10/2025.03.19KST10:09:10/ckpt/0/model.safetensors"
    )
    # state_dict.pop("_ada_weight.weight")
    model.load_state_dict(state_dict, strict=False)

    n_gpus = 1
    n_grad_accum = 2
    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=512 // n_gpus // n_grad_accum,
        shuffle=True,
        dataset_path=Path("./"),
        workspace=Path(f"./test.workspace/sct-cifar10/{stamp}"),
    )
    trainer.optim = torch.optim.RAdam(
        trainer.model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    trainer.scheduler = InverseSquareRootScheduler(
        trainer.optim,
        0.0001,
        t_ref=4000,
    )

    trainer.train(
        total=400000 * n_grad_accum,
        mixed_precision="no",
        gradient_accumulation_steps=n_grad_accum,
        half_life_ema=500000,
    )


if __name__ == "__main__":
    reproduce_sct_cifar10()
