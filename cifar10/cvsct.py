from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from safetensors.torch import load_model

from ddpmpp import DDPMpp
from flowmodels import RectifiedFlow
from flowmodels.cvsct import ConstantVelocityConsistencyModels
from trainer import Cifar10Trainer, _LossDDPWrapper


def reproduce_cvsct_cifar10():
    # model definition
    backbone = DDPMpp(
        resolution=32,
        in_channels=3,
        nf=128,
        ch_mult=[1, 2, 2, 2],
        attn_resolutions=[16],
        num_res_blocks=4,
        init_scale=0.0,
        skip_rescale=True,
        dropout=0.20,
        pe_scale=0.02,
        use_shift_scale_norm=True,
        use_double_norm=True,
    )
    teacher = RectifiedFlow(backbone)
    load_model(
        _LossDDPWrapper(teacher),
        "./test.workspace/rf-cifar10/2025.05.24KST14:12:00/ckpt/2652/model.safetensors",
    )

    model = ConstantVelocityConsistencyModels(backbone)

    n_gpus = 1
    n_grad_accum = 2
    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=225, #  // n_gpus // n_grad_accum,
        shuffle=True,
        dataset_path=Path("./"),
        workspace=Path(f"./test.workspace/cvsct-cifar10/{stamp}"),
    )
    trainer.optim = torch.optim.RAdam(
        trainer.model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    trainer.train(
        total=400000 * n_grad_accum,
        mixed_precision="no",
        gradient_accumulation_steps=n_grad_accum,
        half_life_ema=500000,
        _eval_interval=20 * n_grad_accum,
        _fid_steps=2,
    )


if __name__ == "__main__":
    reproduce_cvsct_cifar10()
