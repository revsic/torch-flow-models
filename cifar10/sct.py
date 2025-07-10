from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from safetensors.torch import load_model

from ddpmpp import DDPMpp
from flowmodels.sct import ScaledContinuousCM, ScaledContinuousCMScheduler
from trainer import Cifar10Trainer, _LossDDPWrapper


def reproduce_sct_cifar10():
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
    model = ScaledContinuousCM(
        backbone,
        ScaledContinuousCMScheduler(),
        tangent_warmup=10000,
    )
    load_model(
        _LossDDPWrapper(model),
        "./test.workspace/trigflow-cifar10/2025.05.16KST21:12:35-lr5e4-fixproposal/ckpt/4081/model.safetensors",
    )

    n_gpus = 2
    n_grad_accum = 1
    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=225,  # 512 // n_gpus // n_grad_accum,
        shuffle=True,
        dataset_path=Path("./"),
        workspace=Path(f"./test.workspace/sct-cifar10/{stamp}"),
    )
    trainer.optim = torch.optim.RAdam(  # pyright: ignore
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
    reproduce_sct_cifar10()
