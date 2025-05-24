from datetime import datetime, timezone, timedelta
from pathlib import Path

from ddpmpp import DDPMpp
from flowmodels import RectifiedFlow
from trainer import Cifar10Trainer


def reproduce_trigflow_cifar10():
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
        dropout=0.15,
        pe_scale=0.02,  # 16 on official repo
        use_shift_scale_norm=True,  # False on official repo
        use_double_norm=True,  # False on official repo
    )
    model = RectifiedFlow(backbone)

    n_gpus = 2
    n_grad_accum = 1
    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=512 // n_gpus // n_grad_accum,  # 128 on official repo
        lr=5e-4,  # 2e-4 on official repo
        betas=(0.9, 0.999),
        eps=1e-8,
        shuffle=True,
        dataset_path=Path("./"),
        workspace=Path(f"./test.workspace/rf-cifar10/{stamp}"),
    )

    trainer.train(
        total=1300000 * n_grad_accum,  # 1.3M on official repo
        mixed_precision="no",
        gradient_accumulation_steps=n_grad_accum,
        clip_grad_norm=1.0,
        _eval_interval=20 * n_grad_accum,
        _fid_steps=18,
    )


if __name__ == "__main__":
    reproduce_trigflow_cifar10()
