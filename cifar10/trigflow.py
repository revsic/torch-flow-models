from datetime import datetime, timezone, timedelta
from pathlib import Path

from ddpmpp import DDPMpp
from flowmodels.sct import TrigFlow, ScaledContinuousCMScheduler
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
        dropout=0.13,
        pe_scale=0.02,
        use_shift_scale_norm=True,
        use_double_norm=True,
    )
    model = TrigFlow(
        backbone,
        ScaledContinuousCMScheduler(
            p_mean=-1.0,
            p_std=1.4,
        ),
    )

    n_gpus = 2
    n_grad_accum = 1
    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=512 // n_gpus // n_grad_accum,
        lr=0.0005,  # paper: 0.001 (diverge)
        betas=(0.9, 0.999),
        eps=1e-8,
        shuffle=True,
        dataset_path=Path("./"),
        workspace=Path(f"./test.workspace/trigflow-cifar10/{stamp}"),
    )

    trainer.train(
        total=400000 * n_grad_accum,
        mixed_precision="no",
        gradient_accumulation_steps=n_grad_accum,
        _eval_interval=20 * n_grad_accum,
        _fid_steps=18,
    )


if __name__ == "__main__":
    reproduce_trigflow_cifar10()
