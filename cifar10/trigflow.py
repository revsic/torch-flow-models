from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch

from flowmodels.sct import TrigFlow, ScaledContinuousCMScheduler
from nvidiaedm import SongUNet
from trainer import Cifar10Trainer


def reproduce_trigflow_cifar10():
    # model definition
    backbone = SongUNet(
        input_size=32,
        in_channels=3,
        out_channels=3,
        num_classes=0,
        augment_dim=0,
        model_channels=128,
        channel_mult=[2, 2, 2],  # [1, 2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.2,  # 0.13,
        label_dropout=0,
        embedding_type="positional",
        channel_mult_noise=1,
        encoder_type="standard",
        decoder_type="standard",
        resample_filter=[1, 1]
    )
    model = TrigFlow(
        backbone,
        ScaledContinuousCMScheduler(
            sigma_d=1.0,
            p_mean=-1.0,
            p_std=1.4,
        ),
    )

    n_gpus = 1
    n_grad_accum = 2
    # timestamp
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y.%m.%dKST%H:%M:%S")
    trainer = Cifar10Trainer(
        model,
        batch_size=768 // n_gpus // n_grad_accum,
        lr=0.0002,  # paper: 0.001 (diverge)
        betas=(0.9, 0.95),  # paper: (0.9, 0.999)
        eps=1e-8,
        weight_decay=0.0,
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
