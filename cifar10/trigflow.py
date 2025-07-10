import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ddpmpp import DDPMpp, ModelConfig
from flowmodels.sct import TrigFlow, ScaledContinuousCMScheduler
from trainer import Cifar10Trainer, TrainConfig


@dataclass
class Config:
    p_mean: float = -1.0
    p_std: float = 1.4
    model: ModelConfig = field(default_factory=ModelConfig)  # pyright: ignore
    train: TrainConfig = field(default_factory=TrainConfig)  # pyright: ignore


def reproduce_trigflow_cifar10():
    config = Config(
        model=ModelConfig(
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
            n_classes=10 + 1,  # +1 for uncond
        ),
        train=TrainConfig(
            n_gpus=2,
            n_grad_accum=2,
            mixed_precision="no",
            batch_size=1024,
            n_classes=10 + 1,
            lr=0.001,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.0,
            clip_grad_norm=None,
            nan_to_num=None,
            total=400000,
            half_life_ema=500000,
            label_dropout=0.1,
            uncond_label=10,
            fid_steps=18,
        ),
    )

    # model definition
    backbone = DDPMpp(config.model)
    model = TrigFlow(
        backbone,
        ScaledContinuousCMScheduler(p_mean=config.p_mean, p_std=config.p_std),
    )

    # timestamp
    workspace = Path(f"./test.workspace/trigflow-cifar10-cond/{config.train.stamp}")
    workspace.mkdir(parents=True, exist_ok=True)
    with open(workspace / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    config.train.workspace = workspace.as_posix()
    trainer = Cifar10Trainer(model, config.train)
    try:
        trainer.train()
    except:
        with open(workspace / "exception", "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    reproduce_trigflow_cifar10()
