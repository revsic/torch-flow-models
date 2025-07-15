import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from safetensors.torch import load_model

from ddpmpp import DDPMpp, ModelConfig
from flowmodels.sct import ScaledContinuousCM, ScaledContinuousCMScheduler
from trainer import Cifar10Trainer, TrainConfig, _LossDDPWrapper


@dataclass
class Config:
    p_mean: float = -1.0
    p_std: float = 1.4
    tangent_warmup: int = 10000
    approx_jvp: bool = True
    dt: float = 0.005

    model: ModelConfig = field(default_factory=ModelConfig)  # pyright: ignore
    train: TrainConfig = field(default_factory=TrainConfig)  # pyright: ignore


def reproduce_sct_cifar10():
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
            dropout=0.20,
            pe_scale=0.02,
            use_shift_scale_norm=True,
            use_double_norm=True,
            n_classes=10 + 1,  # +1 for uncond
        ),
        train=TrainConfig(
            n_gpus=1,
            n_grad_accum=3,
            mixed_precision="no",
            batch_size=768,
            n_classes=10 + 1,
            lr=0.0001,
            beta1=0.9,
            beta2=0.99,
            eps=1e-8,
            weight_decay=0.0,
            clip_grad_norm=0.1,
            nan_to_num=0.0,
            total=400000,
            half_life_ema=500000,
            label_dropout=0.1,
            uncond_label=10,
            fid_steps=2,
        ),
    )
    # model definition
    backbone = DDPMpp(config.model)
    model = ScaledContinuousCM(
        backbone,
        ScaledContinuousCMScheduler(),
        tangent_warmup=config.tangent_warmup,
        _approx_jvp=config.approx_jvp,
        _dt=config.dt,
    )

    # timestamp
    workspace = Path(f"./test.workspace/sct-cifar10-cond/{config.train.stamp}")
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
    reproduce_sct_cifar10()
