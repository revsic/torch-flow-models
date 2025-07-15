from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Protocol

import git
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from safetensors.torch import load_model, save_model
from torch.utils.tensorboard import SummaryWriter  # pyright: ignore
from tqdm import tqdm

from fid import compute_fid_with_model
from flowmodels.basis import SamplingSupports, ScoreModel, ODEModel
from flowmodels.utils import EMASupports


class Loggable(Protocol):
    def log(self, msg: str): ...


class DefaultLogger(Loggable):
    def __init__(self):
        try:
            from loguru import logger  # pyright: ignore
        except ImportError:
            logger = None
        self.logger = logger

    def log(self, msg: str, level: str = "INFO"):
        if self.logger is not None:
            self.logger.log(level, msg)
        else:
            print(f"[{level}] {msg}")


class _LossDDPWrapper(nn.Module):
    def __init__(self, model: ODEModel | ScoreModel):
        super().__init__()
        assert isinstance(model, nn.Module)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.loss(*args, **kwargs)


def _revision() -> str | None:
    try:
        repo = git.Repo(search_parent_directories=True)
        branch = repo.active_branch.name
        hexsha = repo.head.object.hexsha
        return f"{branch}:{hexsha}"
    except Exception as e:
        print(f"[*] FAILED TO RETRIEVE REVISION: {e}")
        return None


def _stamp() -> str:
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST).strftime("%Y.%m.%dKST%H:%M:%S")


@dataclass
class TrainConfig:
    n_gpus: int
    n_grad_accum: int
    mixed_precision: str = "no"

    batch_size: int = 512
    n_classes: int | None = None

    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    clip_grad_norm: float | None = None
    nan_to_num: float | None = None

    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    dataset_path: str = "./cifar-10-batches-py/.."

    workspace: str = "./workspace"

    total: int = 400000

    start_step: int = 0
    load_ckpt: str | None = None
    load_ema_ckpt: str | None = None

    half_life_ema: int | None = None

    label_dropout: float = 0.0
    uncond_label: int | None = None

    num_samples: int = 10
    eval_interval: int = 20
    fid_steps: int | None = None

    revision: str | None = field(default_factory=_revision)
    stamp: str = field(default_factory=_stamp)


class Cifar10Trainer:
    def __init__(
        self,
        model: ScoreModel | ODEModel,
        config: TrainConfig,
        _logger: Loggable = DefaultLogger(),
    ):
        assert isinstance(model, nn.Module)
        self.model = model
        self.config = config

        self.trainset = torchvision.datasets.CIFAR10(
            config.dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(p=0.5),
                ]
            ),
        )
        self.testset = torchvision.datasets.CIFAR10(
            config.dataset_path,
            train=False,
            transform=transforms.ToTensor(),
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=config.batch_size // config.n_gpus // config.n_grad_accum,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=config.batch_size // config.n_gpus // config.n_grad_accum,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self.workspace = Path(config.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.train_log = SummaryWriter(self.workspace / "logs" / "train")
        self.test_log = SummaryWriter(self.workspace / "logs" / "test")
        self._logger = _logger

    def train(
        self,
        _preproc: Callable[[torch.Tensor], torch.Tensor] | None = None,
        _postproc: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        # validity check
        assert self.config.eval_interval > 0

        self.model.train()
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.n_grad_accum,
            log_with="tensorboard",
            project_dir=self.workspace / "logs",
        )
        _main_proc = accelerator.is_main_process
        if _main_proc:
            accelerator.init_trackers("train")

        model, optimizer, train_loader = accelerator.prepare(
            _LossDDPWrapper(self.model),
            self.optim,
            self.train_loader,
        )
        scheduler = None
        if self.scheduler is not None:
            scheduler = accelerator.prepare(self.scheduler)

        if self.config.load_ckpt is not None:
            torch.serialization.add_safe_globals(
                [
                    np._core.multiarray.scalar,  # pyright: ignore
                    np.dtypes.Float64DType,
                ]
            )
            accelerator.load_state(self.config.load_ckpt)

        ema, mu = None, 0.0
        if self.config.half_life_ema:
            ema = EMASupports(self.model)
            # compute ema factor
            steps = self.config.half_life_ema / self.config.batch_size
            mu = 0.5 ** (1 / steps)
            self._logger.log(
                f"Estimated mu: {mu:.4f}, and halved({mu ** int(steps):.2f}) in {int(steps)} steps"
            )

            if self.config.load_ema_ckpt:
                load_model(ema.module, self.config.load_ema_ckpt)

        # identity mapping
        if _preproc is None:
            _preproc = lambda x: x
        if _postproc is None:
            _postproc = lambda x: x

        if self.config.n_classes and self.config.uncond_label is None:
            self.config.uncond_label = self.config.n_classes + 1

        def _preproc_label(labels: torch.Tensor) -> torch.Tensor | None:
            if not self.config.n_classes:
                return None
            assert isinstance(self.config.uncond_label, int)
            if self.config.label_dropout > 0.0:
                mask = torch.rand(len(labels)) < self.config.label_dropout
                labels[mask] = self.config.uncond_label
            return labels

        step = self.config.start_step
        total = self.config.total * self.config.n_grad_accum
        epochs = -(-total // len(train_loader))
        epoch = 0
        for epoch in tqdm(range(epochs), disable=not _main_proc):
            with tqdm(train_loader, leave=False, disable=not _main_proc) as pbar:
                for sample, labels in pbar:
                    with accelerator.accumulate(model):
                        loss = model(
                            _preproc(sample * 2 - 1),
                            label=_preproc_label(labels),
                        )
                        accelerator.backward(loss)
                        if clip_grad_norm := self.config.clip_grad_norm:
                            accelerator.clip_grad_norm_(
                                model.parameters(), clip_grad_norm
                            )
                        # early collection
                        loss = loss.item()
                        # for numerical stability
                        if (ntn := self.config.nan_to_num) is not None:
                            for param in model.parameters():
                                if param.grad is not None:
                                    torch.nan_to_num_(
                                        param.grad,
                                        nan=ntn,
                                        posinf=ntn,
                                        neginf=ntn,
                                    )

                        if updated := accelerator.sync_gradients:
                            step += 1
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            # compute gradient norm before zero-grad
                            with torch.no_grad():
                                _grad_norms = [
                                    torch.norm(p.grad).item()
                                    for p in model.parameters()
                                    if p.grad is not None
                                ]
                            optimizer.zero_grad()

                    if updated and _main_proc:
                        if ema is not None:
                            ema.update(self.model, mu)

                        pbar.set_postfix({"loss": loss, "step": step})
                        self.train_log.add_scalar("loss", loss, step)

                        for k, v in (
                            getattr(self.model, "_debug_purpose", None) or {}
                        ).items():
                            self.train_log.add_scalar(f"debug-pupose:{k}", v, step)

                        with torch.no_grad():
                            self.train_log.add_scalar(
                                "common/grad-norm",
                                np.mean(_grad_norms),  # pyright: ignore
                                step,
                            )
                            param_norm = np.mean(
                                [torch.norm(p).item() for p in model.parameters()]
                            )
                            self.train_log.add_scalar(
                                "common/param-norm", param_norm, step
                            )
                            self.train_log.add_scalar(
                                "common/lr", self.optim.param_groups[0]["lr"], step
                            )

            if _main_proc and epoch % max(1, epochs // self.config.eval_interval) == 0:
                with torch.no_grad():
                    model.eval()
                    _device, _dtype = (
                        accelerator.device,
                        next(self.model.parameters()).dtype,
                    )
                    losses = [
                        self.model.loss(
                            _preproc(bunch.to(_device, dtype=_dtype) * 2 - 1),
                            label=_preproc_label(labels.to(_device)),
                        ).item()
                        for bunch, labels in tqdm(self.test_loader, leave=False)
                    ]
                    self.test_log.add_scalar(f"loss", np.mean(losses), step)

                    # plot image
                    _sample, _ = next(iter(self.test_loader))
                    _, c, h, w = _sample.shape
                    _labels = None
                    if self.config.n_classes:
                        _labels = torch.arange(self.config.n_classes, device=_device)
                        _labels = _labels[:, None].repeat(
                            1, -(-self.config.num_samples // self.config.n_classes)
                        )
                        _labels = _labels.view(-1)[: self.config.num_samples]
                    sample, trajectories = self.model.sample(
                        torch.randn(
                            *(self.config.num_samples, c, h, w),
                            generator=torch.Generator(_device).manual_seed(0),
                            dtype=_dtype,
                            device=_device,
                        ),
                        _labels,
                        steps=self.config.fid_steps,
                        verbose=lambda x: tqdm(x, leave=False),
                    )
                    for i, img in enumerate(sample):
                        img = ((_postproc(img) + 1) * 0.5).clamp(0.0, 1.0)
                        self.test_log.add_image(f"sample/{i}", img, step)

                        for j, traj in list(enumerate(trajectories))[
                            :: -(-len(trajectories) // 4)
                        ]:
                            point = ((_postproc(traj[i]) + 1) * 0.5).clamp(0.0, 1.0)
                            self.test_log.add_image(f"sample/{i}/traj@{j}", point, step)

                    fid = compute_fid_with_model(
                        self.model,
                        steps=self.config.fid_steps,
                        num_samples=10000,
                        n_classes=self.config.n_classes,
                        inception_batch_size=self.test_loader.batch_size,  # pyright: ignore
                        sampling_batch_size=self.test_loader.batch_size,  # pyright: ignore
                        device=accelerator.device,
                        scaler=lambda x: (
                            ((_postproc(x) + 1) * 0.5).clamp(0.0, 1.0) * 255
                        ).to(torch.uint8),
                    )
                    model.train()
                    self.test_log.add_scalar("metric/fid10k", fid, step)

                _path = self.workspace / "ckpt" / str(step)
                accelerator.save_state(_path.as_posix())
                if ema:
                    save_model(ema.module, (_path / "_ema.safetensors").as_posix())

        _path = self.workspace / "ckpt" / str(step)
        accelerator.save_state(_path.as_posix())
        if ema:
            save_model(ema.module, (_path / "_ema.safetensors").as_posix())
