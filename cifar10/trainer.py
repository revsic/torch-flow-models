from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from safetensors.torch import load_model, save_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fid import compute_fid_with_model
from flowmodels.basis import ScoreModel, ODEModel
from flowmodels.utils import EMASupports


class Loggable(Protocol):
    def log(self, msg: str): ...


class DefaultLogger(Loggable):
    def __init__(self):
        try:
            from loguru import logger
        except ImportError:
            logger = None
        self.logger = logger

    def log(self, msg: str, level: str = "INFO"):
        if self.logger is not None:
            self.logger.log(level, msg)
        else:
            print(f"[{level}] {msg}")


class Cifar10Trainer:
    def __init__(
        self,
        model: ScoreModel | ODEModel,
        batch_size: int = 512,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        dataset_path: Path = Path("./cifar-10-batches-py/.."),
        workspace: Path = Path("./workspace"),
        _logger: Loggable = DefaultLogger(),
    ):
        self.model = model
        self.workspace = workspace
        self.trainset = torchvision.datasets.CIFAR10(
            dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.RandomHorizontalFlip(p=0.5),
                ]
            ),
        )
        self.testset = torchvision.datasets.CIFAR10(
            dataset_path,
            train=False,
            transform=transforms.ToTensor(),
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
        )

        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        workspace.mkdir(parents=True, exist_ok=True)
        self.train_log = SummaryWriter(workspace / "logs" / "train")
        self.test_log = SummaryWriter(workspace / "logs" / "test")
        self._logger = _logger

    def train(
        self,
        total: int = 400000,
        mixed_precision: str = "no",  # "fp16"
        gradient_accumulation_steps: int = 1,
        num_samples: int = 10,
        load_ckpt: Path | None = None,
        half_life_ema: int | None = None,
        _eval_interval: int = 20,
        _load_ema_ckpt: Path | None = None,
    ):
        self.model.train()
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=self.workspace / "logs",
        )
        _main_proc = accelerator.is_main_process
        if _main_proc:
            accelerator.init_trackers("train")

        model, optimizer, train_loader, test_loader = accelerator.prepare(
            self.model, self.optim, self.train_loader, self.test_loader
        )
        scheduler = None
        if self.scheduler is not None:
            scheduler = accelerator.prepare(self.scheduler)

        if load_ckpt is not None:
            torch.serialization.add_safe_globals(
                [
                    np._core.multiarray.scalar,
                    np.dtypes.Float64DType,
                ]
            )
            accelerator.load_state(load_ckpt.absolute().as_posix())

        _model = model
        if isinstance(_model, torch._dynamo.OptimizedModule):
            _model = model._orig_mod
        if isinstance(_model, torch.nn.parallel.DistributedDataParallel):
            _model = model.module

        ema, mu = None, 0.0
        if half_life_ema:
            ema = EMASupports(_model)
            # compute batch size again for multi-gpu settings
            batch_size = len(self.trainset) // len(train_loader)
            self._logger.log(f"Estimated batch-size: {batch_size}")
            # compute ema factor
            steps = half_life_ema / batch_size
            mu = 0.5 ** (1 / steps)
            self._logger.log(
                f"Estimated mu: {mu:.4f}, and halved({mu ** int(steps):.2f}) in {int(steps)} steps"
            )

            if _load_ema_ckpt:
                load_model(
                    ema.module, _load_ema_ckpt, device=next(ema.parameters()).device
                )

        step = 0
        epochs = -(-total // len(train_loader))
        for epoch in tqdm(range(epochs), disable=not _main_proc):
            with tqdm(train_loader, leave=False, disable=not _main_proc) as pbar:
                for sample, labels in pbar:
                    with accelerator.accumulate(model):
                        loss = _model.loss(sample * 2 - 1)
                        accelerator.backward(loss)

                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()

                    step += 1
                    if _main_proc:
                        if ema is not None:
                            ema.update(_model, mu)

                        pbar.set_postfix({"loss": loss.item(), "step": step})
                        self.train_log.add_scalar("loss", loss.item(), step)

                        for k, v in (getattr(_model, "_debug_purpose", None) or {}).items():
                            self.train_log.add_scalar(f"debug-pupose:{k}", v, step)

                        with torch.no_grad():
                            _grad_norms = [
                                torch.norm(p.grad).item()
                                for p in model.parameters()
                                if p.grad is not None
                            ]
                            if _grad_norms:
                                self.train_log.add_scalar(
                                    "common/grad-norm", np.mean(_grad_norms), step
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

            if _main_proc and epoch % (epochs // _eval_interval) == 0:
                with torch.no_grad():
                    losses = [
                        _model.loss(bunch * 2 - 1).item()
                        for bunch, labels in tqdm(test_loader, leave=False)
                    ]
                    self.test_log.add_scalar(f"loss", np.mean(losses), step)

                    # plot image
                    model.eval()
                    sample, _ = next(iter(test_loader))
                    _, c, h, w = sample.shape
                    sample, trajectories = _model.sample(
                        torch.randn(
                            *(num_samples, c, h, w),
                            generator=torch.Generator(sample.device).manual_seed(0),
                            dtype=sample.dtype,
                            device=sample.device,
                        ),
                        verbose=lambda x: tqdm(x, leave=False),
                    )
                    model.train()
                    for i, img in enumerate(sample):
                        img = ((img + 1) * 0.5).clamp(0.0, 1.0)
                        self.test_log.add_image(f"sample/{i}", img, step)

                        for j, traj in list(enumerate(trajectories))[
                            :: -(-len(trajectories) // 4)
                        ]:
                            point = ((traj[i] + 1) * 0.5).clamp(0.0, 1.0)
                            self.test_log.add_image(f"sample/{i}/traj@{j}", point, step)

                    fid = compute_fid_with_model(
                        _model,
                        num_samples=10000,
                        sampling_batch_size=self.test_loader.batch_size,
                        device=accelerator.device,
                        scaler=lambda x: (
                            (x - x.amin()) / (x.amax() - x.amin()) * 255
                        ).to(torch.uint8),
                    )
                    self.test_log.add_scalar("metric/fid10k", fid, step)

                accelerator.save_state(self.workspace / "ckpt" / str(epoch))
                if ema:
                    save_model(
                        ema.module,
                        self.workspace / "ckpt" / str(epoch) / "_ema.safetensors",
                    )

        accelerator.save_state(self.workspace / "ckpt" / str(epoch))
        if ema:
            save_model(
                ema.module, self.workspace / "ckpt" / str(epoch) / "_ema.safetensors"
            )
