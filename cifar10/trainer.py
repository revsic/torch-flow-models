from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fid import compute_fid_with_model
from flowmodels.basis import ScoreModel, ODEModel


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
    ):
        self.model = model
        self.workspace = workspace
        self.trainset = torchvision.datasets.CIFAR10(
            dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5),
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

        workspace.mkdir(parents=True, exist_ok=True)
        self.train_log = SummaryWriter(workspace / "logs" / "train")
        self.test_log = SummaryWriter(workspace / "logs" / "test")

    def train(
        self,
        total: int = 400000,
        mixed_precision: str = "no",  # "fp16"
        gradient_accumulation_steps: int = 1,
        num_samples: int = 10,
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
        _model = model
        if isinstance(_model, torch._dynamo.OptimizedModule):
            _model = model._orig_mod
        if isinstance(_model, torch.nn.parallel.DistributedDataParallel):
            _model = model.module

        step = 0
        epochs = -(-total // len(self.train_loader))
        for epoch in tqdm(range(epochs), disable=not _main_proc):
            with tqdm(train_loader, leave=False, disable=not _main_proc) as pbar:
                for sample, labels in pbar:
                    with accelerator.accumulate(model):
                        loss = _model.loss(sample * 2 - 1)
                        accelerator.backward(loss)

                        optimizer.step()
                        optimizer.zero_grad()

                    step += 1
                    if _main_proc:
                        pbar.set_postfix({"loss": loss.item(), "step": step})
                        self.train_log.add_scalar("loss", loss.item(), step)

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

            if _main_proc and epoch % (epochs // 20) == 0:
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
                    )
                    self.test_log.add_scalar("metric/fid10k", fid, step)

                accelerator.save_state(self.workspace / "ckpt" / str(epoch))

        accelerator.save_state(self.workspace / "ckpt" / str(epoch))
