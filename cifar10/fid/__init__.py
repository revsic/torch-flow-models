import sys

import pickle
from pathlib import Path
from PIL import Image
from typing import Callable

import numpy as np
import scipy.linalg
import torch
from tqdm.auto import tqdm

from flowmodels.basis import SamplingSupports


DEFAULT_INCEPTION_PATH = Path("./inception-2015-12-05.pkl")
DEFAULT_REFERECNE_PATH = Path("./cifar10-32x32.npz")


def _load_inception(inception: Path = DEFAULT_INCEPTION_PATH):
    import fid.torch_utils as torch_utils
    import fid.dnnlib as dnnlib

    sys.modules.update({"torch_utils": torch_utils, "dnnlib": dnnlib})
    # https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
    with open(inception or DEFAULT_INCEPTION_PATH, "rb") as f:
        detector = pickle.load(f)
    sys.modules.pop("torch_utils")
    sys.modules.pop("dnnlib")
    return detector


def calculate_inception_stats(
    dataloader: torch.utils.data.DataLoader,
    inception: Path = DEFAULT_INCEPTION_PATH,
    feature_dim: int = 2048,
    device: torch.device | str = "cuda:0",
):
    detector = _load_inception(inception).to(device)
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images in tqdm(dataloader, leave=False):
        # [B, C]
        features = detector(images.to(device), return_features=True).to(torch.float64)
        # [C]
        mu += features.sum(dim=0)
        # [C, C]
        sigma += features.T @ features

    mu /= len(dataloader.dataset)
    sigma -= mu.ger(mu) * len(dataloader.dataset)
    sigma /= len(dataloader.dataset) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(
    mu: np.ndarray,
    sigma: np.ndarray,
    mu_ref: np.ndarray,
    sigma_ref: np.ndarray,
) -> float:
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def compute_fid(
    images: torch.utils.data.DataLoader,
    inception: Path = DEFAULT_INCEPTION_PATH,
    cache: Path = DEFAULT_REFERECNE_PATH,
    device: torch.device | str = "cuda:0",
) -> float:
    # https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
    ref = np.load(cache)
    mu, sigma = calculate_inception_stats(images, inception, device=device)
    return calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])


def compute_fid_with_model(
    sampler: SamplingSupports,
    steps: int | None = None,
    num_samples: int = 50000,
    sampling_batch_size: int = 1,
    inception_batch_size: int = 1,
    shape: list[int] = [3, 32, 32],
    device: torch.device | str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    inception: Path = DEFAULT_INCEPTION_PATH,
    cache: Path = DEFAULT_REFERECNE_PATH,
    scaler: Callable[[torch.Tensor], torch.Tensor] = (
        lambda x: ((x + 1) * 127.5).to(torch.uint8)
    ),
    _save_images: Path | None = None,
) -> float:

    class _SyntheticDataset(torch.utils.data.IterableDataset):
        def __len__(self):
            return num_samples

        def __iter__(self):
            images, _id = [], 0
            generator = torch.Generator(device).manual_seed(0)
            for _ in range(num_samples):
                if not images:
                    sampled, _ = sampler.sample(
                        torch.randn(
                            sampling_batch_size,
                            *shape,
                            generator=generator,
                            device=device,
                            dtype=dtype,
                        ),
                        steps=steps,
                        verbose=lambda x: tqdm(x, leave=False),
                    )
                    images = [*scaler(sampled)]
                    if _save_images:
                        for image in images:
                            _id += 1
                            Image.fromarray(
                                image.detach().cpu().permute(1, 2, 0).numpy()
                            ).save(_save_images / f"{_id}.png")
                yield images.pop()

    return compute_fid(
        torch.utils.data.DataLoader(
            _SyntheticDataset(),
            batch_size=inception_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        ),
        inception=inception,
        cache=cache,
        device=device,
    )
