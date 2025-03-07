import copy
from typing import Self

import torch
import torch.nn as nn

from flowmodels.basis import VelocitySupports


class EMASupports[T: nn.Module](nn.Module):
    def __init__(self, module: T):
        super().__init__()
        self.module = copy.deepcopy(module)

    @torch.no_grad()
    def update(self, module: nn.Module, mu: float | torch.Tensor):
        given = dict(module.named_parameters())
        for name, param in self.module.named_parameters():
            assert name in given, f"parameters not found; named `{name}`"
            param.copy_(mu * param.data + (1 - mu) * given[name].data)

    @classmethod
    def reduce(cls, self_: T, ema: T | Self | None = None) -> T:
        if ema is None:
            return self_
        if isinstance(ema, EMASupports):
            return ema.module
        return ema


class VelocityInverter(VelocitySupports):
    def __init__(self, model: VelocitySupports):
        self.model = model

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Invert the estimated velocity"""
        return -self.model.velocity(x_t, 1 - t)
