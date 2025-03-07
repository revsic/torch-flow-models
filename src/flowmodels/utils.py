import torch

from flowmodels.basis import VelocitySupports


class VelocityInverter(VelocitySupports):
    def __init__(self, model: VelocitySupports):
        self.model = model

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Invert the estimated velocity"""
        return -self.model.velocity(x_t, 1 - t)
