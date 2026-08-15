"""Minimal spatial 2x super-resolution model."""

import torch
from torch import nn


class SpatialSR2x(nn.Module):
    """Upscale float NCHW RGB tensors by a factor of two."""

    scale_factor = 2

    def __init__(self, feature_channels: int = 16) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        self.network = nn.Sequential(
            nn.Conv2d(3, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(feature_channels, 3 * self.scale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale_factor),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("input must have shape [N, 3, H, W]")
        if image.shape[1] != 3:
            raise ValueError("input must contain exactly three RGB channels")
        if image.shape[2] <= 0 or image.shape[3] <= 0:
            raise ValueError("input height and width must be positive")
        return self.network(image)
